// agent_pruning.rs
// This is clearly overshoot, but interesting to implement. Now that the collection of the game state is fixed , this is of minor performance gain.

// Entropy-based detection and pruning of under-performing agents in a
// multi-agent FlappyBird setting.
//
// ── Concepts ────────────────────────────────────────────────────────────────
//
//  • ENTROPY  of a policy π is   H(π) = -Σ p_i · log(p_i)
//    For a 2-action policy: H ∈ [0,  ln 2 ≈ 0.693].
//    A value near 0 means the agent is completely deterministic (may be stuck).
//    A value near ln 2 means the agent is essentially random (hasn't learned).
//
//  • SCORE TRACKER  keeps an exponential moving average (EMA) of the episode
//    return for each agent so we can detect stagnation independent of entropy.
//
//  • PRUNING CRITERIA  (any one triggers pruning):
//      1. Policy entropy stays below `entropy_floor` for `patience` episodes
//         → agent has collapsed to a degenerate deterministic policy.
//      2. Policy entropy stays above `entropy_ceiling` for `patience` episodes
//         → agent never left the random-walk regime.
//      3. EMA score stays below `score_floor` for `patience` episodes
//         → agent is consistently dying immediately regardless of entropy.
//
//  • REPLACEMENT  A pruned agent is replaced by a *cloned-and-perturbed* copy
//    of the best-scoring survivor, giving it a fresh optimiser state while
//    keeping useful learned weights.

// ─────────────────────────────────────────────────────────────────────────────
// 1.  ENTROPY HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Shannon entropy of a probability vector (any length).
///
/// H = -Σ p_i · log(p_i),  with  0·log(0) := 0
pub fn shannon_entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum()
}

/// Maximum possible entropy for `n_actions` actions.
///   H_max = ln(n_actions)
pub fn max_entropy(n_actions: usize) -> f32 {
    (n_actions as f32).ln()
}

/// Normalised entropy ∈ [0, 1].
pub fn normalised_entropy(probs: &[f32]) -> f32 {
    let h_max = max_entropy(probs.len());
    if h_max == 0.0 {
        return 1.0;
    }
    shannon_entropy(probs) / h_max
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  CONFIGURATION
// ─────────────────────────────────────────────────────────────────────────────

/// All tunable knobs for the pruning system.
#[derive(Debug, Clone)]
pub struct PruningConfig {
    // ── Entropy thresholds (normalised, ∈ [0, 1]) ─────────────────────────
    /// Entropy below this → agent is pathologically deterministic.
    pub entropy_floor: f32,
    /// Entropy above this → agent never learned to exploit.
    pub entropy_ceiling: f32,
    /// EMA smoothing for entropy.
    pub entropy_alpha: f32,

    // ── Score threshold ───────────────────────────────────────────────────
    /// EMA episode return below this → agent is consistently failing.
    pub score_floor: f32,
    /// EMA smoothing for score.
    pub score_alpha: f32,

    // ── Timing ────────────────────────────────────────────────────────────
    /// Episodes before we even consider pruning (let agents warm up).
    pub warmup_episodes: u64,
    /// Consecutive violations required before pruning fires.
    pub patience: u32,

    // ── Replacement ───────────────────────────────────────────────────────
    /// Noise scale added to the parent weights when cloning.
    pub noise_scale: f32,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            //models must be very surprising in their decisions ( flappy bird is a surprise based game )
            entropy_floor: 0.1,
            entropy_ceiling: 0.9,
            //degrade the entropc bounds very slowly
            entropy_alpha: 0.01, //
            score_floor: 1.,     // tune to your reward scale
            score_alpha: 0.5,    // you have to improve at least by a percent each episode

            warmup_episodes: 30,
            patience: 10,

            noise_scale: 0.01,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  ENTROPY MEASUREMENT  (burn tensor → f32 slice)
// ─────────────────────────────────────────────────────────────────────────────

/// Measure the average normalised entropy of `net` over a batch of states.
///
/// `states` – raw feature rows, each `[6]`; stacked into `[n, 6]`.
///
/// Returns a value in [0, 1].
///

// a sum of entropy for a model. is defined within a trait since it comes attach itself to a forward function to reduce
pub trait EntropyTracker {
    fn summarize(&self) -> String;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  POPULATION MANAGER
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a population of agents: tracks their stats, flag them if entropicishes
pub struct PopulationManager {
    //pub agents: Vec<FlappyGradientAgent<B>>,
    //pub stats: Vec<AgentStats>,
    pub cfg: PruningConfig,
}
impl PopulationManager {
    pub fn new() -> Self {
        Self {
            cfg: PruningConfig::default(),
        }
    }
}
