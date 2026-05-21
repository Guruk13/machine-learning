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

use burn::tensor::backend::AutodiffBackend;

use super::agent_utils::AgentStats;
use super::model::FlappyGradientAgent;
use bevy::prelude::warn;
use std::collections::HashMap;

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
            entropy_floor: 0.05,   // below 5 % of max → collapsed
            entropy_ceiling: 0.95, // above 95 % of max → still random
            entropy_alpha: 0.1,    // "if you jump one percent of the occasions you have , it's ok"

            score_floor: -0.2, // tune to your reward scale
            score_alpha: 0.1,

            warmup_episodes: 20,
            patience: 20,

            noise_scale: 0.001,
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
pub fn measure_policy_entropy<B: AutodiffBackend>(agent: &FlappyGradientAgent<B>) -> f32 {
    let n = agent.state.episode.len();
    if n == 0 {
        return 0.0;
    }
    agent.stats.entropy_sum / n as f32
}

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

    // ── Per-episode hook ──────────────────────────────────────────────────

    /// Call at the end of every episode **after** `finish_episode()` has been
    /// called on each agent.
    ///
    /// * `episode_states` – the states visited by each agent this episode
    ///   (used only for entropy measurement; pass `&[]` to skip).
    /// * `episode_returns` – total undiscounted return per agent.
    ///
    /// Returns the indices of agents that need to be pruned and replaced as well as the index of the most "performant" bird.
    pub fn spot_entropicishes<B: AutodiffBackend>(
        &self,
        agents: &mut HashMap<u32, FlappyGradientAgent<B>>,
    ) -> (Vec<u32>, u32) {
        let mut to_prune: Vec<u32> = Vec::new();
        //let mut warn = true;
        // ── Measure entropy for each agent ────────────────────────────────
        agents
            .iter_mut()
            .for_each(|(_key, agent): (&u32, &mut FlappyGradientAgent<B>)| {
                let entropy = measure_policy_entropy(agent);
                //if (warn && *key == 1 as u32) {
                //    warn!("{:?}", agent.stats.entropy_ema);
                //    warn = true;
                //}

                agent.stats.entropy_ema = self.cfg.entropy_alpha * entropy
                    + (1.0 - self.cfg.entropy_alpha) * agent.stats.entropy_ema;
            });

        agents
            .iter()
            .for_each(|(key, agent): (&u32, &FlappyGradientAgent<B>)| {
                bevy::prelude::info!(
                   "Agent {key} | ep={} | entropy_ema={:.3} | score_ema={:.3} | e_streak={} | s_streak={}",
                   agent.stats.episodes,
                   agent.stats.entropy_ema,
                   agent.stats.score_ema,
                   agent.stats.entropy_violation_streak,
                   agent.stats.score_violation_streak,
               );
                if self.should_prune(agent.stats) {
                    to_prune.push(*key);
                }
            });
        let best_idx: u32;
        if let Some((key, _agent)) = agents
            .iter()
            .filter(|(key, _agent)| !to_prune.contains(key))
            .max_by(|(_, a), (_, b)| a.stats.score_ema.partial_cmp(&b.stats.score_ema).unwrap())
        {
            best_idx = *key;
        } else {
            warn!("no best agent ... ");
            best_idx = *agents.keys().next().unwrap();
        }

        for &idx in &to_prune {
            let reason = self.prune_reason(agents[&idx].stats);
            bevy::prelude::warn!(
                "Agent {idx} pruned after {} episodes — {reason} \
                 (entropy_ema={:.3}, score_ema={:.3})",
                agents[&idx].stats.episodes,
                agents[&idx].stats.entropy_ema,
                agents[&idx].stats.score_ema,
            );

            // Reset statistics for the new agent.
        }

        (to_prune, best_idx)
    }
    /// Returns `true` when an agent should be pruned.
    pub fn should_prune(&self, agent_stats: AgentStats) -> bool {
        // Need at least `patience` episodes before we prune.
        if agent_stats.episodes < self.cfg.warmup_episodes {
            return false;
        }
        agent_stats.entropy_violation_streak >= self.cfg.patience
            || agent_stats.score_violation_streak >= self.cfg.patience
    }

    /// Human-readable reason for pruning (for logging).
    pub fn prune_reason(&self, stats: AgentStats) -> &'static str {
        if stats.entropy_ema < self.cfg.entropy_floor {
            "policy-collapsed (entropy too low)"
        } else if stats.entropy_ema > self.cfg.entropy_ceiling {
            "policy-random (entropy too high)"
        } else {
            "score-stagnant (score EMA below floor)"
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8.  UNIT TESTS
// ─────────────────────────────────────────────────────────────────────────────

/* #[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform over 2 actions → H = ln 2 ≈ 0.693
        let h = shannon_entropy(&[0.5, 0.5]);
        assert!((h - 2.0_f32.ln()).abs() < 1e-5, "got {h}");
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        let h = shannon_entropy(&[1.0, 0.0]);
        assert!(h.abs() < 1e-5, "got {h}");
    }

    #[test]
    fn test_normalised_entropy_range() {
        for p in [0.0f32, 0.1, 0.3, 0.5, 0.9, 1.0] {
            let h = normalised_entropy(&[p, 1.0 - p]);
            assert!((0.0..=1.0 + 1e-5).contains(&h), "out of range: {h} for p={p}");
        }
    }

    #[test]
    fn test_agent_stats_warmup_prevents_prune() {
        let cfg = PruningConfig {
            warmup_episodes: 10,
            patience: 3,
            ..Default::default()
        };
        let mut s = AgentStats::new(0);
        // Simulate a very bad agent for 5 episodes (below warmup).
        for _ in 0..5 {
            s.update(0.0, -10.0, &cfg); // entropy=0 → violation every step
        }
        assert!(!s.should_prune(&cfg), "should not prune before warmup");
    }

    #[test]
    fn test_agent_stats_prune_on_entropy_collapse() {
        let cfg = PruningConfig {
            warmup_episodes: 2,
            patience: 3,
            entropy_floor: 0.05,
            ..Default::default()
        };
        let mut s = AgentStats::new(0);
        for _ in 0..10 {
            s.update(0.0, 1.0, &cfg); // entropy collapsed to 0
        }
        assert!(s.should_prune(&cfg));
        assert_eq!(s.prune_reason(&cfg), "policy-collapsed (entropy too low)");
    }

    #[test]
    fn test_agent_stats_prune_on_score_stagnation() {
        let cfg = PruningConfig {
            warmup_episodes: 2,
            patience: 3,
            score_floor: 0.0,
            ..Default::default()
        };
        let mut s = AgentStats::new(0);
        for _ in 0..10 {
            s.update(0.5, -5.0, &cfg); // entropy fine, score terrible
        }
        assert!(s.should_prune(&cfg));
        assert_eq!(s.prune_reason(&cfg), "score-stagnant (score EMA below floor)");
    }
} */
