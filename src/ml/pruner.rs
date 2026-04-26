// agent_pruning.rs
//
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

use crate::{FlappyGradientAgent, ml::EpisodeStep};
use std::{collections::HashMap, panic};

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
// 2.  PER-AGENT STATISTICS
// ─────────────────────────────────────────────────────────────────────────────

/// Rolling statistics for one agent.
#[derive(Debug, Copy, Clone)]
pub struct AgentStats {
    // ── Entropy tracking ──────────────────────────────────────────────────
    /// EMA of normalised policy entropy (α = 0.1 by default).
    pub entropy_ema: f32,
    /// How many consecutive episodes entropy has been outside healthy range.
    pub entropy_violation_streak: u32,

    // ── Score tracking ────────────────────────────────────────────────────
    /// EMA of episode total return (α = 0.05 by default).
    pub score_ema: f32,
    /// How many consecutive episodes the score has been below floor.
    pub score_violation_streak: u32,

    /// Total episodes recorded.
    pub episodes: u64,
}

impl AgentStats {
    pub fn new() -> Self {
        Self {
            entropy_ema: 1.0, // start at max entropy (uninitialised)
            entropy_violation_streak: 0,
            score_ema: 0.0,
            score_violation_streak: 0,
            episodes: 0,
        }
    }

    /// Update EMAs after an episode.
    ///
    /// * `raw_entropy` – normalised entropy ∈ [0, 1] measured over the episode
    /// * `episode_return` – sum of undiscounted rewards for the episode
    pub fn update(&mut self, episode: Vec<EpisodeStep>, cfg: &PruningConfig) {
        self.episodes += 1;
        let episode_return: f32 = episode.iter().fold(0.0, |acc, x| acc + x.reward);
        // Score EMA
        self.score_ema =
            cfg.score_alpha * episode_return + (1.0 - cfg.score_alpha) * self.score_ema;

        // Violation streaks
        let entropy_ok =
            self.entropy_ema >= cfg.entropy_floor && self.entropy_ema <= cfg.entropy_ceiling;
        if entropy_ok {
            self.entropy_violation_streak = 0;
        } else {
            self.entropy_violation_streak += 1;
        }

        let score_ok = self.score_ema >= cfg.score_floor;
        if score_ok {
            self.score_violation_streak = 0;
        } else {
            self.score_violation_streak += 1;
        }
    }
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
            entropy_alpha: 0.10,

            score_floor: -0.5, // tune to your reward scale
            score_alpha: 0.05,

            warmup_episodes: 20,
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
pub fn measure_policy_entropy<B: AutodiffBackend>(agent: &FlappyGradientAgent<B>) -> f32 {
    let n = agent.episode.len();
    agent.entropy_sum / n as f32
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
        // ── Measure entropy for each agent ────────────────────────────────
        agents
            .iter_mut()
            .for_each(|(key, agent): (&u32, &mut FlappyGradientAgent<B>)| {
                let entropy = measure_policy_entropy(agent);
                agent.stats.entropy_ema = self.cfg.entropy_alpha * entropy
                    + (1.0 - self.cfg.entropy_alpha) * agent.stats.entropy_ema;
                if self.should_prune(agent.stats) {
                    to_prune.push(*key);
                };
            });
        let best_idx: u32;
        if let Some((key, _agent)) = agents
            .iter()
            .filter(|(key, _agent)| !to_prune.contains(key))
            .max_by(|(_, a), (_, b)| a.stats.score_ema.partial_cmp(&b.stats.score_ema).unwrap())
        {
            best_idx = *key;
        } else {
            panic!("no best agent ... ")
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
    // ── Convenience getters ───────────────────────────────────────────────

    /*pub fn best_agent_idx(&self) -> usize {
        (0..self.agents.len())
            .max_by(|&a, &b| {
                self.stats[a]
                    .score_ema
                    .partial_cmp(&self.stats[b].score_ema)
                    .unwrap()
            })
            .unwrap_or(0)
    }*/

    /// Print a summary table to stdout (useful during training).
    /* pub fn print_stats(stats: AgentStats) {
        println!(
            "{:>5}  {:>8}  {:>12}  {:>10}  {:>10}",
            "agent", "episodes", "entropy_ema", "score_ema", "violations"
        );
        println!(
            "{:>5}  {:>8}  {:>12.4}  {:>10.3}  e:{} s:{}",
            stats.agent_id,
            stats.episodes,
            stats.entropy_ema,
            stats.score_ema,
            stats.entropy_violation_streak,
            stats.score_violation_streak,
        );
    }*/
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
