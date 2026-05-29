use crate::ml::model::OPTIMIZER_EPSILON;

use super::pruner::PruningConfig;
use bevy::prelude::{Component, Reflect};

#[derive(Debug, Clone)]
pub struct EpisodeStep {
    pub state: GameStateFeatures,
    pub action: Action,
    pub reward: f32,
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

    pub entropy_sum: f32,

    pub total_score: f32,
}

//Exponential Moving Average (EMA)
impl AgentStats {
    pub fn new() -> Self {
        Self {
            entropy_ema: 1.0, // start at max entropy (uninitialised)
            entropy_violation_streak: 0,
            score_ema: 0.0,
            score_violation_streak: 0,
            episodes: 0,
            entropy_sum: 0.0,
            total_score: 0.,
        }
    }

    /// Update EMAs after an episode.
    ///
    /// * `raw_entropy` – normalised entropy ∈ [0, 1] measured over the episode
    /// * `episode_return` – sum of undiscounted rewards for the episode
    pub fn update(&mut self, episode: Vec<EpisodeStep>, cfg: &PruningConfig) {
        //warn!("{:?}", self.episodes);
        let episode_return: f32 = episode.iter().fold(0.0, |acc, x| acc + x.reward);
        self.total_score = episode_return;
        // Score EMA
        //EMA initialization bias correction
        if self.episodes == 1 {
            self.score_ema = episode_return;
        } else {
            self.score_ema =
                cfg.score_alpha * episode_return + (1.0 - cfg.score_alpha) * self.score_ema;
        }

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

#[derive(Reflect, Component, Debug, Clone, Copy, Default)]
pub struct GameStateFeatures {
    pub bird_y: f32,
    pub bird_speed: f32,
    pub next_pipe_top_y: f32,
    pub next_pipe_bottom_y: f32,
    pub remaining_bot_x: f32,
    pub remaining_top_x: f32,
    pub second_top_y: f32,
    pub second_bot_y: f32,
}
impl GameStateFeatures {
    /// Returns a flat [6] array — handy when building batch tensors manually.
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.bird_y,
            self.bird_speed,
            self.next_pipe_top_y,
            self.next_pipe_bottom_y,
            self.second_bot_y,
            self.second_top_y,
            self.remaining_bot_x,
            self.remaining_top_x,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    DoNothing = 0,
    Jump = 1,
}

pub struct RewardPrizes {
    pub dying: f32,
    pub pipe_cleared: f32,
    pub alive: f32,

    pub jump_cost: f32,
}
impl Default for RewardPrizes {
    fn default() -> Self {
        Self {
            dying: -50.0,
            pipe_cleared: 10.,
            alive: 0.01,
            jump_cost: 0.1,
        }
    }
}
#[derive(Component)]
pub struct AgentState {
    pub is_dead: bool,
    pub current_gamestate: Option<GameStateFeatures>,
    pub episode: Vec<EpisodeStep>, //@ref

    pub score: u32,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            is_dead: false,
            current_gamestate: None,
            episode: vec![], //@ref
            score: 0,
        }
    }
    //destroys current game state
    pub fn get_state_features(&mut self) -> GameStateFeatures {
        self.current_gamestate
            .take()
            .expect("no game state on this agent")
    }
    pub fn set_state_features(&mut self, state: Option<GameStateFeatures>) {
        self.current_gamestate = state;
    }
}
/** In ML, gamma (γ) is the discount factor used in reinforcement learning (RL).
 * It's a value between 0 and 1 that determines how much an agent values future rewards relative to immediate rewards.
The Bellman Equation Context
Gamma appears in the return (cumulative reward) calculation:
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...
What the Value Means
GammaBehaviorγ = 0 Fully myopic — only cares about the immediate next rewardγ →
1Fully far-sighted — values future rewards almost as much as current onesγ = 0.99 Common default — slight preference for sooner rewards */

pub struct AgentDefault {
    pub gamma: f32,
    pub learning_rate: f64,
}

impl Default for AgentDefault {
    fn default() -> Self {
        Self {
            gamma: 0.95,
            learning_rate: 1e-4,
        }
    }
}
