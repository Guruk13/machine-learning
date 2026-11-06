use core::f32;

use bevy::prelude::{Component, Reflect};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone)]
pub struct EpisodeStep {
    pub state: GameStateFeatures,
    pub action: Action,
    pub reward: f32,
}

/// Rolling statistics for one agent.
#[derive(Debug, Copy, Clone)]
pub struct AgentStats {
    pub episodes: u64,
    pub total_score: f32,
    pub total_episodes: u64,
    pub rank: u32,
}

//Exponential Moving Average (EMA)
impl AgentStats {
    pub fn new(total_episodes: Option<u64>) -> Self {
        Self {
            episodes: 0,
            total_score: 0.,
            total_episodes: total_episodes.unwrap_or(0),
            rank: 0,
        }
    }
}

#[derive(Debug, Reflect, Component, Clone, Copy, Default)]
pub struct GameStateFeatures {
    pub bird_speed: f32,
    pub next_pipe_top_y: f32,
    pub next_pipe_bottom_y: f32,

    pub remaining_bot_x: f32,
    pub remaining_top_x: f32,
}
impl GameStateFeatures {
    /// Returns a flat [6] array — handy when building batch tensors manually.
    pub fn to_array(&self) -> [f32; 5] {
        [
            self.bird_speed, // multiply speed to make more noise in the tensors
            self.next_pipe_top_y,
            self.next_pipe_bottom_y,
            self.remaining_bot_x,
            self.remaining_top_x,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[wasm_bindgen]
pub enum Action {
    DoNothing = 0,
    Jump = 1,
}

pub struct RewardPrizes {
    pub dying: f32,
    pub pipe_cleared: f32,
    pub alive: f32,

    pub pipe_death: f32,

    pub jump_cost: f32,
}

#[derive(Debug, Component)]
pub struct AgentState {
    pub is_dead: bool,
    pub current_gamestate: Option<GameStateFeatures>,
    pub episode: Vec<EpisodeStep>, //@ref
    pub score: u32,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
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

impl Default for RewardPrizes {
    fn default() -> Self {
        Self {
            dying: -3.,
            pipe_death: -1.1,

            pipe_cleared: 1.,
            alive: 0.05,
            jump_cost: 0.0,
        }
    }
}
