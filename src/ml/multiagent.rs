use std::collections::HashMap;
use super::FlappyGradientAgent; 


use burn::{
    tensor::{ backend::AutodiffBackend},
};



// Define your own trait with a custom insert
trait MyInsert {
    fn insert(&mut self, key: String, value: i32);
}

struct MyMap {
    inner: HashMap<String, i32>,
}

// Override insert behavior — here we sum instead of overwrite
impl MyInsert for MyMap {
    fn insert(&mut self, key: String, value: i32) {
        let entry = self.inner.entry(key).or_insert(0);
        *entry += value; // custom logic
    }
}


// ─────────────────────────────────────────────
// 4.  MULTI-BIRD RUNNER
// ─────────────────────────────────────────────
/// Manages N birds that each own a `FlappyGradientAgent`.
/// Your game loop calls the methods below; this struct owns the agents.
pub struct MultiAgentRunner<B: AutodiffBackend> {
    pub agents: Vec<FlappyGradientAgent<B>>,
}

impl<B: AutodiffBackend> MultiAgentRunner<B> {
    /// Create `n` independent agents, all fresh.
    pub fn new(n: usize, device: B::Device, gamma: f32, lr: f64) -> Self {
        let agents = (0..n)
            .map(|_| FlappyGradientAgent::new(device.clone(), gamma, lr))
            .collect();
        Self { agents }
    }

    /// Ask agent `i` which action to take.
    pub fn select_action(&self, bird_idx: usize, state: &GameStateFeatures) -> Action {
        self.agents[bird_idx].select_action(state)
    }

    /// Record one tick of experience for bird `i`.
    pub fn record_step(
        &mut self,
        bird_idx: usize,
        state: GameStateFeatures,
        action: Action,
        reward: f32,
    ) {
        self.agents[bird_idx].record_step(state, action, reward);
    }

    /// Call when bird `i` dies — triggers its flappy update.
    /// Returns the loss for logging / display.
    pub fn bird_died(&mut self, bird_idx: usize) -> f32 {
        self.agents[bird_idx].finish_episode()
    }
}