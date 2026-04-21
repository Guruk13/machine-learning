use super::Action;
use super::FlappyGradientAgent;
use super::GameStateFeatures;
use burn::tensor::backend::AutodiffBackend;
use burn_wgpu::{Wgpu, WgpuDevice};
use std::collections::HashMap;

//following traits are not really usefull except for bind agent which *may* be influenced by the way you implement the backend , other than that there's not much to keep 

trait BindAgent<B: AutodiffBackend> {
    fn bind_agent(&mut self, key: String, gamma: f32, lr: f64);
}

trait FreeAgent<B: AutodiffBackend> {
    fn free_agent(&mut self, key: String, value: FlappyGradientAgent<B>);
}

impl<B: AutodiffBackend> BindAgent<B> for AgentManager<B> {
    fn bind_agent(&mut self, key: String, gamma: f32, lr: f64) {
        let agent = FlappyGradientAgent::new(self.device.clone(), gamma, lr);
        self.inner.insert(key, agent);
    }      
}

impl<B: AutodiffBackend> FreeAgent<B> for AgentManager<B> {
    fn free_agent(&mut self, key: String, _value: FlappyGradientAgent<B>) {
        self.inner.remove(&key);
    }
}

pub struct AgentManager<B: AutodiffBackend> {
    pub inner: HashMap<String, FlappyGradientAgent<B>>,
    device: B::Device,
}

//"If you want to guarantee this is only ever used with the Wgpu backend, you can add a where clause"
impl<B: AutodiffBackend> AgentManager<B> {
    pub fn new(device: B::Device) -> AgentManager<B> {
        Self {
            device: device,
            inner: HashMap::new(),
        }
    }

    pub fn select_action(&self, key: String, state: &GameStateFeatures) -> Action {
        self.inner[&key].select_action(state)
    }

    /// Record one tick of experience for bird `i`.
    pub fn record_step(
        &mut self,
        key: String,
        state: GameStateFeatures,
        action: Action,
        reward: f32,
    ) {
        match self.inner.get_mut(&key) {
            Some(agent) => agent.record_step(state, action, reward),
            None => panic!("Agent '{}' not found", key),
        }
    }

    /// Call when bird `i` dies — triggers its flappy update.
    /// Returns the loss for logging / display.

    pub fn bird_died(&mut self, key: String) -> f32 {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                return agent.finish_episode();
            }
            None => panic!("Agent '{}' not found", key),
        }
    }
}
