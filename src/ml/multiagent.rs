use super::Action;
use super::FlappyGradientAgent;
use super::GameStateFeatures;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use crate::FlappyNet;
use crate::get_optimizer;
use crate::ml::AgentDefault;
use crate::ml::pruner::PopulationManager;

//following traits are not really usefull except for bind agent which *may* be influenced by the way you implement the backend , other than that there's not much to keep

/* trait BindAgent<B: AutodiffBackend> {
    fn bind_agent(&mut self, key: String, gamma: f32, lr: f64);
} */

pub struct AgentManager<B: AutodiffBackend> {
    pub inner: HashMap<u32, FlappyGradientAgent<B>>,
    device: B::Device,
    pop: PopulationManager<B>,
}

//"If you want to guarantee this is only ever used with the Wgpu backend, you can add a where clause"
impl<B: AutodiffBackend> AgentManager<B> {
    pub fn new(device: B::Device) -> AgentManager<B> {
        Self {
            device: device,
            inner: HashMap::new(),
            pop: PopulationManager<B>
        }
    }

    pub fn select_action(&self, key: u32, state: &GameStateFeatures) -> Action {
        self.inner[&key].select_action(state)
    }

    pub fn bind_agent(&mut self, key: u32) {
        let agent = FlappyGradientAgent::new(
            self.device.clone(),
            AgentDefault::default().gamma,
            AgentDefault::default().learning_rate,
        );
        self.inner.insert(key, agent);
    }

    pub fn unbind_agent(&mut self, key: u32) {
        //remove *should* use the drop function which is freeing the memory of WGPU's garabage collector more efficiently
        self.inner.remove(&key);
    }
    /// swap an agent's net with another
    /// The optimiser state is always freshly initialised.
    pub fn swap_net(&mut self, down: u32, up: u32) -> &mut Self {
        let optimizer = get_optimizer();
        let up_agent_net = self.inner[&up].flappy.clone();
        let newagent: FlappyGradientAgent<B> = FlappyGradientAgent {
            flappy: up_agent_net,
            optimizer: optimizer,
            device: self.device.clone(),
            episode: Vec::new(),
            gamma: AgentDefault::default().gamma,
            lr: AgentDefault::default().learning_rate,
        };
        self.inner.insert(down, newagent);
        self
    }

    /// Record one tick of experience for bird `i`.
    pub fn record_step(&mut self, key: u32, state: GameStateFeatures, action: Action, reward: f32) {
        match self.inner.get_mut(&key) {
            Some(agent) => agent.record_step(state, action, reward),
            None => panic!("Agent '{}' not found", key),
        }
    }

    /// Call when bird `i` dies — triggers its flappy update.
    /// Returns the loss for logging / display.

    pub fn bird_died(&mut self, key: u32) -> f32 {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                return agent.finish_episode();
            }
            None => panic!("Agent '{}' not found", key),
        }
    }
}
