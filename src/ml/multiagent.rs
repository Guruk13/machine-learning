use super::Action;
use super::FlappyGradientAgent;
use super::GameStateFeatures;
use bevy::ecs::error::warn;
use bevy::prelude::warn;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use crate::get_optimizer;
use crate::ml::AgentDefault;
use crate::ml::pruner::AgentStats;
use crate::ml::pruner::PopulationManager;
use crate::ml::pruner::PruningConfig;

//following traits are not really usefull except for bind agent which *may* be influenced by the way you implement the backend , other than that there's not much to keep

/* trait BindAgent<B: AutodiffBackend> {
    fn bind_agent(&mut self, key: String, gamma: f32, lr: f64);
} */

pub struct AgentManager<B: AutodiffBackend> {
    pub inner: HashMap<u32, FlappyGradientAgent<B>>,
    device: B::Device,
    pop: PopulationManager,
}

//"If you want to guarantee this is only ever used with the Wgpu backend, you can add a where clause"
impl<B: AutodiffBackend> AgentManager<B> {
    pub fn new(device: B::Device) -> AgentManager<B> {
        Self {
            device: device,
            inner: HashMap::new(),
            pop: PopulationManager::new(),
        }
    }

    pub fn select_action(&mut self, key: &u32, state: &GameStateFeatures) -> Action {
        let action: Action;
        if let Some(agent) = self.inner.get_mut(key) {
            action = agent.select_action(state);
        } else {
            warn!("agent not found '{}'", key);
            action = Action::DoNothing;
        }
        action
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
            entropy_sum: 0.0,
            stats: AgentStats::new(),
        };
        self.inner.insert(down, newagent);
        self
    }

    /// Record one tick of experience for bird `i`.
    pub fn record_step(&mut self, key: u32, state: GameStateFeatures, action: Action, reward: f32) {
        //warn!("recording");
        match self.inner.get_mut(&key) {
            Some(agent) => agent.record_step(state, action, reward),
            None => panic!("Agent '{}' not found", key),
        }
    }

    // do the swap logic inline or call a free fn
    /**
     * By destructuring self, Rust sees inner and pop as independent borrows rather than two borrows of the same self, which resolves the conflict.
    If swap_net is a method on Self and needs &mut self again, extract it too: */

    pub fn prune_agents(&mut self) -> &mut Self {
        let Self { inner, pop, .. } = self;
        let (to_prune, best) = pop.spot_entropicishes(inner);
        for key in to_prune {
            // swap an agent's net with another
            // The optimiser state is always freshly initialised.
            let optimizer = get_optimizer();
            let up_agent_net = self.inner[&best].flappy.clone();
            let newagent: FlappyGradientAgent<B> = FlappyGradientAgent {
                flappy: up_agent_net,
                optimizer: optimizer,
                device: self.device.clone(),
                episode: Vec::new(),
                gamma: AgentDefault::default().gamma,
                lr: AgentDefault::default().learning_rate,
                entropy_sum: 0.0,
                stats: AgentStats::new(),
            };
            self.inner.insert(key, newagent);
        }
        self
    }

    pub fn update_stats(&mut self) {
        self.inner.iter_mut().for_each(|(_key, agent)| {
            agent
                .stats
                .update(agent.episode.clone(), &PruningConfig::default());
        });
    }
    pub fn clear_episode(&mut self, key: u32) {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                agent.episode.clear();
            }
            None => panic!("Agent '{}' not found", key),
        }
    }

    /// Call when bird `i` dies — triggers its flappy update.
    /// Returns the loss for logging / display.

    pub fn bird_died(&mut self, key: u32) -> f32 {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                // spot entropicishes
                return agent.finish_episode();
            }
            None => panic!("Agent '{}' not found", key),
        }
    }
}
