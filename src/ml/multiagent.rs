use super::model::Action;
use super::model::FlappyGradientAgent;
use super::model::GameStateFeatures;
//use bevy::ecs::error::warn;
use bevy::prelude::warn;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use super::model::get_optimizer;
use crate::ml::agent_utils::AgentState;
use crate::ml::agent_utils::AgentStats;
use crate::ml::model::AgentDefault;
use crate::ml::pruner::PopulationManager;
use crate::ml::pruner::PruningConfig;

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

    pub fn select_action(&mut self, key: u32, state: &GameStateFeatures) -> Action {
        let action: Action;
        if let Some(agent) = self.inner.get_mut(&key) {
            action = agent.select_action(state);
        } else {
            warn!("agent not found '{}'", key);
            action = Action::DoNothing;
        }
        action
    }

    pub fn bind_agent(&mut self, key: u32) {
        self.inner.entry(key).or_insert(FlappyGradientAgent::new(
            self.device.clone(),
            AgentStats::new(),
        ));
    }

    pub fn unbind_agent(&mut self, key: u32) {
        //remove *should* use the drop function which is freeing the memory of WGPU's garabage collector more efficiently
        self.inner.remove(&key);
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
                state: AgentState::new();
                //new agent from net but keep evaluating its progression
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
                agent.state=   AgentState::new();
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
