use super::agent_utils::GameStateFeatures;
use super::model::FlappyGradientAgent;
//use bevy::ecs::error::warn;
//use bevy::prelude::warn;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use super::model::get_optimizer;
use crate::ml::agent_utils::AgentState;
use crate::ml::agent_utils::AgentStats;
use crate::ml::pruner::PopulationManager;
use crate::ml::pruner::PruningConfig;
//@todo make mod agentutils to prevent bad usage
//mod agent_utils;

pub struct AgentMap<B: AutodiffBackend> {
    pub inner: HashMap<u32, FlappyGradientAgent<B>>,

    #[cfg(feature = "tracker")]
    pub trackers: HashMap<u32, Arc<RwLock<dqn_tracker::Agent>>>,
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
            #[cfg(feature = "tracker")]
            registry: dqn_tracker::AgentRegistry::new(),
        }
    }

    pub fn bind_agent(
        &mut self,
        key: u32,
        game_state: GameStateFeatures,
    ) -> &mut FlappyGradientAgent<B> {
        let agent = self
            .inner
            .entry(key)
            .or_insert_with(|| FlappyGradientAgent::new(self.device.clone()));

        agent.state.set_state_features(Some(game_state));
        agent
    }

    pub fn unbind_agent(&mut self, key: u32) {
        //remove *should* use the drop function which is freeing the memory of WGPU's garabage collector more efficiently
        self.inner.remove(&key);
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
                state: AgentState::new(),
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
                .update(agent.state.episode.clone(), &PruningConfig::default());
        });
    }
    pub fn purge_states(&mut self, key: u32) {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                agent.state = AgentState::new();
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

    impl<B: AutodiffBackend> AgentMap<B> {
        pub fn on_episode_end(&mut self, id: u32) {
            #[cfg(feature = "tracker")]
            if let (Some(agent), Some(tracker)) = (self.inner.get(&id), self.trackers.get(&id)) {
                let mut t = tracker.write().unwrap();
                t.episodes.push(agent.state.score as f64);
                t.epsilon = agent.stats.entropy_ema as f64;
            }
        }
    }




}
