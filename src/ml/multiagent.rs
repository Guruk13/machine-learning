use super::agent_utils::GameStateFeatures;
use super::model::FlappyGradientAgent;
use bevy::camera::MsaaWriteback::Off;
//use bevy::ecs::error::warn;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use super::model::get_optimizer;
use crate::ml::agent_utils::AgentDefault;
use crate::ml::agent_utils::AgentState;
use crate::ml::agent_utils::AgentStats;
use crate::ml::critic::Critic;
use crate::ml::pruner::PopulationManager;
use crate::ml::pruner::PruningConfig;
//@todo make mod agentutils to prevent bad usage
//mod agent_utils;
pub struct AgentManager<B: AutodiffBackend> {
    pub inner: HashMap<u32, FlappyGradientAgent<B>>,
    device: B::Device,
    pop: PopulationManager,
    critic: Critic<B>,
}

//"If you want to guarantee this is only ever used with the Wgpu backend, you can add a where clause"
impl<B: AutodiffBackend> AgentManager<B> {
    pub fn new(device: B::Device) -> AgentManager<B> {
        Self {
            device: device.clone(),
            inner: HashMap::new(),
            pop: PopulationManager::new(),
            critic: Critic::new(device.clone()),
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

            let total_episodes =
                self.inner[&key].stats.total_episodes + self.inner[&key].stats.episodes;
            let newagent: FlappyGradientAgent<B> = FlappyGradientAgent {
                flappy: up_agent_net,
                optimizer: optimizer,
                device: self.device.clone(),
                state: AgentState::new(),
                //new agent from net but keep evaluating its progression
                stats: AgentStats::new(Some(total_episodes)),
            };

            self.inner.insert(key, newagent);
        }
        self
    }

    pub fn update_stats(&mut self, keys: &[u32]) {
        for key in keys {
            match self.inner.get_mut(key) {
                Some(agent) => {
                    agent
                        .stats
                        .update(agent.state.episode.clone(), &PruningConfig::default());
                }
                None => panic!("Agent '{}' not found", key),
            }
        }
    }
    pub fn update_metrics(&mut self, key: u32) {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                agent.state = AgentState::new();
                agent.stats.episodes = agent.stats.episodes + 1;
                agent.stats.entropy_sum = 0.;
                agent.state.episode.clear();
            }
            None => panic!("Agent '{}' not found", key),
        }
    }

    ///  may Returns the loss for logging / display.

    pub async fn agents_over(&mut self) {
        let mut all_states = Vec::<f32>::new();
        let mut all_returns = Vec::<f32>::new();
        let mut total_steps = 0usize;

        for (_, agent) in self.inner.iter_mut() {
            let (_actor_loss, states, returns, n) = agent.finish_episode(&self.critic).await;

            all_states.extend(states);
            all_returns.extend(returns);
            total_steps += n;
        }

        // One critic update with everyone's data pooled
        self.critic.update_batch(
            &all_states,
            &all_returns,
            total_steps,
            AgentDefault::default().learning_rate,
        ).await;
    }
    //  match self.inner.get_mut(&key) {
    //      Some(agent) => {
    //          // spot entropicishes
    //          return agent.finish_episode();
    //      }
    //      None => panic!("Agent '{}' not found", key),
    //  }
}
