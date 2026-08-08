use super::agent_utils::GameStateFeatures;
use super::model::FlappyGradientAgent;
//use bevy::ecs::error::warn;
use burn::tensor::backend::AutodiffBackend;
use std::collections::HashMap;

use crate::ml::agent_utils::AgentState;

//@todo make mod agentutils to prevent bad usage
//mod agent_utils;
pub struct AgentManager<B: AutodiffBackend> {
    pub inner: HashMap<u32, FlappyGradientAgent>,
    device: B::Device,
}

//"If you want to guarantee this is only ever used with the Wgpu backend, you can add a where clause"
impl<B: AutodiffBackend> AgentManager<B> {
    pub fn new(device: B::Device) -> AgentManager<B> {

        Self {
            inner: HashMap::new(),
            device: ,
        }
    }

    pub fn bind_agent(
        &mut self,
        key: u32,
        game_state: GameStateFeatures,
    ) -> &mut FlappyGradientAgent<B> {
        let device = self.device.clone();
        let agent = self
            .inner
            .entry(key)
            .or_insert_with(|| FlappyGradientAgent::new(device));
        agent.state.set_state_features(Some(game_state));
        agent
    }

    pub fn unbind_agent(&mut self, key: u32) {
        //remove *should* use the drop function which is freeing the memory of WGPU's garabage collector more efficiently
        self.inner.remove(&key);
    }

    pub fn update_metrics(&mut self, key: u32) {
        match self.inner.get_mut(&key) {
            Some(agent) => {
                agent.state = AgentState::new();
                agent.stats.episodes = agent.stats.episodes + 1;
                //agent.stats.entropy_sum = 0.;
            }
            None => panic!("Agent '{}' not found", key),
        }
    }

    //  match self.inner.get_mut(&key) {
    //      Some(agent) => {
    //          // spot entropicishes
    //          return agent.finish_episode();
    //      }
    //      None => panic!("Agent '{}' not found", key),
    //  }
}
