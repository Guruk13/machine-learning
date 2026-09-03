use super::agent_utils::GameStateFeatures;
use super::model::FlappyGradientAgent;
use std::collections::HashMap;

use std::cell::RefCell;
use std::rc::Rc;

pub struct AgentManager {
    pub inner: HashMap<u32, Rc<RefCell<FlappyGradientAgent>>>,
}

impl AgentManager {
    pub fn new() -> AgentManager {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn bind_agent(&mut self, key: u32, game_state: GameStateFeatures) {
        let agent = self
            .inner
            .entry(key)
            .or_insert_with(|| Rc::new(RefCell::new(FlappyGradientAgent::new())));
        agent
            .borrow_mut()
            .state
            .set_state_features(Some(game_state));
    }

    pub fn unbind_agent(&mut self, key: u32) {
        //remove *should* use the drop function which is freeing the memory of WGPU's garabage collector more efficiently
        self.inner.remove(&key);
    }
}
