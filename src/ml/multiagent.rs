use crate::ml::agent_utils::AgentDefault;

use super::agent_utils::GameStateFeatures;
use super::model::FlappyGradientAgent;
use std::collections::HashMap;

use crate::ReplacementState;
use burn::backend::wgpu::WgpuDevice;
use std::sync::OnceLock;
use std::{cell::RefCell, rc::Rc};

pub struct AgentManager {
    pub inner: HashMap<u8, Rc<RefCell<FlappyGradientAgent>>>,
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentManager {
    pub fn new() -> AgentManager {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn bind_agent(&mut self, key: u8, game_state: GameStateFeatures) {
        let agent = self.inner.entry(key).or_insert_with(|| {
            Rc::new(RefCell::new(FlappyGradientAgent::new(AgentDefault::new())))
        });
        agent
            .borrow_mut()
            .state
            .set_state_features(Some(game_state));
    }

    pub fn unbind_agent(&mut self, key: u8) {
        //remove *should* use the drop function which is freeing the memory of WGPU's garabage collector more efficiently
        self.inner.remove(&key);
    }
}

pub struct AMRessource {
    pub agent_manager: AgentManager,
    pub replacement: Rc<RefCell<ReplacementState>>,
}
//maybe useless
unsafe impl Send for AMRessource {}

static DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

pub fn set_global_device(device: WgpuDevice) {
    DEVICE
        .set(device)
        .unwrap_or_else(|_| panic!("device already initialized"));
}
pub fn download_bytes(bytes: &[u8], filename: &str) {
    use wasm_bindgen::JsCast;
    let arr = js_sys::Uint8Array::from(bytes);
    let parts = js_sys::Array::new();
    parts.push(&arr);
    let blob = web_sys::Blob::new_with_u8_array_sequence(&parts).unwrap();
    let url = web_sys::Url::create_object_url_with_blob(&blob).unwrap();

    let document = web_sys::window().unwrap().document().unwrap();
    let a: web_sys::HtmlAnchorElement = document.create_element("a").unwrap().dyn_into().unwrap();
    a.set_href(&url);
    a.set_download(filename);
    a.click();
    web_sys::Url::revoke_object_url(&url).ok();
}

pub fn get_global_device() -> WgpuDevice {
    DEVICE
        .get()
        .expect("device not initialized — call init() from JS before constructing agents")
        .clone()
}
