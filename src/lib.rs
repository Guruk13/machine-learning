use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;

use crate::ml::multiagent::AgentManager;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
pub mod ml;
pub mod player;
pub mod plugins;

pub mod staticdevice;

pub use plugins::*;

type RegistrySnapshot = HashSet<u32>;

pub struct AMRessource {
    pub agent_manager: AgentManager,
    // Type here must match whatever `registry.0` / `live_inventory.0`
    // actually are — swap `RegistrySnapshot` for the real type.
    pub replacement: Rc<RefCell<ReplacementState>>,
}

#[derive(Clone)]
pub enum ReplacementState {
    Idle,
    Descending,
    Replacing,
    Ready(RegistrySnapshot),
}
type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
#[derive(Debug, Clone)]
pub struct Applying;

unsafe impl Sync for AMRessource {}
