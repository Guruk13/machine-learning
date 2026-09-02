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
    pub replacement: Rc<RefCell<ReplacementState<RegistrySnapshot>>>,
}

// ── shared state, single source of truth (see previous message) ─────────
#[derive(Debug, Copy, Clone)]
pub enum ReplacementState<T> {
    Idle,
    InFlight,
    Ready(T),
}
type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

unsafe impl Sync for AMRessource {}
