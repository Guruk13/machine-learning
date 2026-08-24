use burn::backend::wgpu::{init_device, Wgpu, WgpuDevice, WgpuSetup};
use burn::backend::Autodiff;
use burn::tensor::backend::AutodiffBackend;

use crate::ml::multiagent::AgentManager;

pub mod ml;
pub mod player;
pub mod plugins;
pub mod staticdevice;

pub use plugins::*;

//#[cfg(feature = "tracker")]
//mod dqn_tracker;

pub struct AMRessource {
    pub agent_manager: AgentManager,
}

type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;

unsafe impl Sync for AMRessource {}
