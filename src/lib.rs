use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;

use std::collections::HashSet;
pub mod ml;
pub mod player;
pub mod plugins;

pub mod staticdevice;

pub use plugins::*;

type RegistrySnapshot = HashSet<u32>;

#[derive(Clone)]
pub enum ReplacementState {
    Idle,
    Descending,
    Replacing,
    Ready(RegistrySnapshot),
}
type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
pub enum GameMode {
    Train,
    Play,
    Demo,
}

impl Default for GameMode {
    fn default() -> Self {
        GameMode::Train
    }
}
