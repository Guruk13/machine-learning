use bevy::prelude::*;
use burn::backend::wgpu::Wgpu;
use burn::backend::Autodiff;
use std::collections::HashSet;
pub mod ml;
pub mod player;
pub mod plugins;

pub mod staticdevice;

pub use plugins::*;
#[derive(Event)]
pub struct BeginGame;
#[derive(Event)]
pub struct EndGame;
type RegistrySnapshot = HashSet<u8>;

#[derive(Clone)]
pub enum ReplacementState {
    Idle,
    Descending,
    Replacing,
    Ready(RegistrySnapshot),
}
type MyBackend = Wgpu<f32, i32>;
pub type MyAutodiffBackend = Autodiff<MyBackend>;
#[derive(Clone, Copy, Debug, Reflect, PartialEq)]
pub enum GameMode {
    Train { birdnumber: u8, opacity: f32 },
    Play { birdnumber: u8, opacity: f32 },
    Demo { birdnumber: u8, opacity: f32 },
}

impl GameMode {
    pub fn train() -> Self {
        GameMode::Train {
            birdnumber: 10,
            opacity: 50.0,
        }
    }
    pub fn play() -> Self {
        GameMode::Play {
            birdnumber: 1,
            opacity: 100.0,
        }
    }
    pub fn demo() -> Self {
        GameMode::Demo {
            birdnumber: 1,
            opacity: 100.0,
        }
    }
}

#[derive(Resource, Reflect)]
pub struct GamemodeResource(pub GameMode);

pub trait EndsGame: Event {}
impl EndsGame for SwitchGamemode {}
impl EndsGame for EndGame {}

#[derive(Event)]
pub struct SwitchGamemode(pub GameMode);
