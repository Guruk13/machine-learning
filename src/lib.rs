use bevy::prelude::*;

pub mod ml;
pub mod player;
pub mod plugins;

pub mod staticdevice;

pub use plugins::*;

#[derive(Debug, Clone)]
pub enum ReplacementState {
    Idle,
    Descending,
    Replacing,
    Ready,
}
#[derive(Debug, Clone, Copy, Reflect, PartialEq)]
pub enum GameMode {
    Train { birdnumber: u8, opacity: f32 },
    Play { birdnumber: u8, opacity: f32 },
    Demo { birdnumber: u8, opacity: f32 },
}

impl GameMode {
    pub fn train() -> Self {
        GameMode::Train {
            birdnumber: 8,
            opacity: 70.0,
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
impl Default for GameMode {
    fn default() -> Self {
        Self::train()
    }
}
#[derive(Debug, Resource, Reflect)]
pub struct GamemodeRessource(pub GameMode);
#[derive(Debug, Message)]
pub struct BeginGame;
#[derive(Debug, Message)]
pub struct EndGame;
#[derive(Debug, Message)]
pub struct SwitchGamemode;
#[derive(Debug, Message)]
pub struct ResetGamemode;
