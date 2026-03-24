use bevy::{
    prelude::*,
};


#[derive(Component)]
pub struct Gravity(pub f32);

#[derive(Component, Default,Debug)]
pub struct Velocity(pub f32);

#[derive(Component, Debug)]
#[require(Gravity(1000.), Velocity)]
pub struct Player;

#[derive(Resource, Default)]
pub struct Score(pub u32);

#[derive(Event)]
pub struct ScorePoint;

#[derive(Component)]
pub struct ScoreText;
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GameSets {
    Game,
    AI,
}

