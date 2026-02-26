use bevy::{
    prelude::*,
};


#[derive(Component)]
pub struct Gravity(pub f32);

#[derive(Component, Default)]
pub struct Velocity(pub f32);

#[derive(Component)]
#[require(Gravity(1000.), Velocity)]
pub struct Player;


#[derive(Resource, Default)]
pub struct Score(pub u32);

#[derive(Event)]
pub struct ScorePoint;

#[derive(Component)]
pub struct ScoreText;
