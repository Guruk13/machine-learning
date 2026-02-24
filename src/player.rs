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
