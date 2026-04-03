use crate::CANVAS_SIZE;
use crate::PLAYER_SIZE;
use bevy::prelude::*;
use rand::*;

#[derive(Component)]
pub struct Gravity(pub f32);

#[derive(Component, Default, Debug)]
pub struct Velocity(pub f32);

#[derive(Component, Debug)]
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

#[derive(Component, Debug)]
#[require(Gravity(1000.), Velocity)]
pub struct Bird;

#[derive(Event)]
pub struct BirdDeath {
    pub bird: Entity,
}

//https://docs.rs/bevy_ecs/0.18.0/bevy_ecs/event/struct.EntityTrigger.html don't tell claude about this or i'll be jobless 
#[derive(EntityEvent)]
#[entity_event(propagate)]
pub struct BirdJump(pub Entity);


impl Bird {
    pub fn new(asset_server: &AssetServer, rand: bool) -> (Bird, Sprite, Transform) {
        (
            Bird,
            Sprite {
                custom_size: Some(Vec2::splat(PLAYER_SIZE)),
                image: asset_server.load("bevy-bird.png"),
                color: Srgba::hex("#282828").unwrap().into(),
                ..default()
            },
            if rand {
                Transform::from_xyz(-CANVAS_SIZE.x / rand::random_range(1..=4) as f32, rand::random_range(0..=5)as f32, 1.0)
            } else {
                Transform::from_xyz(-CANVAS_SIZE.x / 4.0, 0.0, 1.0)
            },
        )
    }
}
