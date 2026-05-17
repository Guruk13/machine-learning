use crate::CANVAS_SIZE;
use crate::PLAYER_SIZE;
use bevy::prelude::*;

#[derive(Component)]
pub struct Gravity(pub f32);

#[derive(Component, Default, Debug)]
pub struct Velocity(pub f32);

#[derive(Component, Debug)]
pub struct Player;

#[derive(Component, Debug)]
pub struct Dead;

#[derive(Event)]
pub struct ScorePoint {
    pub bird: Entity,
}

#[derive(Resource, Default)]
pub struct Score(pub u32);

#[derive(Component)]
pub struct ScoreText;
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GameSets {
    Input,   // read player/agent intentions
    Game,    // physics, collisions, bounds
    Cleanup, // despawn, respawn
    AI,      // think, bind agent
}

#[derive(Component, Debug, Reflect)]
#[require(Gravity(1000.), Velocity)]
pub struct Bird {
    pub uid: u32,
    pub dead: bool,
    pub score: u32,
}

#[derive(Event)]
pub struct BirdDeath {
    pub bird: Entity,
}

//https://docs.rs/bevy_ecs/0.18.0/bevy_ecs/event/struct.EntityTrigger.html don't tell claude about this or i'll be jobless
#[derive(EntityEvent)]
#[entity_event(propagate)]
pub struct BirdJump(pub Entity);

impl Bird {
    pub fn new(asset_server: &AssetServer, rand: bool, id: u32) -> (Bird, Sprite, Transform) {
        (
            Bird {
                uid: id,
                dead: false,
                score: 0,
            },
            Sprite {
                custom_size: Some(Vec2::splat(PLAYER_SIZE)),
                image: asset_server.load("bevy-bird.png"),
                color: Srgba::hex("#282828").unwrap().into(),
                ..default()
            },
            if rand {
                Transform::from_xyz(
                    -CANVAS_SIZE.x / rand::random_range(1..=4) as f32,
                    rand::random_range(0..=5) as f32,
                    1.0,
                )
            } else {
                Transform::from_xyz(-CANVAS_SIZE.x / 4.0, 0.0, 1.0)
            },
        )
    }
}

#[derive(Resource, Default)]
pub struct BirdInventory(pub Vec<u32>);

#[derive(Component)]
pub struct Pipe;

#[derive(Component)]
pub struct PipeTop;

#[derive(Component)]
pub struct PipeBottom;

#[derive(Component)]
pub struct PointsGate;
