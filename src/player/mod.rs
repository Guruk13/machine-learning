use crate::CANVAS_SIZE;
use crate::PLAYER_SIZE;
use bevy::prelude::*;

#[derive(Debug, Component)]
pub struct Gravity(pub f32);

#[derive(Debug, Component, Default)]
pub struct Velocity(pub f32);

#[derive(Debug, Component)]
pub struct Player;

#[derive(Debug, Component)]
pub struct Dead;
#[derive(Debug, Component)]
pub struct PipeDeath;

#[derive(Debug, Component)]
pub struct IndividualScore(pub u32);

#[derive(Debug, Resource, Default)]
pub struct Score(pub u32);

#[derive(Debug, Component)]
pub struct ScoreText;
#[derive(Debug, SystemSet, Clone, PartialEq, Eq, Hash)]
pub enum GameSets {
    Input,   // read player/agent intentions
    Game,    // physics, collisions, bounds
    Cleanup, // despawn, respawn
    AI,
    Play,
}

#[derive(Debug, Component)]
#[require(Gravity(950.), Velocity)]
pub struct Bird {
    pub uid: u8,
}

#[derive(Debug, Event)]
pub struct BirdDeath {
    pub bird: Entity,
}

#[derive(Debug, EntityEvent)]
#[entity_event(propagate)]
pub struct BirdJump(pub Entity);

impl Bird {
    pub fn new(
        asset_server: &AssetServer,
        id: u8,
        opacity: f32,
    ) -> (Bird, Sprite, IndividualScore, Transform, Lifetime) {
        let alpha = (opacity / 100.0).clamp(0.0, 1.0);
        let color: Color = Srgba::hex("#282828").unwrap().with_alpha(alpha).into();

        (
            Bird { uid: id },
            Sprite {
                custom_size: Some(Vec2::splat(PLAYER_SIZE)),
                image: asset_server.load("bevy-bird.png"),
                color: color,
                ..default()
            },
            IndividualScore(0),
            Transform::from_xyz(-CANVAS_SIZE.x / 4.0, 0.0, 1.0),
            Lifetime::default(),
        )
    }
}

#[derive(Debug, Component)]
pub struct Pipe;

#[derive(Debug, Component)]
pub struct PipeTop;

#[derive(Debug, Component)]
pub struct PipeBottom;

#[derive(Debug, Component)]
pub struct PointsGate {
    pub has_scored: Vec<u8>,
}

#[derive(Debug, Component, Default)]
pub struct Lifetime {
    pub ticks: u32,
}
