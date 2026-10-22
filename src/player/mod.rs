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
#[derive(Component, Debug)]
pub struct PipeDeath;

#[derive(Component, Debug)]
pub struct IndividualScore(pub u32);

#[derive(Event)]
pub struct ScorePoint {
    pub bird: Entity,
    pub gate: Entity,
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

#[derive(Component, Debug)]
#[require(Gravity(1000.), Velocity)]
pub struct Bird {
    pub uid: u8,
}

#[derive(Event)]
pub struct BirdDeath {
    pub bird: Entity,
}

#[derive(EntityEvent)]
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

#[derive(Component)]
pub struct Pipe;

#[derive(Component)]
pub struct PipeTop;

#[derive(Component)]
pub struct PipeBottom;

#[derive(Component)]
pub struct PointsGate {
    pub has_scored: Vec<u8>,
}

#[derive(Component, Debug, Default)]
pub struct Lifetime {
    pub ticks: u32,
}
