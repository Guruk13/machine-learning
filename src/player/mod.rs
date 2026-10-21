use crate::ml::agent_utils::Action;
use crate::CANVAS_SIZE;
use crate::PLAYER_SIZE;
use bevy::prelude::*;

use std::collections::HashSet;
use std::collections::VecDeque;
#[derive(Component)]
pub struct Gravity(pub f32);

#[derive(Component, Default, Debug)]
pub struct Velocity(pub f32);

#[derive(Component, Debug)]
pub struct Player;

#[derive(Component, Debug)]
pub struct Dead;
#[derive(Component, Debug)]
pub struct Opacity {
    pub value: f32,
}

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

#[derive(Component, Debug)]
#[require(Gravity(1000.), Velocity)]
pub struct Bird {
    pub pending: PendingActions,
    pub uid: u8,
    pub dead: bool,
    pub score: u32,
    pub pipe_death: bool,
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
    ) -> (Bird, Sprite, Transform, Lifetime) {
        let alpha = (opacity / 100.0).clamp(0.0, 1.0);
        let color: Color = Srgba::hex("#282828").unwrap().with_alpha(alpha).into();

        (
            Bird {
                uid: id,
                dead: false,
                score: 0,
                pipe_death: false,
                pending: PendingActions::default(),
            },
            Sprite {
                custom_size: Some(Vec2::splat(PLAYER_SIZE)),
                image: asset_server.load("bevy-bird.png"),
                color: color,
                ..default()
            },
            Transform::from_xyz(-CANVAS_SIZE.x / 4.0, 0.0, 1.0),
            Lifetime::default(),
        )
    }
}

#[derive(Resource, Default, Reflect)]
pub struct BirdInventory(pub HashSet<u8>);

#[derive(Debug, Component, Clone, Copy, PartialEq, Eq)]
pub enum Flag {
    Dead, // @todo use this instead of  Bird state
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

#[derive(Component, Default, Debug)]
pub struct PendingActions(pub VecDeque<Action>);

#[derive(Component, Debug, Default)]
pub struct Lifetime {
    pub ticks: u32,
}
