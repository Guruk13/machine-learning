use super::pipes::*;
use crate::ml::agent_utils::Action;
use crate::player::{Bird, BirdJump, GameSets, Player, Velocity};
use crate::player::{PipeBottom, PipeTop};
use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::tensor::backend::AutodiffBackend;

use crate::ml::multiagent::AgentManager;

use crate::ml::agent_utils::{GameStateFeatures, RewardPrizes};

pub struct AMRessource<B: AutodiffBackend> {
    agent_manager: AgentManager<B>,
}

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

unsafe impl<B: AutodiffBackend> Sync for AMRessource<B> {}

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        let device: <MyAutodiffBackend as burn::tensor::backend::Backend>::Device =
            Default::default();

        let ressource = AMRessource::<MyAutodiffBackend> {
            agent_manager: AgentManager::new(device),
        };

        app.insert_non_send_resource(ressource);
        app.add_systems(
            FixedUpdate,
            (ApplyDeferred, think).chain().in_set(GameSets::AI),
        );
    }
}

//Think Bird.  what will you have after 500 years ?
fn think(
    mut commands: Commands,
    birds: Query<(Entity, &Bird, &Transform, &Velocity, &Aabb), (With<Bird>, Without<Player>)>,

    pipe_tops: Query<(&GlobalTransform, &Aabb), With<PipeTop>>,
    pipe_bottoms: Query<(&GlobalTransform, &Aabb), With<PipeBottom>>,
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
) {
    let all_birds: Vec<_> = birds.iter().collect();

    //warn!( "Look mom , no .... : {:?}",query.iter().count);
    for (entity, bird, transform, velocity, bird_aabb) in &all_birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0).to_angle();
        //let bird_y = transform.translation.y;
        let bird_x = transform.translation.x;
        let mut pipes_above = pipe_tops.iter().filter(|t| {
            t.0.translation().x + t.1.half_extents.x > bird_x - bird_aabb.half_extents.x
        });
        let nearest_top = pipes_above.next();
        let second_top = pipes_above.next();

        let mut pipes_below = pipe_bottoms.iter().filter(|t| {
            t.0.translation().x + t.1.half_extents.x > bird_x - bird_aabb.half_extents.x
        });
        let nearest_bottom = pipes_below.next();
        let second_bottom = pipes_below.next();

        // Then for top pipe: bottom edge = translation.y - half_extents.y
        let nearest_top_bottom_edge = nearest_top
            .map(|(t, aabb)| t.translation().y - aabb.half_extents.y)
            .unwrap_or(500.0);

        // Then for top pipe: bottom edge = translation.y - half_extents.y
        let second_top_bottom_edge = second_top
            .map(|(t, aabb)| t.translation().y - aabb.half_extents.y)
            .unwrap_or(500.0);

        let nearest_bottom_top_edge = nearest_bottom
            .map(|(t, aabb)| t.translation().y + aabb.half_extents.y)
            .unwrap_or(-500.0);

        let second_bottom_top_edge = second_bottom
            .map(|(t, aabb)| t.translation().y + aabb.half_extents.y)
            .unwrap_or(-500.0);

        let top_right_edge = nearest_top
            .map(|(t, aabb)| t.translation().x + aabb.half_extents.x)
            .unwrap_or(f32::NEG_INFINITY); // no entity → treat as already cleared

        let bot_right_edge = nearest_bottom
            .map(|(t, aabb)| t.translation().x + aabb.half_extents.x)
            .unwrap_or(f32::NEG_INFINITY); // no entity → treat as already cleared

        let bird_right_edge = bird_x + bird_aabb.half_extents.x;

        let remaining_top_x = if bird_right_edge >= top_right_edge {
            -0.5 // already past, or not overlapping
        } else {
            top_right_edge - bird_right_edge
        };
        let remaining_bot_x = if bird_right_edge >= bot_right_edge {
            -0.5 // already past, or not overlapping
        } else {
            bot_right_edge - bird_right_edge
        };

        let state = GameStateFeatures {
            bird_y: transform.translation.y,
            bird_speed: calculated_velocity, //calculated_velocity,
            next_pipe_top_y: nearest_top_bottom_edge,
            next_pipe_bottom_y: nearest_bottom_top_edge,
            second_top_y: second_top_bottom_edge,
            second_bot_y: second_bottom_top_edge,
            remaining_bot_x: remaining_bot_x,
            remaining_top_x: remaining_top_x,
        };
        //for debugging purposes only
        commands.entity(*entity).insert(state);
        //get an agent , sync it with Bird's pov
        let agent = am_ressource.agent_manager.bind_agent(bird.uid, state);
        //forward

        let action = agent.select_action();

        //compute reward
        let mut reward: f32 = if bird.dead {
            RewardPrizes::default().dying
        } else {
            RewardPrizes::default().alive
        };

        if bird.score > agent.state.score {
            reward += RewardPrizes::default().pipe_cleared;
            agent.state.score = bird.score;
        }

        let gap_centre = (state.next_pipe_top_y + state.next_pipe_bottom_y) / 2.0;
        let gap_centre_penalty = (state.bird_y - gap_centre).abs() * 0.001;
        // Small bonus for staying near the centre of the gap
        reward = reward - gap_centre_penalty;
        //
        agent.record_step(action.clone(), reward);
        match action {
            Action::DoNothing => { /*  not because you pelican means you pelishould */ }
            Action::Jump => {
                if !bird.dead {
                    commands.trigger(BirdJump(*entity));
                }
            }
        }
    }

    //Gamestate has been processed , process bird's agent  stats
    let birds_dead: Vec<_> = all_birds
        .iter()
        .filter(|(_, bird, _, _, _)| bird.dead == true)
        .collect();

    for bird in &birds_dead {
        am_ressource.agent_manager.bird_died(bird.1.uid);
    }
    // warn!("{:?}", birds_dead.len());

    if !&birds_dead.is_empty() {
        am_ressource.agent_manager.update_stats();
        //am_ressource.agent_manager.prune_agents();
    }

    for bird in &birds_dead {
        am_ressource.agent_manager.purge_states(bird.1.uid);
    }
}

#[derive(Component, Default, Clone, Copy)]
pub struct PartialReward(pub f32);

impl std::ops::Sub<f32> for PartialReward {
    type Output = f32;
    fn sub(self, rhs: f32) -> f32 {
        self.0 - rhs
    }
}
