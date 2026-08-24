use std::process::exit;

use super::pipes::*;
use crate::ml::agent_utils::Action;
use crate::ml::agent_utils::{GameStateFeatures, RewardPrizes};
use crate::ml::multiagent::AgentManager;
use crate::player::{Bird, BirdInventory, BirdJump, GameSets, Player, Velocity};
use crate::player::{PipeBottom, PipeTop};
use crate::AMRessource;
use bevy::camera::primitives::Aabb;
use bevy::prelude::*;

use crate::MyAutodiffBackend;

use burn::tensor::Device;
pub struct BrainPlugin;

// to be able to run updates on every agent without cutting some agent's experience , store the birds here , then respawn once optimizations are through
#[derive(Resource, Default)]
pub struct DeadBirdRegistry(pub Vec<u32>);

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (
                ApplyDeferred,
                think,
                run_forwards_and_optims.run_if(no_birds_in_main_system),
            )
                .chain()
                .in_set(GameSets::AI),
        )
        .insert_resource(DeadBirdRegistry(vec![]));
    }
}

//Think Bird.  what will you have after 500 years ?
fn think(
    mut commands: Commands,
    birds: Query<(Entity, &Bird, &Transform, &Velocity, &Aabb), (With<Bird>, Without<Player>)>,

    pipe_tops: Query<(&GlobalTransform, &Aabb), With<PipeTop>>,
    pipe_bottoms: Query<(&GlobalTransform, &Aabb), With<PipeBottom>>,
    mut am_ressource: NonSendMut<AMRessource>,
    mut registry: ResMut<DeadBirdRegistry>,
) {
    let all_birds: Vec<_> = birds.iter().collect();

    //warn!( "Look mom , no .... : {:?}",query.iter().count);
    for (entity, bird, transform, velocity, bird_aabb) in &all_birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0).to_angle();
        let bird_x = transform.translation.x;
        let bird_left_edge = bird_x + bird_aabb.half_extents.x;
        let bird_right_edge = bird_x - bird_aabb.half_extents.x; // left edge for filter

        // Sort all pipes ahead, independently (they are NOT assumed to be pairs)
        let mut tops_ahead: Vec<_> = pipe_tops
            .iter()
            .filter(|(t, aabb)| t.translation().x + aabb.half_extents.x > bird_right_edge)
            .collect();
        tops_ahead.sort_by(|a, b| {
            a.0.translation()
                .x
                .partial_cmp(&b.0.translation().x)
                .unwrap()
        });

        let mut bots_ahead: Vec<_> = pipe_bottoms
            .iter()
            .filter(|(t, aabb)| t.translation().x + aabb.half_extents.x > bird_right_edge)
            .collect();
        bots_ahead.sort_by(|a, b| {
            a.0.translation()
                .x
                .partial_cmp(&b.0.translation().x)
                .unwrap()
        });

        // Pipe 0 well past check — independent per top/bottom
        let top0_well_past = tops_ahead
            .first()
            .map(|(t, aabb)| {
                bird_left_edge - (t.translation().x + aabb.half_extents.x) > PLAYER_SIZE
            })
            .unwrap_or(true);

        let bot0_well_past = bots_ahead
            .first()
            .map(|(t, aabb)| {
                bird_left_edge - (t.translation().x + aabb.half_extents.x) > PLAYER_SIZE
            })
            .unwrap_or(true);

        let top_active = if top0_well_past { 1 } else { 0 };
        let bot_active = if bot0_well_past { 1 } else { 0 };

        // Active pipes
        let nearest_top = extract_top_pipe(tops_ahead.get(top_active).map(|x| x), bird_left_edge);
        let nearest_bot =
            extract_bottom_pipe(bots_ahead.get(bot_active).map(|x| x), bird_left_edge);

        // Lookahead pipes (next after active)
        let second_top =
            extract_top_pipe(tops_ahead.get(top_active + 1).map(|x| x), bird_left_edge);
        let second_bot =
            extract_bottom_pipe(bots_ahead.get(bot_active + 1).map(|x| x), bird_left_edge);

        let state = GameStateFeatures {
            bird_y: transform.translation.y,
            bird_speed: calculated_velocity,
            next_pipe_top_y: nearest_top.edge_y,
            next_pipe_bottom_y: nearest_bot.edge_y,
            second_top_y: second_top.edge_y,
            second_bot_y: second_bot.edge_y,
            remaining_top_x: nearest_top.remaining_x,
            remaining_bot_x: nearest_bot.remaining_x,
        };

        //for debugging purposes only
        commands.entity(*entity).insert(state);
        //get an agent , sync it with Bird's pov
        let agent = am_ressource.agent_manager.bind_agent(bird.uid, state);
        //forward

        let action = agent.select_action();

        //compute reward
        let mut reward: f32 = if bird.dead {
            //dying soon is shameful

            if bird.pipe_death {
                RewardPrizes::default().pipe_death
            } else {
                RewardPrizes::default().dying
            }
        } else {
            RewardPrizes::default().alive * (agent.state.score as f32).powi(2).max(1.0)
        };
        //prevent association of jump = reward policy
        if action == Action::Jump {
            reward += RewardPrizes::default().jump_cost
        }

        /*         let mut reward: f32 = if bird.dead && !bird.pipe_death {
            //dying soon is shameful
            RewardPrizes::default().dying * -(agent.state.score as i32) as f32
        } else {
            RewardPrizes::default().alive * (1 + bird.score) as f32
        }; */
        if bird.score > agent.state.score {
            reward +=
                RewardPrizes::default().pipe_cleared * (agent.state.score as f32).powi(5).max(1.0);
            agent.state.score = bird.score;
        }

        let gap_centre = (state.next_pipe_top_y + state.next_pipe_bottom_y) / 2.0;
        let gap_centre_penalty = (state.bird_y - gap_centre).abs() * 0.001;
        // Small bonus for staying near the centre of the gap
        reward = reward - gap_centre_penalty;
        // warn!("{:?}", reward);
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

    // warn!("{:?}", birds_dead.len());

    for bird in &birds_dead {
        registry.0.push(bird.1.uid);
    }
}
pub fn run_forwards_and_optims(
    mut registry: ResMut<DeadBirdRegistry>,
    mut am_ressource: NonSendMut<AMRessource>,
    mut live_inventory: ResMut<BirdInventory>,
) {
    //run the forward function

    let best_idx: u32;
    if let Some((key, _agent)) = am_ressource
        .agent_manager
        .inner
        .iter()
        .max_by(|(_, a), (_, b)| {
            a.stats
                .total_score
                .partial_cmp(&b.stats.total_score)
                .unwrap()
        })
    {
        best_idx = *key;
    } else {
        best_idx = *am_ressource.agent_manager.inner.keys().next().unwrap();
    }
    let best_model = am_ressource.agent_manager.inner[&best_idx].flappy.clone();

    for (key, agent) in am_ressource.agent_manager.inner.iter_mut() {
        if *key != best_idx {
            agent.flappy = best_model.clone();
        }
    }

    live_inventory.0 = registry.0.clone();
    registry.0.clear();
}
pub fn no_birds_in_main_system(
    alive: Query<&Bird>,
    live_inventory: Res<BirdInventory>,

    mut commands: Commands,
) -> bool {
    alive.is_empty() && live_inventory.0.is_empty()
}
#[derive(Component, Default, Clone, Copy)]
pub struct PartialReward(pub f32);

impl std::ops::Sub<f32> for PartialReward {
    type Output = f32;
    fn sub(self, rhs: f32) -> f32 {
        self.0 - rhs
    }
}
struct PipeEdgeInfo {
    edge_y: f32,
    remaining_x: f32,
}

fn extract_top_pipe(top: Option<&(&GlobalTransform, &Aabb)>, bird_left_edge: f32) -> PipeEdgeInfo {
    match top {
        Some((t, aabb)) => PipeEdgeInfo {
            edge_y: t.translation().y - aabb.half_extents.y,
            remaining_x: (t.translation().x + aabb.half_extents.x) - bird_left_edge,
        },
        None => PipeEdgeInfo {
            edge_y: CANVAS_SIZE[1] / 2.0 + 1.0,
            remaining_x: -PLAYER_SIZE / 2.0 - 30.0,
        },
    }
}

fn extract_bottom_pipe(
    bottom: Option<&(&GlobalTransform, &Aabb)>,
    bird_left_edge: f32,
) -> PipeEdgeInfo {
    match bottom {
        Some((t, aabb)) => PipeEdgeInfo {
            edge_y: t.translation().y + aabb.half_extents.y,
            remaining_x: (t.translation().x + aabb.half_extents.x) - bird_left_edge,
        },
        None => PipeEdgeInfo {
            edge_y: -CANVAS_SIZE[1] / 2.0 - 1.0,
            remaining_x: -PLAYER_SIZE / 2.0 - 30.0,
        },
    }
}
