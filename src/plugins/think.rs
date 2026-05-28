use super::pipes::*;
use crate::ml::agent_utils::Action;
use crate::player::{Bird, BirdInventory, BirdJump, GameSets, Player, Velocity};
use crate::player::{PipeBottom, PipeTop};
use bevy::camera::primitives::Aabb;
use bevy::pbr::ParallaxMappingMethod::Occlusion;
use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ops::InterpolateMode::Nearest;

use crate::ml::multiagent::AgentManager;

use crate::ml::agent_utils::{GameStateFeatures, RewardPrizes};

pub struct AMRessource<B: AutodiffBackend> {
    agent_manager: AgentManager<B>,
}

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

unsafe impl<B: AutodiffBackend> Sync for AMRessource<B> {}

pub struct BrainPlugin;

// to be able to run updates on every agent without cutting some agent's experience , store the birds here , then respawn once optimizations are through
#[derive(Resource, Default)]
pub struct DeadBirdRegistry(pub Vec<u32>);

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
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
    mut registry: ResMut<DeadBirdRegistry>,
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
        let nearest_top_bottom_edge;
        let nearest_bottom_top_edge;
        let second_top_bottom_edge;
        let second_bottom_top_edge;
        let remaining_top_x;
        let remaining_bot_x;
        let bird_left_edge = bird_x + bird_aabb.half_extents.x;
        if let Some((transform, aabb)) = nearest_top {
            nearest_top_bottom_edge = transform.translation().y - aabb.half_extents.y;
            let top_right_edge;
            top_right_edge = transform.translation().x + aabb.half_extents.x;
            remaining_top_x = top_right_edge - bird_left_edge;
        } else {
            nearest_top_bottom_edge = CANVAS_SIZE[1] / 2. + 1.;
            remaining_top_x = -1.;
        }
        if let Some((transform, aabb)) = nearest_bottom {
            nearest_bottom_top_edge = transform.translation().y + aabb.half_extents.y;
            let bot_right_edge;
            bot_right_edge = transform.translation().x + aabb.half_extents.x;
            remaining_bot_x = bot_right_edge - bird_left_edge
        } else {
            nearest_bottom_top_edge = -CANVAS_SIZE[1] / 2. - 1.;

            remaining_bot_x = -1.;
        }

        if let Some((transform, aabb)) = second_bottom {
            second_top_bottom_edge = transform.translation().y - aabb.half_extents.y;
        } else {
            second_top_bottom_edge = CANVAS_SIZE[1] / 2. + 1.;
        }

        if let Some((transform, aabb)) = second_top {
            second_bottom_top_edge = transform.translation().y + aabb.half_extents.y;
        } else {
            second_bottom_top_edge = -CANVAS_SIZE[1] / 2. - 1.;
        }

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
        let mut reward: f32 = if bird.dead && !bird.pipe_death {
            //dying soon is shameful
            RewardPrizes::default().dying
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
        let gap_half = (state.next_pipe_top_y - state.next_pipe_bottom_y).abs() / 2.0;

        let dist_to_centre = (state.bird_y - gap_centre).abs();
        let centre_bonus = 1.0 - (dist_to_centre / gap_half).clamp(0.0, 1.0); // 1.0 at centre, 0.0 at edge

        reward += centre_bonus;
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
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
    mut live_inventory: ResMut<BirdInventory>,
) {
    //run the forward function
    am_ressource.agent_manager.agents_over();

    //grade general behavior
    am_ressource.agent_manager.update_stats(&registry.0);
    am_ressource.agent_manager.prune_agents();
    for bird in &registry.0 {
        am_ressource.agent_manager.update_metrics(*bird);
    }
    live_inventory.0 = registry.0.clone();
    registry.0.clear();
}
pub fn no_birds_in_main_system(alive: Query<&Bird>, live_inventory: Res<BirdInventory>) -> bool {
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
