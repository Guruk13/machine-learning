use super::pipes::*;
use crate::ml::agent_utils::{Action, GameStateFeatures, GradedRunners, RewardPrizes};
use crate::ml::model::FlappyNet;
use crate::player::{
    Bird, BirdInventory, BirdJump, GameSets, Lifetime, PipeBottom, PipeTop, Player, Velocity,
};

use crate::RegistrySnapshot;
use crate::{AMRessource, ReplacementState};
use itertools::{Itertools, MinMaxResult};
use wasm_bindgen_futures::spawn_local;

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
pub struct BrainPlugin;

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

// to be able to run updates on every agent without cutting some agent's experience , store the birds here , then respawn once optimizations are through
#[derive(Resource, Default)]
pub struct DeadBirdRegistry(pub HashSet<u32>);

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
        .insert_resource(DeadBirdRegistry(HashSet::new()));
    }
}

//Think Bird.  what will you have after 500 years ?
fn think(
    mut commands: Commands,
    birds: Query<
        (Entity, &Bird, &Transform, &Velocity, &Aabb, &Lifetime),
        (With<Bird>, Without<Player>),
    >,

    pipe_tops: Query<(&GlobalTransform, &Aabb), With<PipeTop>>,
    pipe_bottoms: Query<(&GlobalTransform, &Aabb), With<PipeBottom>>,
    mut am_ressource: NonSendMut<AMRessource>,
    mut registry: ResMut<DeadBirdRegistry>,
) {
    let all_birds: Vec<_> = birds.iter().collect();
    for (entity, bird, transform, velocity, bird_aabb, _) in &all_birds {
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
            bird_speed: calculated_velocity * 100.,
            next_pipe_top_y: nearest_top.edge_y,
            next_pipe_bottom_y: nearest_bot.edge_y,
            second_top_y: second_top.edge_y,
            second_bot_y: second_bot.edge_y,
        };

        //for debugging purposes only
        commands.entity(*entity).insert(state);
        //get an agent , sync it with Bird's pov
        am_ressource.agent_manager.bind_agent(bird.uid, state);
        let mut agent = am_ressource.agent_manager.inner[&bird.uid].borrow_mut();

        //forward

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

        /*let mut reward: f32 = if bird.dead && !bird.pipe_death {
            //dying soon is shameful
            RewardPrizes::default().dying * -(agent.state.score as i32) as f32
        } else {
            RewardPrizes::default().alive * (1 + bird.score) as f32
        }; */
        if bird.score > agent.state.score {
            reward += RewardPrizes::default().pipe_cleared;
            agent.state.score = bird.score;
        }

        if nearest_top.exists || nearest_bot.exists {
            let (lo, hi) = (
                state.next_pipe_top_y.min(state.next_pipe_bottom_y),
                state.next_pipe_top_y.max(state.next_pipe_bottom_y),
            );
            let center = (lo + hi) / 2.0;
            let half_gap = (hi - lo) / 2.0;
            let dist_ratio = (state.bird_y - center).abs() / half_gap;
            reward += RewardPrizes::default().alive * (1.0 - dist_ratio); // continuous, not binary
        } else {
            let half_canvas = CANVAS_SIZE[1] / 2.0;
            let sweet_half_height = 0.1 * PLAYER_SIZE; // 3*PLAYER_SIZE total band, centered on 0
            let dist_from_center = state.bird_y.abs();

            let alive = RewardPrizes::default().alive;

            reward += if dist_from_center <= sweet_half_height {
                alive // inside the sweet spot: full reward
            } else {
                // linearly decay from `alive` at the sweet-spot edge to 0 at the canvas edge
                let decay_range = half_canvas - sweet_half_height;
                let dist_past_sweet = dist_from_center - sweet_half_height;
                let falloff = (1.0 - dist_past_sweet / decay_range).max(0.0);
                alive * falloff
            };
        }
        let action = agent.select_action();
        //[...] for the sacred icon has been found
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", state).into());
        if bird.score > 15 {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(&format!("{:?}", agent.settings).into());
            panic!("with it, our path is clear")
        }

        match action {
            Action::DoNothing => {}
            Action::Jump => {
                if !bird.dead {
                    commands.trigger(BirdJump(*entity));
                }
            }
        }

        if action == Action::Jump {
            reward += RewardPrizes::default().jump_cost;
        }

        agent.record_step(action, reward);
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", reward).into());
    }
    //Gamestate has been processed , process bird's agent  stats
    let birds_dead: Vec<_> = all_birds
        .iter()
        .filter(|(_, bird, _, _, _, _)| bird.dead == true)
        .collect();

    // warn!("{:?}", birds_dead.len());

    for bird in &birds_dead {
        registry.0.insert(bird.1.uid);
        am_ressource.agent_manager.inner[&bird.1.uid]
            .borrow_mut()
            .agent_finish_episode();
    }
}
//this function is end of the line for async works : it hands off major tensor tasks to wasm worker before resuming them in another micro task loop
pub fn run_forwards_and_optims(
    mut registry: ResMut<DeadBirdRegistry>,
    mut am_ressource: NonSendMut<AMRessource>,
    mut live_inventory: ResMut<BirdInventory>,
) {
    let replacement = am_ressource.replacement.clone();
    let current_state = am_ressource.replacement.borrow().clone();
    let done: Rc<Cell<Option<bool>>> = Rc::new(Cell::new(None));
    let done_task = done.clone();
    match current_state {
        ReplacementState::Idle => {
            // ── 1. Pick the best agent (cheap, CPU-only — stays sync) ──

            let graded: GradedRunners =
                match am_ressource
                    .agent_manager
                    .inner
                    .iter()
                    .minmax_by(|(_, a), (_, b)| {
                        a.borrow_mut()
                            .stats
                            .total_score
                            .partial_cmp(&b.borrow_mut().stats.total_score)
                            .unwrap()
                    }) {
                    MinMaxResult::NoElements => {
                        // empty map — fall back to some sentinel, adjust as needed
                        panic!("agent_manager.inner is empty")
                    }
                    MinMaxResult::OneElement(_) => {
                        panic!("must have at least two agents ")
                    }
                    MinMaxResult::MinMax((worst_key, _), (best_key, _)) => GradedRunners {
                        best: *best_key,
                        worst: *worst_key,
                    },
                };

            *replacement.borrow_mut() = ReplacementState::Descending(graded.clone());
            //since model is 1-2 frames behind , the death reward must be forcefully recorded
            let a = am_ressource.agent_manager.inner[&graded.best].clone();
            let i = a.borrow().state.episode.len() - 1; // borrow(), not borrow_mut() — read-only
            a.borrow_mut().state.episode[i].reward = RewardPrizes::default().pipe_death;
            drop(a);
            let a = am_ressource.agent_manager.inner[&graded.best].clone();
            let replacement_task = replacement.clone();
            spawn_local(async move {
                let _loss: f32 = a.borrow_mut().update_policy().await;
                #[cfg(target_arch = "wasm32")]
                web_sys::console::log_1(&format!("descend loss: {:?}", _loss).into());

                *replacement_task.borrow_mut() = ReplacementState::Replacing(graded);
            });
        }

        ReplacementState::Descending(_) => {
            // Please standby
        }

        ReplacementState::Replacing(graded) => {
            match done_task.get() {
                None => {
                    done_task.set(Some(false));
                    let best_model = am_ressource.agent_manager.inner[&graded.best]
                        .borrow_mut()
                        .flappy
                        .clone();
                    let mut other_handles: Vec<Rc<RefCell<FlappyNet>>> = Vec::new();
                    let mut to_remove: Option<u32> = None;
                    for (key, agent) in am_ressource.agent_manager.inner.iter() {
                        agent.borrow_mut().agent_clear_episode();

                        if *key == graded.worst {
                            //this is the new innovator agent , it is out of replacement and will be assigned an agent in the next run
                            to_remove = Some(*key);
                            continue;
                        }
                        agent.borrow_mut().new_model_reset_stats();
                        other_handles.push(agent.borrow().flappy.clone());
                    }
                    if let Some(key) = to_remove {
                        am_ressource.agent_manager.inner.remove(&key);
                    }

                    spawn_local(async move {
                        for handle in other_handles.iter() {
                            *handle.borrow_mut() = best_model.borrow().clone();
                        }
                        done_task.set(Some(true));
                    });
                }
                Some(false) => {}
                Some(true) => {
                    let registry_snapshot: RegistrySnapshot = registry.0.clone();
                    let replacement_task = replacement.clone();
                    *replacement_task.borrow_mut() = ReplacementState::Ready(registry_snapshot);
                    drop(done_task)
                }
            }
        }
        ReplacementState::Ready(finalized) => {
            live_inventory.0 = finalized.clone();
            registry.0.clear();
            *replacement.borrow_mut() = ReplacementState::Idle;
        }
    }
    //clean slate
    // for (e, _) in pipes.iter() {
    //    commands.entity(e).despawn();
    // }
}
pub fn no_birds_in_main_system(
    alive: Query<&Bird>,
    live_inventory: Res<BirdInventory>,
    am_ressource: NonSend<AMRessource>,
    _commands: Commands,
) -> bool {
    let mut no_actions_in_flight = true;

    for (id, agent) in am_ressource.agent_manager.inner.iter() {
        match agent.try_borrow() {
            Ok(agent) => {
                let inflight = *agent.action_inflight.borrow();
                if inflight {
                    no_actions_in_flight = false;
                    break;
                }
            }
            Err(_e) => {
                no_actions_in_flight = false;
                #[cfg(target_arch = "wasm32")]
                web_sys::console::log_1(
                    &format!("failed to borrow an agent , need to wait its action to be complete.")
                        .into(),
                );
                break;
            }
        }
    }

    alive.is_empty() && live_inventory.0.is_empty() && no_actions_in_flight
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
    //conendrum: there might be pipes , but no next pipes : hence exists
    exists: bool,
    edge_y: f32,
    remaining_x: f32,
}

fn extract_top_pipe(top: Option<&(&GlobalTransform, &Aabb)>, bird_left_edge: f32) -> PipeEdgeInfo {
    match top {
        Some((t, aabb)) => PipeEdgeInfo {
            exists: true,
            edge_y: t.translation().y - aabb.half_extents.y,
            remaining_x: (t.translation().x + aabb.half_extents.x) - bird_left_edge,
        },
        None => PipeEdgeInfo {
            exists: false,
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
            exists: true,
            edge_y: t.translation().y + aabb.half_extents.y,
            remaining_x: (t.translation().x + aabb.half_extents.x) - bird_left_edge,
        },
        None => PipeEdgeInfo {
            exists: false,
            edge_y: -CANVAS_SIZE[1] / 2.0 - 1.0,
            remaining_x: -PLAYER_SIZE / 2.0 - 30.0,
        },
    }
}
