use super::pipes::*;
use crate::ml::agent_utils::{Action, GameStateFeatures, RewardPrizes};
use crate::ml::model::FlappyNet;
use crate::player::{
    Bird, BirdInventory, BirdJump, GameSets, PipeBottom, PipeTop, Player, Velocity,
};
use crate::RegistrySnapshot;
use crate::{AMRessource, ReplacementState};
use wasm_bindgen_futures::spawn_local;

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
pub struct BrainPlugin;

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
    birds: Query<(Entity, &Bird, &Transform, &Velocity, &Aabb), (With<Bird>, Without<Player>)>,

    pipe_tops: Query<(&GlobalTransform, &Aabb), With<PipeTop>>,
    pipe_bottoms: Query<(&GlobalTransform, &Aabb), With<PipeBottom>>,
    mut am_ressource: NonSendMut<AMRessource>,
    mut registry: ResMut<DeadBirdRegistry>,
) {
    let all_birds: Vec<_> = birds.iter().collect();
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

        /*         let mut reward: f32 = if bird.dead && !bird.pipe_death {
            //dying soon is shameful
            RewardPrizes::default().dying * -(agent.state.score as i32) as f32
        } else {
            RewardPrizes::default().alive * (1 + bird.score) as f32
        }; */
        if bird.score > agent.state.score {
            reward += RewardPrizes::default().pipe_cleared;
            agent.state.score = bird.score;
        }

        let (lo, hi) = (
            state.next_pipe_top_y.min(state.next_pipe_bottom_y),
            state.next_pipe_top_y.max(state.next_pipe_bottom_y),
        );
        let in_gap = state.bird_y > lo && state.bird_y < hi;
        // Small bonus for staying near the centre of the gap
        reward = reward + if in_gap { 0.01 } else { 0.0 };
        let agent = am_ressource.agent_manager.bind_agent(bird.uid, state);
        let action = agent.select_action();

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

        //Gamestate has been processed , process bird's agent  stats
        let birds_dead: Vec<_> = all_birds
            .iter()
            .filter(|(_, bird, _, _, _)| bird.dead == true)
            .collect();

        // warn!("{:?}", birds_dead.len());

        for bird in &birds_dead {
            registry.0.insert(bird.1.uid);
        }
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", "purgatory").into());
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", &birds_dead).into());
    }
}
//this function is end of the line for async works : it hands off major tensor tasks before
pub fn run_forwards_and_optims(
    mut registry: ResMut<DeadBirdRegistry>,
    am_ressource: NonSendMut<AMRessource>,
    mut live_inventory: ResMut<BirdInventory>,
) {
    // ── 0. Catch a precedent future, if one landed since we last ran.
    //    This is the ONLY place the registries are touched, and it only
    //    fires once a background replacement has actually finished.
    {
        let mut slot = am_ressource.replacement.borrow_mut();
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", "slot").into());
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", &*slot).into());
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", "live").into());
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", &live_inventory.0).into());
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", "registry").into());
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!("{:?}", &registry.0).into());
        if let ReplacementState::Ready(finalized) = &*slot {
            live_inventory.0 = finalized.clone();
            registry.0.clear();
            *slot = ReplacementState::Idle;

            return;
        }
    }

    // ── 1. Don't start a new replacement cycle while one is in flight.
    if !matches!(*am_ressource.replacement.borrow(), ReplacementState::Idle) {
        return;
    }

    // ── 2. Pick the best agent (cheap, CPU-only — stays sync) ──────────
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
    // Cloning the Rc handle itself (cheap, just a refcount bump) — the
    // actual weights get cloned once, inside the background task, per
    // recipient, matching your original semantics.
    let best_model = am_ressource.agent_manager.inner[&best_idx].flappy.clone();

    let other_handles: Vec<Rc<RefCell<FlappyNet>>> = am_ressource
        .agent_manager
        .inner
        .iter()
        .filter(|(key, _)| **key != best_idx)
        .map(|(_, agent)| agent.flappy.clone())
        .collect();

    // `registry`/`live_inventory` (the ResMuts) don't live long enough —
    // snapshot the owned data the background task needs to hand back.
    let registry_snapshot: RegistrySnapshot = registry.0.clone();

    let replacement = am_ressource.replacement.clone();
    *replacement.borrow_mut() = ReplacementState::InFlight;

    spawn_local(async move {
        // ── operate the replacement ─────────────────────────────────────
        for handle in other_handles.iter() {
            *handle.borrow_mut() = best_model.borrow().clone();
        }

        // ── stage the finalize; step 0 on a later frame will pick it up ──
        *replacement.borrow_mut() = ReplacementState::Ready(registry_snapshot);
    });
}
pub fn no_birds_in_main_system(
    alive: Query<&Bird>,
    live_inventory: Res<BirdInventory>,

    _commands: Commands,
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
