use super::pipes::*;
use crate::ml::agent_utils::{Action, GameStateFeatures, RewardPrizes};
use crate::ml::model::FlappyNet;
use crate::player::{
    Bird, BirdInventory, BirdJump, GameSets, Lifetime, Opacity, Pipe, PipeBottom, PipeTop, Player,
    Velocity,
};
use crate::RegistrySnapshot;
use crate::{AMRessource, ReplacementState};
use std::collections::HashMap;
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
    mut birds: Query<
        (
            Entity,
            &Bird,
            &Transform,
            &Velocity,
            &Aabb,
            &Lifetime,
            &mut Opacity,
        ),
        (With<Bird>, Without<Player>, Changed<Transform>),
    >,

    pipe_tops: Query<(&GlobalTransform, &Aabb), With<PipeTop>>,
    pipe_bottoms: Query<(&GlobalTransform, &Aabb), With<PipeBottom>>,
    mut am_ressource: NonSendMut<AMRessource>,
    mut registry: ResMut<DeadBirdRegistry>,
) {
    let mut all_birds: Vec<_> = birds.iter_mut().collect();
    //linear distribution
    let n = all_birds.iter().count();
    let denom = (n.saturating_sub(1)).max(1) as f32; // avoid divide by zero when N == 1
    for (entity, bird, transform, velocity, bird_aabb, _lifetime, opacity) in all_birds.iter_mut() {
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

        // Active pipes
        let nearest_top = extract_pipe(
            tops_ahead.first().map(|x| x),
            PipeKind::Top,
            transform.translation.y,
            bird_left_edge,
        );
        let nearest_bot = extract_pipe(
            bots_ahead.first().map(|x| x),
            PipeKind::Bottom,
            transform.translation.y,
            bird_left_edge,
        );

        let state = GameStateFeatures {
            bird_speed: calculated_velocity,
            next_pipe_top_y: nearest_top.edge_y,
            next_pipe_bottom_y: nearest_bot.edge_y,
            remaining_bot_x: nearest_bot.remaining_x,
            remaining_top_x: nearest_top.remaining_x,
        };

        //get an agent , sync it with Bird's pov
        am_ressource.agent_manager.bind_agent(bird.uid, state);
        let mut agent = am_ressource.agent_manager.inner[&bird.uid].borrow_mut();
        let rank = agent.stats.rank as f32;
        opacity.value = 100.0 * (1.0 - rank / denom);

        //compute reward
        let mut reward: f32 = if bird.dead {
            //dying soon is shameful
            if bird.pipe_death {
                RewardPrizes::default().pipe_death
            } else {
                RewardPrizes::default().dying
            }
        } else {
            0.0
        };

        if bird.score > agent.state.score {
            reward += RewardPrizes::default().pipe_cleared;
            agent.state.score = bird.score;
        }

        let alive = RewardPrizes::default().alive;

        let in_zone = {
            let lo = if nearest_bot.exists {
                state.next_pipe_bottom_y
            } else {
                -CANVAS_SIZE.y / 2.0 * 0.25
            };
            let hi = if nearest_top.exists {
                state.next_pipe_top_y
            } else {
                CANVAS_SIZE.y / 2.0 * 0.25
            };
            (lo..=hi).contains(&transform.translation.y)
        };

        if in_zone {
            reward += alive;
        }

        let action = agent.select_action();

        if bird.score % 10 == 0 && bird.score != 0 {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::log_1(
                &format!("{:?} {:?}, {:?}", bird.score, bird.uid, agent.settings).into(),
            );
            //panic!("with it, our path is clear")
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
        //#[cfg(target_arch = "wasm32")]
        //web_sys::console::log_1(&format!("{:?}", reward).into());
    }
    //Gamestate has been processed , process bird's agent  stats
    let birds_dead: Vec<_> = all_birds
        .iter()
        .filter(|(_, bird, _, _, _, _, _)| bird.dead == true)
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
    pipes: Query<(Entity, &Transform), With<Pipe>>,
    mut commands: Commands,
) {
    thread_local! {
        static DONE: Rc<Cell<Option<bool>>> = Rc::new(Cell::new(None));
        // incremented after each successful batch of descent
        static EPOCH: Cell<u32> = Cell::new(0);
    }
    const CULL_EVERY: u32 = 3;

    let done = DONE.with(Rc::clone);
    let done_task = done.clone();

    let replacement = am_ressource.replacement.clone();
    let current_state = am_ressource.replacement.borrow().clone();

    match current_state {
        ReplacementState::Idle => {
            if am_ressource.agent_manager.inner.len() < 2 {
                panic!("must have at least two agents");
            }

            *replacement.borrow_mut() = ReplacementState::Descending;

            // every agent descends on its own episode
            let agents: Vec<_> = am_ressource.agent_manager.inner.values().cloned().collect();
            let replacement_task = replacement.clone();
            let registry_snapshot: RegistrySnapshot = registry.0.clone();
            //
            EPOCH.with(|e| e.set(e.get() + 1));
            let epoch = EPOCH.with(|e| e.get());
            let cull_epoch = epoch > 0 && epoch % CULL_EVERY == 0;

            spawn_local(async move {
                let mut _total = 0.0f32;
                let _n = agents.len() as f32;
                for (_, a) in agents.iter().enumerate() {
                    if a.borrow().state.episode.is_empty() {
                        continue;
                    }

                    _total += a.borrow_mut().update_policy().await;

                    a.borrow_mut().agent_clear_episode();
                    if !cull_epoch {
                        a.borrow_mut().reset_score();
                    }
                }

                if !cull_epoch {
                    let replacement_task = replacement.clone();
                    *replacement_task.borrow_mut() = ReplacementState::Ready(registry_snapshot);
                } else {
                    *replacement_task.borrow_mut() = ReplacementState::Replacing;
                }
            });
        }

        ReplacementState::Descending => {
            // Please standby
        }

        ReplacementState::Replacing => {
            match done_task.get() {
                None => {
                    // (target handle, source handle)
                    let mut copies: Vec<(Rc<RefCell<FlappyNet>>, Rc<RefCell<FlappyNet>>)> =
                        Vec::new();
                    let bottom: HashSet<u32>;
                    let mut middle_sources: HashMap<u32, Rc<RefCell<FlappyNet>>> = HashMap::new();

                    done_task.set(Some(false));
                    // rank worst -> best (stats are still intact: reset happens below)
                    let mut ranked: Vec<(u32, f32)> = am_ressource
                        .agent_manager
                        .inner
                        .iter()
                        .map(|(k, a)| (*k, a.borrow().stats.total_score))
                        .collect();
                    ranked
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                    am_ressource.agent_manager.inner.iter().for_each(|(k, a)| {
                        a.borrow_mut().stats.rank = ranked[*k as usize].0;
                    });

                    let n = ranked.len();
                    let third = (n / 3).max(1);
                    let top_start = n - third;
                    let mid_start = top_start.saturating_sub(third);

                    let keys: Vec<u32> = ranked.iter().map(|(k, _)| *k).collect();
                    let top: Vec<u32> = keys[top_start..].iter().rev().copied().collect(); // best-first
                    let middle: Vec<u32> = keys[mid_start..top_start].to_vec();
                    bottom = keys[..mid_start].iter().copied().collect();

                    // middle third <- clones of top third (round-robin)
                    for (i, k) in middle.iter().enumerate() {
                        let src_key = top[i % top.len()];
                        let src = am_ressource.agent_manager.inner[&src_key]
                            .borrow()
                            .flappy
                            .clone();
                        middle_sources.insert(*k, src);
                    }

                    let mut to_remove: Vec<u32> = Vec::new();
                    for (key, agent) in am_ressource.agent_manager.inner.iter() {
                        if bottom.contains(key) {
                            // removed, new agents are created in the next run
                            to_remove.push(*key);
                            continue;
                        }
                        agent.borrow_mut().new_model_reset_stats();

                        if let Some(src) = middle_sources.get(key) {
                            copies.push((agent.borrow().flappy.clone(), src.clone()));
                        }
                    }
                    for key in to_remove {
                        am_ressource.agent_manager.inner.remove(&key);
                    }

                    spawn_local(async move {
                        for (target, source) in copies.iter() {
                            let model = source.borrow().clone(); // guard dropped at end of statement
                            *target.borrow_mut() = model;
                        }
                        done_task.set(Some(true));
                    });
                }
                Some(false) => {}
                Some(true) => {
                    let registry_snapshot: RegistrySnapshot = registry.0.clone();
                    let replacement_task = replacement.clone();
                    *replacement_task.borrow_mut() = ReplacementState::Ready(registry_snapshot);
                    done.set(None);
                }
            }
        }

        ReplacementState::Ready(finalized) => {
            live_inventory.0 = finalized.clone();
            registry.0.clear();
            *replacement.borrow_mut() = ReplacementState::Idle;
        }
    }

    // clean slate
    for (e, _) in pipes.iter() {
        commands.entity(e).despawn();
    }
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
                //#[cfg(target_arch = "wasm32")]
                //web_sys::console::log_1(
                // &format!( "failed to borrow agent {:?} , need to wait its
                // action to be complete.", id ) .into(),
                //);
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

#[derive(Clone, Copy)]
enum PipeKind {
    Top,    // gap ceiling = bottom edge of the pipe
    Bottom, // gap floor   = top edge of the pipe
}

fn extract_pipe(
    pipe: Option<&(&GlobalTransform, &Aabb)>,
    kind: PipeKind,
    bird_y: f32,
    bird_left_edge: f32,
) -> PipeEdgeInfo {
    match pipe {
        Some((t, aabb)) => {
            let center = t.translation() + Vec3::from(aabb.center);
            let half = Vec3::from(aabb.half_extents) * t.compute_transform().scale;

            let edge_y = match kind {
                PipeKind::Top => center.y - half.y,
                PipeKind::Bottom => center.y + half.y,
            };

            PipeEdgeInfo {
                exists: true,
                edge_y: edge_y - bird_y,
                remaining_x: (center.x + half.x) - bird_left_edge,
            }
        }
        None => PipeEdgeInfo {
            exists: false,
            edge_y: match kind {
                PipeKind::Top => CANVAS_SIZE[1] / 2.0,     // canvas ceiling
                PipeKind::Bottom => -CANVAS_SIZE[1] / 2.0, // canvas floor
            } - bird_y,
            remaining_x: CANVAS_SIZE[0],
        },
    }
}
