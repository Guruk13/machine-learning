use super::pipes::*;
use crate::ml::agent_utils::Action;
use crate::player::{
    Bird, BirdDeath, BirdInventory, BirdJump, GameSets, Player, ScorePoint, Velocity,
};
use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;
use burn::tensor::backend::AutodiffBackend;

use crate::ml::multiagent::AgentManager;

use crate::ml::agent_utils::{RewardPrizes, GameStateFeatures} ;

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
        app.add_observer(bird_death_reward);
        app.add_observer(bird_pass_reward);
        app.add_systems(
            FixedUpdate,
            (ApplyDeferred, bird_bind_agent, bird_alive_reward, think)
                .chain()
                .in_set(GameSets::AI),
        );
    }
}

#[derive(Component)]
pub struct Pipe;

#[derive(Component)]
pub struct PipeTop;

#[derive(Component)]
pub struct PipeBottom;

#[derive(Component)]
pub struct PointsGate;

//memories of a bird, "This is essentially a simple form of A3C (Asynchronous Advantage Actor-Critic)"
//#[derive(Component, Default)]
//pub struct BirdEpisode {
//    pub steps: Vec<Step>,
//}

//this might spawn zombies if no proper check
fn bird_bind_agent(
    mut commands: Commands,
    query: Query<(Entity, &Bird), Added<Bird>>,
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
) {
    //warn!("Nuée de : {:?}",&query.iter().count());
    for entity in &query {
        am_ressource.agent_manager.bind_agent(entity.1.uid);
        commands.entity(entity.0).insert(AgentState::new(false));

        //warn!("Debout , joli bouton d'or : {:?}",entity.index().to_string());
    }
}

//  Birds think .  what will you have after 500 years ?

fn think(
    mut commands: Commands,
    birds: Query<
        (
            Entity,
            &GameStateFeatures,
            &PartialReward,
            &Bird,
            &AgentState,
        ),
        (With<Bird>, Without<Player>),
    >,
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
) {
    let all_birds: Vec<_> = birds.iter().collect();

    //warn!( "Look mom , no .... : {:?}",query.iter().count);
    for (entity, state, reward, bird, dead_state) in &all_birds {
        //warn!("{:?}", state);
        let uid = bird.uid;
        let action: Action = am_ressource.agent_manager.select_action(uid, &state);
        //@todo match dead birds and exclude them from action "Jump"
        match action {
            Action::DoNothing => { /*  not because you pelican means you pelishould */ }
            Action::Jump => {
                if !dead_state.is_dead {
                    commands.trigger(BirdJump(*entity));
                }
            }
        }
        let gap_centre = (state.next_pipe_top_y + state.next_pipe_bottom_y) / 2.0;
        let gap_centre_penalty = (state.bird_y - gap_centre).abs() * 0.001;
        // Small bonus for staying near the centre of the gap
        let reward = **reward - gap_centre_penalty;

        am_ressource
            .agent_manager
            .record_step(uid, **state, action, reward);
        commands.entity(*entity).remove::<PartialReward>();
    }
    //Gamestate has been processed , process bird's agent  stats
    let birds_dead: Vec<_> = all_birds
        .iter()
        .filter(|(_, _, _, _, dead)| dead.is_dead == true)
        .collect();

    for bird in &birds_dead {
        am_ressource.agent_manager.bird_died(bird.3.uid);
    }
    // warn!("{:?}", birds_dead.len());

    if !&birds_dead.is_empty() {
        am_ressource.agent_manager.update_stats();
        am_ressource.agent_manager.prune_agents();
    }

    for bird in &birds_dead {
        am_ressource.agent_manager.clear_episode(bird.3.uid);
        commands.entity(bird.0).insert(AgentState::new(false));
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

// bird is alive: reward it ,  snapshot its POV
fn bird_alive_reward(
    mut commands: Commands,
    birds: Query<
        (&Transform, &Velocity, Entity, Option<&PartialReward>),
        (With<Bird>, Without<Player>),
    >,
    pipe_tops: Query<(&GlobalTransform, &Aabb), With<PipeTop>>,
    pipe_bottoms: Query<(&GlobalTransform, &Aabb), With<PipeBottom>>,
) {
    for (transform, velocity, entity, maybe_reward) in &birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0).to_angle();
        let bird_y = transform.translation.y;
        let bird_x = transform.translation.x;
        let nearest_top = pipe_tops.iter().find(|t| t.0.translation().x > bird_x);
        let nearest_bottom = pipe_bottoms.iter().find(|t| t.0.translation().x > bird_x);

        let dist_x = nearest_top
            .map(|t| t.0.translation().x - bird_x)
            .unwrap_or(500.0);

        // Then for top pipe: bottom edge = translation.y - half_extents.y
        let nearest_top_edge = nearest_top
            .map(|(t, aabb)| t.translation().y - aabb.half_extents.y)
            .unwrap_or(500.0);

        // For bottom pipe: top edge = translation.y + half_extents.y
        let nearest_bottom_edge = nearest_bottom
            .map(|(t, aabb)| t.translation().y + aabb.half_extents.y)
            .unwrap_or(-500.0);

        let state = GameStateFeatures {
            bird_y: transform.translation.y,
            bird_speed: calculated_velocity, //calculated_velocity,
            next_pipe_top_y: nearest_top_edge,
            next_pipe_bottom_y: nearest_bottom_edge,
            dist_x: dist_x,
        };
        commands.entity(entity).insert(state);

        let new_score = match maybe_reward {
            Some(score) => PartialReward(score.0 + RewardPrizes::default().alive), // increment
            None => PartialReward(RewardPrizes::default().alive),                  // insert fresh
        };

        commands.entity(entity).insert(new_score);
    }
}
//@Todo extract gameplay insert dead
fn bird_death_reward(
    entity_event: On<BirdDeath>,
    mut commands: Commands,
    birds: Query<(Entity, &Bird, Option<&PartialReward>)>,
    mut birdinv: ResMut<BirdInventory>,
) {
    if let Ok((entity, bird, partial_reward)) = birds.get(entity_event.bird) {
        let new_score = match partial_reward {
            Some(score) => PartialReward(score.0 + RewardPrizes::default().dying),
            None => PartialReward(RewardPrizes::default().dying),
        };
        commands.entity(entity_event.bird).insert(new_score);

        if !birdinv.0.contains(&bird.uid) {
            birdinv.0.push(bird.uid);
        }


fn bird_pass_reward(
    entity_event: On<ScorePoint>,
    mut commands: Commands,
    birds: Query<(Entity, Option<&PartialReward>), With<Bird>>,
) {
    if let Ok((_entity, partial_reward)) = birds.get(entity_event.bird) {
        let new_score = match partial_reward {
            Some(score) => PartialReward(score.0 + RewardPrizes::default().pipe_cleared),
            None => PartialReward(RewardPrizes::default().pipe_cleared),
        };
        commands.entity(entity_event.bird).insert(new_score);
    }
}
