use bevy::{image::ImageLoaderSettings, prelude::*};

use burn::backend::Autodiff;
use burn::backend::wgpu::Wgpu;

pub mod ml;
use crate::ml::*;

use crate::ml::multiagent::AgentManager;

pub mod player;
use crate::player::*;
use burn::tensor::backend::AutodiffBackend;

pub const CANVAS_SIZE: Vec2 = Vec2::new(480., 270.);
pub const PLAYER_SIZE: f32 = 25.0;
const PIPE_SIZE: Vec2 = Vec2::new(32., CANVAS_SIZE.y);
const _GAP_SIZE: f32 = 100.0;
pub const PIPE_SPEED: f32 = 200.0;

pub struct AMRessource<B: AutodiffBackend> {
    agent_manager: AgentManager<B>,
}

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

unsafe impl<B: AutodiffBackend> Sync for AMRessource<B> {}

pub struct PipePlugin;

impl Plugin for PipePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (
                despawn_pipes,
                shift_pipes_to_the_left,
                //spawn_pipes.run_if(on_timer(Duration::from_millis(1000))),
            ),
        );
    }
}

pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        let device: <MyAutodiffBackend as burn::tensor::backend::Backend>::Device =
            Default::default();

        let ressource = AMRessource::<MyAutodiffBackend> {
            agent_manager: AgentManager::new(device),
        };

        app.insert_non_send_resource(ressource);
        app.add_systems(Update, bird_alive_reward);
        app.add_observer(bird_death_reward);
        app.add_observer(bird_pass_reward);
        app.add_systems(Update, bird_bind_agent);
        app.add_systems(
            Update,
            (bird_alive_reward, ApplyDeferred, sync_agent_state).chain(),
        );
        app.add_systems(FixedUpdate, think.in_set(GameSets::AI));
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
#[derive(Component, Default)]
pub struct BirdEpisode {
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone)]
pub struct Step {
    pub state: GameStateFeatures,
    pub jumped: bool,
    pub prob: f32, // the raw sigmoid output at that frame
    pub reward: f32,
}

fn _spawn_pipes(mut commands: Commands, asset_server: Res<AssetServer>, time: Res<Time>) {
    let image =
        asset_server.load_with_settings("pipe.png", |settings: &mut ImageLoaderSettings| {
            settings
                .sampler
                .get_or_init_descriptor()
                .set_filter(bevy::image::ImageFilterMode::Nearest);
        });

    let image_mode = SpriteImageMode::Sliced(TextureSlicer {
        border: BorderRect::axes(8., 19.),
        center_scale_mode: SliceScaleMode::Stretch,
        ..default()
    });

    let transform = Transform::from_xyz(CANVAS_SIZE.x / 2., 0.0, 1.0);
    let gap_y_position = (time.elapsed_secs() * 4.2309875).sin() * CANVAS_SIZE.y / 4.;
    let pipe_offset = PIPE_SIZE.y / 2.0 + _GAP_SIZE / 2.0;

    commands.spawn((
        transform,
        Visibility::Visible,
        Pipe,
        children![
            (
                Sprite {
                    image: image.clone(),
                    custom_size: Some(PIPE_SIZE),
                    image_mode: image_mode.clone(),
                    ..default()
                },
                Transform::from_xyz(0.0, pipe_offset + gap_y_position, 1.0,),
                PipeTop
            ),
            (
                Visibility::Hidden,
                Sprite {
                    color: Color::WHITE,
                    custom_size: Some(Vec2::new(10.0, _GAP_SIZE,)),
                    ..default()
                },
                Transform::from_xyz(0.0, gap_y_position, 1.0,),
                PointsGate,
            ),
            (
                Sprite {
                    image,
                    custom_size: Some(PIPE_SIZE),
                    image_mode,
                    ..default()
                },
                Transform::from_xyz(0.0, -pipe_offset + gap_y_position, 1.0,),
                PipeBottom,
            )
        ],
    ));
}

pub fn shift_pipes_to_the_left(mut pipes: Query<&mut Transform, With<Pipe>>, time: Res<Time>) {
    for mut pipe in &mut pipes {
        pipe.translation.x -= PIPE_SPEED * time.delta_secs();
    }
}

fn despawn_pipes(mut commands: Commands, pipes: Query<(Entity, &Transform), With<Pipe>>) {
    for (entity, transform) in pipes.iter() {
        if transform.translation.x < -(CANVAS_SIZE.x / 2.0 + PIPE_SIZE.x) {
            commands.entity(entity).despawn();
        }
    }
}

fn bird_bind_agent(
    mut commands: Commands,
    query: Query<Entity, Added<Bird>>,
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
) {
    //warn!("Nuée de : {:?}",&query.iter().count());
    for entity in &query {
        am_ressource
            .agent_manager
            .bind_agent(entity.index().index());
        commands.entity(entity).insert(AgentState::new(false));

        //warn!("Debout , joli bouton d'or : {:?}",entity.index().to_string());
    }
}

//  Birds think .  what will you have after 500 years ?

fn think(
    mut commands: Commands,
    birds: Query<
        (Entity, &GameStateFeatures, &PartialReward, &AgentState),
        (With<Bird>, Without<Player>),
    >,
    mut am_ressource: NonSendMut<AMRessource<MyAutodiffBackend>>,
) {
    //make birds think
    let all_birds: Vec<_> = birds.iter().collect();

    //warn!( "Look mom , no .... : {:?}",query.iter().count);
    for (entity, state, reward, dead_state) in &all_birds {
        let action: Action = am_ressource
            .agent_manager
            .select_action(&entity.index().index(), &state);
        //@todo match dead birds and exclude them from action "Jump"
        match action {
            Action::DoNothing => { /*  not because you pelican means you pelishould */ }
            Action::Jump => {
                //warn!("Look mom , no user input  : {:?}", entity.index().to_string());
                if !dead_state.is_dead {
                    commands.trigger(BirdJump(*entity));
                }
            }
        }

        // Small bonus for staying near the centre of the gap
        let gap_centre_penalty = (state.next_pipe_top_y - state.next_pipe_bottom_y).abs() * 0.001;
        let reward = **reward - gap_centre_penalty;
        //warn!( "Sad {:?}",state);
        am_ressource
            .agent_manager
            .record_step(entity.index().index(), **state, action, reward);
        commands.entity(*entity).remove::<PartialReward>();
        //Gamestate has been processed , process bird's agent  stats
        let birds_dead: Vec<_> = all_birds
            .iter()
            .filter(|(_, _, _, dead)| dead.is_dead == true)
            .collect();

        for bird in &birds_dead {
            am_ressource.agent_manager.bird_died(bird.0.index().index());
        }

        if !&birds_dead.is_empty() {
            // drop the &
            am_ressource.agent_manager.update_stats();
            am_ressource.agent_manager.prune_agents();
        }

        for bird in &birds_dead {
            am_ressource
                .agent_manager
                .clear_episode(bird.0.index().index());
            commands.entity(bird.0).insert(AgentState::new(false));
        }
    }
}

/* there is a duality , events and rewards are computed on "framerate update", "thinking" is a bruteforce thread which consumes ressources far too fast.
in the system and event catcher below a RewardHolder is added , on fixed update (bruteforce) we look for component and run the computation if it is found
. Game and Ai thread should balance out and meet in the middle with this hack.  */

#[derive(Component)]
pub struct AgentState {
    pub is_dead: bool,
    // you could fold GameStateFeatures in here too
    //
}
impl AgentState {
    pub fn new(is_dead: bool) -> Self {
        Self { is_dead: is_dead }
    }
}

fn sync_agent_state(mut commands: Commands, agents: Query<(Entity, &Dead), With<Bird>>) {
    for bird in &agents {
        commands.entity(bird.0).insert(AgentState::new(true));
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
    pipe_tops: Query<&GlobalTransform, With<PipeTop>>,
    pipe_bottoms: Query<&GlobalTransform, With<PipeBottom>>,
) {
    for (transform, velocity, entity, maybe_reward) in &birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0).to_angle();
        let bird_y = transform.translation.y;
        let bird_x = transform.translation.x;
        let nearest_top = pipe_tops.iter().find(|t| t.translation().x > bird_x);
        let nearest_bottom = pipe_bottoms.iter().find(|t| t.translation().x > bird_x);
        let dist_to_top = nearest_top
            .map(|t| t.translation().y - bird_y)
            .unwrap_or(500.0);

        let dist_to_bottom = nearest_bottom
            .map(|t| bird_y - t.translation().y)
            .unwrap_or(500.0);
        let nearest_top_y = nearest_top.map(|t| t.translation().y).unwrap_or(-1.0);
        let nearest_bottom_y = nearest_bottom.map(|t| t.translation().y).unwrap_or(-1.0);
        //warn!("{:#?}", nearest_bottom_y);
        //warn!("{:#?}", nearest_top_y);
        let state = GameStateFeatures {
            bird_y: transform.translation.y,
            bird_speed: calculated_velocity, //calculated_velocity,
            next_pipe_top_y: nearest_top_y,
            next_pipe_bottom_y: nearest_bottom_y,
            dist_top: dist_to_top,
            dist_bot: dist_to_bottom,
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
    birds: Query<(Entity, Option<&PartialReward>), With<Bird>>,
) {
    if let Ok((_entity, partial_reward)) = birds.get(entity_event.bird) {
        let new_score = match partial_reward {
            Some(score) => PartialReward(score.0 + RewardPrizes::default().dying),
            None => PartialReward(RewardPrizes::default().dying),
        };
        commands.entity(entity_event.bird).insert(new_score);
        commands.entity(entity_event.bird).insert(Dead);
        commands
            .entity(entity_event.bird)
            .insert(AgentState::new(true));
    }
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
