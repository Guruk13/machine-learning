use std::time::Duration;

use bevy::{image::ImageLoaderSettings, prelude::*, time::common_conditions::on_timer};

use burn::prelude::Backend;
use burn::tensor::{Distribution, Tensor, Tolerance};
use burn::backend::Autodiff;

use burn::backend::wgpu::{Wgpu,WgpuDevice, WgpuRuntime};

pub mod ml;
use crate::ml::*;

use crate::ml::multiagent::AgentManager;

pub mod player;
use crate::player::*;
use burn::tensor::backend::AutodiffBackend;

pub const CANVAS_SIZE: Vec2 = Vec2::new(480., 270.);
pub const PLAYER_SIZE: f32 = 25.0;
const PIPE_SIZE: Vec2 = Vec2::new(32., CANVAS_SIZE.y);
const GAP_SIZE: f32 = 100.0;
pub const PIPE_SPEED: f32 = 200.0;

pub struct AMRessource<B: AutodiffBackend> {
    agent_manager: AgentManager<B>,
}

/* impl<B: AutodiffBackend<Device = WgpuDevice>> AMRessource<B> {
    pub fn new() -> Self {
        let device = WgpuDevice::default();
        Self {
            agent_manager: AgentManager::new(device),
        }
    }
} */
type MyBackend = burn_wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>;
type MyAutodiffBackend = burn_autodiff::Autodiff<MyBackend>;

impl<B> Resource for AMRessource<B>
where
    B: AutodiffBackend + Send + Sync + 'static,
    AgentManager<B>: Send + Sync,
{
}

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
        let device = WgpuDevice::default();

        let device: <MyAutodiffBackend as burn::tensor::backend::Backend>::Device =
            Default::default();

        let resource = AMRessource::<MyAutodiffBackend> {
            agent_manager: AgentManager::new(device.clone()),
        };

        app.insert_non_send_resource(ressource);
        //app.add_observer(attach_episodes);
        app.add_systems(FixedUpdate, (think).in_set(player::GameSets::AI));
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

fn spawn_pipes(mut commands: Commands, asset_server: Res<AssetServer>, time: Res<Time>) {
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
    let pipe_offset = PIPE_SIZE.y / 2.0 + GAP_SIZE / 2.0;

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
                    custom_size: Some(Vec2::new(10.0, GAP_SIZE,)),
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

/* fn bird_bind_agent(query: Query<Entity, Added<Bird>>,  mut resam:  NonSend<AMRessource<B: AutodiffBackend>>,) {
    for entity in &query {
        println!("Enemy spawned: {:?}", entity);
    }
} */

//  Birds think .  what will you have after 500 years ?
pub fn think(
    mut commands: Commands,

    birds: Query<(&Transform, &Velocity, Entity), (With<Bird>, Without<Player>)>,
    pipe_tops: Query<&GlobalTransform, With<PipeTop>>,
    pipe_bottoms: Query<&GlobalTransform, With<PipeBottom>>,
    brain: NonSend<AMRessource<B>>,
) {
    //collect GameState for each bird
    for (transform, velocity, entity) in &birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0).to_angle();
        let bird_y = transform.translation.y;
        let bird_x = transform.translation.x;
        let nearest_top = pipe_tops.iter().find(|t| t.translation().x > bird_x);
        let nearest_bottom = pipe_bottoms.iter().find(|t| t.translation().x > bird_x);
        let dist_to_top = nearest_top
            .map(|t| t.translation().y - bird_y)
            .unwrap_or(f32::MAX);

        let dist_to_bottom = nearest_bottom
            .map(|t| bird_y - t.translation().y)
            .unwrap_or(f32::MAX);

        //warn!("{:#?}", flap);
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
        //do the actual thinking

        /* let tensor = state.to_tensor(&device);
        let tensor = brain.model.forward(tensor); */

        //record game state
        // in your game update system:

        //LIFO 3 seconds
        /*             episode.steps.push(Step {
            state: [state.bird_y, state.bird_speed,
                    state.next_pipe_top_y, state.next_pipe_bottom_y,
                    state.next_pipe_distance],
            jumped,
            prob,
            reward: 0.0, // filled in later
        }); */

        //shine on
        if brain.model.should_jump(tensor) {
            commands.trigger(BirdJump(entity));
        } else {
            warn!("not because you pelican means you pelishould");
        }
    }
}
