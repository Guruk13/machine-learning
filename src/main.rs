#![allow(warnings)]
use bevy::{
    camera::ScalingMode,
    image::ImageAddressMode,
    math::bounding::{Aabb2d, BoundingCircle, IntersectsVolume},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
    sprite_render::{Material2d, Material2dPlugin},
};
use bevy::{color::palettes::tailwind::RED_400, image::ImageLoaderSettings};
use flappy_bird::{ml::agent_utils::GameStateFeatures, *};

use crate::ml::multiagent::set_global_device;
use crate::ml::multiagent::AMRessource;
use crate::ml::multiagent::AgentManager;
use crate::player::Pipe;
use crate::player::*;
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Device;
use flappy_bird::BrainPlugin;
use std::rc::Rc;
use std::time::Duration;
use std::{cell::RefCell, collections::HashSet};

#[derive(Resource, Clone)]
pub struct BurnDevice(pub WgpuDevice);
fn main() {
    // sync main — just kicks off the async work and doesn't wait for it
    wasm_bindgen_futures::spawn_local(async_main());
}
async fn async_main() {
    let device: Device<MyAutodiffBackend> = Default::default();
    burn::backend::wgpu::init_setup_async::<burn::backend::wgpu::graphics::WebGpu>(
        &device,
        Default::default(),
    )
    .await;
    set_global_device(device.clone()); // still needed for code outside Bevy

    App::new()
        // Plugins
        .add_plugins(DefaultPlugins)
        .add_plugins((
            PipePlugin,
            BrainPlugin,
            Material2dPlugin::<BackgroundMaterial>::default(),
        ))
        // Resources
        .insert_resource(BurnDevice(device))
        .insert_resource(BirdInventory(HashSet::new()))
        .insert_resource(GamemodeResource(GameMode::train()))
        .init_resource::<Score>()
        // Set order
        .configure_sets(
            FixedUpdate,
            (
                GameSets::Input,
                GameSets::Game,
                GameSets::AI,
                GameSets::Cleanup,
            )
                .chain(),
        )
        // Startup
        .add_systems(Startup, (startup, set_time_scale))
        // Update
        .add_systems(
            Update,
            (
                best_bird,
                enforce_bird_direction,
                toggle_pause,
                controls
                    .run_if(any_with_component::<Player>)
                    .in_set(GameSets::Input),
            ),
        )
        // FixedUpdate
        .add_systems(
            FixedUpdate,
            (gravity, check_in_bounds, check_collisions).in_set(GameSets::Game),
        )
        // FixedPostUpdate
        .add_systems(FixedPostUpdate, (despawn_deads).in_set(GameSets::Cleanup))
        // Observers
        .add_observer(spawn_birds)
        .add_observer(endgame::<SwitchGamemode>)
        .add_observer(endgame::<EndGame>)
        .add_observer(on_bird_jump)
        .add_observer(on_score)
        .run();
}

fn startup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut config_store: ResMut<GizmoConfigStore>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
) {
    let (config, _) = config_store.config_mut::<DefaultGizmoConfigGroup>();
    config.enabled = false;

    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scaling_mode: ScalingMode::AutoMax {
                max_width: CANVAS_SIZE.x,
                max_height: CANVAS_SIZE.y,
            },
            ..OrthographicProjection::default_2d()
        }),
    ));

    commands.spawn((
        Node {
            width: percent(100.),
            margin: px(20.).top(),
            ..default()
        },
        Text::new("0"),
        TextLayout::new(Justify::Center, LineBreak::AnyCharacter),
        TextFont {
            font_size: bevy::prelude::FontSize::Px(33.0),
            ..default()
        },
        TextColor(Srgba::hex("#282828").unwrap().into()),
        ScoreText,
    ));

    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(CANVAS_SIZE.x, CANVAS_SIZE.x))),
        MeshMaterial2d(materials.add(BackgroundMaterial {
            color_texture: asset_server.load_with_settings(
                "background_color_grass.png",
                |settings: &mut ImageLoaderSettings| {
                    settings
                        .sampler
                        .get_or_init_descriptor()
                        .set_address_mode(ImageAddressMode::Repeat);
                },
            ),
        })),
    ));
    commands.trigger(BeginGame);
}

fn gravity(mut transforms: Query<(&mut Transform, &mut Velocity, &Gravity)>, time: Res<Time>) {
    for (mut transform, mut velocity, gravity) in &mut transforms {
        velocity.0 -= gravity.0 * time.delta_secs();

        transform.translation.y += velocity.0 * time.delta_secs();
    }
}

fn controls(
    mut commands: Commands,
    player: Single<Entity, With<Player>>,
    buttons: Res<ButtonInput<MouseButton>>,
) {
    if buttons.any_just_pressed([MouseButton::Left, MouseButton::Right]) {
        commands.trigger(BirdJump(*player));
    }
}

fn check_in_bounds(
    mut birds: Query<(Entity, &Transform, &mut Bird), With<Bird>>,
    mut commands: Commands,
    mut gamemode: Res<GamemodeResource>,
) {
    // check each bird
    for (entity, transform, mut bird) in birds.iter_mut() {
        if transform.translation.y < -CANVAS_SIZE.y / 2.0 - PLAYER_SIZE
            || transform.translation.y > CANVAS_SIZE.y / 2.0 + PLAYER_SIZE
        {
            bird.dead = true;

            match gamemode.0 {
                GameMode::Train { .. } => {
                    commands.entity(e).insert(Dead);
                }
                GameMode::Play { .. } => {
                    commands.trigger(EndGame);
                }
                GameMode::Demo { .. } => {
                    commands.trigger(EndGame);
                }
            }
        }
    }
}

fn handle_endgame(
    _: On<EndGame>,
    mut commands: Commands,
    player: Option<Single<Entity, With<Player>>>,
    mut score: ResMut<Score>,
) {
    match player {
        None => { /* not spawned yet, do nothing */ }
        Some(player) => {
            score.0 = 0;
            commands.entity(*player).insert((
                Transform::from_xyz(-CANVAS_SIZE.x / 4.0, 0.0, 1.0),
                Velocity(0.),
            ));
        }
    }
}

fn spawn_birds(
    event: On<BeginGame>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gamemode: Res<GamemodeResource>,
) {
    match gamemode.0 {
        GameMode::Demo { .. } => {}
        GameMode::Play {
            birdnumber,
            opacity,
        } => {
            commands.spawn((Player, Bird::new(&*asset_server, 0, opacity)));
        }
        GameMode::Train {
            birdnumber,
            opacity,
        } => {
            for n in 0..birdnumber {
                commands.spawn(Bird::new(&*asset_server, n, opacity));
            }
        }
    }
}

fn despawn_deads(
    mut commands: Commands,
    mut birdinv: ResMut<BirdInventory>,
    birds: Query<(Entity, &Bird)>,
) {
    for (entity, bird) in birds {
        if bird.dead {
            commands.entity(entity).despawn();
            //pushed in think there's probably a way to implement a trait here
        };
    }
}

fn check_collisions(
    mut commands: Commands,
    mut birds: Query<(Entity, &mut Bird, Has<Player>), With<Bird>>,
    pipe_segments: Query<(&Sprite, Entity), Or<(With<PipeTop>, With<PipeBottom>)>>,
    mut pipe_gaps: Query<(&Sprite, Entity, &mut PointsGate), With<PointsGate>>,
    mut gizmos: Gizmos,
    transform_helper: TransformHelper,
    gamemode: Res<GamemodeResource>,
) -> Result<()> {
    for (mut e, mut bird, player) in birds.iter_mut() {
        let bird_transform = transform_helper.compute_global_transform(e)?;
        let bird_collider =
            BoundingCircle::new(bird_transform.translation().xy(), PLAYER_SIZE / 2.);

        gizmos.circle_2d(bird_transform.translation().xy(), PLAYER_SIZE / 2., RED_400);

        for (sprite, entity) in &pipe_segments {
            let pipe_transform = transform_helper.compute_global_transform(entity)?;
            let pipe_collider = Aabb2d::new(
                pipe_transform.translation().xy(),
                sprite.custom_size.unwrap() / 2.,
            );

            gizmos.rect_2d(
                pipe_transform.translation().xy(),
                sprite.custom_size.unwrap(),
                RED_400,
            );
            if bird_collider.intersects(&pipe_collider) {
                match gamemode.0 {
                    GameMode::Train { .. } => {
                        commands.entity(e).insert(PipeDeath);
                    }
                    _ => {
                        commands.trigger(EndGame);
                    }
                }
            }
        }

        for (sprite, entity, mut pointgate) in pipe_gaps.iter_mut() {
            let gap_transform = transform_helper.compute_global_transform(entity)?;
            let gap_collider = Aabb2d::new(
                gap_transform.translation().xy(),
                sprite.custom_size.unwrap() / 2.,
            );

            gizmos.rect_2d(
                gap_transform.translation().xy(),
                sprite.custom_size.unwrap().xy(),
                RED_400,
            );

            if bird_collider.intersects(&gap_collider) & !pointgate.has_scored.contains(&bird.uid) {
                commands.trigger(ScorePoint {
                    bird: e,
                    gate: entity,
                });
            }
        }
    }
    Ok(())
}

fn on_score(
    e: On<ScorePoint>,
    mut birds: Query<&mut Bird>,
    mut score: ResMut<Score>,
    mut text: Query<&mut Text, With<ScoreText>>,
) {
    if let Ok(mut bird) = birds.get_mut(e.bird) {
        bird.score += 1;
        score.0 = bird.score;
    }
    for mut t in &mut text {
        t.0 = score.0.to_string();
    }
}

pub fn best_bird(bird_query: Query<&Bird>, mut query: Query<&mut Text, With<ScoreText>>) {
    let best_score = bird_query.iter().map(|bird| bird.score).max();
    for mut span in &mut query {
        if let Some(highest) = best_score {
            span.0 = highest.to_string();
        }
    }
}

fn enforce_bird_direction(mut birds: Query<(&mut Transform, &Velocity), With<Bird>>) {
    for (mut transform, velocity) in &mut birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0);
        transform.rotation = Quat::from_rotation_z(calculated_velocity.to_angle());
    }
}

fn on_bird_jump(event: On<BirdJump>, mut velocities: Query<&mut Velocity, With<Bird>>) {
    if let Ok(mut velocity) = velocities.get_mut(event.0) {
        velocity.0 = 300.;
    }
}

fn endgame<E: EndsGame>(
    _: On<E>,
    mut commands: Commands,
    mut moves: Query<Entity, With<Transform>>,
) {
    for e in moves {
        commands.entity(e).despawn();
    }
    commands.trigger(BeginGame);
}

//debug helper
fn set_time_scale(mut time: ResMut<Time<Virtual>>) {
    time.set_relative_speed(0.5); // 2x faster
}
fn toggle_pause(mut time: ResMut<Time<Virtual>>, keys: Res<ButtonInput<KeyCode>>) {
    if keys.just_pressed(KeyCode::Space) {
        if time.is_paused() {
            time.unpause();
        } else {
            time.pause();
        }
    }
}
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct BackgroundMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub color_texture: Handle<Image>,
}

impl Material2d for BackgroundMaterial {
    fn fragment_shader() -> ShaderRef {
        "background.wgsl".into()
    }
}
