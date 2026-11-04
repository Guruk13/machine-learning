#![allow(warnings)]
use bevy::{
    asset::AssetMetaCheck,
    camera::ScalingMode,
    image::ImageAddressMode,
    prelude::*,
    prelude::{Aabb2d, BoundingCircle, IntersectsVolume},
    render::{render_resource::AsBindGroup, renderer::RenderGraphSystems::Begin},
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
use crate::{BeginGame, EndGame, ResetGamemode, SwitchGamemode};
use bevy::color::Color;
use bevy_egui::{EguiGlobalSettings, PrimaryEguiContext};
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::Device;
use flappy_bird::BrainPlugin;
use std::rc::Rc;
use std::time::Duration;
use std::{cell::RefCell, collections::HashSet};

use bevy::image::{ImageSampler, ImageSamplerDescriptor};

fn main() {
    // sync main — just kicks off the async work and doesn't wait for it
    wasm_bindgen_futures::spawn_local(async_main());
}
async fn async_main() {
    let device = Device::wgpu_options()
        .init_async()
        .await
        .expect("Unable to initialize WebGPU for inference")
        .autodiff();
    set_global_device(device);

    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            meta_check: AssetMetaCheck::Never,
            ..default()
        }))
        .add_plugins((
            PipePlugin,
            BrainPlugin,
            Material2dPlugin::<BackgroundMaterial>::default(),
        ))
        .insert_resource(GamemodeRessource(GameMode::train()))
        .init_resource::<Score>()
        .configure_sets(
            FixedUpdate,
            GameSets::Play.run_if(|m: Res<GamemodeRessource>| matches!(m.0, GameMode::Play { .. })),
        )
        .configure_sets(
            Update,
            GameSets::Play.run_if(|m: Res<GamemodeRessource>| matches!(m.0, GameMode::Play { .. })),
        )
        .configure_sets(
            FixedPostUpdate,
            GameSets::Play.run_if(|m: Res<GamemodeRessource>| matches!(m.0, GameMode::Play { .. })),
        )
        .configure_sets(
            FixedLast,
            GameSets::Play.run_if(|m: Res<GamemodeRessource>| matches!(m.0, GameMode::Play { .. })),
        )
        .add_systems(Startup, (startup, set_time_scale))
        .add_message::<BeginGame>()
        .add_message::<EndGame>()
        .add_message::<SwitchGamemode>()
        .add_message::<ResetGamemode>()
        .add_systems(
            FixedFirst,
            (
                switch_gamemode,
                end_listener::<SwitchGamemode>,
                begin_listener::<BeginGame>,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                enforce_bird_direction,
                toggle_pause,
                controls.in_set(GameSets::Play),
            ),
        )
        .add_systems(
            FixedUpdate,
            (gravity, check_in_bounds, check_collisions).in_set(GameSets::Game),
        )
        .add_systems(
            FixedLast,
            ((
                despawn_deads,
                (no_bird).in_set(GameSets::Play),
                end_listener::<EndGame>,
            )
                .chain(),),
        )
        // this should be a message , but i need to finish this project
        .add_observer(on_bird_jump)
        .insert_resource(EguiGlobalSettings::default())
        .run();
}

fn startup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut config_store: ResMut<GizmoConfigStore>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<BackgroundMaterial>>,
    mut begin: MessageWriter<BeginGame>,
    mut egui_settings: ResMut<EguiGlobalSettings>,
) {
    egui_settings.auto_create_primary_context = false; // before spawning the camera
    config_store
        .config_mut::<DefaultGizmoConfigGroup>()
        .0
        .enabled = false;
    commands.spawn((
        Camera2d,
        PrimaryEguiContext, // <- Egui context lives on this camera
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
            color_texture: {
                asset_server
                    .load_builder()
                    .with_settings(|settings: &mut ImageLoaderSettings| {
                        settings
                            .sampler
                            .get_or_init_descriptor()
                            .set_address_mode(ImageAddressMode::Repeat);
                    })
                    .load("background_color_grass.png")
            },
        })),
    ));
    begin.write(BeginGame);
}

const PLAY_TIMEOUT_SECS: f32 = 5.0;
fn switch_gamemode(
    mouse: Res<ButtonInput<MouseButton>>,
    //za warudo
    time: Res<Time<Virtual>>,
    mut gamemode: ResMut<GamemodeRessource>,
    mut last_click: Local<Option<f32>>,
    mut switch: MessageWriter<ResetGamemode>,
    mut begin: MessageWriter<BeginGame>,
) {
    let now = time.elapsed_secs();
    if mouse.just_pressed(MouseButton::Left) {
        *last_click = Some(now);
        // Any mode other than Play -> Play on click
        if !matches!(gamemode.0, GameMode::Play { .. }) {
            gamemode.0 = GameMode::play();
            switch.write(ResetGamemode);
        }
    }
    // Play -> Train if the last click was 5+ seconds ago
    if matches!(gamemode.0, GameMode::Play { .. }) {
        if let Some(t) = *last_click {
            if now - t >= PLAY_TIMEOUT_SECS {
                begin.write(BeginGame);
                switch.write(ResetGamemode);
                gamemode.0 = GameMode::train();
                *last_click = None;
            }
        }
    }
}

fn gravity(mut transforms: Query<(&mut Transform, &mut Velocity, &Gravity)>, time: Res<Time>) {
    for (mut transform, mut velocity, gravity) in &mut transforms {
        velocity.0 -= gravity.0 * time.delta_secs();

        transform.translation.y += velocity.0 * time.delta_secs();
    }
}

fn controls(
    mut commands: Commands,
    player: Query<Entity, With<Player>>,
    buttons: Res<ButtonInput<MouseButton>>,
) {
    if !buttons.any_just_pressed([MouseButton::Left, MouseButton::Right]) {
        return;
    }
    match player.single() {
        Ok(e) => commands.trigger(BirdJump(e)),
        Err(err) => {}
    }
}

fn check_in_bounds(
    mut birds: Query<(Entity, &Transform), With<Bird>>,
    mut commands: Commands,
    mut gamemode: Res<GamemodeRessource>,
    mut end: MessageWriter<EndGame>,
) {
    for (entity, transform) in birds.iter_mut() {
        if transform.translation.y < -CANVAS_SIZE.y / 2.0 - PLAYER_SIZE
            || transform.translation.y > CANVAS_SIZE.y / 2.0 + PLAYER_SIZE
        {
            match gamemode.0 {
                GameMode::Train { .. } => {
                    commands.entity(entity).insert(Dead);
                }
                GameMode::Play { .. } => {
                    end.write(EndGame);
                }
                GameMode::Demo { .. } => {
                    end.write(EndGame);
                }
            }
        }
    }
}
pub fn no_bird(
    alive: Query<&Bird>,
    mut commands: Commands,
    mut end: MessageWriter<EndGame>,
    mut begin: MessageWriter<BeginGame>,
) {
    if !alive.is_empty() {
        return;
    }
    end.write(EndGame);
    begin.write(BeginGame);
}

fn despawn_deads(
    mut commands: Commands,
    birds: Query<(Entity), Or<(With<Dead>, With<PipeDeath>)>>,
) {
    for e in birds {
        commands.entity(e).despawn();
    }
}

fn check_collisions(
    mut commands: Commands,
    mut birds: Query<(Entity, &mut Bird, &mut IndividualScore, Has<Player>), With<Bird>>,
    pipe_segments: Query<(&Sprite, Entity), Or<(With<PipeTop>, With<PipeBottom>)>>,
    mut pipe_gaps: Query<(&Sprite, Entity, &mut PointsGate), With<PointsGate>>,
    mut gizmos: Gizmos,
    transform_helper: TransformHelper,
    gamemode: Res<GamemodeRessource>,
    mut end: MessageWriter<EndGame>,
    mut text: Query<&mut Text, With<ScoreText>>,
) -> Result<()> {
    for (mut e, mut bird, mut birdscore, player) in birds.iter_mut() {
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
                        end.write(EndGame);
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
                pointgate.has_scored.push(bird.uid);
                birdscore.0 += 1;
                for mut t in &mut text {
                    t.0 = birdscore.0.to_string();
                }
            }
        }
    }
    Ok(())
}

fn enforce_bird_direction(mut birds: Query<(&mut Transform, &Velocity), With<Bird>>) {
    for (mut transform, velocity) in &mut birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, velocity.0);
        transform.rotation = Quat::from_rotation_z(calculated_velocity.to_angle());
    }
}

fn on_bird_jump(event: On<BirdJump>, mut velocities: Query<&mut Velocity, With<Bird>>) {
    if let Ok(mut velocity) = velocities.get_mut(event.0) {
        velocity.0 = 460.;
    }
}

fn end_listener<E: Message>(
    mut end: MessageReader<E>,
    mut text: Query<&mut Text, With<ScoreText>>,
    movers: Query<Entity, Or<(With<Pipe>, With<Bird>)>>,
    mut commands: Commands,
) {
    if end.read().count() > 0 {
        for e in &movers {
            //this should be a despawn
            commands.entity(e).despawn();
        }
        for mut t in &mut text {
            t.0 = String::from("0");
        }
    }
}

fn begin_listener<E: Message>(
    mut begin: MessageReader<E>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    gamemode: Res<GamemodeRessource>,
) {
    if begin.read().count() > 0 {
        match gamemode.0 {
            GameMode::Demo { .. } => {}
            GameMode::Play { opacity, .. } => {
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
}

//debug helper
//maybe accelerate when training ?
fn set_time_scale(mut time: ResMut<Time<Virtual>>) {
    time.set_relative_speed(1.);
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
#[derive(Debug, Asset, TypePath, AsBindGroup, Clone)]
pub struct BackgroundMaterial {
    #[texture(0)]
    #[sampler(1)]
    pub color_texture: Handle<Image>,
}

impl Material2d for BackgroundMaterial {
    fn fragment_shader() -> ShaderRef {
        "background.wesl".into()
    }
}
