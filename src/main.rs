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

use crate::ml::multiagent::AgentManager;
use crate::player::Pipe;
use crate::player::*;
use crate::staticdevice::set_global_device;
use burn::tensor::Device;
use flappy_bird::AMRessource;
use flappy_bird::BrainPlugin;
use std::cell::RefCell;
use std::rc::Rc;

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
    set_global_device(device.clone());

    App::new()
        .init_resource::<Score>()
        .insert_non_send_resource(AMRessource {
            agent_manager: AgentManager::new(),

            replacement: Rc::new(RefCell::new(ReplacementState::Idle)),
        })
        .insert_resource(BirdInventory(vec![]))
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
        .add_plugins(DefaultPlugins)
        .add_plugins((
            PipePlugin,
            Material2dPlugin::<BackgroundMaterial>::default(),
        ))
        .add_systems(Startup, (startup, spawn_birds))
        .add_systems(
            FixedUpdate,
            (gravity, check_in_bounds, check_collisions).chain(),
        )
        .add_systems(
            Update,
            (
                //score_update.run_if(resource_changed::<Score>),
                best_bird,
                enforce_bird_direction,
            ),
        )
        .add_observer(respawn_on_endgame)
        .add_observer(on_bird_jump)
        .add_observer(|_trigger: On<ScorePoint>, mut score: ResMut<Score>| {
            score.0 += 1;
        })
        // Input
        .add_systems(
            Update,
            controls
                .run_if(any_with_component::<Player>)
                .in_set(GameSets::Input),
        )
        // Game
        .add_systems(
            FixedUpdate,
            (gravity, check_in_bounds, check_collisions)
                .chain()
                .in_set(GameSets::Game),
        )
        // usually you'd rather move an entity instead of despawn /respawn.
        // here pipes can obstruct the spawn point. spawn a bird only in somewhat fair conditions
        .add_systems(
            FixedPostUpdate,
            (ApplyDeferred, despawn_deads, bird_respawn)
                .chain()
                .in_set(GameSets::Cleanup),
        )
        .add_systems(Startup, set_time_scale)
        .add_systems(Update, toggle_pause)
        // AI
        .add_plugins(BrainPlugin)
        .run();
}

#[derive(Event)]
struct EndGame;

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
}

fn gravity(mut transforms: Query<(&mut Transform, &mut Velocity, &Gravity)>, time: Res<Time>) {
    for (mut transform, mut velocity, gravity) in &mut transforms {
        velocity.0 -= gravity.0 * time.delta_secs();

        transform.translation.y += velocity.0 * time.delta_secs();
    }
}

fn controls(
    mut commands: Commands,
    player: Option<Single<Entity, With<Player>>>,
    buttons: Res<ButtonInput<MouseButton>>,
) {
    match player {
        None => { /* not spawned yet, do nothing */ }
        Some(player) => {
            if buttons.any_just_pressed([MouseButton::Left, MouseButton::Right]) {
                //velocity.0 = 400.;
                commands.trigger(BirdJump(*player));
            }
        }
    }
}

fn check_in_bounds(
    mut birds: Query<(Entity, &Transform, &mut Bird, Has<Player>), With<Bird>>,
    mut commands: Commands,
) {
    let wheelie_bounds = false;
    // check each bird
    for (entity, transform, mut bird, is_player) in birds.iter_mut() {
        if !wheelie_bounds {
            if transform.translation.y < -CANVAS_SIZE.y / 2.0 - PLAYER_SIZE
                || transform.translation.y > CANVAS_SIZE.y / 2.0 + PLAYER_SIZE
            {
                bird.dead = true;
                if is_player {
                    commands.trigger(EndGame);
                } else {
                    commands.trigger(BirdDeath { bird: entity });
                }
            }
        } else {
            if transform.translation.y < -CANVAS_SIZE.y / 2.0 - PLAYER_SIZE {
                commands.trigger(BirdJump(entity));
            }
        }
    }
}

fn respawn_on_endgame(
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

fn spawn_birds(mut commands: Commands, asset_server: Res<AssetServer>) {
    for n in 0..5 {
        commands.spawn(Bird::new(&*asset_server, false, n));
    }
}

fn _spawn_player(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((Player, Bird::new(&*asset_server, true, 0)));
}

fn despawn_deads(
    mut commands: Commands,
    mut birdinv: ResMut<BirdInventory>,
    birds: Query<(Entity, &Bird)>,
) {
    //warn! {"there'll be another time"};
    for (entity, bird) in birds {
        if bird.dead {
            //warn! {"there'll be another time {:?}", bird.uid}
            commands.entity(entity).despawn();
            //pushed in think there's probably a way to implement a trait here
            //birdinv.0.push(bird.uid)
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
) -> Result<()> {
    for mut bird in birds.iter_mut() {
        let bird_transform = transform_helper.compute_global_transform(bird.0)?;
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
                //is_player
                if bird.2 {
                    //commands.trigger(EndGame);
                } else {
                    bird.1.dead = true;
                    bird.1.pipe_death = true;

                    commands.trigger(BirdDeath { bird: bird.0 });
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

            if bird_collider.intersects(&gap_collider) & !pointgate.has_scored.contains(&bird.1.uid)
            {
                //commands.trigger(ScorePoint { bird: bird.0 });
                bird.1.score = bird.1.score + 1;
                pointgate.has_scored.push(bird.1.uid)
            }
        }
    }
    Ok(())
}

fn _score_update(mut query: Query<&mut Text, With<ScoreText>>, score: Res<Score>) {
    for mut span in &mut query {
        span.0 = score.0.to_string();
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

fn enforce_bird_direction(birds: Query<(&mut Transform, &Velocity), With<Bird>>) {
    for mut player in birds {
        let calculated_velocity = Vec2::new(PIPE_SPEED, player.1 .0);
        player.0.rotation = Quat::from_rotation_z(calculated_velocity.to_angle());
    }
}

//on postfixed update to give time for proper thinking and avoid collision
fn bird_respawn(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut birdinv: ResMut<BirdInventory>,
    pipe_segments: Query<(&Sprite, Entity), Or<(With<PipeTop>, With<PipeBottom>)>>,
    transform_helper: TransformHelper,
    pipe_gaps: Query<(&Sprite, Entity), With<PointsGate>>,
) -> Result<()> {
    let translation = Vec2::new(-CANVAS_SIZE.x / 4.0, 0.0);
    let bird_collider = BoundingCircle::new(translation, 75.0); // spawn is clear of 50% a Bird's size

    let mut has_intersected = pipe_segments.iter().any(|(sprite, entity)| {
        let pipe_transform = transform_helper.compute_global_transform(entity).unwrap();
        let pipe_collider = Aabb2d::new(
            pipe_transform.translation().xy(),
            sprite.custom_size.unwrap() / 2.,
        );
        bird_collider.intersects(&pipe_collider)
    });

    for (sprite, entity) in pipe_gaps {
        let gap_transform = transform_helper.compute_global_transform(entity)?;
        let gap_collider = Aabb2d::new(
            gap_transform.translation().xy(),
            sprite.custom_size.unwrap() / 2.,
        );

        if bird_collider.intersects(&gap_collider) {
            has_intersected = true;
        }
    }

    if !has_intersected {
        for value in birdinv.0.drain(..) {
            commands.spawn(Bird::new(&asset_server, false, value));
            //warn!("spawn {:?}", value)
        }

        //warn!("{:?}", birdinv.0.len());
        // drain already clears, no need for .clear()
    }
    Ok(())
}

fn on_bird_jump(event: On<BirdJump>, mut velocities: Query<&mut Velocity, With<Bird>>) {
    if let Ok(mut velocity) = velocities.get_mut(event.0) {
        velocity.0 = 300.; // this is the ONE place that knows how jumping works
    }
}

//debug helper
fn set_time_scale(mut time: ResMut<Time<Virtual>>) {
    time.set_relative_speed(0.1); // 2x faster
                                  // time.set_relative_speed(0.5); // half speed
                                  // time.set_relative_speed(0.0); // pause
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
