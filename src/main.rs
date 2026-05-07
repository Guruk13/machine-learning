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
use flappy_bird::*;

use crate::player::*;
use player::GameSets;

fn main() -> AppExit {
    App::new()
        .init_resource::<Score>()
        .insert_resource(BirdInventory(vec![]))
        .configure_sets(FixedUpdate, (GameSets::Game, GameSets::AI).chain())
        .add_plugins(DefaultPlugins)
        .add_plugins((
            PipePlugin,
            Material2dPlugin::<BackgroundMaterial>::default(),
        ))
        .add_systems(Startup, (startup, spawn_birds).in_set(GameSets::Game))
        .add_systems(
            FixedUpdate,
            (gravity, check_in_bounds, check_collisions).chain(),
        )
        .add_systems(
            Update,
            (
                controls.run_if(any_with_component::<Player>),
                score_update.run_if(resource_changed::<Score>),
                enforce_bird_direction,
            ),
        )
        .add_systems(FixedPostUpdate, (bird_respawn).in_set(GameSets::AI))
        .add_observer(respawn_on_endgame)
        .add_observer(on_bird_jump)
        .add_observer(|_trigger: On<ScorePoint>, mut score: ResMut<Score>| {
            score.0 += 1;
        })
        .add_plugins(BrainPlugin)
        .run()
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
        TextLayout::new_with_justify(Justify::Center),
        TextFont {
            font_size: 33.0,
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
    birds: Query<(Entity, &Transform, Has<Player>), With<Bird>>,
    mut commands: Commands,
) {
    let wheelie_bounds = false;
    // check each bird
    for (entity, transform, is_player) in birds.iter() {
        if !wheelie_bounds {
            if transform.translation.y < -CANVAS_SIZE.y / 2.0 - PLAYER_SIZE
                || transform.translation.y > CANVAS_SIZE.y / 2.0 + PLAYER_SIZE
            {
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

fn check_collisions(
    mut commands: Commands,
    birds: Query<(&Sprite, Entity, Has<Player>), With<Bird>>,
    pipe_segments: Query<(&Sprite, Entity), Or<(With<PipeTop>, With<PipeBottom>)>>,
    pipe_gaps: Query<(&Sprite, Entity), With<PointsGate>>,
    mut gizmos: Gizmos,
    transform_helper: TransformHelper,
) -> Result<()> {
    for bird in birds.iter() {
        let bird_transform = transform_helper.compute_global_transform(bird.1)?;
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
                    commands.trigger(BirdDeath { bird: bird.1 });
                }
            }
        }

        for (sprite, entity) in &pipe_gaps {
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

            if bird_collider.intersects(&gap_collider) {
                commands.trigger(ScorePoint { bird: bird.1 });
                //commands.entity(entity).despawn();
            }
        }
    }
    Ok(())
}

fn score_update(mut query: Query<&mut Text, With<ScoreText>>, score: Res<Score>) {
    for mut span in &mut query {
        span.0 = score.0.to_string();
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

fn enforce_bird_direction(players: Query<(&mut Transform, &Velocity), With<Bird>>) {
    for mut player in players {
        let calculated_velocity = Vec2::new(PIPE_SPEED, player.1.0);
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
) -> Result<()> {
    let translation = Vec2::new(-CANVAS_SIZE.x / 4.0, 0.0);
    //spawn point with player size +10%
    let bird_collider = BoundingCircle::new(translation, PLAYER_SIZE / 2. * 1.1);
    if !&pipe_segments.is_empty() {
        for (sprite, entity) in &pipe_segments {
            let pipe_transform = transform_helper.compute_global_transform(entity)?;
            let pipe_collider = Aabb2d::new(
                pipe_transform.translation().xy(),
                sprite.custom_size.unwrap() / 2.,
            );

            if !bird_collider.intersects(&pipe_collider) {
                warn! { "does not intersect "}
                if !birdinv.0.is_empty() {
                    birdinv.0.iter().for_each(|value| {
                        commands.spawn(Bird::new(&*asset_server, false, *value));
                    });
                    birdinv.0.clear();
                }
            }
        }
    } else {
        if !birdinv.0.is_empty() {
            birdinv.0.iter().for_each(|value| {
                commands.spawn(Bird::new(&*asset_server, false, *value));
            });
            birdinv.0.clear();
        }
    }
    Ok(())
}

fn on_bird_jump(event: On<BirdJump>, mut velocities: Query<&mut Velocity, With<Bird>>) {
    if let Ok(mut velocity) = velocities.get_mut(event.0) {
        velocity.0 = 400.; // this is the ONE place that knows how jumping works
    }
}
