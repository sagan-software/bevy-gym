//! Licensed skinned mannequin; animation projects movement and combat state.

#[path = "robot/animation.rs"]
mod animation;
#[path = "robot/locomotion.rs"]
mod locomotion;
pub(super) use locomotion::pose;
#[path = "robot/grip.rs"]
mod grip;
pub(super) use grip::pose as grip_pose;

use super::{scene::Character, Game};
use bevy::{animation::AnimationTargetId, prelude::*};

/// Right-hand bone used to place the held pistol after the skeleton loads.
#[derive(Component)]
pub(super) struct Hand;

/// Source model and named animation clips loaded together.
#[derive(Resource)]
pub(super) struct Model(Handle<Gltf>);

/// Load the in-place mannequin beneath the authoritative capsule projection.
pub(super) fn spawn(commands: &mut Commands<'_, '_>, assets: &AssetServer) {
    let path = "robots/survival/mannequin.glb";
    commands.insert_resource(Model(assets.load(path)));
    commands.spawn((
        LoadingNotice,
        Text::new("Loading models…"),
        TextFont {
            font: assets.load("fonts/MonaSans-VariableFont.ttf"),
            font_size: 18.0,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            left: px(16),
            top: px(104),
            ..default()
        },
        BackgroundColor(Color::srgb(0.09, 0.11, 0.12)),
    ));
    commands
        .spawn((Character, Transform::default(), Visibility::default()))
        .with_children(|root| {
            root.spawn((
                SceneRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(path))),
                bevy_ragdoll::Ragdoll::default(),
                bevy_ragdoll::runtime::components::RagdollMode::Kinematic,
                bevy_ragdoll::runtime::components::RagdollDrive::new(0.0, 0.0),
                bevy_ragdoll::runtime::components::RagdollBlend::new(0.0),
                Transform::from_xyz(0.0, -0.9, 0.0)
                    .with_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
            ));
        });
}

/// Build the graph only after the model's skeleton and named clips exist.
pub(super) fn load(
    mut commands: Commands<'_, '_>,
    model: Res<'_, Model>,
    models: Res<'_, Assets<Gltf>>,
    mut graphs: ResMut<'_, Assets<AnimationGraph>>,
    players: Query<'_, '_, Entity, (With<AnimationPlayer>, Without<animation::Rig>)>,
    targets: Query<'_, '_, (Entity, &AnimationTargetId, &bevy::animation::AnimatedBy)>,
    hierarchy: Query<'_, '_, (&Name, Option<&ChildOf>)>,
) {
    let Some(model) = models.get(&model.0) else {
        return;
    };
    for player in &players {
        let (mut graph, rig) = animation::Rig::new(model);
        for (entity, target, owner) in &targets {
            if owner.0 == player {
                if hierarchy
                    .get(entity)
                    .is_ok_and(|(name, _)| name.as_str() == "hand_r")
                {
                    commands.entity(entity).insert(Hand);
                }
                if let Ok((name, _)) = hierarchy.get(entity) {
                    if let Some(arm) = grip::Arm::named(name.as_str()) {
                        commands.entity(entity).insert(arm);
                    }
                    let bone = match name.as_str() {
                        "pelvis" => Some(locomotion::Bone::Pelvis),
                        "spine_01" => Some(locomotion::Bone::Spine),
                        _ => None,
                    };
                    if let Some(bone) = bone {
                        commands.entity(entity).insert(bone);
                    }
                }
                graph.add_target_to_mask_group(*target, u32::from(!upper_body(entity, &hierarchy)));
            }
        }
        commands
            .entity(player)
            .insert((AnimationGraphHandle(graphs.add(graph)), rig));
    }
}

/// Classify the whole spine subtree, including fingers, without relying on bone ordering.
fn upper_body(mut entity: Entity, hierarchy: &Query<'_, '_, (&Name, Option<&ChildOf>)>) -> bool {
    while let Ok((name, parent)) = hierarchy.get(entity) {
        if name.as_str() == "spine_01" {
            return true;
        }
        let Some(parent) = parent else {
            break;
        };
        entity = parent.parent();
    }
    false
}

/// Blend measured locomotion with an independent upper-body pistol pose.
pub(super) fn animate(
    game: Res<'_, Game>,
    time: Res<'_, Time>,
    mut graphs: ResMut<'_, Assets<AnimationGraph>>,
    mut players: Query<
        '_,
        '_,
        (
            &mut AnimationPlayer,
            &AnimationGraphHandle,
            &mut animation::Rig,
        ),
    >,
) {
    for (mut player, graph, mut rig) in &mut players {
        if let Some(graph) = graphs.get_mut(&graph.0) {
            rig.advance(&game, time.delta_secs(), &mut player, graph);
        }
    }
}

/// Asset-loading feedback; it disappears once the animated character can be controlled.
#[derive(Component)]
pub(super) struct LoadingNotice;

/// Do not consume survival time or accept combat actions before all visible scenes load.
pub(super) fn ready(
    assets: Res<'_, AssetServer>,
    scenes: Query<'_, '_, &SceneRoot>,
    players: Query<'_, '_, &animation::Rig>,
) -> bool {
    !players.is_empty()
        && scenes
            .iter()
            .all(|scene| assets.is_loaded_with_dependencies(&scene.0))
}

/// Report missing model or texture files without leaving the player invisible and vulnerable.
pub(super) fn loading_notice(
    assets: Res<'_, AssetServer>,
    scenes: Query<'_, '_, &SceneRoot>,
    players: Query<'_, '_, &animation::Rig>,
    mut notices: Query<'_, '_, (&mut Text, &mut Visibility), With<LoadingNotice>>,
) {
    let loaded = !players.is_empty()
        && scenes
            .iter()
            .all(|scene| assets.is_loaded_with_dependencies(&scene.0));
    let failed = scenes.iter().any(|scene| {
        assets
            .get_load_state(scene.0.id())
            .is_some_and(|state| state.is_failed())
            || assets
                .get_recursive_dependency_load_state(scene.0.id())
                .is_some_and(|state| state.is_failed())
    });
    for (mut text, mut visibility) in &mut notices {
        *visibility = if loaded {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        };
        text.0 = if failed {
            "Model loading failed. Reload to retry.".to_owned()
        } else {
            "Loading models…".to_owned()
        };
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use bevy::ecs::system::{RunSystemOnce, SystemState};

    /// Load actual glTF scenes and animate their skeletons without creating a GPU device.
    fn model_app() -> App {
        model_app_with(|_| {})
    }

    /// Install optional scene plugins before the asset-loading test app starts.
    pub(crate) fn model_app_with(configure: impl FnOnce(&mut App)) -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            AssetPlugin {
                file_path: concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets").to_owned(),
                meta_check: bevy::asset::AssetMetaCheck::Never,
                ..default()
            },
            bevy::scene::ScenePlugin,
            TransformPlugin,
            AnimationPlugin,
            bevy::gltf::GltfPlugin::default(),
        ))
        .init_asset::<Mesh>()
        .init_asset::<Image>()
        .init_asset::<StandardMaterial>()
        .register_type::<MeshMaterial3d<StandardMaterial>>()
        .init_asset::<bevy::mesh::skinning::SkinnedMeshInverseBindposes>()
        .init_asset::<Font>()
        .init_resource::<Game>()
        .init_resource::<crate::controls::Input>()
        .insert_resource(Time::<Fixed>::from_seconds(0.02))
        .insert_resource(bevy::image::CompressedImageFormatSupport(
            bevy::image::CompressedImageFormats::NONE,
        ))
        .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_millis(20),
        ))
        .add_systems(
            Startup,
            |mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>| {
                spawn(&mut commands, &assets);
            },
        )
        .add_systems(FixedUpdate, crate::advance.run_if(ready))
        .add_systems(Update, (load, animate, loading_notice).chain());
        configure(&mut app);
        app.finish();
        app.cleanup();
        app
    }

    /// Bound asynchronous file loading while keeping the simulation clock deterministic.
    pub(crate) fn until(app: &mut App, mut done: impl FnMut(&mut World) -> bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            app.update();
            if done(app.world_mut()) {
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "Asset loading deadline"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    fn actual_model_loads_its_hand_and_missing_scene_stops_play() {
        let mut app = model_app();
        assert!(!app
            .world_mut()
            .run_system_once(ready)
            .expect("Ready system"));
        until(&mut app, |world| {
            world.run_system_once(ready).expect("Ready system")
        });
        app.update();
        let world = app.world_mut();
        assert!(world.resource::<Game>().run.elapsed() > std::time::Duration::ZERO);
        assert_eq!(
            world
                .query_filtered::<Entity, With<Hand>>()
                .iter(world)
                .count(),
            1
        );
        assert_eq!(world.query::<&animation::Rig>().iter(world).count(), 1);
        let missing = world
            .resource::<AssetServer>()
            .load(GltfAssetLabel::Scene(0).from_asset("robots/survival/missing-model.glb"));
        for mut scene in world.query::<&mut SceneRoot>().iter_mut(world) {
            scene.0 = missing.clone();
        }
        let elapsed = world.resource::<Game>().run.elapsed();
        until(&mut app, |world| {
            world
                .query_filtered::<&Text, With<LoadingNotice>>()
                .iter(world)
                .any(|text| text.0 == "Model loading failed. Reload to retry.")
        });
        assert!(!app
            .world_mut()
            .run_system_once(ready)
            .expect("Ready system"));
        assert_eq!(app.world().resource::<Game>().run.elapsed(), elapsed);
    }

    #[test]
    fn upper_body_mask_includes_hand_descendants_but_excludes_legs() {
        let mut world = World::new();
        let pelvis = world.spawn(Name::new("pelvis")).id();
        let spine = world.spawn((Name::new("spine_01"), ChildOf(pelvis))).id();
        let hand = world.spawn((Name::new("hand_r"), ChildOf(spine))).id();
        let finger = world.spawn((Name::new("index_01_r"), ChildOf(hand))).id();
        let thigh = world.spawn((Name::new("thigh_r"), ChildOf(pelvis))).id();
        let unnamed = world.spawn_empty().id();
        let mut state = SystemState::<Query<'_, '_, (&Name, Option<&ChildOf>)>>::new(&mut world);
        let hierarchy = state.get(&world);
        for entity in [spine, hand, finger] {
            assert!(upper_body(entity, &hierarchy));
        }
        for entity in [pelvis, thigh, unnamed] {
            assert!(!upper_body(entity, &hierarchy));
        }
    }
}
