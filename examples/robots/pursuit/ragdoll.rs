//! Physical death and reset for the loaded mannequin.

use bevy::prelude::*;

/// Install the compatible physics runtime before spawning the mannequin.
pub(super) fn install(app: &mut App) {
    use bevy_ragdoll::{runtime::sets::RagdollSystems, RagdollPlugin};
    use bevy_ragdoll_rapier3d::{RapierRagdollHooks, RapierRagdollPlugin};
    use bevy_rapier3d::plugin::{RapierPhysicsPlugin, TimestepMode};
    // The adapter's forced-sleep timer survives reset and can sleep animated bodies.
    // Keep Rapier's automatic sleep; disable only that optional timer for this viewer.
    app.insert_resource(bevy_ragdoll::runtime::settings::RagdollPhysicsSettings {
        force_sleep_after: 0.0,
        ..default()
    })
    .insert_resource(TimestepMode::Fixed {
        dt: 0.02,
        substeps: 2,
    })
    .add_plugins((
        RagdollPlugin::default(),
        RapierPhysicsPlugin::<RapierRagdollHooks<'_, '_>>::default().in_fixed_schedule(),
        RapierRagdollPlugin,
    ))
    .add_systems(Startup, cover)
    .add_systems(Update, transition.before(super::robot::animate))
    .add_systems(
        PostUpdate,
        super::weapons::project
            .after(RagdollSystems::Writeback)
            .before(TransformSystems::Propagate),
    )
    .configure_sets(
        PostUpdate,
        RagdollSystems::CaptureTargets.after(super::robot::grip_pose),
    );
}

/// Reuse every authoritative solid box, including the rotated pipe sections.
fn cover(mut commands: Commands<'_, '_>, game: Res<'_, super::Game>) {
    use bevy_rapier3d::prelude::{Collider, RigidBody};
    for block in game.arena.blocks() {
        commands.spawn((
            RigidBody::Fixed,
            Collider::cuboid(block.half.x, block.half.y, block.half.z),
            Transform::from_translation(block.centre).with_rotation(block.rotation),
        ));
    }
}

/// Health selects animation following or free physical motion; reset restores following.
fn transition(
    game: Res<'_, super::Game>,
    mut modes: Query<
        '_,
        '_,
        (
            &mut bevy_ragdoll::runtime::components::RagdollMode,
            &mut bevy_ragdoll::runtime::components::RagdollBlend,
        ),
    >,
) {
    use bevy_ragdoll::runtime::components::RagdollMode;
    let wanted = if game.robot_health.is_alive() {
        RagdollMode::Kinematic
    } else {
        RagdollMode::Dynamic
    };
    for (mut mode, mut blend) in &mut modes {
        if *mode != wanted {
            *mode = wanted;
            // Living animation and grip constraints must not inherit interpolated physics lag.
            blend.set(if wanted == RagdollMode::Dynamic {
                1.0
            } else {
                0.0
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use bevy_ragdoll::runtime::components::{RagdollBodies, RagdollMode};

    /// Mean body height measures a physical fall independently of animation tracks.
    fn height(world: &mut World) -> f32 {
        let (sum, count) = world
            .query::<&bevy_ragdoll::runtime::body::BodyPhysicsPose>()
            .iter(world)
            .fold((0.0, 0.0), |(sum, count), pose| {
                (sum + pose.current.translation.y, count + 1.0)
            });
        assert!(count > 0.0);
        sum / count
    }

    #[test]
    fn loaded_mannequin_becomes_dynamic_on_death_and_resets_to_animation() {
        let mut app = crate::robot::tests::model_app_with(|app| {
            app.add_plugins(super::install)
                .add_systems(Update, crate::scene::project)
                .add_systems(
                    PostUpdate,
                    (crate::robot::pose, crate::robot::grip_pose)
                        .chain()
                        .after(bevy::app::AnimationSystems)
                        .before(TransformSystems::Propagate),
                );
        });
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
        app.world_mut()
            .resource_mut::<crate::Game>()
            .act(crate::firing::Action::PickUp);
        crate::robot::tests::until(&mut app, |world| {
            world
                .query::<&RagdollBodies>()
                .iter(world)
                .any(|bodies| !bodies.is_empty())
        });
        let world = app.world_mut();
        let (root, bodies, mode) = world
            .query::<(Entity, &RagdollBodies, &RagdollMode)>()
            .single(world)
            .expect("One mannequin");
        assert_eq!(*mode, RagdollMode::Kinematic);
        let body_count = bodies.len();
        assert!(body_count >= 10);
        let standing_height = height(app.world_mut());
        crate::robot::tests::until(&mut app, |world| {
            !world.resource::<crate::Game>().robot_health.is_alive()
        });
        app.update();
        assert_eq!(
            *app.world().get::<RagdollMode>(root).expect("Mode"),
            RagdollMode::Dynamic
        );
        let world = app.world_mut();
        assert!(world
            .query::<&AnimationPlayer>()
            .iter(world)
            .all(|player| player.playing_animations().next().is_none()));
        // Wait beyond the adapter's six-second forced-sleep timer before resetting.
        for _ in 0..600 {
            app.update();
        }
        let fallen_height = height(app.world_mut());
        assert!(
            fallen_height < standing_height - 0.2,
            "The body must fall under gravity"
        );
        assert!(fallen_height > 0.0, "The floor must stop the body");
        app.world_mut().resource_mut::<crate::Game>().reset();
        app.world_mut()
            .resource_mut::<crate::Game>()
            .act(crate::firing::Action::PickUp);
        app.update();
        assert_eq!(
            *app.world().get::<RagdollMode>(root).expect("Mode"),
            RagdollMode::Kinematic
        );
        // Continue the restored animation long enough to exercise subsequent sleep updates.
        for _ in 0..100 {
            app.update();
        }
        assert!(height(app.world_mut()) > fallen_height + 0.2);
        let world = app.world_mut();
        assert_eq!(
            world
                .query::<&bevy_ragdoll::runtime::components::RagdollBodyOf>()
                .iter(world)
                .count(),
            body_count
        );
    }
}
