//! Original pistol geometry and the licensed drone target, driven only by combat state.

use super::{
    combat::{health::RotorHealth, pistol::Pistol},
    drone_model, Game,
};
use bevy::prelude::*;

/// Mutually exclusive roles for the three weapon display roots.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(super) enum Visual {
    /// Root of the damageable drone model.
    Drone,
    /// The uncollected pistol on the ground.
    Pickup,
    /// The collected pistol in the robot's hand.
    Held,
}

/// Load the existing licensed drone and create matching ground and held pistol meshes.
pub(super) fn setup(
    mut commands: Commands<'_, '_>,
    assets: Res<'_, AssetServer>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
) {
    let model = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/drone.glb"));
    commands
        .spawn((Visual::Drone, Transform::default(), Visibility::default()))
        .with_children(|root| {
            root.spawn((SceneRoot(model), drone_model::model_alignment()));
        });
    let metal = materials.add(Color::srgb(0.13, 0.17, 0.19));
    let grip = materials.add(Color::srgb(0.89, 0.57, 0.13));
    let barrel = meshes.add(Cuboid::new(0.07, 0.09, 0.25));
    let handle = meshes.add(Cuboid::new(0.065, 0.15, 0.08));
    for held in [false, true] {
        let mut entity = commands.spawn((Transform::default(), Visibility::default()));
        if held {
            entity.insert(Visual::Held);
        } else {
            entity.insert(Visual::Pickup);
        }
        entity.with_children(|root| {
            root.spawn((
                Mesh3d(barrel.clone()),
                MeshMaterial3d(metal.clone()),
                Transform::from_xyz(0.0, 0.0, -0.08),
            ));
            root.spawn((
                Mesh3d(handle.clone()),
                MeshMaterial3d(grip.clone()),
                Transform::from_xyz(0.0, -0.09, 0.0),
            ));
        });
    }
}

/// Project ownership and authoritative target health into model visibility and poses.
pub(super) fn project(
    game: Res<'_, Game>,
    mut roots: Query<'_, '_, (&Visual, &mut Transform, &mut Visibility), Without<Name>>,
) {
    let target = game.combat.target();
    for (kind, mut transform, mut visible) in &mut roots {
        let shown = match kind {
            Visual::Drone => {
                transform.translation = target.position();
                transform.rotation = target.rotation();
                target.health().is_alive()
            }
            Visual::Pickup => {
                transform.translation = Pistol::LOCATION - Vec3::Y * 0.65;
                !game.combat.is_armed()
            }
            Visual::Held => {
                *transform = held_pose(&game);
                game.combat.is_armed()
            }
        };
        *visible = if shown {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
    }
}

/// The same source-mesh pivot mapping drives both drone viewers.
pub(super) fn rotors(
    game: Res<'_, Game>,
    time: Res<'_, Time>,
    mut rotors: Query<'_, '_, (&Name, &mut Transform, &mut Visibility)>,
) {
    for (name, mut transform, mut visible) in &mut rotors {
        if let Some((pivot, motor, sign)) = drone_model::rotor(name.as_str()) {
            *visible = if game.combat.target().health().rotor(motor) == RotorHealth::Destroyed {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            };
            *transform = drone_model::spin_about(pivot, time.elapsed_secs() * 100.0 * sign);
        }
    }
}

/// Draw a bounded tracer and impact pulse; mark a rotor after its first hit.
pub(super) fn traces(game: Res<'_, Game>, mut gizmos: Gizmos<'_, '_>) {
    if let Some(trace) = game.combat.trace() {
        gizmos.line(trace.from, trace.to, Color::srgb(1.0, 0.78, 0.3));
        gizmos.sphere(Isometry3d::from_translation(trace.to), 0.05, Color::WHITE);
    }
    let target = game.combat.target();
    if target.health().is_alive() {
        for motor in bevy_gym::robots::DroneMotor::ALL {
            if target.health().rotor(motor) == RotorHealth::Damaged {
                gizmos.sphere(
                    Isometry3d::from_translation(target.rotor_centre(motor)),
                    0.12,
                    Color::srgb(1.0, 0.6, 0.1),
                );
            }
        }
    }
}

/// Derive the pistol pose from the same shoulder hinge used by the robot animation.
fn held_pose(game: &Game) -> Transform {
    let facing = Quat::from_rotation_y(game.facing());
    let pitch = game.pitch();
    let shoulder = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2 + pitch);
    let hand = Vec3::new(0.31, 0.46, 0.0) + shoulder * Vec3::new(0.0, -0.50, 0.0);
    let barrel = Quat::from_rotation_x(pitch);
    Transform::from_translation(game.arena.position() + facing * (hand + barrel * Vec3::Y * 0.09))
        .with_rotation(facing * barrel)
}
