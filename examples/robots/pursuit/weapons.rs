//! Licensed pistol and drone models, driven only by combat state.

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

/// Load licensed scenes once and share their assets between ground and held pistols.
pub(super) fn setup(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
    let model = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/drone.glb"));
    commands
        .spawn((Visual::Drone, Transform::default(), Visibility::default()))
        .with_children(|root| {
            root.spawn((SceneRoot(model), drone_model::model_alignment()));
        });
    let pistol =
        assets.load(GltfAssetLabel::Scene(0).from_asset("robots/survival/pistol/Gun_Pistol.gltf"));
    for kind in [Visual::Pickup, Visual::Held] {
        commands.spawn((
            kind,
            SceneRoot(pistol.clone()),
            Transform::default(),
            Visibility::default(),
        ));
    }
}

/// Project ownership and authoritative target health into model visibility and poses.
pub(super) fn project(
    game: Res<'_, Game>,
    hand: Query<'_, '_, Entity, With<super::robot::Hand>>,
    mut poses: ParamSet<
        '_,
        '_,
        (
            TransformHelper<'_, '_>,
            Query<'_, '_, (&Visual, &mut Transform, &mut Visibility), Without<Name>>,
        ),
    >,
) {
    // Animation has written local bones, but global propagation has not run yet.
    let hand = hand
        .single()
        .ok()
        .and_then(|entity| poses.p0().compute_global_transform(entity).ok());
    let target = game.combat.target();
    for (kind, mut transform, mut visible) in &mut poses.p1() {
        let shown = match kind {
            Visual::Drone => {
                transform.translation = target.position();
                transform.rotation = target.rotation();
                target.health().is_alive()
            }
            Visual::Pickup => {
                // The source mesh is 6.84 cm thick. Lay it flat just above the floor.
                *transform = Transform::from_xyz(Pistol::LOCATION.x, 0.035, Pistol::LOCATION.z)
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
                !game.combat.is_armed()
            }
            Visual::Held => {
                *transform = held_pose(&game, hand.as_ref());
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

/// Place the grip at the animated hand; the native barrel points along negative X.
fn held_pose(game: &Game, hand: Option<&GlobalTransform>) -> Transform {
    let facing = Quat::from_rotation_y(game.facing());
    let direction = game.aim.map_or(facing * Vec3::NEG_Z, |aim| *aim);
    let pitch = direction.y.clamp(-1.0, 1.0).asin();
    let position = hand.map_or_else(
        || game.arena.position() + facing * Vec3::new(0.25, 0.4, -0.4),
        GlobalTransform::translation,
    );
    // Apply yaw and elevation separately so diagonal aim cannot roll the grip sideways.
    Transform::from_translation(position).with_rotation(
        facing * Quat::from_rotation_x(pitch) * Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
    )
}
