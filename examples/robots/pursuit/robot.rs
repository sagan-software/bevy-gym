//! Original articulated robot geometry; walking follows measured ground displacement.

use super::{scene::Character, Game};
use bevy::prelude::*;

/// Hinge joints driven by the robot's walking phase.
#[derive(Component, Clone, Copy)]
pub(super) enum Joint {
    /// Left shoulder swings opposite the left hip.
    LeftShoulder,
    /// Right shoulder swings opposite the right hip.
    RightShoulder,
    /// Left hip swings forward and backward.
    LeftHip,
    /// Right hip swings opposite the left hip.
    RightHip,
}

/// Construct a robot with a dark chassis, orange panels, and a pale face plate.
pub(super) fn spawn(
    commands: &mut Commands<'_, '_>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let chassis = materials.add(Color::srgb(0.10, 0.13, 0.15));
    let panel = materials.add(Color::srgb(0.92, 0.49, 0.11));
    let face = materials.add(Color::srgb(0.82, 0.87, 0.86));
    let torso = meshes.add(Cuboid::new(0.46, 0.52, 0.30));
    let head = meshes.add(Cuboid::new(0.28, 0.28, 0.26));
    let visor = meshes.add(Cuboid::new(0.22, 0.065, 0.015));
    let arm = meshes.add(Capsule3d::new(0.065, 0.40));
    let leg = meshes.add(Capsule3d::new(0.085, 0.50));
    let foot = meshes.add(Cuboid::new(0.17, 0.10, 0.30));
    commands
        .spawn((Character, Transform::default(), Visibility::default()))
        .with_children(|root| {
            root.spawn((
                Mesh3d(torso),
                MeshMaterial3d(panel.clone()),
                Transform::from_xyz(0.0, 0.26, 0.0),
            ));
            root.spawn((
                Mesh3d(head),
                MeshMaterial3d(face),
                Transform::from_xyz(0.0, 0.68, 0.0),
            ));
            root.spawn((
                Mesh3d(visor),
                MeshMaterial3d(chassis.clone()),
                Transform::from_xyz(0.0, 0.69, -0.137),
            ));
            limbs(root, &arm, &leg, &foot, &chassis, &panel);
        });
}

/// Stop the walking cycle when collision prevents horizontal movement.
pub(super) fn animate(game: Res<'_, Game>, mut joints: Query<'_, '_, (&Joint, &mut Transform)>) {
    let swing = if game.motion.length() > 0.001 {
        (game.distance * 5.0).sin() * 0.45
    } else {
        0.0
    };
    for (joint, mut transform) in &mut joints {
        let sign = match joint {
            Joint::LeftShoulder | Joint::RightHip => -1.0,
            Joint::RightShoulder | Joint::LeftHip => 1.0,
        };
        transform.rotation = if matches!(joint, Joint::RightShoulder) && game.combat.is_armed() {
            Quat::from_rotation_x(std::f32::consts::FRAC_PI_2 + game.pitch())
        } else {
            Quat::from_rotation_x(swing * sign)
        };
    }
}

/// Attach alternating shoulder and hip hinges to the robot body.
fn limbs(
    root: &mut ChildSpawnerCommands<'_>,
    arm: &Handle<Mesh>,
    leg: &Handle<Mesh>,
    foot: &Handle<Mesh>,
    chassis: &Handle<StandardMaterial>,
    panel: &Handle<StandardMaterial>,
) {
    for (sign, shoulder, hip) in [
        (-1.0, Joint::LeftShoulder, Joint::LeftHip),
        (1.0, Joint::RightShoulder, Joint::RightHip),
    ] {
        root.spawn((
            shoulder,
            Transform::from_xyz(sign * 0.31, 0.46, 0.0),
            Visibility::default(),
        ))
        .with_children(|joint| {
            joint.spawn((
                Mesh3d(arm.clone()),
                MeshMaterial3d(chassis.clone()),
                Transform::from_xyz(0.0, -0.24, 0.0),
            ));
        });
        root.spawn((
            hip,
            Transform::from_xyz(sign * 0.14, -0.10, 0.0),
            Visibility::default(),
        ))
        .with_children(|joint| {
            joint.spawn((
                Mesh3d(leg.clone()),
                MeshMaterial3d(chassis.clone()),
                Transform::from_xyz(0.0, -0.34, 0.0),
            ));
            joint.spawn((
                Mesh3d(foot.clone()),
                MeshMaterial3d(panel.clone()),
                Transform::from_xyz(0.0, -0.74, -0.05),
            ));
        });
    }
}
