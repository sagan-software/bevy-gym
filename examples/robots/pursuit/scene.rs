//! Render the authoritative blockout and follow the player.

use super::{arena::layout::Surface, robot, Game};
use bevy::prelude::*;

/// Rendered robot root, positioned at the collision capsule centre.
#[derive(Component)]
pub(super) struct Character;

/// Build meshes from the exact boxes used by the collision world.
pub(super) fn setup(
    mut commands: Commands<'_, '_>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
    game: Res<'_, Game>,
) {
    let floor = materials.add(Color::srgb(0.39, 0.43, 0.44));
    let wall = materials.add(Color::srgb(0.65, 0.67, 0.66));
    let cover = materials.add(Color::srgb(0.73, 0.29, 0.09));
    let pipe = materials.add(Color::srgb(0.22, 0.37, 0.41));
    for block in game.arena.blocks() {
        let material = match block.surface {
            Surface::Floor => &floor,
            Surface::Wall => &wall,
            Surface::Cover => &cover,
            Surface::Pipe => &pipe,
        };
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(block.half * 2.0))),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(block.centre).with_rotation(block.rotation),
        ));
    }
    robot::spawn(&mut commands, &mut meshes, &mut materials);
    commands.insert_resource(ClearColor(Color::srgb(0.65, 0.73, 0.76)));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((Camera3d::default(), Transform::default()));
}

/// Follow from above and behind; the capsule remains the only movement authority.
pub(super) fn project(
    game: Res<'_, Game>,
    mut character: Query<'_, '_, &mut Transform, (With<Character>, Without<Camera3d>)>,
    mut camera: Query<'_, '_, &mut Transform, (With<Camera3d>, Without<Character>)>,
) {
    let position = game.arena.position();
    for mut transform in &mut character {
        transform.translation = position;
        transform.rotation = Quat::from_rotation_y(game.heading);
    }
    let target = position + Vec3::Y * 0.4;
    let desired = target + Vec3::new(0.0, 3.3, 6.0);
    for mut transform in &mut camera {
        *transform = Transform::from_translation(
            game.arena
                .obstruction(target, desired)
                .map_or(desired, |distance| {
                    target + (desired - target).normalize() * (distance - 0.2).max(0.1)
                }),
        )
        .looking_at(target, Vec3::Y);
    }
}
