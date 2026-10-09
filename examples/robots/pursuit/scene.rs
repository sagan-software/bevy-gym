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
    assets: Res<'_, AssetServer>,
) {
    let floor = materials.add(Color::srgb(0.39, 0.43, 0.44));
    let wall = materials.add(Color::srgb(0.65, 0.67, 0.66));
    let cover = materials.add(Color::srgb(0.73, 0.29, 0.09));
    let pipe = materials.add(Color::srgb(0.22, 0.37, 0.41));
    for block in game.arena.blocks() {
        if block.surface == Surface::Cover {
            commands.spawn((
                SceneRoot(
                    assets.load(
                        GltfAssetLabel::Scene(0)
                            .from_asset("robots/survival/cover/Prop_Crate_Large.gltf"),
                    ),
                ),
                cover_pose(block),
            ));
            continue;
        }
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
    robot::spawn(&mut commands, &assets);
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
    commands.spawn((
        Camera3d::default(),
        super::view::camera(),
        SpatialListener::new(0.2),
        Transform::default(),
    ));
}

/// Fit the source crate's measured bounds to the authoritative solid cover box.
fn cover_pose(block: &super::arena::layout::Block) -> Transform {
    // Metres from the source glTF POSITION accessor; its origin lies at the base.
    let min = Vec3::new(-1.733_024_2, 0.001_476_556, -0.750_000_1);
    let max = Vec3::new(1.733_024_4, 1.501_478_1, 0.750_000_1);
    let scale = block.half * 2.0 / (max - min);
    Transform {
        translation: block.centre - block.rotation * ((min + max) * 0.5 * scale),
        rotation: block.rotation,
        scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_bounds_match_every_cover_collider() {
        for block in super::super::arena::layout::blocks()
            .iter()
            .filter(|block| block.surface == Surface::Cover)
        {
            let pose = cover_pose(block);
            for (point, sign) in [
                (Vec3::new(-1.733_024_2, 0.001_476_556, -0.750_000_1), -1.0),
                (Vec3::new(1.733_024_4, 1.501_478_1, 0.750_000_1), 1.0),
            ] {
                let expected = block.centre + block.rotation * (block.half * sign);
                assert!((pose.transform_point(point) - expected).length() < 1e-5);
            }
        }
    }
}

/// Follow from above and behind; the capsule remains the only movement authority.
pub(super) fn project(
    game: Res<'_, Game>,
    mut character: Query<'_, '_, &mut Transform, (With<Character>, Without<Camera3d>)>,
) {
    let position = game.arena.position();
    for mut transform in &mut character {
        transform.translation = position;
        transform.rotation = Quat::from_rotation_y(game.facing());
    }
}
