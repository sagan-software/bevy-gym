//! Read-only projection from ecosystem snapshots into Bevy sprites.

use bevy::prelude::*;
use bevy_gym::ecosystem::{VisualObjectType, VisualSpecies};

use super::ActiveSession;

/// Spawn a camera fitted to the ecosystem's fixed square map.
pub(crate) fn setup_camera(mut commands: Commands<'_, '_>) {
    commands.spawn((
        Camera2d,
        Projection::from(OrthographicProjection {
            scale: 0.085,
            ..OrthographicProjection::default_2d()
        }),
        Name::new("Browser ecosystem camera"),
    ));
}

/// Draw the current authoritative snapshot without mutating the simulation.
pub(crate) fn draw_world(active: NonSend<'_, ActiveSession>, mut gizmos: Gizmos<'_, '_>) {
    let Some(session) = active.0.as_ref() else {
        return;
    };
    let snapshot = session.snapshot();
    let extent = snapshot.map_half_extent;
    gizmos.rect_2d(
        Vec2::ZERO,
        Vec2::splat(extent * 2.0),
        Color::srgb_u8(84, 108, 91),
    );

    // Resource and obstacle colors stay stable across every route.
    for object in &snapshot.objects {
        let color = match object.kind {
            VisualObjectType::Food => Color::srgb_u8(244, 211, 94),
            VisualObjectType::Well => Color::srgb_u8(54, 162, 235),
            VisualObjectType::Tree => Color::srgb_u8(54, 130, 78),
            VisualObjectType::Rock => Color::srgb_u8(130, 139, 145),
            VisualObjectType::Thorn => Color::srgb_u8(214, 79, 139),
            VisualObjectType::Shelter => Color::srgb_u8(144, 116, 219),
        };
        gizmos
            .circle_2d(Vec2::from_array(object.position), object.radius, color)
            .resolution(20);
    }

    // Faint exact perception sectors make policy inputs visible without hiding bodies.
    for ray in &snapshot.rays {
        gizmos.line_2d(
            Vec2::from_array(ray.start),
            Vec2::from_array(ray.end),
            Color::srgba(0.54, 0.68, 0.58, 0.14),
        );
    }

    for agent in &snapshot.agents {
        let color = if !agent.is_alive {
            Color::srgb_u8(76, 81, 78)
        } else {
            match agent.species {
                VisualSpecies::Bunny => Color::srgb_u8(235, 239, 230),
                VisualSpecies::Fox => Color::srgb_u8(232, 116, 57),
            }
        };
        let position = Vec2::from_array(agent.position);
        let rotation = Rot2::radians(agent.heading);
        gizmos
            .ellipse_2d(
                Isometry2d::new(position, rotation),
                Vec2::new(0.78, 0.58),
                color,
            )
            .resolution(20);
        let forward = rotation * Vec2::X;
        gizmos.line_2d(position, position + forward * 1.1, color);
    }
}
