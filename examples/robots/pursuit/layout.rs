//! One immutable blockout description for physics, sight, and rendering.

use bevy::math::{Quat, Vec3};

/// Material groups identify landmarks without changing collision behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Surface {
    /// Walkable concrete floor.
    Floor,
    /// House and perimeter concrete.
    Wall,
    /// Scattered orange cover blocks.
    Cover,
    /// Blue-gray pipe segments.
    Pipe,
}

/// An oriented solid box shared by the mesh and collider constructors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    /// World-space box centre in metres.
    pub(crate) centre: Vec3,
    /// Half dimensions along local X, Y, and Z, in metres.
    pub(crate) half: Vec3,
    /// World rotation of the local box axes.
    pub(crate) rotation: Quat,
    /// Visual material group.
    pub(crate) surface: Surface,
}

impl Block {
    /// Describe an axis-aligned solid; all inputs are authored layout constants.
    const fn new(centre: Vec3, half: Vec3, surface: Surface) -> Self {
        Self {
            centre,
            half,
            rotation: Quat::IDENTITY,
            surface,
        }
    }
}

/// Build the 26-metre arena, windowed house, open pipe, and scattered cover.
pub(super) fn blocks() -> Vec<Block> {
    let mut blocks = vec![Block::new(
        Vec3::new(0.0, -0.1, 0.0),
        Vec3::new(13.0, 0.1, 13.0),
        Surface::Floor,
    )];
    for sign in [-1.0, 1.0] {
        blocks.push(Block::new(
            Vec3::new(sign * 13.0, 1.25, 0.0),
            Vec3::new(0.2, 1.25, 13.0),
            Surface::Wall,
        ));
        blocks.push(Block::new(
            Vec3::new(0.0, 1.25, sign * 13.0),
            Vec3::new(13.0, 1.25, 0.2),
            Surface::Wall,
        ));
    }
    house(&mut blocks);
    pipe(&mut blocks);
    for (centre, half) in [
        (Vec3::new(-3.0, 0.6, 5.0), Vec3::new(1.1, 0.6, 0.5)),
        (Vec3::new(1.8, 0.9, 5.0), Vec3::new(0.5, 0.9, 1.0)),
        (Vec3::new(-8.0, 0.7, 7.0), Vec3::new(0.6, 0.7, 1.5)),
        (Vec3::new(8.5, 1.0, -7.0), Vec3::new(1.2, 1.0, 0.5)),
    ] {
        blocks.push(Block::new(centre, half, Surface::Cover));
    }
    blocks
}

/// House bounds X=-8..-2, Z=-6..0; front window and east door are actual gaps.
fn house(blocks: &mut Vec<Block>) {
    for (centre, half) in [
        // Front window: X=-6..-4, Y=1.0..2.5, through Z=0.
        (Vec3::new(-5.0, 0.5, 0.0), Vec3::new(3.0, 0.5, 0.15)),
        (Vec3::new(-5.0, 2.95, 0.0), Vec3::new(3.0, 0.45, 0.15)),
        (Vec3::new(-7.0, 1.75, 0.0), Vec3::new(1.0, 0.75, 0.15)),
        (Vec3::new(-3.0, 1.75, 0.0), Vec3::new(1.0, 0.75, 0.15)),
        // East door: Z=-4..-2, Y=0..2.4, through X=-2.
        (Vec3::new(-2.0, 2.9, -3.0), Vec3::new(0.15, 0.5, 3.0)),
        (Vec3::new(-2.0, 1.2, -5.0), Vec3::new(0.15, 1.2, 1.0)),
        (Vec3::new(-2.0, 1.2, -1.0), Vec3::new(0.15, 1.2, 1.0)),
        (Vec3::new(-8.0, 1.7, -3.0), Vec3::new(0.15, 1.7, 3.0)),
        (Vec3::new(-5.0, 1.7, -6.0), Vec3::new(3.0, 1.7, 0.15)),
        (Vec3::new(-5.0, 3.5, -3.0), Vec3::new(3.15, 0.1, 3.15)),
    ] {
        blocks.push(Block::new(centre, half, Surface::Wall));
    }
}

/// Twelve overlapping wall segments leave both ends of the six-metre pipe open.
fn pipe(blocks: &mut Vec<Block>) {
    for index in 0..12 {
        let angle = index as f32 * std::f32::consts::TAU / 12.0;
        blocks.push(Block {
            centre: Vec3::new(
                1.7_f32.mul_add(angle.cos(), 5.0),
                1.7_f32.mul_add(angle.sin(), 1.7),
                1.0,
            ),
            half: Vec3::new(0.48, 0.12, 3.0),
            rotation: Quat::from_rotation_z(angle + std::f32::consts::FRAC_PI_2),
            surface: Surface::Pipe,
        });
    }
}
