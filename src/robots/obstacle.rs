//! Validated static boxes for a private flight collision world.

use std::fmt;

use bevy::math::{Quat, Vec3, Vec4};
use rapier3d::prelude::{ColliderBuilder, PhysicsWorld, RigidBodyBuilder, Vector};

/// An immutable solid box described by its centre, half extents, and rotation.
///
/// Construct with `DroneObstacle::try_from((centre, half_extents, rotation))`.
/// Centres are metres within ±1,000 on each axis. Half extents are metres in
/// `(0, 1,000]`. All components must be finite. Nonzero quaternions are normalized.
/// Validation checks centre, half extents, then rotation.
///
/// ```compile_fail
/// use bevy_gym::robots::DroneObstacle;
/// use bevy::math::{Quat, Vec3};
/// let mut obstacle = DroneObstacle::try_from((Vec3::ZERO, Vec3::ONE, Quat::IDENTITY)).unwrap();
/// obstacle.half_extents = Vec3::ZERO;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneObstacle {
    /// World-space centre in metres.
    centre: Vec3,
    /// Positive local half extents in metres.
    half_extents: Vec3,
    /// Unit rotation from local coordinates to the world frame.
    rotation: Quat,
}

/// The first invalid field in a static flight obstacle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidDroneObstacle {
    /// A centre component is non-finite or outside ±1,000 metres.
    Centre,
    /// A half extent is non-finite, nonpositive, or greater than 1,000 metres.
    HalfExtents,
    /// The quaternion is zero or contains a non-finite component.
    Rotation,
}

impl TryFrom<(Vec3, Vec3, Quat)> for DroneObstacle {
    type Error = InvalidDroneObstacle;

    fn try_from((centre, half_extents, rotation): (Vec3, Vec3, Quat)) -> Result<Self, Self::Error> {
        // Reject invalid geometry before any value can reach the collision solver.
        if !centre.is_finite() || centre.abs().max_element() > 1_000.0 {
            return Err(InvalidDroneObstacle::Centre);
        }
        if !half_extents.is_finite()
            || half_extents.min_element() <= 0.0
            || half_extents.max_element() > 1_000.0
        {
            return Err(InvalidDroneObstacle::HalfExtents);
        }
        let scale = Vec4::from_array(rotation.to_array()).abs().max_element();
        if !rotation.is_finite() || scale <= 0.0 {
            return Err(InvalidDroneObstacle::Rotation);
        }
        // Scaling first avoids overflow and underflow when normalizing finite input.
        Ok(Self {
            centre,
            half_extents,
            rotation: (rotation / scale).normalize(),
        })
    }
}

impl DroneObstacle {
    /// Insert one fixed box using Rapier 0.36's world translation and scaled-axis rotation.
    ///
    /// <https://docs.rs/rapier3d/0.36.0/rapier3d/dynamics/struct.RigidBodyBuilder.html>
    pub(super) fn insert(self, world: &mut PhysicsWorld) {
        world.insert(
            RigidBodyBuilder::fixed()
                .translation(Vector::from_array(self.centre.to_array()))
                .rotation(Vector::from_array(
                    self.rotation.to_scaled_axis().to_array(),
                )),
            ColliderBuilder::cuboid(
                self.half_extents.x,
                self.half_extents.y,
                self.half_extents.z,
            ),
        );
    }
}

impl fmt::Display for InvalidDroneObstacle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Centre => "Obstacle centre must be finite and within +/-1000 metres.",
            Self::HalfExtents => "Obstacle half extents must be finite and in (0, 1000] metres.",
            Self::Rotation => "Obstacle rotation must be finite and nonzero.",
        })
    }
}

impl std::error::Error for InvalidDroneObstacle {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_retains_rotation_and_insertion_keeps_boxes_fixed() {
        let rotation = Quat::from_rotation_z(0.7);
        for scale in [-2.0, 1.0, 2.0] {
            let obstacle = DroneObstacle::try_from((
                Vec3::new(1.0, 2.0, 3.0),
                Vec3::new(0.2, 0.3, 0.4),
                rotation * scale,
            ))
            .expect("Valid box");
            assert!((obstacle.rotation.length() - 1.0).abs() < 1e-6);
            assert!((obstacle.rotation * Vec3::X - rotation * Vec3::X).length() < 1e-6);
            let mut world = PhysicsWorld::default();
            obstacle.insert(&mut world);
            world.step();
            let (_, body) = world.bodies.iter().next().expect("One fixed body");
            assert!(body.is_fixed());
            assert_eq!(
                body.translation().to_array().map(f32::to_bits),
                obstacle.centre.to_array().map(f32::to_bits)
            );
            let (_, collider) = world.colliders.iter().next().expect("One cuboid");
            let shape = collider.shape().as_cuboid().expect("Box");
            assert_eq!(
                shape.half_extents.to_array().map(f32::to_bits),
                obstacle.half_extents.to_array().map(f32::to_bits)
            );
        }
    }
}
