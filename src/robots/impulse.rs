//! Validated body-local impact points and world-space momentum for drone hits.

use std::{error::Error, fmt};

use bevy::math::Vec3;
use rapier3d::prelude::{RigidBody, Vector};

/// A bounded contact point in metres and momentum in newton-seconds.
///
/// Construct with `DroneImpulse::try_from((body_local_point, world_momentum))`.
/// Point components must be within ±1 metre; momentum magnitude must be at most
/// 10 N·s. All components must be finite. Zero momentum is valid. Validation checks
/// the point first. These bounds belong to this flight lesson.
///
/// ```compile_fail
/// use bevy_gym::robots::DroneImpulse;
/// use bevy::math::Vec3;
/// let unchecked = DroneImpulse { point: Vec3::NAN, momentum: Vec3::ZERO };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneImpulse {
    /// Contact point in body-local metres.
    point: Vec3,
    /// Momentum in world-space newton-seconds.
    momentum: Vec3,
}

/// The first invalid field in an impact request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidDroneImpulse {
    /// The point is nonfinite or has a component outside ±1 metre.
    Point,
    /// Momentum is nonfinite or its magnitude exceeds 10 newton-seconds.
    Momentum,
}

/// An ended episode cannot accept another physical hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroneImpulseRejected;

impl TryFrom<(Vec3, Vec3)> for DroneImpulse {
    type Error = InvalidDroneImpulse;

    fn try_from((point, momentum): (Vec3, Vec3)) -> Result<Self, Self::Error> {
        // Validate the contact before momentum so malformed requests have stable errors.
        if !point.is_finite() || point.abs().max_element() > 1.0 {
            return Err(InvalidDroneImpulse::Point);
        }
        // Squared magnitude is in (N·s)²; 100 is the square of the 10 N·s limit.
        if !momentum.is_finite() || momentum.length_squared() > 100.0 {
            return Err(InvalidDroneImpulse::Momentum);
        }
        Ok(Self { point, momentum })
    }
}

impl DroneImpulse {
    /// Rapier 0.36 accepts world momentum and derives torque at the world contact.
    ///
    /// <https://docs.rs/rapier3d/0.36.0/rapier3d/dynamics/struct.RigidBody.html#method.apply_impulse_at_point>
    pub(super) fn apply(self, body: &mut RigidBody) {
        let local = Vector::from_array(self.point.to_array());
        let point = body.translation() + body.rotation() * local;
        body.apply_impulse_at_point(Vector::from_array(self.momentum.to_array()), point, true);
    }
}

impl fmt::Display for InvalidDroneImpulse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Point => "drone contact components must be finite and within ±1 metre",
            Self::Momentum => "drone momentum must be finite with magnitude at most 10 N·s",
        })
    }
}

impl Error for InvalidDroneImpulse {}

impl fmt::Display for DroneImpulseRejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cannot apply an impulse after the drone episode has ended")
    }
}

impl Error for DroneImpulseRejected {}

#[cfg(test)]
mod tests {
    use super::*;
    use rapier3d::prelude::RigidBodyBuilder;

    #[test]
    fn zero_impulse_preserves_private_body_pose_and_mass() {
        let mut body = RigidBodyBuilder::dynamic()
            .translation(Vector::Y)
            .additional_mass(2.0)
            .build();
        let before = (*body.position(), body.mass());
        DroneImpulse {
            point: Vec3::ONE,
            momentum: Vec3::ZERO,
        }
        .apply(&mut body);
        assert_eq!((*body.position(), body.mass()), before);
        assert_eq!(body.linvel(), Vector::ZERO);
        assert_eq!(body.angvel(), Vector::ZERO);
    }
}
