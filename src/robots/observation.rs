//! Read-only position, orientation, and velocity in the world frame.

use bevy::math::{Quat, Vec3};

/// Drone state in a right-handed frame with +X right, +Y up, and -Z forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneObservation {
    /// Body centre position in metres.
    pub(super) position: Vec3,
    /// Unit quaternion rotating body coordinates into the world frame.
    pub(super) orientation: Quat,
    /// World linear velocity in metres per second.
    pub(super) linear_velocity: Vec3,
    /// World angular velocity in radians per second.
    pub(super) angular_velocity: Vec3,
}

impl DroneObservation {
    /// Return the body centre position in metres.
    #[must_use]
    pub const fn position(self) -> Vec3 {
        self.position
    }

    /// Return the unit quaternion rotating body coordinates into world coordinates.
    #[must_use]
    pub const fn orientation(self) -> Quat {
        self.orientation
    }

    /// Return world linear velocity in metres per second.
    #[must_use]
    pub const fn linear_velocity(self) -> Vec3 {
        self.linear_velocity
    }

    /// Return world angular velocity in radians per second.
    #[must_use]
    pub const fn angular_velocity(self) -> Vec3 {
        self.angular_velocity
    }
}
