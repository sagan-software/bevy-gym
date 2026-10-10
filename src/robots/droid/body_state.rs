//! Read-only physical state of one articulated segment.

use bevy::math::{Quat, Vec3};

/// Segment state in metres and seconds, with +Y up and bind-pose forward -Z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroidBodyState {
    /// Centre of the collider in world metres.
    pub(super) position: Vec3,
    /// Rotation from segment coordinates to world coordinates.
    pub(super) orientation: Quat,
    /// World linear velocity in metres per second.
    pub(super) linear_velocity: Vec3,
    /// World angular velocity in radians per second.
    pub(super) angular_velocity: Vec3,
    /// Whether this segment has an active floor contact.
    pub(super) floor_contact: bool,
}
impl DroidBodyState {
    /// Return the segment centre in world metres.
    #[must_use]
    pub const fn position(self) -> Vec3 {
        self.position
    }
    /// Return the unit quaternion from segment to world coordinates.
    #[must_use]
    pub const fn orientation(self) -> Quat {
        self.orientation
    }
    /// Return world velocity in metres per second.
    #[must_use]
    pub const fn linear_velocity(self) -> Vec3 {
        self.linear_velocity
    }
    /// Return world angular velocity in radians per second.
    #[must_use]
    pub const fn angular_velocity(self) -> Vec3 {
        self.angular_velocity
    }
    /// Report active physical floor contact.
    #[must_use]
    pub const fn floor_contact(self) -> bool {
        self.floor_contact
    }
}
