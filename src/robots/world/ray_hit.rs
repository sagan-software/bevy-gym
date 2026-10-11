//! Read-only physical hit and its measured distance.
use super::RobotRayTarget;
/// Nearest solid hit; callers cannot forge a collision or its distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobotRayHit {
    /// Identity derived from the hit collider's actual owner.
    pub(super) target: RobotRayTarget,
    /// Finite nonnegative distance along the normalized ray, in metres.
    pub(super) metres: f32,
}
impl RobotRayHit {
    /// Inspect the hit solid without exposing mutable physics.
    #[must_use]
    pub const fn target(self) -> RobotRayTarget {
        self.target
    }
    /// Read the hit distance in metres; zero means the origin is inside the solid.
    #[must_use]
    pub const fn metres(self) -> f32 {
        self.metres
    }
}
