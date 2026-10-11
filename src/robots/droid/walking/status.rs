//! Distinct running, physical fall and invalid-physics outcomes.

use super::super::DroidBody;

/// Machine-readable walking lifecycle, returned as environment information.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DroidWalkingStatus {
    /// The task still accepts policy-chosen torques.
    #[default]
    Running,
    /// A finite physical state violates the walking task's bounds.
    Fallen(
        /// Measured finite reason for physical termination.
        DroidWalkingFall,
    ),
    /// A segment has nonfinite position, rotation or velocity; physics failed.
    Failed(
        /// First invalid segment in `DroidBody::ALL` order.
        DroidBody,
    ),
}

/// Finite physical termination, distinct from malformed commands or failed physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroidWalkingFall {
    /// Pelvis height is below 0.5 m.
    LowPelvis,
    /// The identified segment is outside X/Z ±29 m or Y [0, 3] m.
    OutsideBounds(
        /// First out-of-bounds segment in `DroidBody::ALL` order.
        DroidBody,
    ),
    /// A segment other than either foot touches the floor.
    NonFootContact(
        /// First contacting non-foot segment in `DroidBody::ALL` order.
        DroidBody,
    ),
}
