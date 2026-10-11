//! Own-body walking observations with their immutable task command.

use super::super::DroidObservation;
use super::DroidVelocityCommand;

/// One physical snapshot and the command fixed throughout its episode.
/// No other agents, rewards or critic-only values are exposed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroidWalkingObservation {
    /// The original thirteen-segment observation profile.
    pub(super) physical: DroidObservation,
    /// Validated velocity requested by the lesson setup.
    pub(super) command: DroidVelocityCommand,
}

impl DroidWalkingObservation {
    /// Encode own-body state and the command as 207 fixed actor inputs.
    ///
    /// Root XYZ groups are height-only target displacement in metres, world up,
    /// linear velocity divided by 2 m/s, and angular velocity divided by 4 rad/s.
    /// Twelve other segments follow `DroidBody::ALL` excluding pelvis: relative
    /// position in metres, up, forward, relative linear velocity divided by 2 m/s,
    /// relative angular velocity divided by 4 rad/s, and contact as 0 or 1.
    /// All vectors use the pelvis frame. The final three inputs are the command's
    /// world `(X, 0, Z)` velocity divided by 1 m/s and rotated into that frame.
    /// Tilt can therefore produce a nonzero command Y component.
    ///
    /// The height target is 0.99 m. Horizontal translation does not create a
    /// return-to-origin target. Equivalent quaternion signs retain the same inputs.
    /// Nonfinite measurements remain nonfinite for rejection by the policy boundary.
    /// No reward, work, foot sliding, other robot or critic state enters these inputs.
    /// Encoding takes fixed O(13) time and returns a fixed-size array.
    #[must_use]
    pub fn actor_features(&self) -> [f32; 207] {
        super::encoding::encode(self)
    }

    /// Borrow the body's measured state without exposing the solver.
    #[must_use]
    pub const fn physical(&self) -> &DroidObservation {
        &self.physical
    }

    /// Read the episode's validated X/Z velocity request.
    #[must_use]
    pub const fn command(&self) -> DroidVelocityCommand {
        self.command
    }
}
