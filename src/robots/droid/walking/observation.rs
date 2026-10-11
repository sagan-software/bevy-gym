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
