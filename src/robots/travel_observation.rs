//! Read-only physical state and destination for learned travel control.

use super::{DroneDestination, DroneObservation};

/// A single physical snapshot with the episode's immutable task destination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneTravelObservation {
    /// State produced by the shared physics solver.
    pub(super) body: DroneObservation,
    /// Position and heading supplied as task information to the policy.
    pub(super) destination: DroneDestination,
}

impl DroneTravelObservation {
    /// Read the physical snapshot without exposing mutable solver state.
    #[must_use]
    pub const fn body(self) -> DroneObservation {
        self.body
    }

    /// Read the fixed destination for this episode.
    #[must_use]
    pub const fn destination(self) -> DroneDestination {
        self.destination
    }
}
