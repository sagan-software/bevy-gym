//! Read-only measurements at one common decision boundary.

use super::{RobotFrame, RobotSlot};
use crate::robots::{DroidObservation, DroneObservation};
use std::time::Duration;

/// All six authoritative observations captured before any subsequent actuator frame.
#[derive(Debug, Clone)]
pub struct RobotSnapshot {
    /// Opaque source boundary, used to reject stale and wrong-world actions.
    pub(super) frame: RobotFrame,
    /// Drone states in the stable slot order.
    pub(super) drones: [DroneObservation; 3],
    /// Complete articulated states in the stable slot order.
    pub(super) droids: [DroidObservation; 3],
}

impl RobotSnapshot {
    /// Borrow the identity that binds actions to this physical boundary.
    #[must_use]
    pub const fn frame(&self) -> &RobotFrame {
        &self.frame
    }

    /// Read one drone without exposing a body handle or mutable backend.
    #[must_use]
    pub const fn drone(&self, slot: RobotSlot) -> DroneObservation {
        let [first, second, third] = self.drones;
        match slot {
            RobotSlot::First => first,
            RobotSlot::Second => second,
            RobotSlot::Third => third,
        }
    }

    /// Read all thirteen physical segments of one droid.
    #[must_use]
    pub const fn droid(&self, slot: RobotSlot) -> DroidObservation {
        let [first, second, third] = self.droids;
        match slot {
            RobotSlot::First => first,
            RobotSlot::Second => second,
            RobotSlot::Third => third,
        }
    }

    /// Read policy time derived from complete physical frames since reset.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.frame.elapsed()
    }
}
