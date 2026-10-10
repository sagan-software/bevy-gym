//! Agent identity with team derived from its closed variant.

use super::{RobotSlot, RobotTeam};

/// A robot's team and slot cannot contradict one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotId {
    /// One drone slot.
    Drone(RobotSlot),
    /// One droid slot.
    Droid(RobotSlot),
}

impl RobotId {
    /// Read the team from the identity instead of storing a second mutable value.
    #[must_use]
    pub const fn team(self) -> RobotTeam {
        match self {
            Self::Drone(_) => RobotTeam::Drones,
            Self::Droid(_) => RobotTeam::Droids,
        }
    }

    /// Read the slot within the derived team.
    #[must_use]
    pub const fn slot(self) -> RobotSlot {
        match self {
            Self::Drone(slot) | Self::Droid(slot) => slot,
        }
    }
}
