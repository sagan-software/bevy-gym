//! Collision identity derived from the owning physical body.
use super::RobotSlot;
use crate::robots::DroidBody;
/// A solid belongs to exactly one physical robot segment or to the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotRayTarget {
    /// Floor or other nonrobot solid geometry.
    Environment,
    /// One drone's original collision proxy.
    Drone(RobotSlot),
    /// One physical segment of an articulated droid.
    Droid(RobotSlot, DroidBody),
}

impl RobotRayTarget {
    /// Derive an optional robot identity from its physical collision classification.
    pub(super) const fn owner(self) -> Option<super::RobotId> {
        match self {
            Self::Environment => None,
            Self::Drone(slot) => Some(super::RobotId::Drone(slot)),
            Self::Droid(slot, _) => Some(super::RobotId::Droid(slot)),
        }
    }
    /// Stable priority for equal-distance solids, independent of collider iteration order.
    pub(super) const fn rank(self) -> usize {
        match self {
            Self::Environment => 0,
            Self::Drone(slot) => 1 + slot.index(),
            Self::Droid(slot, body) => 4 + slot.index() * 13 + body as usize,
        }
    }
}
