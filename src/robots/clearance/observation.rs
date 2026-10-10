//! One immutable physical snapshot pairs task information with body-frame ranges.

use crate::robots::{DroneRanges, DroneTravelObservation};

/// Task, physical body and body-frame ranges sampled without an intervening action.
///
/// Fields remain private so public callers cannot combine unrelated physical snapshots.
/// A copy retains its readings after the environment advances or resets.
///
/// ```compile_fail
/// use bevy::math::{Vec2, Vec3};
/// use bevy_gym::robots::{DroneClearance, DroneClearanceObservation, DroneDestination};
/// let goal = DroneDestination::try_from((Vec3::Y * 2.0, Vec2::NEG_Y)).unwrap();
/// let snapshot = DroneClearance::new(goal, []).observation();
/// let fabricated = DroneClearanceObservation {
///     travel: snapshot.travel(),
///     ranges: snapshot.ranges(),
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneClearanceObservation {
    /// Shared body's physical state and immutable destination.
    pub(super) travel: DroneTravelObservation,
    /// Six local rays measured from that same body's pose.
    pub(super) ranges: DroneRanges,
}

impl DroneClearanceObservation {
    /// Read the physical body and immutable task destination.
    #[must_use]
    pub const fn travel(self) -> DroneTravelObservation {
        self.travel
    }

    /// Read the six nearest-solid range readings, including the floor.
    #[must_use]
    pub const fn ranges(self) -> DroneRanges {
        self.ranges
    }
}
