//! Bounded read-only hull hit counts.
use super::hull::{DroidHull, DroneHull};
/// Zero through six hits derived from authoritative health, without a raw constructor.
///
/// ```compile_fail
/// use bevy_gym::robots::RobotHitCount;
/// let invalid = RobotHitCount::from(7_u8);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobotHitCount {
    /// Closed count prevents invalid integers from remaining representable.
    count: Count,
}
/// Every legal outgoing hit count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Count {
    /// Destroyed hull.
    Zero,
    /// One remaining hit.
    One,
    /// Two remaining hits.
    Two,
    /// Three remaining hits.
    Three,
    /// Four remaining hits.
    Four,
    /// Five remaining hits.
    Five,
    /// Six remaining hits.
    Six,
}
impl RobotHitCount {
    /// Derived count for either destroyed team.
    pub(super) const ZERO: Self = Self { count: Count::Zero };
    /// Project the bounded count for indicators or serialization.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.count as u8
    }
}
impl From<DroneHull> for RobotHitCount {
    fn from(hull: DroneHull) -> Self {
        let count = match hull {
            DroneHull::Six => Count::Six,
            DroneHull::Five => Count::Five,
            DroneHull::Four => Count::Four,
            DroneHull::Three => Count::Three,
            DroneHull::Two => Count::Two,
            DroneHull::One => Count::One,
        };
        Self { count }
    }
}
impl From<DroidHull> for RobotHitCount {
    fn from(hull: DroidHull) -> Self {
        let count = match hull {
            DroidHull::Three => Count::Three,
            DroidHull::Two => Count::Two,
            DroidHull::One => Count::One,
        };
        Self { count }
    }
}
