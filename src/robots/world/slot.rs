//! Three closed slots shared by both teams.

/// Position within a three-agent team, independent of physical body handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotSlot {
    /// First agent in its team.
    First,
    /// Second agent in its team.
    Second,
    /// Third agent in its team.
    Third,
}

impl RobotSlot {
    /// Stable array and action order for both teams.
    pub const ALL: [Self; 3] = [Self::First, Self::Second, Self::Third];

    /// Map a closed slot to its fixed array position.
    pub(super) const fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
            Self::Third => 2,
        }
    }
}
