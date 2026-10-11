//! Closed living hull states for each team.
/// A drone hull has exactly one through six remaining hits.
#[derive(Debug, Clone, Copy)]
pub(super) enum DroneHull {
    /// Six remaining hits.
    Six,
    /// Five remaining hits.
    Five,
    /// Four remaining hits.
    Four,
    /// Three remaining hits.
    Three,
    /// Two remaining hits.
    Two,
    /// One remaining hit.
    One,
}
/// A droid hull cannot represent more than three remaining hits.
#[derive(Debug, Clone, Copy)]
pub(super) enum DroidHull {
    /// Three remaining hits.
    Three,
    /// Two remaining hits.
    Two,
    /// One remaining hit.
    One,
}
impl DroneHull {
    /// Consume one hit; absence is destruction, never a living zero.
    pub(super) const fn next(self) -> Option<Self> {
        match self {
            Self::Six => Some(Self::Five),
            Self::Five => Some(Self::Four),
            Self::Four => Some(Self::Three),
            Self::Three => Some(Self::Two),
            Self::Two => Some(Self::One),
            Self::One => None,
        }
    }
}
impl DroidHull {
    /// Consume one hit; absence is destruction, never a living zero.
    pub(super) const fn next(self) -> Option<Self> {
        match self {
            Self::Three => Some(Self::Two),
            Self::Two => Some(Self::One),
            Self::One => None,
        }
    }
}
