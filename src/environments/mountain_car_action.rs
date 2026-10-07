//! Closed force vocabulary for `MountainCar`.

/// The three engine actions in Gymnasium's `Discrete(3)` action space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountainCarAction {
    /// Apply a negative engine increment.
    Left,
    /// Apply no engine increment.
    Coast,
    /// Apply a positive engine increment.
    Right,
}

/// An index outside `MountainCar`'s three-action vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidMountainCarAction;

impl TryFrom<usize> for MountainCarAction {
    type Error = InvalidMountainCarAction;

    fn try_from(index: usize) -> Result<Self, Self::Error> {
        match index {
            0 => Ok(Self::Left),
            1 => Ok(Self::Coast),
            2 => Ok(Self::Right),
            _ => Err(InvalidMountainCarAction),
        }
    }
}

impl From<MountainCarAction> for usize {
    fn from(action: MountainCarAction) -> Self {
        match action {
            MountainCarAction::Left => 0,
            MountainCarAction::Coast => 1,
            MountainCarAction::Right => 2,
        }
    }
}

impl std::fmt::Display for InvalidMountainCarAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("MountainCar action must be 0, 1, or 2")
    }
}

impl std::error::Error for InvalidMountainCarAction {}
