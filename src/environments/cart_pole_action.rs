//! Closed force vocabulary for `CartPole`.

/// The two forces in Gymnasium's `Discrete(2)` action space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartPoleAction {
    /// Apply a force of minus 10 newtons.
    Left,
    /// Apply a force of 10 newtons.
    Right,
}

/// An index outside `CartPole`'s two-action vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidCartPoleAction;

impl TryFrom<usize> for CartPoleAction {
    type Error = InvalidCartPoleAction;

    fn try_from(index: usize) -> Result<Self, Self::Error> {
        match index {
            0 => Ok(Self::Left),
            1 => Ok(Self::Right),
            _ => Err(InvalidCartPoleAction),
        }
    }
}

impl From<CartPoleAction> for usize {
    fn from(action: CartPoleAction) -> Self {
        match action {
            CartPoleAction::Left => 0,
            CartPoleAction::Right => 1,
        }
    }
}

impl std::fmt::Display for InvalidCartPoleAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CartPole action must be 0 or 1")
    }
}

impl std::error::Error for InvalidCartPoleAction {}
