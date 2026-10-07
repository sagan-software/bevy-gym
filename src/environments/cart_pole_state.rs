//! Finite `CartPole` state with documented physical units.

/// Cart position, cart velocity, pole angle, and pole angular velocity.
///
/// Values use metres, metres per second, radians, and radians per second.
/// Finite states beyond the episode limits remain valid for terminal-state tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartPoleState {
    /// Ordered finite state in Gymnasium's internal double precision.
    values: [f64; 4],
}

/// At least one state coordinate is nonfinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidCartPoleState;

impl TryFrom<[f64; 4]> for CartPoleState {
    type Error = InvalidCartPoleState;

    fn try_from(values: [f64; 4]) -> Result<Self, Self::Error> {
        if values.iter().all(|value| value.is_finite()) {
            Ok(Self { values })
        } else {
            Err(InvalidCartPoleState)
        }
    }
}

impl std::fmt::Display for InvalidCartPoleState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("CartPole initial state must contain four finite coordinates")
    }
}

impl std::error::Error for InvalidCartPoleState {}

impl CartPoleState {
    /// Borrow coordinates in position, velocity, angle, angular-velocity order.
    #[must_use]
    pub const fn values(&self) -> &[f64; 4] {
        &self.values
    }
}
