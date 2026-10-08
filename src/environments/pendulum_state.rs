//! Validated diagnostic Pendulum state in the upstream coordinate order.

/// Unwrapped angle in radians and angular velocity in radians per second.
///
/// The angle and squared angular velocity must be finite. The latter restriction
/// prevents an infinite reward through the infallible [`crate::Env::step`] API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PendulumState {
    /// Coordinates retained in Gymnasium's double precision.
    values: [f64; 2],
}

/// A nonfinite angle or angular velocity whose square is nonfinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidPendulumState;

impl TryFrom<[f64; 2]> for PendulumState {
    type Error = InvalidPendulumState;
    fn try_from(values: [f64; 2]) -> Result<Self, Self::Error> {
        let [angle, velocity] = values;
        if angle.is_finite() && velocity.powi(2).is_finite() {
            Ok(Self { values })
        } else {
            Err(InvalidPendulumState)
        }
    }
}
impl PendulumState {
    /// Borrow the angle and angular velocity in source coordinate order.
    #[must_use]
    pub const fn values(&self) -> &[f64; 2] {
        &self.values
    }
}
impl std::fmt::Display for InvalidPendulumState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .write_str("Pendulum requires a finite angle and a finite squared angular velocity")
    }
}
impl std::error::Error for InvalidPendulumState {}
