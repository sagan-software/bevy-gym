//! Finite `MountainCar` state with documented physical units.

/// Position and velocity in Gymnasium coordinate order.
///
/// Position uses the source coordinate units; velocity is displacement per transition.
/// Finite states beyond the episode limits remain valid for terminal-state tests.
/// The terrain angle `3 * position` must remain finite; the infallible step API
/// cannot represent Python's domain error for an infinite cosine argument.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MountainCarState {
    /// Ordered finite state in Gymnasium's internal double precision.
    values: [f64; 2],
}

/// A coordinate or the derived terrain angle is nonfinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidMountainCarState;

impl TryFrom<[f64; 2]> for MountainCarState {
    type Error = InvalidMountainCarState;

    fn try_from(values: [f64; 2]) -> Result<Self, Self::Error> {
        let [position, velocity] = values;
        if (3.0 * position).is_finite() && velocity.is_finite() {
            Ok(Self { values })
        } else {
            Err(InvalidMountainCarState)
        }
    }
}

impl std::fmt::Display for InvalidMountainCarState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("MountainCar state requires finite velocity and finite 3 * position")
    }
}

impl std::error::Error for InvalidMountainCarState {}

impl MountainCarState {
    /// Borrow coordinates in position, velocity order.
    #[must_use]
    pub const fn values(&self) -> &[f64; 2] {
        &self.values
    }
}
