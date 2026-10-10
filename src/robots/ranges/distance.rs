//! Validated hit distance in the fixed ten-metre range profile.

/// Finite distance from the body centre to a solid hit, in [0, 10] metres.
///
/// Construct through `TryFrom<f32>`. Accepted bits, including negative zero,
/// are preserved; nonfinite values are rejected before finite range checks.
///
/// ```compile_fail
/// use bevy_gym::robots::DroneRangeDistance;
/// let unchecked = DroneRangeDistance(-1.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DroneRangeDistance(f32);

impl DroneRangeDistance {
    /// Inclusive sensor limit; a hit at this distance differs from no hit.
    pub const MAX: Self = Self(10.0);

    /// Read the measured distance in metres.
    #[must_use]
    pub const fn metres(self) -> f32 {
        self.0
    }
}

/// Invalid distance supplied at the public numeric boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidDroneRangeDistance {
    /// NaN or either infinity.
    NonFinite,
    /// A finite value outside [0, 10] metres.
    OutOfRange,
}

impl TryFrom<f32> for DroneRangeDistance {
    type Error = InvalidDroneRangeDistance;

    fn try_from(metres: f32) -> Result<Self, Self::Error> {
        // Reject malformed values before checking the sensor profile's finite bounds.
        if !metres.is_finite() {
            return Err(InvalidDroneRangeDistance::NonFinite);
        }
        if !(0.0..=Self::MAX.metres()).contains(&metres) {
            return Err(InvalidDroneRangeDistance::OutOfRange);
        }
        Ok(Self(metres))
    }
}

impl std::fmt::Display for InvalidDroneRangeDistance {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NonFinite => "Range distance must be finite.",
            Self::OutOfRange => "Range distance must be in [0, 10] metres.",
        })
    }
}

impl std::error::Error for InvalidDroneRangeDistance {}
