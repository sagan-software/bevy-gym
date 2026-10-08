//! Validated motor commands in front-left, front-right, rear-right, rear-left order.

use std::{error::Error, fmt};

/// Four finite motor thrust fractions in inclusive `[0, 1]`.
///
/// Motor order is front-left, front-right, rear-right, rear-left, viewed from above.
/// Use [`TryFrom`] to validate commands before stepping an environment.
///
/// ```compile_fail
/// use bevy_gym::robots::DroneAction;
/// let unchecked = DroneAction([f32::NAN; 4]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneAction([f32; 4]);

impl DroneAction {
    /// Return motor fractions in front-left, front-right, rear-right, rear-left order.
    #[must_use]
    pub const fn fractions(self) -> [f32; 4] {
        self.0
    }
}

impl TryFrom<[f32; 4]> for DroneAction {
    type Error = InvalidDroneAction;

    fn try_from(fractions: [f32; 4]) -> Result<Self, Self::Error> {
        // Reject the whole command before any motor can receive an invalid value.
        if fractions
            .iter()
            .all(|fraction| fraction.is_finite() && (0.0..=1.0).contains(fraction))
        {
            Ok(Self(fractions))
        } else {
            Err(InvalidDroneAction)
        }
    }
}

/// At least one motor fraction was non-finite or outside inclusive `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidDroneAction;

impl fmt::Display for InvalidDroneAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("drone motor commands must be four finite fractions between 0 and 1")
    }
}

impl Error for InvalidDroneAction {}
