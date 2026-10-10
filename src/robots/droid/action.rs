//! Finite bounded joint torque fractions.

use super::DroidActuator;
use std::{error::Error, fmt};

/// Twenty-six finite torque fractions in inclusive `[-1, 1]`.
///
/// Order is [`DroidActuator::ALL`]. Zero requests no actuator torque.
/// Positive values torque the child along the parent's positive axis and the
/// parent oppositely. Passive joint constraints and contacts remain active.
///
/// ```compile_fail
/// use bevy_gym::robots::DroidAction;
/// let unchecked = DroidAction([f32::NAN; 26]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroidAction([f32; 26]);

impl DroidAction {
    /// Return the fraction for a physical actuator.
    ///
    /// # Panics
    /// Panics only if the internal array and closed identity enum disagree.
    /// Public construction preserves their matching lengths.
    #[must_use]
    pub fn fraction(self, actuator: DroidActuator) -> f32 {
        *self
            .0
            .get(actuator as usize)
            .expect("closed actuator index")
    }
}

impl TryFrom<[f32; 26]> for DroidAction {
    type Error = InvalidDroidAction;
    fn try_from(fractions: [f32; 26]) -> Result<Self, Self::Error> {
        // Validate the entire action before it can reach any actuator.
        if fractions
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
        {
            Ok(Self(fractions))
        } else {
            Err(InvalidDroidAction)
        }
    }
}

/// A torque fraction is non-finite or outside inclusive `[-1, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidDroidAction;

impl fmt::Display for InvalidDroidAction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("droid torque commands must be 26 finite fractions between -1 and 1")
    }
}
impl Error for InvalidDroidAction {}
