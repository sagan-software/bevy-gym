//! Convert typed observations and actions at the neural-network boundary.

use std::{error::Error, fmt};

use bevy::math::Vec3;
use bevy_gym::robots::{DroneAction, DroneObservation, InvalidDroneAction};

/// Failure to convert a network output into a four-motor command.
#[derive(Debug)]
pub(crate) enum ActionDecodeError {
    /// The policy emitted a different number of motors.
    Width(usize),
    /// A motor value failed the environment's existing validation.
    Fraction(InvalidDroneAction),
}

impl fmt::Display for ActionDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Width(width) => write!(formatter, "expected four motor values, received {width}"),
            Self::Fraction(error) => error.fmt(formatter),
        }
    }
}

impl Error for ActionDecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Width(_) => None,
            Self::Fraction(error) => Some(error),
        }
    }
}

/// Encode target displacement, up, linear velocity, and angular velocity.
///
/// Each quantity uses XYZ body coordinates. Divide displacement by 2 metres,
/// velocity by 2 m/s, and angular velocity by 2 rad/s; world up has unit length.
pub(crate) fn encode(observation: DroneObservation) -> [f32; 12] {
    // Rotate world quantities into body coordinates before applying unit scales.
    let inverse = observation.orientation().inverse();
    let displacement = inverse * (Vec3::new(0.0, 2.0, 0.0) - observation.position()) / 2.0;
    let up = inverse * Vec3::Y;
    let velocity = inverse * observation.linear_velocity() / 2.0;
    let angular_velocity = inverse * observation.angular_velocity() / 2.0;
    [
        displacement.x,
        displacement.y,
        displacement.z,
        up.x,
        up.y,
        up.z,
        velocity.x,
        velocity.y,
        velocity.z,
        angular_velocity.x,
        angular_velocity.y,
        angular_velocity.z,
    ]
}

/// Reject malformed policy output before it can reach the environment.
pub(crate) fn decode_action(values: &[f32]) -> Result<DroneAction, ActionDecodeError> {
    // Check the tensor width before validating individual motor fractions.
    let fractions: [f32; 4] = values
        .try_into()
        .map_err(|_length_error| ActionDecodeError::Width(values.len()))?;
    DroneAction::try_from(fractions).map_err(ActionDecodeError::Fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn decode_errors_preserve_width_and_fraction_causes() {
        let width = decode_action(&[0.5]).unwrap_err();
        assert_eq!(width.to_string(), "expected four motor values, received 1");
        assert!(width.source().is_none());
        let fraction = decode_action(&[f32::NAN; 4]).unwrap_err();
        assert_eq!(fraction.to_string(), InvalidDroneAction.to_string());
        assert!(fraction.source().unwrap().is::<InvalidDroneAction>());
    }
}
