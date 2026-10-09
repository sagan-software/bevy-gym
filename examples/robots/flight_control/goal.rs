//! Validated position and heading requested from the flight pilot.

use bevy::math::{Dir2, Vec3};
use std::{error::Error, fmt};

/// A finite position inside the flight box and a horizontal unit heading.
///
/// X and Z must be strictly between -10 and 10 metres; Y between 0 and 10 metres.
/// Dir2's X and Y components map to world X and Z. This does not avoid obstacles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FlightGoal {
    /// Requested world position in metres.
    position: Vec3,
    /// Requested horizontal heading, mapping XY into world XZ.
    heading: Dir2,
}

/// A goal position is non-finite or outside the open flight box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InvalidFlightGoal;

impl TryFrom<(Vec3, Dir2)> for FlightGoal {
    type Error = InvalidFlightGoal;
    fn try_from((position, heading): (Vec3, Dir2)) -> Result<Self, Self::Error> {
        // Reject non-finite coordinates before checking the environment's open bounds.
        if !position.is_finite()
            || position.x.abs() >= 10.0
            || position.z.abs() >= 10.0
            || position.y <= 0.0
            || position.y >= 10.0
        {
            return Err(InvalidFlightGoal);
        }
        Ok(Self { position, heading })
    }
}
impl fmt::Display for InvalidFlightGoal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "flight goal requires finite coordinates inside X/Z (-10, 10) and Y (0, 10) metres",
        )
    }
}
impl Error for InvalidFlightGoal {}

impl FlightGoal {
    /// Requested world position in metres.
    pub(super) const fn position(self) -> Vec3 {
        self.position
    }

    /// Convert the horizontal heading into the world's XZ plane.
    pub(super) fn heading(self) -> Vec3 {
        Vec3::new(self.heading.x, 0.0, self.heading.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn invalid_goal_has_a_stable_diagnostic_and_no_source() {
        let error = FlightGoal::try_from((Vec3::ZERO, Dir2::X)).unwrap_err();
        assert_eq!(
            error.to_string(),
            "flight goal requires finite coordinates inside X/Z (-10, 10) and Y (0, 10) metres"
        );
        assert!(error.source().is_none());
    }
}
