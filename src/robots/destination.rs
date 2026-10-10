//! Validated immutable position and horizontal heading for a flight episode.

use std::fmt;

use bevy::math::{Dir2, Vec2, Vec3};

/// An interior flight position in metres and a normalized world-XZ heading.
///
/// Construct with `DroneDestination::try_from((position, heading))`. Position must
/// be finite, with X/Z strictly between -10 and 10 metres and Y strictly between
/// 0 and 10 metres. Heading must be finite and nonzero. Its X/Y components map
/// to world X/Z; normalization preserves direction, not input magnitude.
/// Position is validated before heading. No coordinates are clamped.
///
/// ```compile_fail
/// use bevy::math::{Vec2, Vec3};
/// use bevy_gym::robots::DroneDestination;
/// let mut goal = DroneDestination::try_from((Vec3::Y * 2.0, Vec2::X)).unwrap();
/// goal.position = Vec3::splat(f32::NAN);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneDestination {
    /// World-space position in metres.
    position: Vec3,
    /// Unit heading whose components correspond to world X and Z.
    heading: Dir2,
}

/// The first rejected destination field, checked before constructing an environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidDroneDestination {
    /// Position is non-finite or outside the open flight-region interior.
    Position,
    /// Heading is non-finite or zero.
    Heading,
}

impl TryFrom<(Vec3, Vec2)> for DroneDestination {
    type Error = InvalidDroneDestination;

    fn try_from((position, heading): (Vec3, Vec2)) -> Result<Self, Self::Error> {
        // Validate position first so malformed tasks never reach the physics boundary.
        if !position.is_finite()
            || position.x.abs() >= 10.0
            || position.z.abs() >= 10.0
            || position.y <= 0.0
            || position.y >= 10.0
        {
            return Err(InvalidDroneDestination::Position);
        }
        // Scale before normalization to preserve finite tiny and huge directions.
        let scale = heading.abs().max_element();
        if !heading.is_finite() || scale == 0.0 {
            return Err(InvalidDroneDestination::Heading);
        }
        let heading = Dir2::new(heading / scale)
            .map_err(|_normalization_error| InvalidDroneDestination::Heading)?;
        Ok(Self { position, heading })
    }
}

impl DroneDestination {
    /// Read the world-space target position in metres.
    #[must_use]
    pub const fn position(self) -> Vec3 {
        self.position
    }

    /// Read the normalized heading in world-XZ coordinates.
    #[must_use]
    pub const fn heading(self) -> Dir2 {
        self.heading
    }
}

impl fmt::Display for InvalidDroneDestination {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Position => {
                "Drone destination must be finite, with X/Z in (-10, 10) and Y in (0, 10) metres."
            }
            Self::Heading => "Drone destination heading must be finite and nonzero.",
        })
    }
}

impl std::error::Error for InvalidDroneDestination {}
