//! Validate the origin and normalize one fixed-range aim direction.

use bevy::math::{Dir3, InvalidDirectionError, Vec3};

/// Finite world-space origin and unit direction for a thirty-metre shot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Aim {
    /// Origin in metres, within the game's numerical query bounds.
    origin: Vec3,
    /// Normalized at construction; retained as the established direction type.
    direction: Dir3,
}

/// Malformed aim cannot consume ammunition or reach geometry queries.
#[derive(Debug, PartialEq)]
pub(crate) enum InvalidAim {
    /// Origin is non-finite or outside ±1,000 metres on any axis.
    Origin,
    /// Direction is zero or contains a non-finite component.
    Direction(InvalidDirectionError),
}

impl TryFrom<(Vec3, Vec3)> for Aim {
    type Error = InvalidAim;

    fn try_from((origin, direction): (Vec3, Vec3)) -> Result<Self, Self::Error> {
        if !valid_position(origin) {
            return Err(InvalidAim::Origin);
        }
        // Scale before taking a length, preserving finite subnormal and large inputs.
        let scale = direction.abs().max_element();
        let direction = if direction.is_finite() && scale > 0.0 {
            direction / scale
        } else {
            direction
        };
        let direction = Dir3::new(direction).map_err(InvalidAim::Direction)?;
        Ok(Self { origin, direction })
    }
}

impl Aim {
    /// Pistol range in metres, including an impact exactly at this distance.
    pub(crate) const RANGE: f32 = 30.0;

    /// Read the validated muzzle position for traces and presentation.
    pub(crate) const fn origin(self) -> Vec3 {
        self.origin
    }

    /// Read the normalized direction without exposing mutable state.
    pub(crate) const fn direction(self) -> Dir3 {
        self.direction
    }

    /// Point on the ray at a distance in metres selected by the query.
    pub(crate) fn point(self, distance: f32) -> Vec3 {
        self.origin + *self.direction * distance
    }
}

/// Bound query coordinates so local subtraction and squared distances remain finite.
pub(super) fn valid_position(position: Vec3) -> bool {
    position.is_finite() && position.abs().max_element() <= 1_000.0
}
