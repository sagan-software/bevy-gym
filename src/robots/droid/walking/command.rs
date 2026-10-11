//! Validated horizontal walking velocities in world X/Z coordinates.

use bevy::math::Vec2;
use std::{error::Error, fmt};

/// Finite world X/Z velocity with speed at most 1 m/s.
///
/// `Vec2.x` is world X velocity; `Vec2.y` is world Z velocity. Components
/// retain their f32 values and signed zeros. Validation compares component
/// squares in f64 against 1 m²/s² without normalization.
/// This command defines a lesson; it never moves a body or selects torques.
///
/// ```compile_fail
/// use bevy::math::Vec2;
/// use bevy_gym::robots::DroidVelocityCommand;
/// let unchecked = DroidVelocityCommand(Vec2::new(f32::NAN, 0.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroidVelocityCommand(
    /// Validated velocity components in metres per second, ordered X then Z.
    Vec2,
);

impl DroidVelocityCommand {
    /// Return world X/Z velocity in metres per second, without changing components.
    #[must_use]
    pub const fn velocity(self) -> Vec2 {
        self.0
    }
}

impl Default for DroidVelocityCommand {
    /// Request 0.5 m/s along world positive X.
    fn default() -> Self {
        Self(Vec2::new(0.5, 0.0))
    }
}

impl TryFrom<Vec2> for DroidVelocityCommand {
    type Error = InvalidDroidVelocityCommand;

    fn try_from(velocity: Vec2) -> Result<Self, Self::Error> {
        // Reject invalid components before comparing squared speeds.
        if !velocity.is_finite() {
            return Err(InvalidDroidVelocityCommand);
        }
        let x = f64::from(velocity.x);
        let z = f64::from(velocity.y);
        let x_squared = x * x;
        let z_squared = z * z;
        let larger = x_squared.max(z_squared);
        let smaller = x_squared.min(z_squared);
        // Each product is exact for f32 inputs. Subtract the larger square first
        // so a sum just above 1 m²/s² cannot round down to the accepted boundary.
        if larger > 1.0 || smaller > 1.0 - larger {
            Err(InvalidDroidVelocityCommand)
        } else {
            Ok(Self(velocity))
        }
    }
}

/// A velocity component is nonfinite or the speed exceeds 1 m/s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidDroidVelocityCommand;

impl fmt::Display for InvalidDroidVelocityCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(
            "droid walking commands must be finite X/Z velocities with speed at most 1 m/s",
        )
    }
}

impl Error for InvalidDroidVelocityCommand {}
