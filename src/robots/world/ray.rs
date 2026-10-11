//! Validated world-space ray geometry.
use super::InvalidRobotRay;
use bevy::math::Vec3;
/// Immutable ray with a finite origin, unit direction and positive range in metres.
/// Construction normalizes any finite nonzero direction, including subnormal values.
#[derive(Debug, Clone, Copy)]
pub struct RobotRay {
    /// Finite world-space origin in metres.
    pub(super) origin: Vec3,
    /// Normalized dimensionless world-space direction.
    pub(super) direction: Vec3,
    /// Positive finite inclusive range in metres.
    pub(super) metres: f32,
}
impl TryFrom<(Vec3, Vec3, f32)> for RobotRay {
    type Error = InvalidRobotRay;
    /// Validate origin, direction, distance and endpoint in that order.
    fn try_from((origin, direction, metres): (Vec3, Vec3, f32)) -> Result<Self, Self::Error> {
        // Validate each ingress value before performing geometric arithmetic.
        if !origin.is_finite() {
            return Err(InvalidRobotRay::Origin);
        }
        if !direction.is_finite() || direction == Vec3::ZERO {
            return Err(InvalidRobotRay::Direction);
        }
        if !metres.is_finite() || metres <= 0.0 {
            return Err(InvalidRobotRay::Distance);
        }
        // f64 normalization avoids overflow and underflow for every accepted f32 direction.
        let direction = direction.as_dvec3().normalize().as_vec3();
        let endpoint = (origin.as_dvec3() + direction.as_dvec3() * f64::from(metres)).as_vec3();
        if !endpoint.is_finite() {
            return Err(InvalidRobotRay::Endpoint);
        }
        Ok(Self {
            origin,
            direction,
            metres,
        })
    }
}
