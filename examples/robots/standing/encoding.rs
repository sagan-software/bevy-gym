//! Body-state observations and validated joint-action decoding.

use bevy::math::{Quat, Vec3};
use bevy_gym::robots::{
    DroidAction, DroidBody, DroidBodyState, DroidObservation, InvalidDroidAction,
};
use std::{error::Error, fmt};

/// Twelve root values followed by sixteen values for each of twelve other segments.
pub(crate) const FEATURES: usize = 204;

/// Encode only the droid's own physical state; no reward or critic-only information.
///
/// Root order: pelvis-frame displacement to `(0, 0.99, 0)` / 1 m, world up,
/// linear velocity / 2 m/s, angular velocity / 4 rad/s, each XYZ.
/// Other segments follow `DroidBody::ALL` excluding pelvis. Each contributes
/// relative position / 1 m, local up, local forward, relative linear velocity
/// / 2 m/s, relative angular velocity / 4 rad/s, then floor contact as 0 or 1.
/// All relative vectors use the pelvis frame. Rotation uses two unit vectors,
/// preserving equivalent quaternion signs without an angle discontinuity.
pub(crate) fn encode(observation: &DroidObservation) -> [f32; FEATURES] {
    let pelvis = observation.body(DroidBody::Pelvis);
    let inverse = pelvis.orientation().inverse();
    let displacement = inverse * (Vec3::new(0.0, 0.99, 0.0) - pelvis.position());
    let up = inverse * Vec3::Y;
    let velocity = inverse * pelvis.linear_velocity() / 2.0;
    let angular = inverse * pelvis.angular_velocity() / 4.0;
    let mut features = [0.0; FEATURES];
    let (root, segments) = features.split_at_mut(12);
    root.copy_from_slice(&[
        displacement.x,
        displacement.y,
        displacement.z,
        up.x,
        up.y,
        up.z,
        velocity.x,
        velocity.y,
        velocity.z,
        angular.x,
        angular.y,
        angular.z,
    ]);
    // Fixed chunks borrow the output buffer; no intermediate observation collection is built.
    for (identity, chunk) in DroidBody::ALL
        .into_iter()
        .skip(1)
        .zip(segments.chunks_exact_mut(16))
    {
        chunk.copy_from_slice(&segment(observation.body(identity), pelvis, inverse));
    }
    features
}

/// Express a segment relative to the same pre-step pelvis snapshot.
fn segment(body: DroidBodyState, pelvis: DroidBodyState, inverse: Quat) -> [f32; 16] {
    let displacement = inverse * (body.position() - pelvis.position());
    let rotation = inverse * body.orientation();
    let up = rotation * Vec3::Y;
    let forward = rotation * Vec3::NEG_Z;
    // These are differences of world velocities rotated into the pelvis frame,
    // not time derivatives in a rotating coordinate system.
    let velocity = inverse * (body.linear_velocity() - pelvis.linear_velocity()) / 2.0;
    let angular = inverse * (body.angular_velocity() - pelvis.angular_velocity()) / 4.0;
    [
        displacement.x,
        displacement.y,
        displacement.z,
        up.x,
        up.y,
        up.z,
        forward.x,
        forward.y,
        forward.z,
        velocity.x,
        velocity.y,
        velocity.z,
        angular.x,
        angular.y,
        angular.z,
        f32::from(body.floor_contact()),
    ]
}

/// Policy output cannot be interpreted as a complete bounded joint action.
#[derive(Debug)]
pub(crate) enum DecodeError {
    /// The output has the wrong number of actuators.
    Width(usize),
    /// At least one torque fraction is invalid.
    Fraction(InvalidDroidAction),
}
impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Width(width) => write!(formatter, "expected 26 joint values, received {width}"),
            Self::Fraction(error) => error.fmt(formatter),
        }
    }
}
impl Error for DecodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Width(_) => None,
            Self::Fraction(error) => Some(error),
        }
    }
}

/// Check output width before individual fractions; never clamp or substitute actions.
pub(crate) fn decode(values: &[f32]) -> Result<DroidAction, DecodeError> {
    let fractions: [f32; 26] = values
        .try_into()
        .map_err(|_error| DecodeError::Width(values.len()))?;
    DroidAction::try_from(fractions).map_err(DecodeError::Fraction)
}
