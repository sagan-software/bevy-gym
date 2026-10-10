//! Goal-relative motor-policy inputs shared by travel collection and inference.

use bevy::math::Vec3;
use bevy_gym::robots::{DroneDestination, DroneObservation, DroneTravelObservation};

use super::motor_encoding::encode_target;

/// Encode twelve existing motor features followed by signed heading error divided by pi.
///
/// Heading error is `atan2(cross_y, dot)` between world body-forward and the horizontal
/// destination heading. Positive values rotate around +Y; values lie in [-1, 1].
/// At a vertical body-forward singularity, atan2(0, 0) supplies zero; the up vector
/// and recurrent state still describe the tilt. Heading is task information only.
pub(crate) fn encode_goal(body: DroneObservation, destination: DroneDestination) -> [f32; 13] {
    // Reuse the exact hover feature order and scales so its RL actor can transfer.
    let [dx, dy, dz, ux, uy, uz, vx, vy, vz, wx, wy, wz] =
        encode_target(body, destination.position());
    let turn = heading_error(body, destination) / std::f32::consts::PI;
    [dx, dy, dz, ux, uy, uz, vx, vy, vz, wx, wy, wz, turn]
}

/// Encode a snapshot from the shared physical environment.
pub(crate) fn encode(observation: DroneTravelObservation) -> [f32; 13] {
    encode_goal(observation.body(), observation.destination())
}

/// Signed world-Y rotation from body forward to desired heading, in radians.
pub(crate) fn heading_error(body: DroneObservation, destination: DroneDestination) -> f32 {
    let heading = destination.heading();
    let desired = Vec3::new(heading.x, 0.0, heading.y);
    let forward = body.orientation() * Vec3::NEG_Z;
    signed_heading(forward, desired)
}

/// Resolve an exact vertical singularity before measuring horizontal rotation.
fn signed_heading(forward: Vec3, desired: Vec3) -> f32 {
    // A vertical forward vector has no horizontal heading; encode its exact singularity as zero.
    if forward.x == 0.0 && forward.z == 0.0 {
        return 0.0;
    }
    // Both atan2 operands are dimensionless; the result is in radians.
    forward.cross(desired).y.atan2(forward.dot(desired))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Signed zero at an undefined heading must not become an artificial half-turn.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn vertical_projection_is_zero_and_horizontal_turns_keep_their_sign() {
        for forward in [Vec3::Y, Vec3::NEG_Y, Vec3::new(-0.0, -1.0, -0.0)] {
            assert_eq!(
                signed_heading(forward, Vec3::X).to_bits(),
                0.0_f32.to_bits()
            );
        }
        assert!((signed_heading(Vec3::NEG_Z, Vec3::X) + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(
            (signed_heading(Vec3::NEG_Z, Vec3::NEG_X) - std::f32::consts::FRAC_PI_2).abs() < 1e-6
        );
        assert!((signed_heading(Vec3::NEG_Z, Vec3::Z).abs() - std::f32::consts::PI).abs() < 1e-6);
    }
}
