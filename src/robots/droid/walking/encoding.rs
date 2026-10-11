//! Fixed own-body and command inputs for walking actors.

use super::DroidWalkingObservation;
use crate::robots::{DroidBody, DroidBodyState};
use bevy::math::{Quat, Vec3};

/// 204 own-body values followed by three pelvis-frame command components.
const FEATURES: usize = 207;

/// Derive fixed inputs from one immutable physical decision boundary.
pub(super) fn encode(observation: &DroidWalkingObservation) -> [f32; FEATURES] {
    let physical = observation.physical();
    let pelvis = physical.body(DroidBody::Pelvis);
    let inverse = pelvis.orientation().inverse();
    // Height is a task target; horizontal position must remain translation invariant.
    let displacement = inverse * Vec3::new(0.0, 0.99 - pelvis.position().y, 0.0);
    let up = inverse * Vec3::Y;
    let velocity = inverse * pelvis.linear_velocity() / 2.0;
    let angular = inverse * pelvis.angular_velocity() / 4.0;
    let mut features = [0.0; FEATURES];
    let (root, segments) = features[..204].split_at_mut(12);
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
        chunk.copy_from_slice(&segment(physical.body(identity), pelvis, inverse));
    }
    // The horizontal world command requires three axes in the tilted pelvis frame.
    let command = observation.command().velocity();
    let velocity = inverse * Vec3::new(command.x, 0.0, command.y);
    features[204..].copy_from_slice(&velocity.to_array());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        robots::{DroidVelocityCommand, DroidWalking},
        Env,
    };
    use bevy::math::Vec2;

    /// Construct exact isolated measurement values without mutating a solver.
    fn fixture() -> DroidWalkingObservation {
        let mut observation =
            DroidWalking::new(DroidVelocityCommand::try_from(Vec2::new(0.3, -0.4)).unwrap())
                .reset(Some(42))
                .observation;
        for (index, body) in observation.physical.bodies.iter_mut().enumerate() {
            body.position = Vec3::new(index as f32 + 0.25, 1.0, 2.0);
            body.orientation = Quat::from_rotation_x(0.2);
            body.linear_velocity = Vec3::new(index as f32, 2.0, -3.0);
            body.angular_velocity = Vec3::new(-1.0, index as f32, 4.0);
            body.floor_contact = index % 2 == 0;
        }
        observation.physical.bodies[0].orientation = Quat::from_rotation_z(0.3);
        observation
    }

    /// Every physical group retains its stated coordinate order and scale.
    #[test]
    fn all_physical_groups_and_both_contact_values_match_the_contract() {
        let observation = fixture();
        let features = encode(&observation);
        let pelvis = observation.physical.body(DroidBody::Pelvis);
        let inverse = pelvis.orientation().inverse();
        assert_eq!(
            &features[6..9],
            &(inverse * pelvis.linear_velocity() / 2.0).to_array()
        );
        assert_eq!(
            &features[9..12],
            &(inverse * pelvis.angular_velocity() / 4.0).to_array()
        );
        for (identity, chunk) in DroidBody::ALL
            .into_iter()
            .skip(1)
            .zip(features[12..204].chunks_exact(16))
        {
            let body = observation.physical.body(identity);
            let rotation = inverse * body.orientation();
            assert_eq!(&chunk[3..6], &(rotation * Vec3::Y).to_array());
            assert_eq!(&chunk[6..9], &(rotation * Vec3::NEG_Z).to_array());
            assert_eq!(
                &chunk[9..12],
                &(inverse * (body.linear_velocity() - pelvis.linear_velocity()) / 2.0).to_array()
            );
            assert_eq!(
                &chunk[12..15],
                &(inverse * (body.angular_velocity() - pelvis.angular_velocity()) / 4.0).to_array()
            );
            assert_eq!(
                chunk[15].to_bits(),
                f32::from(body.floor_contact()).to_bits()
            );
        }
        let command = inverse * Vec3::new(0.3, 0.0, -0.4);
        assert_eq!(&features[204..], &command.to_array());
        assert!(features[205].abs() > 0.0);
    }

    /// Horizontal relocation cannot introduce a standing return-to-origin target.
    #[test]
    fn horizontal_translation_preserves_all_actor_inputs() {
        let mut observation = fixture();
        let before = encode(&observation);
        for body in &mut observation.physical.bodies {
            body.position += Vec3::new(8.0, 0.0, -16.0);
        }
        assert_eq!(
            encode(&observation).map(f32::to_bits),
            before.map(f32::to_bits)
        );
    }

    /// Quaternion sign is a representation choice rather than a policy observation change.
    #[test]
    fn equivalent_quaternion_signs_preserve_all_actor_inputs() {
        let mut observation = fixture();
        let before = encode(&observation);
        for body in &mut observation.physical.bodies {
            body.orientation = -body.orientation;
        }
        assert!(encode(&observation)
            .iter()
            .zip(before)
            .all(|(actual, expected)| (*actual - expected).abs() < 1e-6));
    }

    /// Invalid physical measurements remain visible to the policy validation boundary.
    #[test]
    fn nonfinite_measurements_are_never_replaced() {
        let mut observation = fixture();
        observation.physical.bodies[0].linear_velocity.x = f32::NAN;
        assert!(encode(&observation).iter().any(|value| !value.is_finite()));
    }
}
