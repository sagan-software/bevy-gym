//! Read-only absolute actuator work at the original 5 ms physics boundaries.
//!
//! Rapier 0.36 torque persists during a solver step; the existing operator replaces
//! it at the next boundary. <https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/>.
//! This local trapezoidal estimate does not claim exact continuous work or electrical energy.

use super::super::{geometry, physics::PHYSICS_INTERVAL, DroidAction, DroidObservation};
use bevy::math::Vec3;

/// Sum each actuator separately, so opposing actuator powers cannot cancel.
///
/// Runs in O(26) time with O(1) auxiliary storage. Inputs must already be finite.
pub(super) fn estimate(
    before: &DroidObservation,
    after: &DroidObservation,
    action: DroidAction,
) -> f64 {
    let mut joules = 0.0;
    for joint in geometry::JOINTS {
        let parent = before.body(joint.parent);
        let relative_before = before.body(joint.child).angular_velocity().as_dvec3()
            - parent.angular_velocity().as_dvec3();
        let relative_after = after.body(joint.child).angular_velocity().as_dvec3()
            - after.body(joint.parent).angular_velocity().as_dvec3();
        for (direction, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().zip(joint.axes) {
            if let Some(axis) = axis {
                // Match the applied f32 world torque exactly, then widen before power arithmetic.
                let torque = (parent.orientation()
                    * direction
                    * (action.fraction(axis.actuator) * axis.torque_limit))
                    .as_dvec3();
                let before_watts = torque.dot(relative_before);
                let after_watts = torque.dot(relative_after);
                joules += integrated_absolute_power(before_watts, after_watts);
            }
        }
    }
    joules
}

/// Nm times rad/s gives watts; average absolute endpoint watts times seconds gives joules.
fn integrated_absolute_power(before_watts: f64, after_watts: f64) -> f64 {
    before_watts.abs().midpoint(after_watts.abs()) * PHYSICS_INTERVAL.as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robots::{DroidActuator, DroidBody, DroidWalking};
    use crate::Env;
    use bevy::math::Quat;

    /// Unit rotations make the independently calculated actuator powers exact.
    fn snapshot() -> DroidObservation {
        let mut observation = *DroidWalking::default()
            .reset(Some(42))
            .observation
            .physical();
        for body in &mut observation.bodies {
            body.orientation = Quat::IDENTITY;
        }
        observation
    }

    /// The requested spine torque is 80 Nm; endpoint speeds 1 and 3 rad/s give 0.8 J.
    #[test]
    fn work_integrates_each_endpoint_and_both_torque_signs() {
        let mut before = snapshot();
        before.bodies[DroidBody::Torso as usize].angular_velocity = Vec3::X;
        let mut after = before;
        after.bodies[DroidBody::Torso as usize].angular_velocity = Vec3::X * 3.0;
        for fraction in [-1.0, 1.0] {
            let mut fractions = [0.0; 26];
            fractions[DroidActuator::SpineX as usize] = fraction;
            let action = DroidAction::try_from(fractions).expect("isolated actuator fixture");
            assert!((estimate(&before, &after, action) - 0.8).abs() < 1e-12);
        }
        let passive = DroidAction::try_from([0.0; 26]).expect("isolated passive fixture");
        assert_eq!(
            estimate(&before, &after, passive).to_bits(),
            0.0_f64.to_bits()
        );
    }

    /// Spine +80 W and neck -80 W contribute 0.8 J rather than cancelling.
    #[test]
    fn opposing_actuators_do_not_cancel_work() {
        let mut observation = snapshot();
        observation.bodies[DroidBody::Torso as usize].angular_velocity = Vec3::X;
        observation.bodies[DroidBody::Head as usize].angular_velocity = Vec3::X * -9.0;
        let mut fractions = [0.0; 26];
        fractions[DroidActuator::SpineX as usize] = 1.0;
        fractions[DroidActuator::NeckX as usize] = 1.0;
        let action = DroidAction::try_from(fractions).expect("isolated opposing-power fixture");
        assert!((estimate(&observation, &observation, action) - 0.8).abs() < 1e-12);
    }

    /// Widening before angular subtraction retains finite extreme measurements.
    #[test]
    fn finite_angular_extremes_do_not_overflow_the_estimate() {
        let mut observation = snapshot();
        observation.bodies[DroidBody::Pelvis as usize].angular_velocity = Vec3::X * f32::MAX;
        observation.bodies[DroidBody::Torso as usize].angular_velocity = Vec3::X * -f32::MAX;
        let mut fractions = [0.0; 26];
        fractions[DroidActuator::SpineX as usize] = 1.0;
        let action = DroidAction::try_from(fractions).expect("isolated finite fixture");
        let work = estimate(&observation, &observation, action);
        assert!(work.is_finite() && work > f64::from(f32::MAX) * 0.5);
    }
}
