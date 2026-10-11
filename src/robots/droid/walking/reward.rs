//! Post-step velocity tracking and physical posture measurements.

use super::super::{geometry, DroidBody};
use super::DroidWalkingObservation;
use bevy::math::Vec3;

/// Measure grounded velocity tracking without selecting or altering an action.
///
/// Reward is foot support × uprightness × height fraction × exp(-error² / scale²).
/// Support is 0 or 1; uprightness is torso up-Y clamped to [0, 1]. Height fraction
/// divides pelvis height in metres by its original 0.99 m bind height, then clamps.
/// Error² is squared X/Z velocity error in m²/s²; scale² is 0.25 m²/s².
/// All inputs belong to the same validated own-body snapshot. Borrowing that
/// snapshot and returning a scalar takes constant time and auxiliary space.
pub(super) fn measure(observation: &DroidWalkingObservation) -> f64 {
    let physical = observation.physical();
    // Unsupported motion earns no grounded-walking reward; contact never selects torques.
    if !physical.body(DroidBody::LeftFoot).floor_contact()
        && !physical.body(DroidBody::RightFoot).floor_contact()
    {
        return 0.0;
    }
    let pelvis = physical.body(DroidBody::Pelvis);
    let upright =
        f64::from((physical.body(DroidBody::Torso).orientation() * Vec3::Y).y).clamp(0.0, 1.0);
    let nominal_height_m = f64::from(geometry::segment(DroidBody::Pelvis).centre.y);
    let height = (f64::from(pelvis.position().y) / nominal_height_m).clamp(0.0, 1.0);
    // Convert both m/s components to f64 before subtraction and squared-error arithmetic.
    let requested = observation.command().velocity();
    let velocity = pelvis.linear_velocity();
    let error_x = f64::from(velocity.x) - f64::from(requested.x);
    let error_z = f64::from(velocity.z) - f64::from(requested.y);
    let squared_error = error_x.mul_add(error_x, error_z * error_z);
    upright * height * (-squared_error / 0.25).exp()
}

#[cfg(test)]
mod tests {
    use super::measure;
    use crate::{
        robots::{
            DroidBody, DroidBodyState, DroidVelocityCommand, DroidWalking, DroidWalkingObservation,
        },
        Env,
    };
    use bevy::math::{Quat, Vec2, Vec3};

    /// Build an isolated grounded measurement fixture with unit uprightness and height.
    fn grounded() -> DroidWalkingObservation {
        let mut observation = DroidWalking::default().reset(Some(42)).observation;
        body_mut(&mut observation, DroidBody::Pelvis).position.y = 0.99;
        body_mut(&mut observation, DroidBody::Pelvis).linear_velocity = Vec3::ZERO;
        body_mut(&mut observation, DroidBody::Torso).orientation = Quat::IDENTITY;
        body_mut(&mut observation, DroidBody::LeftFoot).floor_contact = true;
        observation
    }

    /// Alter only an isolated own-body measurement, never the production solver.
    fn body_mut(
        observation: &mut DroidWalkingObservation,
        identity: DroidBody,
    ) -> &mut DroidBodyState {
        observation
            .physical
            .bodies
            .get_mut(identity as usize)
            .expect("closed body identity")
    }

    /// The worked 0.5 m/s error earns exp(-1); matching velocity earns one.
    #[test]
    fn velocity_error_uses_both_horizontal_components_and_ignores_vertical_velocity() {
        let mut observation = grounded();
        assert!((measure(&observation) - 0.367_879_441_171_442_33).abs() < 1e-14);
        body_mut(&mut observation, DroidBody::Pelvis).linear_velocity = Vec3::new(0.5, -12.0, 0.0);
        assert_eq!(measure(&observation).to_bits(), 1.0_f64.to_bits());
        body_mut(&mut observation, DroidBody::Pelvis).linear_velocity = Vec3::new(0.5, 0.0, 0.5);
        assert!((measure(&observation) - 0.367_879_441_171_442_33).abs() < 1e-14);
        body_mut(&mut observation, DroidBody::Pelvis)
            .linear_velocity
            .x = f32::MAX;
        assert_eq!(measure(&observation).to_bits(), 0.0_f64.to_bits());
        let mut observation = grounded();
        observation.command =
            DroidVelocityCommand::try_from(Vec2::new(0.25, 0.5)).expect("bounded X/Z command");
        body_mut(&mut observation, DroidBody::Pelvis).linear_velocity = Vec3::new(0.25, 0.0, 0.5);
        assert_eq!(measure(&observation).to_bits(), 1.0_f64.to_bits());
    }

    /// Either foot can supply support; two feet do not double the reward.
    #[test]
    fn support_requires_at_least_one_foot_contact() {
        let mut observation = grounded();
        let grounded_reward = measure(&observation);
        body_mut(&mut observation, DroidBody::LeftFoot).floor_contact = false;
        assert_eq!(measure(&observation).to_bits(), 0.0_f64.to_bits());
        body_mut(&mut observation, DroidBody::RightFoot).floor_contact = true;
        assert_eq!(measure(&observation).to_bits(), grounded_reward.to_bits());
        body_mut(&mut observation, DroidBody::LeftFoot).floor_contact = true;
        assert_eq!(measure(&observation).to_bits(), grounded_reward.to_bits());
    }

    /// Height and up-projection clamps prevent negative or amplified posture factors.
    #[test]
    fn posture_factors_keep_the_reward_between_zero_and_one() {
        let original = grounded();
        let mut observation = original;
        body_mut(&mut observation, DroidBody::Pelvis).position.y = 0.495;
        assert!((measure(&observation) - 0.183_939_720_585_721_17).abs() < 1e-14);
        body_mut(&mut observation, DroidBody::Pelvis).position.y = 3.0;
        assert_eq!(
            measure(&observation).to_bits(),
            measure(&original).to_bits()
        );
        body_mut(&mut observation, DroidBody::Pelvis).position.y = -0.99;
        assert_eq!(measure(&observation).to_bits(), 0.0_f64.to_bits());
        let mut observation = original;
        body_mut(&mut observation, DroidBody::Torso).orientation =
            Quat::from_rotation_x(std::f32::consts::PI);
        assert_eq!(measure(&observation).to_bits(), 0.0_f64.to_bits());
    }
}
