//! Frozen arrival gates on independent physical episodes.
use super::super::encoding as motor_encoding;
use super::{encode, TravelTask};
use bevy::math::Vec3;
use bevy_gym::training::RecurrentPpoPolicy;
use bevy_gym::{Env, TimeLimit};
use serde::Serialize;
use std::error::Error;
/// Original-physics frozen-policy outcome, independent of the rollout collector.
#[derive(Serialize)]
pub(crate) struct Outcome {
    /// Episode root.
    pub(crate) seed: u64,
    /// Applied action count.
    pub(crate) steps: usize,
    /// Original travel return.
    pub(crate) reward: f64,
    /// Final-window actions meeting every frozen gate.
    pub(crate) stable_tail_actions: usize,
    /// Final destination distance in metres.
    pub(crate) final_distance_m: f32,
    /// Final linear speed in metres per second.
    pub(crate) final_speed_m_s: f32,
    /// Final unitless forward-heading dot product.
    pub(crate) final_heading_alignment: f32,
    /// All five limits hold throughout the final hundred actions.
    pub(crate) passed: bool,
}
/// Check every final two-second action against the predeclared travel, heading and motion limits.
pub(crate) fn evaluate(policy: &RecurrentPpoPolicy, seed: u64) -> Result<Outcome, Box<dyn Error>> {
    let mut environment = TimeLimit::new(TravelTask::new(), 500)?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut reward = 0.0;
    let mut steps = 0;
    let mut stable_tail_actions = 0;
    for action_number in 1..=500 {
        let decision = policy.mean_action(&encode(observation), &memory)?;
        let action = motor_encoding::decode_action(&decision.action)?;
        memory = decision.next_memory;
        let result = environment.step(action);
        observation = result.observation;
        reward += result.reward;
        steps = action_number;
        let body = observation.body();
        let goal = observation.destination();
        let heading = goal.heading();
        let alignment =
            (body.orientation() * Vec3::NEG_Z).dot(Vec3::new(heading.x, 0.0, heading.y));
        if action_number > 400
            && stable([
                body.position().distance(goal.position()),
                body.linear_velocity().length(),
                (body.orientation() * Vec3::Y).y,
                body.angular_velocity().length(),
                alignment,
            ])
        {
            stable_tail_actions += 1;
        }
        if result.status.is_done() {
            break;
        }
    }
    let body = observation.body();
    let goal = observation.destination();
    let heading = goal.heading();
    Ok(Outcome {
        seed,
        steps,
        reward,
        stable_tail_actions,
        final_distance_m: body.position().distance(goal.position()),
        final_speed_m_s: body.linear_velocity().length(),
        final_heading_alignment: (body.orientation() * Vec3::NEG_Z)
            .dot(Vec3::new(heading.x, 0.0, heading.y)),
        passed: steps == 500 && stable_tail_actions == 100,
    })
}

/// Check distance (m), speed (m/s), up-Y, angular speed (rad/s), and heading dot product.
fn stable([distance_m, speed_m_s, up_y, angular_rad_s, heading_dot]: [f32; 5]) -> bool {
    distance_m <= 0.3
        && speed_m_s <= 0.3
        && up_y >= 0.95
        && angular_rad_s <= 0.5
        && heading_dot >= 15.0_f32.to_radians().cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Each inclusive limit passes; independently crossing any limit or supplying NaN fails.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gates_reject_each_limit_independently() {
        let boundary = [0.3, 0.3, 0.95, 0.5, 15.0_f32.to_radians().cos()];
        assert!(stable(boundary));
        let invalid = [
            0.300_001,
            0.300_001,
            0.949_999,
            0.500_001,
            boundary[4] - 0.000_001,
        ];
        for index in 0..5 {
            let mut outside = boundary;
            outside[index] = invalid[index];
            assert!(!stable(outside));
            outside[index] = f32::NAN;
            assert!(!stable(outside));
        }
    }
}
