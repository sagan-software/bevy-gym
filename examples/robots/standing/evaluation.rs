//! Independent standing episodes and fixed promotion gates.

use super::encoding;
use bevy::math::Vec3;
use bevy_gym::{
    robots::{DroidBody, DroidObservation, DroidStanding},
    training::RecurrentPpoPolicy,
    Env, TimeLimit,
};
use serde::Serialize;
use std::error::Error;

/// Twenty seconds at fifty policy actions per second.
pub(crate) const HORIZON: u16 = 1_000;
/// Ordered selection roots, separate from the collector's training partition.
pub(crate) const SELECTION_SEEDS: [u64; 5] = [0, 1, 2, 42, u64::MAX];
/// Smallest accepted torso-up projection, the f32 approximation of cos(15 degrees).
const MIN_UPRIGHT: f32 = 0.965_925_8;

/// One complete frozen-policy episode; measurements do not become optimizer input.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Score {
    /// Independent physical reset root.
    pub(crate) seed: u64,
    /// Applied actions, including a terminal action.
    pub(crate) steps: u16,
    /// Sum of unscaled environment rewards.
    pub(crate) reward: f64,
    /// Whether the horizon was reached without physical termination.
    pub(crate) survived: bool,
    /// Consecutive final actions within every balance and support boundary.
    pub(crate) stable_actions: u16,
    /// Final pelvis height in metres.
    pub(crate) final_height: f32,
    /// Final torso-up projection on world up, clamped to [-1, 1] for roundoff.
    pub(crate) final_upright: f32,
    /// Final horizontal pelvis distance from the origin in metres.
    pub(crate) final_distance: f32,
    /// Final pelvis linear speed in metres per second.
    pub(crate) final_speed: f32,
    /// Whether both feet have active floor contact at the final boundary.
    pub(crate) supported: bool,
}

/// Require the exact nonempty seed set and every episode's independent physical gates.
pub(crate) fn passes(scores: &[Score], seeds: &[u64]) -> bool {
    !seeds.is_empty()
        && scores.len() == seeds.len()
        && scores.iter().zip(seeds).all(|(score, seed)| {
            score.seed == *seed
                && score.survived
                && score.steps == HORIZON
                && (100..=HORIZON).contains(&score.stable_actions)
                && score.reward.is_finite()
                && score.reward >= 800.0
                && stable(score)
        })
}

/// Test balance without using reward to conceal missing physical measurements.
fn stable(score: &Score) -> bool {
    (0.85..=1.15).contains(&score.final_height)
        && (MIN_UPRIGHT..=1.0).contains(&score.final_upright)
        && (0.0..=0.5).contains(&score.final_distance)
        && (0.0..=0.25).contains(&score.final_speed)
        && score.supported
}

/// Final roots occupy the high-bit partition, excluding the maximum selection root.
pub(crate) fn held_out_seeds() -> [u64; 32] {
    std::array::from_fn(|index| u64::MAX - index as u64 - 1)
}

/// Evaluate without optimizer calls, sharing no recurrent memory with training or other episodes.
pub(crate) fn evaluate(
    policy: &RecurrentPpoPolicy,
    seeds: &[u64],
) -> Result<Vec<Score>, Box<dyn Error>> {
    seeds.iter().map(|&seed| episode(policy, seed)).collect()
}

/// Run one physical episode with exclusively policy-selected actuator commands.
fn episode(policy: &RecurrentPpoPolicy, seed: u64) -> Result<Score, Box<dyn Error>> {
    let mut environment = TimeLimit::new(DroidStanding::default(), usize::from(HORIZON))?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut score = measure(&observation, seed);
    loop {
        let action = policy.mean_action(&encoding::encode(&observation), &memory)?;
        let result = environment.step(encoding::decode(&action.action)?);
        observation = result.observation;
        memory = action.next_memory;
        // Final-state measurements are captured before any reset can replace them.
        let mut next = measure(&observation, seed);
        next.steps = score.steps + 1;
        next.reward = score.reward + result.reward;
        next.stable_actions = if stable(&next) {
            score.stable_actions + 1
        } else {
            0
        };
        next.survived = result.status.is_truncated();
        score = next;
        if result.status.is_done() {
            return Ok(score);
        }
    }
}

/// Read physical quantities without commanding movement or changing the task.
fn measure(observation: &DroidObservation, seed: u64) -> Score {
    let pelvis = observation.body(DroidBody::Pelvis);
    let position = pelvis.position();
    let torso_up = observation.body(DroidBody::Torso).orientation() * Vec3::Y;
    Score {
        seed,
        steps: 0,
        reward: 0.0,
        survived: false,
        stable_actions: 0,
        final_height: position.y,
        final_upright: torso_up.y.clamp(-1.0, 1.0),
        final_distance: position.x.hypot(position.z),
        final_speed: pelvis.linear_velocity().length(),
        supported: observation.body(DroidBody::LeftFoot).floor_contact()
            && observation.body(DroidBody::RightFoot).floor_contact(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A complete success fixture for testing the gate, not a trained-policy result.
    fn success() -> Score {
        Score {
            seed: 42,
            steps: HORIZON,
            reward: 800.0,
            survived: true,
            stable_actions: 100,
            final_height: 0.99,
            final_upright: 1.0,
            final_distance: 0.0,
            final_speed: 0.0,
            supported: true,
        }
    }

    /// Every geometric threshold includes its boundary and rejects the tested invalid values.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn exact_physical_boundaries() {
        for (height, upright, distance, speed) in
            [(0.85, MIN_UPRIGHT, 0.5, 0.25), (1.15, 1.0, 0.0, 0.0)]
        {
            let mut score = success();
            score.final_height = height;
            score.final_upright = upright;
            score.final_distance = distance;
            score.final_speed = speed;
            assert!(passes(&[score], &[42]));
        }
        for (field, values) in [
            (
                0,
                vec![0.849, 1.151, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            ),
            (
                1,
                vec![
                    MIN_UPRIGHT - f32::EPSILON,
                    1.001,
                    f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                ],
            ),
            (
                2,
                vec![-0.001, 0.501, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            ),
            (
                3,
                vec![-0.001, 0.251, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            ),
        ] {
            for value in values {
                let mut score = success();
                match field {
                    0 => score.final_height = value,
                    1 => score.final_upright = value,
                    2 => score.final_distance = value,
                    _ => score.final_speed = value,
                }
                assert!(!passes(&[score], &[42]));
            }
        }
    }

    /// Missing support, incomplete episodes and non-finite returns cannot pass.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn episode_and_return_gates() {
        for reward in [799.999, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut score = success();
            score.reward = reward;
            assert!(!passes(&[score], &[42]));
        }
        for steps in [0, 999, 1_001] {
            let mut score = success();
            score.steps = steps;
            assert!(!passes(&[score], &[42]));
        }
        for stable_actions in [0, 99, 1_001] {
            let mut score = success();
            score.stable_actions = stable_actions;
            assert!(!passes(&[score], &[42]));
        }
        let mut score = success();
        score.survived = false;
        assert!(!passes(&[score], &[42]));
        let mut score = success();
        score.supported = false;
        assert!(!passes(&[score], &[42]));
        assert!(!passes(&[success()], &[42, 43]));
        assert!(!passes(&[success()], &[]));
        let mut second = success();
        second.seed = 43;
        assert!(passes(&[success(), second.clone()], &[42, 43]));
        assert!(!passes(&[second, success()], &[42, 43]));
    }
}
