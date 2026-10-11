//! Frozen limb-motion checks supplement balance without changing standing physics.

use super::{encoding, evaluation};
use bevy_gym::{
    robots::{DroidBody, DroidStanding},
    training::RecurrentPpoPolicy,
    Env, TimeLimit,
};
use serde::Serialize;
use std::error::Error;

/// Allow two seconds for landing before checking every remaining action boundary.
const SETTLING_ACTIONS: u16 = 100;
/// Maximum world-frame segment angular speed in radians/second after settling.
const MAX_SPEED_RAD_S: f64 = 8.0;

/// Independent physical motion measurements; absent peaks cannot pass.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Score {
    /// Ordered reset root, never a training or ranking input from the held-out suite.
    seed: u64,
    /// Applied actions, including a terminal action.
    steps: u16,
    /// Post-settling action boundaries actually inspected.
    checked_actions: u16,
    /// Fastest of thirteen segment speeds across inspected boundaries, in radians/second.
    peak_rad_s: Option<f64>,
    /// The original time limit was reached without physical termination.
    survived: bool,
}

/// Require all 900 post-settling boundaries in every requested episode.
pub(crate) fn passes(scores: &[Score], seeds: &[u64]) -> bool {
    !seeds.is_empty()
        && scores.len() == seeds.len()
        && scores.iter().zip(seeds).all(|(score, seed)| {
            score.seed == *seed
                && score.survived
                && score.steps == evaluation::HORIZON
                && score.checked_actions == evaluation::HORIZON - SETTLING_ACTIONS
                && score
                    .peak_rad_s
                    .is_some_and(|peak| peak.is_finite() && (0.0..=MAX_SPEED_RAD_S).contains(&peak))
        })
}

/// Reloaded frozen inference retains independent memory for each physical episode.
pub(crate) fn evaluate(
    policy: &RecurrentPpoPolicy,
    seeds: &[u64],
) -> Result<Vec<Score>, Box<dyn Error>> {
    seeds.iter().map(|&seed| episode(policy, seed)).collect()
}

/// Inspect actual post-action physics; the checker never modifies an actuator request.
fn episode(policy: &RecurrentPpoPolicy, seed: u64) -> Result<Score, Box<dyn Error>> {
    let mut environment =
        TimeLimit::new(DroidStanding::default(), usize::from(evaluation::HORIZON))?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut score = Score {
        seed,
        steps: 0,
        checked_actions: 0,
        peak_rad_s: None,
        survived: false,
    };
    loop {
        let action = policy.mean_action(&encoding::encode(&observation), &memory)?;
        let result = environment.step(encoding::decode(&action.action)?);
        observation = result.observation;
        memory = action.next_memory;
        score.steps += 1;
        // If landing has finished, inspect all segments without f32 square overflow.
        if score.steps > SETTLING_ACTIONS {
            let maximum = DroidBody::ALL
                .into_iter()
                .map(|body| {
                    let [x, y, z] = observation
                        .body(body)
                        .angular_velocity()
                        .to_array()
                        .map(f64::from);
                    z.mul_add(z, y.mul_add(y, x * x)).sqrt()
                })
                .fold(0.0, f64::max);
            score.peak_rad_s = Some(score.peak_rad_s.map_or(maximum, |peak| peak.max(maximum)));
            score.checked_actions += 1;
        }
        if result.status.is_done() {
            score.survived = result.status.is_truncated();
            return Ok(score);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gate fixture is not a trained-policy qualification result.
    fn success() -> Score {
        Score {
            seed: 42,
            steps: 1_000,
            checked_actions: 900,
            peak_rad_s: Some(8.0),
            survived: true,
        }
    }

    /// Accept inclusive speeds; reject missing, nonfinite and out-of-range measurements.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn speed_boundaries() {
        for peak in [0.0, 8.0] {
            let mut score = success();
            score.peak_rad_s = Some(peak);
            assert!(passes(&[score], &[42]));
        }
        for peak in [
            None,
            Some(-0.001),
            Some(8.001),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            let mut score = success();
            score.peak_rad_s = peak;
            assert!(!passes(&[score], &[42]));
        }
    }

    /// Missing boundaries, physical termination and wrong seed sets cannot pass.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn episode_boundaries() {
        assert!(!passes(&[], &[]));
        assert!(!passes(&[success()], &[42, 43]));
        assert!(!passes(&[success()], &[43]));
        for steps in [0, 999, 1_001] {
            let mut score = success();
            score.steps = steps;
            assert!(!passes(&[score], &[42]));
        }
        for checked in [0, 899, 901] {
            let mut score = success();
            score.checked_actions = checked;
            assert!(!passes(&[score], &[42]));
        }
        let mut score = success();
        score.survived = false;
        assert!(!passes(&[score], &[42]));
        let mut second = success();
        second.seed = 43;
        assert!(passes(&[success(), second], &[42, 43]));
    }
}
