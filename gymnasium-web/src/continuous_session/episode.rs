//! Original continuous `MountainCar` episodes with a 999-transition time limit.
use crate::observation::encode_mountain_car;
use bevy_gym::environments::{ContinuousMountainCar, ContinuousMountainCarAction};
use bevy_gym::training::RecurrentPpoError;
use bevy_gym::wrappers::time_limit::TimeLimit;
use bevy_gym::{Env, Step};

/// One physical environment and its unmodified reward accounting.
#[derive(Debug)]
pub(super) struct EpisodeState {
    /// Original dynamics and external time limit.
    environment: TimeLimit<ContinuousMountainCar>,
    /// Current exact environment observation.
    pub observation: [f32; 2],
    /// Raw environment reward sum in the current episode.
    pub reward: f64,
    /// Number of completed episodes on this lane.
    pub completed: u64,
}

impl EpisodeState {
    /// Start one deterministic reset stream.
    pub(super) fn new(seed: u64) -> Self {
        let mut environment =
            TimeLimit::new(ContinuousMountainCar::default(), 999).expect("positive episode limit");
        let observation = environment.reset(Some(seed)).observation;
        Self {
            environment,
            observation,
            reward: 0.0,
            completed: 0,
        }
    }

    /// Scale position from [-1.2, 0.6] and velocity from [-0.07, 0.07] to [-1, 1].
    pub(super) fn encoded(&self) -> [f32; 2] {
        encode(self.observation)
    }

    /// Validate the policy output before applying the original environment transition.
    pub(super) fn step(&mut self, action: &[f32]) -> Result<Step<[f32; 2]>, RecurrentPpoError> {
        let [force] = action else {
            return Err(RecurrentPpoError::DimensionMismatch {
                field: "continuous force",
                expected: 1,
                actual: action.len(),
            });
        };
        let force = ContinuousMountainCarAction::try_from(*force).map_err(|_error| {
            RecurrentPpoError::NonFiniteValue {
                field: "continuous force",
                dimension: 0,
            }
        })?;
        let result = self.environment.step(force);
        self.observation = result.observation;
        self.reward += result.reward;
        Ok(result)
    }

    /// Reset only after the terminal observation and raw return have been consumed.
    pub(super) fn reset(&mut self, seed: Option<u64>) {
        self.completed += 1;
        self.observation = self.environment.reset(seed).observation;
        self.reward = 0.0;
    }

    /// Preserve the worker's four-coordinate physical-state profile.
    pub(super) fn state(&self) -> [f64; 4] {
        let [position, velocity] = self.environment.inner().state();
        [position, velocity, 0.0, 0.0]
    }
}

/// Share the discrete example's normalization without allocating a transition vector.
pub(super) fn encode(observation: [f32; 2]) -> [f32; 2] {
    encode_mountain_car(observation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::EpisodeStatus;

    #[test]
    fn invalid_force_vectors_leave_the_environment_unchanged() {
        let mut episode = EpisodeState::new(42);
        let before = episode.state().map(f64::to_bits);
        for action in [
            vec![],
            vec![0.0, 0.0],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
        ] {
            episode.step(&action).expect_err("invalid force vector");
            assert_eq!(episode.state().map(f64::to_bits), before);
            assert_eq!(episode.reward.to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn time_limit_preserves_original_rewards_and_resets_only_after_accounting() {
        let mut episode = EpisodeState::new(42);
        for _ in 0..998 {
            assert_eq!(
                episode.step(&[0.0]).expect("finite force").status,
                EpisodeStatus::Continuing
            );
        }
        assert_eq!(
            episode.step(&[0.0]).expect("time limit").status,
            EpisodeStatus::Truncated
        );
        assert_eq!(episode.reward.to_bits(), 0.0_f64.to_bits());
        episode.reset(Some(17));
        assert_eq!(episode.completed, 1);
        assert_eq!(episode.reward.to_bits(), 0.0_f64.to_bits());
        assert_eq!(episode.observation[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(
            episode.step(&[0.0]).expect("new episode").status,
            EpisodeStatus::Continuing
        );
    }
}
