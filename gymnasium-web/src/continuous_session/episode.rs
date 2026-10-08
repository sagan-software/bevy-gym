//! Original continuous-action environments with external episode limits.
use crate::{continuous_task::ContinuousTask, observation::Observation};
use bevy_gym::environments::{
    ContinuousMountainCar, ContinuousMountainCarAction, Pendulum, PendulumAction,
};
use bevy_gym::training::RecurrentPpoError;
use bevy_gym::wrappers::time_limit::TimeLimit;
use bevy_gym::{Env, Step};

/// One physical environment and its unmodified reward accounting.
#[derive(Debug)]
pub(super) struct EpisodeState {
    /// Original dynamics and the matching time limit.
    environment: Environment,
    /// Raw environment reward sum in the current episode.
    pub reward: f64,
    /// Number of completed episodes on this lane.
    pub completed: u64,
}

/// Each variant retains its original action, state precision, and episode cap.
#[derive(Debug)]
enum Environment {
    /// Engine force and the 999-transition cap.
    MountainCar(TimeLimit<ContinuousMountainCar>),
    /// Torque and the 200-transition cap.
    Pendulum(TimeLimit<Pendulum>),
}

impl EpisodeState {
    /// Start one deterministic reset stream for the requested task.
    pub(super) fn new(task: ContinuousTask, seed: u64) -> Self {
        let environment = match task {
            ContinuousTask::MountainCar => Environment::MountainCar(
                TimeLimit::new(ContinuousMountainCar::default(), 999)
                    .expect("positive episode limit"),
            ),
            ContinuousTask::Pendulum => Environment::Pendulum(
                TimeLimit::new(Pendulum::default(), 200).expect("positive episode limit"),
            ),
        };
        let mut episode = Self {
            environment,
            reward: 0.0,
            completed: 0,
        };
        episode.reset_environment(Some(seed));
        episode
    }

    /// Derive observations from the authoritative physical state without a second cache.
    pub(super) fn observation(&self) -> Observation {
        match &self.environment {
            Environment::MountainCar(env) => {
                Observation::MountainCar(env.inner().state().map(|value| value as f32))
            }
            Environment::Pendulum(env) => {
                let [angle, velocity] = env.inner().state();
                Observation::Pendulum([angle.cos() as f32, angle.sin() as f32, velocity as f32])
            }
        }
    }

    /// Normalize only the policy input; physical state and rewards retain source units.
    pub(super) fn encoded(&self) -> Observation {
        self.observation().encoded()
    }

    /// Validate the policy output before applying an original environment transition.
    pub(super) fn step(&mut self, action: &[f32]) -> Result<Step<Observation>, RecurrentPpoError> {
        let field = match &self.environment {
            Environment::MountainCar(_) => "continuous force",
            Environment::Pendulum(_) => "pendulum torque",
        };
        let [raw] = action else {
            return Err(RecurrentPpoError::DimensionMismatch {
                field,
                expected: 1,
                actual: action.len(),
            });
        };
        // Reject malformed policy outputs before either environment can change state.
        let (observation, reward, status) = match &mut self.environment {
            Environment::MountainCar(env) => {
                let action = ContinuousMountainCarAction::try_from(*raw).map_err(|_error| {
                    RecurrentPpoError::NonFiniteValue {
                        field,
                        dimension: 0,
                    }
                })?;
                let result = env.step(action);
                (
                    Observation::MountainCar(result.observation),
                    result.reward,
                    result.status,
                )
            }
            Environment::Pendulum(env) => {
                let action = PendulumAction::try_from(*raw).map_err(|_error| {
                    RecurrentPpoError::NonFiniteValue {
                        field,
                        dimension: 0,
                    }
                })?;
                let result = env.step(action);
                (
                    Observation::Pendulum(result.observation),
                    result.reward,
                    result.status,
                )
            }
        };
        self.reward += reward;
        Ok(Step {
            observation,
            reward,
            status,
            info: (),
        })
    }

    /// Reset only after the terminal observation and raw return have been consumed.
    pub(super) fn reset(&mut self, seed: Option<u64>) {
        self.completed += 1;
        self.reset_environment(seed);
        self.reward = 0.0;
    }

    /// Reset the physics without counting construction as a completed episode.
    fn reset_environment(&mut self, seed: Option<u64>) {
        match &mut self.environment {
            Environment::MountainCar(env) => {
                env.reset(seed);
            }
            Environment::Pendulum(env) => {
                env.reset(seed);
            }
        }
    }

    /// Project source state into the worker's four-coordinate rendering profile.
    pub(super) fn state(&self) -> [f64; 4] {
        match &self.environment {
            Environment::MountainCar(env) => {
                let [position, velocity] = env.inner().state();
                [position, velocity, 0.0, 0.0]
            }
            Environment::Pendulum(env) => {
                let [angle, velocity] = env.inner().state();
                [
                    angle,
                    velocity,
                    f64::from(env.inner().last_torque().unwrap_or(0.0)),
                    0.0,
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::EpisodeStatus;

    #[test]
    fn invalid_action_vectors_leave_both_environments_unchanged() {
        for (task, field) in [
            (ContinuousTask::MountainCar, "continuous force"),
            (ContinuousTask::Pendulum, "pendulum torque"),
        ] {
            let mut episode = EpisodeState::new(task, 42);
            let before = episode.state().map(f64::to_bits);
            for action in [vec![], vec![0.0, 0.0], vec![f32::NAN, 0.0]] {
                assert!(matches!(
                    episode.step(&action),
                    Err(RecurrentPpoError::DimensionMismatch {
                        field: actual_field, expected: 1, actual,
                    }) if actual_field == field && actual == action.len()
                ));
                assert_eq!(episode.state().map(f64::to_bits), before);
                assert_eq!(episode.reward.to_bits(), 0.0_f64.to_bits());
            }
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert!(matches!(
                    episode.step(&[value]),
                    Err(RecurrentPpoError::NonFiniteValue {
                        field: actual_field, dimension: 0,
                    }) if actual_field == field
                ));
                assert_eq!(episode.state().map(f64::to_bits), before);
                assert_eq!(episode.reward.to_bits(), 0.0_f64.to_bits());
            }
        }
    }

    #[test]
    fn time_limit_preserves_original_rewards_and_resets_only_after_accounting() {
        let mut episode = EpisodeState::new(ContinuousTask::MountainCar, 42);
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
        assert_eq!(
            episode.observation().as_ref()[1].to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            episode.step(&[0.0]).expect("new episode").status,
            EpisodeStatus::Continuing
        );
    }
}
