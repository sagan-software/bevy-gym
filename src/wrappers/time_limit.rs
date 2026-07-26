//! External episode time-limit wrapper.

use std::error::Error;
use std::fmt;

use crate::{Env, EpisodeStatus, HasSpaces, Reset, Step};

/// Adds an external maximum episode length to an environment.
#[derive(Debug, Clone)]
pub struct TimeLimit<E> {
    /// Wrapped environment.
    inner: E,

    /// Maximum steps allowed in one episode.
    max_episode_steps: usize,

    /// Steps elapsed since the last reset.
    elapsed_steps: usize,
}

impl<E> TimeLimit<E> {
    /// Wrap `inner` with a positive maximum episode length.
    ///
    /// # Errors
    ///
    /// Returns [`TimeLimitError::ZeroLimit`] when `max_episode_steps` is zero.
    pub fn new(inner: E, max_episode_steps: usize) -> Result<Self, TimeLimitError> {
        if max_episode_steps == 0 {
            return Err(TimeLimitError::ZeroLimit);
        }

        Ok(Self {
            inner,
            max_episode_steps,
            elapsed_steps: 0,
        })
    }

    /// Borrow the wrapped environment.
    #[must_use]
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    /// Mutably borrow the wrapped environment.
    #[must_use]
    pub const fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    /// Consume the wrapper and return the environment.
    #[must_use]
    pub fn into_inner(self) -> E {
        self.inner
    }

    /// Return the configured maximum episode length.
    #[must_use]
    pub const fn max_episode_steps(&self) -> usize {
        self.max_episode_steps
    }

    /// Return steps elapsed since the last reset.
    #[must_use]
    pub const fn elapsed_steps(&self) -> usize {
        self.elapsed_steps
    }
}

impl<E: Env> Env for TimeLimit<E> {
    type Observation = E::Observation;
    type Action = E::Action;
    type Info = E::Info;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        self.elapsed_steps = 0;
        self.inner.reset(seed)
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let mut result = self.inner.step(action);
        self.elapsed_steps = self.elapsed_steps.saturating_add(1);

        if result.status == EpisodeStatus::Continuing
            && self.elapsed_steps >= self.max_episode_steps
        {
            result.status = EpisodeStatus::Truncated;
        }

        result
    }

    fn episode_extras(&self) -> std::collections::HashMap<String, f64> {
        self.inner.episode_extras()
    }
}

impl<E: HasSpaces> HasSpaces for TimeLimit<E> {
    type ActionSpace = E::ActionSpace;
    type ObservationSpace = E::ObservationSpace;

    fn action_space(&self) -> Self::ActionSpace {
        self.inner.action_space()
    }

    fn observation_space(&self) -> Self::ObservationSpace {
        self.inner.observation_space()
    }
}

/// Invalid time-limit configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeLimitError {
    /// A time limit must allow at least one step.
    ZeroLimit,
}

impl fmt::Display for TimeLimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit => formatter.write_str("time limit must be at least one step"),
        }
    }
}

impl Error for TimeLimitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct CounterEnv {
        steps: usize,
        terminate_at: Option<usize>,
    }

    impl Env for CounterEnv {
        type Observation = usize;
        type Action = ();
        type Info = ();

        fn reset(&mut self, _seed: Option<u64>) -> Reset<Self::Observation> {
            self.steps = 0;
            Reset {
                observation: 0,
                info: (),
            }
        }

        fn step(&mut self, (): Self::Action) -> Step<Self::Observation> {
            self.steps += 1;
            Step {
                observation: self.steps,
                reward: 1.0,
                status: if self.terminate_at == Some(self.steps) {
                    EpisodeStatus::Terminated
                } else {
                    EpisodeStatus::Continuing
                },
                info: (),
            }
        }
    }

    #[test]
    fn continuing_episode_truncates_at_limit_and_can_bootstrap() {
        let env = CounterEnv {
            steps: 0,
            terminate_at: None,
        };
        let mut limited = TimeLimit::new(env, 2).expect("positive time limit");

        limited.reset(Some(7));
        assert_eq!(limited.step(()).status, EpisodeStatus::Continuing);
        let final_step = limited.step(());

        assert_eq!(final_step.status, EpisodeStatus::Truncated);
        assert!((final_step.status.bootstrap_mask() - 1.0).abs() < f64::EPSILON);
        assert_eq!(limited.elapsed_steps(), 2);
    }

    #[test]
    fn natural_termination_wins_on_the_limit_step() {
        let env = CounterEnv {
            steps: 0,
            terminate_at: Some(2),
        };
        let mut limited = TimeLimit::new(env, 2).expect("positive time limit");

        limited.reset(None);
        limited.step(());
        let final_step = limited.step(());

        assert_eq!(final_step.status, EpisodeStatus::Terminated);
        assert!(final_step.status.bootstrap_mask().abs() < f64::EPSILON);
    }

    #[test]
    fn reset_clears_elapsed_steps_and_zero_limit_is_rejected() {
        let env = CounterEnv {
            steps: 0,
            terminate_at: None,
        };
        assert!(matches!(
            TimeLimit::new(env, 0),
            Err(TimeLimitError::ZeroLimit)
        ));

        let env = CounterEnv {
            steps: 0,
            terminate_at: None,
        };
        let mut limited = TimeLimit::new(env, 2).expect("positive time limit");
        limited.reset(None);
        limited.step(());
        assert_eq!(limited.elapsed_steps(), 1);
        limited.reset(None);
        assert_eq!(limited.elapsed_steps(), 0);
    }
}
