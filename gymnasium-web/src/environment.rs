//! Typed environment dispatch keeps task, action vocabulary, and state together.
use crate::{observation::Observation, Task};
use bevy_gym::environments::{CartPole, CartPoleAction, MountainCar, MountainCarAction};
use bevy_gym::training::DqnError;
use bevy_gym::wrappers::time_limit::TimeLimit;
use bevy_gym::{Env, Step};

/// Supported task state, including its external episode cap.
#[derive(Debug)]
pub(crate) enum Environment {
    /// Exact `CartPole` dynamics, capped at 500 transitions.
    CartPole(TimeLimit<CartPole>),
    /// Exact `MountainCar` dynamics, capped at 200 transitions.
    MountainCar(TimeLimit<MountainCar>),
}
impl Environment {
    /// Construct only the requested simulation.
    pub(crate) fn new(task: Task) -> Self {
        match task {
            Task::CartPole => {
                Self::CartPole(TimeLimit::new(CartPole::default(), 500).expect("positive limit"))
            }
            Task::MountainCar => Self::MountainCar(
                TimeLimit::new(MountainCar::default(), 200).expect("positive limit"),
            ),
        }
    }
    /// Reset the task's original state and retain its independent random stream.
    pub(crate) fn reset(&mut self, seed: Option<u64>) -> Observation {
        match self {
            Self::CartPole(env) => Observation::CartPole(env.reset(seed).observation),
            Self::MountainCar(env) => Observation::MountainCar(env.reset(seed).observation),
        }
    }
    /// Validate the model action before dispatching one transition.
    pub(crate) fn step(&mut self, index: usize) -> Result<Step<Observation>, DqnError> {
        match self {
            Self::CartPole(env) => {
                let action = CartPoleAction::try_from(index).map_err(|_error| {
                    DqnError::ActionOutOfBounds {
                        action_index: index,
                        action_dim: 2,
                    }
                })?;
                let result = env.step(action);
                Ok(Step {
                    observation: Observation::CartPole(result.observation),
                    reward: result.reward,
                    status: result.status,
                    info: (),
                })
            }
            Self::MountainCar(env) => {
                let action = MountainCarAction::try_from(index).map_err(|_error| {
                    DqnError::ActionOutOfBounds {
                        action_index: index,
                        action_dim: 3,
                    }
                })?;
                let result = env.step(action);
                Ok(Step {
                    observation: Observation::MountainCar(result.observation),
                    reward: result.reward,
                    status: result.status,
                    info: (),
                })
            }
        }
    }
    /// Preserve `CartPole`'s existing four-coordinate snapshot shape.
    /// `MountainCar` occupies the first two coordinates; the remaining two are zero.
    pub(crate) const fn state(&self) -> [f64; 4] {
        match self {
            Self::CartPole(env) => *env.inner().state(),
            Self::MountainCar(env) => {
                let [position, velocity] = *env.inner().state();
                [position, velocity, 0.0, 0.0]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Environment, Task};
    use bevy_gym::training::DqnError;

    #[test]
    fn invalid_actions_cannot_advance_either_task() {
        for (task, invalid) in [(Task::CartPole, 2), (Task::MountainCar, 3)] {
            let mut env = Environment::new(task);
            env.reset(Some(42));
            let before = env.state().map(f64::to_bits);
            assert!(matches!(
                env.step(invalid),
                Err(DqnError::ActionOutOfBounds { .. })
            ));
            assert_eq!(env.state().map(f64::to_bits), before);
        }
    }
}
