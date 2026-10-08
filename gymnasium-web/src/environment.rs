//! Typed environment dispatch keeps task, action vocabulary, and state together.
use crate::{discrete_task::DiscreteTask, observation::Observation};
use bevy_gym::environments::{
    Acrobot, AcrobotAction, CartPole, CartPoleAction, MountainCar, MountainCarAction,
};
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
    /// Exact Acrobot book dynamics, capped at 500 transitions.
    Acrobot(TimeLimit<Acrobot>),
}
impl Environment {
    /// Construct only the requested simulation.
    pub(crate) fn new(task: DiscreteTask) -> Self {
        match task {
            DiscreteTask::CartPole => {
                Self::CartPole(TimeLimit::new(CartPole::default(), 500).expect("positive limit"))
            }
            DiscreteTask::MountainCar => Self::MountainCar(
                TimeLimit::new(MountainCar::default(), 200).expect("positive limit"),
            ),
            DiscreteTask::Acrobot => {
                Self::Acrobot(TimeLimit::new(Acrobot::default(), 500).expect("positive limit"))
            }
        }
    }
    /// Reset the task's original state and retain its independent random stream.
    pub(crate) fn reset(&mut self, seed: Option<u64>) -> Observation {
        match self {
            Self::CartPole(env) => Observation::CartPole(env.reset(seed).observation),
            Self::MountainCar(env) => Observation::MountainCar(env.reset(seed).observation),
            Self::Acrobot(env) => Observation::Acrobot(env.reset(seed).observation),
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
            Self::Acrobot(env) => {
                let action = AcrobotAction::try_from(index).map_err(|_error| {
                    DqnError::ActionOutOfBounds {
                        action_index: index,
                        action_dim: 3,
                    }
                })?;
                let result = env.step(action);
                Ok(Step {
                    observation: Observation::Acrobot(result.observation),
                    reward: result.reward,
                    status: result.status,
                    info: (),
                })
            }
        }
    }
    /// Preserve `CartPole`'s existing four-coordinate snapshot shape.
    /// `MountainCar` occupies the first two coordinates; the remaining two are zero.
    /// Acrobot stores both angles followed by both angular velocities.
    pub(crate) const fn state(&self) -> [f64; 4] {
        match self {
            Self::CartPole(env) => *env.inner().state(),
            Self::Acrobot(env) => *env.inner().state(),
            Self::MountainCar(env) => {
                let [position, velocity] = *env.inner().state();
                [position, velocity, 0.0, 0.0]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DiscreteTask, Environment};
    use bevy_gym::training::DqnError;

    #[test]
    fn invalid_actions_cannot_advance_any_discrete_task() {
        for (task, invalid) in [
            (DiscreteTask::CartPole, 2),
            (DiscreteTask::MountainCar, 3),
            (DiscreteTask::Acrobot, 3),
        ] {
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

    #[test]
    fn acrobot_accepts_every_torque_and_rejects_large_indices_before_stepping() {
        for action in 0..3 {
            let mut env = Environment::new(DiscreteTask::Acrobot);
            env.reset(Some(42));
            env.step(action).expect("valid torque");
            let before = env.state().map(f64::to_bits);
            let error = env.step(usize::MAX).expect_err("invalid action");
            assert!(matches!(
                error,
                DqnError::ActionOutOfBounds {
                    action_index: usize::MAX,
                    action_dim: 3,
                }
            ));
            assert_eq!(env.state().map(f64::to_bits), before);
        }
    }
}
