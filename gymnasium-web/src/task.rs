//! Supported browser tasks and their learner configurations.

/// Closed catalog of implemented browser environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// Default `CartPole`-v1 with the 500-step limit.
    #[default]
    CartPole,
    /// Default `MountainCar`-v0 with the 200-step limit.
    MountainCar,
    /// Continuous engine force with the 999-step limit.
    MountainCarContinuous,
    /// Bounded Pendulum torque with the 200-step limit.
    Pendulum,
    /// Three elbow torques with the 500-step limit.
    Acrobot,
}

/// A browser task name outside the supported catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidTask;

impl std::str::FromStr for Task {
    type Err = InvalidTask;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "" | "cartpole" => Ok(Self::CartPole),
            "mountain-car" => Ok(Self::MountainCar),
            "mountain-car-continuous" => Ok(Self::MountainCarContinuous),
            "pendulum" => Ok(Self::Pendulum),
            "acrobot" => Ok(Self::Acrobot),
            _ => Err(InvalidTask),
        }
    }
}
impl std::fmt::Display for InvalidTask {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(
            "task must be cartpole, mountain-car, mountain-car-continuous, pendulum, or acrobot",
        )
    }
}
impl std::error::Error for InvalidTask {}

#[cfg(test)]
mod tests {
    use super::{InvalidTask, Task};
    #[test]
    fn worker_names_form_a_closed_task_vocabulary() {
        for name in ["", "cartpole"] {
            assert_eq!(name.parse::<Task>(), Ok(Task::CartPole));
        }
        assert_eq!("mountain-car".parse::<Task>(), Ok(Task::MountainCar));
        assert_eq!("acrobot".parse::<Task>(), Ok(Task::Acrobot));
        assert_eq!("pendulum".parse::<Task>(), Ok(Task::Pendulum));
        assert_eq!(
            "mountain-car-continuous".parse::<Task>(),
            Ok(Task::MountainCarContinuous)
        );
        for name in [
            "MountainCar",
            "mountain-car ",
            "unknown",
            "Acrobot",
            "acrobot ",
        ] {
            assert_eq!(name.parse::<Task>(), Err(InvalidTask));
        }
        assert_eq!(
            InvalidTask.to_string(),
            "task must be cartpole, mountain-car, mountain-car-continuous, pendulum, or acrobot"
        );
    }
}
