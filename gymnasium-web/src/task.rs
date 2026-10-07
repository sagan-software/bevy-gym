//! Supported browser tasks and their learner configurations.
use bevy_gym::training::DqnConfig;

/// Closed catalog of implemented browser environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Task {
    /// Default `CartPole`-v1 with the 500-step limit.
    #[default]
    CartPole,
    /// Default `MountainCar`-v0 with the 200-step limit.
    MountainCar,
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
            _ => Err(InvalidTask),
        }
    }
}
impl std::fmt::Display for InvalidTask {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("task must be cartpole or mountain-car")
    }
}
impl std::error::Error for InvalidTask {}

impl Task {
    /// Observation and action dimensions in policy-record order.
    pub(crate) const fn dimensions(self) -> (usize, usize) {
        match self {
            Self::CartPole => (4, 2),
            Self::MountainCar => (2, 3),
        }
    }

    /// Native/browser learner settings; shaping never changes reported returns.
    pub(crate) fn config(self) -> DqnConfig {
        match self {
            Self::CartPole => DqnConfig::default(),
            Self::MountainCar => DqnConfig {
                hidden_sizes: vec![64, 64],
                gamma: 0.99,
                learning_rate: 0.001,
                replay_capacity: 100_000,
                min_replay_size: 2_000,
                batch_size: 64,
                target_update_interval: 1_000,
                epsilon_start: 1.0,
                epsilon_end: 0.05,
                epsilon_decay_steps: 100_000,
                ..DqnConfig::default()
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{InvalidTask, Task};
    #[test]
    fn worker_names_form_a_closed_task_vocabulary() {
        for name in ["", "cartpole"] {
            assert_eq!(name.parse::<Task>(), Ok(Task::CartPole));
        }
        assert_eq!("mountain-car".parse::<Task>(), Ok(Task::MountainCar));
        for name in ["MountainCar", "mountain-car ", "unknown"] {
            assert_eq!(name.parse::<Task>(), Err(InvalidTask));
        }
        assert_eq!(
            InvalidTask.to_string(),
            "task must be cartpole or mountain-car"
        );
    }
}
