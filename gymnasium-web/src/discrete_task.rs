//! Tasks whose policies select a discrete action index.
use bevy_gym::training::DqnConfig;

/// A DQN task cannot select a continuous-action environment.
#[derive(Debug, Clone, Copy)]
pub(crate) enum DiscreteTask {
    /// Two horizontal forces.
    CartPole,
    /// Left, neutral, and right engine forces.
    MountainCar,
    /// Negative, zero, and positive elbow torques.
    Acrobot,
}

impl DiscreteTask {
    /// Observation and action dimensions in policy-record order.
    pub(crate) const fn dimensions(self) -> (usize, usize) {
        match self {
            Self::CartPole => (4, 2),
            Self::MountainCar => (2, 3),
            Self::Acrobot => (6, 3),
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
            Self::Acrobot => DqnConfig {
                hidden_sizes: vec![128, 128],
                gamma: 0.99,
                learning_rate: 0.0003,
                replay_capacity: 200_000,
                min_replay_size: 5_000,
                batch_size: 128,
                target_update_interval: 1_000,
                epsilon_start: 1.0,
                epsilon_end: 0.05,
                epsilon_decay_steps: 200_000,
                ..DqnConfig::default()
            },
        }
    }
}
