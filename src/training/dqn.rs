//! DQN trainer API root.

use std::marker::PhantomData;

use crate::Env;

use super::config::{RunConfig, RunPaths};

/// Configuration for the DQN trainer boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct DqnConfig {
    /// Discount factor.
    pub gamma: f32,

    /// Learning rate for later optimizer setup.
    pub learning_rate: f32,

    /// Replay buffer capacity.
    pub replay_capacity: usize,

    /// Minibatch size.
    pub batch_size: usize,

    /// Target-network update interval in optimizer steps.
    pub target_update_interval: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            learning_rate: 1e-3,
            replay_capacity: 50_000,
            batch_size: 64,
            target_update_interval: 1_000,
        }
    }
}

/// DQN trainer boundary type.
#[derive(Debug, Clone)]
pub struct DqnTrainer<E: Env> {
    /// Shared run configuration.
    pub run: RunConfig,

    /// DQN-specific configuration.
    pub dqn: DqnConfig,

    /// Retains the environment type without storing an environment instance.
    _env: PhantomData<E>,
}

impl<E: Env> DqnTrainer<E> {
    /// Create a DQN trainer boundary.
    #[must_use]
    pub const fn new(run: RunConfig, dqn: DqnConfig) -> Self {
        Self {
            run,
            dqn,
            _env: PhantomData,
        }
    }

    /// Resolve the run paths this trainer will use.
    #[must_use]
    pub fn run_paths(&self) -> RunPaths {
        self.run.paths()
    }
}

/// Summary returned by future DQN training/eval slices.
#[derive(Debug, Clone, PartialEq)]
pub struct DqnReport {
    /// Run paths used for artifacts.
    pub paths: RunPaths,

    /// Number of environment steps consumed.
    pub global_steps: u64,

    /// Best deterministic eval reward if one has been selected.
    pub best_eval_reward: Option<f64>,
}
