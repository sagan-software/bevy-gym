//! PPO trainer API root.

use std::marker::PhantomData;

use crate::Env;

use super::config::{RunConfig, RunPaths};

/// Configuration for the PPO trainer boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoConfig {
    /// Discount factor.
    pub gamma: f32,

    /// Generalized advantage estimation lambda.
    pub gae_lambda: f32,

    /// Number of rollout steps before an update.
    pub rollout_steps: usize,

    /// Minibatch size for later PPO updates.
    pub minibatch_size: usize,

    /// Optimization epochs per rollout.
    pub epochs: usize,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            gae_lambda: 0.95,
            rollout_steps: 128,
            minibatch_size: 64,
            epochs: 4,
        }
    }
}

/// PPO trainer boundary type.
#[derive(Debug, Clone)]
pub struct PpoTrainer<E: Env> {
    /// Shared run configuration.
    pub run: RunConfig,

    /// PPO-specific configuration.
    pub ppo: PpoConfig,

    /// Retains the environment type without storing an environment instance.
    _env: PhantomData<E>,
}

impl<E: Env> PpoTrainer<E> {
    /// Create a PPO trainer boundary.
    #[must_use]
    pub const fn new(run: RunConfig, ppo: PpoConfig) -> Self {
        Self {
            run,
            ppo,
            _env: PhantomData,
        }
    }

    /// Resolve the run paths this trainer will use.
    #[must_use]
    pub fn run_paths(&self) -> RunPaths {
        self.run.paths()
    }
}

/// Summary returned by future PPO training/eval slices.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoReport {
    /// Run paths used for artifacts.
    pub paths: RunPaths,

    /// Number of environment steps consumed.
    pub global_steps: u64,

    /// Best deterministic eval reward if one has been selected.
    pub best_eval_reward: Option<f64>,
}
