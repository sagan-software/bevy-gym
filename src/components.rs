use bevy::prelude::*;

use crate::Env;

/// The simulation state for one environment instance.
///
/// One entity per environment instance is spawned by `BevyGymPlugin`.
///
/// # Bevy ECS parallelism
///
/// Each entity's components are independent, so `Query<&mut EnvComponent<E>>`
/// can step many environments in parallel.
#[derive(Component, Debug)]
pub struct EnvComponent<E: Env + Send + Sync + 'static> {
    /// The wrapped reinforcement-learning environment.
    pub env: E,
}

impl<E: Env + Send + Sync + 'static> EnvComponent<E> {
    /// Wrap an environment as a Bevy component.
    pub const fn new(env: E) -> Self {
        Self { env }
    }
}

/// The action queued for this environment's next `step()` call.
#[derive(Component, Debug)]
pub(crate) struct QueuedAction<E: Env + Send + Sync + 'static> {
    /// The action to consume on the next simulation step.
    pub(crate) action: Option<E::Action>,
}

impl<E: Env + Send + Sync + 'static> Default for QueuedAction<E> {
    fn default() -> Self {
        Self { action: None }
    }
}

/// The most recent observation from this environment.
///
/// Updated after steps and resets. `ActionRequest<E>` carries cloned values for
/// the beginner policy path; this component remains useful for advanced ECS
/// systems that need direct state access.
#[derive(Component, Debug)]
pub struct CurrentObservation<E: Env + Send + Sync + 'static> {
    /// Latest observation emitted by the environment.
    pub observation: E::Observation,

    /// Latest auxiliary info emitted by the environment.
    pub info: E::Info,
}

/// Per-episode and overall statistics for one environment instance.
///
/// Useful for logging, debugging, and external trainer integration.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct EnvStats {
    /// Total reward accumulated in the current episode.
    pub episode_reward: f64,

    /// Number of steps taken in the current episode.
    pub episode_steps: usize,

    /// Total number of completed episodes.
    pub total_episodes: usize,

    /// Total number of steps taken across all episodes.
    pub total_steps: usize,
}

impl EnvStats {
    /// Accumulate reward and step counters for one environment step.
    pub(crate) fn record_step(&mut self, reward: f64) {
        self.episode_reward += reward;
        self.episode_steps += 1;
        self.total_steps += 1;
    }

    /// Finish the current episode and reset per-episode counters.
    pub(crate) const fn record_episode_end(&mut self) {
        self.total_episodes += 1;
        self.episode_reward = 0.0;
        self.episode_steps = 0;
    }
}

/// Stable index identifying this environment instance within the pool.
///
/// Ranges from `0..num_envs`. Used to correlate messages and queries
/// with specific environment entities.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvId(pub usize);
