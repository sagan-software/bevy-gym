//! Serializable learning and simulation evidence.
/// Return from one completed episode.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Episode {
    /// Total transitions when this episode ended.
    pub transition: u64,
    /// Sum of unmodified environment rewards.
    pub reward: f64,
}
/// Evidence emitted after a bounded worker batch.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Snapshot {
    /// Total environment transitions.
    pub transitions: u64,
    /// Total updates per optimizer; always zero during inference.
    pub optimizer_steps: u64,
    /// Current Adam learning rate; absent during inference.
    pub learning_rate: Option<f64>,
    /// Current critic Adam rate; present only during PPO training.
    pub critic_learning_rate: Option<f64>,
    /// Most recent PPO critic loss; absent before the first update and during inference.
    pub critic_loss: Option<f64>,
    /// Number of contributing training environments, or one during inference.
    pub parallel_environments: usize,
    /// Current epsilon; absent during inference.
    pub epsilon: Option<f64>,
    /// Most recent DQN TD loss or PPO actor loss; absent before the first update and during inference.
    pub loss: Option<f64>,
    /// `CartPole` physical state in metres and radians, or `MountainCar` position
    /// and per-step velocity followed by two zeros. Pendulum uses angle in radians,
    /// angular velocity in radians per second, last torque in newton metres, and zero.
    pub state: [f64; 4],
    /// Unmodified reward sum in the current episode.
    pub episode_return: f64,
    /// Number of completed episodes.
    pub episode_count: u64,
    /// Episodes completed in this batch only; at most 256 entries.
    pub completed: Vec<Episode>,
}
