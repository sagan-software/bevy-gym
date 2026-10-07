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
    /// Total optimizer updates; always zero during inference.
    pub optimizer_steps: u64,
    /// Current Adam learning rate; absent during inference.
    pub learning_rate: Option<f64>,
    /// Current epsilon; absent during inference.
    pub epsilon: Option<f64>,
    /// Most recent TD loss; absent before the first update and during inference.
    pub loss: Option<f64>,
    /// Physical state for rendering, in metres and radians.
    pub state: [f64; 4],
    /// Unmodified reward sum in the current episode.
    pub episode_return: f64,
    /// Number of completed episodes.
    pub episode_count: u64,
    /// Episodes completed in this batch only; at most 256 entries.
    pub completed: Vec<Episode>,
}
