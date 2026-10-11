//! Diagnostics for the complete joint action distribution.
/// Joint PPO measurements over valid recurrent transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentCombatUpdate {
    /// Completed steps for each optimizer since this learner's construction.
    pub optimizer_steps: u64,
    /// Actor and critic minibatch updates completed by this call.
    pub optimizer_updates: u64,
    /// Unique valid transitions, excluding temporal padding and repeated epochs.
    pub valid_samples: u64,
    /// Mean clipped joint actor objective, including its entropy term.
    pub actor_loss: f64,
    /// Mean centralized critic half-squared residual over valid transitions.
    pub critic_loss: f64,
    /// Bounded continuous entropy estimate plus exact categorical Shannon entropy.
    /// The continuous estimate uses fresh Gaussian samples and the configured Jacobian floor.
    pub entropy: f64,
    /// Mean behavior-to-current joint log-probability difference.
    pub approximate_kl: f64,
    /// Actor learning rate used by this call.
    pub actor_learning_rate: f64,
    /// Critic learning rate used by this call.
    pub critic_learning_rate: f64,
}
