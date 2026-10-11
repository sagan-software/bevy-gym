//! One intact agent trajectory containing both kinds of policy choice.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use super::super::RecurrentPpoError;
use super::super::RecurrentPpoSequence;
use super::signal::CombatSignal;

/// Continuous trajectories and categorical choices share each action boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentCombatSequence {
    /// Existing contiguous physical trajectory; old probabilities contain the joint log density.
    pub continuous: RecurrentPpoSequence,
    /// Exactly one categorical signal for each continuous action, in temporal order.
    pub signals: Vec<CombatSignal>,
}
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
impl RecurrentCombatSequence {
    /// Validate the existing continuous contract before categorical cardinality.
    pub(super) fn validate(
        &self,
        observation_dim: usize,
        global_state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        value_count: usize,
    ) -> Result<(), RecurrentPpoError> {
        self.continuous.validate(
            observation_dim,
            global_state_dim,
            action_dim,
            hidden_size,
            value_count,
        )?;
        if self.signals.len() != self.continuous.observations.len() {
            return Err(RecurrentPpoError::invalid_sequence(
                "categorical signals must match continuous trajectory length",
            ));
        }
        Ok(())
    }
}
