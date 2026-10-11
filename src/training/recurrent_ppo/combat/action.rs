//! One complete joint policy decision.
use super::super::RecurrentAction;
use super::CombatSignal;
/// Continuous actuator and categorical requests from one recurrent action boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentCombatAction {
    /// Continuous choice and complete next recurrent state.
    pub continuous: RecurrentAction,
    /// Joint trigger and alert choice.
    pub signal: CombatSignal,
    /// Joint behavior log probability for PPO.
    pub log_probability: f32,
}
