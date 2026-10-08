//! Validated inference and episode memory for the bundled recovery policy.

use std::error::Error;

use bevy_gym::robots::{DroneAction, DroneObservation};
use bevy_gym::training::{RecurrentMemory, RecurrentPpoError, RecurrentPpoPolicy};

use super::{encoding, model};

/// One fixed policy and the memory of its current episode.
pub(super) struct RecoveryPilot {
    /// Qualified weights retained across episode resets.
    policy: RecurrentPpoPolicy,
    /// Recurrent state belonging only to this episode.
    memory: RecurrentMemory,
}

impl RecoveryPilot {
    /// Load the shared architecture without advancing an environment.
    pub(super) fn load(bytes: Vec<u8>) -> Result<Self, RecurrentPpoError> {
        model::load_policy(bytes).map(Self::from)
    }

    /// Clear episode memory while retaining the learned weights.
    pub(super) fn reset(&mut self) {
        self.memory = self.policy.initial_memory();
    }

    /// Validate the mean action before committing its next recurrent memory.
    pub(super) fn command(
        &mut self,
        observation: DroneObservation,
    ) -> Result<DroneAction, Box<dyn Error>> {
        let features = encoding::encode(observation);
        let prediction = self.policy.mean_action(&features, &self.memory)?;
        let action = encoding::decode_action(&prediction.action)?;
        self.memory = prediction.next_memory;
        Ok(action)
    }
}

impl From<RecurrentPpoPolicy> for RecoveryPilot {
    fn from(policy: RecurrentPpoPolicy) -> Self {
        let memory = policy.initial_memory();
        Self { policy, memory }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::robots::DroneHover;
    use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};
    use bevy_gym::Env;

    #[test]
    fn rejected_predictions_preserve_episode_memory() {
        let observation = DroneHover::disturbed().reset(Some(42)).observation;
        for (inputs, motors, low, high) in [(11, 4, 0.0, 1.0), (12, 3, 0.0, 1.0), (12, 4, 2.0, 3.0)]
        {
            let agent = RecurrentPpoAgent::new(
                inputs,
                12,
                1,
                &vec![low; motors],
                &vec![high; motors],
                model::learning_config(),
                SeedConfig::from_root(7),
            )
            .unwrap();
            let mut pilot = RecoveryPilot::from(agent.policy());
            let initial = pilot.memory.clone();
            assert!(pilot.command(observation).is_err());
            assert_eq!(pilot.memory, initial);
        }
    }

    #[test]
    fn reset_clears_memory_and_reproduces_the_first_command() {
        let mut pilot =
            RecoveryPilot::load(include_bytes!("../../../assets/robots/recovery.mpk").to_vec())
                .unwrap();
        let observation = DroneHover::disturbed().reset(Some(42)).observation;
        let initial = pilot.memory.clone();
        let first = pilot.command(observation).unwrap();
        assert_ne!(pilot.memory, initial);
        pilot.command(observation).unwrap();
        pilot.reset();
        assert_eq!(pilot.memory, initial);
        assert_eq!(pilot.command(observation).unwrap(), first);
    }
}
