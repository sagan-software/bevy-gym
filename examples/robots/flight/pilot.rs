//! Validated inference and episode memory for the bundled recovery policy.

use std::error::Error;

use bevy_gym::robots::{DroneAction, DroneObservation};
use bevy_gym::training::{RecurrentMemory, RecurrentPpoError, RecurrentPpoPolicy};

use super::{damage_encoding, encoding, model};

/// Maximum supported checkpoint size before decoding, in bytes.
pub(super) const MAX_CHECKPOINT_BYTES: usize = 1_048_576;

/// Known observation recipes accepted by the file loader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_arch = "wasm32"), derive(clap::ValueEnum))]
pub(super) enum CheckpointKind {
    /// Twelve body-frame motion features.
    Recovery,
    /// Twelve motion features followed by four actuator-health indicators.
    MotorFailure,
}

/// Where the user obtained these weights, independent of their measured performance.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    /// The repository's qualified recovery checkpoint.
    Bundled,
    /// A checkpoint supplied by the browser training panel.
    Browser,
    /// A local file with an explicitly selected observation recipe.
    File(CheckpointKind),
}

/// One fixed policy and the memory of its current episode.
pub(super) struct RecoveryPilot {
    /// Frozen weights retained across episode resets.
    policy: RecurrentPpoPolicy,
    /// Recurrent state belonging only to this episode.
    memory: RecurrentMemory,
    /// Source shown in the viewer without claiming that new weights are qualified.
    origin: Origin,
}

impl RecoveryPilot {
    /// Load the shared architecture without advancing an environment.
    pub(super) fn load(bytes: Vec<u8>) -> Result<Self, RecurrentPpoError> {
        model::load_policy(bytes).map(Self::from)
    }

    /// Identify the repository's bundled model for its selection highlight.
    pub(super) fn is_bundled(&self) -> bool {
        self.origin == Origin::Bundled
    }

    /// Load the known repository checkpoint and identify its measured provenance.
    pub(super) fn bundled() -> Result<Self, RecurrentPpoError> {
        let mut pilot = Self::load(include_bytes!("../../../assets/robots/recovery.mpk").to_vec())?;
        pilot.origin = Origin::Bundled;
        Ok(pilot)
    }

    /// Validate the selected architecture before retaining a local file's weights.
    pub(super) fn load_file(bytes: Vec<u8>, kind: CheckpointKind) -> Result<Self, Box<dyn Error>> {
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err("Checkpoint exceeds 1 MiB.".into());
        }
        let policy = match kind {
            CheckpointKind::Recovery => model::load_policy(bytes)?,
            CheckpointKind::MotorFailure => RecurrentPpoPolicy::load_bytes(
                bytes,
                16,
                16,
                1,
                &[0.0; 4],
                &[1.0; 4],
                &model::learning_config(),
            )?,
        };
        let mut pilot = Self::from(policy);
        pilot.origin = Origin::File(kind);
        Ok(pilot)
    }

    /// Read the recipe implied by the validated source without exposing mutable state.
    pub(super) const fn kind(&self) -> CheckpointKind {
        match self.origin {
            Origin::File(kind) => kind,
            Origin::Bundled | Origin::Browser => CheckpointKind::Recovery,
        }
    }

    /// Name provenance without claiming that uploaded weights passed qualification.
    pub(super) const fn label(&self) -> &'static str {
        match self.origin {
            Origin::Bundled => "Bundled policy",
            Origin::Browser => "Browser checkpoint",
            Origin::File(CheckpointKind::Recovery) => "Recovery checkpoint",
            Origin::File(CheckpointKind::MotorFailure) => "Motor-failure checkpoint",
        }
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
        let prediction = match self.kind() {
            CheckpointKind::Recovery => self.policy.mean_action(&features, &self.memory)?,
            CheckpointKind::MotorFailure => self.policy.mean_action(
                &damage_encoding::with_motor_health(observation, features),
                &self.memory,
            )?,
        };
        let action = encoding::decode_action(&prediction.action)?;
        self.memory = prediction.next_memory;
        Ok(action)
    }
}

impl From<RecurrentPpoPolicy> for RecoveryPilot {
    fn from(policy: RecurrentPpoPolicy) -> Self {
        let memory = policy.initial_memory();
        Self {
            policy,
            memory,
            origin: Origin::Browser,
        }
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

#[cfg(test)]
mod file_tests {
    use super::*;
    use bevy_gym::{
        robots::{DroneHover, DroneMotor},
        training::{RecurrentPpoAgent, SeedConfig},
        Env,
    };

    #[test]
    fn motor_failure_files_use_health_inputs_and_reset_episode_memory() {
        let policy = RecurrentPpoAgent::new(
            16,
            16,
            1,
            &[0.0; 4],
            &[1.0; 4],
            model::learning_config(),
            SeedConfig::from_root(7),
        )
        .unwrap()
        .policy();
        let mut pilot =
            RecoveryPilot::load_file(policy.to_bytes().unwrap(), CheckpointKind::MotorFailure)
                .unwrap();
        let mut drone = DroneHover::default();
        drone.reset(Some(42));
        drone.fail_motor(DroneMotor::FrontLeft).unwrap();
        let observation = drone.observation();
        let mut features = Vec::from(encoding::encode(observation));
        features.extend([0.0, 1.0, 1.0, 1.0]);
        let expected = policy
            .mean_action(&features, &policy.initial_memory())
            .unwrap();
        let first = pilot.command(observation).unwrap();
        assert_eq!(first, encoding::decode_action(&expected.action).unwrap());
        assert_eq!(pilot.label(), "Motor-failure checkpoint");
        assert_eq!(pilot.kind(), CheckpointKind::MotorFailure);
        assert!(!pilot.is_bundled());
        pilot.command(observation).unwrap();
        pilot.reset();
        assert_eq!(pilot.command(observation).unwrap(), first);
    }

    #[test]
    fn motor_failure_prediction_errors_preserve_memory() {
        let mut pilot = RecoveryPilot::bundled().unwrap();
        // Exercise the defensive inference boundary with an inconsistent private fixture.
        pilot.origin = Origin::File(CheckpointKind::MotorFailure);
        let initial = pilot.memory.clone();
        let observation = DroneHover::default().reset(Some(42)).observation;
        assert!(pilot.command(observation).is_err());
        assert_eq!(pilot.memory, initial);
    }

    #[test]
    fn file_kind_rejects_the_other_architecture_and_preserves_existing_recovery() {
        let healthy = include_bytes!("../../../assets/robots/recovery.mpk").to_vec();
        assert!(RecoveryPilot::load_file(healthy.clone(), CheckpointKind::MotorFailure).is_err());
        let mut loaded =
            RecoveryPilot::load_file(healthy.clone(), CheckpointKind::Recovery).unwrap();
        let mut existing = RecoveryPilot::load(healthy).unwrap();
        let observation = DroneHover::disturbed().reset(Some(42)).observation;
        assert_eq!(
            loaded.command(observation).unwrap(),
            existing.command(observation).unwrap()
        );
        assert_eq!(loaded.label(), "Recovery checkpoint");
        let damage = RecurrentPpoAgent::new(
            16,
            16,
            1,
            &[0.0; 4],
            &[1.0; 4],
            model::learning_config(),
            SeedConfig::from_root(7),
        )
        .unwrap()
        .policy()
        .to_bytes()
        .unwrap();
        assert!(RecoveryPilot::load_file(damage, CheckpointKind::Recovery).is_err());
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;

    #[test]
    fn size_validation_precedes_decoding_and_includes_the_exact_limit() {
        for kind in [CheckpointKind::Recovery, CheckpointKind::MotorFailure] {
            let too_large = RecoveryPilot::load_file(vec![0; MAX_CHECKPOINT_BYTES + 1], kind)
                .err()
                .unwrap();
            assert_eq!(too_large.to_string(), "Checkpoint exceeds 1 MiB.");
            for length in [0, MAX_CHECKPOINT_BYTES] {
                let invalid_model = RecoveryPilot::load_file(vec![0; length], kind)
                    .err()
                    .unwrap();
                assert_ne!(invalid_model.to_string(), "Checkpoint exceeds 1 MiB.");
            }
        }
    }
}
