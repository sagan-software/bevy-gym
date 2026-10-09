//! Frozen imitation policy with zero recurrent memory on every action.

use super::{encoding::ActionDecodeError, FlightGoal};
use bevy::math::Vec3;
use bevy_gym::{
    robots::{DroneAction, DroneObservation},
    training::{RecurrentMemory, RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy},
};
use std::{error::Error, fmt};

/// A bundled flight controller; action calls never retain recurrent history.
pub(crate) struct FlightPilot {
    /// Frozen 13-input, 64-hidden-unit actor.
    policy: RecurrentPpoPolicy,
    /// Immutable zero state reused independently for every inference.
    zero_memory: RecurrentMemory,
}

/// A policy failure or an invalid four-motor command.
#[derive(Debug)]
pub(crate) enum FlightControlError {
    /// Network inference failed before decoding an action.
    Policy(RecurrentPpoError),
    /// The network output failed motor-command validation.
    Action(ActionDecodeError),
}

impl FlightPilot {
    /// Load the bundled, native-qualified imitation checkpoint.
    pub(crate) fn bundled() -> Result<Self, RecurrentPpoError> {
        Self::load(include_bytes!("../../../assets/robots/tracking.mpk").to_vec())
    }

    /// Validate the checkpoint against the frozen actor and critic architecture.
    fn load(bytes: Vec<u8>) -> Result<Self, RecurrentPpoError> {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![64, 64],
            ..RecurrentPpoConfig::default()
        };
        let policy =
            RecurrentPpoPolicy::load_bytes(bytes, 13, 13, 1, &[0.0; 4], &[1.0; 4], &config)?;
        let zero_memory = policy.initial_memory();
        Ok(Self {
            policy,
            zero_memory,
        })
    }
    /// Compute a motor action without changing controller state.
    pub(crate) fn action(
        &self,
        observation: DroneObservation,
        goal: FlightGoal,
    ) -> Result<DroneAction, FlightControlError> {
        // Discard next_memory: training and qualification used independent one-step actions.
        let prediction = self
            .policy
            .mean_action(&features(observation, goal), &self.zero_memory)
            .map_err(FlightControlError::Policy)?;
        super::encoding::decode_action(&prediction.action).map_err(FlightControlError::Action)
    }
}
impl fmt::Display for FlightControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(error) => error.fmt(formatter),
            Self::Action(error) => error.fmt(formatter),
        }
    }
}
impl Error for FlightControlError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Policy(error) => Some(error),
            Self::Action(error) => Some(error),
        }
    }
}

/// Encode bounded displacement and heading alongside the existing body-frame features.
fn features(observation: DroneObservation, goal: FlightGoal) -> [f32; 13] {
    let mut input = [0.0; 13];
    input[..12].copy_from_slice(&super::encoding::encode(observation));
    // Limit requested displacement to three metres before rotation and /2 metre scaling.
    let displacement = observation.orientation().inverse()
        * (goal.position() - observation.position()).clamp_length_max(3.0)
        / 2.0;
    input[..3].copy_from_slice(&displacement.to_array());
    // Signed horizontal heading error uses radians; division by pi removes the unit.
    let forward = observation.orientation() * Vec3::NEG_Z;
    let heading = goal.heading();
    input[12] = forward.cross(heading).y.atan2(forward.dot(heading)) / std::f32::consts::PI;
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Dir2;
    use bevy_gym::{robots::DroneHover, Env};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn checkpoint_errors_reject_corruption_and_the_old_architecture() {
        assert!(matches!(
            FlightPilot::load(vec![]),
            Err(RecurrentPpoError::Checkpoint(_))
        ));
        let old = include_bytes!("../../../assets/robots/recovery.mpk").to_vec();
        assert!(matches!(
            FlightPilot::load(old),
            Err(RecurrentPpoError::InvalidConfig {
                field: "checkpoint",
                reason: "parameter shape differs from the declared architecture"
            })
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn features_preserve_body_frame_units_and_bound_only_displacement() {
        let mut drone = DroneHover::disturbed();
        let observation = drone.reset(Some(42)).observation;
        let original = super::super::encoding::encode(observation);
        for distance in [0.0, 1.0, 3.0, 8.0] {
            let position = observation.position() + Vec3::X * distance;
            let goal = FlightGoal::try_from((position, Dir2::NEG_Y)).unwrap();
            let input = features(observation, goal);
            let expected = observation.orientation().inverse() * Vec3::X * distance.min(3.0) / 2.0;
            let actual = Vec3::from_slice(&input[..3]);
            assert!(actual.abs_diff_eq(expected, 0.000_001));
            assert_eq!(&input[3..12], &original[3..12]);
            let forward = observation.orientation() * Vec3::NEG_Z;
            let error = forward.cross(Vec3::NEG_Z).y.atan2(forward.dot(Vec3::NEG_Z));
            assert_eq!(
                input[12].to_bits(),
                (error / std::f32::consts::PI).to_bits()
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn inference_errors_preserve_their_sources() {
        let mut pilot = FlightPilot::bundled().unwrap();
        // Break a private invariant to exercise the network-failure boundary directly.
        pilot.zero_memory.cell.clear();
        let observation = DroneHover::default().reset(Some(42)).observation;
        let goal = FlightGoal::try_from((Vec3::new(0.0, 2.0, 0.0), Dir2::X)).unwrap();
        let error = pilot.action(observation, goal).unwrap_err();
        assert!(error.source().unwrap().is::<RecurrentPpoError>());
        assert_eq!(error.to_string(), error.source().unwrap().to_string());
        let error = FlightControlError::Action(ActionDecodeError::Width(3));
        assert!(error.source().unwrap().is::<ActionDecodeError>());
        assert_eq!(error.to_string(), "expected four motor values, received 3");
    }
}
