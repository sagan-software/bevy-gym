//! Frozen recurrent inference with no route to manual or fallback motor commands.

use bevy_gym::robots::{DroneAction, DroneHover, DroneObservation};
use bevy_gym::training::{RecurrentMemory, RecurrentPpoPolicy};
use bevy_gym::{Env, EpisodeStatus, TimeLimit};

use super::{encoding, model};

/// Either usable frozen weights and episode memory, or a permanent inference stop.
enum Inference {
    /// Retain recurrent memory across actions, separately from frozen weights.
    Ready {
        /// Weights are loaded once and never updated during playback.
        policy: Box<RecurrentPpoPolicy>,
        /// Hidden state belonging only to this episode.
        memory: RecurrentMemory,
    },
    /// Inference or action validation failed before physics advanced.
    Failed(String),
}

/// Private physics and policy state shared by the two standalone scenes.
pub(crate) struct Session {
    /// The same ten-second environment used by curriculum evaluation.
    environment: TimeLimit<DroneHover>,
    /// Latest authoritative pose and velocities, including the initial reset.
    observation: DroneObservation,
    /// Environment completion; inference failures remain a separate diagnostic.
    status: EpisodeStatus,
    /// The only source of actuator commands.
    inference: Inference,
    /// Last applied motor values for read-only visualization.
    last_action: Option<DroneAction>,
}

impl Session {
    /// Validate weights before creating a playable environment.
    pub(crate) fn load(
        bytes: Vec<u8>,
        factory: fn() -> DroneHover,
        seed: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let policy = model::load_policy(bytes)?;
        let memory = policy.initial_memory();
        let mut environment = TimeLimit::new(factory(), 500)?;
        let observation = environment.reset(Some(seed)).observation;
        Ok(Self {
            environment,
            observation,
            status: EpisodeStatus::Continuing,
            inference: Inference::Ready {
                policy: Box::new(policy),
                memory,
            },
            last_action: None,
        })
    }

    /// Infer and validate one action before advancing physics; failures are sticky.
    pub(crate) fn step(&mut self) {
        if self.status.is_done() {
            return;
        }
        let Inference::Ready { policy, memory } = &mut self.inference else {
            return;
        };
        // Inference sees only the observation, never reward or critic features.
        let inferred = match policy.mean_action(&encoding::encode(self.observation), memory) {
            Ok(inferred) => inferred,
            Err(error) => {
                self.inference = Inference::Failed(error.to_string());
                return;
            }
        };
        // Invalid actuator output cannot reach the environment or update memory.
        let action = match encoding::decode_action(&inferred.action) {
            Ok(action) => action,
            Err(error) => {
                self.inference = Inference::Failed(error.to_string());
                return;
            }
        };
        *memory = inferred.next_memory;
        let result = self.environment.step(action);
        self.observation = result.observation;
        self.status = result.status;
        self.last_action = Some(action);
    }

    /// Reset physics and recurrent memory; a failed policy remains unusable.
    pub(crate) fn reset(&mut self, seed: u64) {
        self.observation = self.environment.reset(Some(seed)).observation;
        self.status = EpisodeStatus::Continuing;
        self.last_action = None;
        if let Inference::Ready { policy, memory } = &mut self.inference {
            *memory = policy.initial_memory();
        }
    }

    /// Read the authoritative observation without exposing physics mutation.
    pub(crate) const fn observation(&self) -> DroneObservation {
        self.observation
    }

    /// Read the environment's completion status.
    pub(crate) const fn status(&self) -> EpisodeStatus {
        self.status
    }

    /// Read the actual number of applied actions.
    pub(crate) const fn steps(&self) -> usize {
        self.environment.elapsed_steps()
    }

    /// Read motor values only after a policy action has reached physics.
    pub(crate) const fn last_action(&self) -> Option<DroneAction> {
        self.last_action
    }

    /// Expose the reason playback stopped so the scene can display it.
    pub(crate) fn error(&self) -> Option<&str> {
        match &self.inference {
            Inference::Ready { .. } => None,
            Inference::Failed(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corrupt memory must stop before a command reaches physics, even after reset.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn inference_failure_is_visible_and_sticky() {
        let mut session = Session::load(
            include_bytes!("../../../docs/progress/drone-curriculum.mpk").to_vec(),
            DroneHover::default,
            42,
        )
        .expect("valid checkpoint");
        if let Inference::Ready { memory, .. } = &mut session.inference {
            memory.hidden.fill(f32::NAN);
        }
        let initial = session.observation();
        session.step();
        assert!(session.error().is_some());
        assert_eq!(session.observation(), initial);
        assert_eq!(session.steps(), 0);
        assert!(session.last_action().is_none());
        session.step();
        session.reset(42);
        session.step();
        assert_eq!(session.steps(), 0);
        assert!(session.error().is_some());
        session.status = EpisodeStatus::Terminated;
        session.step();
        assert_eq!(session.steps(), 0);
    }

    /// A policy with incompatible actuator bounds cannot advance physical state.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn invalid_motor_output_stops_before_physics() {
        let bytes = include_bytes!("../../../docs/progress/drone-curriculum.mpk");
        let mut session =
            Session::load(bytes.to_vec(), DroneHover::default, 42).expect("valid checkpoint");
        let incompatible = RecurrentPpoPolicy::load_bytes(
            bytes.to_vec(),
            12,
            12,
            1,
            &[2.0; 4],
            &[3.0; 4],
            &model::learning_config(),
        )
        .expect("network with incompatible actuator scaling");
        session.inference = Inference::Ready {
            memory: incompatible.initial_memory(),
            policy: Box::new(incompatible),
        };
        let initial = session.observation();
        session.step();
        assert!(session.error().is_some());
        assert_eq!(session.observation(), initial);
        assert_eq!(session.steps(), 0);
        assert!(session.last_action().is_none());
    }
}
