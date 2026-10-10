//! Frozen standing inference; only validated PPO torque outputs reach physical bodies.

use bevy_gym::robots::{DroidAction, DroidObservation, DroidStanding};
use bevy_gym::training::{RecurrentMemory, RecurrentPpoPolicy};
use bevy_gym::{Env, EpisodeStatus, TimeLimit};

use crate::standing::{checkpoint, encoding, evaluation};

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

/// Private physics and policy state shared by the standalone skill scenes.
pub(crate) struct Session {
    /// Validated identity and claimed update count for the retained weights.
    record: checkpoint::Record,
    /// The same bounded physical task used by curriculum evaluation.
    environment: TimeLimit<DroidStanding>,
    /// Latest authoritative pose and velocities, including the initial reset.
    observation: DroidObservation,
    /// Environment completion; inference failures remain a separate diagnostic.
    status: EpisodeStatus,
    /// The only source of actuator commands.
    inference: Inference,
    /// Last applied joint values for read-only visualization.
    last_action: Option<DroidAction>,
}

/// Validate the immutable checkpoint record before constructing physics.
pub(crate) fn load(
    bytes: Vec<u8>,
    metadata: &[u8],
    seed: u64,
) -> Result<Session, Box<dyn std::error::Error>> {
    let (policy, record) = checkpoint::from_bytes(bytes, metadata)?;
    let memory = policy.initial_memory();
    let mut environment =
        TimeLimit::new(DroidStanding::default(), usize::from(evaluation::HORIZON))?;
    let observation = environment.reset(Some(seed)).observation;
    Ok(Session {
        record,
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

impl Session {
    /// Read the update count from the validated checkpoint instead of a display constant.
    pub(crate) const fn checkpoint_update(&self) -> std::num::NonZeroU32 {
        self.record.update()
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
        let inferred = match policy.mean_action(&encoding::encode(&self.observation), memory) {
            Ok(inferred) => inferred,
            Err(error) => {
                self.inference = Inference::Failed(error.to_string());
                return;
            }
        };
        // Invalid actuator output cannot reach the environment or update memory.
        let action = match encoding::decode(&inferred.action) {
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
    pub(crate) const fn observation(&self) -> DroidObservation {
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

    /// Read the configured horizon without storing a duplicate count.
    pub(crate) const fn horizon(&self) -> usize {
        self.environment.max_episode_steps()
    }

    /// Read joint values only after a policy action has reached physics.
    pub(crate) const fn last_action(&self) -> Option<DroidAction> {
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

    /// Invalid recurrent state cannot issue torques, including after reset.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn inference_failure_is_sticky_before_physics() {
        let mut session = load(
            include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec(),
            include_bytes!("../../../docs/progress/droid-standing-trial.json"),
            42,
        )
        .expect("recorded PPO candidate");
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
        assert!(session.error().is_some());
        assert_eq!(session.observation(), initial);
        assert_eq!(session.steps(), 0);
    }

    /// Incompatible actuator scaling stops before the validated action reaches physics.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn invalid_torque_output_stops_before_physics() {
        let bytes = include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec();
        let metadata = include_bytes!("../../../docs/progress/droid-standing-trial.json");
        let mut session = load(bytes.clone(), metadata, 42).expect("RL candidate");
        let policy = RecurrentPpoPolicy::load_bytes(
            bytes,
            204,
            204,
            1,
            &[2.0; 26],
            &[3.0; 26],
            &bevy_gym::training::RecurrentPpoConfig::default(),
        )
        .expect("same shape with invalid scaling");
        session.inference = Inference::Ready {
            memory: policy.initial_memory(),
            policy: Box::new(policy),
        };
        let initial = session.observation();
        session.step();
        assert!(session.error().is_some());
        assert!(session.last_action().is_none());
        assert_eq!(session.steps(), 0);
        assert_eq!(session.observation(), initial);
    }
}
