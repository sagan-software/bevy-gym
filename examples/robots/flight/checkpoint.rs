//! Validate browser checkpoints before handing them to the Bevy session.

use std::error::Error;

use super::pilot::{CheckpointKind, RecoveryPilot, MAX_CHECKPOINT_BYTES};

/// At most one validated policy waits for the next Bevy update.
#[derive(Default)]
struct CheckpointInbox {
    /// A newer valid selection replaces this one before it can reach physics.
    next: Option<RecoveryPilot>,
}

impl CheckpointInbox {
    /// Reject invalid uploads before replacing a previously validated selection.
    fn queue(&mut self, bytes: Vec<u8>) -> Result<(), Box<dyn Error>> {
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err("Checkpoint exceeds 1 MiB.".into());
        }
        let pilot = RecoveryPilot::load(bytes)?;
        self.next = Some(pilot);
        Ok(())
    }

    /// Keep the previous selection until a local file passes its selected recipe.
    fn queue_file(&mut self, bytes: Vec<u8>, kind: CheckpointKind) -> Result<(), Box<dyn Error>> {
        let pilot = RecoveryPilot::load_file(bytes, kind)?;
        self.next = Some(pilot);
        Ok(())
    }

    /// Move the selected policy into the simulation exactly once.
    const fn take(&mut self) -> Option<RecoveryPilot> {
        self.next.take()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "browser"))]
thread_local! {
    /// The browser and Bevy share one main thread; the trainer owns another worker.
    static INBOX: std::cell::RefCell<CheckpointInbox> = const {
        std::cell::RefCell::new(CheckpointInbox { next: None })
    };
}

/// Validate an explicitly selected training checkpoint before the next scene update.
///
/// # Errors
/// Returns a JavaScript diagnostic for oversized, malformed, or incompatible weights.
/// A rejected upload leaves the previous pending selection unchanged.
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn watch_checkpoint(bytes: Vec<u8>) -> Result<(), wasm_bindgen::JsValue> {
    INBOX
        .with(|inbox| inbox.borrow_mut().queue(bytes))
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Watch a local twelve-input recovery checkpoint after validating its size and shape.
///
/// # Errors
/// Returns a diagnostic without replacing the previous pending policy on failure.
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn watch_recovery_file(bytes: Vec<u8>) -> Result<(), wasm_bindgen::JsValue> {
    queue_file(bytes, CheckpointKind::Recovery)
}

/// Watch a local sixteen-input motor-failure checkpoint from a calm start.
///
/// # Errors
/// Returns a diagnostic without replacing the previous pending policy on failure.
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn watch_motor_failure_file(bytes: Vec<u8>) -> Result<(), wasm_bindgen::JsValue> {
    queue_file(bytes, CheckpointKind::MotorFailure)
}

/// Convert a file-validation error at the JavaScript boundary.
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
fn queue_file(bytes: Vec<u8>, kind: CheckpointKind) -> Result<(), wasm_bindgen::JsValue> {
    INBOX
        .with(|inbox| inbox.borrow_mut().queue_file(bytes, kind))
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Apply one browser selection before processing the scene's other controls.
#[cfg(all(target_arch = "wasm32", feature = "browser"))]
pub(super) fn apply(mut session: bevy::prelude::ResMut<'_, super::session::Session>) {
    if let Some(pilot) = INBOX.with(|inbox| inbox.borrow_mut().take()) {
        session.watch_policy(pilot);
    }
}

#[cfg(test)]
mod tests {
    use super::CheckpointInbox;
    use crate::session::{Playback, Session, StartProfile};

    #[test]
    fn a_new_valid_selection_replaces_the_pending_policy() {
        use crate::{encoding, model};
        use bevy_gym::robots::DroneHover;
        use bevy_gym::Env;

        let first = include_bytes!("../../../assets/robots/recovery.mpk").to_vec();
        let second = include_bytes!("../../../docs/progress/drone-browser-training.mpk").to_vec();
        let expected = model::load_policy(second.clone()).expect("browser checkpoint");
        let observation = DroneHover::disturbed().reset(Some(42)).observation;
        let mut inbox = CheckpointInbox::default();
        inbox.queue(first).expect("first policy");
        inbox.queue(second).expect("replacement policy");
        let mut actual = inbox.take().expect("latest selection");
        let command = expected
            .mean_action(&encoding::encode(observation), &expected.initial_memory())
            .expect("valid policy output");
        assert_eq!(
            actual
                .command(observation)
                .expect("selected policy action")
                .fractions(),
            encoding::decode_action(&command.action)
                .expect("expected action")
                .fractions()
        );
        assert!(inbox.take().is_none());
    }

    #[test]
    fn rejected_uploads_preserve_the_pending_checkpoint() {
        let bytes = include_bytes!("../../../assets/robots/recovery.mpk").to_vec();
        let mut inbox = CheckpointInbox::default();
        assert!(inbox.take().is_none());
        inbox.queue(bytes).expect("valid policy");
        inbox
            .queue(vec![0; 1_048_577])
            .expect_err("oversized input");
        inbox.queue(Vec::new()).expect_err("malformed input");
        let mut session = Session::default();
        session.watch_policy(inbox.take().expect("valid policy preserved"));
        assert_eq!(session.start_profile(), StartProfile::Disturbed);
        assert_eq!(session.playback(), Playback::Running);
        assert_eq!(session.steps(), 0);
        session.advance();
        assert_eq!(session.steps(), 1);
        assert!(!session.is_bundled());
        assert!(inbox.take().is_none());
        session.reset();
        assert!(!session.is_bundled());
        session.select_learned();
        assert!(session.is_bundled());
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use crate::{
        model,
        session::{Playback, Session, StartProfile},
    };
    use bevy_gym::{
        robots::{DroneMotor, DroneMotorState},
        training::{RecurrentPpoAgent, SeedConfig},
    };

    #[test]
    fn file_validation_keeps_the_latest_valid_recipe_until_the_scene_takes_it() {
        let healthy = include_bytes!("../../../assets/robots/recovery.mpk").to_vec();
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
        let mut inbox = CheckpointInbox::default();
        inbox
            .queue_file(healthy.clone(), CheckpointKind::Recovery)
            .unwrap();
        inbox
            .queue_file(policy.to_bytes().unwrap(), CheckpointKind::MotorFailure)
            .unwrap();
        inbox
            .queue_file(healthy, CheckpointKind::MotorFailure)
            .expect_err("wrong recipe");
        inbox
            .queue_file(Vec::new(), CheckpointKind::MotorFailure)
            .expect_err("corrupt file");
        inbox
            .queue_file(
                vec![0; MAX_CHECKPOINT_BYTES + 1],
                CheckpointKind::MotorFailure,
            )
            .expect_err("oversized file");
        let mut session = Session::default();
        session.watch_policy(inbox.take().unwrap());
        assert_eq!(session.start_profile(), StartProfile::Calm);
        assert_eq!(session.playback(), Playback::Running);
        assert_eq!(session.policy_label(), Some("Motor-failure checkpoint"));
        assert!(inbox.take().is_none());
        session.advance();
        assert_eq!(session.steps(), 1);
        session.fail_motor(DroneMotor::FrontLeft);
        assert_eq!(
            session.observation().motor_state(DroneMotor::FrontLeft),
            DroneMotorState::Failed
        );
        session.reset();
        assert_eq!(session.steps(), 0);
        assert_eq!(session.playback(), Playback::Paused);
        assert_eq!(session.policy_label(), Some("Motor-failure checkpoint"));
        assert_eq!(
            session.observation().motor_state(DroneMotor::FrontLeft),
            DroneMotorState::Working
        );
    }
}
