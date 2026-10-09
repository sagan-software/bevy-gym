//! Validate browser checkpoints before handing them to the Bevy session.

use std::error::Error;

use super::pilot::RecoveryPilot;

/// At most one validated policy waits for the next Bevy update.
#[derive(Default)]
struct CheckpointInbox {
    /// A newer valid selection replaces this one before it can reach physics.
    next: Option<RecoveryPilot>,
}

impl CheckpointInbox {
    /// Reject invalid uploads before replacing a previously validated selection.
    fn queue(&mut self, bytes: Vec<u8>) -> Result<(), Box<dyn Error>> {
        if bytes.len() > 1_048_576 {
            return Err("Checkpoint exceeds 1 MiB.".into());
        }
        let pilot = RecoveryPilot::load(bytes)?;
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
