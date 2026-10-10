//! Permanent inference and shared-world rejection boundaries.

use super::*;
use bevy_gym::{
    robots::RobotWorldFailure,
    training::{RecurrentPpoAgent, RecurrentPpoConfig, SeedConfig},
};

/// Construct the same retained RL session used by the external playback seam.
fn session() -> Session {
    Session::load(
        DRONE_REFERENCE.to_vec(),
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk")
            .to_vec(),
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.json"),
    )
    .expect("retained RL candidates")
}

/// A malformed test-only actor cannot advance physics or revive after reset.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn inference_failure_revokes_all_six_controllers() {
    let mut session = session();
    let wrong = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[0.0; 4],
        &[1.0; 4],
        RecurrentPpoConfig::default(),
        SeedConfig::from_root(42),
    )
    .expect("test-only wrong input width")
    .policy();
    let droid = crate::standing::model::load_policy(
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk")
            .to_vec(),
    )
    .expect("standing candidate");
    session.inference = Inference::Ready(Controllers::new(wrong, droid));
    let before = session.snapshot().clone();
    session.step();
    assert_eq!(session.snapshot().frame(), before.frame());
    assert_eq!(session.snapshot().elapsed(), Duration::ZERO);
    let failure = session.error().expect("inference stops");
    assert!(failure.to_string().starts_with("Drone(First):"));
    assert!(failure.source().is_some());
    for slot in RobotSlot::ALL {
        assert!(session.last_drone_action(slot).is_none());
        assert!(session.last_droid_action(slot).is_none());
        assert_eq!(session.snapshot().drone(slot), before.drone(slot));
        assert_eq!(session.snapshot().droid(slot), before.droid(slot));
    }
    let message = failure.to_string();
    session.step();
    session.reset();
    session.step();
    assert_eq!(
        session.error().expect("failure is permanent").to_string(),
        message
    );
    assert_eq!(session.snapshot().elapsed(), Duration::ZERO);
}

/// A foreign snapshot stops the common world without publishing staged requests.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn world_failure_preserves_last_published_actions() {
    let mut session = session();
    session.step();
    let last_drone = session.last_drone_action(RobotSlot::First);
    let last_droid = session.last_droid_action(RobotSlot::Third);
    session.snapshot = RobotWorld::default().snapshot().expect("foreign world");
    session.step();
    assert!(matches!(
        session.inference,
        Inference::Failed(Failure::World(bevy_gym::robots::RobotWorldError::Rejected(
            RobotWorldFailure::WrongFrame
        )))
    ));
    assert_eq!(session.snapshot().elapsed(), Duration::ZERO);
    assert_eq!(session.last_drone_action(RobotSlot::First), last_drone);
    assert_eq!(session.last_droid_action(RobotSlot::Third), last_droid);
    let error = session.error().expect("world stop");
    assert!(error.source().is_some());
    assert_eq!(
        error.to_string(),
        error.source().expect("world error").to_string()
    );
    session.reset();
    session.step();
    assert!(session.error().is_some());
    assert_eq!(session.snapshot().elapsed(), Duration::ZERO);
    assert!(session.last_drone_action(RobotSlot::First).is_none());
    assert!(session.last_droid_action(RobotSlot::Third).is_none());
}
