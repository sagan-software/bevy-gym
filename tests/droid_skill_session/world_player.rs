//! Spectator commands change timing while retaining exact six-policy inference.

use crate::{
    standing::world_session::Session,
    world_player::{Command, Player},
};
use std::time::Duration;

/// Load the recorded policies without a renderer or mutable actuator interface.
fn load() -> Session {
    Session::load(
        include_bytes!("../../docs/progress/drone-hover.mpk").to_vec(),
        include_bytes!("../../docs/progress/standing-seed17-update22940/checkpoint.mpk").to_vec(),
        include_bytes!("../../docs/progress/standing-seed17-update22940/checkpoint.json"),
    )
    .expect("recorded frozen RL policies")
}

/// Compare physical observations and actual applied actions, excluding world-specific identity.
fn assert_same(actual: &Session, expected: &Session) {
    assert_eq!(actual.snapshot().elapsed(), expected.snapshot().elapsed());
    for slot in bevy_gym::robots::RobotSlot::ALL {
        assert_eq!(
            actual.snapshot().drone(slot),
            expected.snapshot().drone(slot)
        );
        assert_eq!(
            actual.snapshot().droid(slot),
            expected.snapshot().droid(slot)
        );
        assert_eq!(
            actual.last_drone_action(slot),
            expected.last_drone_action(slot)
        );
        assert_eq!(
            actual.last_droid_action(slot),
            expected.last_droid_action(slot)
        );
    }
}

/// Missing models block run and step; speed and reset remain presentation operations.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn playback_readiness_speed_and_reset_match_direct_inference() {
    let mut player = Player::new(Ok(load()));
    player.apply(Command::Play, false);
    player.advance(false);
    player.apply(Command::Step, false);
    assert!(!player.running());
    assert_eq!(
        player.session().expect("session").snapshot().elapsed(),
        Duration::ZERO
    );
    player.apply(Command::Play, true);
    assert!(player.running());
    player.advance(false);
    assert_eq!(
        player.session().expect("session").snapshot().elapsed(),
        Duration::ZERO
    );
    player.apply(Command::Play, true);
    assert!(!player.running());
    let mut direct = load();
    for count in [1, 4, 16] {
        assert_eq!(player.rate(), count);
        player.apply(Command::Play, true);
        player.advance(true);
        player.apply(Command::Play, true);
        for _ in 0..count {
            direct.step();
        }
        let actual = player.session().expect("session");
        assert_same(actual, &direct);
        player.apply(Command::Speed, true);
    }
    assert_eq!(player.rate(), 1);
    player.apply(Command::Play, true);
    player.apply(Command::Step, true);
    direct.step();
    assert!(!player.running());
    assert_same(player.session().expect("session"), &direct);
    player.apply(Command::Reset, false);
    let fresh = load();
    assert_same(player.session().expect("session"), &fresh);
    player.apply(Command::Step, true);
    let mut fresh = fresh;
    fresh.step();
    assert_same(player.session().expect("session"), &fresh);
}

/// Clip completion pauses all six policies and blocks playback until an explicit reset.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn completed_clip_pauses_and_requires_reset_before_playback() {
    let mut player = Player::new(Ok(load()));
    player.apply(Command::Speed, true);
    player.apply(Command::Speed, true);
    player.apply(Command::Play, true);
    for _ in 0..63 {
        player.advance(true);
    }
    assert_eq!(player.rate(), 16);
    assert!(!player.running());
    let session = player.session().expect("session");
    assert!(session.finished());
    assert_eq!(session.snapshot().elapsed(), Duration::from_secs(20));
    player.apply(Command::Play, true);
    player.advance(true);
    player.apply(Command::Step, true);
    assert!(!player.running());
    assert_eq!(
        player.session().expect("session").snapshot().elapsed(),
        Duration::from_secs(20)
    );
    player.apply(Command::Reset, true);
    assert!(!player.running());
    assert_eq!(
        player.session().expect("session").snapshot().elapsed(),
        Duration::ZERO
    );
    player.apply(Command::Play, true);
    assert!(player.running());
}
