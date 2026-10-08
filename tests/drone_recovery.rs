//! Public reset-distribution and feedback-control requirements for drone recovery.

#![cfg(feature = "robots")]

use bevy::math::{EulerRot, Vec3};
use bevy_gym::robots::{DroneAction, DroneHover};
use bevy_gym::{Env, EpisodeStatus};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn disturbed_resets_have_bounded_tilt_and_velocity() {
    let mut drone = DroneHover::disturbed();
    for seed in (0..100).chain([u64::MAX]) {
        let state = drone.reset(Some(seed)).observation;
        let position = state.position();
        assert!(position.is_finite());
        assert!(position.x.abs() <= 0.2 && position.z.abs() <= 0.2);
        assert!((1.9..=2.1).contains(&position.y));
        let orientation = state.orientation();
        assert!(orientation.is_finite() && orientation.is_normalized());
        let (yaw, pitch, roll) = orientation.to_euler(EulerRot::YXZ);
        assert!(yaw.abs() <= std::f32::consts::PI);
        let tilt_bound = std::f32::consts::PI / 12.0 + 1e-6;
        assert!(pitch.abs() <= tilt_bound && roll.abs() <= tilt_bound);
        for velocity in [state.linear_velocity(), state.angular_velocity()] {
            assert!(velocity.is_finite());
            assert!(velocity.abs().cmple(Vec3::splat(0.5)).all());
            assert!(velocity.length_squared() > 0.0);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn constructed_episode_matches_an_explicit_seed_zero_reset() {
    let mut constructed = DroneHover::disturbed();
    let mut reset = DroneHover::disturbed();
    reset.reset(Some(0));
    let action = DroneAction::try_from([0.5; 4]).expect("half thrust is valid");
    assert_eq!(constructed.step(action), reset.step(action));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn disturbed_seed_stream_and_profile_survive_terminal_resets() {
    let mut left = DroneHover::disturbed();
    let mut right = DroneHover::disturbed();
    let initial = left.reset(Some(42));
    assert_eq!(initial, right.reset(Some(42)));
    let off = DroneAction::try_from([0.0; 4]).expect("power off is valid");
    let terminal = (0..500)
        .map(|_| left.step(off))
        .find(bevy_gym::Step::is_done)
        .expect("unpowered drone reaches the ground");
    assert_eq!(terminal.status, EpisodeStatus::Terminated);
    assert_eq!(terminal, left.step(off));
    let next = left.reset(None);
    assert_ne!(initial, next);
    assert_eq!(next, right.reset(None));
    for _ in 0..10 {
        assert_eq!(left.step(off), right.step(off));
    }
    assert_eq!(initial, left.reset(Some(42)));
    assert_eq!(initial, right.reset(Some(42)));
    assert_eq!(left.step(off), right.step(off));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn constant_half_thrust_fails_the_held_out_recovery_seeds() {
    let action = DroneAction::try_from([0.5; 4]).expect("half thrust is valid");
    for seed in [0, 1, 2, 42, u64::MAX] {
        let mut drone = DroneHover::disturbed();
        let initial = drone.reset(Some(seed)).observation;
        let terminal = (0..500)
            .map(|_| drone.step(action))
            .find(bevy_gym::Step::is_done)
            .expect("fixed thrust cannot recover the disturbed drone");
        assert_eq!(terminal.status, EpisodeStatus::Terminated);
        assert!(terminal.observation.position().distance(initial.position()) > 0.5);
    }
}
