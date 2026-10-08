//! Gymnasium continuous `MountainCar` contracts shared by native and browser consumers.

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde as _;
use serde_json as _;
use shakmaty as _;
use tokio as _;

use bevy_gym::environments::{
    ContinuousMountainCar, ContinuousMountainCarAction, MountainCarState,
};
use bevy_gym::{Env, EpisodeStatus};

#[test]
fn pinned_numpy_trajectories_match_observation_bits_and_original_rewards() {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/gymnasium/continuous_mountain_car_sequences.json"
    ))
    .expect("pinned trajectories");
    let mut transitions = 0;
    for case in fixtures["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let state: [f64; 2] = serde_json::from_value(case["initial_state"].clone()).expect("state");
        let mut env = ContinuousMountainCar::from_state(
            MountainCarState::try_from(state).expect("finite state"),
        );
        for expected in case["steps"].as_array().expect("steps") {
            let raw = expected["action"].as_f64().expect("action") as f32;
            let step = env.step(ContinuousMountainCarAction::try_from(raw).expect("finite action"));
            let bits: [u32; 2] =
                serde_json::from_value(expected["observation_bits"].clone()).expect("bits");
            assert_eq!(
                step.observation.map(f32::to_bits),
                bits,
                "case {name}, transition {transitions}"
            );
            assert!((step.reward - expected["reward"].as_f64().expect("reward")).abs() <= 1e-12);
            assert_eq!(
                step.status == EpisodeStatus::Terminated,
                expected["terminated"].as_bool().expect("termination")
            );
            assert_eq!(
                env.state().map(f64::to_bits),
                step.observation.map(f64::from).map(f64::to_bits)
            );
            transitions += 1;
        }
    }
    assert_eq!(transitions, 201);
}

#[test]
fn finite_raw_actions_are_preserved_for_reward_before_force_clipping() {
    for raw in [-f32::MAX, -2.0, -1.0, 0.0, 1.0, 2.0, f32::MAX] {
        let action = ContinuousMountainCarAction::try_from(raw).expect("finite");
        assert_eq!(f32::from(action).to_bits(), raw.to_bits());
        let mut env = ContinuousMountainCar::default();
        let step = env.step(action);
        assert!(step.reward.is_finite());
        assert!(step.observation.iter().all(|value| value.is_finite()));
    }
    for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = ContinuousMountainCarAction::try_from(raw).expect_err("nonfinite");
        assert_eq!(
            error.to_string(),
            "Continuous MountainCar action must be finite"
        );
        assert!(std::error::Error::source(&error).is_none());
    }
}

#[test]
fn reset_is_seeded_and_time_limit_remains_external() {
    let mut first = ContinuousMountainCar::default();
    let mut second = ContinuousMountainCar::default();
    assert_eq!(first.reset(Some(42)), second.reset(Some(42)));
    assert_eq!(first.reset(None), second.reset(None));
    let mut mean = 0.0_f64;
    for seed in 0..1000 {
        first.reset(Some(seed));
        let [position, velocity] = first.state();
        assert!((-0.6..=-0.4).contains(&position));
        assert_eq!(velocity.to_bits(), 0.0_f64.to_bits());
        mean += position / 1000.0;
    }
    assert!((mean + 0.5).abs() < 0.01);
    let mut limited =
        bevy_gym::TimeLimit::new(ContinuousMountainCar::default(), 999).expect("positive limit");
    limited.reset(Some(42));
    let coast = ContinuousMountainCarAction::try_from(0.0).expect("finite");
    for _ in 0..998 {
        assert_eq!(limited.step(coast).status, EpisodeStatus::Continuing);
    }
    assert_eq!(limited.step(coast).status, EpisodeStatus::Truncated);
}
