//! Pendulum's pinned numerical contract at the shared native/browser boundary.

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
use shakmaty as _;
use tokio as _;

use bevy_gym::environments::{Pendulum, PendulumAction, PendulumState};
use bevy_gym::{Env, EpisodeStatus, TimeLimit};

#[test]
fn pinned_numpy_trajectories_preserve_double_state_and_float_observations() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gymnasium/pendulum_sequences.json"))
            .expect("pinned trajectories");
    let mut transitions = 0;
    for case in fixtures["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let state: [f64; 2] = serde_json::from_value(case["initial_state"].clone()).expect("state");
        let mut env = Pendulum::from_state(PendulumState::try_from(state).expect("finite state"));
        for expected in case["steps"].as_array().expect("steps") {
            let raw = expected["action"].as_f64().expect("action") as f32;
            let step = env.step(PendulumAction::try_from(raw).expect("finite torque"));
            let bits: [u32; 3] =
                serde_json::from_value(expected["observation_bits"].clone()).expect("bits");
            assert_eq!(
                step.observation.map(f32::to_bits),
                bits,
                "case {name}, step {transitions}"
            );
            let state: [f64; 2] = serde_json::from_value(expected["state"].clone()).expect("state");
            for (actual, expected) in env.state().into_iter().zip(state) {
                assert!(
                    (actual - expected).abs() <= 1e-10,
                    "case {name}, step {transitions}: {actual} != {expected}"
                );
            }
            assert!((step.reward - expected["reward"].as_f64().expect("reward")).abs() <= 1e-10);
            assert_eq!(
                env.last_torque(),
                Some(expected["last_torque"].as_f64().expect("torque") as f32)
            );
            assert_eq!(step.status, EpisodeStatus::Continuing);
            transitions += 1;
        }
    }
    assert_eq!(transitions, 420);
}

#[test]
fn reset_restores_default_bounds_and_only_the_wrapper_truncates() {
    let mut first = Pendulum::default();
    let mut second = Pendulum::default();
    assert_eq!(first.reset(Some(42)), second.reset(Some(42)));
    assert_eq!(first.reset(None), second.reset(None));
    let mut means = [0.0_f64; 2];
    let mut second_moments = [0.0_f64; 2];
    for seed in 0..1000 {
        first.reset(Some(seed));
        let [angle, velocity] = first.state();
        assert!((-std::f64::consts::PI..std::f64::consts::PI).contains(&angle));
        assert!((-1.0..1.0).contains(&velocity));
        assert_eq!(first.last_torque(), None);
        means[0] += angle / 1000.0;
        means[1] += velocity / 1000.0;
        second_moments[0] += angle.powi(2) / 1000.0;
        second_moments[1] += velocity.powi(2) / 1000.0;
    }
    assert!(means.into_iter().all(|mean| mean.abs() < 0.1));
    assert!((2.9..3.7).contains(&second_moments[0]));
    assert!((0.29..0.37).contains(&second_moments[1]));
    let torque = PendulumAction::try_from(0.0).expect("finite");
    for _ in 0..201 {
        assert_eq!(first.step(torque).status, EpisodeStatus::Continuing);
    }
    first.reset(None);
    assert_eq!(first.last_torque(), None);
    let mut limited = TimeLimit::new(first, 200).expect("positive time limit");
    limited.reset(Some(42));
    for _ in 0..199 {
        assert_eq!(limited.step(torque).status, EpisodeStatus::Continuing);
    }
    assert_eq!(limited.step(torque).status, EpisodeStatus::Truncated);
}

#[test]
fn finite_torque_is_retained_and_clipped_before_cost_and_acceleration() {
    for raw in [-f32::MAX, -3.0, -2.0, 0.0, 2.0, 3.0, f32::MAX] {
        let action = PendulumAction::try_from(raw).expect("finite torque");
        assert_eq!(f32::from(action).to_bits(), raw.to_bits());
        let mut env = Pendulum::from_state(PendulumState::try_from([0.0, 0.0]).expect("upright"));
        let step = env.step(action);
        let clipped = raw.clamp(-2.0, 2.0);
        assert_eq!(env.last_torque(), Some(clipped));
        assert_eq!(
            step.reward.to_bits(),
            (-f64::from(0.001_f32 * clipped.powi(2))).to_bits()
        );
        assert!((env.state()[1] - f64::from(3.0_f32 * clipped) * 0.05).abs() < 1e-15);
    }
    for raw in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = PendulumAction::try_from(raw).expect_err("nonfinite torque");
        assert_eq!(error.to_string(), "Pendulum torque must be finite");
        assert!(std::error::Error::source(&error).is_none());
    }
}

#[test]
fn invalid_diagnostic_states_cannot_enter_the_environment() {
    for values in [
        [f64::NAN, 0.0],
        [f64::INFINITY, 0.0],
        [f64::NEG_INFINITY, 0.0],
        [0.0, f64::NAN],
        [0.0, f64::INFINITY],
        [0.0, f64::NEG_INFINITY],
        [0.0, f64::MAX],
    ] {
        let error = PendulumState::try_from(values).expect_err("invalid state");
        assert_eq!(
            error.to_string(),
            "Pendulum requires a finite angle and a finite squared angular velocity"
        );
        assert!(std::error::Error::source(&error).is_none());
    }
    let state =
        PendulumState::try_from([17.0 * std::f64::consts::PI, 8.0]).expect("unwrapped angle");
    assert_eq!(state.values()[1].to_bits(), 8.0_f64.to_bits());
    let state = PendulumState::try_from([f64::MAX, f64::MAX.sqrt() / 2.0])
        .expect("large finite state with finite squared velocity");
    let mut env = Pendulum::from_state(state);
    let step = env.step(PendulumAction::try_from(0.0).expect("finite torque"));
    assert!(step.reward.is_finite());
    assert!(step.observation.into_iter().all(f32::is_finite));
}
