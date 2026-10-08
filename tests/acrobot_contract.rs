//! Acrobot's pinned numerical contract at the shared native/browser boundary.

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

use bevy_gym::environments::{Acrobot, AcrobotAction, AcrobotState};
use bevy_gym::{Env, EpisodeStatus, TimeLimit};

#[test]
fn pinned_numpy_trajectories_preserve_book_dynamics() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gymnasium/acrobot_sequences.json"))
            .expect("pinned trajectories");
    let mut transitions = 0;
    for case in fixtures["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let state: [f64; 4] = serde_json::from_value(case["initial_state"].clone()).expect("state");
        let mut env = Acrobot::from_state(AcrobotState::try_from(state).expect("bounded state"));
        for expected in case["steps"].as_array().expect("steps") {
            let index = expected["action"].as_u64().expect("action") as usize;
            let step = env.step(AcrobotAction::try_from(index).expect("three actions"));
            let state: [f64; 4] = serde_json::from_value(expected["state"].clone()).expect("state");
            for (actual, expected) in env.state().iter().copied().zip(state) {
                assert!(
                    (actual - expected).abs() <= 1e-10,
                    "case {name}, step {transitions}: {actual} != {expected}"
                );
            }
            let bits: [u32; 6] =
                serde_json::from_value(expected["observation_bits"].clone()).expect("bits");
            assert_eq!(
                step.observation.map(f32::to_bits),
                bits,
                "case {name}, step {transitions}"
            );
            assert_eq!(
                step.reward.to_bits(),
                expected["reward"].as_f64().expect("reward").to_bits()
            );
            let status = if expected["terminated"].as_bool().expect("terminated") {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            };
            assert_eq!(step.status, status);
            transitions += 1;
        }
    }
    assert_eq!(transitions, 2133);
}

#[test]
fn action_indices_and_bounded_state_reject_every_invalid_boundary() {
    for (index, action) in [
        AcrobotAction::Negative,
        AcrobotAction::Coast,
        AcrobotAction::Positive,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(AcrobotAction::try_from(index), Ok(action));
        assert_eq!(usize::from(action), index);
    }
    for index in [3, usize::MAX] {
        let error = AcrobotAction::try_from(index).expect_err("outside three actions");
        assert_eq!(error.to_string(), "Acrobot action must be 0, 1, or 2");
        assert!(std::error::Error::source(&error).is_none());
    }
    let pi = std::f64::consts::PI;
    for (coordinate, limit) in [pi, pi, 4.0 * pi, 9.0 * pi].into_iter().enumerate() {
        for value in [-limit, 0.0, limit] {
            let mut state = [0.0; 4];
            state[coordinate] = value;
            let state = AcrobotState::try_from(state).expect("inclusive bounds");
            assert_eq!(state.values()[coordinate].to_bits(), value.to_bits());
        }
        for value in [
            -limit - 1e-9,
            limit + 1e-9,
            f64::NAN,
            f64::NEG_INFINITY,
            f64::INFINITY,
        ] {
            let mut state = [0.0; 4];
            state[coordinate] = value;
            let error = AcrobotState::try_from(state).expect_err("invalid coordinate");
            assert_eq!(
                error.to_string(),
                "Acrobot requires angles within ±pi and velocities within ±4pi and ±9pi"
            );
            assert!(std::error::Error::source(&error).is_none());
        }
    }
}

#[test]
fn reset_rounds_state_to_float32_and_preserves_the_uniform_distribution() {
    let mut env = Acrobot::default();
    let mut replay = Acrobot::default();
    assert_eq!(env.reset(Some(42)), replay.reset(Some(42)));
    assert_eq!(env.reset(None), replay.reset(None));
    let mut sum = [0.0_f64; 4];
    let mut squares = [0.0_f64; 4];
    for seed in 0..1000 {
        let reset = env.reset(Some(seed));
        for (index, value) in env.state().iter().copied().enumerate() {
            assert!((-f64::from(0.1_f32)..=f64::from(0.1_f32)).contains(&value));
            assert_eq!(value.to_bits(), f64::from(value as f32).to_bits());
            sum[index] += value;
            squares[index] = value.mul_add(value, squares[index]);
        }
        assert_eq!(
            reset.observation[4].to_bits(),
            (env.state()[2] as f32).to_bits()
        );
        assert_eq!(
            reset.observation[5].to_bits(),
            (env.state()[3] as f32).to_bits()
        );
    }
    assert!(sum.into_iter().all(|value| (value / 1000.0).abs() < 0.01));
    assert!(squares
        .into_iter()
        .all(|value| (0.0028..0.0038).contains(&(value / 1000.0))));
}

#[test]
fn reset_observations_follow_numpy_float32_trigonometry() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gymnasium/acrobot_sequences.json"))
            .expect("pinned resets");
    let mut env = Acrobot::default();
    for expected in fixtures["resets"].as_array().expect("resets") {
        let seed = expected["seed"].as_u64().expect("seed");
        let reset = env.reset(Some(seed));
        let state: [u32; 4] =
            serde_json::from_value(expected["state_bits"].clone()).expect("state bits");
        assert_eq!(
            env.state().map(f64::to_bits),
            state.map(f32::from_bits).map(f64::from).map(f64::to_bits)
        );
        let bits: [u32; 6] =
            serde_json::from_value(expected["observation_bits"].clone()).expect("bits");
        assert_eq!(
            reset.observation.map(f32::to_bits),
            bits,
            "reset seed {seed}"
        );
    }
}

#[test]
fn only_the_wrapper_applies_the_five_hundred_transition_limit() {
    let mut core = Acrobot::default();
    for _ in 0..501 {
        assert_eq!(
            core.step(AcrobotAction::Coast).status,
            EpisodeStatus::Continuing
        );
    }
    let mut limited = TimeLimit::new(Acrobot::default(), 500).expect("positive limit");
    for _ in 0..499 {
        assert_eq!(
            limited.step(AcrobotAction::Coast).status,
            EpisodeStatus::Continuing
        );
    }
    assert_eq!(
        limited.step(AcrobotAction::Coast).status,
        EpisodeStatus::Truncated
    );
}
