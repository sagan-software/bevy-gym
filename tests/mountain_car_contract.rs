//! Gymnasium `MountainCar` contracts shared by native and browser consumers.

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

use bevy_gym::environments::{MountainCar, MountainCarAction, MountainCarState};
use bevy_gym::{Env, EpisodeStatus};

#[test]
fn pinned_oracle_transitions_preserve_double_state_and_float_observations() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gymnasium/transition_cases.json"))
            .expect("oracle JSON");
    let mut count = 0;
    for case in fixtures["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|case| case["registry_id"] == "MountainCar-v0")
    {
        let state: [f64; 2] =
            serde_json::from_value(case["injection"]["internal_state"].clone()).expect("state");
        let action = MountainCarAction::try_from(
            case["injection"]["action"].as_u64().expect("action") as usize,
        )
        .expect("valid action");
        let mut env =
            MountainCar::from_state(MountainCarState::try_from(state).expect("finite state"));
        let result = env.step(action);
        let expected: [f64; 2] =
            serde_json::from_value(case["expected"]["internal_state"].clone()).expect("state");
        for (actual, expected) in env.state().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        let observation: [f32; 2] =
            serde_json::from_value(case["expected"]["observation"].clone()).expect("observation");
        assert_eq!(
            result.observation.map(f32::to_bits),
            observation.map(f32::to_bits)
        );
        assert_eq!(result.reward.to_bits(), (-1.0_f64).to_bits());
        assert_eq!(result.status, EpisodeStatus::Continuing);
        count += 1;
    }
    assert_eq!(count, 2);
}

#[test]
fn goal_requires_position_and_nonnegative_velocity() {
    for (state, expected) in [
        (
            [0.5, (3.0_f64 * 0.5).cos() * 0.0025],
            EpisodeStatus::Terminated,
        ),
        ([0.55, -0.02], EpisodeStatus::Continuing),
        ([0.5, 0.01], EpisodeStatus::Terminated),
        ([0.49, 0.0], EpisodeStatus::Continuing),
    ] {
        let mut env = MountainCar::from_state(MountainCarState::try_from(state).expect("finite"));
        assert_eq!(env.step(MountainCarAction::Coast).status, expected);
    }
}

#[test]
fn velocity_and_position_clip_before_the_left_wall_stops_the_car() {
    for (state, action, expected) in [
        ([0.59, 1.0], MountainCarAction::Right, [0.6, 0.07]),
        ([-1.19, -1.0], MountainCarAction::Left, [-1.2, 0.0]),
        ([-0.5, -1.0], MountainCarAction::Left, [-0.57, -0.07]),
    ] {
        let mut env = MountainCar::from_state(MountainCarState::try_from(state).expect("finite"));
        env.step(action);
        for (actual, expected) in env.state().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn action_and_state_boundaries_reject_invalid_inputs() {
    for (index, action) in [
        (0, MountainCarAction::Left),
        (1, MountainCarAction::Coast),
        (2, MountainCarAction::Right),
    ] {
        assert_eq!(MountainCarAction::try_from(index), Ok(action));
        assert_eq!(usize::from(action), index);
    }
    for index in [3, usize::MAX] {
        MountainCarAction::try_from(index).expect_err("outside action vocabulary");
    }
    for coordinate in 0..2 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut state = [0.0; 2];
            state[coordinate] = invalid;
            MountainCarState::try_from(state).expect_err("nonfinite coordinate");
        }
    }
}

#[test]
fn reset_is_seeded_uniform_and_time_limit_is_external() {
    let mut first = MountainCar::default();
    let mut second = MountainCar::default();
    assert_eq!(
        first.reset(Some(42)).observation.map(f32::to_bits),
        second.reset(Some(42)).observation.map(f32::to_bits)
    );
    for _ in 0..100 {
        let result = first.reset(None);
        assert!((-0.6..=-0.4).contains(&result.observation[0]));
        assert_eq!(result.observation[1].to_bits(), 0.0_f32.to_bits());
        assert_eq!(
            result.observation.map(f32::to_bits),
            second.reset(None).observation.map(f32::to_bits)
        );
    }
    let mut limited = bevy_gym::wrappers::time_limit::TimeLimit::new(MountainCar::default(), 200)
        .expect("positive limit");
    limited.reset(Some(42));
    for _ in 0..199 {
        assert_eq!(
            limited.step(MountainCarAction::Coast).status,
            EpisodeStatus::Continuing
        );
    }
    assert_eq!(
        limited.step(MountainCarAction::Coast).status,
        EpisodeStatus::Truncated
    );
    let mut unwrapped = MountainCar::default();
    for _ in 0..201 {
        assert_eq!(
            unwrapped.step(MountainCarAction::Coast).status,
            EpisodeStatus::Continuing
        );
    }
}

#[test]
fn boundary_errors_describe_the_rejected_contract() {
    for position in [f64::MAX / 4.0, -f64::MAX / 4.0] {
        MountainCarState::try_from([position, 0.0]).expect("finite terrain angle");
    }
    for position in [f64::MAX, -f64::MAX] {
        MountainCarState::try_from([position, 0.0]).expect_err("terrain angle would overflow");
    }
    assert_eq!(
        bevy_gym::environments::InvalidMountainCarAction.to_string(),
        "MountainCar action must be 0, 1, or 2"
    );
    assert_eq!(
        bevy_gym::environments::InvalidMountainCarState.to_string(),
        "MountainCar state requires finite velocity and finite 3 * position"
    );
}
