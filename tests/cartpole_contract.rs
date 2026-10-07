//! Gymnasium CartPole contracts shared by native and browser consumers.

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

use bevy_gym::environments::{CartPole, CartPoleAction, CartPoleState};
use bevy_gym::{Env, EpisodeStatus};

#[test]
fn zero_state_transition_matches_the_pinned_python_oracle() {
    let state = CartPoleState::try_from([0.0; 4]).expect("finite initial state");
    let mut environment = CartPole::from_state(state);
    let result = environment.step(CartPoleAction::Right);
    let expected = [0.0, 0.195_121_951_219_512_2, 0.0, -0.292_682_926_829_268_3];
    for (actual, expected) in environment.state().iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-12);
    }
    assert_eq!(result.reward.to_bits(), 1.0_f64.to_bits());
    assert_eq!(result.status, EpisodeStatus::Continuing);
}

#[test]
fn all_cartpole_fixtures_match_internal_and_observation_precision() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gymnasium/transition_cases.json"))
            .expect("valid oracle fixtures");
    for case in fixtures["cases"]
        .as_array()
        .expect("case array")
        .iter()
        .filter(|case| case["registry_id"] == "CartPole-v1")
    {
        let state: [f64; 4] = serde_json::from_value(case["injection"]["internal_state"].clone())
            .expect("four state coordinates");
        let action = CartPoleAction::try_from(
            case["injection"]["action"]
                .as_u64()
                .expect("integer action") as usize,
        )
        .expect("valid action");
        let mut environment =
            CartPole::from_state(CartPoleState::try_from(state).expect("finite state"));
        let result = environment.step(action);
        let expected: [f64; 4] = serde_json::from_value(case["expected"]["internal_state"].clone())
            .expect("expected state");
        for (actual, expected) in environment.state().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        let observation: [f32; 4] = serde_json::from_value(case["expected"]["observation"].clone())
            .expect("expected observation");
        assert_eq!(result.observation, observation);
    }
}

#[test]
fn action_and_initial_state_boundaries_are_validated() {
    for (index, action) in [(0, CartPoleAction::Left), (1, CartPoleAction::Right)] {
        assert_eq!(
            CartPoleAction::try_from(index).expect("valid index"),
            action
        );
        assert_eq!(usize::from(action), index);
    }
    for invalid in [2, usize::MAX] {
        assert!(matches!(
            CartPoleAction::try_from(invalid),
            Err(bevy_gym::environments::InvalidCartPoleAction)
        ));
    }
    for coordinate in 0..4 {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut values = [0.0; 4];
            values[coordinate] = invalid;
            assert!(matches!(
                CartPoleState::try_from(values),
                Err(bevy_gym::environments::InvalidCartPoleState)
            ));
        }
    }
}

#[test]
fn inclusive_termination_limits_and_post_terminal_reward_match_gymnasium() {
    let angle_limit = 12.0 * 2.0 * std::f64::consts::PI / 360.0;
    for boundary in [
        [-2.4, 0.0, 0.0, 0.0],
        [2.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, -angle_limit, 0.0],
        [0.0, 0.0, angle_limit, 0.0],
    ] {
        let mut env =
            CartPole::from_state(CartPoleState::try_from(boundary).expect("finite boundary"));
        assert_eq!(
            env.step(CartPoleAction::Right).status,
            EpisodeStatus::Continuing
        );
    }
    for outside in [
        [-2.400_001, 0.0, 0.0, 0.0],
        [2.400_001, 0.0, 0.0, 0.0],
        [0.0, 0.0, -angle_limit - 0.000_001, 0.0],
        [0.0, 0.0, angle_limit + 0.000_001, 0.0],
    ] {
        let mut env =
            CartPole::from_state(CartPoleState::try_from(outside).expect("finite outside state"));
        let terminal = env.step(CartPoleAction::Right);
        assert_eq!(terminal.status, EpisodeStatus::Terminated);
        assert_eq!(terminal.reward.to_bits(), 1.0_f64.to_bits());
    }
}

#[test]
fn reset_reseeds_or_continues_within_official_bounds() {
    let mut env = CartPole::default();
    let first = env.reset(Some(42));
    assert_eq!(first, env.reset(Some(42)));
    let next = env.reset(None);
    assert_ne!(first, next);
    assert!(first
        .observation
        .iter()
        .chain(&next.observation)
        .all(|value| (-0.05..=0.05).contains(value)));
}

#[test]
fn repeated_terminal_steps_earn_zero_until_reset() {
    let mut env =
        CartPole::from_state(CartPoleState::try_from([3.0, 0.0, 0.0, 0.0]).expect("finite state"));
    assert_eq!(
        env.step(CartPoleAction::Right).reward.to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        env.step(CartPoleAction::Right).reward.to_bits(),
        0.0_f64.to_bits()
    );
    env.reset(Some(42));
    assert_eq!(
        env.step(CartPoleAction::Right).reward.to_bits(),
        1.0_f64.to_bits()
    );
}

#[test]
fn rejected_inputs_explain_the_valid_action_and_state() {
    let action_error = CartPoleAction::try_from(2).expect_err("invalid action");
    assert_eq!(action_error.to_string(), "CartPole action must be 0 or 1");
    let state_error = CartPoleState::try_from([f64::NAN; 4]).expect_err("invalid state");
    assert_eq!(
        state_error.to_string(),
        "CartPole initial state must contain four finite coordinates"
    );
}
