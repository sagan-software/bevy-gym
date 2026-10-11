//! Replay the selected fifteen-feature travel actor through native and browser physics.
#![cfg(all(
    feature = "robots",
    any(not(target_arch = "wasm32"), feature = "browser-training")
))]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[path = "../examples/robots/arrival/mod.rs"]
mod arrival;
#[path = "../examples/robots/learning/encoding.rs"]
mod encoding;
#[path = "../examples/robots/learning/model.rs"]
mod model;
#[path = "../examples/robots/skill_scene/session.rs"]
mod session;

/// Every reserved root must meet the frozen arrival gates without modifying weights.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn selected_actor_passes_all_reserved_arrivals() {
    let bytes = include_bytes!("../docs/progress/drone-arrival.mpk");
    let policy = arrival::load_policy(bytes.to_vec()).expect("selected actor");
    let before = policy.to_bytes().expect("frozen bytes");
    for offset in 1..=32 {
        let seed = u64::MAX - offset;
        let outcome = arrival::evaluate(&policy, seed).expect("valid inference");
        assert!(outcome.passed, "held-out root {seed}");
        assert_eq!(outcome.steps, 500);
        assert_eq!(outcome.stable_tail_actions, 100);
    }
    assert_eq!(policy.to_bytes().expect("unchanged bytes"), before);
}

/// Playback and evaluation use the same sampled task, features and private episode memory.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn playback_reproduces_frozen_actions_and_resets_memory() {
    use bevy_gym::{Env, TimeLimit};
    let bytes = include_bytes!("../docs/progress/drone-arrival.mpk");
    let policy = arrival::load_policy(bytes.to_vec()).expect("selected actor");
    let mut scene = arrival::load(bytes.to_vec(), 42).expect("playable arrival");
    let mut environment = TimeLimit::new(arrival::TravelTask::new(), 500).expect("horizon");
    let initial = environment.reset(Some(42)).observation;
    let mut observation = initial;
    let mut memory = policy.initial_memory();
    assert_eq!(scene.observation(), initial);
    assert_eq!(scene.horizon(), 500);
    for step in 1..=500 {
        let inferred = policy
            .mean_action(&arrival::encode(observation), &memory)
            .expect("inference");
        memory = inferred.next_memory;
        let action = encoding::decode_action(&inferred.action).expect("motors");
        let result = environment.step(action);
        observation = result.observation;
        scene.step();
        assert_eq!(scene.observation(), observation);
        assert_eq!(scene.last_action(), Some(action));
        assert_eq!(scene.status(), result.status);
        assert_eq!(scene.steps(), step);
    }
    scene.step();
    assert_eq!(scene.steps(), 500);
    scene.reset(42);
    assert_eq!(scene.observation(), initial);
    assert_eq!(scene.steps(), 0);
    assert!(scene.last_action().is_none());
    let mut fresh = arrival::load(bytes.to_vec(), 42).expect("fresh episode");
    scene.step();
    fresh.step();
    assert_eq!(scene.observation(), fresh.observation());
    assert_eq!(scene.last_action(), fresh.last_action());
    assert!(scene.error().is_none());
}

/// Missing, corrupt and older travel checkpoints cannot create a playable arrival.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_checkpoints_stop_before_session_creation() {
    for invalid in [
        Vec::new(),
        vec![0, 1, 2],
        include_bytes!("../docs/progress/drone-travel-trial.mpk").to_vec(),
    ] {
        assert!(arrival::load(invalid, 42).is_err());
    }
}

/// Failed control and incompatible actors cannot acquire a passing evaluation.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn untrained_and_incompatible_evaluations_fail() {
    use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoPolicy, SeedConfig};
    let config = model::learning_config();
    let agent = RecurrentPpoAgent::new(
        15,
        15,
        1,
        &[0.0; 4],
        &[1.0; 4],
        config.clone(),
        SeedConfig::from_root(0),
    )
    .expect("untrained fixture");
    let outcome = arrival::evaluate(&agent.policy(), 42).expect("finite untrained inference");
    assert!(!outcome.passed);
    assert!(outcome.steps < 500);
    let flight =
        model::load_policy(include_bytes!("../docs/progress/drone-curriculum.mpk").to_vec())
            .expect("wrong feature profile");
    assert!(arrival::evaluate(&flight, 42).is_err());
    let incompatible = RecurrentPpoPolicy::load_bytes(
        include_bytes!("../docs/progress/drone-arrival.mpk").to_vec(),
        15,
        15,
        1,
        &[2.0; 4],
        &[3.0; 4],
        &config,
    )
    .expect("wrong scaling");
    assert!(arrival::evaluate(&incompatible, 42).is_err());
}

/// A surviving failed parent must fail the complete final-window gate.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn surviving_parent_with_unstable_arrival_is_rejected() {
    let parent =
        arrival::load_policy(include_bytes!("../docs/progress/drone-arrival-parent.mpk").to_vec())
            .expect("failed parent fixture");
    let outcome = arrival::evaluate(&parent, u64::MAX).expect("valid failed evaluation");
    assert_eq!(outcome.steps, 500);
    assert!(outcome.stable_tail_actions > 0);
    assert!(outcome.stable_tail_actions < 100);
    assert!(!outcome.passed);
}
