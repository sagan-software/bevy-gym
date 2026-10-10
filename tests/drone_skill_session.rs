//! The standalone scene uses the same frozen actions and physics as evaluation.
#![cfg(all(
    feature = "robots",
    any(not(target_arch = "wasm32"), feature = "browser-training")
))]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[path = "../examples/robots/learning/encoding.rs"]
mod encoding;
#[path = "../examples/robots/learning/model.rs"]
mod model;
#[path = "../examples/robots/skill_scene/session.rs"]
mod session;

use bevy_gym::robots::{DroneHover, DroneObservation};
use bevy_gym::{Env, TimeLimit};
use session::Session;

/// Recorded weights are identical in standalone inference and scene playback.
const CHECKPOINT: &[u8] = include_bytes!("../docs/progress/drone-curriculum.mpk");

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn playback_matches_direct_policy_and_reset_clears_memory() {
    for factory in [DroneHover::default, DroneHover::disturbed] {
        assert_playback(factory);
    }
}

/// Compare a full episode and reset against direct frozen inference.
fn assert_playback(factory: fn() -> DroneHover) {
    let mut scene = Session::load(CHECKPOINT.to_vec(), factory, 42).expect("valid checkpoint");
    let policy = model::load_policy(CHECKPOINT.to_vec()).expect("same checkpoint");
    let mut memory = policy.initial_memory();
    let mut environment = TimeLimit::new(factory(), 500).expect("positive limit");
    let initial = environment.reset(Some(42)).observation;
    let mut observation = initial;
    assert_eq!(scene.observation(), initial);
    assert_eq!(scene.steps(), 0);
    assert_eq!(scene.horizon(), 500);
    assert!(scene.last_action().is_none());
    for step in 1..=500 {
        let inferred = policy
            .mean_action(&encoding::encode(observation), &memory)
            .expect("valid inference");
        memory = inferred.next_memory;
        let action = encoding::decode_action(&inferred.action).expect("valid motors");
        let result = environment.step(action);
        observation = result.observation;
        scene.step();
        assert_eq!(scene.observation(), observation);
        assert_eq!(scene.last_action(), Some(action));
        assert_eq!(scene.status(), result.status);
        assert_eq!(scene.steps(), step);
    }
    scene.step();
    assert_eq!(scene.steps(), 500, "completed episodes cannot advance");
    assert!(scene.error().is_none());
    assert_reset(&mut scene, factory, initial);
}

/// The same seed restores physical state and clears the actor's recurrent memory.
fn assert_reset(scene: &mut Session, factory: fn() -> DroneHover, initial: DroneObservation) {
    scene.reset(42);
    assert_eq!(scene.observation(), initial);
    assert_eq!(scene.steps(), 0);
    assert!(scene.last_action().is_none());
    let mut fresh = Session::load(CHECKPOINT.to_vec(), factory, 42).expect("fresh scene");
    scene.step();
    fresh.step();
    assert_eq!(scene.last_action(), fresh.last_action());
    assert_eq!(scene.observation(), fresh.observation());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_checkpoints_never_create_a_playable_session() {
    for bytes in [
        Vec::new(),
        vec![0, 1, 2],
        include_bytes!("../assets/robots/tracking.mpk").to_vec(),
    ] {
        assert!(Session::load(bytes, DroneHover::default, 42).is_err());
    }
}

#[path = "../examples/robots/skill_scene/travel.rs"]
mod travel;

/// The unqualified trial scene must reproduce the training environment and frozen policy exactly.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn travel_scene_replays_the_shared_task_without_controller_substitutes() {
    let bytes = include_bytes!("../docs/progress/drone-travel-trial.mpk");
    let mut scene = travel::load(bytes.to_vec(), 42).expect("RL trial checkpoint");
    let policy = travel::load_policy(bytes.to_vec()).expect("same frozen policy");
    let mut memory = policy.initial_memory();
    assert_eq!(scene.horizon(), 1_000);
    let mut environment = TimeLimit::new(
        travel::environment::TravelTask::new(travel::stage::Stage::Near),
        1_000,
    )
    .expect("travel horizon");
    let initial = environment.reset(Some(42)).observation;
    let mut observation = initial;
    assert_eq!(scene.observation(), initial);
    for step in 1..=1_000 {
        let inferred = policy
            .mean_action(&travel::encoding::encode(observation), &memory)
            .expect("frozen inference");
        memory = inferred.next_memory;
        let action = encoding::decode_action(&inferred.action).expect("policy motors");
        let result = environment.step(action);
        observation = result.observation;
        scene.step();
        assert_eq!(scene.observation(), observation);
        assert_eq!(scene.last_action(), Some(action));
        assert_eq!(scene.status(), result.status);
        assert_eq!(scene.steps(), step);
    }
    scene.step();
    assert_eq!(scene.steps(), 1_000);
    scene.reset(42);
    assert_eq!(scene.observation(), initial);
    let mut fresh = travel::load(bytes.to_vec(), 42).expect("fresh episode");
    scene.step();
    fresh.step();
    assert_eq!(scene.observation(), fresh.observation());
    assert_eq!(scene.last_action(), fresh.last_action());
    assert!(travel::load(CHECKPOINT.to_vec(), 42).is_err());
}
