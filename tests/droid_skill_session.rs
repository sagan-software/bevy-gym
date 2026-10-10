//! Frozen standing playback must match direct policy inference and physical state.
#![cfg(all(
    feature = "robots",
    any(not(target_arch = "wasm32"), feature = "browser-training")
))]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[expect(
    unused_imports,
    dead_code,
    reason = "Only the shared standing architecture is needed here."
)]
#[path = "../examples/robots/learning/mod.rs"]
mod learning;
#[path = "../examples/robots/standing_scene/session.rs"]
mod session;
#[expect(dead_code, reason = "The native CLI has separate executable tests.")]
#[path = "../examples/robots/standing/mod.rs"]
mod standing;

use bevy_gym::{robots::DroidStanding, Env, TimeLimit};

/// Preserved PPO update 100; failed selection, never qualified standing evidence.
const CHECKPOINT: &[u8] = include_bytes!("../docs/progress/droid-standing-trial.mpk");
/// Exact byte identity and recorded PPO counters for this candidate.
const RECORD: &[u8] = include_bytes!("../docs/progress/droid-standing-trial.json");

/// Every rendered episode state must follow the same frozen action and physical step.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn standing_playback_matches_direct_inference_and_freezes_at_terminal() {
    let mut scene = session::load(CHECKPOINT.to_vec(), RECORD, 42).expect("RL candidate");
    let (policy, _) = standing::checkpoint::from_bytes(CHECKPOINT.to_vec(), RECORD)
        .expect("same frozen candidate");
    let mut memory = policy.initial_memory();
    let mut environment = TimeLimit::new(DroidStanding::default(), 1_000).expect("horizon");
    let mut observation = environment.reset(Some(42)).observation;
    assert_eq!(scene.observation(), observation);
    assert_eq!(scene.horizon(), 1_000);
    assert_eq!(scene.checkpoint_update().get(), 100);
    assert_eq!(scene.steps(), 0);
    for step in 1..=1_000 {
        let inferred = policy
            .mean_action(&standing::encoding::encode(&observation), &memory)
            .expect("valid inference");
        memory = inferred.next_memory;
        let action = standing::encoding::decode(&inferred.action).expect("joint torques");
        let result = environment.step(action);
        observation = result.observation;
        scene.step();
        assert_eq!(scene.observation(), observation);
        assert_eq!(scene.last_action(), Some(action));
        assert_eq!(scene.status(), result.status);
        assert_eq!(scene.steps(), step);
        if result.status.is_done() {
            break;
        }
    }
    assert!(scene.status().is_done());
    assert!(scene.error().is_none());
    let steps = scene.steps();
    scene.step();
    assert_eq!(scene.steps(), steps);
    assert_eq!(scene.observation(), observation);
    scene.reset(42);
    let mut fresh = session::load(CHECKPOINT.to_vec(), RECORD, 42).expect("fresh episode");
    assert_eq!(scene.observation(), fresh.observation());
    assert!(scene.last_action().is_none());
    scene.step();
    fresh.step();
    assert_eq!(scene.last_action(), fresh.last_action());
    assert_eq!(scene.observation(), fresh.observation());
}

/// Missing, changed or incompatible policy identity cannot construct a playable scene.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_standing_checkpoint_has_no_playback() {
    assert!(session::load(Vec::new(), RECORD, 42).is_err());
    assert!(session::load(CHECKPOINT.to_vec(), b"{}", 42).is_err());
    assert!(session::load(vec![0, 1, 2], RECORD, 42).is_err());
}

/// Display counters follow validated metadata claims without establishing qualification.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn standing_display_uses_the_validated_update_count() {
    let mut record: serde_json::Value = serde_json::from_slice(RECORD).expect("record");
    *record.get_mut("update").expect("update") = serde_json::json!(101);
    *record.get_mut("transitions").expect("transitions") = serde_json::json!(101 * 512);
    let metadata = serde_json::to_vec(&record).expect("fixture claims");
    let scene = session::load(CHECKPOINT.to_vec(), &metadata, 42).expect("consistent fixture");
    assert_eq!(scene.checkpoint_update().get(), 101);
}

/// Native playback retains the captured physical trajectory of the failed candidate.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn standing_native_playback_matches_the_candidate_trace() {
    use bevy_gym::robots::DroidActuator;
    let mut scene = session::load(CHECKPOINT.to_vec(), RECORD, 42).expect("RL candidate");
    let mut frames = Vec::new();
    for _ in 0..scene.horizon() {
        let input = standing::encoding::encode(&scene.observation());
        scene.step();
        assert!(
            scene.error().is_none(),
            "frozen inference must remain usable"
        );
        let action = scene.last_action().expect("one learned action");
        let torques: Vec<_> = DroidActuator::ALL
            .into_iter()
            .map(|axis| action.fraction(axis))
            .collect();
        let features = standing::encoding::encode(&scene.observation());
        let status = scene.status();
        frames.push(serde_json::json!({
            "action": torques,
            "input": input.as_slice(),
            "features": features.as_slice(),
            "status": format!("{status:?}"),
        }));
        if scene.status().is_done() {
            break;
        }
    }
    assert!(
        scene.status().is_done(),
        "episode must stop at its physical limit"
    );
    // Explicit native capture mode produces evidence; it does not validate a reference trace.
    #[cfg(not(target_arch = "wasm32"))]
    if std::env::var_os("STANDING_CAPTURE_GOLDEN").is_some() {
        let capture = serde_json::to_string(&frames).expect("finite native trace");
        println!("STANDING_NATIVE_TRACE={capture}");
        return;
    }
    let expected: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../docs/progress/droid-standing-native-trace.json"
    ))
    .expect("preserved native trace");
    assert_eq!(frames.len(), expected.len(), "terminal action count");
    for (actual, expected) in frames.iter().zip(&expected) {
        assert_eq!(actual.get("status"), expected.get("status"));
        for (key, tolerance) in [
            ("action", 0.000_01),
            ("input", 0.000_2),
            ("features", 0.000_2),
        ] {
            let actual = actual
                .get(key)
                .expect("trace field")
                .as_array()
                .expect("vector");
            let expected = expected
                .get(key)
                .expect("trace field")
                .as_array()
                .expect("vector");
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                let difference = (actual.as_f64().expect("finite value")
                    - expected.as_f64().expect("finite value"))
                .abs();
                assert!(
                    difference <= tolerance,
                    "{key} native/browser difference {difference}"
                );
            }
        }
    }
}

/// Cross-platform policy parity uses identical observations and separate recurrent memory.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn standing_browser_policy_matches_native_actions_on_identical_inputs() {
    let expected: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../docs/progress/droid-standing-native-trace.json"
    ))
    .expect("preserved native trace");
    let (policy, _) = standing::checkpoint::from_bytes(CHECKPOINT.to_vec(), RECORD)
        .expect("same frozen candidate");
    let mut memory = policy.initial_memory();
    assert_eq!(expected.len(), 62, "complete native recurrent episode");
    for (frame, expected) in expected.iter().enumerate() {
        let input: Vec<f32> = expected["input"]
            .as_array()
            .expect("actor input")
            .iter()
            .map(|value| value.as_f64().expect("finite input") as f32)
            .collect();
        assert_eq!(input.len(), standing::encoding::FEATURES);
        let inferred = policy
            .mean_action(&input, &memory)
            .expect("reference inference");
        memory = inferred.next_memory;
        let actions = expected["action"].as_array().expect("torques");
        assert_eq!(inferred.action.len(), actions.len());
        assert_eq!(actions.len(), 26);
        for (axis, (actual, expected)) in inferred.action.iter().zip(actions).enumerate() {
            let difference = (f64::from(*actual) - expected.as_f64().expect("finite torque")).abs();
            assert!(
                difference <= 0.000_01,
                "frame {frame} axis {axis} action difference {difference}"
            );
        }
    }
}
