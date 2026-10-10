//! Identical-input native/WASM action parity for all six retained recurrent actors.

use super::{DROID, DRONE, RECORD};
use crate::{learning, standing};
use bevy_gym::robots::RobotSlot;
use serde::{Deserialize, Serialize};

/// Native inputs and corresponding requests at one common physical boundary.
#[derive(Deserialize, Serialize)]
struct Frame {
    /// Twelve own-body hover inputs per drone, in closed slot order.
    drone_inputs: [Vec<f32>; 3],
    /// Two hundred and four own-body standing inputs per droid.
    droid_inputs: [Vec<f32>; 3],
    /// Four validated rotor fractions per drone.
    drone_actions: [Vec<f32>; 3],
    /// Twenty-six validated torque fractions per droid.
    droid_actions: [Vec<f32>; 3],
}

/// Read the complete native fixture without relying on platform-specific physical drift.
fn reference() -> Vec<Frame> {
    let frames: Vec<Frame> = serde_json::from_str(include_str!(
        "../../docs/progress/robot-world-native-actions.json"
    ))
    .expect("recorded finite native trace");
    assert_eq!(frames.len(), 128, "2.56 s native recurrent sequence");
    frames
}

/// Compare actor actions on identical input sequences, with distinct memory per robot.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn all_six_actors_match_native_actions_on_identical_inputs() {
    let drone = learning::load_policy(DRONE.to_vec()).expect("same hover bytes");
    let (droid, _) =
        standing::checkpoint::from_bytes(DROID.to_vec(), RECORD).expect("same standing bytes");
    let mut drone_memory = RobotSlot::ALL.map(|_| drone.initial_memory());
    let mut droid_memory = RobotSlot::ALL.map(|_| droid.initial_memory());
    for frame in reference() {
        for index in 0..3 {
            let inferred = drone
                .mean_action(&frame.drone_inputs[index], &drone_memory[index])
                .expect("native own-drone input");
            drone_memory[index] = inferred.next_memory;
            assert_actions(&inferred.action, &frame.drone_actions[index], 4);
            let inferred = droid
                .mean_action(&frame.droid_inputs[index], &droid_memory[index])
                .expect("native own-droid input");
            droid_memory[index] = inferred.next_memory;
            assert_actions(&inferred.action, &frame.droid_actions[index], 26);
        }
    }
}

/// The unitless action-fraction tolerance is fixed at 0.00001 for both teams.
fn assert_actions(actual: &[f32], expected: &[f32], width: usize) {
    assert_eq!(actual.len(), width);
    assert_eq!(expected.len(), width);
    for (&actual, &expected) in actual.iter().zip(expected) {
        let difference = (actual - expected).abs();
        assert!(
            difference <= 0.000_01,
            "action-fraction difference {difference}"
        );
    }
}

/// Capture mode emits evidence; ordinary mode checks that the source preserves that evidence.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn native_shared_clip_matches_recorded_inputs_and_actions() {
    let frames = capture_native();
    if std::env::var_os("ROBOT_WORLD_CAPTURE_NATIVE").is_some() {
        let capture = serde_json::to_string(&frames).expect("finite native actor trace");
        println!("ROBOT_WORLD_NATIVE_ACTIONS={capture}");
        return;
    }
    let expected = reference();
    assert_eq!(
        serde_json::to_value(&frames).expect("current trace"),
        serde_json::to_value(&expected).expect("frozen trace")
    );
}

/// Record inputs before inference and applied requests after the common physical step.
#[cfg(not(target_arch = "wasm32"))]
fn capture_native() -> Vec<Frame> {
    use bevy_gym::robots::DroidActuator;
    let mut session =
        standing::world_session::Session::load(DRONE.to_vec(), DROID.to_vec(), RECORD)
            .expect("retained RL candidates");
    let mut frames = Vec::with_capacity(128);
    for _ in 0..128 {
        let drone_inputs =
            RobotSlot::ALL.map(|slot| super::drone_input(session.snapshot(), slot).to_vec());
        let droid_inputs =
            RobotSlot::ALL.map(|slot| super::droid_input(session.snapshot(), slot).to_vec());
        session.step();
        assert!(session.error().is_none(), "native inference stays usable");
        let drone_actions = RobotSlot::ALL.map(|slot| {
            let action = session
                .last_drone_action(slot)
                .expect("applied rotor requests");
            action.fractions().to_vec()
        });
        let droid_actions = RobotSlot::ALL.map(|slot| {
            let action = session
                .last_droid_action(slot)
                .expect("applied joint requests");
            DroidActuator::ALL
                .into_iter()
                .map(|axis| action.fraction(axis))
                .collect()
        });
        frames.push(Frame {
            drone_inputs,
            droid_inputs,
            drone_actions,
            droid_actions,
        });
    }
    frames
}
