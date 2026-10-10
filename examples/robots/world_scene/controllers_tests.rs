//! Private staging failures and per-robot recurrent-state isolation.

use super::*;
use bevy_gym::{
    robots::RobotWorld,
    training::{RecurrentPpoAgent, RecurrentPpoConfig, SeedConfig},
};

/// Load the retained RL policies without constructing a session or committing memory.
fn controllers() -> Controllers {
    Controllers::new(
        learning::load_policy(super::super::DRONE_REFERENCE.to_vec()).expect("hover RL"),
        standing::model::load_policy(
            include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk")
                .to_vec(),
        )
        .expect("standing RL candidate"),
    )
}

/// Isolated malformed-architecture fixture; these random weights never drive an example.
fn fixture(inputs: usize, outputs: usize, low: f32, high: f32) -> RecurrentPpoPolicy {
    RecurrentPpoAgent::new(
        inputs,
        inputs,
        1,
        &vec![low; outputs],
        &vec![high; outputs],
        RecurrentPpoConfig::default(),
        SeedConfig::from_root(42),
    )
    .expect("test architecture")
    .policy()
}

/// Changing one robot's memory cannot change another robot's inferred request or memory.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn recurrent_memory_is_local_and_staging_is_read_only() {
    let mut controllers = controllers();
    let snapshot = RobotWorld::default().snapshot().expect("common reset");
    let anchors = Anchors::new(&snapshot);
    let baseline = controllers.infer(&snapshot, &anchors).expect("baseline");
    controllers.drone_memory[0].hidden.fill(0.25);
    controllers.droid_memory[1].hidden.fill(0.25);
    let old_drones = controllers.drone_memory.clone();
    let old_droids = controllers.droid_memory.clone();
    let changed = controllers.infer(&snapshot, &anchors).expect("own memory");
    assert_eq!(controllers.drone_memory, old_drones);
    assert_eq!(controllers.droid_memory, old_droids);
    assert_ne!(changed.drones[0], baseline.drones[0]);
    assert_ne!(changed.droids[1], baseline.droids[1]);
    for slot in [RobotSlot::Second, RobotSlot::Third] {
        assert_eq!(
            element(&changed.drones, slot),
            element(&baseline.drones, slot)
        );
        assert_eq!(
            element(&changed.drone_memory, slot),
            element(&baseline.drone_memory, slot)
        );
    }
    for slot in [RobotSlot::First, RobotSlot::Third] {
        assert_eq!(
            element(&changed.droids, slot),
            element(&baseline.droids, slot)
        );
        assert_eq!(
            element(&changed.droid_memory, slot),
            element(&baseline.droid_memory, slot)
        );
    }
    controllers.commit(changed.drone_memory, changed.droid_memory);
    assert_ne!(controllers.drone_memory, old_drones);
    assert_ne!(controllers.droid_memory, old_droids);
    controllers.reset();
    assert_eq!(
        controllers.drone_memory,
        RobotSlot::ALL.map(|_| controllers.drone_policy.initial_memory())
    );
    assert_eq!(
        controllers.droid_memory,
        RobotSlot::ALL.map(|_| controllers.droid_policy.initial_memory())
    );
}

/// Every slot failure preserves all current memories, including staged earlier robots.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn later_inference_failure_commits_no_memory() {
    let snapshot = RobotWorld::default().snapshot().expect("common reset");
    let anchors = Anchors::new(&snapshot);
    for index in 0..3 {
        for drone in [true, false] {
            let mut controllers = controllers();
            let memories = if drone {
                &mut controllers.drone_memory
            } else {
                &mut controllers.droid_memory
            };
            memories[index].hidden.clear();
            let old_drones = controllers.drone_memory.clone();
            let old_droids = controllers.droid_memory.clone();
            let failure = controllers
                .infer(&snapshot, &anchors)
                .err()
                .expect("invalid memory");
            let slot = RobotSlot::ALL[index];
            let expected = if drone {
                RobotId::Drone(slot)
            } else {
                RobotId::Droid(slot)
            };
            assert!(matches!(failure, Failure::Inference { robot, .. } if robot == expected));
            assert_eq!(controllers.drone_memory, old_drones);
            assert_eq!(controllers.droid_memory, old_droids);
        }
    }
}

/// Width and bound failures from either team stop staging before a complete actuator frame.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn malformed_actor_actions_preserve_decoder_causes() {
    let snapshot = RobotWorld::default().snapshot().expect("common reset");
    let anchors = Anchors::new(&snapshot);
    for (drone, outputs, low, high) in [
        (true, 3, 0.0, 1.0),
        (true, 4, 2.0, 3.0),
        (false, 25, -1.0, 1.0),
        (false, 26, 2.0, 3.0),
    ] {
        let mut controllers = controllers();
        let inputs = if drone {
            12
        } else {
            standing::encoding::FEATURES
        };
        let policy = fixture(inputs, outputs, low, high);
        if drone {
            controllers.drone_memory = RobotSlot::ALL.map(|_| policy.initial_memory());
            *controllers.drone_policy = policy;
        } else {
            controllers.droid_memory = RobotSlot::ALL.map(|_| policy.initial_memory());
            *controllers.droid_policy = policy;
        }
        let failure = controllers
            .infer(&snapshot, &anchors)
            .err()
            .expect("invalid action");
        let message = failure.to_string();
        assert!(message.contains("First"));
        let source = std::error::Error::source(&failure).expect("original decoder error");
        assert!(message.contains(&source.to_string()));
        if drone {
            assert!(matches!(
                failure,
                Failure::DroneAction {
                    slot: RobotSlot::First,
                    ..
                }
            ));
            assert!(source
                .downcast_ref::<learning::encoding::ActionDecodeError>()
                .is_some());
        } else {
            assert!(matches!(
                failure,
                Failure::DroidAction {
                    slot: RobotSlot::First,
                    ..
                }
            ));
            assert!(source
                .downcast_ref::<standing::encoding::DecodeError>()
                .is_some());
        }
    }
}
