//! Shared-world physics through the public snapshot and actuator boundary.
#![cfg(feature = "robots")]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

use bevy::math::Vec3;
use bevy_gym::robots::{DroidBody, RobotSlot, RobotWorld};
use std::time::Duration;

/// Six physical snapshots retain the original rotor body and complete articulated geometry.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn shared_world_contains_three_drones_and_three_complete_droids() {
    let world = RobotWorld::default();
    let snapshot = world.snapshot().expect("ready physical world");
    assert_eq!(snapshot.elapsed(), Duration::ZERO);
    for (slot, z) in RobotSlot::ALL.into_iter().zip([-2.0, 0.0, 2.0]) {
        assert!(
            snapshot
                .drone(slot)
                .position()
                .distance(Vec3::new(-3.0, 2.0, z))
                < 1e-6
        );
        let droid = snapshot.droid(slot);
        assert!(
            droid
                .body(DroidBody::Pelvis)
                .position()
                .distance(Vec3::new(3.0, 1.02, z + 0.028))
                < 1e-6
        );
        assert!(
            droid
                .body(DroidBody::Torso)
                .position()
                .distance(Vec3::new(3.0, 1.28, z + 0.008))
                < 1e-6
        );
        assert!(
            droid
                .body(DroidBody::Head)
                .position()
                .distance(Vec3::new(3.0, 1.61, z))
                < 1e-6
        );
        for segment in DroidBody::ALL {
            let body = droid.body(segment);
            assert!(body.position().is_finite());
            assert!(body.orientation().is_finite());
            assert!(body.linear_velocity().is_finite());
            assert!(body.angular_velocity().is_finite());
        }
    }
}

/// Deterministic actuator fixtures exercise physics only; they are not agent controllers.
fn fixture(snapshot: &bevy_gym::robots::RobotSnapshot) -> bevy_gym::robots::RobotActions {
    use bevy_gym::robots::{DroidAction, DroneAction, RobotActions};
    RobotActions::new(
        snapshot,
        [DroneAction::try_from([0.5; 4]).expect("finite thrust fixture"); 3],
        [DroidAction::try_from([0.0; 26]).expect("finite torque fixture"); 3],
    )
}

/// All bodies advance at one common boundary, and a retained source becomes stale.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn complete_actions_advance_once_and_stale_actions_stop_before_physics() {
    use bevy_gym::robots::{RobotWorldError, RobotWorldFailure};
    let mut world = RobotWorld::default();
    let before = world.snapshot().unwrap();
    let requests = fixture(&before);
    let after = world.advance(requests.clone()).unwrap();
    assert_eq!(after.elapsed(), Duration::from_millis(20));
    assert_ne!(before.frame(), after.frame());
    for slot in RobotSlot::ALL {
        assert!(
            after
                .drone(slot)
                .position()
                .distance(before.drone(slot).position())
                < 0.001
        );
        assert!(
            after.droid(slot).body(DroidBody::Pelvis).position().y
                < before.droid(slot).body(DroidBody::Pelvis).position().y
        );
    }
    assert_eq!(
        world.advance(requests).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(
        world.snapshot().unwrap_err(),
        RobotWorldError::Stopped(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(
        world.advance(fixture(&after)).unwrap_err(),
        RobotWorldError::Stopped(RobotWorldFailure::WrongFrame)
    );
    let reset = world.reset().unwrap();
    assert_eq!(reset.elapsed(), Duration::ZERO);
    assert_ne!(reset.frame(), before.frame());
    assert_eq!(
        world.advance(fixture(&before)).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
}

/// A world move retains its scope; another world cannot accept its source snapshot.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frame_identity_survives_move_and_rejects_another_world() {
    use bevy_gym::robots::{RobotWorldError, RobotWorldFailure};
    let world = RobotWorld::default();
    let source = world.snapshot().unwrap();
    let mut moved = world;
    assert_eq!(
        moved.advance(fixture(&source)).unwrap().elapsed(),
        Duration::from_millis(20)
    );
    let mut other = RobotWorld::default();
    assert_ne!(other.snapshot().unwrap().frame(), source.frame());
    assert_eq!(
        other.advance(fixture(&source)).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
}

/// Closed identity variants derive their team and retain every legal slot.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn identities_derive_teams_and_clock_crosses_one_second() {
    use bevy_gym::robots::{RobotId, RobotTeam};
    for slot in RobotSlot::ALL {
        assert_eq!(RobotId::Drone(slot).team(), RobotTeam::Drones);
        assert_eq!(RobotId::Droid(slot).team(), RobotTeam::Droids);
        assert_eq!(RobotId::Drone(slot).slot(), slot);
        assert_eq!(RobotId::Droid(slot).slot(), slot);
    }
    let mut world = RobotWorld::default();
    let mut snapshot = world.snapshot().unwrap();
    for _ in 0..51 {
        snapshot = world.advance(fixture(&snapshot)).unwrap();
    }
    assert_eq!(snapshot.elapsed(), Duration::from_millis(1020));
}

/// Rejections retain their diagnostic source and distinguish a stopped world.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn errors_preserve_each_failure_source() {
    use bevy_gym::robots::{RobotId, RobotWorldError, RobotWorldFailure};
    use std::error::Error;
    for cause in [
        RobotWorldFailure::WrongFrame,
        RobotWorldFailure::ClockExhausted,
        RobotWorldFailure::NonFinite(RobotId::Droid(RobotSlot::Third)),
    ] {
        assert!(!cause.to_string().is_empty());
        for error in [
            RobotWorldError::Rejected(cause),
            RobotWorldError::Stopped(cause),
        ] {
            assert_eq!(
                error.source().unwrap().downcast_ref::<RobotWorldFailure>(),
                Some(&cause)
            );
            assert!(error.to_string().contains(&cause.to_string()));
        }
    }
    assert!(format!("{:?}", RobotWorld::default()).starts_with("RobotWorld"));
}

/// Different motor fixtures expose repeated, missing or interleaved common solver steps.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn all_motor_patterns_share_one_physical_interval() {
    use bevy_gym::robots::{DroidAction, DroneAction, RobotActions};
    let mut world = RobotWorld::default();
    let before = world.snapshot().expect("source frame");
    let drones = [0.0, 0.5, 1.0]
        .map(|fraction| DroneAction::try_from([fraction; 4]).expect("bounded motor fixture"));
    let droids = [DroidAction::try_from([0.0; 26]).expect("zero torque fixture"); 3];
    let after = world
        .advance(RobotActions::new(&before, drones, droids))
        .expect("physical frame");
    let elapsed = Duration::from_millis(20);
    assert_eq!(after.elapsed(), elapsed);
    // Acceleration is 9.81 m/s²; elapsed converts milliseconds to seconds; speed is m/s.
    let speed = 9.81_f32 * elapsed.as_secs_f32();
    for (slot, direction) in RobotSlot::ALL.into_iter().zip([-1.0_f32, 0.0, 1.0]) {
        let error = direction.mul_add(-speed, after.drone(slot).linear_velocity().y);
        assert!(error.abs() < 1e-5);
    }
    assert!(
        after.drone(RobotSlot::First).position().y < before.drone(RobotSlot::First).position().y
    );
    assert!(
        after.drone(RobotSlot::Third).position().y > before.drone(RobotSlot::Third).position().y
    );
}
