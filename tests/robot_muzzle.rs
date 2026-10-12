//! Muzzle geometry follows authoritative bodies and never chooses a target.
#![cfg(feature = "robots")]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use bevy::math::Vec3;
use bevy_gym::robots::{
    DroidAction, DroidBody, DroneAction, RobotActions, RobotId, RobotMuzzle, RobotRayTarget,
    RobotSlot, RobotWorld, RobotWorldError, RobotWorldFailure,
};

/// Body-fixed drone geometry excludes its own collider while tracing another drone.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn drone_muzzle_uses_physical_pose_and_current_frame() {
    let world = RobotWorld::default();
    let snapshot = world.snapshot().unwrap();
    let owner = RobotId::Drone(RobotSlot::Third);
    let muzzle = RobotMuzzle::try_from((&snapshot, owner)).unwrap();
    let body = snapshot.drone(RobotSlot::Third);
    assert_eq!(muzzle.owner(), owner);
    assert_eq!(muzzle.frame(), snapshot.frame());
    assert_eq!(
        muzzle.origin(),
        body.position() + body.orientation() * Vec3::new(0.0, 0.0, -0.317)
    );
    assert_eq!(muzzle.direction(), body.orientation() * Vec3::NEG_Z);
    assert_eq!(muzzle.range_metres().to_bits(), 36.0_f32.to_bits());
    let hit = world.cast_muzzle(&muzzle).unwrap().unwrap();
    assert_eq!(hit.target(), RobotRayTarget::Drone(RobotSlot::Second));
    assert_eq!(world.snapshot().unwrap().frame(), snapshot.frame());
}

/// The right forearm's actual longitudinal axis defines pistol direction in the bind pose.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn droid_muzzle_follows_right_forearm_without_target_selection() {
    let world = RobotWorld::default();
    let snapshot = world.snapshot().unwrap();
    let owner = RobotId::Droid(RobotSlot::First);
    let muzzle = RobotMuzzle::try_from((&snapshot, owner)).unwrap();
    let forearm = snapshot
        .droid(RobotSlot::First)
        .body(DroidBody::RightForearm);
    assert_eq!(muzzle.owner(), owner);
    assert_eq!(
        muzzle.origin(),
        forearm.position() + forearm.orientation() * Vec3::new(0.197, 0.0, 0.0)
    );
    assert_eq!(muzzle.direction(), forearm.orientation() * Vec3::X);
    assert_eq!(muzzle.range_metres().to_bits(), 30.0_f32.to_bits());
    assert!(world.cast_muzzle(&muzzle).unwrap().is_none());
}

/// Rejected stale, reset and foreign queries cannot stop or advance a valid world.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn muzzle_frames_reject_stale_reset_and_foreign_worlds_without_mutation() {
    let mut world = RobotWorld::default();
    let before = world.snapshot().unwrap();
    let muzzle = RobotMuzzle::try_from((&before, RobotId::Drone(RobotSlot::First))).unwrap();
    let foreign = RobotWorld::default();
    assert_eq!(
        foreign.cast_muzzle(&muzzle).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    foreign.snapshot().unwrap();
    let actions = RobotActions::new(
        &before,
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    world.advance(actions).unwrap();
    let current = world.snapshot().unwrap();
    assert_eq!(
        world.cast_muzzle(&muzzle).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(world.snapshot().unwrap().frame(), current.frame());
    let fresh = RobotMuzzle::try_from((&current, RobotId::Droid(RobotSlot::Second))).unwrap();
    world.cast_muzzle(&fresh).unwrap();
    world.reset().unwrap();
    assert_eq!(
        world.cast_muzzle(&fresh).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    world.snapshot().unwrap();
}

/// A stopped world's original cause takes precedence over a muzzle's source frame.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn stopped_world_preserves_its_original_failure_for_muzzle_queries() {
    let mut world = RobotWorld::default();
    let foreign = RobotWorld::default().snapshot().unwrap();
    let muzzle = RobotMuzzle::try_from((&foreign, RobotId::Drone(RobotSlot::First))).unwrap();
    let actions = RobotActions::new(
        &foreign,
        [DroneAction::try_from([0.0; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    assert_eq!(
        world.advance(actions).unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(
        world.cast_muzzle(&muzzle).unwrap_err(),
        RobotWorldError::Stopped(RobotWorldFailure::WrongFrame)
    );
}
