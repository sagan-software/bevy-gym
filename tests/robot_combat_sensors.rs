//! Combat sensor queries remain read-only and obey match lifecycle capabilities.
#![cfg(feature = "robots")]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use bevy::math::Vec3;
use bevy_gym::{
    robots::{
        DroidAction, DroneAction, RobotActions, RobotBattle, RobotId, RobotMatch, RobotMatchError,
        RobotMatchResult, RobotRay, RobotRayTarget, RobotSlot,
    },
    training::CombatSignal,
};
use std::time::Duration;

/// A robot can query physical sensors without advancing time or selecting an action.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn combat_ray_queries_preserve_the_common_decision_boundary() {
    let battle = RobotBattle::default();
    let before = battle.snapshot().unwrap();
    let ray = RobotRay::try_from((Vec3::new(-3.0, 2.0, 5.0), Vec3::NEG_Z, 10.0)).unwrap();
    let hit = battle.cast_ray(&ray, None).unwrap().unwrap();
    assert_eq!(hit.target(), RobotRayTarget::Drone(RobotSlot::Third));
    let excluded = battle
        .cast_ray(&ray, Some(RobotId::Drone(RobotSlot::Third)))
        .unwrap()
        .unwrap();
    assert_eq!(excluded.target(), RobotRayTarget::Drone(RobotSlot::Second));
    let empty = RobotRay::try_from((Vec3::new(20.0, 2.0, 20.0), Vec3::Y, 1.0)).unwrap();
    assert!(battle.cast_ray(&empty, None).unwrap().is_none());
    let after = battle.snapshot().unwrap();
    assert_eq!(after.physical().frame(), before.physical().frame());
    assert_eq!(after.physical().elapsed(), before.physical().elapsed());
    for slot in RobotSlot::ALL {
        for owner in [RobotId::Drone(slot), RobotId::Droid(slot)] {
            assert_eq!(
                after.health(owner).body_hits_remaining(),
                before.health(owner).body_hits_remaining()
            );
            assert_eq!(
                after.health(owner).is_alive(),
                before.health(owner).is_alive()
            );
            assert_eq!(after.weapon_phase(owner), before.weapon_phase(owner));
        }
    }
}

/// Terminal matches cannot query a retired solver; reset restores sensor capability.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn finished_matches_retire_sensor_queries_until_reset() {
    let mut game = RobotMatch::try_from(Duration::from_millis(20)).unwrap();
    let ray = RobotRay::try_from((Vec3::new(-3.0, 2.0, 5.0), Vec3::NEG_Z, 10.0)).unwrap();
    assert!(game.cast_ray(&ray, None).unwrap().is_some());
    let before = game.snapshot().unwrap();
    let actions = RobotActions::new(
        before.physical(),
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::default(); 3],
    );
    game.advance(
        actions,
        [CombatSignal::IdleReleased; 3],
        [CombatSignal::IdleReleased; 3],
    )
    .unwrap();
    assert_eq!(
        game.cast_ray(&ray, None),
        Err(RobotMatchError::Finished(RobotMatchResult::Timeout))
    );
    game.reset();
    assert!(game.cast_ray(&ray, None).unwrap().is_some());
}

/// Failed combat cannot expose a mutable or query-capable retired backend.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn failed_combat_sensor_queries_retain_the_original_failure() {
    use bevy_gym::robots::{RobotWorld, RobotWorldError, RobotWorldFailure};
    let ray = RobotRay::try_from((Vec3::ZERO, Vec3::Y, 1.0)).unwrap();
    let foreign = RobotWorld::default().snapshot().unwrap();
    let actions = RobotActions::new(
        &foreign,
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::default(); 3],
    );
    let mut battle = RobotBattle::default();
    assert_eq!(
        battle
            .advance(
                actions.clone(),
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3]
            )
            .unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(
        battle.cast_ray(&ray, None),
        Err(RobotWorldError::Stopped(RobotWorldFailure::WrongFrame))
    );
    let mut game = RobotMatch::try_from(Duration::from_secs(1)).unwrap();
    assert_eq!(
        game.advance(
            actions,
            [CombatSignal::IdleReleased; 3],
            [CombatSignal::IdleReleased; 3]
        )
        .unwrap_err(),
        RobotMatchError::Failed(RobotWorldError::Rejected(RobotWorldFailure::WrongFrame))
    );
    assert_eq!(
        game.cast_ray(&ray, None),
        Err(RobotMatchError::Failed(RobotWorldError::Rejected(
            RobotWorldFailure::WrongFrame
        )))
    );
}
