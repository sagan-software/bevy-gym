//! Shared combat advances mechanics only from complete policy requests.
#![cfg(feature = "robots")]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use bevy_gym::{
    robots::{
        DroidAction, DroneAction, RobotActions, RobotBattle, RobotId, RobotSlot, RobotWeaponPhase,
        RobotWorld, RobotWorldError, RobotWorldFailure,
    },
    training::CombatSignal,
};

/// Released triggers advance common physics without firing or choosing aim.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn released_triggers_leave_weapons_ready_and_health_intact() {
    let mut battle = RobotBattle::default();
    let before = battle.snapshot().unwrap();
    let actions = RobotActions::new(
        before.physical(),
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    let after = battle
        .advance(
            actions,
            [CombatSignal::IdleReleased; 3],
            [CombatSignal::IdleReleased; 3],
        )
        .unwrap();
    assert_ne!(after.physical().frame(), before.physical().frame());
    for slot in RobotSlot::ALL {
        for owner in [RobotId::Drone(slot), RobotId::Droid(slot)] {
            assert_eq!(after.weapon_phase(owner), RobotWeaponPhase::Ready);
            assert_eq!(
                after.health(owner).body_hits_remaining(),
                before.health(owner).body_hits_remaining()
            );
        }
    }
}

/// Input rejection precedes weapon updates and retains the original world failure.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn foreign_actions_cannot_consume_ammunition_or_start_charge() {
    let mut battle = RobotBattle::default();
    let foreign = RobotWorld::default().snapshot().unwrap();
    let actions = RobotActions::new(
        &foreign,
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    let error = battle
        .advance(
            actions.clone(),
            [CombatSignal::PursuitRequested; 3],
            [CombatSignal::PursuitRequested; 3],
        )
        .unwrap_err();
    assert_eq!(
        error,
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(
        battle.snapshot().unwrap_err(),
        RobotWorldError::Stopped(RobotWorldFailure::WrongFrame)
    );
    assert_eq!(
        battle
            .advance(
                actions,
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3]
            )
            .unwrap_err(),
        RobotWorldError::Stopped(RobotWorldFailure::WrongFrame)
    );
}

/// A charged drone round travels before a physical teammate collision reduces body health.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn drone_round_travels_before_one_body_hit() {
    let mut battle = RobotBattle::default();
    for step in 1..=44 {
        let before = battle.snapshot().unwrap();
        let actions = RobotActions::new(
            before.physical(),
            [DroneAction::try_from([0.5; 4]).unwrap(); 3],
            [DroidAction::try_from([0.0; 26]).unwrap(); 3],
        );
        let after = battle
            .advance(
                actions,
                [
                    CombatSignal::IdleReleased,
                    CombatSignal::IdleReleased,
                    CombatSignal::PursuitRequested,
                ],
                [CombatSignal::IdleReleased; 3],
            )
            .unwrap();
        let expected = if step < 44 { 6 } else { 5 };
        assert_eq!(
            after
                .health(RobotId::Drone(RobotSlot::Second))
                .body_hits_remaining()
                .get(),
            expected,
            "decision {step}"
        );
        assert_eq!(
            after
                .health(RobotId::Drone(RobotSlot::Third))
                .body_hits_remaining()
                .get(),
            6
        );
    }
}

/// A requested pistol shot starts its actual cooldown without aiming at a target.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pistol_request_starts_cooldown_from_the_physical_forearm() {
    let mut battle = RobotBattle::default();
    let before = battle.snapshot().unwrap();
    let actions = RobotActions::new(
        before.physical(),
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    let after = battle
        .advance(
            actions,
            [CombatSignal::IdleReleased; 3],
            [CombatSignal::SearchRequested; 3],
        )
        .unwrap();
    for slot in RobotSlot::ALL {
        assert_eq!(
            after.weapon_phase(RobotId::Droid(slot)),
            RobotWeaponPhase::Cooling {
                remaining: std::time::Duration::from_millis(250)
            }
        );
    }
}

/// Reset rebuilds mechanical state and invalidates retained physical requests.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reset_restores_weapons_health_and_a_fresh_source_frame() {
    let mut battle = RobotBattle::default();
    let before = battle.snapshot().unwrap();
    let stale = RobotActions::new(
        before.physical(),
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    battle
        .advance(
            stale.clone(),
            [CombatSignal::PursuitRequested; 3],
            [CombatSignal::PursuitRequested; 3],
        )
        .unwrap();
    let reset = battle.reset().unwrap();
    assert_ne!(reset.physical().frame(), before.physical().frame());
    for slot in RobotSlot::ALL {
        for owner in [RobotId::Drone(slot), RobotId::Droid(slot)] {
            assert_eq!(reset.weapon_phase(owner), RobotWeaponPhase::Ready);
            assert!(reset.health(owner).is_alive());
        }
    }
    assert_eq!(
        battle
            .advance(
                stale,
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3]
            )
            .unwrap_err(),
        RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
    );
}

/// Mechanical shutdown uses a valid zero-torque value on every named joint actuator.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn default_droid_action_requests_zero_on_every_actuator() {
    let zero = DroidAction::default();
    for actuator in bevy_gym::robots::DroidActuator::ALL {
        assert_eq!(zero.fraction(actuator).to_bits(), 0.0_f32.to_bits());
    }
}

/// Shared fatal-contact sensors read actual collisions without changing the physical boundary.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fatal_contacts_follow_solver_contacts_without_advancing_queries() {
    let mut world = RobotWorld::default();
    let before = world.snapshot().unwrap();
    for slot in RobotSlot::ALL {
        assert!(!world.fatal_contact(RobotId::Drone(slot)).unwrap());
        assert!(!world.fatal_contact(RobotId::Droid(slot)).unwrap());
    }
    assert_eq!(world.snapshot().unwrap().frame(), before.frame());
    for _ in 0..100 {
        let snapshot = world.snapshot().unwrap();
        let actions = RobotActions::new(
            &snapshot,
            [DroneAction::try_from([0.0; 4]).unwrap(); 3],
            [DroidAction::default(); 3],
        );
        world.advance(actions).unwrap();
    }
    let after = world.snapshot().unwrap();
    for slot in RobotSlot::ALL {
        assert!(world.fatal_contact(RobotId::Drone(slot)).unwrap());
    }
    assert_eq!(world.snapshot().unwrap().frame(), after.frame());
}

/// Physical crashes destroy drone health and disable later motor and trigger requests.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn physical_crashes_destroy_health_and_disable_weapons() {
    let mut battle = RobotBattle::default();
    for _ in 0..100 {
        let snapshot = battle.snapshot().unwrap();
        let actions = RobotActions::new(
            snapshot.physical(),
            [DroneAction::try_from([0.0; 4]).unwrap(); 3],
            [DroidAction::default(); 3],
        );
        battle
            .advance(
                actions,
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3],
            )
            .unwrap();
    }
    let snapshot = battle.snapshot().unwrap();
    for slot in RobotSlot::ALL {
        let owner = RobotId::Drone(slot);
        assert!(!snapshot.health(owner).is_alive());
        assert_eq!(snapshot.weapon_phase(owner), RobotWeaponPhase::Disabled);
        for motor in bevy_gym::robots::DroneMotor::ALL {
            assert_eq!(
                snapshot.physical().drone(slot).motor_state(motor),
                bevy_gym::robots::DroneMotorState::Failed
            );
        }
    }
}

/// Match duration accepts only positive complete 20 ms physical frames.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn match_duration_rejects_partial_and_zero_frames() {
    use bevy_gym::robots::{RobotMatch, RobotMatchError};
    use std::time::Duration;
    for duration in [
        Duration::ZERO,
        Duration::from_millis(1),
        Duration::from_millis(21),
        Duration::MAX,
    ] {
        assert_eq!(
            RobotMatch::try_from(duration).unwrap_err(),
            RobotMatchError::InvalidDuration
        );
    }
    for duration in [Duration::from_millis(20), Duration::from_millis(40)] {
        RobotMatch::try_from(duration).unwrap();
    }
}

/// Timeout freezes physical state and reset creates a fresh scope.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn match_timeout_is_terminal_and_reset_restores_running() {
    use bevy_gym::robots::{RobotMatch, RobotMatchError, RobotMatchResult, RobotMatchStatus};
    let mut game = RobotMatch::try_from(std::time::Duration::from_millis(20)).unwrap();
    assert_eq!(game.status(), RobotMatchStatus::Running);
    let before = game.snapshot().unwrap();
    let actions = RobotActions::new(
        before.physical(),
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::default(); 3],
    );
    let finished = game
        .advance(
            actions.clone(),
            [CombatSignal::IdleReleased; 3],
            [CombatSignal::IdleReleased; 3],
        )
        .unwrap();
    assert_eq!(
        game.status(),
        RobotMatchStatus::Finished(RobotMatchResult::Timeout)
    );
    assert_eq!(
        game.advance(
            actions,
            [CombatSignal::PursuitRequested; 3],
            [CombatSignal::PursuitRequested; 3]
        )
        .unwrap_err(),
        RobotMatchError::Finished(RobotMatchResult::Timeout)
    );
    assert_eq!(
        game.snapshot().unwrap().physical().frame(),
        finished.physical().frame()
    );
    game.reset();
    assert_eq!(game.status(), RobotMatchStatus::Running);
    assert_ne!(
        game.snapshot().unwrap().physical().frame(),
        before.physical().frame()
    );
}

/// Source rejection retires the combat backend and preserves one sticky diagnostic.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn match_failure_preserves_cause_until_reset() {
    use bevy_gym::robots::{RobotMatch, RobotMatchError, RobotMatchStatus};
    let mut game = RobotMatch::try_from(std::time::Duration::from_millis(40)).unwrap();
    let foreign = RobotWorld::default().snapshot().unwrap();
    let actions = RobotActions::new(
        &foreign,
        [DroneAction::try_from([0.5; 4]).unwrap(); 3],
        [DroidAction::default(); 3],
    );
    let cause = RobotWorldError::Rejected(RobotWorldFailure::WrongFrame);
    let error = game
        .advance(
            actions.clone(),
            [CombatSignal::PursuitRequested; 3],
            [CombatSignal::PursuitRequested; 3],
        )
        .unwrap_err();
    assert_eq!(error, RobotMatchError::Failed(cause));
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(game.status(), RobotMatchStatus::Failed(cause));
    assert_eq!(game.snapshot().unwrap_err(), error);
    assert_eq!(
        game.advance(
            actions,
            [CombatSignal::IdleReleased; 3],
            [CombatSignal::IdleReleased; 3]
        )
        .unwrap_err(),
        error
    );
    game.reset();
    assert_eq!(game.status(), RobotMatchStatus::Running);
    game.snapshot().unwrap();
}

/// The last complete frame ends a match; earlier boundaries retain running capabilities.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn match_consumes_exact_complete_duration() {
    use bevy_gym::robots::{RobotMatch, RobotMatchResult, RobotMatchStatus};
    let mut game = RobotMatch::try_from(std::time::Duration::from_millis(40)).unwrap();
    for expected in [
        RobotMatchStatus::Running,
        RobotMatchStatus::Finished(RobotMatchResult::Timeout),
    ] {
        let snapshot = game.snapshot().unwrap();
        let actions = RobotActions::new(
            snapshot.physical(),
            [DroneAction::try_from([0.5; 4]).unwrap(); 3],
            [DroidAction::default(); 3],
        );
        let after = game
            .advance(
                actions,
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3],
            )
            .unwrap();
        assert_ne!(after.physical().frame(), snapshot.physical().frame());
        assert_eq!(game.status(), expected);
    }
}

/// Actual fatal contacts finish elimination before timeout and retain a frozen final boundary.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn match_elimination_follows_actual_physical_deaths() {
    use bevy_gym::robots::{RobotMatch, RobotMatchError, RobotMatchStatus};
    let mut game = RobotMatch::try_from(std::time::Duration::from_secs(10)).unwrap();
    for _ in 0..100 {
        if !matches!(game.status(), RobotMatchStatus::Running) {
            break;
        }
        let snapshot = game.snapshot().unwrap();
        let actions = RobotActions::new(
            snapshot.physical(),
            [DroneAction::try_from([0.0; 4]).unwrap(); 3],
            [DroidAction::default(); 3],
        );
        game.advance(
            actions,
            [CombatSignal::IdleReleased; 3],
            [CombatSignal::IdleReleased; 3],
        )
        .unwrap();
    }
    let RobotMatchStatus::Finished(result) = game.status() else {
        panic!("physical deaths did not finish the match");
    };
    let snapshot = game.snapshot().unwrap();
    assert_elimination_result(&snapshot, result);
    let actions = RobotActions::new(
        snapshot.physical(),
        [DroneAction::try_from([1.0; 4]).unwrap(); 3],
        [DroidAction::try_from([1.0; 26]).unwrap(); 3],
    );
    assert_eq!(
        game.advance(
            actions,
            [CombatSignal::PursuitRequested; 3],
            [CombatSignal::PursuitRequested; 3]
        )
        .unwrap_err(),
        RobotMatchError::Finished(result)
    );
    assert_eq!(
        game.snapshot().unwrap().physical().frame(),
        snapshot.physical().frame()
    );
}

/// Compare the final result with independently measured living team health.
fn assert_elimination_result(
    snapshot: &bevy_gym::robots::RobotBattleSnapshot,
    result: bevy_gym::robots::RobotMatchResult,
) {
    use bevy_gym::robots::RobotMatchResult;
    let drones_alive = RobotSlot::ALL
        .into_iter()
        .any(|slot| snapshot.health(RobotId::Drone(slot)).is_alive());
    let droids_alive = RobotSlot::ALL
        .into_iter()
        .any(|slot| snapshot.health(RobotId::Droid(slot)).is_alive());
    assert_ne!(
        result,
        RobotMatchResult::Timeout,
        "100 frames cannot exhaust the 500-frame duration"
    );
    assert_eq!(drones_alive, result == RobotMatchResult::DronesWin);
    assert_eq!(droids_alive, result == RobotMatchResult::DroidsWin);
}
