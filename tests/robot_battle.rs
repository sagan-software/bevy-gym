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
