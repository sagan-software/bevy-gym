//! Mechanical weapon requests never derive firing from sensor observations.
#![cfg(feature = "robots")]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use bevy_gym::{
    robots::{RobotTeam, RobotWeapon, RobotWeaponOutcome},
    training::CombatSignal,
};

/// Every released alert category remains incapable of starting a discharge.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn released_signals_cannot_charge_or_discharge() {
    for signal in [
        CombatSignal::IdleReleased,
        CombatSignal::SuspiciousReleased,
        CombatSignal::PursuitReleased,
        CombatSignal::SearchReleased,
    ] {
        for team in [RobotTeam::Drones, RobotTeam::Droids] {
            let mut weapon = RobotWeapon::from(team);
            for _ in 0..100 {
                assert_eq!(weapon.advance(signal), RobotWeaponOutcome::NoRequest);
            }
        }
    }
}

/// Held requests pay the complete charge and each inter-round delay.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn drone_charge_and_volley_have_exact_boundaries() {
    let mut weapon = RobotWeapon::from(RobotTeam::Drones);
    assert_eq!(
        weapon.advance(CombatSignal::PursuitRequested),
        RobotWeaponOutcome::Charging
    );
    for _ in 0..39 {
        assert_eq!(
            weapon.advance(CombatSignal::PursuitRequested),
            RobotWeaponOutcome::Charging
        );
    }
    assert_eq!(
        weapon.advance(CombatSignal::PursuitRequested),
        RobotWeaponOutcome::Fired
    );
    for _ in 0..5 {
        assert_eq!(
            weapon.advance(CombatSignal::IdleRequested),
            RobotWeaponOutcome::VolleyDelay
        );
    }
    assert_eq!(
        weapon.advance(CombatSignal::IdleRequested),
        RobotWeaponOutcome::Fired
    );
    for _ in 0..5 {
        assert_eq!(
            weapon.advance(CombatSignal::SearchRequested),
            RobotWeaponOutcome::VolleyDelay
        );
    }
    assert_eq!(
        weapon.advance(CombatSignal::SearchRequested),
        RobotWeaponOutcome::Fired
    );
    for _ in 0..60 {
        assert_eq!(
            weapon.advance(CombatSignal::SuspiciousRequested),
            RobotWeaponOutcome::Cooling
        );
    }
    assert_eq!(
        weapon.advance(CombatSignal::SuspiciousRequested),
        RobotWeaponOutcome::Charging
    );
}

/// Magazine exhaustion cannot reload itself and cooling cannot create a request.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn droid_magazine_and_cooldown_are_mechanical_limits() {
    let mut weapon = RobotWeapon::from(RobotTeam::Droids);
    for _ in 0..12 {
        assert_eq!(
            weapon.advance(CombatSignal::PursuitRequested),
            RobotWeaponOutcome::Fired
        );
        for _ in 0..13 {
            assert_eq!(
                weapon.advance(CombatSignal::IdleReleased),
                RobotWeaponOutcome::Cooling
            );
        }
    }
    assert_eq!(weapon.phase(), bevy_gym::robots::RobotWeaponPhase::Empty);
    assert_eq!(
        weapon.advance(CombatSignal::PursuitRequested),
        RobotWeaponOutcome::Empty
    );
    weapon.disable();
    assert_eq!(
        weapon.advance(CombatSignal::PursuitRequested),
        RobotWeaponOutcome::Disabled
    );
}

/// Every requested alert category can request a shot without inferred target state.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn requested_categories_are_direct_trigger_requests() {
    for signal in [
        CombatSignal::IdleRequested,
        CombatSignal::SuspiciousRequested,
        CombatSignal::PursuitRequested,
        CombatSignal::SearchRequested,
    ] {
        let mut droid = RobotWeapon::from(RobotTeam::Droids);
        assert_eq!(droid.advance(signal), RobotWeaponOutcome::Fired);
        let mut drone = RobotWeapon::from(RobotTeam::Drones);
        for _ in 0..40 {
            assert_eq!(drone.advance(signal), RobotWeaponOutcome::Charging);
        }
        assert_eq!(drone.advance(signal), RobotWeaponOutcome::Fired);
    }
}

/// A released trigger discards all charge progress; a new request pays the full charge.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn releasing_charge_discards_progress() {
    let mut weapon = RobotWeapon::from(RobotTeam::Drones);
    for _ in 0..40 {
        weapon.advance(CombatSignal::PursuitRequested);
    }
    assert_eq!(
        weapon.advance(CombatSignal::PursuitReleased),
        RobotWeaponOutcome::NoRequest
    );
    assert_eq!(weapon.phase(), bevy_gym::robots::RobotWeaponPhase::Ready);
    for _ in 0..40 {
        assert_eq!(
            weapon.advance(CombatSignal::SearchRequested),
            RobotWeaponOutcome::Charging
        );
    }
    assert_eq!(
        weapon.advance(CombatSignal::SearchRequested),
        RobotWeaponOutcome::Fired
    );
}

/// Cancelling either later volley round pays the full recovery before charging again.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn cancelling_each_volley_position_enters_full_cooldown() {
    use bevy_gym::robots::RobotWeaponPhase;
    use std::time::Duration;
    for steps in [41, 47] {
        let mut weapon = RobotWeapon::from(RobotTeam::Drones);
        for _ in 0..steps {
            weapon.advance(CombatSignal::IdleRequested);
        }
        assert_eq!(
            weapon.advance(CombatSignal::IdleReleased),
            RobotWeaponOutcome::Cooling
        );
        assert_eq!(
            weapon.phase(),
            RobotWeaponPhase::Cooling {
                remaining: Duration::from_millis(1200)
            }
        );
        for _ in 0..60 {
            assert_eq!(
                weapon.advance(CombatSignal::SearchRequested),
                RobotWeaponOutcome::Cooling
            );
        }
        assert_eq!(weapon.phase(), RobotWeaponPhase::Ready);
        assert_eq!(
            weapon.advance(CombatSignal::SearchRequested),
            RobotWeaponOutcome::Charging
        );
    }
}

/// The250ms rule keeps its exact remainder and becomes ready only at the260ms boundary.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pistol_delay_is_measured_without_rounding_the_initial_rule() {
    use bevy_gym::robots::RobotWeaponPhase;
    use std::time::Duration;
    let mut weapon = RobotWeapon::from(RobotTeam::Droids);
    weapon.advance(CombatSignal::PursuitRequested);
    assert_eq!(
        weapon.phase(),
        RobotWeaponPhase::Cooling {
            remaining: Duration::from_millis(250)
        }
    );
    for _ in 0..12 {
        assert_eq!(
            weapon.advance(CombatSignal::PursuitRequested),
            RobotWeaponOutcome::Cooling
        );
    }
    assert_eq!(
        weapon.phase(),
        RobotWeaponPhase::Cooling {
            remaining: Duration::from_millis(10)
        }
    );
    assert_eq!(
        weapon.advance(CombatSignal::PursuitRequested),
        RobotWeaponOutcome::Cooling
    );
    assert_eq!(weapon.phase(), RobotWeaponPhase::Ready);
    assert_eq!(
        weapon.advance(CombatSignal::IdleReleased),
        RobotWeaponOutcome::NoRequest
    );
    assert_eq!(
        weapon.advance(CombatSignal::IdleRequested),
        RobotWeaponOutcome::Fired
    );
}

/// Every reachable phase can be disabled; no later signal can reactivate it.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn disable_is_terminal_from_every_phase() {
    use bevy_gym::robots::RobotWeaponPhase;
    for (team, steps) in [
        (RobotTeam::Drones, 0),
        (RobotTeam::Drones, 1),
        (RobotTeam::Drones, 41),
        (RobotTeam::Drones, 53),
        (RobotTeam::Droids, 0),
        (RobotTeam::Droids, 1),
        (RobotTeam::Droids, 168),
    ] {
        let mut weapon = RobotWeapon::from(team);
        for _ in 0..steps {
            weapon.advance(CombatSignal::IdleRequested);
        }
        weapon.disable();
        weapon.disable();
        for signal in CombatSignal::ALL {
            assert_eq!(weapon.advance(signal), RobotWeaponOutcome::Disabled);
            assert_eq!(weapon.phase(), RobotWeaponPhase::Disabled);
        }
        let reset = RobotWeapon::from(team);
        assert_eq!(reset.phase(), RobotWeaponPhase::Ready);
    }
}

/// Inspection and cloned independent weapons cannot advance another weapon's timing.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn phase_measurements_preserve_independent_weapon_state() {
    use bevy_gym::robots::{RobotVolleyRounds, RobotWeaponPhase};
    use std::time::Duration;
    let mut first = RobotWeapon::from(RobotTeam::Drones);
    let second = first;
    first.advance(CombatSignal::IdleRequested);
    assert_eq!(second.phase(), RobotWeaponPhase::Ready);
    assert_eq!(
        first.phase(),
        RobotWeaponPhase::Charging {
            remaining: Duration::from_millis(800)
        }
    );
    for _ in 0..40 {
        first.advance(CombatSignal::IdleRequested);
    }
    assert_eq!(
        first.phase(),
        RobotWeaponPhase::Volley {
            remaining: Duration::from_millis(120),
            rounds: RobotVolleyRounds::Two
        }
    );
    for _ in 0..6 {
        first.advance(CombatSignal::IdleRequested);
    }
    assert_eq!(
        first.phase(),
        RobotWeaponPhase::Volley {
            remaining: Duration::from_millis(120),
            rounds: RobotVolleyRounds::One
        }
    );
}
