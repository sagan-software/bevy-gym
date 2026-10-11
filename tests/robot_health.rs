//! Public damage boundaries, independent of agent decisions.
#![cfg(feature = "robots")]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use bevy_gym::robots::{DroneMotor, RobotDamage, RobotHealth, RobotRotorHealth, RobotTeam};

/// Body damage preserves team identity and ends exactly at its configured hit count.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn body_damage_has_exact_team_limits_and_absorbing_death() {
    for (team, limit) in [(RobotTeam::Drones, 6), (RobotTeam::Droids, 3)] {
        let mut health = RobotHealth::from(team);
        assert_eq!(health.team(), team);
        assert!(health.is_alive());
        assert_eq!(health.body_hits_remaining().get(), limit);
        for remaining in (1..limit).rev() {
            assert_eq!(health.hit_body(), RobotDamage::Hit);
            assert_eq!(health.body_hits_remaining().get(), remaining);
        }
        assert_eq!(health.hit_body(), RobotDamage::Destroyed);
        assert_eq!(health.body_hits_remaining().get(), 0);
        assert!(!health.is_alive());
        assert_eq!(health.team(), team);
        assert_eq!(health.hit_body(), RobotDamage::Ignored);
        assert_eq!(health.destroy(), RobotDamage::Ignored);
    }
}

/// Rotor damage is independent of hull life and reports its one-time destruction.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn each_rotor_has_two_hits_without_destroying_other_parts() {
    for motor in DroneMotor::ALL {
        let mut health = RobotHealth::from(RobotTeam::Drones);
        assert_eq!(health.rotor(motor).unwrap(), RobotRotorHealth::Intact);
        assert_eq!(health.hit_rotor(motor).unwrap(), RobotDamage::Hit);
        assert_eq!(health.rotor(motor).unwrap(), RobotRotorHealth::Damaged);
        assert_eq!(
            health.hit_rotor(motor).unwrap(),
            RobotDamage::RotorDestroyed(motor)
        );
        assert_eq!(health.rotor(motor).unwrap(), RobotRotorHealth::Destroyed);
        assert_eq!(health.hit_rotor(motor).unwrap(), RobotDamage::Ignored);
        assert!(health.is_alive());
        assert_eq!(health.body_hits_remaining().get(), 6);
        for other in DroneMotor::ALL {
            if other != motor {
                assert_eq!(health.rotor(other).unwrap(), RobotRotorHealth::Intact);
            }
        }
    }
}

/// Motor availability follows rotor integrity while alive and hull death afterward.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn failed_rotors_and_dead_hulls_disable_motors() {
    use bevy_gym::robots::DroneMotorState;
    let mut health = RobotHealth::from(RobotTeam::Drones);
    for motor in DroneMotor::ALL {
        assert_eq!(health.motor_state(motor).unwrap(), DroneMotorState::Working);
        assert_eq!(health.hit_rotor(motor).unwrap(), RobotDamage::Hit);
        assert_eq!(health.motor_state(motor).unwrap(), DroneMotorState::Working);
        assert_eq!(
            health.hit_rotor(motor).unwrap(),
            RobotDamage::RotorDestroyed(motor)
        );
        assert_eq!(health.motor_state(motor).unwrap(), DroneMotorState::Failed);
    }
    assert!(health.is_alive());
    assert_eq!(health.destroy(), RobotDamage::Destroyed);
    for motor in DroneMotor::ALL {
        assert_eq!(health.rotor(motor).unwrap(), RobotRotorHealth::Destroyed);
        assert_eq!(health.motor_state(motor).unwrap(), DroneMotorState::Failed);
        assert_eq!(health.hit_rotor(motor).unwrap(), RobotDamage::Ignored);
    }
}

/// Hull death retains rotor observations but disables even previously working motors.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hull_destruction_retains_rotors_without_leaving_live_actuators() {
    use bevy_gym::robots::DroneMotorState;
    let mut health = RobotHealth::from(RobotTeam::Drones);
    health.hit_rotor(DroneMotor::FrontLeft).unwrap();
    assert_eq!(health.destroy(), RobotDamage::Destroyed);
    assert_eq!(
        health.rotor(DroneMotor::FrontLeft).unwrap(),
        RobotRotorHealth::Damaged
    );
    for motor in DroneMotor::ALL {
        let before = health.rotor(motor).unwrap();
        assert_eq!(health.motor_state(motor).unwrap(), DroneMotorState::Failed);
        assert_eq!(health.hit_rotor(motor).unwrap(), RobotDamage::Ignored);
        assert_eq!(health.rotor(motor).unwrap(), before);
    }
    assert_eq!(health.body_hits_remaining().get(), 0);
    assert_eq!(health.hit_body(), RobotDamage::Ignored);
}

/// Wrong hull kind is rejected before lifecycle denial and cannot change hull life.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn droids_reject_rotor_locations_before_and_after_death() {
    use bevy_gym::robots::RobotRotorHitRejected;
    use std::error::Error;
    let mut health = RobotHealth::from(RobotTeam::Droids);
    for alive in [true, false] {
        assert_eq!(health.is_alive(), alive);
        let before = health.body_hits_remaining();
        for motor in DroneMotor::ALL {
            assert_eq!(health.rotor(motor), Err(RobotRotorHitRejected));
            assert_eq!(health.motor_state(motor), Err(RobotRotorHitRejected));
            let error = health.hit_rotor(motor).unwrap_err();
            assert_eq!(error, RobotRotorHitRejected);
            assert_eq!(format!("{error}"), "droids have no drone rotors");
            assert!(error.source().is_none());
            assert_eq!(health.body_hits_remaining(), before);
            assert_eq!(health.team(), RobotTeam::Droids);
        }
        health.destroy();
    }
}

/// Fatal contact emits destruction once for either team, with no underflow or revival.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fatal_contact_is_absorbing_and_reset_is_explicit_construction() {
    for team in [RobotTeam::Drones, RobotTeam::Droids] {
        let mut health = RobotHealth::from(team);
        let original = health.body_hits_remaining();
        assert_eq!(health.destroy(), RobotDamage::Destroyed);
        assert_eq!(health.destroy(), RobotDamage::Ignored);
        assert_eq!(health.hit_body(), RobotDamage::Ignored);
        assert!(!health.is_alive());
        assert_eq!(health.team(), team);
        let fresh = RobotHealth::from(team);
        assert!(fresh.is_alive());
        assert_eq!(fresh.team(), team);
        assert_eq!(fresh.body_hits_remaining(), original);
        if team == RobotTeam::Drones {
            for motor in DroneMotor::ALL {
                assert_eq!(fresh.rotor(motor).unwrap(), RobotRotorHealth::Intact);
            }
        }
    }
}

/// Inspection copies contain no mutable alias to the authoritative health.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn copied_health_changes_independently() {
    let health = RobotHealth::from(RobotTeam::Drones);
    let mut copy = health;
    assert_eq!(copy.hit_body(), RobotDamage::Hit);
    assert_eq!(copy.body_hits_remaining().get(), 5);
    assert_eq!(health.body_hits_remaining().get(), 6);
    assert_eq!(
        copy.hit_rotor(DroneMotor::RearLeft).unwrap(),
        RobotDamage::Hit
    );
    assert_eq!(
        health.rotor(DroneMotor::RearLeft).unwrap(),
        RobotRotorHealth::Intact
    );
}
