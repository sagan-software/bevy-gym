//! Public actuator failure, reset, and terminal-state contracts.

#![cfg(feature = "robots")]

use bevy_gym::robots::{
    DroneAction, DroneEpisodeEnded, DroneHover, DroneMotor, DroneMotorState, DroneObservation,
};
use bevy_gym::Env;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

/// Compare physical state exactly while allowing different motor health.
fn same_motion(left: DroneObservation, right: DroneObservation) {
    assert_eq!(left.position(), right.position());
    assert_eq!(left.orientation(), right.orientation());
    assert_eq!(left.linear_velocity(), right.linear_velocity());
    assert_eq!(left.angular_velocity(), right.angular_velocity());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn every_failure_mask_matches_explicit_zero_commands() {
    assert_eq!(
        DroneMotor::ALL,
        [
            DroneMotor::FrontLeft,
            DroneMotor::FrontRight,
            DroneMotor::RearRight,
            DroneMotor::RearLeft,
        ]
    );
    for mask in 0_u8..16 {
        let mut damaged = DroneHover::disturbed();
        let mut reference = DroneHover::disturbed();
        damaged.reset(Some(42));
        reference.reset(Some(42));
        let mut fractions = [0.35, 0.55, 0.65, 0.45];
        for (index, motor) in DroneMotor::ALL.into_iter().enumerate() {
            if mask & (1 << index) != 0 {
                damaged.fail_motor(motor).expect("active episode");
                *fractions.get_mut(index).expect("four motors") = 0.0;
            }
        }
        let commanded = DroneAction::try_from([0.35, 0.55, 0.65, 0.45]).expect("valid");
        let masked = DroneAction::try_from(fractions).expect("valid");
        for _ in 0..20 {
            let actual = damaged.step(commanded);
            let expected = reference.step(masked);
            same_motion(actual.observation, expected.observation);
            assert_eq!(actual.reward.to_bits(), expected.reward.to_bits());
            assert_eq!(actual.status, expected.status);
            for (index, motor) in DroneMotor::ALL.into_iter().enumerate() {
                let state = if mask & (1 << index) == 0 {
                    DroneMotorState::Working
                } else {
                    DroneMotorState::Failed
                };
                assert_eq!(actual.observation.motor_state(motor), state);
                assert_eq!(
                    expected.observation.motor_state(motor),
                    DroneMotorState::Working
                );
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn repeated_failure_is_idempotent_and_reset_restores_the_seed_stream() {
    for motor in DroneMotor::ALL {
        let mut once = DroneHover::disturbed();
        let mut twice = DroneHover::disturbed();
        let mut healthy = DroneHover::disturbed();
        once.reset(Some(7));
        twice.reset(Some(7));
        healthy.reset(Some(7));
        once.fail_motor(motor).expect("active");
        twice.fail_motor(motor).expect("active");
        twice.fail_motor(motor).expect("idempotent");
        let action = DroneAction::try_from([0.5; 4]).expect("valid");
        assert_eq!(once.step(action), twice.step(action));
        let reset = once.reset(None);
        assert_eq!(reset, twice.reset(None));
        assert_eq!(reset, healthy.reset(None));
        for corner in DroneMotor::ALL {
            assert_eq!(
                reset.observation.motor_state(corner),
                DroneMotorState::Working
            );
        }
        assert_eq!(once.step(action), healthy.step(action));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn terminal_failure_cannot_change_the_absorbing_observation() {
    let mut drone = DroneHover::default();
    let off = DroneAction::try_from([0.0; 4]).expect("valid");
    let mut terminal = None;
    for _ in 0..500 {
        let step = drone.step(off);
        if step.is_done() {
            terminal = Some(step);
            break;
        }
    }
    let terminal = terminal.expect("power-off crash");
    for motor in DroneMotor::ALL {
        assert_eq!(drone.fail_motor(motor), Err(DroneEpisodeEnded));
        assert_eq!(drone.step(off), terminal);
    }
    drone.reset(Some(0));
    drone
        .fail_motor(DroneMotor::FrontLeft)
        .expect("reset resumes flight");
    assert_eq!(
        DroneEpisodeEnded.to_string(),
        "cannot fail a motor after the drone episode has ended"
    );
    assert!(std::error::Error::source(&DroneEpisodeEnded).is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reading_damage_does_not_advance_motion_or_random_state() {
    let mut drone = DroneHover::disturbed();
    let mut reference = DroneHover::disturbed();
    let initial = drone.reset(Some(42)).observation;
    reference.reset(Some(42));
    assert_eq!(drone.observation(), initial);
    drone.fail_motor(DroneMotor::FrontLeft).expect("active");
    let damaged = drone.observation();
    same_motion(damaged, initial);
    assert_eq!(
        damaged.motor_state(DroneMotor::FrontLeft),
        DroneMotorState::Failed
    );
    assert_eq!(drone.observation(), damaged);
    assert_eq!(drone.reset(None), reference.reset(None));
}
