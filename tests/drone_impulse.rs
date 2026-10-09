//! Physical impulse and terminal-state contracts through the public drone API.

#![cfg(feature = "robots")]

use bevy::math::Vec3;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);
use bevy_gym::{
    robots::{DroneAction, DroneHover, DroneImpulse, DroneImpulseRejected, InvalidDroneImpulse},
    Env,
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn a_centre_hit_changes_velocity_and_an_offset_hit_also_turns_the_body() {
    let mut centre = DroneHover::default();
    centre
        .apply_impulse(
            DroneImpulse::try_from((Vec3::ZERO, Vec3::X * 0.4)).expect("Valid active impact"),
        )
        .expect("Valid active impact");
    assert!((centre.observation().linear_velocity() - Vec3::X * 0.4).length() < 1e-5);
    assert!(centre.observation().angular_velocity().length() < 1e-5);
    let mut offset = DroneHover::default();
    offset
        .apply_impulse(
            DroneImpulse::try_from((Vec3::Z * 0.25, Vec3::X * 0.4)).expect("Valid active impact"),
        )
        .expect("Valid active impact");
    assert!((offset.observation().linear_velocity() - Vec3::X * 0.4).length() < 1e-5);
    assert!(offset.observation().angular_velocity().y > 0.1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_impact_fields_cannot_enter_the_solver() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.001, -1.001] {
        assert_eq!(
            DroneImpulse::try_from((Vec3::new(value, 0.0, 0.0), Vec3::ZERO)),
            Err(InvalidDroneImpulse::Point)
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 10.001, -10.001] {
        assert_eq!(
            DroneImpulse::try_from((Vec3::ZERO, Vec3::new(value, 0.0, 0.0))),
            Err(InvalidDroneImpulse::Momentum)
        );
    }
    assert_eq!(
        DroneImpulse::try_from((Vec3::NAN, Vec3::NAN)),
        Err(InvalidDroneImpulse::Point)
    );
    DroneImpulse::try_from((Vec3::ONE, Vec3::X * 10.0)).expect("Inclusive limits");
    DroneImpulse::try_from((-Vec3::ONE, -Vec3::X * 10.0)).expect("Negative inclusive limits");
    DroneImpulse::try_from((Vec3::ZERO, Vec3::ZERO)).expect("Zero impulse");
    assert_eq!(
        DroneImpulse::try_from((Vec3::ZERO, Vec3::splat(10.0))),
        Err(InvalidDroneImpulse::Momentum)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ended_episodes_reject_impulses_and_reset_clears_their_motion() {
    let mut drone = DroneHover::default();
    let hit = DroneImpulse::try_from((Vec3::Z * 0.25, Vec3::X * 0.4)).expect("Valid active impact");
    drone.apply_impulse(hit).expect("Valid active impact");
    let reset = drone.reset(Some(42));
    let reference = DroneHover::default().reset(Some(42));
    assert_eq!(reset, reference);
    let off = DroneAction::try_from([0.0; 4]).expect("Valid active impact");
    for _ in 0..1000 {
        if drone.step(off).status.is_done() {
            break;
        }
    }
    let terminal = drone.observation();
    assert_eq!(drone.apply_impulse(hit), Err(DroneImpulseRejected));
    assert_eq!(drone.observation(), terminal);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rotated_contact_uses_body_coordinates_without_advancing_pose() {
    let mut drone = DroneHover::disturbed();
    drone.reset(Some(9));
    let before = drone.observation();
    let momentum = before.orientation() * Vec3::Z * 0.4;
    let hit = DroneImpulse::try_from((Vec3::Z * 0.25, momentum)).expect("Valid hit");
    drone.apply_impulse(hit).expect("Flying");
    let after = drone.observation();
    assert_eq!(before.position(), after.position());
    assert_eq!(before.orientation(), after.orientation());
    assert!((after.linear_velocity() - before.linear_velocity() - momentum).length() < 1e-5);
    assert!((after.angular_velocity() - before.angular_velocity()).length() < 1e-5);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn every_axis_checks_finite_bounds_and_errors_retain_their_categories() {
    for axis in 0..3 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.001, 1.001] {
            let mut point = Vec3::ZERO;
            *point.as_mut().get_mut(axis).expect("Three axes") = value;
            assert_eq!(
                DroneImpulse::try_from((point, Vec3::ZERO)),
                Err(InvalidDroneImpulse::Point)
            );
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -10.001, 10.001] {
            let mut momentum = Vec3::ZERO;
            *momentum.as_mut().get_mut(axis).expect("Three axes") = value;
            assert_eq!(
                DroneImpulse::try_from((Vec3::ZERO, momentum)),
                Err(InvalidDroneImpulse::Momentum)
            );
        }
        for value in [-10.0, 0.0, 10.0] {
            let mut momentum = Vec3::ZERO;
            *momentum.as_mut().get_mut(axis).expect("Three axes") = value;
            DroneImpulse::try_from((Vec3::ONE, momentum)).expect("Axis boundary");
        }
    }
    assert_eq!(
        InvalidDroneImpulse::Point.to_string(),
        "drone contact components must be finite and within ±1 metre"
    );
    assert_eq!(
        InvalidDroneImpulse::Momentum.to_string(),
        "drone momentum must be finite with magnitude at most 10 N·s"
    );
    assert_eq!(
        DroneImpulseRejected.to_string(),
        "cannot apply an impulse after the drone episode has ended"
    );
    assert!(std::error::Error::source(&InvalidDroneImpulse::Point).is_none());
    assert!(std::error::Error::source(&DroneImpulseRejected).is_none());
}
