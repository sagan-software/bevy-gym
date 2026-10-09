//! Static geometry enters flight through validated values, never a mutable solver.

#![cfg(feature = "robots")]

use bevy::math::{Quat, Vec3};
use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor, DroneObstacle, InvalidDroneObstacle};
use bevy_gym::{Env, EpisodeStatus};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn empty_obstacles_preserve_existing_trajectories_exactly() {
    let mut original = DroneHover::default();
    let mut scene = DroneHover::with_obstacles([]);
    for seed in [0, 42, u64::MAX] {
        assert_eq!(
            original.reset(Some(seed)).observation,
            scene.reset(Some(seed)).observation
        );
        for fractions in [[0.5; 4], [0.6; 4], [0.55, 0.45, 0.45, 0.55], [0.0; 4]] {
            let command = DroneAction::try_from(fractions).expect("Valid command");
            for _ in 0..100 {
                let expected = original.step(command);
                let actual = scene.step(command);
                assert_eq!(actual.observation, expected.observation);
                assert_eq!(actual.reward.to_bits(), expected.reward.to_bits());
                assert_eq!(actual.status, expected.status);
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rotated_platform_survives_reset_and_contact_is_absorbing() {
    // Rotate the thin local X axis into world Y; the top face is y=1 metre.
    let platform = DroneObstacle::try_from((
        Vec3::new(0.0, 0.9, 0.0),
        Vec3::new(0.1, 2.0, 2.0),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
    ))
    .expect("Platform");
    let distant = DroneObstacle::try_from((Vec3::new(8.0, 2.0, 8.0), Vec3::ONE, Quat::IDENTITY))
        .expect("Distant box");
    let mut drone = DroneHover::with_obstacles([distant, platform]);
    let off = DroneAction::try_from([0.0; 4]).expect("Valid command");
    for seed in [0, 42] {
        drone.reset(Some(seed));
        let mut terminal = None;
        for _ in 0..100 {
            let step = drone.step(off);
            if step.is_done() {
                terminal = Some(step);
                break;
            }
        }
        let contact = terminal.expect("Platform contact within two seconds");
        assert_eq!(contact.status, EpisodeStatus::Terminated);
        assert!(contact.observation.position().y > 1.0);
        assert_eq!(contact.reward.to_bits(), 0.0_f64.to_bits());
        assert!(drone.fail_motor(DroneMotor::FrontLeft).is_err());
        assert_eq!(drone.step(off).observation, contact.observation);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn obstacle_validation_rejects_each_invalid_component_in_order() {
    for axis in 0..3 {
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1000.0_f32.to_bits() + 1),
            -f32::from_bits(1000.0_f32.to_bits() + 1),
        ] {
            let mut centre = Vec3::ZERO;
            centre[axis] = value;
            assert_eq!(
                DroneObstacle::try_from((centre, Vec3::ZERO, Quat::from_array([0.0; 4]))),
                Err(InvalidDroneObstacle::Centre)
            );
        }
        for value in [
            0.0,
            -0.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1000.0_f32.to_bits() + 1),
        ] {
            let mut half = Vec3::ONE;
            half[axis] = value;
            assert_eq!(
                DroneObstacle::try_from((Vec3::ZERO, half, Quat::from_array([0.0; 4]))),
                Err(InvalidDroneObstacle::HalfExtents)
            );
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn obstacle_rotation_rejects_zero_and_each_nonfinite_component() {
    for axis in 0..4 {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut components = [0.0, 0.0, 0.0, 1.0];
            components[axis] = value;
            assert_eq!(
                DroneObstacle::try_from((Vec3::ZERO, Vec3::ONE, Quat::from_array(components))),
                Err(InvalidDroneObstacle::Rotation)
            );
        }
    }
    assert_eq!(
        DroneObstacle::try_from((Vec3::ZERO, Vec3::ONE, Quat::from_array([0.0; 4]))),
        Err(InvalidDroneObstacle::Rotation)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn obstacle_valid_boundaries_accept_extreme_quaternion_scales() {
    for centre in [-1000.0, 0.0, 1000.0] {
        for half in [f32::from_bits(1), 1.0, 1000.0] {
            for scale in [f32::from_bits(1), 1.0, f32::MAX] {
                DroneObstacle::try_from((
                    Vec3::splat(centre),
                    Vec3::splat(half),
                    Quat::from_xyzw(0.0, 0.0, 0.0, scale),
                ))
                .expect("Valid boundary geometry");
            }
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn initial_overlap_ends_only_the_configured_world() {
    let block = DroneObstacle::try_from((Vec3::new(0.0, 2.0, 0.0), Vec3::ONE, Quat::IDENTITY))
        .expect("Valid box");
    let mut blocked = DroneHover::with_obstacles([block]);
    let mut clear = DroneHover::default();
    let hover = DroneAction::try_from([0.5; 4]).expect("Valid command");
    assert_eq!(blocked.step(hover).status, EpisodeStatus::Terminated);
    assert_eq!(clear.step(hover).status, EpisodeStatus::Continuing);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn validation_errors_explain_the_field_without_an_error_source() {
    use std::error::Error;
    for (error, message) in [
        (
            InvalidDroneObstacle::Centre,
            "Obstacle centre must be finite and within +/-1000 metres.",
        ),
        (
            InvalidDroneObstacle::HalfExtents,
            "Obstacle half extents must be finite and in (0, 1000] metres.",
        ),
        (
            InvalidDroneObstacle::Rotation,
            "Obstacle rotation must be finite and nonzero.",
        ),
    ] {
        assert_eq!(error.to_string(), message);
        assert!(error.source().is_none());
    }
}
