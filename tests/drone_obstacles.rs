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

/// Range reads include the floor, exclude the drone and return the closest solid face.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ranges_read_current_geometry_without_advancing_physics() {
    use bevy_gym::robots::DroneRangeDirection;
    let walls = [2.0, 4.0].map(|x| {
        DroneObstacle::try_from((Vec3::new(x, 2.0, 0.0), Vec3::splat(0.5), Quat::IDENTITY))
            .expect("fixed wall")
    });
    let drone = DroneHover::with_obstacles(walls);
    let before = drone.observation();
    let ranges = drone.ranges();
    let right = ranges
        .distance(DroneRangeDirection::Right)
        .expect("near wall");
    let down = ranges.distance(DroneRangeDirection::Down).expect("floor");
    assert!((right.metres() - 1.5).abs() < 0.000_01);
    assert!((down.metres() - 2.0).abs() < 0.000_01);
    for direction in [
        DroneRangeDirection::Forward,
        DroneRangeDirection::Back,
        DroneRangeDirection::Left,
        DroneRangeDirection::Up,
    ] {
        assert!(
            ranges.distance(direction).is_none(),
            "unobstructed {direction:?}"
        );
    }
    assert_eq!(drone.observation(), before, "readonly range query");
    assert_eq!(drone.ranges(), ranges, "repeated reads are stable");
}

/// Distance construction rejects malformed values before the finite range bounds.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn range_distances_validate_each_numeric_boundary() {
    use bevy_gym::robots::{DroneRangeDistance, InvalidDroneRangeDistance};
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            DroneRangeDistance::try_from(value),
            Err(InvalidDroneRangeDistance::NonFinite)
        );
    }
    for value in [
        -1.0,
        -f32::from_bits(1),
        f32::from_bits(10.0_f32.to_bits() + 1),
        f32::MAX,
    ] {
        assert_eq!(
            DroneRangeDistance::try_from(value),
            Err(InvalidDroneRangeDistance::OutOfRange)
        );
    }
    for value in [-0.0_f32, 0.0, f32::from_bits(1), 1.5, 10.0] {
        let distance = DroneRangeDistance::try_from(value).expect("valid metres");
        assert_eq!(distance.metres().to_bits(), value.to_bits());
    }
    assert_eq!(
        DroneRangeDistance::MAX.metres().to_bits(),
        10.0_f32.to_bits()
    );
    assert_eq!(
        InvalidDroneRangeDistance::NonFinite.to_string(),
        "Range distance must be finite."
    );
    assert_eq!(
        InvalidDroneRangeDistance::OutOfRange.to_string(),
        "Range distance must be in [0, 10] metres."
    );
    let error: &dyn std::error::Error = &InvalidDroneRangeDistance::OutOfRange;
    assert!(error.source().is_none());
}

/// Solid origin overlap is zero; a ten-metre hit is distinct from a missed ray.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ranges_preserve_overlap_and_inclusive_limit() {
    use bevy_gym::robots::DroneRangeDirection;
    let wall = |centre_x| {
        DroneObstacle::try_from((
            Vec3::new(centre_x, 2.0, 0.0),
            Vec3::splat(0.5),
            Quat::IDENTITY,
        ))
        .expect("valid fixed box")
    };
    let at_limit = DroneHover::with_obstacles([wall(10.5)]);
    let distance = at_limit
        .ranges()
        .distance(DroneRangeDirection::Right)
        .expect("inclusive hit");
    assert_eq!(distance.metres().to_bits(), 10.0_f32.to_bits());
    let beyond = DroneHover::with_obstacles([wall(f32::from_bits(10.5_f32.to_bits() + 1))]);
    assert!(beyond
        .ranges()
        .distance(DroneRangeDirection::Right)
        .is_none());
    let inside = DroneHover::with_obstacles([wall(0.0)]);
    for direction in DroneRangeDirection::ALL {
        let distance = inside
            .ranges()
            .distance(direction)
            .expect("solid origin overlap");
        assert_eq!(distance.metres().to_bits(), 0.0_f32.to_bits());
    }
}

/// Sensor reads preserve seeded and unseeded resets, physical steps and terminal snapshots.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ranges_preserve_trajectories_and_reset_streams() {
    let mut observed = DroneHover::default();
    let mut control = DroneHover::default();
    let off = DroneAction::try_from([0.0; 4]).expect("isolated test actuator fixture");
    for seed in [Some(42), None, None] {
        for _ in 0..10 {
            let _snapshot = observed.ranges();
        }
        assert_eq!(
            observed.reset(seed).observation,
            control.reset(seed).observation
        );
        for _ in 0..100 {
            let _snapshot = observed.ranges();
            let before = observed.observation();
            let _again = observed.ranges();
            assert_eq!(observed.observation(), before);
            let measured = observed.step(off);
            let expected = control.step(off);
            assert_eq!(measured.observation, expected.observation);
            assert_eq!(measured.reward.to_bits(), expected.reward.to_bits());
            assert_eq!(measured.status, expected.status);
        }
        let before = observed.observation();
        let terminal = observed.ranges();
        observed.step(off);
        assert_eq!(observed.ranges(), terminal);
        assert_eq!(observed.observation(), before);
    }
}

/// Empty geometry preserves disturbed construction, reset streams and actuator trajectories.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn empty_obstacles_preserve_disturbed_trajectories_exactly() {
    let mut original = DroneHover::disturbed();
    let mut scene = DroneHover::disturbed_with_obstacles([]);
    assert_eq!(scene.observation(), original.observation());
    for seed in [Some(0), Some(42), Some(u64::MAX), None, None] {
        assert_eq!(
            scene.reset(seed).observation,
            original.reset(seed).observation
        );
        for fractions in [[0.5; 4], [0.6; 4], [0.55, 0.45, 0.45, 0.55], [0.0; 4]] {
            let command = DroneAction::try_from(fractions).expect("isolated actuator fixture");
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

/// Disturbed reset streams retain geometry, restore motors and preserve terminal snapshots.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn disturbed_obstacles_survive_resets_and_contact_is_absorbing() {
    use bevy_gym::robots::{DroneMotorState, DroneRangeDirection};
    let platform = DroneObstacle::try_from((
        Vec3::new(0.0, 0.9, 0.0),
        Vec3::new(3.0, 0.1, 3.0),
        Quat::IDENTITY,
    ))
    .expect("platform top at one metre");
    let mut drone = DroneHover::disturbed_with_obstacles([platform]);
    let mut control = DroneHover::disturbed();
    assert_eq!(drone.observation(), control.observation());
    assert_ne!(
        drone.observation(),
        DroneHover::with_obstacles([platform]).observation()
    );
    drone.fail_motor(DroneMotor::FrontLeft).expect("live motor");
    for seed in [Some(42), Some(u64::MAX), None, None] {
        let start = drone.reset(seed).observation;
        assert_eq!(
            start,
            control.reset(seed).observation,
            "reset stream independent of geometry"
        );
        for motor in [
            DroneMotor::FrontLeft,
            DroneMotor::FrontRight,
            DroneMotor::RearRight,
            DroneMotor::RearLeft,
        ] {
            assert_eq!(start.motor_state(motor), DroneMotorState::Working);
        }
        let ranges = drone.ranges();
        let down = ranges
            .distance(DroneRangeDirection::Down)
            .expect("platform below body");
        assert!(
            (0.8..1.3).contains(&down.metres()),
            "one-metre platform distance"
        );
        assert_eq!(drone.observation(), start, "readonly sensor");
        assert_platform_contact_is_absorbing(&mut drone);
    }
}

/// Check first contact and repeated terminal reads through the public environment seam.
fn assert_platform_contact_is_absorbing(drone: &mut DroneHover) {
    let off = DroneAction::try_from([0.0; 4]).expect("isolated actuator fixture");
    let mut terminal = None;
    for _ in 0..100 {
        let step = drone.step(off);
        if step.is_done() {
            terminal = Some(step);
            break;
        }
    }
    let contact = terminal.expect("platform contact within two seconds");
    assert_eq!(contact.status, EpisodeStatus::Terminated);
    assert!(
        contact.observation.position().y > 1.0,
        "contact precedes floor"
    );
    assert_eq!(contact.reward.to_bits(), 0.0_f64.to_bits());
    assert!(drone.fail_motor(DroneMotor::FrontLeft).is_err());
    let frozen_ranges = drone.ranges();
    let repeated = drone.step(off);
    assert_eq!(repeated.observation, contact.observation);
    assert_eq!(repeated.reward.to_bits(), contact.reward.to_bits());
    assert_eq!(repeated.status, contact.status);
    assert_eq!(drone.ranges(), frozen_ranges);
}
