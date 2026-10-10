//! Goal-conditioned flight uses the same actuator-driven physics as disturbed hover.
#![cfg(feature = "robots")]

use bevy::math::{Vec2, Vec3};
use bevy_gym::robots::{DroneAction, DroneDestination, DroneHover, DroneTravel};
use bevy_gym::Env;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// A task destination changes observation and reward, never the supplied motor action.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn destination_does_not_choose_or_modify_physical_actions() {
    let goal = DroneDestination::try_from((Vec3::new(3.0, 2.0, -2.0), Vec2::X))
        .expect("interior goal and nonzero heading");
    let mut travel = DroneTravel::new(goal);
    let mut hover = DroneHover::disturbed();
    let initial = travel.reset(Some(42)).observation;
    assert_eq!(initial.body(), hover.reset(Some(42)).observation);
    assert_eq!(initial.destination(), goal);
    for fractions in [[0.5; 4], [0.4, 0.6, 0.5, 0.5], [0.6, 0.4, 0.5, 0.5]] {
        let action = DroneAction::try_from(fractions).expect("bounded motor fixture");
        let actual = travel.step(action);
        let expected = hover.step(action);
        assert_eq!(actual.observation.body(), expected.observation);
        assert_eq!(actual.status, expected.status);
        assert_eq!(actual.observation.destination(), goal);
    }
}

/// Reject every invalid coordinate before inspecting the heading.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn destination_validates_each_open_position_boundary_before_heading() {
    use bevy_gym::robots::InvalidDroneDestination;
    for axis in 0..3 {
        let lower = if axis == 1 { 0.0 } else { -10.0 };
        for invalid in [
            lower - 1.0,
            lower,
            10.0,
            11.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut position = Vec3::new(0.0, 2.0, 0.0);
            position[axis] = invalid;
            assert_eq!(
                DroneDestination::try_from((position, Vec2::ZERO)),
                Err(InvalidDroneDestination::Position)
            );
        }
        for valid in [lower + 0.001, 9.999] {
            let mut position = Vec3::new(0.0, 2.0, 0.0);
            position[axis] = valid;
            let destination =
                DroneDestination::try_from((position, Vec2::X)).expect("interior position");
            assert_eq!(destination.position(), position);
        }
    }
    assert!(InvalidDroneDestination::Position
        .to_string()
        .contains("(-10, 10)"));
}

/// Heading validation accepts every finite nonzero magnitude and preserves direction.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn destination_normalizes_valid_headings_and_rejects_invalid_components() {
    use bevy_gym::robots::InvalidDroneDestination;
    let position = Vec3::new(0.0, 2.0, 0.0);
    for heading in [
        Vec2::ZERO,
        Vec2::splat(-0.0),
        Vec2::new(f32::NAN, 1.0),
        Vec2::new(1.0, f32::NAN),
        Vec2::new(f32::INFINITY, 1.0),
        Vec2::new(f32::NEG_INFINITY, 1.0),
        Vec2::new(1.0, f32::INFINITY),
        Vec2::new(1.0, f32::NEG_INFINITY),
    ] {
        assert_eq!(
            DroneDestination::try_from((position, heading)),
            Err(InvalidDroneDestination::Heading)
        );
    }
    for magnitude in [f32::from_bits(1), 1.0, f32::MAX] {
        for heading in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y, Vec2::ONE] {
            let destination = DroneDestination::try_from((position, heading * magnitude))
                .expect("finite nonzero heading");
            let actual = destination.heading();
            assert!(actual.as_vec2().distance(heading.normalize()) < 1e-6);
        }
    }
    assert!(InvalidDroneDestination::Heading
        .to_string()
        .contains("nonzero"));
    assert!(std::error::Error::source(&InvalidDroneDestination::Heading).is_none());
}

/// Construction and repeated resets retain the task while restarting or advancing physics seeds.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reset_and_terminal_lifecycle_preserve_destination_without_fallback() {
    let destination = DroneDestination::try_from((Vec3::new(-3.0, 3.0, 4.0), Vec2::NEG_Y))
        .expect("interior destination");
    let mut left = DroneTravel::new(destination);
    let mut right = DroneTravel::new(destination);
    assert_eq!(left.observation(), right.reset(Some(0)).observation);
    let initial = left.reset(Some(u64::MAX));
    assert_eq!(initial, right.reset(Some(u64::MAX)));
    let off = DroneAction::try_from([0.0; 4]).expect("valid unpowered fixture");
    let terminal = (0..500)
        .map(|_| left.step(off))
        .find(bevy_gym::Step::is_done)
        .expect("unpowered test fixture crashes");
    assert!(terminal.status.is_terminal());
    assert_eq!(terminal.reward.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        terminal,
        left.step(DroneAction::try_from([1.0; 4]).expect("valid full-power fixture"))
    );
    assert_eq!(terminal.observation.destination(), destination);
    let continued = left.reset(None);
    assert_ne!(initial, continued);
    assert_eq!(continued, right.reset(None));
    assert_eq!(initial, left.reset(Some(u64::MAX)));
    assert!(format!("{left:?}").contains("DroneTravel"));
}
