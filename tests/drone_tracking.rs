//! Qualify the bundled flight pilot through typed observations and motor actions.
#![cfg(feature = "robots")]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[path = "../examples/robots/flight_control/mod.rs"]
mod flight_control;

use bevy::math::{Dir2, Vec3};
use bevy_gym::{robots::DroneHover, Env};
use flight_control::{FlightGoal, FlightPilot};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn action_history_cannot_change_the_same_command() {
    let pilot = FlightPilot::bundled().expect("bundled model loads");
    let mut drone = DroneHover::disturbed();
    let observation = drone.reset(Some(42)).observation;
    let east = FlightGoal::try_from((Vec3::new(8.0, 2.0, 0.0), Dir2::X)).unwrap();
    let west = FlightGoal::try_from((Vec3::new(-8.0, 2.0, 0.0), Dir2::NEG_X)).unwrap();
    let action = pilot.action(observation, east).unwrap();
    let other = pilot.action(observation, west).unwrap();
    assert_ne!(action, other);
    assert_eq!(action, pilot.action(observation, east).unwrap());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn goals_reject_each_nonfinite_coordinate_and_each_box_face() {
    for axis in 0..3 {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut position = Vec3::new(0.0, 2.0, 0.0);
            position[axis] = invalid;
            FlightGoal::try_from((position, Dir2::X)).expect_err("invalid coordinate is rejected");
        }
        let lower = if axis == 1 { 0.0 } else { -10.0 };
        for invalid in [lower - 1.0, lower, 10.0, 11.0] {
            let mut position = Vec3::new(0.0, 2.0, 0.0);
            position[axis] = invalid;
            FlightGoal::try_from((position, Dir2::X)).expect_err("invalid coordinate is rejected");
        }
        for valid in [lower + 0.001, 9.999] {
            let mut position = Vec3::new(0.0, 2.0, 0.0);
            position[axis] = valid;
            FlightGoal::try_from((position, Dir2::X)).expect("interior coordinate is valid");
        }
    }
}

/// Run the fixed ten-second qualification episode and return its heading-weighted reward.
#[expect(
    clippy::manual_midpoint,
    clippy::suboptimal_flops,
    reason = "Preserve the archived qualification reward's exact floating-point operation order."
)]
fn qualify(pilot: &FlightPilot, seed: u64, position: Vec3, heading: Dir2) -> f64 {
    let goal = FlightGoal::try_from((position, heading)).expect("qualification goal is valid");
    let desired = Vec3::new(heading.x, 0.0, heading.y);
    let mut drone = DroneHover::disturbed();
    let mut observation = drone.reset(Some(seed)).observation;
    let mut reward = 0.0;
    let mut settled = 0;
    for _ in 0..500 {
        let action = pilot
            .action(observation, goal)
            .expect("qualified policy emits a valid action");
        let step = drone.step(action);
        assert!(
            !step.is_done(),
            "seed {seed}, goal {position:?}, heading {heading:?}"
        );
        observation = step.observation;
        let forward = observation.orientation() * Vec3::NEG_Z;
        let up = observation.orientation() * Vec3::Y;
        let distance = observation.position().distance(position);
        let heading_error = forward.cross(desired).y.atan2(forward.dot(desired));
        // Reward matches the archived heading qualification, with dimensionless factors.
        reward += f64::from((up.y + 1.0) / 2.0 / (1.0 + distance * distance))
            * f64::from(0.25 + 0.75 * (forward.dot(desired) + 1.0) / 2.0);
        if distance < 0.5 && heading_error.abs() < std::f32::consts::PI / 12.0 {
            settled += 1;
        } else {
            settled = 0;
        }
    }
    assert!(observation.position().distance(position) <= 0.5);
    assert!(
        settled >= 100,
        "seed {seed}, goal {position:?}: {settled} final actions in band"
    );
    reward
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frozen_pilot_qualifies_all_128_held_out_heading_cases() {
    let pilot = FlightPilot::bundled().unwrap();
    for heading in [Dir2::NEG_Y, Dir2::X, Dir2::Y, Dir2::NEG_X] {
        for offset in 1001..=1032 {
            let reward = qualify(&pilot, u64::MAX - offset, Vec3::new(0.0, 2.0, 0.0), heading);
            assert!(
                reward >= 400.0,
                "heading {heading:?}, offset {offset}: {reward}"
            );
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frozen_pilot_qualifies_all_80_bounded_waypoint_cases() {
    let pilot = FlightPilot::bundled().unwrap();
    for heading in [Dir2::NEG_Y, Dir2::X, Dir2::Y, Dir2::NEG_X] {
        for radius in [0.0, 2.0, 4.0, 8.0] {
            let position = Vec3::new(heading.x * radius, 2.0, heading.y * radius);
            for seed in [0, 1, 2, 42, u64::MAX] {
                qualify(&pilot, seed, position, heading);
            }
        }
    }
}
