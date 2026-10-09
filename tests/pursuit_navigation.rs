//! Programmed search consumes filtered sightings without access to hidden actors.
#![cfg(feature = "robots")]

#[expect(
    dead_code,
    reason = "Movement and sight are covered by their dedicated suites."
)]
#[path = "../examples/robots/pursuit/arena.rs"]
mod arena;
#[path = "../examples/robots/pursuit/camera.rs"]
mod camera;
#[path = "../examples/robots/flight_control/mod.rs"]
mod flight_control;
#[path = "../examples/robots/pursuit/navigation/mod.rs"]
mod navigation;
#[expect(
    dead_code,
    reason = "The navigation boundary receives already-filtered contacts."
)]
#[path = "../examples/robots/pursuit/sight.rs"]
mod sight;

use bevy::math::Vec3;
use bevy_gym::{robots::DroneHover, Env};
use navigation::Navigator;
use sight::Contact;
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn a_remembered_sighting_starts_investigation_without_a_visible_sample() {
    let observation = DroneHover::default().reset(Some(42)).observation;
    let mut search = Navigator::default();
    let scanning = search.goal(observation, Contact::Unknown);
    let investigating = search.goal(
        observation,
        Contact::Remembered {
            point: Vec3::new(-5.0, 1.5, -3.0),
            age: Duration::from_secs(1),
        },
    );
    assert!(scanning.position().abs_diff_eq(
        Vec3::new(observation.position().x, 2.0, observation.position().z),
        0.01
    ));
    assert!(investigating.position().distance(scanning.position()) > 0.25);
    flight_control::FlightPilot::bundled()
        .expect("Bundled flight pilot")
        .action(observation, investigating)
        .expect("Valid investigation motor action");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reset_replays_the_same_filtered_contacts_without_rebuilding_the_world() {
    let observation = DroneHover::default().reset(Some(73)).observation;
    let contacts = [
        Contact::Unknown,
        Contact::Visible(Vec3::new(3.0, 1.5, 8.0)),
        Contact::Unknown,
    ];
    let mut search = Navigator::default();
    let expected = contacts.map(|contact| search.goal(observation, contact));
    search.reset();
    let actual = contacts.map(|contact| search.goal(observation, contact));
    assert_eq!(actual, expected);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn stationary_pursuit_survives_a_minute_with_real_sight_and_motor_actions() {
    let mut arena = arena::Arena::default();
    let obstacles = arena
        .blocks()
        .iter()
        .filter(|block| block.surface != arena::layout::Surface::Floor)
        .map(|block| {
            bevy_gym::robots::DroneObstacle::try_from((block.centre, block.half, block.rotation))
                .expect("Authored static box")
        });
    let mut drone = DroneHover::with_obstacles(obstacles);
    drone.reset(Some(42));
    let pilot = flight_control::FlightPilot::bundled().expect("Bundled flight pilot");
    let mut navigation = Navigator::default();
    let mut sight = sight::Sight::default();
    let mut visible = 0;
    for _ in 0..600 {
        let goal = navigation.goal(drone.observation(), sight.contact());
        for _ in 0..5 {
            arena.step(arena::Movement::Idle);
            let action = pilot
                .action(drone.observation(), goal)
                .expect("Valid motor action");
            let step = drone.step(action);
            assert!(!step.is_done());
            let own = step.observation;
            let eye = own.position() + own.orientation() * camera::EYE_OFFSET;
            let contact = sight.sample(
                &arena,
                eye,
                own.orientation() * bevy::math::Dir3::NEG_Z,
                arena.position(),
                Duration::from_millis(20),
            );
            visible += usize::from(matches!(contact, Contact::Visible(_)));
        }
    }
    assert!(
        visible >= 1800,
        "At least 60% visible samples, received {visible}"
    );
}
