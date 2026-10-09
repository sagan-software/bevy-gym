//! Follow a sensed robot with programmed search and learned flight.
//!
//! Run `cargo run --features robots --example pursuit-search`.

#[expect(
    dead_code,
    reason = "This guide keeps the character still; pursuit-walk covers movement."
)]
#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/camera.rs"]
mod camera;
#[path = "flight_control/mod.rs"]
mod flight_control;
#[path = "pursuit/navigation/mod.rs"]
mod navigation;
#[path = "pursuit/sight.rs"]
mod sight;

use arena::{layout::Surface, Arena, Movement};
use bevy::math::Dir3;
use bevy_gym::{
    robots::{DroneHover, DroneObstacle},
    Env,
};
use flight_control::FlightPilot;
use navigation::Navigator;
use sight::{Contact, Sight};
use std::time::Duration;

/// Only the sensor reads the robot's position; navigation receives filtered contact.
fn main() {
    let mut arena = Arena::default();
    let obstacles = arena
        .blocks()
        .iter()
        .filter(|block| block.surface != Surface::Floor)
        .map(|block| {
            DroneObstacle::try_from((block.centre, block.half, block.rotation))
                .expect("Authored static box")
        });
    let mut drone = DroneHover::with_obstacles(obstacles);
    let pilot = FlightPilot::bundled().expect("Bundled learned motor pilot");
    let mut search = Navigator::default();
    let mut sight = Sight::default();

    // Each reset clears observations while retaining the map and learned weights.
    for episode in 0..2 {
        arena.reset();
        drone.reset(Some(42));
        search.reset();
        sight.forget();
        let mut visible = 0;
        for _ in 0..600 {
            let goal = search.goal(drone.observation(), sight.contact());
            // Hold each navigation goal for five 20-millisecond motor actions.
            for _ in 0..5 {
                arena.step(Movement::Idle);
                let action = pilot
                    .action(drone.observation(), goal)
                    .expect("Motor action");
                let step = drone.step(action);
                assert!(!step.is_done(), "Drone crashed");
                let own = step.observation;
                let eye = own.position() + own.orientation() * camera::EYE_OFFSET;
                let contact = sight.sample(
                    &arena,
                    eye,
                    own.orientation() * Dir3::NEG_Z,
                    arena.position(),
                    Duration::from_millis(20),
                );
                visible += usize::from(matches!(contact, Contact::Visible(_)));
            }
        }
        assert!(
            visible >= 1800,
            "Tracking requires at least 60% visible samples"
        );
        println!("Episode {episode}: {visible}/3000 visible samples");
    }
}
