//! Fly to a position and face east using a bundled imitation-trained pilot.
//!
//! Run with `cargo run --no-default-features --features robots --example drone-track`.
//! This lesson controls healthy motors in empty space; it does not avoid obstacles.

mod flight_control;

use bevy::math::{Dir2, Vec3};
use bevy_gym::{robots::DroneHover, Env, TimeLimit};
use flight_control::{FlightGoal, FlightPilot};

/// Run ten seconds of flight from a disturbed starting pose.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pilot = FlightPilot::bundled()?;
    // Positions use metres. Dir2::X faces east along the world's positive X axis.
    let goal = FlightGoal::try_from((Vec3::new(8.0, 2.0, 0.0), Dir2::X))?;
    let mut drone = TimeLimit::new(DroneHover::disturbed(), 500)?;
    let mut observation = drone.reset(Some(42)).observation;

    // The pilot turns the goal into four motor commands, once every 20 ms.
    loop {
        let action = pilot.action(observation, goal)?;
        let step = drone.step(action);
        observation = step.observation;
        if step.is_done() {
            break;
        }
    }
    let position = observation.position();
    println!("Final position in metres: {position:?}");
    Ok(())
}
