//! Compare a disturbed start with the calm hover guide.

use bevy_gym::robots::{DroneAction, DroneHover};
use bevy_gym::{Env, TimeLimit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // This lesson starts with tilt and motion. The physics and motor API stay the same.
    let mut drone = TimeLimit::new(DroneHover::disturbed(), 500)?;
    let initial = drone.reset(Some(42)).observation;
    let velocity = initial.linear_velocity();
    println!("Initial velocity in metres per second: {velocity:?}");

    // Equal thrust balances weight only while upright. This baseline cannot recover.
    let half_thrust = DroneAction::try_from([0.5; 4])?;
    for tick in 1..=500 {
        let result = drone.step(half_thrust);
        if result.is_done() {
            let position = result.observation.position();
            let status = result.status;
            println!("After {tick} actions: {status:?}, position in metres: {position:?}");
            break;
        }
    }
    Ok(())
}
