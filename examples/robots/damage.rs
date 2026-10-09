//! Disable one motor during a seeded flight.
//!
//! Run with `cargo run --no-default-features --features robots --example drone-damage`.
//! This actuator-failure baseline keeps the body mass and collision shape unchanged.
//! Constant commands cannot recover from the failure; this example does not learn.

use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor};
use bevy_gym::Env;

/// Hover for two seconds, fail the front-left motor, then observe the crash.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut drone = DroneHover::default();
    drone.reset(Some(42));
    let hover = DroneAction::try_from([0.5; 4])?;

    // One action lasts 20 ms. Failure removes thrust and reaction torque.
    for _ in 0..100 {
        drone.step(hover);
    }
    drone.fail_motor(DroneMotor::FrontLeft)?;

    // The remaining three motors still receive their original half-power commands.
    for elapsed in 1..=500 {
        let step = drone.step(hover);
        if step.is_done() {
            let position = step.observation.position();
            println!("Episode ended {elapsed} actions after failure at {position:?} metres");
            break;
        }
    }
    Ok(())
}
