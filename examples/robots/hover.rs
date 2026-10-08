//! Hold a drone aloft with four balanced motor commands.
//!
//! Run with `cargo run --no-default-features --features robots --example drone-hover`.
//! This constant command is a physics baseline. It does not learn or correct drift.

use bevy_gym::robots::{DroneAction, DroneHover};
use bevy_gym::{Env, TimeLimit};

/// Run ten seconds of hovering, then report the resulting position.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One action lasts 20 ms. The wrapper ends the episode after 500 actions.
    let mut drone = TimeLimit::new(DroneHover::default(), 500)?;
    let initial = drone.reset(Some(42)).observation.position();
    println!("Initial position in metres: {initial:?}");

    // Motor order: front-left, front-right, rear-right, rear-left.
    // Half power on all four motors balances this upright drone's weight.
    let hover = DroneAction::try_from([0.5; 4])?;
    loop {
        let step = drone.step(hover);
        if step.is_done() {
            let position = step.observation.position();
            println!("Final position in metres: {position:?}");
            break;
        }
    }
    Ok(())
}
