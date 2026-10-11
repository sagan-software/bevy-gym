//! Inspect a walking-task reset without choosing actions or claiming a learned skill.

use bevy_gym::robots::{DroidBody, DroidWalking};
use bevy_gym::Env;

/// Print the requested velocity and physical body centres without advancing physics.
fn main() {
    let observation = DroidWalking::default().reset(Some(42)).observation;
    let velocity = observation.command().velocity();
    println!("Walking command: {velocity:?} m/s (X/Z)");
    for body in DroidBody::ALL {
        let position = observation.physical().body(body).position();
        println!("{body:?}: {position:?} m");
    }
}
