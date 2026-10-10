//! Inspect the articulated standing reset contract without commanding an agent.

use bevy_gym::robots::{DroidBody, DroidStanding};
use bevy_gym::Env;

/// Print the authoritative physics body centres for a reproducible reset.
fn main() {
    let observation = DroidStanding::default().reset(Some(42)).observation;
    for body in DroidBody::ALL {
        let position = observation.body(body).position();
        println!("{body:?}: {position:?} m");
    }
}
