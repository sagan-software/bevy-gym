//! Hear an event without learning the hidden emitter's exact position.
//!
//! Run `cargo run --features robots --example pursuit-hearing`.

#[expect(
    dead_code,
    reason = "This guide uses the arena's obstruction query only."
)]
#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/hearing.rs"]
mod hearing;

use arena::Arena;
use bevy::math::Vec3;
use hearing::{Hearing, Noise};
use std::time::Duration;

/// Emit one shot behind a wall and let its coarse bearing expire.
fn main() {
    let arena = Arena::default();
    let mut hearing = Hearing::default();
    let listener = Vec3::new(-7.0, 1.7, 3.0);
    let source = Vec3::new(-7.0, 1.7, -2.0);

    // The wall halves range: a five-metre footstep is silent, but a shot is audible.
    hearing.hear(&arena, listener, source, Noise::Footstep);
    assert!(hearing.latest().is_none());
    hearing.hear(&arena, listener, source, Noise::Gunshot);
    let cue = hearing.latest().expect("Audible shot");
    let noise = cue.noise().label();
    let bearing = cue.bearing().label();
    println!("Heard {noise} toward {bearing}");

    // No event means no new information, even if the hidden character moves.
    hearing.advance(Duration::from_secs(2));
    assert!(hearing.latest().is_none());

    // Reset clears hearing independently of sight or flight.
    hearing.forget();
}
