//! See through a window, remember a lost target, and forget stale information.
//!
//! Run `cargo run --features robots --example pursuit-sight`.

#[expect(
    dead_code,
    reason = "This guide queries sight; pursuit-walk demonstrates movement."
)]
#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/sight.rs"]
mod sight;

use arena::Arena;
use bevy::math::{Dir3, Vec3};
use sight::{Contact, Sight};
use std::time::Duration;

/// The actor receives a measured point, never an occluded character position.
fn main() {
    let arena = Arena::default();
    let mut sight = Sight::default();
    let eye = Vec3::new(-5.0, 1.7, 3.0);

    // Look through the house window. The first exposed body point becomes visible.
    let character = Vec3::new(-5.0, 1.1, -2.0);
    let visible = sight.sample(&arena, eye, Dir3::NEG_Z, character, Duration::ZERO);
    assert!(matches!(visible, Contact::Visible(_)));
    println!("Through the window: {visible:?}");

    // The character moves behind the wall. Only the previous sighting survives.
    let hidden = Vec3::new(-7.0, 1.1, -2.0);
    sight.sample(&arena, eye, Dir3::NEG_Z, hidden, Duration::from_secs(1));
    let remembered = sight.contact();
    assert!(matches!(remembered, Contact::Remembered { .. }));
    println!("Behind the wall: {remembered:?}");

    // Three seconds without sight removes the position from the observation.
    let expired = sight.sample(&arena, eye, Dir3::NEG_Z, hidden, Duration::from_secs(2));
    assert_eq!(expired, Contact::Unknown);
    println!("After memory expires: {expired:?}");

    // Episode reset explicitly discards every previous sighting.
    sight.forget();
}
