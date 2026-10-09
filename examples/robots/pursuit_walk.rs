//! Advance the same collision-aware character used by the playable pursuit scene.
//!
//! Run `cargo run --features robots --example pursuit-walk`.

#[expect(
    dead_code,
    reason = "This guide uses movement only; the shared helper also serves rendering and sight queries."
)]
#[path = "pursuit/arena.rs"]
mod arena;

use arena::{Arena, Movement};

/// A controller supplies one movement choice for each fixed simulation action.
fn main() {
    let mut arena = Arena::default();
    for _ in 0..100 {
        arena.step(Movement::Forward);
    }
    let position = arena.position();
    println!("Character position after two seconds: {position:?}");
    arena.reset();
}
