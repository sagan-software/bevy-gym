//! Stage 7: shared-policy predator and prey training.
//!
//! Bunnies eat spawned food. Foxes eat bunnies. Both species must drink from
//! the finite refillable well. Joint actions are applied simultaneously, then
//! every bunny transition updates one shared bunny learner and every fox
//! transition updates one shared fox learner. The bunny policy transfers from
//! stage 2; the fox policy begins as a separate role-specific policy.
//!
//! Running without arguments starts live training in the visual ecosystem:
//!
//! ```text
//! cargo run --example ecosystem-predator-prey
//! ```
//!
//! Headless curriculum training is explicit:
//!
//! ```text
//! cargo run --no-default-features --release --example ecosystem-predator-prey -- train --resume <competition-best.mpk>
//! cargo run --release --example ecosystem-predator-prey -- eval --checkpoint <run-dir>
//! ```
//!
//! The default render feature supplies the Inspector-egui HUD, camera controls,
//! checkpoint playback, and checkpoint video generation.

use tokio as _;

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::PredatorPrey)
}
