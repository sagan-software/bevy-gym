//! Stage 8: predator-prey curriculum with procedural hazards.
//!
//! This stage transfers both stage-3 policies into procedurally generated maps.
//! Trees and rocks are solid `Avian2D` obstacles. Thorn bushes are traversable
//! sensors that reduce health and hit points. The evaluation checks obstacle
//! penetration, retained survival, and reduced thorn exposure without assuming
//! that entering thorns is always wrong during a predator escape.
//!
//! Running without arguments starts live training in the visual ecosystem:
//!
//! ```text
//! cargo run --example ecosystem-obstacles
//! ```
//!
//! Headless curriculum training is explicit:
//!
//! ```text
//! cargo run --no-default-features --release --example ecosystem-obstacles -- train --resume <predator-prey-bunny.mpk> --fox-resume <predator-prey-fox.mpk>
//! cargo run --release --example ecosystem-obstacles -- eval --checkpoint <run-dir>
//! ```
//!
//! The default render feature supplies the Inspector-egui HUD, camera controls,
//! checkpoint playback, and checkpoint video generation.

use tokio as _;

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Obstacles)
}
