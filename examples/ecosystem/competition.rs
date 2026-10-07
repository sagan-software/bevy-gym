//! Stage 6: multi-agent competition for finite resources.
//!
//! Several `Avian2D` bunny bodies act simultaneously in one world. They collide,
//! block, and push one another while competing for spawned food and well water.
//! All valid bunny sequences train one parameter-sharing recurrent PPO policy. A
//! qualifying curriculum run initializes that policy from stage 1.
//!
//! Running without arguments starts live training in the visual ecosystem:
//!
//! ```text
//! cargo run --example ecosystem-competition
//! ```
//!
//! Headless curriculum training is explicit:
//!
//! ```text
//! cargo run --no-default-features --release --example ecosystem-competition -- train --resume <survival-best.mpk>
//! cargo run --release --example ecosystem-competition -- eval --checkpoint <run-or-mpk>
//! ```
//!
//! The default render feature supplies the Inspector-egui HUD, camera controls,
//! checkpoint playback, and checkpoint video generation.

use tokio as _;

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Competition)
}
