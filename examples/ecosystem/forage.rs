//! Stage 1: single-agent food foraging.
//!
//! One bunny learns the stable ecosystem action, perception, and recurrent
//! memory contract while seeking food under one active physiological need.
//!
//! ```text
//! cargo run --example ecosystem-forage
//! cargo run --no-default-features --release --example ecosystem-forage -- train
//! ```

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Forage)
}
