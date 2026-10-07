//! Stage 3: single-agent gorge bridge crossing.
//!
//! One bunny retains ephemeral-food sprinting while food alternates between
//! opposite banks. The center gorge is lethal outside its visible bridge.
//!
//! ```text
//! cargo run --example ecosystem-gorge
//! cargo run --no-default-features --release --example ecosystem-gorge -- train
//! ```

use shakmaty as _;
use tokio as _;

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Gorge)
}
