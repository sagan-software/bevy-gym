//! Stage 2: single-agent ephemeral-food sprinting.
//!
//! One bunny retains the forage policy while learning to accelerate, steer,
//! and reach visible food before its five-second lifetime ends. Earlier
//! contact earns a larger reward.
//!
//! ```text
//! cargo run --example ecosystem-sprint
//! cargo run --no-default-features --release --example ecosystem-sprint -- train
//! ```

use shakmaty as _;
use tokio as _;

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Sprint)
}
