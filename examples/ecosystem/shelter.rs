//! Stage 5: single-agent food, water, and shelter regulation.
//!
//! The survival bunny retains foraging and drinking while learning to reduce
//! environmental exposure inside shelter during adverse weather.
//!
//! ```text
//! cargo run --example ecosystem-shelter
//! cargo run --no-default-features --release --example ecosystem-shelter -- train
//! ```

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Shelter)
}
