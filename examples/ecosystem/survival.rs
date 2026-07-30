//! Stage 1: single-agent ecosystem survival.
//!
//! One bunny receives only local ray perception, physiology, and its Burn LSTM
//! state. It must find spawned food and a finite refillable well.
//! Satiation and hydration eventually reach zero and reduce HP.
//!
//! Running without arguments starts live training in the visual ecosystem:
//!
//! ```text
//! cargo run --example ecosystem-survival
//! ```
//!
//! Headless training is explicit. Checkpoint playback and video generation use
//! the default `render` feature:
//!
//! ```text
//! cargo run --no-default-features --release --example ecosystem-survival -- train
//! cargo run --release --example ecosystem-survival -- eval --checkpoint <run-or-mpk>
//! cargo run --release --example ecosystem-survival -- watch --checkpoint <run-or-mpk>
//! cargo run --release --example ecosystem-survival -- video --checkpoint <run-dir>
//! ```
//!
//! See [`examples/ecosystem/README.md`](README.md) for the observation,
//! evidence, and video contracts.

#[path = "shared/mod.rs"]
mod shared;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    shared::run(shared::CurriculumStage::Survival)
}
