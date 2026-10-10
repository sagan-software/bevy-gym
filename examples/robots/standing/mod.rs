//! Shared physical standing training and frozen inference.

pub(crate) mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
mod cli;
pub(crate) mod encoding;
pub(crate) mod evaluation;
pub(crate) mod model;
#[cfg(not(target_arch = "wasm32"))]
mod training;

use crate::learning::RecoveryBatch;
use bevy_gym::{robots::DroidStanding, training::RecurrentPpoPolicy};
use std::num::NonZeroU16;

/// Start eight independent physical episodes and recurrent memories.
pub(crate) fn batch(seed: u64, policy: &RecurrentPpoPolicy) -> RecoveryBatch<DroidStanding, 204> {
    RecoveryBatch::with_actions(
        seed,
        policy,
        DroidStanding::default,
        encoding::encode,
        |values| Ok(encoding::decode(values)?),
        NonZeroU16::new(1_000).expect("positive standing horizon"),
    )
}

/// Dispatch standalone training or frozen evaluation.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    cli::run()
}
