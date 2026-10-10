//! Shared physical standing training and frozen inference.

pub(crate) mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
mod cli;
pub(crate) mod encoding;
pub(crate) mod evaluation;
pub(crate) mod model;
pub(crate) mod reward;
/// Frozen execution shared by physical inspection and rendered playback.
#[path = "../standing_scene/session.rs"]
pub(crate) mod session;
/// Stream one frozen episode as read-only diagnostic evidence.
#[cfg(not(target_arch = "wasm32"))]
mod trace;
#[cfg(not(target_arch = "wasm32"))]
mod training;
/// Frozen six-agent inference over one shared physical world.
#[path = "../world_scene/session.rs"]
pub(crate) mod world_session;

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

/// Collect using a fixed reward profile while retaining the original action/observation contract.
pub(crate) fn batch_with_recipe(
    seed: u64,
    policy: &RecurrentPpoPolicy,
    recipe: reward::Recipe,
) -> RecoveryBatch<reward::TrainingTask, 204> {
    RecoveryBatch::with_actions(
        seed,
        policy,
        recipe.factory(),
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
