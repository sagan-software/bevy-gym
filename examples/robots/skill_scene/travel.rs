//! Bind the travel scene to the same sampled task and encoder used by PPO training.

#[path = "../travel/encoding.rs"]
pub(crate) mod encoding;
#[path = "../travel/environment.rs"]
pub(crate) mod environment;
#[expect(
    dead_code,
    reason = "The scene shares the stage type; selection-only helpers are exercised by curriculum tests."
)]
#[path = "../travel/stage.rs"]
pub(crate) mod stage;

use bevy_gym::training::{RecurrentPpoError, RecurrentPpoPolicy};
use std::num::NonZeroU16;

use super::{encoding as motor_encoding, model, session::Session};

/// Validate the exact thirteen-input architecture before creating a physical scene.
pub(crate) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(
        bytes,
        13,
        13,
        1,
        &[0.0; 4],
        &[1.0; 4],
        &model::learning_config(),
    )
}

/// Load a frozen travel actor and an independent episode with no training capability.
pub(crate) fn load(
    bytes: Vec<u8>,
    seed: u64,
) -> Result<Session<environment::TravelTask, 13>, RecurrentPpoError> {
    let policy = load_policy(bytes)?;
    Ok(Session::from_policy(
        policy,
        environment::TravelTask::factory(stage::Stage::Near)(),
        seed,
        NonZeroU16::new(1_000).expect("positive travel horizon"),
        encoding::encode,
    ))
}
