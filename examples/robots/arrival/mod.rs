//! Frozen fifteen-feature travel inference shared by evaluation and playback.
mod encoding;
mod environment;
mod evaluation;
use super::{model, session::Session};
use bevy_gym::training::{RecurrentPpoError, RecurrentPpoPolicy};
pub(crate) use encoding::encode;
pub(crate) use environment::TravelTask;
pub(crate) use evaluation::evaluate;
use std::num::NonZeroU16;

/// Load the distinct fifteen-feature actor and critic without accepting older travel profiles.
pub(crate) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(
        bytes,
        15,
        15,
        1,
        &[0.0; 4],
        &[1.0; 4],
        &model::learning_config(),
    )
}

/// Validate weights before constructing an episode with its own recurrent state.
pub(crate) fn load(
    bytes: Vec<u8>,
    seed: u64,
) -> Result<Session<TravelTask, 15>, RecurrentPpoError> {
    let policy = load_policy(bytes)?;
    Ok(Session::from_policy(
        policy,
        TravelTask::new(),
        seed,
        NonZeroU16::new(500).expect("positive arrival horizon"),
        encode,
    ))
}
