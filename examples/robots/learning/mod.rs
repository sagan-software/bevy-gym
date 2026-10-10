//! Shared mechanics behind the recovery training and inference lessons.

pub(crate) mod encoding;
mod episode;
pub(crate) mod evaluation;
mod model;
mod rollout;

use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoError, SeedConfig};

pub(crate) use encoding::{decode_action, encode};
pub(crate) use evaluation::{baseline, evaluate, SELECTION_SEEDS};
pub(crate) use model::{learning_config, load_policy, GAE_LAMBDA, GAMMA};
pub(crate) use rollout::RecoveryBatch;

/// Initialize a reproducible actor and critic without demonstrations.
pub(crate) fn new_agent(seed: u64) -> Result<RecurrentPpoAgent, RecurrentPpoError> {
    RecurrentPpoAgent::new(
        12,
        12,
        1,
        &[0.0; 4],
        &[1.0; 4],
        learning_config(),
        SeedConfig::from_root(seed),
    )
}
