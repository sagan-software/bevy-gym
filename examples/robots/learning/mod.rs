//! Shared mechanics behind the recovery training and inference lessons.

mod encoding;
mod evaluation;
mod rollout;

use bevy_gym::training::{
    RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy, SeedConfig,
};

pub(crate) use encoding::{decode_action, encode};
pub(crate) use evaluation::{baseline, evaluate, SELECTION_SEEDS};
pub(crate) use rollout::RecoveryBatch;

/// Discount per 20 ms action, without units.
const GAMMA: f32 = 0.995;
/// Generalized-advantage trace decay, without units.
const GAE_LAMBDA: f32 = 0.95;

/// Construct the fixed network and optimizer recipe used by this lesson.
pub(crate) fn learning_config() -> RecurrentPpoConfig {
    RecurrentPpoConfig {
        actor_hidden_size: 32,
        critic_hidden_sizes: vec![64, 64],
        gamma: GAMMA,
        gae_lambda: GAE_LAMBDA,
        critic_learning_rate: 0.001,
        initial_log_std: -2.0,
        entropy_coefficient: 0.001,
        ..RecurrentPpoConfig::default()
    }
}

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

/// Restore a checkpoint using this lesson's exact network architecture.
pub(crate) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(bytes, 12, 12, 1, &[0.0; 4], &[1.0; 4], &learning_config())
}
