//! One architecture for recovery training, checkpoint loading, and visual inference.

use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy};

/// Discount per 20 ms action, without units.
pub(crate) const GAMMA: f32 = 0.995;
/// Generalized-advantage trace decay, without units.
pub(crate) const GAE_LAMBDA: f32 = 0.95;

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

/// Restore a checkpoint using this lesson's exact network architecture.
pub(crate) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(bytes, 12, 12, 1, &[0.0; 4], &[1.0; 4], &learning_config())
}
