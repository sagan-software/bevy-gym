//! Fixed standing actor and critic architecture.

use super::encoding::FEATURES;
use crate::learning::{GAE_LAMBDA, GAMMA};
use bevy_gym::training::{
    RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy, SeedConfig,
};

/// Standing uses a 64-unit LSTM and a separate 128/64-unit critic.
///
/// The collector and optimizer use the same discount and GAE constants.
/// Actor and critic receive the same own-body features in this single-agent lesson.
fn configuration() -> RecurrentPpoConfig {
    RecurrentPpoConfig {
        gamma: GAMMA,
        gae_lambda: GAE_LAMBDA,
        critic_learning_rate: 0.001,
        initial_log_std: -2.0,
        entropy_coefficient: 0.001,
        ..RecurrentPpoConfig::default()
    }
}

/// Start the standing actor and critic from seeded random parameters.
pub(crate) fn new_agent(seed: u64) -> Result<RecurrentPpoAgent, RecurrentPpoError> {
    RecurrentPpoAgent::new(
        FEATURES,
        FEATURES,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        configuration(),
        SeedConfig::from_root(seed),
    )
}

/// Restore the exact observation and action architecture without fallback.
pub(crate) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(
        bytes,
        FEATURES,
        FEATURES,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        &configuration(),
    )
}
