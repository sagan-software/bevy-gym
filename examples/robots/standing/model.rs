//! Fixed standing actor and critic architecture.

use super::encoding::FEATURES;
use crate::learning::{GAE_LAMBDA, GAMMA};
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};
use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy};

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
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
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

/// Import the same actor/critic architecture while resetting both Adam optimizers.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn warm_start(
    seed: u64,
    bytes: Vec<u8>,
) -> Result<RecurrentPpoAgent, RecurrentPpoError> {
    RecurrentPpoAgent::load_bytes(
        bytes,
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

/// Initialize the original architecture with an explicitly recorded native optimizer recipe.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn new_agent_configured(
    seed: u64,
    profile: super::optimization::OptimizerProfile,
) -> Result<RecurrentPpoAgent, RecurrentPpoError> {
    let mut agent = RecurrentPpoAgent::new(
        FEATURES,
        FEATURES,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        profile.configure(configuration()),
        SeedConfig::from_root(seed),
    )?;
    // If gentle initialization is selected, change fresh parameters before collecting any actions.
    if profile == super::optimization::OptimizerProfile::GentleV1 {
        agent.reduce_action_mean_head();
    }
    Ok(agent)
}

/// Import mean/critic weights with fresh Adam and the explicitly selected exploration recipe.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn warm_start_configured(
    seed: u64,
    bytes: Vec<u8>,
    profile: super::optimization::OptimizerProfile,
) -> Result<RecurrentPpoAgent, RecurrentPpoError> {
    // Reject fresh-only initialization before loading or changing any learned parameters.
    if profile == super::optimization::OptimizerProfile::GentleV1 {
        return Err(RecurrentPpoError::InvalidConfig {
            field: "optimizer_profile",
            reason: "gentle-v1 requires fresh initialization",
        });
    }
    let mut agent = RecurrentPpoAgent::load_bytes(
        bytes,
        FEATURES,
        FEATURES,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        profile.configure(configuration()),
        SeedConfig::from_root(seed),
    )?;
    if matches!(profile, super::optimization::OptimizerProfile::LowNoiseV1) {
        agent.reset_exploration();
    }
    Ok(agent)
}
