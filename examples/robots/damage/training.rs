//! Damage-aware policy inputs and curriculum tasks, separate from healthy checkpoints.

#[path = "training/encoding.rs"]
mod encoding;
#[path = "training/environment.rs"]
mod environment;

#[path = "training/lesson.rs"]
mod lesson;
pub(crate) use lesson::Lesson;

/// Add actuator health while retaining the healthy recipe's motion features.
pub(crate) fn encode(observation: bevy_gym::robots::DroneObservation) -> [f32; 16] {
    encoding::with_motor_health(observation, crate::learning::encode(observation))
}
pub(crate) use environment::DamageTask;

use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoError, RecurrentPpoPolicy, SeedConfig};

/// Initialize a sixteen-input actor and critic with the existing PPO recipe.
pub(crate) fn new_agent(seed: u64) -> Result<RecurrentPpoAgent, RecurrentPpoError> {
    RecurrentPpoAgent::new(
        16,
        16,
        1,
        &[0.0; 4],
        &[1.0; 4],
        crate::learning::learning_config(),
        SeedConfig::from_root(seed),
    )
}

/// Restore only a damage-aware checkpoint with this lesson's sixteen inputs.
pub(crate) fn load_policy(bytes: Vec<u8>) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(
        bytes,
        16,
        16,
        1,
        &[0.0; 4],
        &[1.0; 4],
        &crate::learning::learning_config(),
    )
}
