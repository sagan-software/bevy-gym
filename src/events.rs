use std::collections::HashMap;

use bevy::prelude::*;

use crate::{Env, EpisodeStatus, Transition};

/// Fired after every successful environment step.
#[derive(Message, Clone)]
pub struct TransitionEvent<E: Env + Send + Sync + 'static> {
    /// Which environment instance produced this transition.
    pub env_id: usize,

    /// Entity that owns the environment components.
    pub entity: Entity,

    /// Full transition data.
    pub transition: Transition<E::Observation, E::Action, E::Info>,
}

/// Fired when an episode ends, whether by termination or truncation.
///
/// Carries the final episode statistics. Useful for logging training
/// progress without having to subscribe to every transition.
#[derive(Message, Debug, Clone)]
pub struct EpisodeEndEvent {
    /// Which environment instance finished.
    pub env_id: usize,

    /// How the episode ended.
    pub status: EpisodeStatus,

    /// Total undiscounted reward accumulated during the episode.
    pub total_reward: f64,

    /// Number of steps the episode lasted.
    pub episode_steps: usize,

    /// Optional per-episode metrics from the environment.
    pub extras: HashMap<String, f64>,
}

/// Request for the next action for one environment.
#[derive(Message, Clone)]
pub struct ActionRequest<E: Env + Send + Sync + 'static> {
    /// Which environment instance needs an action.
    pub env_id: usize,

    /// Entity that should receive the action response.
    pub entity: Entity,

    /// Observation to pass to the policy.
    pub observation: E::Observation,

    /// Auxiliary observation metadata.
    pub info: E::Info,
}

/// Response carrying an action for one environment entity.
#[derive(Message, Clone)]
pub struct ActionResponse<E: Env + Send + Sync + 'static> {
    /// Entity from the matching [`ActionRequest`].
    pub entity: Entity,

    /// Action to apply on the next runner step.
    pub action: E::Action,
}
