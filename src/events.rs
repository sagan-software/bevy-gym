use std::collections::HashMap;
use std::marker::PhantomData;

use bevy::prelude::*;

use crate::{Env, EpisodeStatus, Transition};

/// Fired after every successful environment step.
#[derive(Message, Debug, Clone)]
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
#[derive(Message, Debug, Clone)]
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
#[derive(Message, Debug, Clone)]
pub struct ActionResponse<E: Env + Send + Sync + 'static> {
    /// Entity from the matching [`ActionRequest`].
    pub entity: Entity,

    /// Action to apply on the next runner step.
    pub action: E::Action,
}

/// Private lifecycle signal that public reporting consumers cannot drain.
#[derive(Message, Debug)]
pub(crate) struct EpisodeFinished<E: Env + Send + Sync + 'static> {
    /// Exact entity that produced the terminal transition.
    pub(crate) entity: Entity,
    /// Keeps different environment types in separate message queues.
    environment: PhantomData<fn() -> E>,
}

impl<E: Env + Send + Sync + 'static> EpisodeFinished<E> {
    /// Record one completed environment without copying its observation or action.
    pub(crate) const fn new(entity: Entity) -> Self {
        Self {
            entity,
            environment: PhantomData,
        }
    }
}
