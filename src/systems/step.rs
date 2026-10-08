use std::collections::HashMap;
use std::sync::Mutex;

use bevy::prelude::*;

use crate::components::{CurrentObservation, EnvComponent, EnvId, EnvStats, QueuedAction};
use crate::events::{
    ActionRequest, ActionResponse, EpisodeEndEvent, EpisodeFinished, TransitionEvent,
};
use crate::{Env, Transition};

/// Query data needed to advance one environment by one action.
type StepQuery<'world, 'state, E> = Query<
    'world,
    'state,
    (
        Entity,
        &'static EnvId,
        &'static mut EnvComponent<E>,
        &'static mut QueuedAction<E>,
        &'static mut CurrentObservation<E>,
        &'static mut EnvStats,
    ),
>;

/// Advance environments with queued actions during `FixedUpdate`.
///
/// Discard multiple responses for the same entity. Step environments in parallel,
/// then publish each transition before its next-action request or completion.
/// Private completion messages keep automatic resets independent of public readers.
///
/// For `r` responses and `n` environments, bookkeeping takes expected O(r + n) time
/// and O(r + n) extra storage, excluding environment work and observation/action data.
pub(crate) fn step_system<E: Env + Send + Sync + 'static>(
    mut responses: MessageReader<'_, '_, ActionResponse<E>>,
    mut query: StepQuery<'_, '_, E>,
    mut transition_writer: MessageWriter<'_, TransitionEvent<E>>,
    mut episode_writer: MessageWriter<'_, EpisodeEndEvent>,
    mut finished_writer: MessageWriter<'_, EpisodeFinished<E>>,
    mut action_writer: MessageWriter<'_, ActionRequest<E>>,
) {
    let mut actions_by_entity: HashMap<Entity, Option<E::Action>> = HashMap::new();
    for response in responses.read() {
        actions_by_entity
            .entry(response.entity)
            .and_modify(|action| *action = None)
            .or_insert_with(|| Some(response.action.clone()));
    }

    for (entity, _, _, mut queued, _, _) in &mut query {
        if let Some(action) = actions_by_entity.remove(&entity) {
            queued.action = action;
        }
    }

    let outcomes = Mutex::new(Vec::new());
    query.par_iter_mut().for_each(
        |(entity, id, mut env_comp, mut pending, mut obs, mut stats)| {
            if let Some(outcome) = step_environment(
                entity,
                id.0,
                &mut env_comp.env,
                &mut pending,
                &mut obs,
                &mut stats,
            ) {
                outcomes
                    .lock()
                    .expect("step outcome mutex poisoned")
                    .push(outcome);
            }
        },
    );

    // Publish after parallel stepping, while Bevy's message writers are exclusive.
    for outcome in outcomes.into_inner().expect("step outcome mutex poisoned") {
        emit_step(
            outcome,
            &mut transition_writer,
            &mut episode_writer,
            &mut finished_writer,
            &mut action_writer,
        );
    }
}

/// One step's owned transition and the episode statistics captured alongside it.
struct StepOutcome<E: Env> {
    /// Entity that owns this environment.
    entity: Entity,
    /// Index within this environment's pool.
    env_id: usize,
    /// Authoritative action, observations, reward, and completion status.
    transition: Transition<E::Observation, E::Action, E::Info>,
    /// Accumulated reward before a possible automatic reset.
    episode_reward: f64,
    /// Accumulated step count before a possible automatic reset.
    episode_steps: usize,
    /// Environment-provided metrics, populated only at completion.
    episode_extras: HashMap<String, f64>,
}

/// Consume one queued action and capture its result before any reset can run.
fn step_environment<E: Env + Send + Sync + 'static>(
    entity: Entity,
    env_id: usize,
    env: &mut E,
    pending: &mut QueuedAction<E>,
    observation: &mut CurrentObservation<E>,
    stats: &mut EnvStats,
) -> Option<StepOutcome<E>> {
    let action = pending.action.take()?;
    let previous = observation.observation.clone();
    let result = env.step(action.clone());
    observation.observation = result.observation.clone();
    observation.info = result.info.clone();
    stats.record_step(result.reward);

    // Completion metrics belong to the finished episode, before counters reset.
    let episode_extras = if result.is_done() {
        env.episode_extras()
    } else {
        HashMap::new()
    };
    Some(StepOutcome {
        entity,
        env_id,
        transition: Transition {
            observation: previous,
            action,
            reward: result.reward,
            next_observation: result.observation,
            status: result.status,
            info: result.info,
        },
        episode_reward: stats.episode_reward,
        episode_steps: stats.episode_steps,
        episode_extras,
    })
}

/// Publish the transition, then its next-action request or completion messages.
fn emit_step<E: Env + Send + Sync + 'static>(
    outcome: StepOutcome<E>,
    transition_writer: &mut MessageWriter<'_, TransitionEvent<E>>,
    episode_writer: &mut MessageWriter<'_, EpisodeEndEvent>,
    finished_writer: &mut MessageWriter<'_, EpisodeFinished<E>>,
    action_writer: &mut MessageWriter<'_, ActionRequest<E>>,
) {
    let status = outcome.transition.status;
    // Derive the follow-up from the transition rather than storing duplicate state.
    let next_action = if status.is_done() {
        None
    } else {
        Some(ActionRequest {
            entity: outcome.entity,
            env_id: outcome.env_id,
            observation: outcome.transition.next_observation.clone(),
            info: outcome.transition.info.clone(),
        })
    };
    transition_writer.write(TransitionEvent {
        env_id: outcome.env_id,
        entity: outcome.entity,
        transition: outcome.transition,
    });
    if let Some(request) = next_action {
        action_writer.write(request);
    } else {
        // Public reporting consumers cannot drain the private lifecycle signal.
        finished_writer.write(EpisodeFinished::new(outcome.entity));
        episode_writer.write(EpisodeEndEvent {
            env_id: outcome.env_id,
            status,
            total_reward: outcome.episode_reward,
            episode_steps: outcome.episode_steps,
            extras: outcome.episode_extras,
        });
    }
}
