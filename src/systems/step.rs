use std::collections::HashMap;
use std::sync::Mutex;

use bevy::prelude::*;

use crate::components::{CurrentObservation, EnvComponent, EnvId, EnvStats, QueuedAction};
use crate::events::{ActionRequest, ActionResponse, EpisodeEndEvent, TransitionEvent};
use crate::{Env, EpisodeStatus, Transition};

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

/// The core RL tick system. Runs in `FixedUpdate`.
///
/// For each environment entity that received a valid `ActionResponse`:
/// 1. Calls `env.step(action)`
/// 2. Updates `CurrentObservation` with the result
/// 3. Updates `EnvStats`
/// 4. Fires `TransitionEvent` with the full transition
/// 5. Fires `EpisodeEndEvent` if the episode is over
/// 6. Fires `ActionRequest` only when the stepped state still expects an action
///
/// # Parallelism
///
/// Each environment entity's components are independent, so we use
/// `par_iter_mut()` for the compute-heavy step phase. However, Bevy's
/// `EventWriter` is not `Send`, so we can't fire events from within
/// the parallel closure directly.
///
/// Solution: collect step results into a `Mutex<Vec>` in parallel,
/// then drain and fire events serially. The serial phase is O(n) over
/// only the envs that actually stepped -- the expensive part ran in parallel.
pub(crate) fn step_system<E: Env + Send + Sync + 'static>(
    mut responses: MessageReader<'_, '_, ActionResponse<E>>,
    mut query: StepQuery<'_, '_, E>,
    mut transition_writer: MessageWriter<'_, TransitionEvent<E>>,
    mut episode_writer: MessageWriter<'_, EpisodeEndEvent>,
    mut action_writer: MessageWriter<'_, ActionRequest<E>>,
) {
    struct StepOutcome<O, A, I> {
        entity: Entity,
        env_id: usize,
        transition: Transition<O, A, I>,
        episode_status: EpisodeStatus,
        next_observation: O,
        next_info: I,
        episode_reward: f64,
        episode_steps: usize,
        episode_extras: HashMap<String, f64>,
    }

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

    let outcomes: Mutex<Vec<StepOutcome<E::Observation, E::Action, E::Info>>> =
        Mutex::new(Vec::new());

    query.par_iter_mut().for_each(
        |(entity, id, mut env_comp, mut pending, mut obs, mut stats)| {
            let Some(action) = pending.action.take() else {
                return;
            };

            let prev_obs = obs.observation.clone();
            let result = env_comp.env.step(action.clone());
            let reward = result.reward;
            let status = result.status;
            let episode_done = status.is_done();

            obs.observation = result.observation.clone();
            obs.info = result.info.clone();

            stats.record_step(reward);

            // Capture stats now -- we can't re-query inside the serial phase
            // since par_iter_mut already has exclusive access to these components.
            let episode_reward = stats.episode_reward;
            let episode_steps = stats.episode_steps;
            let episode_extras = if episode_done {
                env_comp.env.episode_extras()
            } else {
                HashMap::new()
            };

            let transition = Transition {
                observation: prev_obs,
                action,
                reward,
                next_observation: result.observation.clone(),
                status,
                info: result.info.clone(),
            };

            outcomes
                .lock()
                .expect("step outcome mutex poisoned")
                .push(StepOutcome {
                    entity,
                    env_id: id.0,
                    transition,
                    episode_status: status,
                    next_observation: result.observation,
                    next_info: result.info,
                    episode_reward,
                    episode_steps,
                    episode_extras,
                });
        },
    );

    for outcome in outcomes.into_inner().expect("step outcome mutex poisoned") {
        transition_writer.write(TransitionEvent {
            env_id: outcome.env_id,
            entity: outcome.entity,
            transition: outcome.transition,
        });

        if outcome.episode_status.is_done() {
            episode_writer.write(EpisodeEndEvent {
                env_id: outcome.env_id,
                status: outcome.episode_status,
                total_reward: outcome.episode_reward,
                episode_steps: outcome.episode_steps,
                extras: outcome.episode_extras,
            });
        } else {
            action_writer.write(ActionRequest {
                env_id: outcome.env_id,
                entity: outcome.entity,
                observation: outcome.next_observation,
                info: outcome.next_info,
            });
        }
    }
}
