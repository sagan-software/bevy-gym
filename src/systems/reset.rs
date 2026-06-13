use bevy::prelude::*;
use rl_traits::Environment;

use crate::components::{CurrentObservation, EnvId, EnvStats, EnvironmentComponent, PendingAction};
use crate::events::{ActionRequestEvent, EpisodeEndEvent};

/// Query data needed to reset an environment after episode completion.
type AutoResetQuery<'world, 'state, E> = Query<
    'world,
    'state,
    (
        Entity,
        &'static EnvId,
        &'static mut EnvironmentComponent<E>,
        &'static mut CurrentObservation<E>,
        &'static mut EnvStats,
        &'static mut PendingAction<E>,
    ),
>;

/// Watches for `EpisodeEndEvent`s and automatically resets the
/// corresponding environment, then fires `ActionRequestEvent` so the
/// policy knows to provide the first action of the new episode.
///
/// This runs in `FixedUpdate`, ordered *after* `step_system`. The reset
/// happens within the same tick that the episode ended, so there is never
/// a tick where an environment sits idle between episodes.
pub fn auto_reset_system<E: Environment + Send + Sync + 'static>(
    mut episode_end_events: MessageReader<'_, '_, EpisodeEndEvent>,
    mut query: AutoResetQuery<'_, '_, E>,
    mut action_req_writer: MessageWriter<'_, ActionRequestEvent>,
) {
    for event in episode_end_events.read() {
        for (entity, id, mut env_comp, mut obs, mut stats, mut pending) in query.iter_mut() {
            if id.0 != event.env_id {
                continue;
            }

            let (new_obs, new_info) = env_comp.env.reset(None);
            obs.observation = new_obs;
            obs.info = new_info;
            pending.action = None;
            stats.record_episode_end();

            action_req_writer.write(ActionRequestEvent {
                env_id: id.0,
                entity,
            });

            break;
        }
    }
}

/// Marker component for environments that should be reset on the next tick.
///
/// Add this component to an environment entity to trigger a manual reset.
/// The reset system will remove the marker after resetting.
///
/// Useful for curriculum learning (reset to a specific state), evaluation
/// (reset with a fixed seed), or recovering from invalid states.
#[derive(Component, Debug, Clone, Copy)]
pub struct ResetRequested {
    /// Optional seed for deterministic reset. `None` for random.
    pub seed: Option<u64>,
}

/// Query data needed to perform a manually requested environment reset.
type ManualResetQuery<'world, 'state, E> = Query<
    'world,
    'state,
    (
        Entity,
        &'static EnvId,
        &'static mut EnvironmentComponent<E>,
        &'static mut CurrentObservation<E>,
        &'static mut EnvStats,
        &'static mut PendingAction<E>,
        &'static ResetRequested,
    ),
>;

/// Handles manually-requested resets via the `ResetRequested` marker component.
pub fn manual_reset_system<E: Environment + Send + Sync + 'static>(
    mut commands: Commands<'_, '_>,
    mut query: ManualResetQuery<'_, '_, E>,
    mut action_req_writer: MessageWriter<'_, ActionRequestEvent>,
) {
    for (entity, id, mut env_comp, mut obs, mut stats, mut pending, reset_req) in query.iter_mut() {
        let (new_obs, new_info) = env_comp.env.reset(reset_req.seed);
        obs.observation = new_obs;
        obs.info = new_info;
        pending.action = None;
        stats.record_episode_end();

        action_req_writer.write(ActionRequestEvent {
            env_id: id.0,
            entity,
        });

        commands.entity(entity).remove::<ResetRequested>();
    }
}
