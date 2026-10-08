use bevy::prelude::*;

use crate::components::{CurrentObservation, EnvComponent, EnvId, EnvStats, QueuedAction};
use crate::events::{ActionRequest, EpisodeFinished};
use crate::plugin::ResetSeedSchedule;
use crate::Env;

/// Query data needed to reset an environment after episode completion.
type AutoResetQuery<'world, 'state, E> = Query<
    'world,
    'state,
    (
        Entity,
        &'static EnvId,
        &'static mut EnvComponent<E>,
        &'static mut CurrentObservation<E>,
        &'static mut EnvStats,
        &'static mut QueuedAction<E>,
        &'static ResetSeedSchedule,
    ),
>;

/// Watches for private completion signals and automatically resets the
/// corresponding environment, then fires `ActionRequest` so the
/// policy knows to provide the first action of the new episode.
///
/// This runs in `FixedUpdate`, ordered *after* `step_system`. The reset
/// happens within the same tick that the episode ended, so there is never
/// a tick where an environment sits idle between episodes.
pub(crate) fn auto_reset_system<E: Env + Send + Sync + 'static>(
    mut completions: MessageReader<'_, '_, EpisodeFinished<E>>,
    mut query: AutoResetQuery<'_, '_, E>,
    mut action_writer: MessageWriter<'_, ActionRequest<E>>,
) {
    for event in completions.read() {
        // Environment IDs are local to a pool. The typed message and entity locate
        // exactly one owner, even when different plugins both assign EnvId(0).
        let Ok((entity, id, mut env_comp, mut obs, mut stats, mut pending, seeds)) =
            query.get_mut(event.entity)
        else {
            // A despawned entity or a different environment type cannot be reset.
            continue;
        };

        let episode = stats.total_episodes.saturating_add(1) as u64;
        let reset = env_comp.env.reset(seeds.seed_for(id.0, episode));
        obs.observation = reset.observation.clone();
        obs.info = reset.info.clone();
        pending.action = None;
        stats.record_episode_end();

        action_writer.write(ActionRequest {
            env_id: id.0,
            entity,
            observation: reset.observation,
            info: reset.info,
        });
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
        &'static mut EnvComponent<E>,
        &'static mut CurrentObservation<E>,
        &'static mut EnvStats,
        &'static mut QueuedAction<E>,
        &'static ResetRequested,
    ),
>;

/// Handles manually-requested resets via the `ResetRequested` marker component.
pub(crate) fn manual_reset_system<E: Env + Send + Sync + 'static>(
    mut commands: Commands<'_, '_>,
    mut query: ManualResetQuery<'_, '_, E>,
    mut action_writer: MessageWriter<'_, ActionRequest<E>>,
) {
    for (entity, id, mut env_comp, mut obs, mut stats, mut pending, reset_req) in &mut query {
        let reset = env_comp.env.reset(reset_req.seed);
        obs.observation = reset.observation.clone();
        obs.info = reset.info.clone();
        pending.action = None;
        stats.record_episode_end();

        action_writer.write(ActionRequest {
            env_id: id.0,
            entity,
            observation: reset.observation,
            info: reset.info,
        });

        commands.entity(entity).remove::<ResetRequested>();
    }
}
