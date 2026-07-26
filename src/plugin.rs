use std::marker::PhantomData;
use std::sync::Arc;

use bevy::prelude::*;

use crate::components::{CurrentObservation, EnvComponent, EnvId, EnvStats, QueuedAction};
use crate::events::{ActionRequest, ActionResponse, EpisodeEndEvent, TransitionEvent};
use crate::systems::{
    reset::{auto_reset_system, manual_reset_system},
    step::step_system,
};
use crate::Env;

/// Configuration for how the gym plugin runs.
#[derive(Resource, Debug, Clone, Copy)]
pub struct GymConfig {
    /// Number of parallel environment instances.
    pub num_envs: usize,

    /// Fixed tick rate in Hz. `None` means uncapped fixed-time execution.
    pub tick_rate: Option<f64>,

    /// Whether ended episodes reset automatically in the same fixed update.
    pub auto_reset: bool,
}

/// Deterministic seed schedule used by automatic environment resets.
#[derive(Resource, Clone)]
pub(crate) struct ResetSeedSchedule {
    /// Maps `(environment id, zero-based episode index)` to a reset seed.
    seed_for: Arc<dyn Fn(usize, u64) -> Option<u64> + Send + Sync>,
}

impl ResetSeedSchedule {
    /// Resolve the reset seed for one environment episode.
    pub(crate) fn seed_for(&self, env_id: usize, episode: u64) -> Option<u64> {
        (self.seed_for)(env_id, episode)
    }
}

impl Default for GymConfig {
    fn default() -> Self {
        Self {
            num_envs: 1,
            tick_rate: Some(60.0),
            auto_reset: true,
        }
    }
}

/// System set for ordering RL systems within `FixedUpdate`.
///
/// Ordering: `Step` -> `AutoReset` -> `ManualReset` -> `RequestActions`
///
/// This ensures resets happen in the same tick as episode completion,
/// and manual resets are processed after automatic ones.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GymSet {
    /// Step environments that have pending actions.
    Step,

    /// Reset environments that just finished an episode.
    AutoReset,

    /// Reset environments marked with `ResetRequested`.
    ManualReset,

    /// User policy systems should usually run in this set.
    RequestActions,
}

/// Bevy plugin that runs N parallel RL environments at a fixed tick rate.
///
/// # Usage
///
/// ```rust,ignore
/// use bevy::prelude::*;
/// use bevy_gym::BevyGymPlugin;
///
/// App::new()
///     .add_plugins(BevyGymPlugin::<MyEnv>::new(env_factory).with_envs(16).uncapped())
///     .add_systems(FixedUpdate, my_policy_system.in_set(GymSet::RequestActions))
///     .run();
/// ```
///
/// # Type parameters
///
/// `E` is the environment implementation. The plugin is generic over `E` so
/// observation and action types
/// are known at compile time throughout.
///
/// # Environment factory
///
/// Rather than requiring `E: Default`, the plugin takes a factory closure
/// `Fn(usize) -> E` where the argument is the environment index `0..num_envs`.
/// This lets you seed environments differently, give them different configs, etc.
pub struct BevyGymPlugin<E: Env> {
    /// Factory used to construct each environment instance.
    env_factory: Arc<dyn Fn(usize) -> E + Send + Sync>,

    /// Number of parallel environment instances to spawn.
    num_envs: usize,

    /// Fixed tick rate in Hz, or `None` for uncapped headless execution.
    tick_rate: Option<f64>,

    /// Whether ended episodes reset automatically.
    auto_reset: bool,

    /// Seed schedule for initial and automatic resets.
    reset_seed_schedule: ResetSeedSchedule,

    /// Retains the generic environment type.
    _phantom: PhantomData<E>,
}

impl<E: Env> std::fmt::Debug for BevyGymPlugin<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BevyGymPlugin")
            .field("num_envs", &self.num_envs)
            .field("tick_rate", &self.tick_rate)
            .field("auto_reset", &self.auto_reset)
            .field("reset_seed_schedule", &"<closure>")
            .finish_non_exhaustive()
    }
}

impl<E: Env + Send + Sync + 'static> BevyGymPlugin<E> {
    /// Create a new plugin with one environment instance.
    ///
    /// `factory` receives the environment index. Use [`Self::with_envs`] to
    /// spawn multiple independent instances.
    pub fn new(factory: impl Fn(usize) -> E + Send + Sync + 'static) -> Self {
        Self {
            env_factory: Arc::new(factory),
            num_envs: 1,
            tick_rate: Some(60.0),
            auto_reset: true,
            reset_seed_schedule: ResetSeedSchedule {
                seed_for: Arc::new(|env_id, episode| (episode == 0).then_some(env_id as u64)),
            },
            _phantom: PhantomData,
        }
    }

    /// Set the number of parallel environment instances.
    #[must_use]
    pub const fn with_envs(mut self, num_envs: usize) -> Self {
        self.num_envs = num_envs;
        self
    }

    /// Set the fixed tick rate in Hz.
    #[must_use]
    pub const fn with_tick_rate(mut self, hz: f64) -> Self {
        self.tick_rate = Some(hz);
        self
    }

    /// Run the fixed update schedule without a configured fixed tick rate.
    #[must_use]
    pub const fn uncapped(mut self) -> Self {
        self.tick_rate = None;
        self
    }

    /// Reset ended episodes automatically. This is the default.
    #[must_use]
    pub const fn autoreset(mut self) -> Self {
        self.auto_reset = true;
        self
    }

    /// Leave ended episodes idle until the user inserts `ResetRequested`.
    #[must_use]
    pub const fn without_autoreset(mut self) -> Self {
        self.auto_reset = false;
        self
    }

    /// Set the reset seed schedule for every environment and episode.
    ///
    /// `episode` is zero for the initial reset and increments after each
    /// completed episode. The schedule is independent of ECS execution order.
    #[must_use]
    pub fn with_reset_seeds(
        mut self,
        seed_for: impl Fn(usize, u64) -> Option<u64> + Send + Sync + 'static,
    ) -> Self {
        self.reset_seed_schedule = ResetSeedSchedule {
            seed_for: Arc::new(seed_for),
        };
        self
    }
}

impl<E: Env + Send + Sync + 'static> Plugin for BevyGymPlugin<E> {
    fn build(&self, app: &mut App) {
        if let Some(hz) = self.tick_rate {
            app.insert_resource(Time::<Fixed>::from_hz(hz));
        }

        app.insert_resource(GymConfig {
            num_envs: self.num_envs,
            tick_rate: self.tick_rate,
            auto_reset: self.auto_reset,
        });
        app.insert_resource(self.reset_seed_schedule.clone());

        app.add_message::<TransitionEvent<E>>();
        app.add_message::<EpisodeEndEvent>();
        app.add_message::<ActionRequest<E>>();
        app.add_message::<ActionResponse<E>>();

        app.configure_sets(
            FixedUpdate,
            (
                GymSet::Step,
                GymSet::AutoReset,
                GymSet::ManualReset,
                GymSet::RequestActions,
            )
                .chain(),
        );

        app.add_systems(FixedUpdate, step_system::<E>.in_set(GymSet::Step));
        if self.auto_reset {
            app.add_systems(
                FixedUpdate,
                auto_reset_system::<E>.in_set(GymSet::AutoReset),
            );
        }
        app.add_systems(
            FixedUpdate,
            manual_reset_system::<E>.in_set(GymSet::ManualReset),
        );

        let factory = Arc::clone(&self.env_factory);
        let reset_seed_schedule = self.reset_seed_schedule.clone();
        let num_envs = self.num_envs;
        app.add_systems(
            Startup,
            spawn_environments_seeded(move |i| factory(i), num_envs, reset_seed_schedule),
        );
    }
}

/// Startup system that spawns all environment entities and requests their first actions.
///
/// Called automatically by `BevyGymPlugin`. Can also be used standalone if you
/// need to spawn environments separately from the plugin setup.
pub fn spawn_environments<E: Env + Send + Sync + 'static>(
    factory: impl Fn(usize) -> E + Send + Sync + 'static,
    num_envs: usize,
) -> impl FnMut(Commands<'_, '_>, MessageWriter<'_, ActionRequest<E>>) {
    spawn_environments_seeded(
        factory,
        num_envs,
        ResetSeedSchedule {
            seed_for: Arc::new(|env_id, _episode| Some(env_id as u64)),
        },
    )
}

/// Spawn environments using the plugin's episode seed schedule.
fn spawn_environments_seeded<E: Env + Send + Sync + 'static>(
    factory: impl Fn(usize) -> E + Send + Sync + 'static,
    num_envs: usize,
    reset_seed_schedule: ResetSeedSchedule,
) -> impl FnMut(Commands<'_, '_>, MessageWriter<'_, ActionRequest<E>>) {
    move |mut commands: Commands<'_, '_>, mut action_writer: MessageWriter<'_, ActionRequest<E>>| {
        for i in 0..num_envs {
            let mut env = factory(i);
            let initial = env.reset(reset_seed_schedule.seed_for(i, 0));

            let entity = commands
                .spawn((
                    EnvId(i),
                    EnvComponent::new(env),
                    CurrentObservation::<E> {
                        observation: initial.observation.clone(),
                        info: initial.info.clone(),
                    },
                    QueuedAction::<E>::default(),
                    EnvStats::default(),
                ))
                .id();

            action_writer.write(ActionRequest {
                env_id: i,
                entity,
                observation: initial.observation,
                info: initial.info,
            });
        }
    }
}
