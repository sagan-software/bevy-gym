//! Bevy ECS plugin for parallelised RL environment simulation.
//!
//! `bevy-gym` runs repo-owned [`Env`] implementations inside Bevy's ECS,
//! giving you:
//!
//! - **Free parallelism**: N environment instances as separate Bevy entities,
//!   stepped in parallel via `par_iter_mut()` each `FixedUpdate` tick.
//!
//! - **Decoupled tick rate**: RL simulation runs in `FixedUpdate` at a
//!   configurable Hz or uncapped for throughput-focused simulation.
//!
//! - **Typed policy integration**: [`ActionRequest`] carries the observation
//!   and info to a policy system, and [`ActionResponse`] carries the action
//!   back to the runner.
//!
//! - **Burn trainer boundary**: [`training`] exposes the first-class
//!   Burn-backed trainer layer for model, run, metrics, and checkpoint
//!   responsibilities without expanding [`Env`].
//!
//! # Quick start
//!
//! ```rust,no_run
//! use bevy::prelude::*;
//! use bevy_gym::{ActionRequest, ActionResponse, BevyGymPlugin, Env, EpisodeStatus, GymSet, Reset, Step};
//!
//! struct Counter(u32);
//!
//! impl Env for Counter {
//!     type Observation = u32;
//!     type Action = bool;
//!     type Info = ();
//!
//!     fn reset(&mut self, _seed: Option<u64>) -> Reset<u32> {
//!         self.0 = 0;
//!         Reset { observation: self.0, info: () }
//!     }
//!
//!     fn step(&mut self, action: bool) -> Step<u32> {
//!         if action {
//!             self.0 += 1;
//!         }
//!
//!         Step {
//!             observation: self.0,
//!             reward: f64::from(self.0),
//!             status: if self.0 >= 10 { EpisodeStatus::Terminated } else { EpisodeStatus::Continuing },
//!             info: (),
//!         }
//!     }
//! }
//!
//! fn policy(
//!     mut requests: MessageReader<ActionRequest<Counter>>,
//!     mut actions: MessageWriter<ActionResponse<Counter>>,
//! ) {
//!     for req in requests.read() {
//!         actions.write(ActionResponse { entity: req.entity, action: true });
//!     }
//! }
//!
//! App::new()
//!     .add_plugins(MinimalPlugins)
//!     .add_plugins(BevyGymPlugin::new(|_| Counter(0)).with_envs(16).uncapped())
//!     .add_systems(FixedUpdate, policy.in_set(GymSet::RequestActions))
//!     .run();
//! ```

#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
// Cargo exposes dev-dependencies to the library test target. Ecosystem example
// tests own the actual Avian use, while this keeps per-target dependency linting exact.
#[cfg(test)]
use avian2d as _;
// Ecosystem examples own these serialization dependencies. Keep library builds
// free of false unused-dependency diagnostics.
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
#[cfg(feature = "render")]
use serde as _;
use serde_json as _;
#[cfg(feature = "render")]
use tokio as _;

/// ECS components used to store environment state.
pub mod components;
/// Core reinforcement-learning environment vocabulary.
pub mod core;
/// Messages emitted by the gym systems.
pub mod events;
/// Bevy plugin and startup helpers.
pub mod plugin;
/// Fixed-update systems that step and reset environments.
pub mod systems;
/// Burn-backed trainer API boundary.
pub mod training;
/// Reusable environment wrappers.
pub mod wrappers;

pub use components::{CurrentObservation, EnvComponent, EnvId, EnvStats};
pub use core::{
    check_env_with_action, BoxSpace, DiscreteSpace, Env, EnvCheckError, EpisodeStatus, HasSpaces,
    Reset, Space, Step, Transition,
};
pub use events::{ActionRequest, ActionResponse, EpisodeEndEvent, TransitionEvent};
pub use plugin::{spawn_environments, BevyGymPlugin, GymConfig, GymSet};
pub use systems::reset::ResetRequested;
pub use wrappers::time_limit::{TimeLimit, TimeLimitError};
