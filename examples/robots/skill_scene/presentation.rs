//! Read-only projection of the two typed playback session shapes.

use bevy::math::Vec3;
use bevy_gym::robots::{DroneAction, DroneDestination, DroneObservation};
use bevy_gym::EpisodeStatus;

use super::{session::Session, travel};

/// Keep physical task observations typed until the renderer reads their common body state.
pub(super) enum SceneSession {
    /// Existing hover or recovery episode.
    Control(
        /// Frozen control policy and its physical episode.
        Session,
    ),
    /// Goal-conditioned travel episode using the shared PPO task.
    Travel(
        /// Frozen travel policy and its independently sampled task.
        Session<travel::environment::TravelTask, 13>,
    ),
}

impl SceneSession {
    /// Request one frozen policy action through the existing guarded session boundary.
    pub(super) fn step(&mut self) {
        match self {
            Self::Control(session) => session.step(),
            Self::Travel(session) => session.step(),
        }
    }

    /// Restart an episode and its private recurrent memory.
    pub(super) fn reset(&mut self, seed: u64) {
        match self {
            Self::Control(session) => session.reset(seed),
            Self::Travel(session) => session.reset(seed),
        }
    }

    /// Read physical pose without exposing a writable environment.
    pub(super) const fn observation(&self) -> DroneObservation {
        match self {
            Self::Control(session) => session.observation(),
            Self::Travel(session) => session.observation().body(),
        }
    }

    /// Expose a goal only for the goal-conditioned task.
    pub(super) const fn destination(&self) -> Option<DroneDestination> {
        match self {
            Self::Control(_) => None,
            Self::Travel(session) => Some(session.observation().destination()),
        }
    }

    /// Derive the task's target from its authoritative observation or fixed hover contract.
    pub(super) fn target(&self) -> Vec3 {
        self.destination()
            .map_or(Vec3::new(0.0, 2.0, 0.0), DroneDestination::position)
    }

    /// Read the absolute travel heading error in radians.
    pub(super) fn heading_error(&self) -> Option<f32> {
        self.destination()
            .map(|goal| travel::encoding::heading_error(self.observation(), goal).abs())
    }

    /// Read environment completion independently of qualification status.
    pub(super) const fn status(&self) -> EpisodeStatus {
        match self {
            Self::Control(session) => session.status(),
            Self::Travel(session) => session.status(),
        }
    }

    /// Read the number of actions that actually reached physics.
    pub(super) const fn steps(&self) -> usize {
        match self {
            Self::Control(session) => session.steps(),
            Self::Travel(session) => session.steps(),
        }
    }

    /// Read the environment's configured horizon instead of duplicating its limit.
    pub(super) const fn horizon(&self) -> usize {
        match self {
            Self::Control(session) => session.horizon(),
            Self::Travel(session) => session.horizon(),
        }
    }

    /// Read motor fractions for rotor illustration only.
    pub(super) const fn last_action(&self) -> Option<DroneAction> {
        match self {
            Self::Control(session) => session.last_action(),
            Self::Travel(session) => session.last_action(),
        }
    }

    /// Preserve sticky inference errors for the user-facing readout.
    pub(super) fn error(&self) -> Option<&str> {
        match self {
            Self::Control(session) => session.error(),
            Self::Travel(session) => session.error(),
        }
    }
}
