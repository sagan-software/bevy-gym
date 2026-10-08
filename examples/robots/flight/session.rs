//! Deterministic playback and controller selection for the rendered hover lesson.

use bevy::prelude::Resource;
use bevy_gym::robots::{DroneAction, DroneHover, DroneObservation};
use bevy_gym::{Env, EpisodeStatus, Step, TimeLimit};

use super::pilot::RecoveryPilot;

/// Whether fixed updates may request another action; completion comes from `Step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Playback {
    /// Advance only when the user requests one step.
    Paused,
    /// Apply one action per fixed update until the environment ends.
    Running,
}

/// Initial conditions selected by the viewer, independent of motor commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StartProfile {
    /// Upright and stationary near the target.
    Calm,
    /// Random tilt, heading, and velocity near the target.
    Disturbed,
}

/// Four diagnostic commands; none is a learned controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MotorPreset {
    /// Turn all four motors off.
    PowerOff,
    /// Balance an upright body's weight with equal forces.
    Hover,
    /// Apply equal forces above the hover level.
    Climb,
    /// Increase left thrust and decrease right thrust to tilt the body.
    Tilt,
}

impl MotorPreset {
    /// Construct the validated command associated with this closed preset.
    pub(super) fn action(self) -> DroneAction {
        let fractions = match self {
            Self::PowerOff => [0.0; 4],
            Self::Hover => [0.5; 4],
            Self::Climb => [0.6; 4],
            Self::Tilt => [0.55, 0.45, 0.45, 0.55],
        };
        DroneAction::try_from(fractions).expect("presets contain finite motor fractions")
    }
}

/// Exactly one source of motor commands, or the failure that stopped inference.
enum Controller {
    /// A user-selected command with no policy memory.
    Manual(MotorPreset),
    /// A learned policy with its own episode memory.
    Learned(Box<RecoveryPilot>),
    /// Loading or inference failed; no further commands may reach physics.
    Failed(String),
}

/// Physics owns the pose; the viewer retains its last result and playback controls.
#[derive(Resource)]
pub(super) struct Session {
    /// Ten-second task using the same environment as the headless guide.
    environment: TimeLimit<DroneHover>,
    /// Most recent atomic observation, reward, and completion status.
    last: Step<DroneObservation>,
    /// User's playback choice, independent of the environment's completion status.
    playback: Playback,
    /// Source of the next validated command, or the diagnostic stopping playback.
    controller: Controller,
    /// Last command applied to physics, used only to illustrate rotor motion.
    last_action: DroneAction,
    /// Policy actions applied since reset, bounded by the 500-step time limit.
    steps: usize,
    /// Initial conditions retained when the user resets the episode.
    start_profile: StartProfile,
}

impl Default for Session {
    fn default() -> Self {
        Self::starting(StartProfile::Calm)
    }
}

impl Session {
    /// Build a paused, seeded episode for either documented start distribution.
    fn starting(start_profile: StartProfile) -> Self {
        let drone = match start_profile {
            StartProfile::Calm => DroneHover::default(),
            StartProfile::Disturbed => DroneHover::disturbed(),
        };
        let mut environment = TimeLimit::new(drone, 500).expect("time limit is positive");
        let initial = environment.reset(Some(42));
        Self {
            environment,
            last: Step {
                observation: initial.observation,
                reward: 0.0,
                status: EpisodeStatus::Continuing,
                info: (),
            },
            playback: Playback::Paused,
            controller: Controller::Manual(MotorPreset::Hover),
            last_action: MotorPreset::PowerOff.action(),
            steps: 0,
            start_profile,
        }
    }

    /// Read the physics pose without allowing a renderer to alter it.
    pub(super) const fn observation(&self) -> DroneObservation {
        self.last.observation
    }

    /// Read completion from the authoritative step result.
    pub(super) const fn status(&self) -> EpisodeStatus {
        self.last.status
    }

    /// Read the user's playback choice.
    pub(super) const fn playback(&self) -> Playback {
        self.playback
    }

    /// Read the current motor command preset.
    pub(super) const fn preset(&self) -> Option<MotorPreset> {
        match self.controller {
            Controller::Manual(preset) => Some(preset),
            Controller::Learned(_) | Controller::Failed(_) => None,
        }
    }

    /// Identify learned inference without exposing policy weights or memory.
    pub(super) const fn is_learned(&self) -> bool {
        matches!(self.controller, Controller::Learned(_))
    }

    /// Read a controller failure for the visible recovery message.
    pub(super) fn error(&self) -> Option<&str> {
        match &self.controller {
            Controller::Failed(error) => Some(error),
            Controller::Manual(_) | Controller::Learned(_) => None,
        }
    }

    /// Read the command that produced the displayed pose.
    pub(super) const fn last_action(&self) -> DroneAction {
        self.last_action
    }

    /// Completion and controller failure independently prevent another action.
    pub(super) const fn can_step(&self) -> bool {
        !self.status().is_done() && !matches!(self.controller, Controller::Failed(_))
    }

    /// Read the number of applied policy actions.
    pub(super) const fn steps(&self) -> usize {
        self.steps
    }

    /// Read the initial-condition choice without exposing mutable physics state.
    pub(super) const fn start_profile(&self) -> StartProfile {
        self.start_profile
    }

    /// Select initial conditions and replace the episode with its paused seed-42 start.
    pub(super) fn select_start(&mut self, profile: StartProfile) {
        self.start_profile = profile;
        self.reset();
    }

    /// Select an action without advancing or resetting the environment.
    pub(super) fn select(&mut self, preset: MotorPreset) {
        self.controller = Controller::Manual(preset);
    }

    /// Start a paused episode using the checkpoint qualified by the training lesson.
    pub(super) fn select_learned(&mut self) {
        self.select_policy(include_bytes!("../../../assets/robots/recovery.mpk").to_vec());
    }

    /// Reject a malformed checkpoint before it can supply any motor command.
    pub(super) fn select_policy(&mut self, bytes: Vec<u8>) {
        match RecoveryPilot::load(bytes) {
            Ok(pilot) => {
                self.controller = Controller::Learned(Box::new(pilot));
                self.reset();
            }
            Err(error) => self.fail(error.to_string()),
        }
    }

    /// Toggle playback only while another action is allowed.
    pub(super) const fn toggle_playback(&mut self) {
        if self.can_step() {
            self.playback = match self.playback {
                Playback::Paused => Playback::Running,
                Playback::Running => Playback::Paused,
            };
        }
    }

    /// Advance once during a running fixed update; ended episodes stay unchanged.
    pub(super) fn advance(&mut self) {
        if self.playback == Playback::Running && self.can_step() {
            self.take_step();
        }
    }

    /// Advance one paused frame; ignore requests during playback or after completion.
    pub(super) fn single_step(&mut self) {
        if self.playback == Playback::Paused && self.can_step() {
            self.take_step();
        }
    }

    /// Restore the seed and pause; retain learned weights but clear episode memory.
    pub(super) fn reset(&mut self) {
        let controller =
            std::mem::replace(&mut self.controller, Controller::Manual(MotorPreset::Hover));
        *self = Self::starting(self.start_profile);
        if let Controller::Learned(mut pilot) = controller {
            pilot.reset();
            self.controller = Controller::Learned(pilot);
        }
    }

    /// Apply one validated action and retain its atomic result for rendering.
    fn take_step(&mut self) {
        let command = match &mut self.controller {
            Controller::Manual(preset) => Ok(preset.action()),
            Controller::Learned(pilot) => pilot.command(self.last.observation),
            Controller::Failed(_) => return,
        };
        // Inference and validation complete before either physics or its counter changes.
        match command {
            Ok(action) => {
                self.last = self.environment.step(action);
                self.last_action = action;
                self.steps += 1;
            }
            Err(error) => self.fail(error.to_string()),
        }
    }

    /// Stop playback and retain a diagnostic until reset or controller selection.
    fn fail(&mut self, error: String) {
        self.controller = Controller::Failed(error);
        self.playback = Playback::Paused;
    }
}

#[cfg(test)]
mod tests {
    use bevy_gym::EpisodeStatus;

    use super::{MotorPreset, Playback, Session, StartProfile};

    #[test]
    fn checkpoint_failures_stop_playback_and_allow_explicit_recovery() {
        let mut session = Session::default();
        session.toggle_playback();
        session.advance();
        let pose = session.observation();
        let action = session.last_action();
        for bytes in [Vec::new(), vec![0xc1]] {
            session.select_policy(bytes);
            assert_eq!(session.playback(), Playback::Paused);
            assert!(session.error().is_some());
            assert!(!session.can_step());
            assert!(!session.is_learned());
            assert_eq!(session.preset(), None);
            session.toggle_playback();
            session.single_step();
            session.advance();
            session.take_step();
            assert_eq!(session.steps(), 1);
            assert_eq!(session.observation(), pose);
            assert_eq!(session.last_action(), action);
        }
        session.select(MotorPreset::Hover);
        assert!(session.can_step());
        assert!(session.error().is_none());
        session.select_policy(Vec::new());
        session.reset();
        assert_eq!(session.preset(), Some(MotorPreset::Hover));
        assert!(session.error().is_none());
        assert_eq!(session.steps(), 0);
        session.select_policy(Vec::new());
        session.select_learned();
        assert!(session.is_learned());
        assert!(session.can_step());
        assert!(session.error().is_none());
    }

    #[test]
    fn inference_failure_cannot_advance_physics_or_leave_playback_running() {
        use super::{Controller, RecoveryPilot};
        use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};

        let agent = RecurrentPpoAgent::new(
            12,
            12,
            1,
            &[0.0; 3],
            &[1.0; 3],
            crate::model::learning_config(),
            SeedConfig::from_root(7),
        )
        .unwrap();
        let mut session = Session::default();
        session.controller = Controller::Learned(Box::new(RecoveryPilot::from(agent.policy())));
        let pose = session.observation();
        let action = session.last_action();
        session.toggle_playback();
        session.advance();
        assert_eq!(
            session.error(),
            Some("expected four motor values, received 3")
        );
        assert_eq!(session.playback(), Playback::Paused);
        assert_eq!(session.steps(), 0);
        assert_eq!(session.observation(), pose);
        assert_eq!(session.last_action(), action);
        assert!(!session.can_step());
    }

    #[test]
    fn learned_recovery_repeats_after_reset_and_preserves_the_start_choice() {
        let mut session = Session::default();
        session.select_start(StartProfile::Disturbed);
        let initial = session.observation();
        session.select_learned();
        assert!(session.is_learned());
        assert_eq!(session.playback(), Playback::Paused);
        session.single_step();
        let first = session.observation();
        let action = session.last_action();
        for _ in 1..500 {
            session.single_step();
        }
        assert_eq!(session.status(), EpisodeStatus::Truncated);
        session.reset();
        assert!(session.is_learned());
        assert_eq!(session.start_profile(), StartProfile::Disturbed);
        assert_eq!(session.observation(), initial);
        assert_eq!(session.steps(), 0);
        session.single_step();
        assert_eq!(session.observation(), first);
        assert_eq!(session.last_action(), action);
        session.select_start(StartProfile::Calm);
        assert!(session.is_learned());
        assert_eq!(session.steps(), 0);
        session.select(MotorPreset::Climb);
        assert!(!session.is_learned());
        session.single_step();
        assert_eq!(session.last_action(), MotorPreset::Climb.action());
    }

    #[test]
    fn selecting_a_start_replaces_the_episode_and_reset_keeps_the_choice() {
        let mut session = Session::default();
        let calm = session.observation();
        session.select(MotorPreset::Climb);
        session.toggle_playback();
        session.advance();
        session.select_start(StartProfile::Disturbed);
        let disturbed = session.observation();
        assert_ne!(calm, disturbed);
        assert_eq!(session.start_profile(), StartProfile::Disturbed);
        assert_eq!(session.steps(), 0);
        assert_eq!(session.playback(), Playback::Paused);
        assert_eq!(session.preset(), Some(MotorPreset::Hover));
        session.single_step();
        session.reset();
        assert_eq!(session.observation(), disturbed);
        assert_eq!(session.start_profile(), StartProfile::Disturbed);
        session.select_start(StartProfile::Calm);
        assert_eq!(session.observation(), calm);
        assert_eq!(session.start_profile(), StartProfile::Calm);
    }

    #[test]
    fn paused_frames_do_not_advance_and_single_step_advances_once() {
        let mut session = Session::default();
        assert_eq!(session.playback(), Playback::Paused);
        let initial = session.observation();
        session.select(MotorPreset::PowerOff);
        session.advance();
        assert_eq!(session.observation(), initial);
        assert_eq!(session.steps(), 0);
        session.single_step();
        assert_eq!(session.steps(), 1);
        assert!(session.observation().position().y < initial.position().y);
    }

    #[test]
    fn running_ignores_single_step_and_pause_stops_progress() {
        let mut session = Session::default();
        session.toggle_playback();
        assert_eq!(session.playback(), Playback::Running);
        session.single_step();
        assert_eq!(session.steps(), 0);
        session.advance();
        assert_eq!(session.steps(), 1);
        session.toggle_playback();
        session.advance();
        assert_eq!(session.steps(), 1);
    }

    #[test]
    fn terminal_controls_are_absorbing_and_reset_restarts_the_seed() {
        let mut session = Session::default();
        let initial = session.observation();
        session.select(MotorPreset::PowerOff);
        session.toggle_playback();
        for _ in 0..100 {
            session.advance();
        }
        assert_eq!(session.status(), EpisodeStatus::Terminated);
        let steps = session.steps();
        let terminal = session.observation();
        session.toggle_playback();
        session.single_step();
        session.advance();
        assert_eq!(session.steps(), steps);
        assert_eq!(session.observation(), terminal);
        session.reset();
        assert_eq!(session.steps(), 0);
        assert_eq!(session.status(), EpisodeStatus::Continuing);
        assert_eq!(session.observation(), initial);
        assert_eq!(session.playback(), Playback::Paused);
        assert_eq!(session.preset(), Some(MotorPreset::Hover));
    }

    #[test]
    fn hover_stops_at_the_time_limit_and_paused_terminal_steps_do_nothing() {
        let mut session = Session::default();
        for _ in 0..500 {
            session.single_step();
        }
        assert_eq!(session.status(), EpisodeStatus::Truncated);
        assert_eq!(session.steps(), 500);
        let final_state = session.observation();
        session.single_step();
        session.toggle_playback();
        session.advance();
        assert_eq!(session.playback(), Playback::Paused);
        assert_eq!(session.steps(), 500);
        assert_eq!(session.observation(), final_state);
    }

    #[test]
    fn every_preset_has_distinct_bounded_commands() {
        assert_eq!(MotorPreset::PowerOff.action().fractions(), [0.0; 4]);
        assert_eq!(MotorPreset::Hover.action().fractions(), [0.5; 4]);
        assert_eq!(MotorPreset::Climb.action().fractions(), [0.6; 4]);
        assert_eq!(
            MotorPreset::Tilt.action().fractions(),
            [0.55, 0.45, 0.45, 0.55]
        );
    }
}
