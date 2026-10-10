//! Presentation commands preserve frozen actions and the common physical timestep.

use crate::standing::world_session::Session;

/// Spectator commands never contain an actuator value.
#[derive(Clone, Copy)]
pub(crate) enum Command {
    /// Toggle continuous playback after all six models are usable.
    Play,
    /// Pause and infer one common frame.
    Step,
    /// Pause and reset physical state and ready-policy memories.
    Reset,
    /// Change the number of common frames per presentation tick.
    Speed,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A load failure retains its diagnostic and never acquires playback capability.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn failed_loading_has_no_playback_or_reset_fallback() {
        let mut player = Player::new(Err("invalid checkpoint".to_owned()));
        for ready in [false, true] {
            for command in [Command::Play, Command::Step, Command::Reset, Command::Speed] {
                player.apply(command, ready);
                player.advance(ready);
                assert!(!player.running());
                assert!(matches!(player.session(), Err("invalid checkpoint")));
            }
        }
    }
}

/// Mutually exclusive spectator playback modes.
#[derive(Default)]
enum Playback {
    /// Wait for a step or resume request.
    #[default]
    Paused,
    /// Infer new actions on fixed presentation ticks.
    Running,
}

/// Closed presentation rates; each inferred physical frame remains 20 ms.
#[derive(Clone, Copy, Default)]
enum Rate {
    /// One common frame per fixed tick.
    #[default]
    One,
    /// Four common frames per fixed tick.
    Four,
    /// Sixteen common frames per fixed tick.
    Sixteen,
}

/// A load failure cannot retain a session or a running playback mode.
enum State {
    /// Preserve the failed checkpoint load without any inference capability.
    Unavailable(
        /// Checkpoint-load diagnostic retained until the page reloads.
        String,
    ),
    /// Only a loaded session can retain spectator playback state.
    Loaded {
        /// Frozen policies and their last complete physical frame.
        session: Box<Session>,
        /// Spectator timing cannot choose policy actions.
        playback: Playback,
    },
}

/// A validated frozen session and independent spectator timing state.
#[derive(bevy::prelude::Resource)]
pub(crate) struct Player {
    /// Mutually exclusive loaded session or checkpoint failure.
    state: State,
    /// Number of complete policy frames consumed per fixed tick.
    rate: Rate,
}

impl Player {
    /// Retain the validated session or its visible load failure.
    pub(crate) fn new(session: Result<Session, String>) -> Self {
        Self {
            state: match session {
                Ok(session) => State::Loaded {
                    session: Box::new(session),
                    playback: Playback::default(),
                },
                Err(error) => State::Unavailable(error),
            },
            rate: Rate::default(),
        }
    }

    /// Borrow physical and policy results without exposing mutation.
    pub(crate) fn session(&self) -> Result<&Session, &str> {
        match &self.state {
            State::Loaded { session, .. } => Ok(session),
            State::Unavailable(error) => Err(error.as_str()),
        }
    }

    /// Read whether the spectator requested continuous playback.
    pub(crate) const fn running(&self) -> bool {
        matches!(
            self.state,
            State::Loaded {
                playback: Playback::Running,
                ..
            }
        )
    }

    /// Read the number of policy frames per fixed presentation tick.
    pub(crate) const fn rate(&self) -> usize {
        match self.rate {
            Rate::One => 1,
            Rate::Four => 4,
            Rate::Sixteen => 16,
        }
    }

    /// Apply a presentation command through the model-readiness boundary.
    pub(crate) fn apply(&mut self, command: Command, models_ready: bool) {
        match command {
            Command::Speed => {
                self.rate = match self.rate {
                    Rate::One => Rate::Four,
                    Rate::Four => Rate::Sixteen,
                    Rate::Sixteen => Rate::One,
                }
            }
            Command::Play => {
                // Loading, failed inference and completed clips cannot resume.
                if let State::Loaded { session, playback } = &mut self.state {
                    if models_ready && session.error().is_none() && !session.finished() {
                        *playback = match playback {
                            Playback::Paused => Playback::Running,
                            Playback::Running => Playback::Paused,
                        };
                    }
                }
            }
            Command::Step => {
                if let State::Loaded { session, playback } = &mut self.state {
                    *playback = Playback::Paused;
                    if models_ready {
                        session.step();
                    }
                }
            }
            Command::Reset => {
                if let State::Loaded { session, playback } = &mut self.state {
                    *playback = Playback::Paused;
                    session.reset();
                }
            }
        }
    }

    /// Consume complete frozen-policy frames only when every model is ready.
    pub(crate) fn advance(&mut self, models_ready: bool) {
        let count = self.rate();
        let State::Loaded { session, playback } = &mut self.state else {
            return;
        };
        if !models_ready || !matches!(playback, Playback::Running) {
            return;
        }
        for _ in 0..count {
            session.step();
        }
        // Derive completion from physics; no separate viewer clock is advanced.
        if session.finished() || session.error().is_some() {
            *playback = Playback::Paused;
        }
    }
}
