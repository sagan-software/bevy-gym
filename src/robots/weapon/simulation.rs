//! Authoritative weapon phases and policy-request advancement.
use super::{
    delay::Delay, magazine::Magazine, RobotVolleyRounds, RobotWeaponOutcome, RobotWeaponPhase,
};
use crate::{robots::RobotTeam, training::CombatSignal};
/// Fixed-team mechanical weapon; sensors and target state are absent from its API.
/// Advancement takes O(1) time and auxiliary space.
///
/// ```compile_fail
/// use bevy_gym::robots::{RobotTeam, RobotWeapon};
/// let mut weapon = RobotWeapon::from(RobotTeam::Drones);
/// weapon.state = weapon.state;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RobotWeapon {
    /// One mutually exclusive team-specific mechanical phase.
    state: State,
}
/// Team is derived from the authoritative state, never stored independently.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Drone-specific charge, volley and recovery rules.
    Drone(#[doc = "Authoritative drone phase."] DroneState),
    /// Droid-specific magazine and pistol recovery rules.
    Droid(#[doc = "Authoritative droid phase."] DroidState),
}
/// Only charged requests can progress into a drone volley.
#[derive(Debug, Clone, Copy)]
enum DroneState {
    /// A request may start charging.
    Ready,
    /// Positive unfinished spin-up.
    Charging(#[doc = "Positive remaining charge."] Delay),
    /// Positive inter-round delay and one or two later rounds.
    Volley(
        #[doc = "Positive inter-round delay."] Delay,
        #[doc = "One or two remaining rounds."] RobotVolleyRounds,
    ),
    /// Positive recovery before another charge.
    Cooling(#[doc = "Positive remaining recovery."] Delay),
    /// Lifecycle shutdown; no request can reactivate this state.
    Disabled,
}
/// Empty magazines cannot remain ready; counts only descend from twelve.
#[derive(Debug, Clone, Copy)]
enum DroidState {
    /// One through twelve remaining rounds can discharge.
    Ready(#[doc = "Validated positive remaining magazine."] Magazine),
    /// Recovery remains mandatory even after the last round.
    Cooling {
        /// Positive time before ready or empty becomes observable.
        delay: Delay,
        /// Absence means the last round was discharged.
        rounds: Option<Magazine>,
    },
    /// No round remains and there is no automatic reload.
    Empty,
    /// Lifecycle shutdown; no request can reactivate this state.
    Disabled,
}
impl From<RobotTeam> for RobotWeapon {
    fn from(team: RobotTeam) -> Self {
        let state = match team {
            RobotTeam::Drones => State::Drone(DroneState::Ready),
            RobotTeam::Droids => State::Droid(DroidState::Ready(Magazine::default())),
        };
        Self { state }
    }
}
impl RobotWeapon {
    /// Read timing without supplying or changing a trigger request.
    #[must_use]
    pub const fn phase(&self) -> RobotWeaponPhase {
        match self.state {
            State::Drone(DroneState::Ready) | State::Droid(DroidState::Ready(_)) => {
                RobotWeaponPhase::Ready
            }
            State::Drone(DroneState::Charging(delay)) => RobotWeaponPhase::Charging {
                remaining: delay.remaining(),
            },
            State::Drone(DroneState::Volley(delay, rounds)) => RobotWeaponPhase::Volley {
                remaining: delay.remaining(),
                rounds,
            },
            State::Drone(DroneState::Cooling(delay))
            | State::Droid(DroidState::Cooling { delay, .. }) => RobotWeaponPhase::Cooling {
                remaining: delay.remaining(),
            },
            State::Droid(DroidState::Empty) => RobotWeaponPhase::Empty,
            State::Drone(DroneState::Disabled) | State::Droid(DroidState::Disabled) => {
                RobotWeaponPhase::Disabled
            }
        }
    }
    /// Advance exactly twenty milliseconds using only the current policy signal.
    /// Alert categories never authorize a released trigger. This emits permission for
    /// one round; a separate physical environment must derive its muzzle and collision.
    pub fn advance(&mut self, signal: CombatSignal) -> RobotWeaponOutcome {
        let requested = requested(signal);
        match self.state {
            State::Drone(state) => {
                let (state, outcome) = advance_drone(state, requested);
                self.state = State::Drone(state);
                outcome
            }
            State::Droid(state) => {
                let (state, outcome) = advance_droid(state, requested);
                self.state = State::Droid(state);
                outcome
            }
        }
    }
    /// Disable all future discharges until a new weapon is constructed at reset.
    pub const fn disable(&mut self) {
        self.state = match self.state {
            State::Drone(_) => State::Drone(DroneState::Disabled),
            State::Droid(_) => State::Droid(DroidState::Disabled),
        };
    }
}
/// Map the closed actor vocabulary directly; no sight or inferred alert transition enters firing.
const fn requested(signal: CombatSignal) -> bool {
    match signal {
        CombatSignal::IdleRequested
        | CombatSignal::SuspiciousRequested
        | CombatSignal::PursuitRequested
        | CombatSignal::SearchRequested => true,
        CombatSignal::IdleReleased
        | CombatSignal::SuspiciousReleased
        | CombatSignal::PursuitReleased
        | CombatSignal::SearchReleased => false,
    }
}
/// Apply drone timing without choosing aim, target or trigger.
fn advance_drone(state: DroneState, requested: bool) -> (DroneState, RobotWeaponOutcome) {
    use RobotWeaponOutcome as Outcome;
    match state {
        DroneState::Disabled => (state, Outcome::Disabled),
        DroneState::Ready => {
            if requested {
                (DroneState::Charging(Delay::charge()), Outcome::Charging)
            } else {
                (state, Outcome::NoRequest)
            }
        }
        DroneState::Charging(delay) => {
            // Releasing before discharge discards all charge progress.
            if !requested {
                return (DroneState::Ready, Outcome::NoRequest);
            }
            delay.elapse().map_or_else(
                || {
                    (
                        DroneState::Volley(Delay::volley(), RobotVolleyRounds::Two),
                        Outcome::Fired,
                    )
                },
                |delay| (DroneState::Charging(delay), Outcome::Charging),
            )
        }
        DroneState::Volley(delay, rounds) => {
            // Cancelling either later round pays the full recovery.
            if !requested {
                return (
                    DroneState::Cooling(Delay::drone_cooling()),
                    Outcome::Cooling,
                );
            }
            if let Some(delay) = delay.elapse() {
                return (DroneState::Volley(delay, rounds), Outcome::VolleyDelay);
            }
            let next = match rounds {
                RobotVolleyRounds::Two => {
                    DroneState::Volley(Delay::volley(), RobotVolleyRounds::One)
                }
                RobotVolleyRounds::One => DroneState::Cooling(Delay::drone_cooling()),
            };
            (next, Outcome::Fired)
        }
        DroneState::Cooling(delay) => {
            // Finishing recovery cannot discharge until the next decision boundary.
            let next = delay
                .elapse()
                .map_or(DroneState::Ready, DroneState::Cooling);
            (next, Outcome::Cooling)
        }
    }
}
/// Consume only a currently requested ready pistol round; elapsed recovery never supplies a shot.
fn advance_droid(state: DroidState, requested: bool) -> (DroidState, RobotWeaponOutcome) {
    use RobotWeaponOutcome as Outcome;
    match state {
        DroidState::Disabled => (state, Outcome::Disabled),
        DroidState::Empty => (state, Outcome::Empty),
        DroidState::Ready(rounds) => {
            if !requested {
                return (state, Outcome::NoRequest);
            }
            (
                DroidState::Cooling {
                    delay: Delay::droid_cooling(),
                    rounds: rounds.discharge(),
                },
                Outcome::Fired,
            )
        }
        DroidState::Cooling { delay, rounds } => {
            let next = delay.elapse().map_or_else(
                || rounds.map_or(DroidState::Empty, DroidState::Ready),
                |delay| DroidState::Cooling { delay, rounds },
            );
            (next, Outcome::Cooling)
        }
    }
}
