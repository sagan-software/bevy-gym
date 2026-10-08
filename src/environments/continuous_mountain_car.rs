//! Gymnasium continuous `MountainCar` with source precision and raw-action rewards.
//!
//! Farama Gymnasium, MIT license, revision `7a1191388aa4aa973d3a5e4b039899cd99cc991f`.
//! <https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/continuous_mountain_car.py>
//! The numerical profile uses `NumPy` 2.4.4 with float32 actions. Reset uses
//! `SplitMix64` with the original uniform bounds, not `NumPy`'s seed stream.
use super::{ContinuousMountainCarAction, MountainCarState};
use crate::training::split_mix64::SplitMix64;
use crate::{Env, EpisodeStatus, Reset, Step};

/// Default Gymnasium continuous `MountainCar`, without an episode time limit.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousMountainCar {
    /// Source precision changes from reset doubles to transition floats.
    state: State,
    /// Independent reset stream.
    random: SplitMix64,
}

/// `NumPy` stores a double reset state and float32 state after each transition.
#[derive(Debug, Clone, Copy)]
enum State {
    /// Reset or diagnostic state before the first step.
    Initial([f64; 2]),
    /// State rounded by the source's post-step float32 array construction.
    Stepped([f32; 2]),
}
impl Default for ContinuousMountainCar {
    fn default() -> Self {
        Self {
            state: State::Initial([-0.5, 0.0]),
            random: SplitMix64::new(0),
        }
    }
}
impl ContinuousMountainCar {
    /// Inject the source's initial double state after finite-angle validation.
    #[must_use]
    pub const fn from_state(state: MountainCarState) -> Self {
        Self {
            state: State::Initial(*state.values()),
            random: SplitMix64::new(0),
        }
    }
    /// Return position and displacement per transition in source coordinate units.
    #[must_use]
    pub fn state(&self) -> [f64; 2] {
        match self.state {
            State::Initial(values) => values,
            State::Stepped(values) => values.map(f64::from),
        }
    }
}
impl Env for ContinuousMountainCar {
    type Action = ContinuousMountainCarAction;
    type Observation = [f32; 2];
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        if let Some(seed) = seed {
            self.random = SplitMix64::new(seed);
        }
        let state = [self.random.f64_between(-0.6, -0.4), 0.0];
        self.state = State::Initial(state);
        Reset {
            observation: state.map(|value| value as f32),
            info: (),
        }
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let raw = f32::from(action);
        // NumPy 2.4.4 keeps reset coordinates in float64. Later steps use float32
        // additions and comparisons before storing another float32 state.
        let (observation, terminated) = match self.state {
            State::Initial([position, velocity]) => {
                let delta = increment(raw, 3.0 * position);
                let velocity = (velocity + delta).clamp(-0.07, 0.07);
                let position = (position + velocity).clamp(-1.2, 0.6);
                let velocity = if position <= -1.2 && velocity < 0.0 {
                    0.0
                } else {
                    velocity
                };
                (
                    [position as f32, velocity as f32],
                    position >= 0.45 && velocity >= 0.0,
                )
            }
            State::Stepped([position, velocity]) => {
                let delta = increment(raw, f64::from(3.0 * position)) as f32;
                let velocity = (velocity + delta).clamp(-0.07, 0.07);
                let position = (position + velocity).clamp(-1.2, 0.6);
                let velocity = if position <= -1.2 && velocity < 0.0 {
                    0.0
                } else {
                    velocity
                };
                ([position, velocity], position >= 0.45 && velocity >= 0.0)
            }
        };
        self.state = State::Stepped(observation);
        // The reward penalizes the raw force, including values outside [-1, 1].
        // The source computes this square and its 0.1 reward scale in float64.
        let raw = f64::from(raw);
        let penalty = raw * raw * 0.1;
        let bonus = if terminated { 100.0 } else { 0.0 };
        Step {
            observation,
            reward: bonus - penalty,
            status: if terminated {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

/// Reproduce `NumPy`'s force promotion before adding the velocity increment.
fn increment(raw: f32, angle: f64) -> f64 {
    // 0.0015 is the velocity increment per force unit and transition; 0.0025
    // scales the gravity velocity increment. The terrain angle is in radians.
    let gravity = 0.0025 * angle.cos();
    if (-1.0..=1.0).contains(&raw) {
        // Unclipped force remains a NumPy float32 scalar; Python gravity is
        // converted to float32 before subtraction under weak scalar promotion.
        let engine = raw * 0.0015;
        f64::from(engine - gravity as f32)
    } else {
        // Python min/max replace out-of-range force with a float64 bound.
        let engine = f64::from(raw.clamp(-1.0, 1.0)) * 0.0015;
        engine - gravity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_restores_double_storage_after_float_transitions() {
        let mut env = ContinuousMountainCar::default();
        env.step(ContinuousMountainCarAction::try_from(0.0).expect("finite"));
        assert!(matches!(env.state, State::Stepped(_)));
        env.reset(Some(42));
        assert!(matches!(env.state, State::Initial(_)));
    }
}
