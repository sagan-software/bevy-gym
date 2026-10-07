//! `CartPole` dynamics shared by native and browser training.
//!
//! Equations and default rewards follow Farama Gymnasium, MIT license, revision
//! `7a1191388aa4aa973d3a5e4b039899cd99cc991f`, `classic_control/cartpole.py`.
//! <https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/cartpole.py>
//! Resets use `SplitMix64` rather than `NumPy`'s `PCG64`, with the same uniform bounds.

use super::{CartPoleAction, CartPoleState};
use crate::training::split_mix64::SplitMix64;
use crate::{Env, EpisodeStatus, Reset, Step};
use std::time::Duration;

/// Gymnasium `CartPole`-v1 default dynamics, without the external time limit.
///
/// Call `reset` after termination. Diagnostic state injection accepts finite
/// coordinates; extreme injected velocities can overflow as in the Python model.
#[derive(Debug, Clone)]
pub struct CartPole {
    /// Position in metres, velocity in m/s, angle in radians, angular speed in rad/s.
    state: [f64; 4],
    /// Independent deterministic reset stream.
    random: SplitMix64,
    /// Whether a previous step terminated since the last reset.
    lifecycle: Lifecycle,
}

/// Reward treatment before and after the first terminal transition.
#[derive(Debug, Clone, Copy)]
enum Lifecycle {
    /// The next terminal transition earns the final unit reward.
    Active,
    /// Further terminal transitions earn zero reward, matching Gymnasium.
    Terminated,
}

impl Default for CartPole {
    fn default() -> Self {
        Self {
            state: [0.0; 4],
            random: SplitMix64::new(0),
            lifecycle: Lifecycle::Active,
        }
    }
}

impl CartPole {
    /// Construct an environment from a validated initial physical state.
    #[must_use]
    pub const fn from_state(state: CartPoleState) -> Self {
        Self {
            state: *state.values(),
            random: SplitMix64::new(0),
            lifecycle: Lifecycle::Active,
        }
    }

    /// Borrow the double-precision physical coordinates for diagnostics or rendering.
    #[must_use]
    pub const fn state(&self) -> &[f64; 4] {
        &self.state
    }
}

impl Env for CartPole {
    type Observation = [f32; 4];
    type Action = CartPoleAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        if let Some(seed) = seed {
            self.random = SplitMix64::new(seed);
        }
        self.state = std::array::from_fn(|_| self.random.f64_between(-0.05, 0.05));
        self.lifecycle = Lifecycle::Active;
        Reset {
            observation: self.state.map(|value| value as f32),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let [position, velocity, angle, angular_velocity] = self.state;
        // Masses are 1 kg and 0.1 kg; pole half-length is 0.5 m, gravity 9.8 m/s².
        // Keep the original operation order and explicit Euler integrator.
        let force = match action {
            CartPoleAction::Left => -10.0,
            CartPoleAction::Right => 10.0,
        };
        let cosine = angle.cos();
        let sine = angle.sin();
        let centrifugal_force = 0.05 * angular_velocity.powi(2) * sine;
        let temp = (force + centrifugal_force) / 1.1;
        let gravity_component = 9.8 * sine;
        let inertia_component = cosine * temp;
        let angular_acceleration = (gravity_component - inertia_component)
            / (0.5 * (4.0 / 3.0 - 0.1 * cosine.powi(2) / 1.1));
        let acceleration = temp - 0.05 * angular_acceleration * cosine / 1.1;
        let seconds = Duration::from_millis(20).as_secs_f64();
        self.state = [
            position + seconds * velocity,
            velocity + seconds * acceleration,
            angle + seconds * angular_velocity,
            angular_velocity + seconds * angular_acceleration,
        ];
        // Inclusive boundary coordinates remain active. Convert 12 degrees to radians.
        let angle_limit = 12.0 * 2.0 * std::f64::consts::PI / 360.0;
        let terminated = self.state[0] < -2.4
            || self.state[0] > 2.4
            || self.state[2] < -angle_limit
            || self.state[2] > angle_limit;
        let (status, reward) = if terminated {
            let reward = match self.lifecycle {
                Lifecycle::Active => 1.0,
                Lifecycle::Terminated => 0.0,
            };
            self.lifecycle = Lifecycle::Terminated;
            (EpisodeStatus::Terminated, reward)
        } else {
            (EpisodeStatus::Continuing, 1.0)
        };
        Step {
            observation: self.state.map(|value| value as f32),
            reward,
            status,
            info: (),
        }
    }
}
