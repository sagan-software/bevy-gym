//! Default Gymnasium Pendulum with double-precision state and float32 actions.
//!
//! Farama Gymnasium, MIT license, revision `7a1191388aa4aa973d3a5e4b039899cd99cc991f`.
//! <https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/pendulum.py>
//! The numerical profile uses `NumPy` 2.4.4. Custom gravity and reset bounds are
//! unsupported. Reset uses the original uniform bounds with a `SplitMix64` stream.
use super::{PendulumAction, PendulumState};
use crate::training::split_mix64::SplitMix64;
use crate::{Env, EpisodeStatus, Reset, Step};

/// Default Pendulum-v1 dynamics; wrap in [`crate::TimeLimit`] for 200-step episodes.
#[derive(Debug, Clone, Copy)]
pub struct Pendulum {
    /// Unwrapped angle in radians and angular velocity in radians per second.
    state: [f64; 2],
    /// Clipped torque in newton metres; absent before the first transition.
    last_torque: Option<f32>,
    /// Independent deterministic reset stream.
    random: SplitMix64,
}

impl Default for Pendulum {
    fn default() -> Self {
        Self::from_state(
            PendulumState::try_from([std::f64::consts::PI, 0.0]).expect("finite state"),
        )
    }
}

impl Pendulum {
    /// Inject a validated diagnostic state without changing the default physics.
    #[must_use]
    pub const fn from_state(state: PendulumState) -> Self {
        Self {
            state: *state.values(),
            last_torque: None,
            random: SplitMix64::new(0),
        }
    }

    /// Return the unwrapped angle and angular velocity in source coordinate order.
    #[must_use]
    pub const fn state(&self) -> [f64; 2] {
        self.state
    }

    /// Return the clipped torque used by the source renderer, in newton metres.
    #[must_use]
    pub const fn last_torque(&self) -> Option<f32> {
        self.last_torque
    }

    /// Project double coordinates to the source's float32 observation boundary.
    fn observation(&self) -> [f32; 3] {
        let [angle, velocity] = self.state;
        [angle.cos() as f32, angle.sin() as f32, velocity as f32]
    }
}

impl Env for Pendulum {
    type Action = PendulumAction;
    type Observation = [f32; 3];
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        if let Some(seed) = seed {
            self.random = SplitMix64::new(seed);
        }
        self.state = [
            self.random
                .f64_between(-std::f64::consts::PI, std::f64::consts::PI),
            self.random.f64_between(-1.0, 1.0),
        ];
        self.last_torque = None;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "The pinned NumPy source rounds multiplication and addition separately"
    )]
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let torque = f32::from(action).clamp(-2.0, 2.0);
        let [angle, velocity] = self.state;
        // Compute cost before integration. NumPy squares float32 torque and applies
        // its 0.001 cost scale before promoting that term into the double state cost.
        let normalized =
            (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        let cost = normalized.powi(2) + 0.1 * velocity.powi(2) + f64::from(0.001 * torque.powi(2));
        // g = 10 m/s², l = 1 m, m = 1 kg: 3g/(2l) = 15 s⁻² and
        // 3/(ml²) = 3 (kg m²)⁻¹. Multiplying acceleration by dt = 0.05 s
        // yields rad/s. NumPy computes the torque term in float32 first.
        let acceleration = 15.0 * angle.sin() + f64::from(3.0 * torque);
        let next_velocity = (velocity + acceleration * 0.05).clamp(-8.0, 8.0);
        // Integrate using the clipped velocity, preserving the unwrapped angle.
        self.state = [angle + next_velocity * 0.05, next_velocity];
        self.last_torque = Some(torque);
        Step {
            observation: self.observation(),
            reward: -cost,
            status: EpisodeStatus::Continuing,
            info: (),
        }
    }
}
