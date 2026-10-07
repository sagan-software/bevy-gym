//! Default `MountainCar` dynamics shared by native and browser consumers.
//!
//! Farama Gymnasium, MIT license, revision `7a1191388aa4aa973d3a5e4b039899cd99cc991f`.
//! <https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/mountain_car.py>
//! Reset uses `SplitMix64` with the original uniform bounds, not `NumPy` `PCG64` seeds.
use super::{MountainCarAction, MountainCarState};
use crate::training::split_mix64::SplitMix64;
use crate::{Env, EpisodeStatus, Reset, Step};

/// Gymnasium `MountainCar`-v0 with default reset bounds and goal velocity zero.
#[derive(Debug, Clone, Copy)]
pub struct MountainCar {
    /// Position and displacement per transition in source coordinate units.
    state: [f64; 2],
    /// Independent reset stream.
    random: SplitMix64,
}
impl Default for MountainCar {
    fn default() -> Self {
        Self {
            state: [-0.5, 0.0],
            random: SplitMix64::new(0),
        }
    }
}
impl MountainCar {
    /// Construct diagnostic state after finite-coordinate validation.
    #[must_use]
    pub const fn from_state(state: MountainCarState) -> Self {
        Self {
            state: *state.values(),
            random: SplitMix64::new(0),
        }
    }
    /// Borrow the original double-precision coordinates.
    #[must_use]
    pub const fn state(&self) -> &[f64; 2] {
        &self.state
    }
}
impl Env for MountainCar {
    type Action = MountainCarAction;
    type Observation = [f32; 2];
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        if let Some(seed) = seed {
            self.random = SplitMix64::new(seed);
        }
        self.state = [self.random.f64_between(-0.6, -0.4), 0.0];
        Reset {
            observation: self.state.map(|value| value as f32),
            info: (),
        }
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let direction = match action {
            MountainCarAction::Left => -1.0,
            MountainCarAction::Coast => 0.0,
            MountainCarAction::Right => 1.0,
        };
        let [position, velocity] = self.state;
        // The source has no elapsed-time integration factor: 0.001 is the engine
        // velocity increment per step, and 0.0025 is the gravity increment scale.
        // Preserve its separate products and additions before velocity clipping.
        let engine_increment = direction * 0.001;
        let gravity_increment = (3.0 * position).cos() * (-0.0025);
        let increment = engine_increment + gravity_increment;
        let velocity = (velocity + increment).clamp(-0.07, 0.07);
        let position = (position + velocity).clamp(-1.2, 0.6);
        let velocity = if position <= -1.2 && velocity < 0.0 {
            0.0
        } else {
            velocity
        };
        self.state = [position, velocity];
        // The default goal requires both coordinates; passing it while moving
        // left does not terminate. The caller applies the 200-step time limit.
        let status = if position >= 0.5 && velocity >= 0.0 {
            EpisodeStatus::Terminated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.state.map(|value| value as f32),
            reward: -1.0,
            status,
            info: (),
        }
    }
}
