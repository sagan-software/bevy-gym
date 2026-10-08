//! Default Acrobot dynamics shared by native and browser consumers.
//!
//! Gymnasium revision `7a1191388aa4aa973d3a5e4b039899cd99cc991f`, `NumPy` 2.4.4.
//! <https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/acrobot.py>
//! The source derives from `RLPy` under BSD-3-Clause; see `LICENSES/ACROBOT-BSD-3-Clause.txt`.
//! Custom reset bounds, link parameters, timestep, torque noise, and NIPS dynamics
//! are unsupported. Reset uses `SplitMix64`; its seeds differ from `NumPy` PCG64.
use super::{AcrobotAction, AcrobotState};
use crate::training::split_mix64::SplitMix64;
use crate::{Env, EpisodeStatus, Reset, Step};

/// Default Acrobot-v1 book dynamics, without an episode time limit.
#[derive(Debug, Clone, Copy)]
pub struct Acrobot {
    /// Joint angles in radians, followed by angular velocities in radians per second.
    state: [f64; 4],
    /// Independent reset stream.
    random: SplitMix64,
}

impl Default for Acrobot {
    fn default() -> Self {
        Self {
            state: [0.0; 4],
            random: SplitMix64::new(0),
        }
    }
}
impl Acrobot {
    /// Inject a bounded diagnostic state without changing the default physics.
    #[must_use]
    pub const fn from_state(state: AcrobotState) -> Self {
        Self {
            state: *state.values(),
            random: SplitMix64::new(0),
        }
    }
    /// Borrow angles and angular velocities in source coordinate order.
    #[must_use]
    pub const fn state(&self) -> &[f64; 4] {
        &self.state
    }

    /// Project the evolved double state to the six float32 observations.
    fn observation(&self) -> [f32; 6] {
        let [first, second, first_velocity, second_velocity] = self.state;
        [
            first.cos() as f32,
            first.sin() as f32,
            second.cos() as f32,
            second.sin() as f32,
            first_velocity as f32,
            second_velocity as f32,
        ]
    }
}
impl Env for Acrobot {
    type Action = AcrobotAction;
    type Observation = [f32; 6];
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        if let Some(seed) = seed {
            self.random = SplitMix64::new(seed);
        }
        // The source rounds reset state before trigonometry, then promotes it in step.
        let state: [f32; 4] = std::array::from_fn(|_| self.random.f64_between(-0.1, 0.1) as f32);
        self.state = state.map(f64::from);
        let [first, second, first_velocity, second_velocity] = state;
        Reset {
            observation: [
                first.cos(),
                first.sin(),
                second.cos(),
                second.sin(),
                first_velocity,
                second_velocity,
            ],
            info: (),
        }
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let torque = match action {
            AcrobotAction::Negative => -1.0,
            AcrobotAction::Coast => 0.0,
            AcrobotAction::Positive => 1.0,
        };
        let next = integrate(self.state, torque);
        // Wrap each angle before clipping its velocity; both angle endpoints are retained.
        let pi = std::f64::consts::PI;
        self.state = [
            wrap(next[0]),
            wrap(next[1]),
            next[2].clamp(-4.0 * pi, 4.0 * pi),
            next[3].clamp(-9.0 * pi, 9.0 * pi),
        ];
        let terminated = terminal(self.state);
        Step {
            observation: self.observation(),
            reward: if terminated { 0.0 } else { -1.0 },
            status: if terminated {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

/// Compute the two angular velocities and two angular accelerations.
#[expect(
    clippy::suboptimal_flops,
    reason = "The pinned NumPy source rounds each product and sum separately"
)]
fn derivatives(state: [f64; 4], torque: f64) -> [f64; 4] {
    let [theta_1, theta_2, velocity_1, velocity_2] = state;
    // Both masses are 1 kg, link length is 1 m, centers are 0.5 m from each joint,
    // and moments of inertia are 1 kg m². Gravity is 9.8 m/s².
    let mass_1 = 1.0;
    let mass_2 = 1.0;
    let length = 1.0_f64;
    let center_1 = 0.5_f64;
    let center_2 = 0.5_f64;
    let inertia_1 = 1.0;
    let inertia_2 = 1.0;
    let gravity = 9.8;
    // Preserve the source's separate arithmetic operations and their evaluation order.
    // d_1 and d_2 have inertia units; phi_1 and phi_2 have torque units.
    let d_1 = mass_1 * center_1.powi(2)
        + mass_2 * (length.powi(2) + center_2.powi(2) + 2.0 * length * center_2 * theta_2.cos())
        + inertia_1
        + inertia_2;
    let d_2 = mass_2 * (center_2.powi(2) + length * center_2 * theta_2.cos()) + inertia_2;
    let phi_2 =
        mass_2 * center_2 * gravity * (theta_1 + theta_2 - std::f64::consts::PI / 2.0).cos();
    let phi_1 = -mass_2 * length * center_2 * velocity_2.powi(2) * theta_2.sin()
        - 2.0 * mass_2 * length * center_2 * velocity_2 * velocity_1 * theta_2.sin()
        + (mass_1 * center_1 + mass_2 * length)
            * gravity
            * (theta_1 - std::f64::consts::PI / 2.0).cos()
        + phi_2;
    // Dividing net torque by inertia gives angular acceleration in radians per second squared.
    let acceleration_2 = (torque + d_2 / d_1 * phi_1
        - mass_2 * length * center_2 * velocity_1.powi(2) * theta_2.sin()
        - phi_2)
        / (mass_2 * center_2.powi(2) + inertia_2 - d_2.powi(2) / d_1);
    let acceleration_1 = -(d_2 * acceleration_2 + phi_1) / d_1;
    [velocity_1, velocity_2, acceleration_1, acceleration_2]
}

/// Integrate one 0.2-second step with the original RK4 stage and summation order.
#[expect(
    clippy::suboptimal_flops,
    reason = "Fused arithmetic changes the pinned RK4 trajectory"
)]
fn integrate(state: [f64; 4], torque: f64) -> [f64; 4] {
    let seconds = 0.2;
    let first = derivatives(state, torque);
    let second = derivatives(add_scaled(state, first, seconds / 2.0), torque);
    let third = derivatives(add_scaled(state, second, seconds / 2.0), torque);
    let fourth = derivatives(add_scaled(state, third, seconds), torque);
    let mut next = state;
    // Each derivative is multiplied by seconds before adding it to its state coordinate.
    for ((((value, first), second), third), fourth) in next
        .iter_mut()
        .zip(first)
        .zip(second)
        .zip(third)
        .zip(fourth)
    {
        *value += seconds / 6.0 * (first + 2.0 * second + 2.0 * third + fourth);
    }
    next
}

/// Advance an RK4 intermediate state by a derivative times elapsed seconds.
#[expect(
    clippy::suboptimal_flops,
    reason = "The pinned RK4 stage rounds its product before addition"
)]
fn add_scaled(mut state: [f64; 4], derivative: [f64; 4], seconds: f64) -> [f64; 4] {
    for (value, slope) in state.iter_mut().zip(derivative) {
        *value += seconds * slope;
    }
    state
}

/// Preserve the source's subtraction order and inclusive ±pi endpoints.
#[expect(
    clippy::while_float,
    reason = "Bounded finite states require the source's endpoint-preserving wrap loops"
)]
fn wrap(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= std::f64::consts::TAU;
    }
    while angle < -std::f64::consts::PI {
        angle += std::f64::consts::TAU;
    }
    angle
}

/// The two 1-metre links must put their free end strictly above 1 metre.
fn terminal([first, second, _, _]: [f64; 4]) -> bool {
    -first.cos() - (second + first).cos() > 1.0
}

#[cfg(test)]
mod tests {
    use super::{terminal, wrap};
    use std::f64::consts::{PI, TAU};

    #[test]
    fn angle_wrapping_retains_both_endpoints_and_repeated_turns() {
        for turns in [1.0, 3.0, 5.0] {
            assert_eq!(wrap(turns * PI).to_bits(), PI.to_bits());
            assert_eq!(wrap(-turns * PI).to_bits(), (-PI).to_bits());
        }
        assert_eq!(wrap(0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(wrap(PI.next_up()).to_bits(), (PI.next_up() - TAU).to_bits());
        assert_eq!(
            wrap((-PI).next_down()).to_bits(),
            ((-PI).next_down() + TAU).to_bits()
        );
    }

    #[test]
    fn adjacent_goal_angles_straddle_the_strict_height_boundary() {
        let boundary = 2.0 * PI / 3.0;
        assert!(!terminal([boundary.next_down(), 0.0, 0.0, 0.0]));
        assert!(!terminal([boundary, 0.0, 0.0, 0.0]));
        assert!(terminal([boundary.next_up(), 0.0, 0.0, 0.0]));
    }
}
