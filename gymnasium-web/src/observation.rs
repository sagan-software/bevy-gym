//! Fixed-size observations avoid allocating a vector on every transition.
use bevy_gym::EpisodeStatus;

/// Original environment observations, before learner-specific normalization.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Observation {
    /// Cart position, velocity, pole angle, and angular velocity.
    CartPole([f32; 4]),
    /// Car position and displacement per step.
    MountainCar([f32; 2]),
    /// Angle cosine, angle sine, and angular velocity in radians per second.
    Pendulum([f32; 3]),
}
impl AsRef<[f32]> for Observation {
    fn as_ref(&self) -> &[f32] {
        match self {
            Self::CartPole(values) => values,
            Self::MountainCar(values) => values,
            Self::Pendulum(values) => values,
        }
    }
}
impl Observation {
    /// Normalize position and speed using each native example's profile.
    pub(crate) fn encoded(self) -> Self {
        match self {
            Self::CartPole(_) => self,
            Self::MountainCar(values) => Self::MountainCar(encode_mountain_car(values)),
            Self::Pendulum([cosine, sine, velocity]) => {
                Self::Pendulum([cosine, sine, velocity / 8.0])
            }
        }
    }

    /// Transform only the optimizer reward; episode accounting retains the raw reward.
    ///
    /// Potential is dimensionless terrain height minus one; scale 25 converts
    /// its change to reward units. Gamma 0.99 matches the `MountainCar` learner.
    /// Pendulum scales its reward by the dimensionless factor 0.1.
    pub(crate) fn training_reward(self, next: Self, reward: f64, status: EpisodeStatus) -> f64 {
        match (self, next) {
            (Self::MountainCar([position, _]), Self::MountainCar([next_position, _])) => {
                let potential = f64::from((3.0 * position).sin().mul_add(0.45, 0.55) - 1.0);
                let next_potential = if status == EpisodeStatus::Terminated {
                    0.0
                } else {
                    f64::from((3.0 * next_position).sin().mul_add(0.45, 0.55) - 1.0)
                };
                25.0_f64.mul_add(0.99_f64.mul_add(next_potential, -potential), reward)
            }
            (Self::Pendulum(_), Self::Pendulum(_)) => reward * 0.1,
            _ => reward,
        }
    }
}

/// Map source position bounds [-1.2, 0.6] and per-step speed bounds [-0.07, 0.07] to [-1, 1].
pub(crate) fn encode_mountain_car([position, velocity]: [f32; 2]) -> [f32; 2] {
    [
        2.0 * (position - (-1.2_f32)) / (0.6_f32 - (-1.2_f32)) - 1.0,
        velocity / 0.07,
    ]
}

#[cfg(test)]
mod tests {
    use super::{EpisodeStatus, Observation};

    #[test]
    fn shaping_zeros_terminal_potential_but_keeps_truncated_bootstrap() {
        let current = Observation::MountainCar([-0.5, 0.0]);
        let next = Observation::MountainCar([0.5, 0.01]);
        let potential = f64::from((-1.5_f32).sin().mul_add(0.45, 0.55) - 1.0);
        let next_potential = f64::from(1.5_f32.sin().mul_add(0.45, 0.55) - 1.0);
        let terminal = current.training_reward(next, -1.0, EpisodeStatus::Terminated);
        let expected = 25.0_f64.mul_add(-potential, -1.0);
        assert!((terminal - expected).abs() < 1e-10);
        for status in [EpisodeStatus::Continuing, EpisodeStatus::Truncated] {
            let expected = 25.0_f64.mul_add(0.99_f64.mul_add(next_potential, -potential), -1.0);
            assert!((current.training_reward(next, -1.0, status) - expected).abs() < 1e-10);
        }
        let cart = Observation::CartPole([0.0; 4]);
        assert_eq!(
            cart.training_reward(cart, 1.0, EpisodeStatus::Terminated)
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn normalization_maps_declared_bounds_and_preserves_cartpole() {
        for (raw, expected) in [([-1.2, -0.07], [-1.0_f32, -1.0]), ([0.6, 0.07], [1.0, 1.0])] {
            let encoded = Observation::MountainCar(raw).encoded();
            assert_eq!(
                encoded
                    .as_ref()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.map(f32::to_bits)
            );
        }
        let raw = [0.1_f32, -0.2, 0.3, -0.4];
        assert_eq!(
            Observation::CartPole(raw)
                .encoded()
                .as_ref()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            raw.map(f32::to_bits)
        );
    }

    #[test]
    fn pendulum_normalizes_speed_and_scales_only_the_optimizer_reward() {
        for (velocity, expected) in [(-8.0, -1.0_f32), (0.0, 0.0), (8.0, 1.0)] {
            let raw = Observation::Pendulum([0.6, -0.8, velocity]);
            assert_eq!(raw.encoded().as_ref(), &[0.6, -0.8, expected]);
            assert_eq!(raw.as_ref(), &[0.6, -0.8, velocity]);
            for status in [
                EpisodeStatus::Continuing,
                EpisodeStatus::Truncated,
                EpisodeStatus::Terminated,
            ] {
                assert_eq!(
                    raw.training_reward(raw, -20.0, status).to_bits(),
                    (-2.0_f64).to_bits()
                );
            }
        }
    }
}
