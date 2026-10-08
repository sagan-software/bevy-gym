//! Continuous action spaces supported by the shared PPO session.

/// A PPO task selects its original observation width and physical action bounds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ContinuousTask {
    /// Position and per-transition velocity with force in [-1, 1].
    MountainCar,
    /// Cosine, sine, and angular velocity with torque in [-2, 2] newton metres.
    Pendulum,
}

impl ContinuousTask {
    /// Actor and critic observation width after task-specific normalization.
    pub(crate) const fn observation_dim(self) -> usize {
        match self {
            Self::MountainCar => 2,
            Self::Pendulum => 3,
        }
    }

    /// Inclusive physical action bounds in the policy record's coordinate order.
    pub(crate) const fn action_bounds(self) -> (&'static [f32], &'static [f32]) {
        match self {
            Self::MountainCar => (&[-1.0], &[1.0]),
            Self::Pendulum => (&[-2.0], &[2.0]),
        }
    }
}
