use std::collections::HashMap;

/// A single reinforcement-learning environment.
///
/// The trait is intentionally small: reset starts a new episode, and step
/// advances the environment by one action. Batching, scheduling, rendering,
/// and training algorithms are runner concerns outside this contract.
pub trait Env {
    /// Observation type produced by [`Env::reset`] and [`Env::step`].
    type Observation: Clone + Send + Sync + 'static;

    /// Action type consumed by [`Env::step`].
    type Action: Clone + Send + Sync + 'static;

    /// Auxiliary information returned alongside observations.
    type Info: Default + Clone + Send + Sync + 'static;

    /// Start a new episode and return the initial observation.
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info>;

    /// Advance the environment by one action.
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info>;

    /// Scalar episode metrics emitted with [`crate::EpisodeEndEvent`].
    fn episode_extras(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

/// Reset output for a new episode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reset<O, I = ()> {
    /// Initial observation for the new episode.
    pub observation: O,

    /// Auxiliary reset metadata.
    pub info: I,
}

/// Output of one environment step.
#[derive(Debug, Clone, PartialEq)]
pub struct Step<O, I = ()> {
    /// Observation after applying the action.
    pub observation: O,

    /// Scalar reward for the transition.
    pub reward: f64,

    /// Whether the episode continues, terminated naturally, or was truncated.
    pub status: EpisodeStatus,

    /// Auxiliary step metadata.
    pub info: I,
}

impl<O, I> Step<O, I> {
    /// Returns true when the episode ended for any reason.
    #[inline]
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.status.is_done()
    }
}

/// Episode lifecycle status for a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpisodeStatus {
    /// The episode is still running.
    Continuing,

    /// The episode reached a natural terminal state; value bootstrapping stops.
    Terminated,

    /// The episode was cut short externally; next-state value can bootstrap.
    Truncated,
}

impl EpisodeStatus {
    /// Returns true for terminal or truncated steps.
    #[inline]
    #[must_use]
    pub const fn is_done(self) -> bool {
        matches!(self, Self::Terminated | Self::Truncated)
    }

    /// Returns true for natural terminal states only.
    #[inline]
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Terminated)
    }

    /// Returns true for external cutoffs only.
    #[inline]
    #[must_use]
    pub const fn is_truncated(self) -> bool {
        matches!(self, Self::Truncated)
    }

    /// Mask used by value-learning algorithms when bootstrapping targets.
    #[inline]
    #[must_use]
    pub const fn bootstrap_mask(self) -> f64 {
        match self {
            Self::Terminated => 0.0,
            Self::Continuing | Self::Truncated => 1.0,
        }
    }
}

/// A full transition produced by applying one action to one environment.
#[derive(Debug, Clone, PartialEq)]
pub struct Transition<O, A, I = ()> {
    /// Observation before the action.
    pub observation: O,

    /// Action applied to the environment.
    pub action: A,

    /// Scalar reward received.
    pub reward: f64,

    /// Observation after the action.
    pub next_observation: O,

    /// Episode status after the action.
    pub status: EpisodeStatus,

    /// Auxiliary step metadata.
    pub info: I,
}

/// A validation space for values of type `T`.
pub trait Space<T> {
    /// Returns true when `value` belongs to the space.
    fn contains(&self, value: &T) -> bool;
}

/// Discrete integer space `0..n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscreteSpace {
    /// Number of valid integer actions.
    pub n: usize,
}

impl Space<usize> for DiscreteSpace {
    fn contains(&self, value: &usize) -> bool {
        *value < self.n
    }
}

/// Fixed-size continuous box space for `f32` vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxSpace<const N: usize> {
    /// Inclusive lower bound per dimension.
    pub low: [f32; N],

    /// Inclusive upper bound per dimension.
    pub high: [f32; N],
}

impl<const N: usize> Space<[f32; N]> for BoxSpace<N> {
    fn contains(&self, value: &[f32; N]) -> bool {
        value
            .iter()
            .zip(self.low.iter().zip(self.high.iter()))
            .all(|(value, (low, high))| value.is_finite() && low <= value && value <= high)
    }
}

/// Optional spaces for environments that want validation metadata.
pub trait HasSpaces: Env {
    /// Action-space representation.
    type ActionSpace: Space<Self::Action>;

    /// Observation-space representation.
    type ObservationSpace: Space<Self::Observation>;

    /// Return the action space.
    fn action_space(&self) -> Self::ActionSpace;

    /// Return the observation space.
    fn observation_space(&self) -> Self::ObservationSpace;
}

/// Error returned by [`check_env_with_action`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvCheckError {
    /// Reset returned an observation outside the declared observation space.
    ResetObservationOutOfSpace,

    /// The provided action is outside the declared action space.
    ActionOutOfSpace,

    /// Step returned an observation outside the declared observation space.
    StepObservationOutOfSpace,

    /// Step returned a non-finite reward.
    RewardNotFinite,
}

/// Validate a single reset/step cycle for an environment with spaces.
///
/// # Errors
///
/// Returns [`EnvCheckError`] when the reset observation, provided action, step
/// observation, or reward violates the declared environment spaces.
pub fn check_env_with_action<E>(
    env: &mut E,
    seed: Option<u64>,
    action: E::Action,
) -> Result<(), EnvCheckError>
where
    E: HasSpaces,
{
    let reset = env.reset(seed);
    if !env.observation_space().contains(&reset.observation) {
        return Err(EnvCheckError::ResetObservationOutOfSpace);
    }

    if !env.action_space().contains(&action) {
        return Err(EnvCheckError::ActionOutOfSpace);
    }

    let step = env.step(action);
    if !step.reward.is_finite() {
        return Err(EnvCheckError::RewardNotFinite);
    }

    if !env.observation_space().contains(&step.observation) {
        return Err(EnvCheckError::StepObservationOutOfSpace);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_status_bootstrap_mask_preserves_terminal_distinction() {
        assert!(EpisodeStatus::Terminated.bootstrap_mask().abs() < f64::EPSILON);
        assert!((EpisodeStatus::Truncated.bootstrap_mask() - 1.0).abs() < f64::EPSILON);
        assert!((EpisodeStatus::Continuing.bootstrap_mask() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn discrete_space_accepts_only_values_below_n() {
        let space = DiscreteSpace { n: 2 };

        assert!(space.contains(&0));
        assert!(space.contains(&1));
        assert!(!space.contains(&2));
    }

    #[test]
    fn box_space_rejects_out_of_bounds_and_non_finite_values() {
        let space = BoxSpace {
            low: [-1.0, 0.0],
            high: [1.0, 2.0],
        };

        assert!(space.contains(&[0.0, 1.0]));
        assert!(!space.contains(&[2.0, 1.0]));
        assert!(!space.contains(&[0.0, f32::NAN]));
    }
}
