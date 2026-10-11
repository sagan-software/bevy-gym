//! Research-only additive motion cost over unchanged physical dynamics.
use bevy_gym::{
    robots::{DroidAction, DroidBody, DroidObservation},
    Env, Reset, Step,
};
/// The original posture-height task with an additive resulting-state cost.
pub(crate) struct ControlTask {
    /// Original body, dynamics and termination owner.
    inner: crate::standing::reward::TrainingTask,
    /// A naturally terminated frame cannot incur another reward or torque request.
    stopped: Option<Box<DroidObservation>>,
}
impl Default for ControlTask {
    fn default() -> Self {
        Self {
            inner: crate::standing::reward::TrainingTask::posture_height(),
            stopped: None,
        }
    }
}
impl Env for ControlTask {
    type Observation = DroidObservation;
    type Action = DroidAction;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        self.stopped = None;
        self.inner.reset(seed)
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        if let Some(observation) = &self.stopped {
            return Step {
                observation: **observation,
                reward: 0.0,
                status: bevy_gym::EpisodeStatus::Terminated,
                info: (),
            };
        }
        // Forward policy torques without changing physics, observations or natural termination.
        let mut step = self.inner.step(action);
        // Angular components are rad/s; widen before squaring and compare all thirteen segments.
        let peak_squared_rad2_s2 = DroidBody::ALL
            .into_iter()
            .map(|body| {
                step.observation
                    .body(body)
                    .angular_velocity()
                    .as_dvec3()
                    .length_squared()
            })
            .fold(0.0_f64, f64::max);
        step.reward = reward(step.reward, peak_squared_rad2_s2);
        // Retain the final physical frame while freezing later requests.
        if step.status.is_done() {
            self.stopped = Some(Box::new(step.observation));
        }
        step
    }
}
/// Divide squared rad/s by (8 rad/s)^2; 0.1 multiplies the dimensionless motion cost.
fn reward(base: f64, peak_squared_rad2_s2: f64) -> f64 {
    base - 0.1 * peak_squared_rad2_s2 / 64.0
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Exact cost values remain finite for every finite f32 angular vector.
    #[test]
    fn additive_cost_does_not_saturate_at_large_speeds() {
        assert_eq!(reward(1.0, 0.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(reward(1.0, 64.0).to_bits(), 0.9_f64.to_bits());
        assert_eq!(reward(1.0, 256.0).to_bits(), 0.6_f64.to_bits());
        let largest = 3.0 * f64::from(f32::MAX).powi(2);
        assert!(reward(1.0, largest).is_finite());
        assert!(reward(1.0, largest) < reward(1.0, 256.0));
    }
    /// The cost cannot change any physical frame or natural terminal state.
    #[test]
    fn resulting_physics_and_terminal_frames_match_original() {
        let mut task = ControlTask::default();
        let mut original = crate::standing::reward::TrainingTask::posture_height();
        let action = DroidAction::try_from([0.0; 26]).expect("isolated mechanics fixture");
        for seed in [0, 1, 2, 42, u64::MAX] {
            assert_eq!(
                task.reset(Some(seed)).observation,
                original.reset(Some(seed)).observation
            );
            let mut terminated = false;
            for _ in 0..100 {
                let actual = task.step(action);
                let expected = original.step(action);
                assert_eq!(actual.observation, expected.observation);
                assert_eq!(actual.status, expected.status);
                if expected.status.is_done() {
                    let stopped = task.step(action);
                    let reference = original.step(action);
                    assert_eq!(stopped.observation, reference.observation);
                    assert_eq!(stopped.status, reference.status);
                    assert_eq!(stopped.reward.to_bits(), 0.0_f64.to_bits());
                    terminated = true;
                    break;
                }
            }
            assert!(terminated);
        }
    }
}
