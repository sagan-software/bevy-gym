//! Research-only stronger training failures; original selection remains unchanged.
use bevy_gym::{
    robots::{DroidAction, DroidActuator, DroidBody, DroidObservation},
    Env, EpisodeStatus, Reset, Step,
};
/// A stronger training task over the unchanged body and torque decoder.
pub(crate) struct ControlTask {
    /// Original posture-height reward task and physical owner.
    inner: crate::standing::reward::TrainingTask,
    /// Terminal frames cannot receive another torque request.
    state: State,
}
/// Mutually exclusive training lifecycle states.
enum State {
    /// Physical actions remain permitted.
    Ready,
    /// A natural or stronger training failure freezes this frame.
    Stopped(Box<DroidObservation>),
}
impl Default for ControlTask {
    fn default() -> Self {
        Self {
            inner: crate::standing::reward::TrainingTask::posture_height(),
            state: State::Ready,
        }
    }
}
impl Env for ControlTask {
    type Observation = DroidObservation;
    type Action = DroidAction;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        self.state = State::Ready;
        self.inner.reset(seed)
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        if let State::Stopped(observation) = &self.state {
            return Step {
                observation: **observation,
                reward: 0.0,
                status: EpisodeStatus::Terminated,
                info: (),
            };
        }
        // Forward policy torques unchanged before measuring the authoritative resulting bodies.
        let mut step = self.inner.step(action);
        // Widen rad/s components before computing the norm to avoid finite f32 overflow.
        let peak = DroidBody::ALL
            .into_iter()
            .map(|body| {
                step.observation
                    .body(body)
                    .angular_velocity()
                    .as_dvec3()
                    .length()
            })
            .fold(0.0_f64, f64::max);
        // Torque fractions and their mean square are dimensionless; 0.1 is a reward coefficient.
        let effort = DroidActuator::ALL
            .into_iter()
            .map(|axis| f64::from(action.fraction(axis)).powi(2))
            .sum::<f64>()
            / 26.0;
        step.reward = reward(step.reward, peak, effort);
        // The training failure is stricter than selection, which retains its original profile.
        if peak > 8.0 {
            step.status = EpisodeStatus::Terminated;
        }
        if step.status.is_done() {
            self.state = State::Stopped(Box::new(step.observation));
        }
        step
    }
}
/// Compare world segment speed in rad/s with the 8 rad/s training boundary.
fn reward(base: f64, peak_rad_s: f64, mean_squared_fraction: f64) -> f64 {
    if peak_rad_s > 8.0 {
        -1.0
    } else {
        base - 0.1 * mean_squared_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_speed_and_effort_boundaries() {
        assert_eq!(reward(1.0, 8.0, 0.0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(reward(1.0, 8.000001, 0.0).to_bits(), (-1.0_f64).to_bits());
        assert!((reward(1.0, 0.0, 1.0) - 0.9).abs() < f64::EPSILON);
    }
    #[test]
    fn physical_reset_and_steps_match_original_until_a_stronger_failure() {
        use bevy_gym::{
            robots::{DroidAction, DroidBody},
            Env,
        };
        let mut task = ControlTask::default();
        let mut base = crate::standing::reward::TrainingTask::posture_height();
        assert_eq!(
            task.reset(Some(42)).observation,
            base.reset(Some(42)).observation
        );
        let action = DroidAction::try_from([0.0; 26]).expect("passive diagnostic fixture");
        let mut stopped = false;
        for _ in 0..100 {
            let actual = task.step(action);
            let expected = base.step(action);
            assert_eq!(actual.observation, expected.observation);
            let speed = DroidBody::ALL
                .into_iter()
                .map(|b| expected.observation.body(b).angular_velocity().length())
                .fold(0.0_f32, f32::max);
            if speed > 8.0 {
                assert!(actual.status.is_done());
                assert_eq!(actual.reward.to_bits(), (-1.0_f64).to_bits());
                stopped = true;
                let frozen = task.step(action);
                assert_eq!(actual.observation, frozen.observation);
                assert_eq!(frozen.reward.to_bits(), 0.0_f64.to_bits());
                break;
            }
            assert_eq!(actual.status, expected.status);
        }
        assert!(stopped, "the fixture must exercise the stronger failure");
        assert_eq!(
            task.reset(Some(42)).observation,
            base.reset(Some(42)).observation
        );
        assert_eq!(task.step(action).observation, base.step(action).observation);
    }
}
