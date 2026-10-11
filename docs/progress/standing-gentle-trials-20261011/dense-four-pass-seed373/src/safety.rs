//! Research-only dense posture costs over original standing dynamics.
use bevy_gym::{
    robots::{DroidAction, DroidBody, DroidObservation},
    Env, Reset, Step,
};
/// Original standing dynamics with a dense resulting-state reward.
#[derive(Default)]
pub(crate) struct ControlTask {
    /// Original body, dynamics and termination owner.
    inner: bevy_gym::robots::DroidStanding,
    /// A naturally terminated frame cannot incur another reward or torque request.
    stopped: Option<Box<DroidObservation>>,
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
        step.reward = project_reward(&step.observation);
        // Retain the final physical frame while freezing later requests.
        if step.status.is_done() {
            // A natural fall forfeits the remaining standing task; charge once before freezing.
            step.reward -= 1000.0;
            self.stopped = Some(Box::new(step.observation));
        }
        step
    }
}
/// Project the same post-step body snapshot into dense, dimensionless training costs.
fn project_reward(observation: &DroidObservation) -> f64 {
    let pelvis = observation.body(DroidBody::Pelvis);
    let position_m = pelvis.position().as_dvec3();
    let linear_squared_m2_s2 = pelvis.linear_velocity().as_dvec3().length_squared();
    let horizontal_drift_squared_m2 = position_m.x.powi(2) + position_m.z.powi(2);
    let upright =
        f64::from((observation.body(DroidBody::Torso).orientation() * bevy::math::Vec3::Y).y)
            .clamp(-1.0, 1.0);
    let peak_squared_rad2_s2 = DroidBody::ALL
        .into_iter()
        .map(|body| {
            observation
                .body(body)
                .angular_velocity()
                .as_dvec3()
                .length_squared()
        })
        .fold(0.0_f64, f64::max);
    dense_reward(
        upright,
        position_m.y,
        linear_squared_m2_s2,
        horizontal_drift_squared_m2,
        peak_squared_rad2_s2,
        observation.body(DroidBody::LeftFoot).floor_contact(),
        observation.body(DroidBody::RightFoot).floor_contact(),
    )
}

/// Up is dimensionless; height and target are metres, normalized by 0.15 m.
/// Divide squared velocity by (0.25 m/s)², drift by (0.5 m)², and angular speed by (8 rad/s)².
/// Each unsupported foot costs 0.25; no term commands an actuator or changes physics.
fn dense_reward(
    up: f64,
    height_m: f64,
    speed_squared_m2_s2: f64,
    drift_squared_m2: f64,
    angular_squared_rad2_s2: f64,
    left: bool,
    right: bool,
) -> f64 {
    let height_error = (height_m - 1.0) / 0.15;
    let missing_feet = f64::from(!left) + f64::from(!right);
    1.0 - 10.0 * (1.0 - up)
        - height_error.powi(2)
        - 0.1 * speed_squared_m2_s2 / 0.0625
        - 0.1 * drift_squared_m2 / 0.25
        - 0.1 * angular_squared_rad2_s2 / 64.0
        - 0.25 * missing_feet
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Dense posture costs distinguish a surviving lean from upright standing.
    #[test]
    fn dense_posture_cost_has_independent_physical_terms() {
        assert_eq!(dense_reward(1.0, 1.0, 0.0, 0.0, 0.0, true, true), 1.0);
        assert!((dense_reward(0.67, 1.0, 0.0, 0.0, 0.0, true, true) + 2.3).abs() < 1e-12);
        for height in [0.85, 1.15] {
            assert!(dense_reward(1.0, height, 0.0, 0.0, 0.0, true, true).abs() < 1e-12);
        }
        for (speed_squared, drift_squared, angular_squared) in
            [(0.0625, 0.0, 0.0), (0.0, 0.25, 0.0), (0.0, 0.0, 64.0)]
        {
            assert!(
                (dense_reward(
                    1.0,
                    1.0,
                    speed_squared,
                    drift_squared,
                    angular_squared,
                    true,
                    true
                ) - 0.9)
                    .abs()
                    < 1e-12
            );
        }
        for (left, right, expected) in [
            (false, false, 0.5),
            (true, false, 0.75),
            (false, true, 0.75),
            (true, true, 1.0),
        ] {
            assert_eq!(dense_reward(1.0, 1.0, 0.0, 0.0, 0.0, left, right), expected);
        }
        let maximum = f64::from(f32::MAX);
        assert!(dense_reward(
            -1.0,
            maximum,
            3.0 * maximum.powi(2),
            2.0 * maximum.powi(2),
            3.0 * maximum.powi(2),
            false,
            false
        )
        .is_finite());
    }
    /// The cost cannot change any physical frame or natural terminal state.
    #[test]
    fn resulting_physics_and_terminal_frames_match_original() {
        let mut task = ControlTask::default();
        let mut original = bevy_gym::robots::DroidStanding::default();
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
                if !expected.status.is_done() {
                    assert_eq!(
                        actual.reward.to_bits(),
                        project_reward(&expected.observation).to_bits()
                    );
                }
                if expected.status.is_done() {
                    let expected_reward = project_reward(&expected.observation) - 1000.0;
                    assert_eq!(
                        actual.reward.to_bits(),
                        expected_reward.to_bits(),
                        "The first natural fall incurs its penalty exactly once."
                    );
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
