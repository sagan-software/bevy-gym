//! Compare controllers with identical scheduled failures and post-failure horizons.

use std::error::Error;

use bevy::math::Vec3;
use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor, DroneObservation};
use bevy_gym::Env;
use serde::Serialize;

/// Fixed warm-up durations before one actuator fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureTime {
    /// Apply 100 intact 20 ms actions before failure.
    TwoSeconds,
    /// Apply 250 intact 20 ms actions before failure.
    FiveSeconds,
}

impl FailureTime {
    /// Convert the selected duration to a count of 20 ms environment actions.
    pub(crate) const fn actions(self) -> u16 {
        match self {
            Self::TwoSeconds => 100,
            Self::FiveSeconds => 250,
        }
    }
}

/// Legal results distinguish early termination, damaged termination, and survival.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum Outcome {
    /// The controller ended the episode before the scheduled failure.
    #[serde(rename = "before_failure")]
    ApproachEnded(Measurements),
    /// The episode terminated within the post-failure horizon.
    #[serde(rename = "crashed_after_failure")]
    Crashed(Measurements),
    /// All 500 post-failure actions completed without termination.
    #[serde(rename = "survived_after_failure")]
    Survived(Measurements),
}

/// Measurements for one phase only; intact-flight rewards never enter damage scores.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Measurements {
    /// Commands applied during this phase.
    pub(crate) steps: u16,
    /// Sum of the environment's rewards during this phase.
    pub(crate) reward: f64,
    /// Final distance from the hover target, in metres.
    pub(crate) final_distance_m: f32,
    /// Lowest body-centre height in this phase, including its initial state.
    pub(crate) minimum_height_m: f32,
    /// Greatest angle between body up and world up, in radians.
    pub(crate) peak_tilt_radians: f32,
    /// Final angular velocity around body up, in radians per second.
    pub(crate) final_body_yaw_rate_radians_per_second: f32,
}

impl Measurements {
    /// Begin a new score at the phase boundary without counting a reset as a step.
    fn new(observation: DroneObservation) -> Self {
        let body_velocity = observation.orientation().inverse() * observation.angular_velocity();
        Self {
            steps: 0,
            reward: 0.0,
            final_distance_m: observation.position().distance(Vec3::new(0.0, 2.0, 0.0)),
            minimum_height_m: observation.position().y,
            peak_tilt_radians: tilt(observation),
            final_body_yaw_rate_radians_per_second: body_velocity.y,
        }
    }

    /// Accumulate one accepted action and its resulting physical state.
    fn observe(&mut self, observation: DroneObservation, reward: f64) {
        self.steps += 1;
        self.reward += reward;
        self.final_distance_m = observation.position().distance(Vec3::new(0.0, 2.0, 0.0));
        self.minimum_height_m = self.minimum_height_m.min(observation.position().y);
        self.peak_tilt_radians = self.peak_tilt_radians.max(tilt(observation));
        let body_velocity = observation.orientation().inverse() * observation.angular_velocity();
        self.final_body_yaw_rate_radians_per_second = body_velocity.y;
    }
}

/// Whether physics terminated or consumed every command in the requested phase.
enum PhaseEnd {
    /// Physics terminated; no following phase may begin.
    Terminated(Measurements),
    /// Every requested command completed without termination.
    Horizon(Measurements),
}

/// Evaluate a calm seeded approach, then allow ten seconds after failure.
///
/// Work is linear in the bounded action count; auxiliary storage is constant.
/// The controller sees the changed health before choosing its next action.
pub(crate) fn assess(
    seed: u64,
    motor: DroneMotor,
    failure: FailureTime,
    mut command: impl FnMut(DroneObservation) -> Result<DroneAction, Box<dyn Error>>,
) -> Result<Outcome, Box<dyn Error>> {
    let mut environment = DroneHover::default();
    environment.reset(Some(seed));
    // An early crash belongs to the intact approach, not the damage trial.
    if let PhaseEnd::Terminated(score) = phase(&mut environment, failure.actions(), &mut command)? {
        return Ok(Outcome::ApproachEnded(score));
    }
    environment.fail_motor(motor)?;
    // Start fresh metrics at failure; both schedules receive 500 post-failure actions.
    Ok(match phase(&mut environment, 500, &mut command)? {
        PhaseEnd::Terminated(score) => Outcome::Crashed(score),
        PhaseEnd::Horizon(score) => Outcome::Survived(score),
    })
}

/// Advance one bounded phase, stopping before any action after termination.
fn phase(
    environment: &mut DroneHover,
    actions: u16,
    command: &mut impl FnMut(DroneObservation) -> Result<DroneAction, Box<dyn Error>>,
) -> Result<PhaseEnd, Box<dyn Error>> {
    let mut observation = environment.observation();
    let mut score = Measurements::new(observation);
    for _ in 0..actions {
        let result = environment.step(command(observation)?);
        observation = result.observation;
        score.observe(observation, result.reward);
        if result.status.is_done() {
            return Ok(PhaseEnd::Terminated(score));
        }
    }
    Ok(PhaseEnd::Horizon(score))
}

/// Clamp quaternion roundoff before converting the up-vector dot product to radians.
fn tilt(observation: DroneObservation) -> f32 {
    (observation.orientation() * Vec3::Y)
        .y
        .clamp(-1.0, 1.0)
        .acos()
}
