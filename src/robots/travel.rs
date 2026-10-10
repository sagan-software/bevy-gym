//! Goal-conditioned flight over the shared four-motor rigid-body environment.

use bevy::math::Vec3;

use super::{
    DroneAction, DroneDestination, DroneHover, DroneObstacle, DroneRanges, DroneTravelObservation,
};
use crate::{Env, Reset, Step};

/// Reach a fixed position and heading through four validated motor commands.
///
/// Every reset uses [`DroneHover::disturbed`]'s distribution and retains the
/// destination. Physics, action bounds, 20 ms command interval, collisions and
/// terminal states are identical to that environment. No arrival action or
/// automatic route is generated. Use [`crate::TimeLimit`] for an episode horizon.
///
/// Continuing reward is `upright * alignment / (1 + distance_squared / (1 m²))`.
/// `upright = (1 + body_up.dot(world_up)) / 2` and
/// `alignment = (1 + body_forward.dot(desired_heading)) / 2`, each clamped to
/// `[0, 1]`. Body forward is -Z. Terminated episodes return zero reward and retain
/// their final snapshot. The destination describes a position and heading, not
/// an approach velocity. Later curriculum stages must define their speed gates.
/// This is local task policy; no external reward standard is claimed.
///
/// The environment adds constant work and storage to the shared physics step.
///
/// ```compile_fail
/// use bevy::math::{Vec2, Vec3};
/// use bevy_gym::robots::{DroneDestination, DroneTravel};
/// let goal = DroneDestination::try_from((Vec3::Y * 2.0, Vec2::X)).unwrap();
/// let environment = DroneTravel::new(goal);
/// let unchecked_physics = environment.body;
/// ```
#[derive(Debug)]
pub struct DroneTravel {
    /// Private physics environment shared with the recovery lesson.
    body: DroneHover,
    /// Immutable task condition; no navigation action is derived from it.
    destination: DroneDestination,
}

impl DroneTravel {
    /// Compose clearance geometry without exposing mutable physics to public callers.
    pub(super) fn with_obstacles(
        destination: DroneDestination,
        obstacles: impl IntoIterator<Item = DroneObstacle>,
    ) -> Self {
        Self {
            body: DroneHover::disturbed_with_obstacles(obstacles),
            destination,
        }
    }

    /// Read local ranges from the same private body used for travel observations.
    pub(super) fn ranges(&self) -> DroneRanges {
        self.body.ranges()
    }

    /// Start a disturbed episode at seed zero with the supplied destination.
    #[must_use]
    pub fn new(destination: DroneDestination) -> Self {
        Self {
            body: DroneHover::disturbed(),
            destination,
        }
    }

    /// Read the current task and physical state without advancing the solver.
    #[must_use]
    pub fn observation(&self) -> DroneTravelObservation {
        DroneTravelObservation {
            body: self.body.observation(),
            destination: self.destination,
        }
    }
}

impl Env for DroneTravel {
    type Observation = DroneTravelObservation;
    type Action = DroneAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        self.body.reset(seed);
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        // Forward exactly the caller's validated actuator command to the shared solver.
        let result = self.body.step(action);
        let observation = self.observation();
        // Terminal state stays frozen and cannot earn destination reward.
        let reward = if result.status.is_done() {
            0.0
        } else {
            reward(observation)
        };
        Step {
            observation,
            reward,
            status: result.status,
            info: (),
        }
    }
}

/// Score uprightness, position and heading without selecting any actuator command.
fn reward(observation: DroneTravelObservation) -> f64 {
    let body = observation.body();
    let destination = observation.destination();
    let upright = (body.orientation() * Vec3::Y)
        .y
        .midpoint(1.0)
        .clamp(0.0, 1.0);
    // Squared distance in m² divided by 1 m² gives a dimensionless denominator.
    let distance_squared = body.position().distance_squared(destination.position());
    let heading = destination.heading();
    let desired = Vec3::new(heading.x, 0.0, heading.y);
    let forward = body.orientation() * Vec3::NEG_Z;
    let alignment = forward.dot(desired).midpoint(1.0).clamp(0.0, 1.0);
    f64::from(upright * alignment / (1.0 + distance_squared))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::{Quat, Vec2};

    /// Exact geometric cases distinguish position and heading reward factors.
    #[test]
    fn reward_has_independent_position_heading_and_upright_factors() {
        let body = DroneHover::default().observation();
        let cases = [
            (Vec3::new(0.0, 2.0, 0.0), Vec2::NEG_Y, 1.0),
            (Vec3::new(1.0, 2.0, 0.0), Vec2::NEG_Y, 0.5),
            (Vec3::new(0.0, 2.0, 0.0), Vec2::X, 0.5),
            (Vec3::new(0.0, 2.0, 0.0), Vec2::Y, 0.0),
        ];
        for (position, heading, expected) in cases {
            let destination = DroneDestination::try_from((position, heading)).unwrap();
            assert!((reward(DroneTravelObservation { body, destination }) - expected).abs() < 1e-6);
        }
        let mut inverted = body;
        inverted.orientation = Quat::from_rotation_z(std::f32::consts::PI);
        let destination = DroneDestination::try_from((body.position(), Vec2::NEG_Y)).unwrap();
        assert!(
            reward(DroneTravelObservation {
                body: inverted,
                destination
            })
            .abs()
                < 1e-6
        );
    }
}
