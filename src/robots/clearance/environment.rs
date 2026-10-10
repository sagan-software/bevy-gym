//! Clearance episodes compose the shared disturbed travel physics and range sensors.

use crate::robots::{DroneAction, DroneDestination, DroneObstacle, DroneTravel};
use crate::{Env, Reset, Step};

use super::DroneClearanceObservation;

/// Reach a fixed destination through collision geometry using four motor commands.
///
/// Resets use the disturbed recovery distribution and retain the destination and
/// validated static boxes. Contact or flight-region exit terminates the episode.
/// Empty geometry reproduces [`DroneTravel`] physics, rewards and reset streams.
/// The environment never chooses routes, waypoints, braking or motor actions.
/// Geometry and destinations may overlap; callers must configure feasible tasks.
///
/// Reward and the 20 ms action interval match [`DroneTravel`]. Each observation
/// pairs its position and heading task with six body-frame solid-ray readings.
/// Reads cost O(N) time and O(1) auxiliary space for N colliders; retained geometry
/// costs O(N) space. Use [`crate::TimeLimit`] for an episode horizon.
/// Qualification criteria belong to the curriculum, not this physical constructor.
///
/// ```compile_fail
/// use bevy::math::{Vec2, Vec3};
/// use bevy_gym::robots::{DroneClearance, DroneDestination};
/// let goal = DroneDestination::try_from((Vec3::Y * 2.0, Vec2::NEG_Y)).unwrap();
/// let environment = DroneClearance::new(goal, []);
/// let unchecked_physics = environment.travel;
/// ```
#[derive(Debug)]
pub struct DroneClearance {
    /// Shared travel task; its physics and collision geometry remain private.
    travel: DroneTravel,
}

impl DroneClearance {
    /// Start seed-zero disturbed flight with an immutable destination and static boxes.
    #[must_use]
    pub fn new(
        destination: DroneDestination,
        obstacles: impl IntoIterator<Item = DroneObstacle>,
    ) -> Self {
        Self {
            travel: DroneTravel::with_obstacles(destination, obstacles),
        }
    }

    /// Read the task, body and local ranges from one unchanged physical state.
    ///
    /// # Panics
    /// Panics if private physics violates its finite pose or bounded ray-hit invariants.
    #[must_use]
    pub fn observation(&self) -> DroneClearanceObservation {
        DroneClearanceObservation {
            travel: self.travel.observation(),
            ranges: self.travel.ranges(),
        }
    }
}

impl Env for DroneClearance {
    type Observation = DroneClearanceObservation;
    type Action = DroneAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        // Preserve the shared reset stream, destination and collision geometry.
        self.travel.reset(seed);
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        // Only the caller's validated motor command advances physics and earns reward.
        let result = self.travel.step(action);
        Step {
            observation: self.observation(),
            reward: result.reward,
            status: result.status,
            info: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robots::DroneRangeDirection;
    use bevy::math::{Quat, Vec2, Vec3};

    /// Private task replacement must not leave cached destination or sensor metadata.
    #[test]
    fn snapshot_derives_from_the_current_owned_travel_task() {
        let original = DroneDestination::try_from((Vec3::Y * 2.0, Vec2::NEG_Y)).unwrap();
        let replacement = DroneDestination::try_from((Vec3::new(0.0, 3.0, -4.0), Vec2::X)).unwrap();
        let mut environment = DroneClearance::new(original, []);
        let retained = environment.observation();
        let block = DroneObstacle::try_from((Vec3::Y * 2.0, Vec3::ONE, Quat::IDENTITY)).unwrap();
        // Only this private test can replace the owner; public fields stay inaccessible.
        environment.travel = DroneTravel::with_obstacles(replacement, [block]);
        let current = environment.observation();
        assert_eq!(current.travel().destination(), replacement);
        assert_eq!(retained.travel().destination(), original);
        assert_ne!(current.ranges(), retained.ranges());
        for direction in DroneRangeDirection::ALL {
            let hit = current
                .ranges()
                .distance(direction)
                .expect("inside private test box");
            assert_eq!(hit.metres().to_bits(), 0.0_f32.to_bits());
        }
    }
}
