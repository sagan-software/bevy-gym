//! Walking episodes with policy-driven torques and explicit physical failure states.

use super::super::{
    physics::{Articulation, Floor, PHYSICS_INTERVAL},
    DroidAction, DroidBody, DroidObservation, DroidStanding,
};
use super::{
    reward, work, DroidVelocityCommand, DroidWalkingFall, DroidWalkingObservation,
    DroidWalkingStatus,
};
use crate::{training::SplitMix64, Env, EpisodeStatus, Reset, Step};
use rapier3d::prelude::{Rotation, Vector};
use std::{fmt, time::Duration};

/// Track a bounded horizontal command through the original droid's 26 torques.
///
/// A 30 m half-width floor supports the independent walking task. The body,
/// actuators and 20 ms action interval match standing and the shared world.
/// The command stays fixed until the environment is replaced. Reset samples
/// the original yaw, pitch and roll distribution and clears all body motion.
/// Use [`crate::TimeLimit`] with 1,000 actions for a nominal 20-second episode.
/// No walking policy or qualification is supplied by this mechanical task.
///
/// Validation rejects nonfinite segment state before checking pelvis height,
/// segment bounds and non-foot floor contact, in that order. A physical fall
/// or failed physics freezes the body and returns zero reward until reset.
/// [`DroidWalkingStatus`] distinguishes those outcomes despite their common
/// [`EpisodeStatus::Terminated`] status. Partial failed advancement is not rolled back.
///
/// ```compile_fail
/// use bevy_gym::robots::DroidWalking;
/// let solver = DroidWalking::default().articulation;
/// ```
pub struct DroidWalking {
    /// Sole physical owner; callers cannot mutate its solver or body transforms.
    articulation: Articulation,
    /// Original seeded initial-pose stream, independent of agent decisions.
    random: SplitMix64,
    /// Validated lesson command, fixed for this environment's lifetime.
    command: DroidVelocityCommand,
    /// Closed lifecycle; the common episode status is derived from this value.
    status: DroidWalkingStatus,
    /// Accumulated absolute actuator work estimate in joules; never actor input.
    mechanical_work_joules: f64,
    /// Actual solver time, including a terminating substep.
    elapsed: Duration,
    /// Integrated maximum foot solver-contact sliding estimates in metres, left then right.
    foot_contact_sliding_distance_metres: [f64; 2],
}

impl DroidWalking {
    /// A request advances at most 20 ms; physical termination can stop its four substeps early.
    pub const POLICY_INTERVAL: Duration = DroidStanding::POLICY_INTERVAL;

    /// Create the unchanged body for one validated lesson command.
    #[must_use]
    pub fn new(command: DroidVelocityCommand) -> Self {
        Self::from_random(command, SplitMix64::new(0))
    }

    /// Return absolute mechanical actuator work estimated at solver boundaries, in joules.
    ///
    /// Each 5 ms substep integrates absolute actuator power with a trapezoidal rule.
    /// Torque is the original parent-frame request held throughout that substep.
    /// This estimate excludes gravity, contact work and electrical losses. It resets
    /// with the episode and freezes after a fall. Failed physics returns `None`.
    #[must_use]
    pub const fn mechanical_work_joules(&self) -> Option<f64> {
        match self.status {
            DroidWalkingStatus::Failed(_) => None,
            DroidWalkingStatus::Running | DroidWalkingStatus::Fallen(_) => {
                Some(self.mechanical_work_joules)
            }
        }
    }

    /// Return cumulative foot solver-contact sliding estimates in metres, left then right.
    ///
    /// Each completed 5 ms substep adds its maximum horizontal contact-point speed
    /// multiplied by 0.005 s. Rotation contributes to point velocity. No contact adds
    /// zero. Predictive solver contacts count; this estimate does not identify loaded
    /// contact or exact continuous slip. Reset clears it, a fall freezes it, and failed
    /// physics returns `None`. It never enters observations, rewards or actuation.
    #[must_use]
    pub const fn foot_contact_sliding_distance_metres(&self) -> Option<[f64; 2]> {
        match self.status {
            DroidWalkingStatus::Failed(_) => None,
            DroidWalkingStatus::Running | DroidWalkingStatus::Fallen(_) => {
                Some(self.foot_contact_sliding_distance_metres)
            }
        }
    }

    /// Return actual advanced solver time; terminal requests do not advance it.
    #[must_use]
    pub const fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Retain the original pose distribution without choosing any actuator request.
    fn from_random(command: DroidVelocityCommand, mut random: SplitMix64) -> Self {
        let yaw = random.f64_between(-std::f64::consts::PI, std::f64::consts::PI) as f32;
        let pitch = random.f64_between(-0.01, 0.01) as f32;
        let roll = random.f64_between(-0.01, 0.01) as f32;
        let rotation = Rotation::from_rotation_y(yaw)
            * Rotation::from_rotation_x(pitch)
            * Rotation::from_rotation_z(roll);
        Self {
            articulation: Articulation::new_for_floor(rotation, Vector::Y * 0.03, Floor::Walking),
            random,
            command,
            status: DroidWalkingStatus::Running,
            mechanical_work_joules: 0.0,
            elapsed: Duration::ZERO,
            foot_contact_sliding_distance_metres: [0.0; 2],
        }
    }

    /// Own the bounded thirteen-segment snapshot and its immutable task command.
    fn observation(&self) -> DroidWalkingObservation {
        DroidWalkingObservation {
            physical: self.articulation.observation(),
            command: self.command,
        }
    }

    /// Record one completed solver substep, including physical termination.
    fn record_substep(
        &mut self,
        before: &DroidObservation,
        after: &DroidObservation,
        action: DroidAction,
    ) {
        self.elapsed += PHYSICS_INTERVAL;
        self.status = Self::classify(after);
        // If physics failed, expose unavailable work instead of consuming invalid state.
        if !matches!(self.status, DroidWalkingStatus::Failed(_)) {
            self.mechanical_work_joules += work::estimate(before, after, action);
            // Speed in m/s times the completed solver interval in seconds gives metres.
            for (distance_m, speed_m_s) in self
                .foot_contact_sliding_distance_metres
                .iter_mut()
                .zip(self.articulation.foot_contact_speeds())
            {
                *distance_m = speed_m_s.mul_add(PHYSICS_INTERVAL.as_secs_f64(), *distance_m);
            }
        }
    }

    /// Classify all finite-state failures before ordinary task termination.
    fn classify(observation: &DroidObservation) -> DroidWalkingStatus {
        // Check every segment before comparisons or reward arithmetic consume its values.
        for identity in DroidBody::ALL {
            let body = observation.body(identity);
            if !body.position().is_finite()
                || !body.orientation().is_finite()
                || !body.linear_velocity().is_finite()
                || !body.angular_velocity().is_finite()
            {
                return DroidWalkingStatus::Failed(identity);
            }
        }
        if observation.body(DroidBody::Pelvis).position().y < 0.5 {
            return DroidWalkingStatus::Fallen(DroidWalkingFall::LowPelvis);
        }
        // Check every segment's metre bounds before considering any contact failure.
        for identity in DroidBody::ALL {
            let body = observation.body(identity);
            let position = body.position();
            if position.x.abs() > 29.0
                || position.z.abs() > 29.0
                || !(0.0..=3.0).contains(&position.y)
            {
                return DroidWalkingStatus::Fallen(DroidWalkingFall::OutsideBounds(identity));
            }
        }
        // Only feet may support this walking task; contact never chooses an action.
        for identity in DroidBody::ALL {
            let body = observation.body(identity);
            if body.floor_contact()
                && !matches!(identity, DroidBody::LeftFoot | DroidBody::RightFoot)
            {
                return DroidWalkingStatus::Fallen(DroidWalkingFall::NonFootContact(identity));
            }
        }
        DroidWalkingStatus::Running
    }
}

impl Default for DroidWalking {
    fn default() -> Self {
        Self::new(DroidVelocityCommand::default())
    }
}

impl fmt::Debug for DroidWalking {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DroidWalking")
            .field("command", &self.command)
            .field("status", &self.status)
            .finish_non_exhaustive()
    }
}

impl Env for DroidWalking {
    type Observation = DroidWalkingObservation;
    type Action = DroidAction;
    type Info = DroidWalkingStatus;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        // Rebuild all physics and clear the lifecycle while retaining the lesson command.
        *self = Self::from_random(self.command, seed.map_or(self.random, SplitMix64::new));
        Reset {
            observation: self.observation(),
            info: self.status,
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        if self.status == DroidWalkingStatus::Running {
            // Reject invalid current physics before applying any request.
            self.status = Self::classify(&self.articulation.observation());
            let substeps = Self::POLICY_INTERVAL.as_nanos() / PHYSICS_INTERVAL.as_nanos();
            for _ in 0..substeps {
                if self.status != DroidWalkingStatus::Running {
                    break;
                }
                // Only the caller's validated torques enter the original body model.
                let before = self.articulation.observation();
                self.articulation.apply(action);
                self.articulation.world.step();
                let after = self.articulation.observation();
                self.record_substep(&before, &after, action);
            }
        }
        let observation = self.observation();
        // Derive episode status and reward from the retained lifecycle, never a second flag.
        let (status, reward) = match self.status {
            DroidWalkingStatus::Running => {
                (EpisodeStatus::Continuing, reward::measure(&observation))
            }
            DroidWalkingStatus::Fallen(_) | DroidWalkingStatus::Failed(_) => {
                (EpisodeStatus::Terminated, 0.0)
            }
        };
        Step {
            observation,
            reward,
            status,
            info: self.status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DroidBody, DroidObservation, DroidWalking, DroidWalkingFall, DroidWalkingStatus};
    use crate::{
        robots::{DroidAction, DroidBodyState},
        Env, EpisodeStatus,
    };
    use bevy::math::Vec3;
    use rapier3d::prelude::Vector;

    /// Quantities poisoned only in isolated validation fixtures.
    #[derive(Clone, Copy)]
    enum Measurement {
        /// World centre in metres.
        Position,
        /// Dimensionless quaternion component.
        Orientation,
        /// World velocity in metres per second.
        LinearVelocity,
        /// World angular velocity in radians per second.
        AngularVelocity,
    }

    /// Create a finite own-body snapshot without choosing or advancing any action.
    fn initial() -> DroidObservation {
        *DroidWalking::default()
            .reset(Some(42))
            .observation
            .physical()
    }

    /// Alter an isolated measurement fixture through its closed segment identity.
    fn body_mut(observation: &mut DroidObservation, identity: DroidBody) -> &mut DroidBodyState {
        observation
            .bodies
            .get_mut(identity as usize)
            .expect("closed body identity")
    }

    /// Invalid state, low pelvis, bounds and contact have a fixed diagnostic priority.
    #[test]
    fn classification_preserves_failure_priority() {
        let mut observation = initial();
        body_mut(&mut observation, DroidBody::Pelvis).position.y = 0.0;
        body_mut(&mut observation, DroidBody::Head).position.x = f32::NAN;
        assert_eq!(
            DroidWalking::classify(&observation),
            DroidWalkingStatus::Failed(DroidBody::Head)
        );

        body_mut(&mut observation, DroidBody::Head).position.x = 30.0;
        assert_eq!(
            DroidWalking::classify(&observation),
            DroidWalkingStatus::Fallen(DroidWalkingFall::LowPelvis)
        );

        let mut observation = initial();
        body_mut(&mut observation, DroidBody::Torso).floor_contact = true;
        body_mut(&mut observation, DroidBody::Head).position.x = 30.0;
        assert_eq!(
            DroidWalking::classify(&observation),
            DroidWalkingStatus::Fallen(DroidWalkingFall::OutsideBounds(DroidBody::Head))
        );
    }

    /// Every segment and measured quantity rejects NaN and both infinities.
    #[test]
    fn every_nonfinite_quantity_identifies_its_segment() {
        let original = initial();
        for identity in DroidBody::ALL {
            for quantity in [
                Measurement::Position,
                Measurement::Orientation,
                Measurement::LinearVelocity,
                Measurement::AngularVelocity,
            ] {
                for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    let mut observation = original;
                    let body = body_mut(&mut observation, identity);
                    match quantity {
                        Measurement::Position => body.position.x = invalid,
                        Measurement::Orientation => body.orientation.x = invalid,
                        Measurement::LinearVelocity => body.linear_velocity.x = invalid,
                        Measurement::AngularVelocity => body.angular_velocity.x = invalid,
                    }
                    assert_eq!(
                        DroidWalking::classify(&observation),
                        DroidWalkingStatus::Failed(identity)
                    );
                }
            }
        }
        let mut observation = original;
        body_mut(&mut observation, DroidBody::Pelvis)
            .linear_velocity
            .x = f32::NAN;
        body_mut(&mut observation, DroidBody::Head).position.x = f32::NAN;
        assert_eq!(
            DroidWalking::classify(&observation),
            DroidWalkingStatus::Failed(DroidBody::Pelvis)
        );
    }

    /// X/Z ±29 m and Y [0, 3] m are inclusive, with a separate 0.5 m pelvis minimum.
    #[test]
    fn segment_bounds_keep_exact_edges() {
        let original = initial();
        for identity in DroidBody::ALL {
            let minimum_y_m = if identity == DroidBody::Pelvis {
                0.5
            } else {
                0.0
            };
            for position in [
                Vec3::new(-29.0, 1.0, 0.0),
                Vec3::new(29.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, -29.0),
                Vec3::new(0.0, 1.0, 29.0),
                Vec3::new(0.0, 3.0, 0.0),
                Vec3::new(0.0, minimum_y_m, 0.0),
            ] {
                let mut observation = original;
                body_mut(&mut observation, identity).position = position;
                assert_eq!(
                    DroidWalking::classify(&observation),
                    DroidWalkingStatus::Running
                );
            }
        }
    }

    /// Adjacent horizontal and upper-height values reject the affected segment.
    #[test]
    fn segment_bounds_reject_adjacent_values() {
        let original = initial();
        let outside_horizontal_m = f32::from_bits(29.0_f32.to_bits() + 1);
        let outside_upper_m = f32::from_bits(3.0_f32.to_bits() + 1);
        for identity in DroidBody::ALL {
            for position in [
                Vec3::new(-outside_horizontal_m, 1.0, 0.0),
                Vec3::new(outside_horizontal_m, 1.0, 0.0),
                Vec3::new(0.0, 1.0, -outside_horizontal_m),
                Vec3::new(0.0, 1.0, outside_horizontal_m),
                Vec3::new(0.0, outside_upper_m, 0.0),
            ] {
                let mut observation = original;
                body_mut(&mut observation, identity).position = position;
                assert_eq!(
                    DroidWalking::classify(&observation),
                    DroidWalkingStatus::Fallen(DroidWalkingFall::OutsideBounds(identity))
                );
            }
            let mut observation = original;
            body_mut(&mut observation, identity).position.y = -f32::from_bits(1);
            let reason = if identity == DroidBody::Pelvis {
                DroidWalkingFall::LowPelvis
            } else {
                DroidWalkingFall::OutsideBounds(identity)
            };
            assert_eq!(
                DroidWalking::classify(&observation),
                DroidWalkingStatus::Fallen(reason)
            );
        }
        let mut observation = original;
        body_mut(&mut observation, DroidBody::Pelvis).position.y =
            f32::from_bits(0.5_f32.to_bits() - 1);
        assert_eq!(
            DroidWalking::classify(&observation),
            DroidWalkingStatus::Fallen(DroidWalkingFall::LowPelvis)
        );
    }

    /// Both feet may contact the floor; every other contacting segment causes a fall.
    #[test]
    fn only_feet_can_support_the_walking_body() {
        let original = initial();
        for identity in DroidBody::ALL {
            let mut observation = original;
            body_mut(&mut observation, identity).floor_contact = true;
            let expected = match identity {
                DroidBody::LeftFoot | DroidBody::RightFoot => DroidWalkingStatus::Running,
                DroidBody::Pelvis
                | DroidBody::Torso
                | DroidBody::Head
                | DroidBody::LeftUpperArm
                | DroidBody::LeftForearm
                | DroidBody::RightUpperArm
                | DroidBody::RightForearm
                | DroidBody::LeftThigh
                | DroidBody::LeftCalf
                | DroidBody::RightThigh
                | DroidBody::RightCalf => {
                    DroidWalkingStatus::Fallen(DroidWalkingFall::NonFootContact(identity))
                }
            };
            assert_eq!(DroidWalking::classify(&observation), expected);
        }
    }

    /// A completed invalid substep retains elapsed time but cannot report valid work.
    #[test]
    fn failed_completed_substep_invalidates_work_without_consuming_invalid_floats() {
        let mut walking = DroidWalking::default();
        let before = *walking.reset(Some(42)).observation.physical();
        let mut after = before;
        body_mut(&mut after, DroidBody::Head).angular_velocity.x = f32::NAN;
        let action = DroidAction::try_from([1.0; 26]).expect("isolated failure fixture");
        walking.record_substep(&before, &after, action);
        assert_eq!(walking.status, DroidWalkingStatus::Failed(DroidBody::Head));
        assert_eq!(walking.elapsed(), std::time::Duration::from_millis(5));
        assert_eq!(walking.mechanical_work_joules(), None);
        assert_eq!(walking.foot_contact_sliding_distance_metres(), None);
        assert_eq!(walking.mechanical_work_joules.to_bits(), 0.0_f64.to_bits());
        walking.reset(Some(42));
        assert_eq!(walking.mechanical_work_joules(), Some(0.0));
        assert_eq!(walking.elapsed(), std::time::Duration::ZERO);
    }

    /// Invalid current velocity stops before torque application or solver advancement.
    #[test]
    fn invalid_current_physics_retains_failure_without_actuation_until_reset() {
        let mut walking = DroidWalking::default();
        let reset = walking.reset(Some(42));
        let (_, pelvis) = walking
            .articulation
            .world
            .bodies
            .iter_mut()
            .find(|(_, body)| body.is_dynamic())
            .expect("first dynamic segment");
        assert_eq!(pelvis.mass().to_bits(), 12.0_f32.to_bits());
        pelvis.set_linvel(Vector::new(f32::NAN, 0.0, 0.0), true);
        let action = DroidAction::try_from([1.0; 26]).expect("isolated bounded fixture");
        for _ in 0..2 {
            let stopped = walking.step(action);
            assert_eq!(stopped.info, DroidWalkingStatus::Failed(DroidBody::Pelvis));
            assert_eq!(walking.mechanical_work_joules(), None);
            assert_eq!(walking.foot_contact_sliding_distance_metres(), None);
            assert_eq!(walking.elapsed(), std::time::Duration::ZERO);
            assert_eq!(stopped.status, EpisodeStatus::Terminated);
            assert_eq!(stopped.reward.to_bits(), 0.0_f64.to_bits());
            for identity in DroidBody::ALL {
                assert_eq!(
                    stopped.observation.physical().body(identity).position(),
                    reset.observation.physical().body(identity).position()
                );
                assert_eq!(
                    stopped.observation.physical().body(identity).orientation(),
                    reset.observation.physical().body(identity).orientation()
                );
            }
            assert!(walking
                .articulation
                .world
                .bodies
                .iter()
                .all(|(_, body)| body.user_torque() == Vector::ZERO));
        }
        assert_eq!(walking.reset(Some(42)), reset);
    }
}
