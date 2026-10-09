//! Four-motor rigid-body hover task with a private physics world.

use std::{fmt, time::Duration};

use bevy::math::{Quat, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, ContactPair, IntegrationParameters, PhysicsWorld,
    RigidBodyBuilder, RigidBodyHandle, Rotation, Vector,
};

use super::{DroneAction, DroneEpisodeEnded, DroneMotor, DroneMotorState, DroneObservation};
use crate::training::SplitMix64;
use crate::{Env, EpisodeStatus, Reset, Step};

/// Hover near `(0, 2, 0)` metres using four ideal thrust actuators.
///
/// Each action spans 20 ms. Ground contact or leaving the flight region ends the
/// episode. The region extends 10 metres either side of the target and from
/// ground level to a 10-metre ceiling. Ended episodes earn zero reward until reset.
/// Use [`crate::TimeLimit`] to impose a maximum number of actions.
///
/// Reward is `upright / (1 + distance_squared)`, with distance measured in metres
/// and divided by one square metre. `upright` is `(1 + body_up.dot(world_up)) / 2`,
/// clamped to `[0, 1]`. Termination earns zero. Motors command ideal forces;
/// this task does not model rotor lag, drag, wind, or battery limits.
/// The physics world remains private so callers cannot bypass validated actions.
///
/// ```compile_fail
/// use bevy_gym::robots::DroneHover;
/// let drone = DroneHover::default();
/// let unchecked_world = drone.world;
/// ```
pub struct DroneHover {
    /// Solver state, isolated from callers and from every other environment.
    world: PhysicsWorld,
    /// The sole dynamic body, retained until the entire world is replaced.
    body: RigidBodyHandle,
    /// Drone collision proxy, retained until the entire world is replaced.
    collider: ColliderHandle,
    /// Reset stream; physics stepping never consumes random values.
    random: SplitMix64,
    /// Whether actions can still advance this episode.
    episode: Flight,
    /// Initial-condition distribution retained across episode resets.
    reset_profile: ResetProfile,
    /// Actuator health restored to working whenever the world is reset.
    motor_states: [DroneMotorState; 4],
}

/// Initial conditions selected by the two public constructors.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResetProfile {
    /// Upright and stationary, with only position offsets.
    Calm,
    /// Bounded tilt, heading, and velocity for feedback-control lessons.
    Disturbed,
}

/// Legal lifecycle states for one hover episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flight {
    /// The body is inside the flight region and has not touched the ground.
    Flying,
    /// A terminal transition has occurred; only reset can resume physics.
    Ended,
}

/// Duration held by one policy command.
const POLICY_INTERVAL: Duration = Duration::from_millis(20);
/// Solver interval; motor axes are recomputed before every substep.
const PHYSICS_INTERVAL: Duration = Duration::from_millis(5);
/// Body mass in kilograms, independent of the artist's material choices.
const MASS_KG: f32 = 1.0;
/// Gravitational acceleration magnitude in metres per second squared.
const GRAVITY: f32 = 9.81;
/// Each motor can produce half the body's weight in newtons.
const MAX_THRUST: f32 = MASS_KG * GRAVITY / 2.0;
/// Reaction moment divided by thrust, in metres.
const REACTION_ARM: f32 = 0.016;
/// Hover target position in metres.
const TARGET: Vec3 = Vec3::new(0.0, 2.0, 0.0);
/// Motor offsets in metres and alternating reaction signs, in action order.
///
/// These offsets follow the licensed model's rotor centres after its front faces -Z.
const MOTORS: [(Vector, f32); 4] = [
    (Vector::new(-0.2505, 0.0875, -0.2606), 1.0),
    (Vector::new(0.2505, 0.0875, -0.2606), -1.0),
    (Vector::new(0.2505, 0.0875, 0.2606), 1.0),
    (Vector::new(-0.2505, 0.0875, 0.2606), -1.0),
];

impl DroneHover {
    /// Disable one motor's force and reaction torque until reset.
    ///
    /// Repeated failure during flight succeeds without further changes. Failure
    /// retains body mass, collision geometry, pose, and velocity. It models an
    /// actuator failure, not a detached part.
    ///
    /// # Errors
    ///
    /// Returns `DroneEpisodeEnded` after termination, preserving the terminal state.
    ///
    /// ```compile_fail
    /// use bevy_gym::robots::DroneHover;
    /// DroneHover::default().fail_motor(4);
    /// ```
    pub fn fail_motor(&mut self, motor: DroneMotor) -> Result<(), DroneEpisodeEnded> {
        // Reject terminal mutations before changing any health or solver state.
        if self.episode == Flight::Ended {
            return Err(DroneEpisodeEnded);
        }
        let [front_left, front_right, rear_right, rear_left] = &mut self.motor_states;
        let state = match motor {
            DroneMotor::FrontLeft => front_left,
            DroneMotor::FrontRight => front_right,
            DroneMotor::RearRight => rear_right,
            DroneMotor::RearLeft => rear_left,
        };
        *state = DroneMotorState::Failed;
        Ok(())
    }

    /// Construct a hover lesson with randomized tilt and velocity on every reset.
    ///
    /// The initial episode uses seed zero. Position offsets match the calm task.
    /// Yaw spans ±π radians; pitch and roll span ±π/12 radians. Each world linear
    /// velocity component spans ±0.5 m/s, and each angular component ±0.5 rad/s.
    /// Reset preserves this profile. Actions, rewards, and termination are unchanged.
    #[must_use]
    pub fn disturbed() -> Self {
        let mut environment = Self {
            reset_profile: ResetProfile::Disturbed,
            ..Self::default()
        };
        environment.reset(Some(0));
        environment
    }

    /// Construct a fresh solver and proxy at a finite, internally chosen position.
    fn at_position(position: Vec3, random: SplitMix64) -> Self {
        let mut world = PhysicsWorld {
            gravity: Vector::Y * -GRAVITY,
            integration_parameters: IntegrationParameters {
                dt: PHYSICS_INTERVAL.as_secs_f32(),
                ..Default::default()
            },
            ..Default::default()
        };

        // The floor's upper face is y=0. It extends beyond every flight boundary.
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(30.0, 0.1, 30.0),
        );
        // A box approximates the model body; its mass determines rotational inertia.
        let (body, collider) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::from_array(position.to_array()))
                .can_sleep(false)
                .ccd_enabled(true),
            ColliderBuilder::cuboid(0.287, 0.104, 0.297).mass(MASS_KG),
        );
        Self {
            world,
            body,
            collider,
            random,
            episode: Flight::Flying,
            reset_profile: ResetProfile::Calm,
            motor_states: [DroneMotorState::Working; 4],
        }
    }

    /// Sample bounded initial motion after rebuilding the solver at a safe position.
    fn disturb_start(&mut self) {
        let mut sample = |bound: f64| self.random.f64_between(-bound, bound) as f32;
        // Angles are radians. The rightmost rotation acts first in R_y * R_x * R_z.
        let yaw = sample(std::f64::consts::PI);
        let pitch = sample(std::f64::consts::PI / 12.0);
        let roll = sample(std::f64::consts::PI / 12.0);
        let rotation = Rotation::from_rotation_y(yaw)
            * Rotation::from_rotation_x(pitch)
            * Rotation::from_rotation_z(roll);
        // These world-frame components are m/s and rad/s respectively.
        let linear_velocity = Vector::new(sample(0.5), sample(0.5), sample(0.5));
        let angular_velocity = Vector::new(sample(0.5), sample(0.5), sample(0.5));
        let body = self
            .world
            .bodies
            .get_mut(self.body)
            .expect("private body exists");
        body.set_rotation(rotation, true);
        body.set_linvel(linear_velocity, true);
        body.set_angvel(angular_velocity, true);
    }

    /// Copy the solver's current state across the private engine boundary.
    fn observation(&self) -> DroneObservation {
        let body = self
            .world
            .bodies
            .get(self.body)
            .expect("private body exists");
        DroneObservation {
            position: Vec3::from_array(body.translation().to_array()),
            orientation: Quat::from_array(body.rotation().to_array()),
            linear_velocity: Vec3::from_array(body.linvel().to_array()),
            angular_velocity: Vec3::from_array(body.angvel().to_array()),
            motor_states: self.motor_states,
        }
    }

    /// Replace persistent forces with this command in the body's current frame.
    fn apply_motors(&mut self, action: DroneAction) {
        let body = self
            .world
            .bodies
            .get_mut(self.body)
            .expect("private body exists");
        body.reset_forces(false);
        body.reset_torques(false);
        let rotation = *body.rotation();
        let position = body.translation();
        let up = rotation * Vector::Y;

        // A force in newtons at a world-space point produces torque in newton-metres.
        // Alternating reaction moments cancel when all motor commands are equal.
        for (((offset, sign), fraction), state) in MOTORS
            .into_iter()
            .zip(action.fractions())
            .zip(self.motor_states)
        {
            // A failed actuator contributes neither lift nor its reaction moment.
            let fraction = match state {
                DroneMotorState::Working => fraction,
                DroneMotorState::Failed => 0.0,
            };
            let force = fraction * MAX_THRUST;
            body.add_force_at_point(up * force, position + rotation * offset, true);
            body.add_torque(up * (sign * REACTION_ARM * force), true);
        }
    }

    /// End after contact or exit, including non-finite positions from solver failure.
    fn flight_ended(&self) -> bool {
        let position = self.observation().position();
        let inside = position.is_finite()
            && position.x.abs() <= 10.0
            && position.z.abs() <= 10.0
            && (0.0..=10.0).contains(&position.y);
        !inside
            || self
                .world
                .contact_pairs_with(self.collider)
                .any(ContactPair::has_any_active_contact)
    }
}

impl Default for DroneHover {
    fn default() -> Self {
        Self::at_position(TARGET, SplitMix64::new(0))
    }
}

impl fmt::Debug for DroneHover {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DroneHover")
            .field("observation", &self.observation())
            .field("episode", &self.episode)
            .finish_non_exhaustive()
    }
}

impl Env for DroneHover {
    type Observation = DroneObservation;
    type Action = DroneAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        // Explicit seeds restart the stream; omitted seeds continue it.
        let mut random = seed.map_or(self.random, SplitMix64::new);
        let offset = Vec3::new(
            random.f64_between(-0.2, 0.2) as f32,
            random.f64_between(-0.1, 0.1) as f32,
            random.f64_between(-0.2, 0.2) as f32,
        );
        // Rebuilding also clears cached contacts, islands, and solver impulses.
        let profile = self.reset_profile;
        *self = Self::at_position(TARGET + offset, random);
        self.reset_profile = profile;
        if profile == ResetProfile::Disturbed {
            self.disturb_start();
        }
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        if self.episode == Flight::Flying {
            for _ in 0..(POLICY_INTERVAL.as_nanos() / PHYSICS_INTERVAL.as_nanos()) {
                self.apply_motors(action);
                self.world.step();
                // Stop at the first terminal substep, rather than simulating past it.
                if self.flight_ended() {
                    self.episode = Flight::Ended;
                    break;
                }
            }
        }

        let observation = self.observation();
        let (status, reward) = match self.episode {
            Flight::Flying => {
                let upright = (observation.orientation() * Vec3::Y).y.midpoint(1.0);
                let distance_squared = observation.position().distance_squared(TARGET);
                let reward = upright.clamp(0.0, 1.0) / (1.0 + distance_squared);
                (EpisodeStatus::Continuing, f64::from(reward))
            }
            Flight::Ended => (EpisodeStatus::Terminated, 0.0),
        };
        Step {
            observation,
            reward,
            status,
            info: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flight_region_accepts_boundaries_and_rejects_each_outside_coordinate() {
        let mut drone = DroneHover::default();
        for position in [
            Vec3::new(-10.0, 2.0, 0.0),
            Vec3::new(10.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(0.0, 2.0, -10.0),
            Vec3::new(0.0, 2.0, 10.0),
        ] {
            drone
                .world
                .bodies
                .get_mut(drone.body)
                .expect("body exists")
                .set_translation(Vector::from_array(position.to_array()), true);
            assert!(!drone.flight_ended());
        }
        let outside = f32::from_bits(10.0_f32.to_bits() + 1);
        for position in [
            Vec3::new(-outside, 2.0, 0.0),
            Vec3::new(outside, 2.0, 0.0),
            Vec3::new(0.0, -f32::from_bits(1), 0.0),
            Vec3::new(0.0, outside, 0.0),
            Vec3::new(0.0, 2.0, -outside),
            Vec3::new(0.0, 2.0, outside),
            Vec3::splat(f32::NAN),
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NEG_INFINITY),
        ] {
            drone
                .world
                .bodies
                .get_mut(drone.body)
                .expect("body exists")
                .set_translation(Vector::from_array(position.to_array()), true);
            assert!(drone.flight_ended());
        }
    }

    #[test]
    fn thrust_axis_rotates_between_physics_substeps() {
        let mut drone = DroneHover::default();
        let body = drone.world.bodies.get_mut(drone.body).expect("body exists");
        body.set_rotation(Rotation::from_rotation_z(0.6), true);
        body.set_angvel(Vector::Z * 4.0, true);
        let hover = DroneAction::try_from([0.5; 4]).expect("valid action");
        let step = drone.step(hover);

        // Four force holds start at 0, 5, 10, and 15 ms. Pure local-Z rotation has
        // no gyroscopic moment, so angular speed stays at 4 radians per second.
        let expected = [0.0_f32, 0.005, 0.010, 0.015]
            .into_iter()
            .map(|seconds| {
                let angle = 4.0_f32.mul_add(seconds, 0.6);
                Vec3::new(-angle.sin(), angle.cos() - 1.0, 0.0) * GRAVITY * 0.005
            })
            .sum::<Vec3>();
        let actual = step.observation.linear_velocity();
        assert!(
            actual.distance(expected) < 1e-5,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn ceiling_exit_stops_after_the_first_terminal_substep() {
        let mut drone = DroneHover::at_position(Vec3::Y * 10.0, SplitMix64::new(0));
        let full = DroneAction::try_from([1.0; 4]).expect("full power");
        let step = drone.step(full);
        assert_eq!(step.status, EpisodeStatus::Terminated);
        let expected_velocity = GRAVITY * PHYSICS_INTERVAL.as_secs_f32();
        assert!((step.observation.linear_velocity().y - expected_velocity).abs() < 1e-5);
    }

    #[test]
    fn reward_uses_target_distance_and_body_uprightness() {
        let hover = DroneAction::try_from([0.5; 4]).expect("balanced action");
        let mut centred = DroneHover::default();
        assert!((centred.step(hover).reward - 1.0).abs() < 1e-6);
        let mut offset = DroneHover::at_position(TARGET + Vec3::X, SplitMix64::new(0));
        assert!((offset.step(hover).reward - 0.5).abs() < 1e-6);

        let mut tilted = DroneHover::default();
        tilted
            .world
            .bodies
            .get_mut(tilted.body)
            .expect("body exists")
            .set_rotation(Rotation::from_rotation_z(std::f32::consts::PI), true);
        assert!(tilted.step(hover).reward.abs() < 1e-6);
    }

    #[test]
    fn debug_output_exposes_state_without_a_mutable_world() {
        let drone = DroneHover::default();
        let description = format!("{drone:?}");
        assert!(description.contains("DroneHover"));
        assert!(description.contains("Flying"));
        assert!(description.contains("observation"));
    }

    #[test]
    fn failure_changes_only_health_before_the_next_physics_step() {
        let mut drone = DroneHover::disturbed();
        let before = drone.observation();
        drone
            .fail_motor(DroneMotor::FrontLeft)
            .expect("active flight");
        let after = drone.observation();
        assert_eq!(before.position(), after.position());
        assert_eq!(before.orientation(), after.orientation());
        assert_eq!(before.linear_velocity(), after.linear_velocity());
        assert_eq!(before.angular_velocity(), after.angular_velocity());
        assert_eq!(
            after.motor_state(DroneMotor::FrontLeft),
            DroneMotorState::Failed
        );
    }

    #[test]
    fn failed_motors_clear_previously_applied_force_and_reaction() {
        let mut drone = DroneHover::default();
        let full = DroneAction::try_from([1.0; 4]).expect("full power");
        let asymmetric = DroneAction::try_from([1.0, 0.0, 0.0, 0.0]).expect("one motor");
        drone.apply_motors(asymmetric);
        assert!(
            drone
                .world
                .bodies
                .get(drone.body)
                .expect("body")
                .user_force()
                .y
                > 0.0
        );
        assert!(
            drone
                .world
                .bodies
                .get(drone.body)
                .expect("body")
                .user_torque()
                .length()
                > 0.0
        );
        for motor in DroneMotor::ALL {
            drone.fail_motor(motor).expect("active flight");
        }
        drone.apply_motors(full);
        let body = drone.world.bodies.get(drone.body).expect("body");
        assert_eq!(body.user_force(), Vector::ZERO);
        assert_eq!(body.user_torque(), Vector::ZERO);
    }

    #[test]
    fn resetting_either_profile_discards_accumulated_motor_forces() {
        for mut drone in [DroneHover::default(), DroneHover::disturbed()] {
            let action = DroneAction::try_from([1.0, 0.0, 0.0, 0.0]).expect("one motor");
            drone.step(action);
            let body = drone
                .world
                .bodies
                .get(drone.body)
                .expect("private body exists");
            assert!(body.user_force().length_squared() > 0.0);
            assert!(body.user_torque().length_squared() > 0.0);
            drone.reset(None);
            let body = drone
                .world
                .bodies
                .get(drone.body)
                .expect("private body exists");
            assert_eq!(body.user_force(), Vector::ZERO);
            assert_eq!(body.user_torque(), Vector::ZERO);
        }
    }
}
