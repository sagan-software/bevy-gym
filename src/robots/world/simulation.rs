//! One private solver shared by all six physical robot bodies.

use super::{
    RobotActions, RobotFrame, RobotId, RobotSlot, RobotSnapshot, RobotWorldError, RobotWorldFailure,
};
use crate::robots::{droid::DroidRig, drone::DroneModel, DroneMotorState};
use bevy::math::{Quat, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, IntegrationParameters, PhysicsWorld, RigidBodyBuilder, Rotation, Vector,
};
use std::fmt;

/// Shared physical foundation for the three-versus-three match, without a controller.
/// Competitive health, weapons, tactics and learned-policy qualification remain separate work.
///
/// ```compile_fail
/// use bevy_gym::robots::RobotWorld;
/// let mut world = RobotWorld::default();
/// world.world.step();
/// ```
pub struct RobotWorld {
    /// Sole collision and integration owner; no mutable backend is exposed.
    world: PhysicsWorld,
    /// Three rotor bodies sharing the solver.
    drones: [DroneModel; 3],
    /// Three thirteen-segment articulations sharing the solver.
    droids: [DroidRig; 3],
    /// Current world/reset scope and physical decision boundary.
    frame: RobotFrame,
    /// Mutually exclusive advancement capabilities.
    state: State,
}

/// Only a ready world can advance; failure retains its original diagnostic.
enum State {
    /// Complete validated actuator frames remain permitted.
    Ready,
    /// Only explicit reset can permit advancement again.
    Stopped(RobotWorldFailure),
}

impl Default for RobotWorld {
    fn default() -> Self {
        // The shared floor is application policy; standalone lesson floors remain unchanged.
        let mut world = PhysicsWorld {
            gravity: Vector::NEG_Y * 9.81,
            integration_parameters: IntegrationParameters {
                dt: 0.005,
                ..Default::default()
            },
            ..Default::default()
        };
        let (_, floor) = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(30.0, 0.1, 30.0).friction(0.8),
        );
        let drones = RobotSlot::ALL.map(|slot| {
            let z = lane(slot);
            DroneModel::insert(&mut world, Vec3::new(-3.0, 2.0, z))
        });
        let droids = RobotSlot::ALL.map(|slot| {
            let z = lane(slot);
            DroidRig::insert(
                &mut world,
                floor,
                Rotation::IDENTITY,
                Vector::new(3.0, 0.03, z),
            )
        });
        Self {
            world,
            drones,
            droids,
            frame: RobotFrame::first(),
            state: State::Ready,
        }
    }
}
impl fmt::Debug for RobotWorld {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RobotWorld")
            .field("frame", &self.frame)
            .finish_non_exhaustive()
    }
}
impl RobotWorld {
    /// Capture all six bodies without changing physics or granting mutable access.
    ///
    /// # Errors
    /// Returns the original stopped-world cause or rejects a nonfinite physical state.
    pub fn snapshot(&self) -> Result<RobotSnapshot, RobotWorldError> {
        if let State::Stopped(cause) = self.state {
            return Err(RobotWorldError::Stopped(cause));
        }
        let drones = self
            .drones
            .each_ref()
            .map(|drone| drone.observation(&self.world, [DroneMotorState::Working; 4]));
        let droids = self
            .droids
            .each_ref()
            .map(|droid| droid.observation(&self.world));
        let snapshot = RobotSnapshot {
            frame: self.frame.clone(),
            drones,
            droids,
        };
        // Every pose and velocity must be finite before it can reach a policy.
        for slot in RobotSlot::ALL {
            let drone = snapshot.drone(slot);
            if !finite(
                drone.position(),
                drone.orientation(),
                drone.linear_velocity(),
                drone.angular_velocity(),
            ) {
                return Err(RobotWorldError::Rejected(RobotWorldFailure::NonFinite(
                    RobotId::Drone(slot),
                )));
            }
            for body in crate::robots::DroidBody::ALL {
                let segment = snapshot.droid(slot).body(body);
                if !finite(
                    segment.position(),
                    segment.orientation(),
                    segment.linear_velocity(),
                    segment.angular_velocity(),
                ) {
                    return Err(RobotWorldError::Rejected(RobotWorldFailure::NonFinite(
                        RobotId::Droid(slot),
                    )));
                }
            }
        }
        Ok(snapshot)
    }

    /// Advance four common 5 ms substeps after validating the complete source frame.
    ///
    /// # Errors
    /// Rejects a wrong source frame, exhausted clock or nonfinite physical state.
    /// After rejection, returns the original stopped-world cause until reset.
    ///
    /// # Panics
    /// Panics only if private model arrays disagree with the closed robot slots.
    /// Construction preserves their matching lengths.
    pub fn advance(&mut self, actions: RobotActions) -> Result<RobotSnapshot, RobotWorldError> {
        if let State::Stopped(cause) = self.state {
            return Err(RobotWorldError::Stopped(cause));
        }
        // Reject ownership and clock failures before changing any body or actuator.
        let next = if actions.frame == self.frame {
            self.frame.next()
        } else {
            Err(RobotWorldFailure::WrongFrame)
        };
        let next = match next {
            Ok(next) => next,
            Err(cause) => return Err(self.stop(cause)),
        };
        self.validate()?;
        for _ in 0..4 {
            // All six requests precede one common 5 ms integration and collision step.
            for slot in RobotSlot::ALL {
                let index = slot.index();
                self.drones.get(index).expect("closed robot slot").apply(
                    &mut self.world,
                    *actions.drones.get(index).expect("complete robot actions"),
                    [DroneMotorState::Working; 4],
                );
                self.droids.get(index).expect("closed robot slot").apply(
                    &mut self.world,
                    *actions.droids.get(index).expect("complete robot actions"),
                );
            }
            self.world.step();
            self.validate()?;
        }
        // Publish time only after all four substeps have completed successfully.
        self.frame = next;
        self.snapshot()
    }

    /// Rebuild every physical body and constraint, creating a fresh frame scope.
    ///
    /// # Errors
    /// Rejects a nonfinite physical state after rebuilding the world.
    pub fn reset(&mut self) -> Result<RobotSnapshot, RobotWorldError> {
        *self = Self::default();
        self.snapshot()
    }

    /// Revoke advancement if a physical boundary cannot be safely observed.
    fn validate(&mut self) -> Result<(), RobotWorldError> {
        match self.snapshot() {
            Ok(_) => Ok(()),
            Err(RobotWorldError::Rejected(cause) | RobotWorldError::Stopped(cause)) => {
                Err(self.stop(cause))
            }
        }
    }

    /// Preserve the first failure and revoke further advancement until reset.
    const fn stop(&mut self, cause: RobotWorldFailure) -> RobotWorldError {
        self.state = State::Stopped(cause);
        RobotWorldError::Rejected(cause)
    }
}

/// World-space lane centre in metres for each closed team slot.
const fn lane(slot: RobotSlot) -> f32 {
    match slot {
        RobotSlot::First => -2.0,
        RobotSlot::Second => 0.0,
        RobotSlot::Third => 2.0,
    }
}

/// Check all four physical fields before exposing a pose to a learned policy.
fn finite(
    position: Vec3,
    orientation: Quat,
    linear_velocity: Vec3,
    angular_velocity: Vec3,
) -> bool {
    [
        position.is_finite(),
        orientation.is_finite(),
        linear_velocity.is_finite(),
        angular_velocity.is_finite(),
    ]
    .into_iter()
    .all(core::convert::identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robots::{DroidAction, DroneAction};

    /// Complete fixture requests probe mechanics without acting as a game controller.
    fn requests(snapshot: &RobotSnapshot) -> RobotActions {
        RobotActions::new(
            snapshot,
            [DroneAction::try_from([0.5; 4]).unwrap(); 3],
            [DroidAction::try_from([0.0; 26]).unwrap(); 3],
        )
    }

    /// One solver owns all 42 dynamic bodies, the floor and 36 passive joints.
    #[test]
    fn solver_contains_all_agents_and_original_total_mass() {
        let world = RobotWorld::default();
        assert_eq!(world.world.bodies.len(), 43);
        assert_eq!(world.world.colliders.len(), 43);
        assert_eq!(world.world.impulse_joints.len(), 36);
        let dynamic = world
            .world
            .bodies
            .iter()
            .filter(|(_, body)| body.is_dynamic());
        let masses: Vec<_> = dynamic.map(|(_, body)| body.mass()).collect();
        assert_eq!(masses.len(), 42);
        assert!((masses.iter().sum::<f32>() - 213.0).abs() < 0.001);
    }

    /// Different agents collide in the same narrow phase instead of separate worlds.
    #[test]
    fn cross_agent_contact_is_resolved_by_common_solver() {
        let mut world = RobotWorld::default();
        let first = world.drones[0].body;
        let second = world.drones[1].body;
        let position = world.world.bodies[first].translation();
        // Test-only placement creates an allowed collision; production exposes no teleport API.
        world.world.bodies[second].set_translation(position + Vector::X * 0.4, true);
        let source = world.snapshot().unwrap();
        world.advance(requests(&source)).unwrap();
        let pair = world
            .world
            .contact_pair(world.drones[0].collider, world.drones[1].collider)
            .unwrap();
        assert!(pair.has_any_active_contact());
    }

    /// Nonfinite states cannot reach any actor, including a later slot or droid segment.
    #[test]
    fn nonfinite_physical_states_are_identified_before_policy_reads() {
        for slot in RobotSlot::ALL {
            let mut world = RobotWorld::default();
            let handle = world.drones[slot.index()].body;
            world.world.bodies[handle].set_linvel(Vector::new(f32::NAN, 0.0, 0.0), true);
            assert_eq!(
                world.snapshot().unwrap_err(),
                RobotWorldError::Rejected(RobotWorldFailure::NonFinite(RobotId::Drone(slot)))
            );
        }
        let mut world = RobotWorld::default();
        // Private solver insertion order places all droid segments after floor and drones.
        let handle = world
            .world
            .bodies
            .iter()
            .filter(|(_, body)| body.is_dynamic())
            .last()
            .unwrap()
            .0;
        world.world.bodies[handle].set_angvel(Vector::new(f32::NAN, 0.0, 0.0), true);
        assert_eq!(
            world.snapshot().unwrap_err(),
            RobotWorldError::Rejected(RobotWorldFailure::NonFinite(RobotId::Droid(
                RobotSlot::Third
            )))
        );
    }

    /// Source rejection retains exact physics and the last successfully published time.
    #[test]
    fn wrong_frame_leaves_every_body_unchanged() {
        let mut world = RobotWorld::default();
        let before = world.snapshot().unwrap();
        let other = RobotWorld::default().snapshot().unwrap();
        world.advance(requests(&other)).unwrap_err();
        assert_eq!(world.frame, before.frame);
        for slot in RobotSlot::ALL {
            assert_eq!(
                world.drones[slot.index()].observation(&world.world, [DroneMotorState::Working; 4]),
                before.drone(slot)
            );
            assert_eq!(
                world.droids[slot.index()].observation(&world.world),
                before.droid(slot)
            );
        }
    }
    /// Each nonfinite component independently fails the common physical boundary.
    #[test]
    fn finite_boundary_checks_every_field() {
        let nan = Vec3::splat(f32::NAN);
        assert!(finite(Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO));
        assert!(!finite(nan, Quat::IDENTITY, Vec3::ZERO, Vec3::ZERO));
        assert!(!finite(
            Vec3::ZERO,
            Quat::from_array([f32::NAN; 4]),
            Vec3::ZERO,
            Vec3::ZERO
        ));
        assert!(!finite(Vec3::ZERO, Quat::IDENTITY, nan, Vec3::ZERO));
        assert!(!finite(Vec3::ZERO, Quat::IDENTITY, Vec3::ZERO, nan));
    }

    /// An invalid physical boundary stops before any actuator command is applied.
    #[test]
    fn invalid_boundary_stops_without_integrating() {
        let mut world = RobotWorld::default();
        let source = world.snapshot().unwrap();
        let handle = world.drones[0].body;
        world.world.bodies[handle].set_linvel(Vector::new(f32::NAN, 0.0, 0.0), true);
        let cause = RobotWorldFailure::NonFinite(RobotId::Drone(RobotSlot::First));
        assert_eq!(
            world.advance(requests(&source)).unwrap_err(),
            RobotWorldError::Rejected(cause)
        );
        assert_eq!(world.frame, source.frame);
        assert_eq!(
            world.snapshot().unwrap_err(),
            RobotWorldError::Stopped(cause)
        );
    }
}
