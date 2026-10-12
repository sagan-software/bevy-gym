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
    /// Read fatal physical contact without advancing physics or choosing an action.
    /// Drone contact with any solid is fatal. A droid falls when its pelvis is below
    /// 0.5 m or a non-foot segment contacts the floor. Foot support alone is permitted.
    /// Competition movement does not inherit standalone lesson boundary limits.
    ///
    /// # Errors
    /// Returns the original stopped-world error or rejects nonfinite physical state.
    pub fn fatal_contact(&self, owner: RobotId) -> Result<bool, RobotWorldError> {
        let snapshot = self.snapshot()?;
        match owner {
            RobotId::Drone(slot) => {
                let [first, second, third] = &self.drones;
                let drone = match slot {
                    RobotSlot::First => first,
                    RobotSlot::Second => second,
                    RobotSlot::Third => third,
                };
                Ok(self
                    .world
                    .contact_pairs_with(drone.collider)
                    .any(rapier3d::prelude::ContactPair::has_any_active_contact))
            }
            RobotId::Droid(slot) => Ok(snapshot.droid(slot).fatal_contact()),
        }
    }

    /// Place an isolated collision fixture; this mutation is absent from production builds.
    #[cfg(test)]
    pub(super) fn place_drone_fixture(&mut self, slot: RobotSlot, position: Vec3) {
        let body = self
            .drones
            .get(slot.index())
            .expect("closed fixture slot")
            .body;
        self.world
            .bodies
            .get_mut(body)
            .expect("fixture drone body")
            .set_translation(Vector::from_array(position.to_array()), true);
        // Direct ray queries read collider poses before the next solver propagation.
        let collider = self
            .drones
            .get(slot.index())
            .expect("closed fixture slot")
            .collider;
        self.world
            .colliders
            .get_mut(collider)
            .expect("fixture drone collider")
            .set_translation(Vector::from_array(position.to_array()));
    }

    /// Trace current physical muzzle geometry, excluding all of its owner's colliders.
    /// Invalid queries leave the world unchanged. Physical validity precedes frame validation.
    ///
    /// # Errors
    /// Returns the original world error or rejects a stale, reset, or foreign muzzle frame.
    pub fn cast_muzzle(
        &self,
        muzzle: &super::RobotMuzzle,
    ) -> Result<Option<super::RobotRayHit>, RobotWorldError> {
        // Validate the current world before comparing opaque source boundaries.
        let snapshot = self.snapshot()?;
        if snapshot.frame() != muzzle.frame() {
            return Err(RobotWorldError::Rejected(RobotWorldFailure::WrongFrame));
        }
        self.cast_ray(&muzzle.ray, Some(muzzle.owner()))
    }

    /// Read the nearest actual solid without advancing the world or choosing an action.
    /// The excluded identity removes all of that robot's colliders, including every droid segment.
    /// Equal distances use environment, drone slot, then droid slot/body order.
    /// Direct shape reads include reset poses before Rapier 0.36 updates its query BVH:
    /// <https://rapier.rs/docs/user_guides/rust/scene_queries/>.
    /// Scanning N solids with at most 42 robot-owner comparisons costs O(N) time
    /// for the fixed six-agent roster and O(1) auxiliary space.
    ///
    /// # Errors
    /// Returns the original stopped-world error or rejects nonfinite physical state.
    pub fn cast_ray(
        &self,
        ray: &super::RobotRay,
        excluded: Option<RobotId>,
    ) -> Result<Option<super::RobotRayHit>, RobotWorldError> {
        // Preserve stopped-world and finite-state checks before reading collision geometry.
        self.snapshot()?;
        let physical_ray = rapier3d::prelude::Ray::new(
            Vector::from_array(ray.origin.to_array()),
            Vector::from_array(ray.direction.to_array()),
        );
        let mut nearest: Option<super::RobotRayHit> = None;
        // Direct shape reads include freshly inserted poses before Rapier's first BVH update.
        // With 43 fixed colliders, classification and scanning retain O(1) auxiliary space.
        for (handle, collider) in self.world.colliders.iter() {
            if collider.is_sensor() {
                continue;
            }
            let target = self.ray_target(handle);
            if excluded.is_some() && target.owner() == excluded {
                continue;
            }
            if let Some(metres) =
                collider
                    .shape()
                    .cast_ray(collider.position(), &physical_ray, ray.metres, true)
            {
                // Canonicalize signed zero before applying the total distance ordering.
                let metres = metres.abs();
                let hit = super::RobotRayHit { target, metres };
                // Rank breaks exact-distance ties; it never changes which closer solid wins.
                if nearest.is_none_or(|previous| {
                    metres
                        .total_cmp(&previous.metres)
                        .then_with(|| target.rank().cmp(&previous.target.rank()))
                        .is_lt()
                }) {
                    nearest = Some(hit);
                }
            }
        }
        Ok(nearest)
    }

    /// Map actual collision ownership into a read-only closed target identity.
    fn ray_target(&self, handle: rapier3d::prelude::ColliderHandle) -> super::RobotRayTarget {
        for ((slot, drone), droid) in RobotSlot::ALL
            .into_iter()
            .zip(&self.drones)
            .zip(&self.droids)
        {
            if drone.collider == handle {
                return super::RobotRayTarget::Drone(slot);
            }
            if let Some(body) = droid.body_for_collider(handle) {
                return super::RobotRayTarget::Droid(slot, body);
            }
        }
        super::RobotRayTarget::Environment
    }

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
        self.advance_frame(actions, super::actuation::Actuation::Unrestricted)
    }

    /// Validate a complete source boundary before combat can consume timing or ammunition.
    pub(super) fn check_actions(
        &mut self,
        actions: &RobotActions,
    ) -> Result<RobotFrame, RobotWorldError> {
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
        Ok(next)
    }

    /// Apply current damage constraints without choosing or replacing living policy actions.
    pub(super) fn advance_with_health(
        &mut self,
        actions: RobotActions,
        drones: &mut [crate::robots::RobotHealth; 3],
        droids: &mut [crate::robots::RobotHealth; 3],
    ) -> Result<RobotSnapshot, RobotWorldError> {
        self.advance_frame(
            actions,
            super::actuation::Actuation::Combat { drones, droids },
        )
    }

    /// Integrate one common frame under its explicit actuator and contact mechanics.
    fn advance_frame(
        &mut self,
        actions: RobotActions,
        mut actuation: super::actuation::Actuation<'_>,
    ) -> Result<RobotSnapshot, RobotWorldError> {
        let next = self.check_actions(&actions)?;
        for _ in 0..4 {
            // All six requests precede one common 5 ms integration and collision step.
            for slot in RobotSlot::ALL {
                let index = slot.index();
                self.drones.get(index).expect("closed robot slot").apply(
                    &mut self.world,
                    *actions.drones.get(index).expect("complete robot actions"),
                    actuation.motors(slot),
                );
                self.droids.get(index).expect("closed robot slot").apply(
                    &mut self.world,
                    actuation.droid_action(
                        slot,
                        *actions.droids.get(index).expect("complete robot actions"),
                    ),
                );
            }
            self.world.step();
            self.validate()?;
            // Revoke force immediately after a fatal substep, before the next actuator application.
            actuation.contacts(self)?;
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
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
    use super::*;
    use crate::robots::{DroidAction, DroneAction};

    /// A fully occluded enemy can move without changing any actor image byte.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn camera_images_hide_enemy_motion_behind_solid_cover() {
        let mut world = RobotWorld::default();
        let owner = RobotId::Drone(RobotSlot::Third);
        world.world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(-3.0, 2.0, 1.0)),
            ColliderBuilder::cuboid(5.0, 5.0, 0.1),
        );
        let before = super::super::RobotImage::capture(&world, owner).unwrap();
        assert!(before
            .pixels()
            .iter()
            .flatten()
            .all(|pixel| *pixel == [96, 104, 112]));
        let original = world.snapshot().unwrap().droid(RobotSlot::First);
        let [rig, _, _] = &world.droids;
        let segments = world
            .world
            .colliders
            .iter()
            .filter_map(|(handle, collider)| {
                rig.body_for_collider(handle)
                    .map(|_| (handle, collider.parent().unwrap()))
            })
            .collect::<Vec<_>>();
        assert_eq!(segments.len(), 13);
        for (collider, body) in segments {
            let offset = Vector::new(-2.0, 0.0, -1.0);
            let position = world.world.bodies[body].translation();
            world.world.bodies[body].set_translation(position + offset, true);
            let position = world.world.colliders[collider].translation();
            world.world.colliders[collider].set_translation(position + offset);
        }
        let changed = world.snapshot().unwrap().droid(RobotSlot::First);
        assert_ne!(original, changed);
        let after = super::super::RobotImage::capture(&world, owner).unwrap();
        assert_eq!(before, after);
    }

    /// The camera follows body rotation rather than retaining a spectator or target direction.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn camera_images_follow_the_physical_body_rotation() {
        let mut world = RobotWorld::default();
        let owner = RobotId::Drone(RobotSlot::Third);
        let before = super::super::RobotImage::capture(&world, owner).unwrap();
        assert_eq!(before.pixels()[6][8], [32, 176, 208]);
        let [_, _, drone] = &world.drones;
        let rotation = Rotation::from_rotation_y(std::f32::consts::FRAC_PI_2);
        world.world.bodies[drone.body].set_rotation(rotation, true);
        world.world.colliders[drone.collider].set_rotation(rotation);
        let after = super::super::RobotImage::capture(&world, owner).unwrap();
        assert_eq!(after.pixels()[6][8], [16, 24, 40]);
        assert_eq!(before.frame(), after.frame());
    }

    /// The fixed 20 m image range cannot expose a farther collider or hidden identity.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn camera_images_limit_visibility_to_the_fixed_range() {
        let mut world = RobotWorld::default();
        let owner = RobotId::Drone(RobotSlot::Third);
        world.place_drone_fixture(RobotSlot::First, Vec3::new(20.0, 2.0, 20.0));
        let origin = Vec3::new(-3.0, 2.0, 2.0 - 0.317);
        world.place_drone_fixture(RobotSlot::Second, origin + Vec3::NEG_Z * 19.0);
        assert_eq!(
            super::super::RobotImage::capture(&world, owner)
                .unwrap()
                .pixels()[6][8],
            [32, 176, 208]
        );
        world.place_drone_fixture(RobotSlot::Second, origin + Vec3::NEG_Z * 21.0);
        assert_eq!(
            super::super::RobotImage::capture(&world, owner)
                .unwrap()
                .pixels()[6][8],
            [16, 24, 40]
        );
    }

    /// Invalid physical state fails before an image can conceal it with invented pixels.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn camera_images_reject_nonfinite_physical_state() {
        let mut world = RobotWorld::default();
        let owner = RobotId::Drone(RobotSlot::Third);
        world.place_drone_fixture(RobotSlot::First, Vec3::splat(f32::NAN));
        assert_eq!(
            super::super::RobotImage::capture(&world, owner),
            Err(RobotWorldError::Rejected(RobotWorldFailure::NonFinite(
                RobotId::Drone(RobotSlot::First)
            )))
        );
    }

    /// Complete fixture requests probe mechanics without acting as a game controller.
    fn requests(snapshot: &RobotSnapshot) -> RobotActions {
        RobotActions::new(
            snapshot,
            [DroneAction::try_from([0.5; 4]).unwrap(); 3],
            [DroidAction::try_from([0.0; 26]).unwrap(); 3],
        )
    }

    /// Replay one isolated solver substep with explicit third-drone motor availability.
    fn fixture_substep(world: &mut RobotWorld, third: [DroneMotorState; 4]) {
        for ((slot, drone), droid) in RobotSlot::ALL
            .into_iter()
            .zip(&world.drones)
            .zip(&world.droids)
        {
            let motors = if slot == RobotSlot::Third {
                third
            } else {
                [DroneMotorState::Working; 4]
            };
            drone.apply(
                &mut world.world,
                DroneAction::try_from([1.0; 4]).unwrap(),
                motors,
            );
            droid.apply(&mut world.world, DroidAction::default());
        }
        world.world.step();
    }

    /// The first fatal 5 ms contact suppresses thrust during the remaining three substeps.
    #[test]
    fn fatal_substep_removes_force_before_the_next_substep() {
        let mut actual = RobotWorld::default();
        let mut reference = RobotWorld::default();
        for world in [&mut actual, &mut reference] {
            world.place_drone_fixture(RobotSlot::Third, Vec3::new(-3.0, 0.1, 2.0));
        }
        fixture_substep(&mut reference, [DroneMotorState::Working; 4]);
        assert!(reference
            .fatal_contact(RobotId::Drone(RobotSlot::Third))
            .unwrap());
        for _ in 0..3 {
            fixture_substep(&mut reference, [DroneMotorState::Failed; 4]);
        }
        let snapshot = actual.snapshot().unwrap();
        let actions = RobotActions::new(
            &snapshot,
            [DroneAction::try_from([1.0; 4]).unwrap(); 3],
            [DroidAction::default(); 3],
        );
        let mut drones = [crate::robots::RobotHealth::from(super::super::RobotTeam::Drones); 3];
        let mut droids = [crate::robots::RobotHealth::from(super::super::RobotTeam::Droids); 3];
        let after = actual
            .advance_with_health(actions, &mut drones, &mut droids)
            .unwrap();
        let [_, _, health] = drones;
        assert!(!health.is_alive());
        assert_eq!(
            after.drone(RobotSlot::Third),
            reference.snapshot().unwrap().drone(RobotSlot::Third)
        );
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

#[cfg(test)]
mod ray_tests {
    use super::super::{RobotRay, RobotRayTarget};
    use super::*;
    /// Sensors never occlude and equal-distance solids have stable environment priority.
    #[test]
    fn sensor_filter_and_environment_tie_do_not_depend_on_insertion_order() {
        let mut world = RobotWorld::default();
        let ray = RobotRay::try_from((Vec3::new(0.0, 2.0, 0.0), Vec3::Y, 10.0)).unwrap();
        world.world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, 3.0, 0.0)),
            ColliderBuilder::ball(0.5).sensor(true),
        );
        assert!(world.cast_ray(&ray, None).unwrap().is_none());
        let origin = Vec3::new(-3.0, 2.0, -2.0);
        let ray = RobotRay::try_from((origin, Vec3::Y, 0.1)).unwrap();
        let shape = world.world.colliders[world.drones[0].collider]
            .shared_shape()
            .clone();
        world.world.insert(
            RigidBodyBuilder::fixed().translation(Vector::from_array(origin.to_array())),
            ColliderBuilder::new(shape),
        );
        assert_eq!(
            world.cast_ray(&ray, None).unwrap().unwrap().target(),
            RobotRayTarget::Environment
        );
        assert_eq!(
            world
                .cast_ray(&ray, Some(RobotId::Drone(RobotSlot::First)))
                .unwrap()
                .unwrap()
                .target(),
            RobotRayTarget::Environment
        );
    }
    /// Overlapping robot fixtures retain the first slot and exclusion reveals the next slot.
    #[test]
    fn robot_ties_and_nonfinite_state_use_the_public_query_boundary() {
        let mut world = RobotWorld::default();
        let origin = Vec3::new(-3.0, 2.0, -2.0);
        let second = world.drones[1].collider;
        world.world.colliders[second].set_translation(Vector::from_array(origin.to_array()));
        let ray = RobotRay::try_from((origin, Vec3::Y, 0.001)).unwrap();
        assert_eq!(
            world.cast_ray(&ray, None).unwrap().unwrap().target(),
            RobotRayTarget::Drone(RobotSlot::First)
        );
        assert_eq!(
            world
                .cast_ray(&ray, Some(RobotId::Drone(RobotSlot::First)))
                .unwrap()
                .unwrap()
                .target(),
            RobotRayTarget::Drone(RobotSlot::Second)
        );
        let body = world.drones[0].body;
        world.world.bodies[body].set_linvel(Vector::new(f32::NAN, 0.0, 0.0), true);
        assert_eq!(
            world.cast_ray(&ray, None).unwrap_err(),
            RobotWorldError::Rejected(RobotWorldFailure::NonFinite(RobotId::Drone(
                RobotSlot::First
            )))
        );
    }

    /// Exact-distance ties retain the lower robot slot or body rank.
    #[test]
    fn equal_distance_priority_is_derived_from_target_identity() {
        assert!(
            RobotRayTarget::Environment.rank() < RobotRayTarget::Drone(RobotSlot::First).rank()
        );
        let mut previous = 0;
        for slot in RobotSlot::ALL {
            let rank = RobotRayTarget::Drone(slot).rank();
            assert!(rank > previous);
            previous = rank;
        }
        for slot in RobotSlot::ALL {
            for body in crate::robots::DroidBody::ALL {
                let target = RobotRayTarget::Droid(slot, body);
                assert!(target.rank() > previous);
                previous = target.rank();
                assert_eq!(target.owner(), Some(RobotId::Droid(slot)));
            }
        }
    }
}
