//! Rapier 0.36 articulated construction and direct torque application.
//!
//! Forces persist until reset: each substep clears user torques, then applies
//! equal and opposite torques. No position/velocity motor chooses joint motion.
//! <https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/>.

use super::{geometry, DroidAction, DroidBody, DroidBodyState, DroidObservation};
use bevy::math::{DVec3, Quat, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, ContactPair, GenericJointBuilder, IntegrationParameters,
    JointAxesMask, JointAxis, PhysicsWorld, RigidBody, RigidBodyBuilder, RigidBodyHandle, Rotation,
    Vector,
};
use std::time::Duration;

/// One deterministic solver substep.
pub(super) const PHYSICS_INTERVAL: Duration = Duration::from_millis(5);

/// Handles for one complete articulated body inside its owner's solver.
pub(in crate::robots) struct DroidRig {
    /// Physical bodies in observation order; never exposed outside robot mechanics.
    bodies: [RigidBodyHandle; 13],
    /// Matching segment colliders in observation order.
    colliders: [ColliderHandle; 13],
    /// Shared floor identity, distinct from self-contact and other agents.
    floor: ColliderHandle,
}

impl DroidRig {
    /// Identify a collision handle without exposing mutable solver access.
    pub(in crate::robots) fn body_for_collider(&self, handle: ColliderHandle) -> Option<DroidBody> {
        DroidBody::ALL
            .into_iter()
            .zip(self.colliders)
            .find_map(|(body, collider)| (handle == collider).then_some(body))
    }

    /// Insert the unchanged body and passive joint model into one existing solver.
    pub(in crate::robots) fn insert(
        world: &mut PhysicsWorld,
        floor: ColliderHandle,
        rotation: Rotation,
        offset: Vector,
    ) -> Self {
        let segments = DroidBody::ALL.map(|body| {
            let profile = geometry::segment(body);
            let half = profile.half_extents;
            world.insert(
                RigidBodyBuilder::dynamic()
                    .translation(rotation * profile.centre + offset)
                    .rotation(rotation.to_scaled_axis())
                    .can_sleep(false)
                    .ccd_enabled(true),
                ColliderBuilder::cuboid(half.x, half.y, half.z)
                    .mass(profile.mass)
                    .friction(0.8),
            )
        });
        let bodies = segments.map(|(body, _)| body);
        let colliders = segments.map(|(_, collider)| collider);
        Self::insert_joints(world, &bodies);
        Self {
            bodies,
            colliders,
            floor,
        }
    }

    /// Join the existing segments with passive constraints and coincident bind anchors.
    fn insert_joints(world: &mut PhysicsWorld, bodies: &[RigidBodyHandle; 13]) {
        // Coincident anchors retain the bind pose under every common reset transform.
        for joint in geometry::JOINTS {
            let mut locked = JointAxesMask::LOCKED_SPHERICAL_AXES;
            let mut builder = GenericJointBuilder::new(locked)
                .local_anchor1(joint.anchor - geometry::segment(joint.parent).centre)
                .local_anchor2(joint.anchor - geometry::segment(joint.child).centre)
                .contacts_enabled(false);
            for ((axis, mask), actuator) in [
                (JointAxis::AngX, JointAxesMask::ANG_X),
                (JointAxis::AngY, JointAxesMask::ANG_Y),
                (JointAxis::AngZ, JointAxesMask::ANG_Z),
            ]
            .into_iter()
            .zip(joint.axes)
            {
                if let Some(actuator) = actuator {
                    builder = builder.limits(axis, actuator.limits);
                } else {
                    locked |= mask;
                }
            }
            world.insert_impulse_joint(
                *bodies
                    .get(joint.parent as usize)
                    .expect("closed parent index"),
                *bodies
                    .get(joint.child as usize)
                    .expect("closed child index"),
                builder.locked_axes(locked),
            );
        }
    }

    /// Apply policy-selected actuator torques, with no target posture or damping controller.
    pub(in crate::robots) fn apply(&self, world: &mut PhysicsWorld, action: DroidAction) {
        // Rapier accumulates persistent user torque; replace the prior substep's commands.
        for handle in self.bodies {
            world
                .bodies
                .get_mut(handle)
                .expect("private segment")
                .reset_torques(true);
        }
        for joint in geometry::JOINTS {
            let parent = *self
                .bodies
                .get(joint.parent as usize)
                .expect("closed parent index");
            let child = *self
                .bodies
                .get(joint.child as usize)
                .expect("closed child index");
            let orientation = *world.bodies.get(parent).expect("private parent").rotation();
            for (direction, axis) in [Vector::X, Vector::Y, Vector::Z]
                .into_iter()
                .zip(joint.axes)
            {
                if let Some(axis) = axis {
                    // Dimensionless fraction * Nm limit * world unit axis yields world Nm.
                    let torque = orientation
                        * direction
                        * (action.fraction(axis.actuator) * axis.torque_limit);
                    world
                        .bodies
                        .get_mut(child)
                        .expect("private child")
                        .add_torque(torque, true);
                    world
                        .bodies
                        .get_mut(parent)
                        .expect("private parent")
                        .add_torque(-torque, true);
                }
            }
        }
    }

    /// Copy the authoritative physics state without exposing mutable bodies.
    pub(in crate::robots) fn observation(&self, world: &PhysicsWorld) -> DroidObservation {
        DroidObservation {
            bodies: DroidBody::ALL.map(|identity| {
                let handle = *self
                    .bodies
                    .get(identity as usize)
                    .expect("closed body index");
                let collider = *self
                    .colliders
                    .get(identity as usize)
                    .expect("closed collider index");
                let body = world.bodies.get(handle).expect("private segment");
                let floor_contact = world
                    .contact_pair(collider, self.floor)
                    .is_some_and(ContactPair::has_any_active_contact);
                DroidBodyState {
                    position: Vec3::from_array(body.translation().to_array()),
                    orientation: Quat::from_array(body.rotation().to_array()),
                    linear_velocity: Vec3::from_array(body.linvel().to_array()),
                    angular_velocity: Vec3::from_array(body.angvel().to_array()),
                    floor_contact,
                }
            }),
        }
    }
}

/// Compute horizontal contact-point speed in m/s with widened rigid-body arithmetic.
///
/// Rapier 0.36 point velocity is linear velocity plus angular velocity crossed with
/// the offset from world centre of mass. Widen before subtraction and multiplication so finite
/// f32 state cannot overflow in intermediate arithmetic. The floor is stationary.
fn contact_horizontal_speed(body: &RigidBody, point: Vector) -> f64 {
    let point_m = DVec3::from_array(point.to_array().map(f64::from));
    let centre_m = DVec3::from_array(body.center_of_mass().to_array().map(f64::from));
    let linear_m_s = DVec3::from_array(body.linvel().to_array().map(f64::from));
    let angular_rad_s = DVec3::from_array(body.angvel().to_array().map(f64::from));
    let velocity_m_s = linear_m_s + angular_rad_s.cross(point_m - centre_m);
    velocity_m_s.x.hypot(velocity_m_s.z)
}

/// Standalone lesson owner; its rig uses the same component as the shared robot world.
pub(super) struct Articulation {
    /// The lesson's sole integration and collision owner.
    pub world: PhysicsWorld,
    /// Complete body handles, tied to the owned solver throughout the episode.
    rig: DroidRig,
}

/// Fixed floor profiles; the articulated body remains identical in both tasks.
#[derive(Debug, Clone, Copy)]
pub(super) enum Floor {
    /// Preserve the original standalone floor with 10 m half-width.
    Standing,
    /// Use the shared world's 30 m half-width for commanded walking.
    Walking,
}

impl Articulation {
    /// Rebuild the original standalone floor and body without changing insertion order.
    pub(super) fn new(rotation: Rotation, offset: Vector) -> Self {
        Self::new_for_floor(rotation, offset, Floor::Standing)
    }

    /// Choose only the floor extent; retain body geometry and insertion order.
    pub(super) fn new_for_floor(rotation: Rotation, offset: Vector, floor: Floor) -> Self {
        let mut world = PhysicsWorld {
            gravity: Vector::NEG_Y * 9.81,
            integration_parameters: IntegrationParameters {
                dt: PHYSICS_INTERVAL.as_secs_f32(),
                ..Default::default()
            },
            ..Default::default()
        };
        let half_width_m = match floor {
            Floor::Standing => 10.0,
            Floor::Walking => 30.0,
        };
        let (_, floor) = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(half_width_m, 0.1, half_width_m).friction(0.8),
        );
        let rig = DroidRig::insert(&mut world, floor, rotation, offset);
        Self { world, rig }
    }

    /// Forward only this body's validated actuator requests to its own solver.
    pub(super) fn apply(&mut self, action: DroidAction) {
        self.rig.apply(&mut self.world, action);
    }

    /// Measure each foot's maximum horizontal solver-contact speed in m/s, left then right.
    ///
    /// Rapier 0.36 resolves CoM-local anchors through current body poses:
    /// <https://docs.rs/rapier3d/0.36.0/rapier3d/geometry/struct.ContactManifoldData.html#method.solver_contact_world_points>.
    /// Solver contacts include predictive constraints; this is not a loaded-contact classification.
    pub(super) fn foot_contact_speeds(&self) -> [f64; 2] {
        [DroidBody::LeftFoot, DroidBody::RightFoot].map(|foot| {
            let collider = *self
                .rig
                .colliders
                .get(foot as usize)
                .expect("closed foot index");
            let handle = *self
                .rig
                .bodies
                .get(foot as usize)
                .expect("closed foot index");
            let body = self.world.bodies.get(handle).expect("private foot");
            let Some(pair) = self.world.contact_pair(collider, self.rig.floor) else {
                return 0.0;
            };
            // Bound memory while examining every actual solver contact once.
            let mut maximum_m_s = 0.0_f64;
            for manifold in pair.solver_manifolds() {
                for contact in &manifold.data.solver_contacts {
                    let (first, second) = manifold
                        .data
                        .solver_contact_world_points(contact, &self.world.bodies);
                    let point = if manifold.data.rigid_body1 == Some(handle) {
                        first
                    } else {
                        second
                    };
                    maximum_m_s = maximum_m_s.max(contact_horizontal_speed(body, point));
                }
            }
            maximum_m_s
        })
    }

    /// Read the unchanged physical observation profile.
    pub(super) fn observation(&self) -> DroidObservation {
        self.rig.observation(&self.world)
    }
}

#[cfg(test)]
mod tests {
    use super::super::DroidActuator;
    use super::*;

    /// Point velocity includes rotation, excludes vertical speed and widens finite extremes.
    #[test]
    fn horizontal_contact_speed_uses_the_contact_point() {
        let mut body = RigidBodyBuilder::dynamic().build();
        body.set_linvel(Vector::new(8.0, 7.0, 0.0), false);
        body.set_angvel(Vector::new(0.0, 2.0, 0.0), false);
        assert_eq!(
            contact_horizontal_speed(&body, Vector::new(3.0, 0.0, 0.0)).to_bits(),
            10.0_f64.to_bits()
        );
        body.set_angvel(Vector::new(0.0, -2.0, 0.0), false);
        assert_eq!(
            contact_horizontal_speed(&body, Vector::new(3.0, 0.0, 0.0)).to_bits(),
            10.0_f64.to_bits()
        );
        body.set_linvel(Vector::Y * 7.0, false);
        body.set_angvel(Vector::ZERO, false);
        assert_eq!(
            contact_horizontal_speed(&body, Vector::X).to_bits(),
            0.0_f64.to_bits()
        );
        body.set_linvel(Vector::X * f32::MAX, false);
        body.set_angvel(Vector::Y * f32::MAX, false);
        assert!(contact_horizontal_speed(&body, Vector::X * 3.0).is_finite());
    }

    /// A later-inserted floor resolves the foot's first surface anchor instead of the floor's.
    #[test]
    fn foot_contact_speed_handles_both_collider_orders() {
        let mut articulation = Articulation::new(Rotation::IDENTITY, Vector::Y * 0.03);
        articulation
            .world
            .colliders
            .get_mut(articulation.rig.floor)
            .expect("private floor")
            .set_translation(Vector::NEG_Y * 100.0);
        let (_, floor) = articulation.world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(10.0, 0.1, 10.0).friction(0.8),
        );
        articulation.rig.floor = floor;
        for _ in 0..20 {
            articulation.world.step();
        }
        for foot in [DroidBody::LeftFoot, DroidBody::RightFoot] {
            let handle = *articulation
                .rig
                .bodies
                .get(foot as usize)
                .expect("closed foot index");
            let body = articulation
                .world
                .bodies
                .get_mut(handle)
                .expect("private foot");
            body.set_linvel(Vector::X, false);
            body.set_angvel(Vector::ZERO, false);
        }
        assert_eq!(
            articulation.foot_contact_speeds().map(f64::to_bits),
            [1.0_f64.to_bits(); 2]
        );
    }

    /// All bodies have physical mass and inertia.
    #[test]
    fn dynamic_segments_have_mass_and_inertia() {
        let rotation = Rotation::from_rotation_y(0.7) * Rotation::from_rotation_z(0.01);
        let articulation = Articulation::new(rotation, Vector::Y * 0.03);
        let mut mass = 0.0;
        for handle in articulation.rig.bodies {
            let body = articulation.world.bodies.get(handle).expect("segment");
            assert!(body.is_dynamic());
            assert!(body.mass() > 0.0);
            assert!(
                body.mass_properties()
                    .local_mprops
                    .principal_inertia()
                    .min_element()
                    > 0.0
            );
            mass += body.mass();
        }
        assert!((mass - 70.0).abs() < 1e-4);
    }

    /// Joints retain coincident reset anchors and passive angular limits without motors.
    #[test]
    fn joints_have_anchors_limits_and_no_motors() {
        let rotation = Rotation::from_rotation_y(0.7) * Rotation::from_rotation_z(0.01);
        let articulation = Articulation::new(rotation, Vector::Y * 0.03);
        assert_eq!(articulation.world.impulse_joints.len(), 12);
        for (_, joint) in articulation.world.impulse_joints.iter() {
            let parent = articulation
                .world
                .bodies
                .get(joint.body1())
                .expect("parent");
            let child = articulation.world.bodies.get(joint.body2()).expect("child");
            let anchor1 = parent.position() * joint.data.local_anchor1();
            let anchor2 = child.position() * joint.data.local_anchor2();
            assert!(anchor1.distance(anchor2) < 1e-6);
            assert!(!joint.data.contacts_enabled);
            assert!(joint.data.motor_axes.is_empty());
            assert!(joint
                .data
                .locked_axes
                .contains(JointAxesMask::LOCKED_SPHERICAL_AXES));
            for axis in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ] {
                if let Some(limits) = joint.data.limits(axis) {
                    assert!(limits.min < 0.0 && limits.max > 0.0);
                } else {
                    assert!(joint.data.locked_axes.contains(axis.into()));
                }
            }
        }
    }

    /// Every requested axis acts on exactly two bodies; replacing the command clears torque.
    #[test]
    fn torque_pairs_conserve_total_moment_and_replace_previous_commands() {
        let mut articulation = Articulation::new(Rotation::from_rotation_y(0.7), Vector::Y * 2.0);
        for actuator in DroidActuator::ALL {
            let mut fractions = [0.0; 26];
            *fractions.get_mut(actuator as usize).expect("actuator") = 0.5;
            articulation.apply(DroidAction::try_from(fractions).expect("fixture"));
            let torques: Vec<_> = articulation
                .rig
                .bodies
                .iter()
                .map(|handle| {
                    articulation
                        .world
                        .bodies
                        .get(*handle)
                        .expect("segment")
                        .user_torque()
                })
                .collect();
            assert_eq!(
                torques
                    .iter()
                    .filter(|torque| torque.length_squared() > 0.0)
                    .count(),
                2
            );
            assert!(torques.iter().copied().sum::<Vector>().length() < 1e-6);
            assert!(torques.iter().all(|torque| torque.length() <= 60.001));
        }
        articulation.apply(DroidAction::try_from([0.0; 26]).expect("fixture"));
        for handle in articulation.rig.bodies {
            assert_eq!(
                articulation
                    .world
                    .bodies
                    .get(handle)
                    .expect("segment")
                    .user_torque(),
                Vector::ZERO
            );
        }
    }

    /// A left knee command rotates its child about parent-local X with a 100 Nm bound.
    #[test]
    fn knee_torque_rotates_with_parent_frame() {
        let rotation = Rotation::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let mut articulation = Articulation::new(rotation, Vector::Y * 2.0);
        let mut fractions = [0.0; 26];
        *fractions
            .get_mut(DroidActuator::LeftKneeX as usize)
            .expect("knee") = 1.0;
        articulation.apply(DroidAction::try_from(fractions).expect("fixture"));
        for (identity, expected) in [
            (DroidBody::LeftCalf, Vector::NEG_Z * 100.0),
            (DroidBody::LeftThigh, Vector::Z * 100.0),
        ] {
            let handle = *articulation
                .rig
                .bodies
                .get(identity as usize)
                .expect("identity");
            let actual = articulation
                .world
                .bodies
                .get(handle)
                .expect("segment")
                .user_torque();
            assert!(actual.distance(expected) < 1e-4, "{actual:?}");
        }
    }
}
