//! Rapier 0.36 articulated construction and direct torque application.
//!
//! Forces persist until reset: each substep clears user torques, then applies
//! equal and opposite torques. No position/velocity motor chooses joint motion.
//! <https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/>.

use super::{geometry, DroidAction, DroidBody, DroidBodyState, DroidObservation};
use bevy::math::{Quat, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, ContactPair, GenericJointBuilder, IntegrationParameters,
    JointAxesMask, JointAxis, PhysicsWorld, RigidBodyBuilder, RigidBodyHandle, Rotation, Vector,
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

/// Standalone lesson owner; its rig uses the same component as the shared robot world.
pub(super) struct Articulation {
    /// The lesson's sole integration and collision owner.
    pub world: PhysicsWorld,
    /// Complete body handles, tied to the owned solver throughout the episode.
    rig: DroidRig,
}

impl Articulation {
    /// Rebuild the original standalone floor and body without changing insertion order.
    pub(super) fn new(rotation: Rotation, offset: Vector) -> Self {
        let mut world = PhysicsWorld {
            gravity: Vector::NEG_Y * 9.81,
            integration_parameters: IntegrationParameters {
                dt: PHYSICS_INTERVAL.as_secs_f32(),
                ..Default::default()
            },
            ..Default::default()
        };
        let (_, floor) = world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(10.0, 0.1, 10.0).friction(0.8),
        );
        let rig = DroidRig::insert(&mut world, floor, rotation, offset);
        Self { world, rig }
    }

    /// Forward only this body's validated actuator requests to its own solver.
    pub(super) fn apply(&mut self, action: DroidAction) {
        self.rig.apply(&mut self.world, action);
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
