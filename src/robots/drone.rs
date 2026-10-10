//! Original drone collision, inertial and four-motor model inside an owned solver.
//! Rapier 0.36 forces persist; each command replaces the prior force and torque.
//! <https://rapier.rs/docs/user_guides/rust/rigid_body_forces_and_impulses/>.

use super::{DroneAction, DroneMotorState, DroneObservation};
use bevy::math::{Quat, Vec3};
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, PhysicsWorld, RigidBodyBuilder, RigidBodyHandle, Vector,
};

/// Private handles for one physical rotor body; callers never receive the mutable solver.
pub(super) struct DroneModel {
    /// This body's rigid handle, retained while the owner retains its solver.
    pub(super) body: RigidBodyHandle,
    /// Matching collision proxy, excluded from its own range observations.
    pub(super) collider: ColliderHandle,
}

/// Body mass in kilograms, independent of the artist's material choices.
const MASS_KG: f32 = 1.0;
/// Gravitational acceleration magnitude in metres per second squared.
pub(super) const GRAVITY: f32 = 9.81;
/// Each motor can produce half the body's weight in newtons.
const MAX_THRUST: f32 = MASS_KG * GRAVITY / 2.0;
/// Reaction moment divided by thrust, in metres.
const REACTION_ARM: f32 = 0.016;
/// Motor offsets in metres and alternating reaction signs, in action order.
///
/// These offsets follow the licensed model's rotor centres after its front faces -Z.
const MOTORS: [(Vector, f32); 4] = [
    (Vector::new(-0.2505, 0.0875, -0.2606), 1.0),
    (Vector::new(0.2505, 0.0875, -0.2606), -1.0),
    (Vector::new(0.2505, 0.0875, 0.2606), 1.0),
    (Vector::new(-0.2505, 0.0875, 0.2606), -1.0),
];

impl DroneModel {
    /// Insert the unchanged collision and inertial profile at a finite internal position.
    pub(super) fn insert(world: &mut PhysicsWorld, position: Vec3) -> Self {
        let (body, collider) = world.insert(
            RigidBodyBuilder::dynamic()
                .translation(Vector::from_array(position.to_array()))
                .can_sleep(false)
                .ccd_enabled(true),
            ColliderBuilder::cuboid(0.287, 0.104, 0.297).mass(MASS_KG),
        );
        Self { body, collider }
    }

    /// Read only this body's authoritative pose, velocity and supplied actuator health.
    pub(super) fn observation(
        &self,
        world: &PhysicsWorld,
        motor_states: [DroneMotorState; 4],
    ) -> DroneObservation {
        let body = world.bodies.get(self.body).expect("private body exists");
        DroneObservation {
            position: Vec3::from_array(body.translation().to_array()),
            orientation: Quat::from_array(body.rotation().to_array()),
            linear_velocity: Vec3::from_array(body.linvel().to_array()),
            angular_velocity: Vec3::from_array(body.angvel().to_array()),
            motor_states,
        }
    }

    /// Replace persistent forces with this command in the body's current frame.
    pub(super) fn apply(
        &self,
        world: &mut PhysicsWorld,
        action: DroneAction,
        motor_states: [DroneMotorState; 4],
    ) {
        let body = world
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
        for (((offset, sign), fraction), state) in
            MOTORS.into_iter().zip(action.fractions()).zip(motor_states)
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
}
