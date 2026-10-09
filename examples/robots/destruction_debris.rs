//! A small independent collision world for visual wreckage.

use bevy::prelude::*;
use rapier3d::prelude::{ColliderBuilder, PhysicsWorld, RigidBodyBuilder, RigidBodyHandle, Vector};
use std::time::Duration;

/// Collision bodies owned by the current destruction burst only.
#[derive(Resource, Default)]
pub(super) struct Debris {
    /// No solver is retained before a crash or after cleanup.
    world: Option<PhysicsWorld>,
    /// Remaining lifetime of this burst.
    remaining: Duration,
    /// Unconsumed presentation time, bounded by the fixed step.
    accumulator: Duration,
}

/// One fragment's private body handle; it cannot reach the flight solver.
#[derive(Component)]
pub(super) struct Fragment(pub(super) RigidBodyHandle);

/// Fixed debris solver step, independent of the render frame rate.
const STEP: Duration = Duration::from_micros(16_667);
/// Wreckage remains for five seconds after destruction.
const LIFETIME: Duration = Duration::from_secs(5);

impl Debris {
    /// Create eight collision fragments in a private world with incoming velocity in m/s.
    pub(super) fn burst(
        &mut self,
        pose: Isometry3d,
        velocity: Vec3,
        mut world: PhysicsWorld,
    ) -> Vec<(Fragment, Transform)> {
        world.integration_parameters.dt = STEP.as_secs_f32();
        let mut fragments = Vec::with_capacity(8);
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let offset = Vec3::new(x * 0.15, y * 0.06, z * 0.15);
                    let mut position: Vec3 = pose.transform_point(offset).into();
                    position.y = position.y.max(0.15);
                    // Fragment velocities are m/s; spin is rad/s. The burst is visual energy.
                    let fragment_velocity =
                        velocity + Vec3::new(x * 1.8, y.mul_add(0.5, 2.5), z * 1.8);
                    let (handle, _) = world.insert(
                        RigidBodyBuilder::dynamic()
                            .translation(Vector::from_array(position.to_array()))
                            .rotation(Vector::from_array(
                                pose.rotation.to_scaled_axis().to_array(),
                            ))
                            .linvel(Vector::from_array(fragment_velocity.to_array()))
                            .angvel(Vector::new(z * 5.0, x * 7.0, y * 4.0))
                            .ccd_enabled(true),
                        ColliderBuilder::cuboid(0.11, 0.04, 0.11)
                            .mass(0.125)
                            .restitution(0.3)
                            .friction(0.6),
                    );
                    fragments.push((
                        Fragment(handle),
                        Transform::from_translation(position).with_rotation(pose.rotation),
                    ));
                }
            }
        }
        *self = Self {
            world: Some(world),
            remaining: LIFETIME,
            accumulator: Duration::ZERO,
        };
        fragments
    }

    /// Advance bounded fixed steps; report when all render fragments must be removed.
    pub(super) fn tick(&mut self, delta: Duration) -> bool {
        let Some(world) = self.world.as_mut() else {
            return false;
        };
        let delta = delta.min(Duration::from_millis(100));
        self.remaining = self.remaining.saturating_sub(delta);
        if self.remaining.is_zero() {
            *self = Self::default();
            return true;
        }
        self.accumulator += delta;
        while self.accumulator >= STEP {
            world.step();
            self.accumulator -= STEP;
        }
        false
    }

    /// Read a body's pose without exposing the solver or accepting physics mutations.
    pub(super) fn pose(&self, fragment: &Fragment) -> Option<Transform> {
        let body = self.world.as_ref()?.bodies.get(fragment.0)?;
        Some(
            Transform::from_translation(Vec3::from_array(body.translation().to_array()))
                .with_rotation(Quat::from_array(body.rotation().to_array())),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::{robots::DroneHover, Env};

    #[test]
    fn debris_moves_collides_expires_and_never_changes_the_drone() {
        let mut drone = DroneHover::default();
        let observation = drone.reset(Some(42)).observation;
        let mut debris = Debris::default();
        assert!(!debris.tick(Duration::from_secs(1)));
        let mut world = PhysicsWorld::default();
        world.insert(
            RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
            ColliderBuilder::cuboid(30.0, 0.1, 30.0),
        );
        let fragments = debris.burst(
            Isometry3d::new(observation.position(), observation.orientation()),
            observation.linear_velocity(),
            world,
        );
        assert_eq!(fragments.len(), 8);
        assert!(!debris.tick(STEP));
        assert_ne!(debris.pose(&fragments[0].0).unwrap(), fragments[0].1);
        for _ in 0..40 {
            assert!(!debris.tick(Duration::from_millis(100)));
        }
        for (fragment, _) in &fragments {
            let height = debris.pose(fragment).unwrap().translation.y;
            assert!(
                (0.03..=0.2).contains(&height),
                "fragment must settle on the floor: {height}"
            );
        }
        assert_eq!(drone.observation(), observation);
        for _ in 0..10 {
            debris.tick(Duration::from_millis(100));
        }
        assert!(debris.pose(&fragments[0].0).is_none());
        assert!(!debris.tick(STEP));
    }
}
