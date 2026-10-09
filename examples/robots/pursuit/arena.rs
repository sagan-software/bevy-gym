//! Fixed-step character movement and a shared collision/rendering layout.
//!
//! Uses Rapier 0.36's capsule controller as documented at
//! <https://rapier.rs/docs/user_guides/rust/character_controller_setup/>.

#[path = "layout.rs"]
pub(super) mod layout;
#[path = "movement.rs"]
mod movement;

pub(super) use layout::Block;
pub(super) use movement::Movement;

use bevy::math::Vec3;
use rapier3d::{
    control::{CharacterAutostep, CharacterLength, KinematicCharacterController},
    prelude::{
        ColliderBuilder, PhysicsWorld, Pose, QueryFilter, Ray, RigidBodyBuilder, SharedShape,
        Vector,
    },
};

/// Every controller action advances twenty milliseconds.
const STEP_SECONDS: f32 = 0.02;
/// Initial capsule centre, in metres, clear of the cover blocks.
const SPAWN: Vec3 = Vec3::new(0.0, 0.92, 9.0);

/// The example owns collision state; renderers cannot mutate its solver.
pub(super) struct Arena {
    /// Static collision geometry and query acceleration structures.
    world: PhysicsWorld,
    /// The exact boxes used to construct the collision world.
    blocks: Vec<Block>,
    /// Capsule shape reused by every movement query.
    shape: SharedShape,
    /// Authoritative capsule centre, in metres.
    position: Vec3,
    /// Falling velocity in metres per second.
    vertical_velocity: f32,
    /// Support reported by the last character-controller action.
    grounded: bool,
}

impl Default for Arena {
    fn default() -> Self {
        let blocks = layout::blocks();
        let mut world = PhysicsWorld::default();
        for block in &blocks {
            world.insert(
                RigidBodyBuilder::fixed()
                    .translation(Vector::from_array(block.centre.to_array()))
                    .rotation(Vector::from_array(
                        block.rotation.to_scaled_axis().to_array(),
                    )),
                ColliderBuilder::cuboid(block.half.x, block.half.y, block.half.z),
            );
        }
        // Populate broad-phase queries once; all authored obstacles remain fixed.
        world.step();
        Self {
            world,
            blocks,
            shape: SharedShape::capsule_y(0.6, 0.3),
            position: SPAWN,
            vertical_velocity: 0.0,
            grounded: false,
        }
    }
}

impl Arena {
    /// Test fixture starting two metres above the normal spawn, with the same collision world.
    #[cfg(test)]
    pub(super) fn airborne() -> Self {
        Self {
            position: SPAWN + Vec3::Y * 2.0,
            ..Self::default()
        }
    }

    /// Borrow authored boxes; rendering cannot change the collision layout.
    pub(super) fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Advance one 20-millisecond action through collision-aware movement.
    pub(super) fn step(&mut self, movement: Movement) {
        self.step_relative(movement, 0.0);
    }

    /// Rotate player-local walking by camera yaw, in radians, before collision resolution.
    pub(super) fn step_relative(&mut self, movement: Movement, yaw: f32) {
        // Speed is 4 m/s; gravity is 9.81 m/s². The step is measured in seconds.
        self.vertical_velocity = 9.81_f32.mul_add(-STEP_SECONDS, self.vertical_velocity);
        let desired =
            bevy::math::Quat::from_rotation_y(yaw) * movement.direction() * (4.0 * STEP_SECONDS)
                + Vec3::Y * (self.vertical_velocity * STEP_SECONDS);
        let controller = KinematicCharacterController {
            offset: CharacterLength::Absolute(0.01),
            autostep: Some(CharacterAutostep {
                max_height: CharacterLength::Absolute(0.25),
                min_width: CharacterLength::Absolute(0.2),
                include_dynamic_bodies: false,
            }),
            ..Default::default()
        };
        let pose = Pose::from_translation(Vector::from_array(self.position.to_array()));
        let corrected = controller.move_shape(
            STEP_SECONDS,
            &self.world.query_pipeline(),
            self.shape.as_ref(),
            &pose,
            Vector::from_array(desired.to_array()),
            |_| {},
        );
        self.position += Vec3::from_array(corrected.translation.to_array());
        self.grounded = corrected.grounded;
        if self.grounded {
            self.vertical_velocity = 0.0;
        }
    }

    /// Read the capsule centre without exposing the collision world.
    pub(super) const fn position(&self) -> Vec3 {
        self.position
    }

    /// Read support from the last action; construction and reset have not sampled it.
    pub(super) const fn is_grounded(&self) -> bool {
        self.grounded
    }

    /// Restore the initial character state while retaining immutable obstacles.
    pub(super) const fn reset(&mut self) {
        self.position = SPAWN;
        self.vertical_velocity = 0.0;
        self.grounded = false;
    }

    /// Intersect the same movement capsule with a normalized finite segment.
    pub(super) fn character_hit(&self, from: Vec3, to: Vec3) -> Option<f32> {
        let offset = to - from;
        let distance = offset.length();
        if !from.is_finite() || !to.is_finite() || !distance.is_finite() || distance <= f32::EPSILON
        {
            return None;
        }
        let ray = Ray::new(
            Vector::from_array(from.to_array()),
            Vector::from_array((offset / distance).to_array()),
        );
        let pose = Pose::from_translation(Vector::from_array(self.position.to_array()));
        // Parry 0.31.1: solid=true reports zero when a segment starts inside the capsule.
        self.shape.cast_ray(&pose, &ray, distance, true)
    }

    /// Return the first obstruction distance in metres; invalid coordinates return zero.
    pub(super) fn obstruction(&self, from: Vec3, to: Vec3) -> Option<f32> {
        let direction = to - from;
        let distance = direction.length();
        // Invalid query coordinates fail closed; a zero-length segment has no obstruction.
        if !from.is_finite() || !to.is_finite() || !distance.is_finite() {
            return Some(0.0);
        }
        if distance <= f32::EPSILON {
            return None;
        }
        let ray = Ray::new(
            Vector::from_array(from.to_array()),
            Vector::from_array((direction / distance).to_array()),
        );
        self.world
            .cast_ray(&ray, distance, true, QueryFilter::default())
            .map(|(_, distance)| distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn airborne_character_falls_and_lands_without_floor_penetration() {
        let mut arena = Arena::airborne();
        let initial = arena.position();
        arena.step(Movement::Idle);
        assert!(arena.position().y < initial.y);
        assert!(arena.vertical_velocity < 0.0);
        assert!(!arena.is_grounded());
        for _ in 0..100 {
            arena.step(Movement::Idle);
        }
        assert!((0.90..0.92).contains(&arena.position().y));
        assert!(arena.vertical_velocity.abs() < f32::EPSILON);
        assert!(arena.is_grounded());
    }

    #[test]
    fn static_world_matches_layout_and_retains_it_across_reset() {
        let mut arena = Arena::default();
        assert_eq!(arena.world.colliders.len(), arena.blocks.len());
        assert!(arena.world.bodies.iter().all(|(_, body)| body.is_fixed()));
        let before: Vec<_> = arena
            .world
            .colliders
            .iter()
            .map(|(handle, collider)| (handle, *collider.position()))
            .collect();
        arena.step(Movement::Forward);
        arena.reset();
        let after: Vec<_> = arena
            .world
            .colliders
            .iter()
            .map(|(handle, collider)| (handle, *collider.position()))
            .collect();
        assert_eq!(before, after);
    }
}
