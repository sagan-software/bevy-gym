//! Finite projectiles test the complete travelled segment against arena and character.

use bevy::math::{Dir3, Vec3};
use std::time::Duration;

/// One point projectile with a fixed launch direction and finite lifetime.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Projectile {
    /// Current world position in metres.
    position: Vec3,
    /// Direction measured at discharge; later target movement cannot steer it.
    direction: Dir3,
    /// Remaining simulation time before expiry.
    remaining: Duration,
}

impl Projectile {
    /// Only the validated firing path constructs live projectiles.
    pub(super) const fn new(position: Vec3, direction: Dir3) -> Self {
        Self {
            position,
            direction,
            remaining: Duration::from_secs(2),
        }
    }

    /// Read the visible projectile position without exposing mutation.
    pub(crate) const fn position(self) -> Vec3 {
        self.position
    }
}

/// Three slots cover a full volley; the next warning cannot finish before expiry.
#[derive(Default)]
pub(crate) struct Projectiles {
    /// Empty slots retain no stale position or direction.
    slots: [Option<Projectile>; 3],
}

/// A caller tried to exceed the documented projectile capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Full;

impl Projectiles {
    /// Add an already validated launch without replacing an existing projectile.
    pub(crate) fn launch(&mut self, projectile: Projectile) -> Result<(), Full> {
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or(Full)?;
        *slot = Some(projectile);
        Ok(())
    }

    /// Borrow only live projectiles; projection cannot change their trajectory.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &Projectile> {
        self.slots.iter().filter_map(Option::as_ref)
    }

    /// Advance one twenty-millisecond action and apply collisions to authoritative health.
    pub(crate) fn advance(
        &mut self,
        arena: &crate::arena::Arena,
        health: &mut super::robot_health::RobotHealth,
    ) {
        for slot in &mut self.slots {
            if let Some(projectile) = slot {
                let from = projectile.position;
                // 18 m/s times 0.02 seconds gives a 0.36-metre segment, without homing.
                let to = from + *projectile.direction * (18.0 * 0.02);
                let wall = arena.obstruction(from, to);
                let character = arena.character_hit(from, to);
                // Walls win ties, including a muzzle embedded in solid geometry.
                if robot_precedes_wall(character, wall) {
                    health.hit();
                    *slot = None;
                } else if wall.is_some() {
                    *slot = None;
                } else {
                    projectile.position = to;
                    projectile.remaining = projectile
                        .remaining
                        .saturating_sub(Duration::from_millis(20));
                    if projectile.remaining.is_zero() {
                        *slot = None;
                    }
                }
            }
        }
    }
}

/// Nearest-hit policy gives solid cover priority at equal distances.
fn robot_precedes_wall(character: Option<f32>, wall: Option<f32>) -> bool {
    character.is_some_and(|hit| wall.is_none_or(|wall| hit < wall))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_order_gives_cover_priority_at_exact_ties() {
        assert!(!robot_precedes_wall(None, None));
        assert!(!robot_precedes_wall(None, Some(0.1)));
        assert!(robot_precedes_wall(Some(0.2), None));
        assert!(robot_precedes_wall(Some(0.1), Some(0.2)));
        assert!(!robot_precedes_wall(Some(0.2), Some(0.1)));
        assert!(!robot_precedes_wall(Some(0.2), Some(0.2)));
        assert!(!robot_precedes_wall(Some(0.0), Some(0.0)));
    }
}
