//! Bounded straight projectile flight and swept collision geometry.
use super::{
    RobotId, RobotMuzzle, RobotRay, RobotRayHit, RobotWorld, RobotWorldError, RobotWorldFailure,
};
use bevy::math::Vec3;
use std::num::NonZeroU16;

/// One round retains its physical launch direction, owner and positive remaining lifetime.
#[derive(Debug)]
pub(super) struct Projectile {
    /// Source robot excluded throughout flight.
    owner: RobotId,
    /// Current world-space position in metres.
    position: Vec3,
    /// Normalized launch direction, fixed after discharge.
    direction: Vec3,
    /// Positive remaining 20 ms segments; 100 segments equal two seconds.
    remaining: NonZeroU16,
}
/// A swept segment either retains flight or consumes the projectile once.
#[derive(Debug)]
pub(super) enum Flight {
    /// Lifetime remains after an unobstructed segment.
    Flying(#[doc = "The only retained projectile after this segment."] Projectile),
    /// The final unobstructed segment completed.
    Expired,
    /// The nearest solid consumed this round.
    Hit(#[doc = "Nearest physical solid that consumed the round."] RobotRayHit),
}
impl From<&RobotMuzzle> for Projectile {
    fn from(muzzle: &RobotMuzzle) -> Self {
        Self {
            owner: muzzle.owner(),
            position: muzzle.origin(),
            direction: muzzle.direction(),
            remaining: NonZeroU16::new(100).expect("positive two-second lifetime"),
        }
    }
}
impl Projectile {
    /// Sweep 18 m/s times 0.020 s, including the final segment's endpoint.
    pub(super) fn advance(mut self, world: &RobotWorld) -> Result<Flight, RobotWorldError> {
        let ray = RobotRay::try_from((self.position, self.direction, 0.36)).map_err(
            |_invalid_geometry| RobotWorldError::Rejected(RobotWorldFailure::NonFinite(self.owner)),
        )?;
        if let Some(hit) = world.cast_ray(&ray, Some(self.owner))? {
            return Ok(Flight::Hit(hit));
        }
        self.position += self.direction * 0.36;
        // Consume the last segment instead of retaining an impossible zero-lifetime round.
        self.remaining
            .get()
            .checked_sub(1)
            .and_then(NonZeroU16::new)
            .map_or(Ok(Flight::Expired), |remaining| {
                self.remaining = remaining;
                Ok(Flight::Flying(self))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robots::RobotSlot;

    /// Every unobstructed segment advances once; the hundredth consumes the round.
    #[test]
    fn unobstructed_round_expires_after_one_hundred_segments() {
        let world = RobotWorld::default();
        let snapshot = world.snapshot().unwrap();
        let muzzle = RobotMuzzle::try_from((&snapshot, RobotId::Drone(RobotSlot::First))).unwrap();
        let mut round = Projectile::from(&muzzle);
        for remaining in (2..=100).rev() {
            assert_eq!(round.remaining.get(), remaining);
            let Flight::Flying(next) = round.advance(&world).unwrap() else {
                panic!("unobstructed round must retain flight");
            };
            round = next;
        }
        assert!(matches!(round.advance(&world).unwrap(), Flight::Expired));
    }

    /// Internal corrupt flight state rejects geometry before collision arithmetic.
    #[test]
    fn invalid_projectile_position_preserves_owner_in_error() {
        let world = RobotWorld::default();
        let snapshot = world.snapshot().unwrap();
        let owner = RobotId::Drone(RobotSlot::First);
        let muzzle = RobotMuzzle::try_from((&snapshot, owner)).unwrap();
        let mut round = Projectile::from(&muzzle);
        round.position = Vec3::splat(f32::NAN);
        assert!(
            matches!(round.advance(&world), Err(RobotWorldError::Rejected(RobotWorldFailure::NonFinite(robot))) if robot == owner)
        );
    }
}
