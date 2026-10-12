//! Frame-bound weapon geometry derived from authoritative physical poses.
use super::{InvalidRobotRay, RobotFrame, RobotId, RobotRay, RobotSnapshot};
use bevy::math::Vec3;

/// Immutable muzzle bound to its owner's measured physical frame.
#[derive(Debug, Clone)]
pub struct RobotMuzzle {
    /// Source boundary prevents querying geometry after movement or reset.
    frame: RobotFrame,
    /// Physical owner excluded from the muzzle's collision query.
    owner: RobotId,
    /// Validated world-space geometry.
    pub(super) ray: RobotRay,
}

impl RobotMuzzle {
    /// Read the measured source boundary.
    #[must_use]
    pub const fn frame(&self) -> &RobotFrame {
        &self.frame
    }
    /// Read the physical weapon owner.
    #[must_use]
    pub const fn owner(&self) -> RobotId {
        self.owner
    }
    /// Read the muzzle origin in world-space metres.
    #[must_use]
    pub const fn origin(&self) -> Vec3 {
        self.ray.origin
    }
    /// Read the normalized world-space firing direction.
    #[must_use]
    pub const fn direction(&self) -> Vec3 {
        self.ray.direction
    }
    /// Read the inclusive geometric range in metres.
    #[must_use]
    pub const fn range_metres(&self) -> f32 {
        self.ray.metres
    }
}

impl TryFrom<(&RobotSnapshot, RobotId)> for RobotMuzzle {
    type Error = InvalidRobotRay;
    /// Derive geometry from the owner's body without selecting a target or firing.
    fn try_from((snapshot, owner): (&RobotSnapshot, RobotId)) -> Result<Self, Self::Error> {
        // Mounts sit 0.02 m outside the existing collision half-extents.
        // The drone faces -Z; the right forearm's distal axis is +X in its bind pose.
        let (position, orientation, offset, axis, metres) = match owner {
            RobotId::Drone(slot) => {
                let body = snapshot.drone(slot);
                (
                    body.position(),
                    body.orientation(),
                    Vec3::new(0.0, 0.0, -0.317),
                    Vec3::NEG_Z,
                    36.0,
                )
            }
            RobotId::Droid(slot) => {
                let body = snapshot
                    .droid(slot)
                    .body(crate::robots::DroidBody::RightForearm);
                (
                    body.position(),
                    body.orientation(),
                    Vec3::new(0.197, 0.0, 0.0),
                    Vec3::X,
                    30.0,
                )
            }
        };
        // Validate transformed geometry before retaining the source identity.
        let ray =
            RobotRay::try_from((position + orientation * offset, orientation * axis, metres))?;
        Ok(Self {
            frame: snapshot.frame().clone(),
            owner,
            ray,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robots::{RobotSlot, RobotWorld};
    use bevy::math::Quat;

    /// Rotation transforms both mount displacement and firing direction.
    #[test]
    fn rotated_mount_uses_body_coordinates() {
        let mut snapshot = RobotWorld::default().snapshot().unwrap();
        let [first, _, _] = &mut snapshot.drones;
        first.orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let body = *first;
        let muzzle = RobotMuzzle::try_from((&snapshot, RobotId::Drone(RobotSlot::First))).unwrap();
        assert!((muzzle.direction() - Vec3::NEG_X).length() < 1e-6);
        assert!((muzzle.origin() - (body.position() + Vec3::NEG_X * 0.317)).length() < 1e-6);
    }

    /// Corrupt internal snapshots cannot create usable collision geometry.
    #[test]
    fn invalid_pose_rejects_geometry_before_retaining_frame() {
        let mut snapshot = RobotWorld::default().snapshot().unwrap();
        let owner = RobotId::Drone(RobotSlot::First);
        let [first, _, _] = &mut snapshot.drones;
        first.position = Vec3::splat(f32::NAN);
        assert_eq!(
            RobotMuzzle::try_from((&snapshot, owner)).unwrap_err(),
            InvalidRobotRay::Origin
        );
    }
}
