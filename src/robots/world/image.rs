//! Fixed collision-proxy RGB images from an owner's physical camera.
use super::{
    RobotFrame, RobotId, RobotRay, RobotRayTarget, RobotWorld, RobotWorldError, RobotWorldFailure,
};
use bevy::math::Vec3;

/// Row-major 17-by-13 RGB image from the owner's body-mounted camera.
///
/// The local profile uses a 90-degree horizontal field of view and a 20 m range.
/// Pixels depict collision proxies with flat materials, without identities or depth.
/// This image does not depict the spectator meshes, lighting or weapon effects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobotImage {
    /// Common physical boundary at capture time.
    frame: RobotFrame,
    /// Physical owner whose mount supplies the camera pose.
    owner: RobotId,
    /// Top-to-bottom rows and left-to-right pixels, with red, green and blue bytes.
    pixels: [[[u8; 3]; 17]; 13],
}
impl RobotImage {
    /// Read the source boundary without granting a mutable environment capability.
    #[must_use]
    pub const fn frame(&self) -> &RobotFrame {
        &self.frame
    }
    /// Read the owner of the physical camera.
    #[must_use]
    pub const fn owner(&self) -> RobotId {
        self.owner
    }
    /// Borrow the fixed row-major RGB buffer; no hidden position or identity is encoded.
    #[must_use]
    pub const fn pixels(&self) -> &[[[u8; 3]; 17]; 13] {
        &self.pixels
    }

    /// Render the original collision solids without advancing the solver or choosing actions.
    pub(super) fn capture(world: &RobotWorld, owner: RobotId) -> Result<Self, RobotWorldError> {
        let snapshot = world.snapshot()?;
        // The mount follows only its owner's pose. Offsets sit 0.02 m outside its collider.
        let (position, orientation, offset) = match owner {
            RobotId::Drone(slot) => {
                let body = snapshot.drone(slot);
                (
                    body.position(),
                    body.orientation(),
                    Vec3::new(0.0, 0.0, -0.317),
                )
            }
            RobotId::Droid(slot) => {
                let body = snapshot.droid(slot).body(crate::robots::DroidBody::Head);
                (
                    body.position(),
                    body.orientation(),
                    Vec3::new(0.0, 0.0, -0.1),
                )
            }
        };
        let origin = position + orientation * offset;
        let mut pixels = [[[0; 3]; 17]; 13];
        // Pixel centres sample a pinhole plane at local z=-1 m. The width is 2 m;
        // square pixels give height 26/17 m. Both coordinates are in metres.
        // The fixed RGB buffer uses 663 bytes; queries cost O(221 * solid count).
        for (row, pixels) in pixels.iter_mut().enumerate() {
            for (column, pixel) in pixels.iter_mut().enumerate() {
                let x = 2.0_f32.mul_add(column as f32, -16.0) / 17.0;
                let y = (-2.0_f32).mul_add(row as f32, 12.0) / 17.0;
                let direction = orientation * Vec3::new(x, y, -1.0);
                let ray =
                    RobotRay::try_from((origin, direction, 20.0)).map_err(|_invalid_geometry| {
                        RobotWorldError::Rejected(RobotWorldFailure::NonFinite(owner))
                    })?;
                // Nearest-solid occlusion precedes material projection. Owners are excluded.
                *pixel = match world
                    .cast_ray(&ray, Some(owner))?
                    .map(super::RobotRayHit::target)
                {
                    None => [16, 24, 40],
                    Some(RobotRayTarget::Environment) => [96, 104, 112],
                    Some(RobotRayTarget::Drone(_)) => [32, 176, 208],
                    Some(RobotRayTarget::Droid(_, _)) => [224, 160, 48],
                };
            }
        }
        Ok(Self {
            frame: snapshot.frame().clone(),
            owner,
            pixels,
        })
    }
}
