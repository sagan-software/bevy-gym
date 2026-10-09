//! Shared weapon frame for visible geometry, hand targets, and projectile launch.

use bevy::prelude::*;

/// Build a shoulder-relative grip frame; recoil translates back and lifts the barrel.
pub(super) fn pose(character: Vec3, direction: Dir3, recoil: f32) -> Transform {
    let yaw = Quat::from_rotation_y((-direction.x).atan2(-direction.z));
    let pitch = direction.y.clamp(-1.0, 1.0).asin();
    let aim = yaw * Quat::from_rotation_x(pitch);
    // Keep the reach pivot near the right shoulder, including when aiming vertically.
    let pivot = character + yaw * Vec3::new(0.18, 0.52, 0.12);
    Transform::from_translation(pivot + aim * Vec3::new(0.0, 0.0, recoil.mul_add(0.035, -0.52)))
        .with_rotation(
            aim * Quat::from_rotation_x(recoil * 0.04)
                * Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
        )
}

/// Retract the complete weapon frame before cover, preserving the shared muzzle.
pub(super) fn clear(arena: &super::arena::Arena, mut pose: Transform) -> Transform {
    let chest = arena.position() + Vec3::Y * 0.4;
    let muzzle = muzzle(&pose);
    if let Some(distance) = arena.obstruction(chest, muzzle) {
        let safe = chest + (muzzle - chest).normalize() * (distance - 0.002).max(0.0);
        pose.translation += safe - muzzle;
    }
    pose
}

/// The licensed source mesh's barrel endpoint in metres from its grip pivot.
pub(super) fn muzzle(pose: &Transform) -> Vec3 {
    pose.transform_point(Vec3::new(-0.372, 0.12, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_muzzle_and_recoil_share_the_source_mesh_frame() {
        let still = pose(Vec3::ZERO, Dir3::NEG_Z, 0.0);
        assert!(muzzle(&still).abs_diff_eq(Vec3::new(0.18, 0.64, -0.772), 0.00001));
        let kicked = pose(Vec3::ZERO, Dir3::NEG_Z, 1.0);
        assert!(kicked.translation.z > still.translation.z);
        assert!(muzzle(&kicked).y > muzzle(&still).y);
        for direction in [Dir3::X, Dir3::NEG_X, Dir3::Y, Dir3::NEG_Y, Dir3::Z] {
            let pose = pose(Vec3::ZERO, direction, 0.0);
            assert!(pose.translation.is_finite());
            assert!(pose.rotation.is_normalized());
            assert!(pose.translation.length() < 1.1);
        }
    }
}
