//! Read-only framing of the shared world and each individual robot.

use bevy::math::Vec3;
use bevy_gym::robots::{DroidBody, RobotId, RobotSlot, RobotSnapshot};

/// Spectator camera state is independent of actor observations and policy memory.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, bevy::prelude::Resource)]
pub(crate) enum Mode {
    /// Fit all forty-two physical body centres.
    #[default]
    All,
    /// Fit the selected robot's physical body centres.
    Follow(RobotId),
    /// Retain the current camera and accept presentation-only movement.
    Free,
}

impl Mode {
    /// Cycle the overview and all six follow targets in stable team-slot order.
    pub(crate) const fn next(self) -> Self {
        use RobotSlot::{First, Second, Third};
        match self {
            Self::All => Self::Follow(RobotId::Drone(First)),
            Self::Follow(RobotId::Drone(First)) => Self::Follow(RobotId::Drone(Second)),
            Self::Follow(RobotId::Drone(Second)) => Self::Follow(RobotId::Drone(Third)),
            Self::Follow(RobotId::Drone(Third)) => Self::Follow(RobotId::Droid(First)),
            Self::Follow(RobotId::Droid(First)) => Self::Follow(RobotId::Droid(Second)),
            Self::Follow(RobotId::Droid(Second)) => Self::Follow(RobotId::Droid(Third)),
            Self::Follow(RobotId::Droid(Third)) | Self::Free => Self::All,
        }
    }
    /// Read the exact visible target name.
    pub(crate) const fn label(self) -> &'static str {
        use RobotSlot::{First, Second, Third};
        match self {
            Self::All => "All",
            Self::Free => "Free",
            Self::Follow(RobotId::Drone(First)) => "Drone 1",
            Self::Follow(RobotId::Drone(Second)) => "Drone 2",
            Self::Follow(RobotId::Drone(Third)) => "Drone 3",
            Self::Follow(RobotId::Droid(First)) => "Droid 1",
            Self::Follow(RobotId::Droid(Second)) => "Droid 2",
            Self::Follow(RobotId::Droid(Third)) => "Droid 3",
        }
    }
}

/// Return camera position and target in metres for a finite aspect and FOV in radians.
pub(crate) fn frame(
    snapshot: &RobotSnapshot,
    mode: Mode,
    aspect: f32,
    fov: f32,
) -> Option<(Vec3, Vec3)> {
    if matches!(mode, Mode::Free) {
        return None;
    }
    let positions = RobotSlot::ALL.into_iter().flat_map(|slot| {
        let drone = (mode == Mode::All || mode == Mode::Follow(RobotId::Drone(slot)))
            .then(|| snapshot.drone(slot).position());
        let droids = DroidBody::ALL
            .into_iter()
            .filter(move |_| mode == Mode::All || mode == Mode::Follow(RobotId::Droid(slot)))
            .map(move |body| snapshot.droid(slot).body(body).position());
        drone.into_iter().chain(droids)
    });
    framed_positions(positions, aspect, fov)
}

/// Bound at most forty-two points with constant auxiliary space and no collected positions.
fn framed_positions(
    mut positions: impl Iterator<Item = Vec3>,
    aspect: f32,
    fov: f32,
) -> Option<(Vec3, Vec3)> {
    if !aspect.is_finite()
        || aspect <= 0.0
        || !fov.is_finite()
        || !(0.0..std::f32::consts::PI).contains(&fov)
        || fov == 0.0
    {
        return None;
    }
    let first = positions.next()?;
    if !first.is_finite() {
        return None;
    }
    let mut minimum = first;
    let mut maximum = first;
    for position in positions {
        if !position.is_finite() {
            return None;
        }
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }
    // Coordinates, radius and distance are metres. Aspect and the 1.2 margin are unitless.
    // Vertical FOV and the derived horizontal half-angle are radians.
    let target = minimum * 0.5 + maximum * 0.5;
    let radius = (maximum - minimum).length().mul_add(0.5, 0.4).max(1.0);
    let vertical = fov * 0.5;
    let horizontal = (vertical.tan() * aspect).atan();
    let distance = radius / vertical.sin().min(horizontal.sin()) * 1.2;
    let position = target + Vec3::new(1.2, 0.8, -1.4).normalize() * distance;
    if !position.is_finite() {
        return None;
    }
    Some((position, target))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty, nonfinite and overflowing bounds retain the last spectator camera.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn invalid_point_bounds_do_not_produce_a_camera_pose() {
        assert!(framed_positions(std::iter::empty(), 1.0, 1.0).is_none());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let point = Vec3::splat(invalid);
            assert!(framed_positions([point].into_iter(), 1.0, 1.0).is_none());
            assert!(framed_positions([Vec3::ZERO, point].into_iter(), 1.0, 1.0).is_none());
        }
        assert!(framed_positions(
            [Vec3::splat(-f32::MAX), Vec3::splat(f32::MAX)].into_iter(),
            1.0,
            1.0
        )
        .is_none());
        assert!(framed_positions([Vec3::ZERO].into_iter(), f32::from_bits(1), 1.0).is_none());
        let (position, target) =
            framed_positions([Vec3::ZERO].into_iter(), 1.0, 1.0).expect("finite coincident bounds");
        assert_eq!(target, Vec3::ZERO);
        assert!(position.is_finite());
    }
}
