//! Shared camera mounting point for sensing, navigation, and the rendered lens.

/// Body-local camera centre in metres; forward is local negative Z.
pub(crate) const EYE_OFFSET: bevy::math::Vec3 = bevy::math::Vec3::new(0.0, 0.0, -0.22);
