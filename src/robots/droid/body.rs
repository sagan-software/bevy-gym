//! Physical segments in observation order.

/// Physical segments in observation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DroidBody {
    /// Pelvis segment.
    Pelvis,
    /// Torso segment.
    Torso,
    /// Head segment.
    Head,
    /// Left upper arm segment.
    LeftUpperArm,
    /// Left forearm segment.
    LeftForearm,
    /// Right upper arm segment.
    RightUpperArm,
    /// Right forearm segment.
    RightForearm,
    /// Left thigh segment.
    LeftThigh,
    /// Left calf segment.
    LeftCalf,
    /// Left foot segment.
    LeftFoot,
    /// Right thigh segment.
    RightThigh,
    /// Right calf segment.
    RightCalf,
    /// Right foot segment.
    RightFoot,
}

impl DroidBody {
    /// Return the segment centre in the common bind frame, in metres.
    ///
    /// Bind axes are +X right, +Y up and -Z forward; each segment has identity rotation.
    /// Renderers use this read-only profile to align bones without changing physical state.
    #[must_use]
    pub fn bind_position(self) -> bevy::math::Vec3 {
        bevy::math::Vec3::from_array(super::geometry::segment(self).centre.to_array())
    }

    /// Every body in the stable observation order.
    pub const ALL: [Self; 13] = [
        Self::Pelvis,
        Self::Torso,
        Self::Head,
        Self::LeftUpperArm,
        Self::LeftForearm,
        Self::RightUpperArm,
        Self::RightForearm,
        Self::LeftThigh,
        Self::LeftCalf,
        Self::LeftFoot,
        Self::RightThigh,
        Self::RightCalf,
        Self::RightFoot,
    ];
}
