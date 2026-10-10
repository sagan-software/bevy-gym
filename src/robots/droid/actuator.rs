//! Torque axes in action order, in each parent body frame.

/// Torque axes in action order, in each parent body frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DroidActuator {
    /// Spine x right-handed torque axis.
    SpineX,
    /// Spine y right-handed torque axis.
    SpineY,
    /// Spine z right-handed torque axis.
    SpineZ,
    /// Neck x right-handed torque axis.
    NeckX,
    /// Neck y right-handed torque axis.
    NeckY,
    /// Neck z right-handed torque axis.
    NeckZ,
    /// Left shoulder x right-handed torque axis.
    LeftShoulderX,
    /// Left shoulder y right-handed torque axis.
    LeftShoulderY,
    /// Left shoulder z right-handed torque axis.
    LeftShoulderZ,
    /// Left elbow y right-handed torque axis.
    LeftElbowY,
    /// Right shoulder x right-handed torque axis.
    RightShoulderX,
    /// Right shoulder y right-handed torque axis.
    RightShoulderY,
    /// Right shoulder z right-handed torque axis.
    RightShoulderZ,
    /// Right elbow y right-handed torque axis.
    RightElbowY,
    /// Left hip x right-handed torque axis.
    LeftHipX,
    /// Left hip y right-handed torque axis.
    LeftHipY,
    /// Left hip z right-handed torque axis.
    LeftHipZ,
    /// Left knee x right-handed torque axis.
    LeftKneeX,
    /// Left ankle x right-handed torque axis.
    LeftAnkleX,
    /// Left ankle z right-handed torque axis.
    LeftAnkleZ,
    /// Right hip x right-handed torque axis.
    RightHipX,
    /// Right hip y right-handed torque axis.
    RightHipY,
    /// Right hip z right-handed torque axis.
    RightHipZ,
    /// Right knee x right-handed torque axis.
    RightKneeX,
    /// Right ankle x right-handed torque axis.
    RightAnkleX,
    /// Right ankle z right-handed torque axis.
    RightAnkleZ,
}

impl DroidActuator {
    /// Every actuator in the stable action order.
    pub const ALL: [Self; 26] = [
        Self::SpineX,
        Self::SpineY,
        Self::SpineZ,
        Self::NeckX,
        Self::NeckY,
        Self::NeckZ,
        Self::LeftShoulderX,
        Self::LeftShoulderY,
        Self::LeftShoulderZ,
        Self::LeftElbowY,
        Self::RightShoulderX,
        Self::RightShoulderY,
        Self::RightShoulderZ,
        Self::RightElbowY,
        Self::LeftHipX,
        Self::LeftHipY,
        Self::LeftHipZ,
        Self::LeftKneeX,
        Self::LeftAnkleX,
        Self::LeftAnkleZ,
        Self::RightHipX,
        Self::RightHipY,
        Self::RightHipZ,
        Self::RightKneeX,
        Self::RightAnkleX,
        Self::RightAnkleZ,
    ];
}
