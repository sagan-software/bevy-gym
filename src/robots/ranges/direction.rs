//! The six body-frame directions in the fixed range profile.

/// Sensor directions relative to the physical drone; forward is local -Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneRangeDirection {
    /// Local -Z, toward the drone's front.
    Forward,
    /// Local +Z, toward the drone's rear.
    Back,
    /// Local -X.
    Left,
    /// Local +X.
    Right,
    /// Local +Y.
    Up,
    /// Local -Y.
    Down,
}

impl DroneRangeDirection {
    /// Fixed reading order, independent of collider insertion order.
    pub const ALL: [Self; 6] = [
        Self::Forward,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Up,
        Self::Down,
    ];
}

impl DroneRangeDirection {
    /// Unit vector in the physical body's local axes.
    pub(super) const fn unit(self) -> bevy::math::Vec3 {
        use bevy::math::Vec3;
        match self {
            Self::Forward => Vec3::NEG_Z,
            Self::Back => Vec3::Z,
            Self::Left => Vec3::NEG_X,
            Self::Right => Vec3::X,
            Self::Up => Vec3::Y,
            Self::Down => Vec3::NEG_Y,
        }
    }
}
