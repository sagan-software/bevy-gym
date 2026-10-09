//! Closed movement actions shared by keyboard input and future policy output.

use bevy::math::Vec3;

/// World-relative walking choices; forward means negative Z.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Movement {
    /// Remain still horizontally while gravity continues.
    #[default]
    Idle,
    /// Walk toward negative Z.
    Forward,
    /// Walk toward positive Z.
    Backward,
    /// Walk toward negative X.
    Left,
    /// Walk toward positive X.
    Right,
    /// Walk toward negative X and negative Z.
    ForwardLeft,
    /// Walk toward positive X and negative Z.
    ForwardRight,
    /// Walk toward negative X and positive Z.
    BackwardLeft,
    /// Walk toward positive X and positive Z.
    BackwardRight,
}

impl Movement {
    /// A unit horizontal vector, or zero for idle.
    pub(super) fn direction(self) -> Vec3 {
        let direction = match self {
            Self::Idle => Vec3::ZERO,
            Self::Forward => Vec3::NEG_Z,
            Self::Backward => Vec3::Z,
            Self::Left => Vec3::NEG_X,
            Self::Right => Vec3::X,
            Self::ForwardLeft => Vec3::new(-1.0, 0.0, -1.0),
            Self::ForwardRight => Vec3::new(1.0, 0.0, -1.0),
            Self::BackwardLeft => Vec3::new(-1.0, 0.0, 1.0),
            Self::BackwardRight => Vec3::new(1.0, 0.0, 1.0),
        };
        direction.normalize_or_zero()
    }
}

/// Convert forward/backward/left/right button states, cancelling opposite inputs.
impl From<[bool; 4]> for Movement {
    fn from([forward, backward, left, right]: [bool; 4]) -> Self {
        match (
            i8::from(right) - i8::from(left),
            i8::from(backward) - i8::from(forward),
        ) {
            (-1, -1) => Self::ForwardLeft,
            (0, -1) => Self::Forward,
            (1, -1) => Self::ForwardRight,
            (-1, 0) => Self::Left,
            (1, 0) => Self::Right,
            (-1, 1) => Self::BackwardLeft,
            (0, 1) => Self::Backward,
            (1, 1) => Self::BackwardRight,
            _ => Self::Idle,
        }
    }
}
