//! Three projectile hits end the humanoid robot's episode.

use std::num::NonZeroU8;

/// Remaining hits are private; absence is an absorbing dead state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RobotHealth {
    /// One through three remaining hits, or death.
    remaining: Option<NonZeroU8>,
}

impl Default for RobotHealth {
    fn default() -> Self {
        Self {
            remaining: NonZeroU8::new(3),
        }
    }
}

impl RobotHealth {
    /// Only a living robot may move or use its pistol.
    pub(crate) const fn is_alive(self) -> bool {
        self.remaining.is_some()
    }

    /// Read the bounded health indicator.
    pub(crate) fn hits_remaining(self) -> u8 {
        self.remaining.map_or(0, NonZeroU8::get)
    }

    /// A collision consumes one hit; later collisions cannot revive or underflow health.
    pub(super) const fn hit(&mut self) {
        if let Some(remaining) = self.remaining {
            self.remaining = NonZeroU8::new(remaining.get() - 1);
        }
    }
}
