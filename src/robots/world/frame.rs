//! Opaque in-process identity for one world and its current physical boundary.

use super::RobotWorldFailure;
use std::{num::NonZeroU64, sync::Arc, time::Duration};

/// Opaque physical snapshot identity, retained across world moves but invalidated by reset.
/// No public constructor or serialization exposes its private scope.
///
/// ```compile_fail
/// use bevy_gym::robots::RobotFrame;
/// let forged = RobotFrame::first();
/// ```
#[derive(Debug, Clone)]
pub struct RobotFrame {
    /// Retention prevents scope allocation reuse while an old frame is still held.
    scope: Arc<Scope>,
    /// Positive boundary ordinal; one identifies the reset boundary.
    ordinal: NonZeroU64,
}

/// A private allocation identity, independent of seeds, addresses of moved worlds and wall time.
#[derive(Debug)]
struct Scope;

impl RobotFrame {
    /// Create a fresh reset scope with no applied action frames.
    pub(super) fn first() -> Self {
        Self {
            scope: Arc::new(Scope),
            ordinal: NonZeroU64::MIN,
        }
    }

    /// Advance the ordinal while retaining the same world/reset scope.
    pub(super) fn next(&self) -> Result<Self, RobotWorldFailure> {
        let ordinal = self
            .ordinal
            .checked_add(1)
            .ok_or(RobotWorldFailure::ClockExhausted)?;
        Ok(Self {
            scope: Arc::clone(&self.scope),
            ordinal,
        })
    }

    /// Derive elapsed policy time from completed 20 ms frames.
    pub(super) fn elapsed(&self) -> Duration {
        // Fifty completed 20 ms action frames form one second; remainder is milliseconds.
        let frames = self.ordinal.get() - 1;
        Duration::from_secs(frames / 50) + Duration::from_millis((frames % 50) * 20)
    }
}

impl PartialEq for RobotFrame {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.scope, &other.scope) && self.ordinal == other.ordinal
    }
}
impl Eq for RobotFrame {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ordinal exhaustion cannot wrap time or reuse a physical boundary.
    #[test]
    fn exhausted_clock_rejects_before_wrap() {
        let mut frame = RobotFrame::first();
        frame.ordinal = NonZeroU64::MAX;
        assert_eq!(frame.next(), Err(RobotWorldFailure::ClockExhausted));
        let frames = u64::MAX - 1;
        assert_eq!(
            frame.elapsed(),
            Duration::from_secs(frames / 50) + Duration::from_millis((frames % 50) * 20)
        );
    }
}
