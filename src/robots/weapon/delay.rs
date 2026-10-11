//! Positive remaining mechanical durations with no caller-supplied construction.
use std::time::Duration;
/// Only fixed positive rules and a positive elapsed remainder can construct a delay.
#[derive(Debug, Clone, Copy)]
pub(super) struct Delay(
    #[doc = "Validated positive duration, retained only before expiry."] Duration,
);
impl Delay {
    /// Drone spin-up time.
    pub(super) const fn charge() -> Self {
        Self(Duration::from_millis(800))
    }
    /// Time between successive drone rounds.
    pub(super) const fn volley() -> Self {
        Self(Duration::from_millis(120))
    }
    /// Recovery after a completed or interrupted drone volley.
    pub(super) const fn drone_cooling() -> Self {
        Self(Duration::from_millis(1200))
    }
    /// Pistol recovery, observed at fixed twenty-millisecond boundaries.
    pub(super) const fn droid_cooling() -> Self {
        Self(Duration::from_millis(250))
    }
    /// Project the validated duration without exposing its constructor.
    pub(super) const fn remaining(self) -> Duration {
        self.0
    }
    /// Return another delay only when its remainder remains positive.
    pub(super) const fn elapse(self) -> Option<Self> {
        let remaining = self.0.saturating_sub(Duration::from_millis(20));
        if remaining.is_zero() {
            None
        } else {
            Some(Self(remaining))
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Fixed positive delays preserve their exact remainder and never retain zero.
    #[test]
    fn durations_only_retain_positive_remainders() {
        for (initial, milliseconds) in [
            (Delay::charge(), 800),
            (Delay::volley(), 120),
            (Delay::drone_cooling(), 1200),
            (Delay::droid_cooling(), 250),
        ] {
            let mut delay = Some(initial);
            let mut expected = milliseconds;
            while let Some(current) = delay {
                assert_eq!(current.remaining(), Duration::from_millis(expected));
                expected = expected.saturating_sub(20);
                delay = current.elapse();
            }
            assert_eq!(expected, 0);
        }
    }
}
