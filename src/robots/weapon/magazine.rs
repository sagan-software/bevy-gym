//! Positive pistol counts that can only descend from twelve.
use std::num::NonZeroU8;
/// A validated remaining count; callers cannot supply an integer or increase it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Magazine {
    /// One through twelve remaining rounds, derived from the full magazine.
    rounds: NonZeroU8,
}
impl Default for Magazine {
    fn default() -> Self {
        Self {
            rounds: NonZeroU8::MIN.saturating_add(11),
        }
    }
}
impl Magazine {
    /// Remove exactly one round; absence means exhaustion.
    pub(super) fn discharge(self) -> Option<Self> {
        NonZeroU8::new(self.rounds.get() - 1).map(|rounds| Self { rounds })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Every reachable count remains in1..=12, and the last discharge has no successor.
    #[test]
    fn counts_descend_once_without_zero_or_reload() {
        let mut magazine = Some(Magazine::default());
        for expected in (1..=12).rev() {
            let current = magazine.unwrap();
            assert_eq!(current.rounds.get(), expected);
            magazine = current.discharge();
        }
        assert!(magazine.is_none());
    }
}
