//! Selection ranking never reads held-out cases and never qualifies a lesson.

use super::checkpoint::Record;
use clap::ValueEnum;
use serde::Serialize;

/// Closed collection policies; existing commands retain their first-pass behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum SelectionPolicy {
    /// Stop at the first checkpoint passing the fixed selection cases.
    #[default]
    FirstPassing,
    /// Finish the budget and rank only checkpoints passing every selection gate.
    BestPassingBudget,
}

/// A checkpoint that passed the unchanged fixed selection gates.
#[derive(Debug, Serialize)]
pub(super) struct Candidate {
    /// Original validated identity and fresh run counters.
    pub(super) checkpoint: Record,
    /// Mean unscaled return across the fixed selection cases, in reward units.
    pub(super) mean_return: f64,
}

/// One run's selection state; a candidate does not imply held-out qualification.
#[derive(Debug)]
pub(super) struct Selection {
    /// Fixed collection policy chosen before any transitions are collected.
    policy: SelectionPolicy,
    /// Highest passing selection return; ties retain the earliest observed update.
    best: Option<Candidate>,
}

impl Selection {
    /// Begin with no passing candidate and no held-out information.
    pub(super) const fn new(policy: SelectionPolicy) -> Self {
        Self { policy, best: None }
    }

    /// Retain passing candidates and decide whether the selected collection policy stops.
    pub(super) fn observe(&mut self, candidate: Option<Candidate>) -> bool {
        let Some(candidate) = candidate else {
            return false;
        };
        if self
            .best
            .as_ref()
            .is_none_or(|best| candidate.mean_return > best.mean_return)
        {
            self.best = Some(candidate);
        }
        self.policy == SelectionPolicy::FirstPassing
    }

    /// Borrow selection evidence without granting lesson promotion.
    pub(super) const fn best(&self) -> Option<&Candidate> {
        self.best.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::{NonZeroU32, NonZeroU64};

    /// Identity fixtures test ranking only; they do not claim trained standing policies.
    fn candidate(update: u32, mean_return: f64) -> Candidate {
        Candidate {
            checkpoint: Record::new(
                42,
                NonZeroU32::new(update).expect("positive fixture"),
                NonZeroU64::MIN,
                b"ranking fixture",
            ),
            mean_return,
        }
    }

    /// Budget collection continues through passing reports; lower returns and ties retain the best.
    #[test]
    fn budget_ranks_only_passing_reports_and_retains_earliest_ties() {
        let mut selection = Selection::new(SelectionPolicy::BestPassingBudget);
        assert!(!selection.observe(None));
        assert!(selection.best().is_none());
        for (update, reward) in [(20, 800.0), (40, 900.0), (60, 850.0), (80, 900.0)] {
            assert!(!selection.observe(Some(candidate(update, reward))));
        }
        assert!(!selection.observe(None));
        let best = selection.best().expect("passing selection retained");
        assert_eq!(best.checkpoint.update().get(), 40);
        assert_eq!(best.mean_return.to_bits(), 900.0_f64.to_bits());
    }

    /// The original policy stops on its first pass and never stops on a failed report.
    #[test]
    fn original_selection_stops_only_on_pass() {
        let mut selection = Selection::new(SelectionPolicy::FirstPassing);
        assert!(!selection.observe(None));
        assert!(selection.observe(Some(candidate(20, 800.0))));
        assert_eq!(
            selection
                .best()
                .expect("selected")
                .checkpoint
                .update()
                .get(),
            20
        );
    }
}
