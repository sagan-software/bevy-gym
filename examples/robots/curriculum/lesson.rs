//! The two task distributions and their independent promotion criterion.

use std::error::Error;

use bevy_gym::robots::DroneHover;
use bevy_gym::training::RecurrentPpoPolicy;

use crate::learning::{self, evaluation::EpisodeScore, RecoveryBatch, SELECTION_SEEDS};

/// Training proceeds in this order while retaining one optimizer.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Lesson {
    /// Learn motor balance from upright, stationary starts.
    Hover,
    /// Recover from the existing randomized tilt and velocity distribution.
    Recovery,
}

impl Lesson {
    /// Stable artifact name for this lesson.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Hover => "hover",
            Self::Recovery => "recovery",
        }
    }

    /// Reset lanes and recurrent memories at the start of a new lesson.
    pub(crate) fn batch(self, seed: u64, policy: &RecurrentPpoPolicy) -> RecoveryBatch {
        match self {
            Self::Hover => RecoveryBatch::with_environment(seed, policy, DroneHover::default),
            Self::Recovery => RecoveryBatch::new(seed, policy),
        }
    }

    /// Evaluate fixed selection seeds without using training lanes or optimizer state.
    pub(crate) fn evaluate(
        self,
        policy: &RecurrentPpoPolicy,
    ) -> Result<Vec<EpisodeScore>, Box<dyn Error>> {
        match self {
            Self::Hover => {
                learning::evaluation::evaluate_with(policy, &SELECTION_SEEDS, DroneHover::default)
            }
            Self::Recovery => learning::evaluate(policy, &SELECTION_SEEDS),
        }
    }
}

/// Require all five survivors and the frozen return and distance thresholds.
pub(crate) fn passes(scores: &[EpisodeScore]) -> bool {
    scores.len() == SELECTION_SEEDS.len()
        && scores
            .iter()
            .zip(SELECTION_SEEDS)
            .all(|(score, seed)| score.seed == seed && score.survived)
        && scores.iter().map(|score| score.reward).sum::<f64>() / 5.0 >= 400.0
        && scores
            .iter()
            .map(|score| f64::from(score.final_distance))
            .sum::<f64>()
            / 5.0
            <= 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores exactly at both inclusive promotion thresholds.
    fn boundary() -> Vec<EpisodeScore> {
        SELECTION_SEEDS
            .into_iter()
            .map(|seed| EpisodeScore {
                seed,
                steps: 500,
                reward: 400.0,
                final_distance: 0.5,
                survived: true,
            })
            .collect()
    }

    #[test]
    fn promotion_requires_the_complete_seed_set_and_every_survivor() {
        let scores = boundary();
        assert!(passes(&scores));
        assert!(!passes(&[]));
        assert!(!passes(&scores[..4]));
        for index in 0..5 {
            let mut crashed = scores.clone();
            crashed[index].survived = false;
            assert!(!passes(&crashed));
            let mut wrong_seed = scores.clone();
            wrong_seed[index].seed = 99;
            assert!(!passes(&wrong_seed));
        }
    }

    #[test]
    fn promotion_uses_inclusive_mean_return_and_distance_boundaries() {
        let mut scores = boundary();
        for score in &mut scores {
            score.reward = 399.999;
        }
        assert!(!passes(&scores));
        let mut scores = boundary();
        for score in &mut scores {
            score.final_distance = f32::from_bits(0.5_f32.to_bits() + 1);
        }
        assert!(!passes(&scores));
        let mut scores = boundary();
        scores[0].reward = 300.0;
        scores[1].reward = 500.0;
        scores[0].final_distance = 0.25;
        scores[1].final_distance = 0.75;
        assert!(passes(&scores));
    }

    #[test]
    fn lessons_use_distinct_profiles_and_independent_evaluation() {
        let policy =
            learning::load_policy(include_bytes!("../../../assets/robots/recovery.mpk").to_vec())
                .unwrap();
        assert_eq!(Lesson::Hover.name(), "hover");
        assert_eq!(Lesson::Recovery.name(), "recovery");
        let calm = Lesson::Hover.evaluate(&policy).unwrap();
        let disturbed = Lesson::Recovery.evaluate(&policy).unwrap();
        assert_ne!(calm, disturbed);
        for lesson in [Lesson::Hover, Lesson::Recovery] {
            let mut batch = lesson.batch(7, &policy);
            let sequences = batch.collect(&policy).unwrap();
            assert_eq!(sequences[0].initial_memory, policy.initial_memory());
            assert_eq!(
                sequences
                    .iter()
                    .map(|s| s.observations.len())
                    .sum::<usize>(),
                512
            );
        }
    }
}
