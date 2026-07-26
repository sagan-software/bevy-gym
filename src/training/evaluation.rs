//! Fixed evaluation suites, aggregate statistics, and proof gates.

use std::error::Error;
use std::fmt;

/// Two-sided confidence interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    /// Lower confidence bound.
    pub lower: f64,

    /// Upper confidence bound.
    pub upper: f64,
}

/// One completed evaluation episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpisodeOutcome {
    /// Undiscounted episode return.
    pub reward: f64,

    /// Episode length in environment steps.
    pub length: usize,

    /// Environment-specific success signal.
    pub success: bool,
}

impl EpisodeOutcome {
    /// Create an evaluation outcome.
    #[must_use]
    pub const fn new(reward: f64, length: usize, success: bool) -> Self {
        Self {
            reward,
            length,
            success,
        }
    }
}

/// Aggregate statistics for a fixed evaluation suite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationSummary {
    /// Number of evaluated episodes.
    pub sample_count: usize,

    /// Arithmetic mean episode return.
    pub mean_reward: f64,

    /// Approximate two-sided 95% interval for the mean return.
    pub mean_reward_interval_95: ConfidenceInterval,

    /// Arithmetic mean episode length.
    pub mean_length: f64,

    /// Minimum episode return.
    pub min_reward: f64,

    /// Maximum episode return.
    pub max_reward: f64,

    /// Number of successful episodes.
    pub success_count: usize,

    /// Successful episode fraction.
    pub success_rate: f64,

    /// Wilson two-sided 95% interval for the success fraction.
    pub success_interval_95: ConfidenceInterval,
}

impl EvaluationSummary {
    /// Aggregate a non-empty set of finite episode outcomes.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::EmptyOutcomes`] for an empty slice and
    /// [`EvaluationError::NonFiniteReward`] when any reward is not finite.
    pub fn from_outcomes(outcomes: &[EpisodeOutcome]) -> Result<Self, EvaluationError> {
        if outcomes.is_empty() {
            return Err(EvaluationError::EmptyOutcomes);
        }

        let mut reward_sum = 0.0;
        let mut length_sum = 0usize;
        let mut min_reward = f64::INFINITY;
        let mut max_reward = f64::NEG_INFINITY;
        let mut success_count = 0usize;

        for (index, outcome) in outcomes.iter().enumerate() {
            if !outcome.reward.is_finite() {
                return Err(EvaluationError::NonFiniteReward { index });
            }
            reward_sum += outcome.reward;
            length_sum = length_sum.saturating_add(outcome.length);
            min_reward = min_reward.min(outcome.reward);
            max_reward = max_reward.max(outcome.reward);
            success_count += usize::from(outcome.success);
        }

        let sample_count = outcomes.len();
        let sample_count_f64 = sample_count as f64;
        let mean_reward = reward_sum / sample_count_f64;
        let mean_length = length_sum as f64 / sample_count_f64;
        let success_rate = success_count as f64 / sample_count_f64;

        Ok(Self {
            sample_count,
            mean_reward,
            mean_reward_interval_95: mean_interval_95(outcomes, mean_reward),
            mean_length,
            min_reward,
            max_reward,
            success_count,
            success_rate,
            success_interval_95: wilson_interval_95(success_count, sample_count),
        })
    }
}

/// Purpose of a fixed seed suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationSuiteKind {
    /// Selects the best checkpoint during training.
    Validation,

    /// Disjoint final proof after fresh checkpoint reload.
    Test,

    /// Fixed rollouts used for chronological video comparison.
    Demo,
}

/// Precomputed seeds reused for every checkpoint in one evaluation purpose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationSuite {
    /// Suite purpose.
    pub kind: EvaluationSuiteKind,

    /// Exact reset seeds in evaluation order.
    pub seeds: Vec<u64>,
}

impl EvaluationSuite {
    /// Build a fixed, non-empty suite from an already partitioned base seed.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError::EmptySuite`] when `sample_count` is zero.
    pub fn new(
        kind: EvaluationSuiteKind,
        base_seed: u64,
        sample_count: usize,
    ) -> Result<Self, EvaluationError> {
        if sample_count == 0 {
            return Err(EvaluationError::EmptySuite);
        }

        let seeds = (0..sample_count)
            .map(|index| base_seed.wrapping_add(index as u64))
            .collect();
        Ok(Self { kind, seeds })
    }
}

/// Predeclared aggregate thresholds for an evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EvaluationGate {
    /// Required mean return, if any.
    pub minimum_mean_reward: Option<f64>,

    /// Required lower 95% mean-return bound, if any.
    pub minimum_mean_reward_lower_95: Option<f64>,

    /// Required lower Wilson 95% success bound, if any.
    pub minimum_success_lower_95: Option<f64>,
}

impl EvaluationGate {
    /// Assess `summary` against every configured threshold.
    #[must_use]
    pub fn assess(self, summary: &EvaluationSummary) -> GateAssessment {
        let mut failed_requirements = Vec::new();

        if self
            .minimum_mean_reward
            .is_some_and(|minimum| summary.mean_reward < minimum)
        {
            failed_requirements.push("minimum_mean_reward");
        }
        if self
            .minimum_mean_reward_lower_95
            .is_some_and(|minimum| summary.mean_reward_interval_95.lower < minimum)
        {
            failed_requirements.push("minimum_mean_reward_lower_95");
        }
        if self
            .minimum_success_lower_95
            .is_some_and(|minimum| summary.success_interval_95.lower < minimum)
        {
            failed_requirements.push("minimum_success_lower_95");
        }

        GateAssessment {
            passed: failed_requirements.is_empty(),
            failed_requirements,
        }
    }
}

/// Result of applying a predeclared evaluation gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateAssessment {
    /// Whether every configured requirement passed.
    pub passed: bool,

    /// Stable keys for requirements that failed.
    pub failed_requirements: Vec<&'static str>,
}

/// Invalid evaluation input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationError {
    /// Aggregate statistics require at least one outcome.
    EmptyOutcomes,

    /// A fixed seed suite requires at least one entry.
    EmptySuite,

    /// Episode returns used for proof must be finite.
    NonFiniteReward {
        /// Index of the invalid episode outcome.
        index: usize,
    },
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOutcomes => formatter.write_str("evaluation requires at least one outcome"),
            Self::EmptySuite => formatter.write_str("evaluation suite requires at least one seed"),
            Self::NonFiniteReward { index } => {
                write!(
                    formatter,
                    "evaluation outcome {index} has a non-finite reward"
                )
            }
        }
    }
}

impl Error for EvaluationError {}

/// Approximate a mean interval with the normal 95% critical value.
fn mean_interval_95(outcomes: &[EpisodeOutcome], mean: f64) -> ConfidenceInterval {
    if outcomes.len() < 2 {
        return ConfidenceInterval {
            lower: mean,
            upper: mean,
        };
    }

    let squared_error_sum = outcomes
        .iter()
        .map(|outcome| (outcome.reward - mean).powi(2))
        .sum::<f64>();
    let sample_variance = squared_error_sum / (outcomes.len() - 1) as f64;
    let standard_error = (sample_variance / outcomes.len() as f64).sqrt();
    let margin = 1.959_963_984_540_054 * standard_error;

    ConfidenceInterval {
        lower: mean - margin,
        upper: mean + margin,
    }
}

/// Compute the two-sided 95% Wilson score interval for a binomial proportion.
fn wilson_interval_95(successes: usize, samples: usize) -> ConfidenceInterval {
    let z = 1.959_963_984_540_054;
    let sample_count = samples as f64;
    let proportion = successes as f64 / sample_count;
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / sample_count;
    let center = (proportion + z_squared / (2.0 * sample_count)) / denominator;
    let variance = proportion.mul_add(1.0 - proportion, z_squared / (4.0 * sample_count));
    let margin = z * (variance / sample_count).sqrt() / denominator;

    ConfidenceInterval {
        lower: (center - margin).clamp(0.0, 1.0),
        upper: (center + margin).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_reports_reward_and_wilson_confidence() {
        let outcomes = [
            EpisodeOutcome::new(1.0, 10, true),
            EpisodeOutcome::new(0.0, 20, false),
            EpisodeOutcome::new(1.0, 30, true),
            EpisodeOutcome::new(1.0, 40, true),
        ];

        let summary = EvaluationSummary::from_outcomes(&outcomes).expect("non-empty outcomes");

        assert_eq!(summary.sample_count, 4);
        assert!((summary.mean_reward - 0.75).abs() < f64::EPSILON);
        assert!((summary.mean_length - 25.0).abs() < f64::EPSILON);
        assert_eq!(summary.success_count, 3);
        assert!(summary.success_interval_95.lower < 0.75);
        assert!(summary.success_interval_95.upper > 0.75);
        assert!(summary.mean_reward_interval_95.lower < summary.mean_reward);
        assert!(summary.mean_reward_interval_95.upper > summary.mean_reward);
    }

    #[test]
    fn gate_uses_declared_point_and_lower_confidence_thresholds() {
        let outcomes = (0..100)
            .map(|index| EpisodeOutcome::new(10.0, 5, index < 95))
            .collect::<Vec<_>>();
        let summary = EvaluationSummary::from_outcomes(&outcomes).expect("non-empty outcomes");

        let passing = EvaluationGate {
            minimum_mean_reward: Some(9.0),
            minimum_mean_reward_lower_95: Some(9.0),
            minimum_success_lower_95: Some(0.85),
        };
        let failing = EvaluationGate {
            minimum_mean_reward: Some(11.0),
            ..passing
        };

        assert!(passing.assess(&summary).passed);
        let assessment = failing.assess(&summary);
        assert!(!assessment.passed);
        assert_eq!(assessment.failed_requirements, vec!["minimum_mean_reward"]);
    }

    #[test]
    fn validation_test_and_demo_suites_are_fixed_and_disjoint() {
        let seeds = crate::training::SeedConfig::from_root(7);
        let validation =
            EvaluationSuite::new(EvaluationSuiteKind::Validation, seeds.validation, 100)
                .expect("positive suite size");
        let test = EvaluationSuite::new(EvaluationSuiteKind::Test, seeds.test, 100)
            .expect("positive suite size");
        let demo = EvaluationSuite::new(EvaluationSuiteKind::Demo, seeds.demo, 8)
            .expect("positive suite size");

        assert_eq!(validation, validation.clone());
        assert!(validation
            .seeds
            .iter()
            .all(|seed| !test.seeds.contains(seed)));
        assert!(validation
            .seeds
            .iter()
            .all(|seed| !demo.seeds.contains(seed)));
        assert!(test.seeds.iter().all(|seed| !demo.seeds.contains(seed)));
        assert!(matches!(
            EvaluationSuite::new(EvaluationSuiteKind::Test, seeds.test, 0),
            Err(EvaluationError::EmptySuite)
        ));
    }
}
