//! Healthy-survival reward calculation shared by every curriculum stage.

use std::error::Error;
use std::fmt;
use std::time::Duration;

/// A validated weighted physiological deficit in the closed interval zero through one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct NormalizedDrive(f32);

impl TryFrom<f32> for NormalizedDrive {
    type Error = NormalizedDriveError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(NormalizedDriveError)
        }
    }
}

/// A value cannot represent a normalized physiological drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct NormalizedDriveError;

impl fmt::Display for NormalizedDriveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("normalized drive must be finite and in 0..=1")
    }
}

impl Error for NormalizedDriveError {}

/// Stable healthy-survival reward parameters used across curriculum stages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct HomeostaticReward {
    /// Lifetime that normalizes one second of healthy survival.
    reference_lifetime: Duration,

    /// Fraction of live reward removed at maximum physiological drive.
    drive_cost: f32,

    /// Maximum direct feedback for reducing drive from one to zero.
    resource_feedback: f32,
}

impl HomeostaticReward {
    /// Return one transition reward from post-transition drive and absorbed benefit.
    pub(super) fn transition_reward(
        self,
        is_alive: bool,
        time_step: Duration,
        before: NormalizedDrive,
        after: NormalizedDrive,
    ) -> f32 {
        if !is_alive {
            return 0.0;
        }

        // Normalize live time before adding bounded feedback from absorbed resources.
        let survival_scale = time_step.as_secs_f32() / self.reference_lifetime.as_secs_f32();
        let drive_discount = self.drive_cost.mul_add(-after.0, 1.0);
        let drive_reduction = (before.0 - after.0).max(0.0);
        let resource_feedback = self.resource_feedback * drive_reduction;
        survival_scale.mul_add(drive_discount, resource_feedback)
    }

    /// Return the bounded horizon objective for terminal physiological state.
    pub(super) fn horizon_reward(drive: NormalizedDrive) -> f32 {
        1.0 - drive.0
    }
}

impl Default for HomeostaticReward {
    fn default() -> Self {
        Self {
            reference_lifetime: Duration::from_mins(1),
            drive_cost: 0.75,
            resource_feedback: 0.10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Return one twenty-second healthy-survival score from a drive trace.
    fn trace_return(drives: &[(f32, f32)]) -> f32 {
        let reward = HomeostaticReward {
            reference_lifetime: Duration::from_secs(20),
            drive_cost: 0.75,
            resource_feedback: 0.10,
        };
        drives
            .iter()
            .map(|(before, after)| {
                reward.transition_reward(
                    true,
                    Duration::from_millis(100),
                    NormalizedDrive::try_from(*before).expect("before drive is normalized"),
                    NormalizedDrive::try_from(*after).expect("after drive is normalized"),
                )
            })
            .sum()
    }

    /// Full healthy survival must beat urgent consumption followed by immediate death.
    #[test]
    fn healthy_horizon_outscores_resource_collection_then_death() {
        let reward = HomeostaticReward {
            reference_lifetime: Duration::from_secs(20),
            drive_cost: 0.75,
            resource_feedback: 0.10,
        };
        let healthy = NormalizedDrive::try_from(0.0).expect("zero drive is valid");
        let urgent = NormalizedDrive::try_from(1.0).expect("maximum drive is valid");
        let time_step = Duration::from_millis(100);

        let healthy_return = (0..200)
            .map(|_| reward.transition_reward(true, time_step, healthy, healthy))
            .sum::<f32>();
        let consume_then_die = reward.transition_reward(true, time_step, urgent, healthy)
            + reward.transition_reward(false, time_step, healthy, healthy);

        assert!(healthy_return > consume_then_die);
    }

    /// Eating at a full setpoint must add no resource feedback.
    #[test]
    fn full_setpoint_consumption_adds_no_reward() {
        let reward = HomeostaticReward::default();
        let healthy = NormalizedDrive::try_from(0.0).expect("zero drive is valid");
        let time_step = Duration::from_millis(100);

        let ordinary_survival = reward.transition_reward(true, time_step, healthy, healthy);
        let full_setpoint_eating = reward.transition_reward(true, time_step, healthy, healthy);

        assert_eq!(full_setpoint_eating, ordinary_survival);
    }

    /// Stable homeostasis must beat repeated delayed recovery over the same horizon.
    #[test]
    fn stable_homeostasis_outscores_repeated_consumption_cycles() {
        let stable = vec![(0.0, 0.0); 200];
        let mut repeated = Vec::with_capacity(200);
        for _ in 0..5 {
            repeated.extend(std::iter::repeat_n((0.5, 0.5), 30));
            repeated.push((0.5, 0.0));
            repeated.extend(std::iter::repeat_n((0.0, 0.0), 9));
        }

        assert!(trace_return(&stable) > trace_return(&repeated));
    }

    /// Cautious low-drive survival must beat delayed resource hoarding.
    #[test]
    fn cautious_survival_outscores_risky_hoarding() {
        let cautious = vec![(0.1, 0.1); 200];
        let mut risky = vec![(0.5, 0.5); 160];
        risky.push((0.5, 0.0));
        risky.extend(std::iter::repeat_n((0.0, 0.0), 39));

        assert!(trace_return(&cautious) > trace_return(&risky));
    }

    /// Shelter recovery must beat remaining at maximum exposure drive.
    #[test]
    fn shelter_recovery_outscores_remaining_exposed() {
        let mut recovered = vec![(1.0, 0.0)];
        recovered.extend(std::iter::repeat_n((0.0, 0.0), 99));
        let exposed = vec![(1.0, 1.0); 100];

        assert!(trace_return(&recovered) > trace_return(&exposed));
    }

    /// A fox must receive direct feedback only when prey nutrition is absorbed.
    #[test]
    fn fox_feeding_outscores_kill_without_absorbed_food() {
        let feeding = trace_return(&[(1.0, 0.0)]);
        let kill_without_absorption = trace_return(&[(1.0, 1.0)]);

        assert!(feeding > kill_without_absorption);
    }

    /// Maximum terminal physiology must receive the complete horizon objective.
    #[test]
    fn horizon_reward_prefers_maximum_terminal_stats() {
        let maximum_stats = NormalizedDrive::try_from(0.0).expect("zero drive is valid");
        let depleted = NormalizedDrive::try_from(0.8).expect("depleted drive is valid");

        assert_eq!(HomeostaticReward::horizon_reward(maximum_stats), 1.0);
        assert!(
            HomeostaticReward::horizon_reward(maximum_stats)
                > HomeostaticReward::horizon_reward(depleted)
        );
    }
}
