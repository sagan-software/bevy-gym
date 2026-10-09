//! Ordered lessons with selection scores and fixed promotion thresholds.

use std::{error::Error, num::NonZeroU16};

use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor, DroneObservation};
use bevy_gym::training::RecurrentPpoPolicy;
use serde::{Serialize, Serializer};

use super::{encode, DamageTask};
use crate::assessment::{assess, FailureTime, Outcome};
use crate::learning::{
    evaluation::{evaluate_episode, EpisodeScore},
    RecoveryBatch, SELECTION_SEEDS,
};

/// Increasing exposure to failure while retaining one actor, critic, and optimizer.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Lesson {
    /// Learn intact hover with explicit health inputs.
    Hover,
    /// Learn the front-left failure before varying its identity.
    FrontLeft,
    /// Learn every motor and both scheduled failure times.
    Scheduled,
}

/// Fresh evaluation episodes, never reused for optimizer updates.
#[derive(Serialize)]
pub(crate) struct Assessment {
    /// Expected matrix used to validate completeness before promotion.
    lesson: Lesson,
    /// Intact calm flight must remain qualified after every lesson.
    healthy: Vec<EpisodeScore>,
    /// Each lesson's complete ordered damage matrix.
    damaged: Vec<DamageTrial>,
}

/// One scheduled-failure selection case.
#[derive(Serialize)]
struct DamageTrial {
    /// Physical reset seed, excluded from training lanes.
    seed: u64,
    /// Typed actuator identity, serialized only at the report boundary.
    #[serde(serialize_with = "serialize_motor")]
    motor: DroneMotor,
    /// Intact-flight duration before failure.
    failure: FailureTime,
    /// Post-failure measurements or an explicitly separate early crash.
    outcome: Outcome,
}

impl Lesson {
    /// Stable checkpoint prefix for the selected lesson.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Hover => "hover",
            Self::FrontLeft => "front-left",
            Self::Scheduled => "scheduled",
        }
    }

    /// Start new independent lanes while leaving optimizer ownership with the caller.
    pub(crate) fn batch(
        self,
        seed: u64,
        policy: &RecurrentPpoPolicy,
    ) -> RecoveryBatch<DamageTask, 16> {
        let (make, limit): (fn() -> DamageTask, u16) = match self {
            Self::Hover => (DamageTask::hover, 500),
            Self::FrontLeft => (DamageTask::front_left, 500),
            Self::Scheduled => (DamageTask::scheduled, 750),
        };
        RecoveryBatch::with_task(
            seed,
            policy,
            make,
            encode,
            NonZeroU16::new(limit).expect("positive lesson horizon"),
        )
    }

    /// Score intact flight and the lesson's full failure matrix with fresh actor memory.
    pub(crate) fn evaluate(
        self,
        policy: &RecurrentPpoPolicy,
    ) -> Result<Assessment, Box<dyn Error>> {
        let mut healthy = Vec::with_capacity(5);
        for seed in SELECTION_SEEDS {
            healthy.push(evaluate_episode(
                seed,
                DroneHover::default(),
                controller(policy),
            )?);
        }
        let mut damaged = Vec::new();
        for (seed, motor, failure) in self.cases() {
            let outcome = assess(seed, motor, failure, controller(policy))?;
            damaged.push(DamageTrial {
                seed,
                motor,
                failure,
                outcome,
            });
        }
        Ok(Assessment {
            lesson: self,
            healthy,
            damaged,
        })
    }
    /// Produce the complete deterministic matrix without allocating a second collection.
    fn cases(self) -> impl Iterator<Item = (u64, DroneMotor, FailureTime)> {
        let motors: &'static [DroneMotor] = match self {
            Self::Hover => &[],
            Self::FrontLeft => &[DroneMotor::FrontLeft],
            Self::Scheduled => &DroneMotor::ALL,
        };
        let times: &'static [FailureTime] = match self {
            Self::Hover | Self::FrontLeft => &[FailureTime::TwoSeconds],
            Self::Scheduled => &[FailureTime::TwoSeconds, FailureTime::FiveSeconds],
        };
        SELECTION_SEEDS.into_iter().flat_map(move |seed| {
            motors.iter().copied().flat_map(move |motor| {
                times
                    .iter()
                    .copied()
                    .map(move |failure| (seed, motor, failure))
            })
        })
    }
}

impl Assessment {
    /// Require retained healthy flight and successful recovery in every requested case.
    pub(crate) fn passes(&self) -> bool {
        // Every intact episode must survive and remain close to the target.
        self.healthy.len() == 5
            && self
                .healthy
                .iter()
                .zip(SELECTION_SEEDS)
                .all(|(score, seed)| {
                    score.seed == seed
                        && score.survived
                        && score.steps == 500
                        && score.reward >= 400.0
                        && score.final_distance <= 0.5
                })
            && self.damaged.len() == self.lesson.cases().count()
            && self
                .damaged
                .iter()
                .zip(self.lesson.cases())
                .all(|(trial, expected)| {
                    (trial.seed, trial.motor, trial.failure) == expected
                        && match trial.outcome {
                            Outcome::Survived(score) => {
                                score.steps == 500
                                    && score.reward >= 400.0
                                    && score.final_distance_m <= 0.5
                                    && score.minimum_height_m >= 1.0
                                    && score.peak_tilt_radians <= std::f32::consts::FRAC_PI_4
                            }
                            Outcome::ApproachEnded(_) | Outcome::Crashed(_) => false,
                        }
                })
    }
}

/// Decode each command before retaining the corresponding recurrent memory.
fn controller(
    policy: &RecurrentPpoPolicy,
) -> impl FnMut(DroneObservation) -> Result<DroneAction, Box<dyn Error>> + '_ {
    let mut memory = policy.initial_memory();
    move |observation| {
        let output = policy.mean_action(&encode(observation), &memory)?;
        let action = crate::learning::decode_action(&output.action)?;
        memory = output.next_memory;
        Ok(action)
    }
}

/// Serialize named motors without exposing unvalidated strings inside the task.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "Serde serialize_with hooks receive a field reference."
)]
fn serialize_motor<S: Serializer>(motor: &DroneMotor, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(match motor {
        DroneMotor::FrontLeft => "front_left",
        DroneMotor::FrontRight => "front_right",
        DroneMotor::RearRight => "rear_right",
        DroneMotor::RearLeft => "rear_left",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::Measurements;

    /// Complete selection matrix exactly at every inclusive threshold.
    fn boundary(lesson: Lesson) -> Assessment {
        let healthy = SELECTION_SEEDS
            .into_iter()
            .map(|seed| EpisodeScore {
                seed,
                steps: 500,
                reward: 400.0,
                final_distance: 0.5,
                survived: true,
            })
            .collect();
        let damaged = lesson
            .cases()
            .map(|(seed, motor, failure)| DamageTrial {
                seed,
                motor,
                failure,
                outcome: Outcome::Survived(Measurements {
                    steps: 500,
                    reward: 400.0,
                    final_distance_m: 0.5,
                    minimum_height_m: 1.0,
                    peak_tilt_radians: std::f32::consts::FRAC_PI_4,
                    final_body_yaw_rate_radians_per_second: 20.0,
                }),
            })
            .collect();
        Assessment {
            lesson,
            healthy,
            damaged,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn promotion_requires_complete_ordered_matrices_and_retained_healthy_flight() {
        for (lesson, count) in [
            (Lesson::Hover, 0),
            (Lesson::FrontLeft, 5),
            (Lesson::Scheduled, 40),
        ] {
            let mut assessment = boundary(lesson);
            assert!(assessment.passes());
            assert_eq!(assessment.damaged.len(), count);
            assessment.healthy[0].survived = false;
            assert!(!assessment.passes());
            assessment = boundary(lesson);
            assessment.healthy.swap(0, 1);
            assert!(!assessment.passes());
            assessment = boundary(lesson);
            assessment.healthy.pop();
            assert!(!assessment.passes());
            if count != 0 {
                assessment = boundary(lesson);
                assessment.damaged.pop();
                assert!(!assessment.passes());
                assessment = boundary(lesson);
                assessment.damaged.swap(0, 1);
                assert!(!assessment.passes());
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn every_damage_threshold_and_early_or_late_crash_blocks_promotion() {
        for invalid in 0..7 {
            let mut assessment = boundary(Lesson::Scheduled);
            let Outcome::Survived(mut score) = assessment.damaged[0].outcome else {
                panic!("survival fixture");
            };
            match invalid {
                0 => score.steps = 499,
                1 => score.reward = 399.999,
                2 => score.final_distance_m = 0.501,
                3 => score.minimum_height_m = 0.999,
                4 => score.peak_tilt_radians = std::f32::consts::FRAC_PI_4 + 0.001,
                _ => {}
            }
            assessment.damaged[0].outcome = match invalid {
                5 => Outcome::ApproachEnded(score),
                6 => Outcome::Crashed(score),
                _ => Outcome::Survived(score),
            };
            assert!(!assessment.passes(), "invalid case {invalid}");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn healthy_thresholds_and_short_episodes_block_promotion() {
        for invalid in 0..3 {
            let mut assessment = boundary(Lesson::Hover);
            match invalid {
                0 => assessment.healthy[0].reward = 399.999,
                1 => assessment.healthy[0].final_distance = 0.501,
                _ => assessment.healthy[0].steps = 499,
            }
            assert!(!assessment.passes());
        }
    }
}
