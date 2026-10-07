//! Typed curriculum qualification gates and missing-evidence handling.

use serde::{Deserialize, Serialize};

/// Closed stage vocabulary used by durable qualification artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QualificationStage {
    /// Single-bunny food lesson.
    Forage,
    /// Single-bunny ephemeral-food speed lesson.
    Sprint,
    /// Single-bunny gorge bridge-crossing lesson.
    Gorge,
    /// Single-bunny food-and-water lesson.
    Survival,
    /// Single-bunny weather-shelter lesson.
    Shelter,
    /// Shared-policy resource competition lesson.
    Competition,
    /// Co-adapting bunny and fox lesson.
    PredatorPrey,
    /// Full procedural ecosystem lesson.
    Obstacles,
}

impl QualificationStage {
    /// Return the stable stage key used by commands and artifacts.
    pub(super) const fn as_key(self) -> &'static str {
        // Match evaluation artifacts to their originating curriculum stage.
        match self {
            Self::Forage => "forage",
            Self::Sprint => "sprint",
            Self::Gorge => "gorge",
            Self::Survival => "survival",
            Self::Shelter => "shelter",
            Self::Competition => "competition",
            Self::PredatorPrey => "predator-prey",
            Self::Obstacles => "obstacles",
        }
    }
}

/// Metrics needed to classify one frozen checkpoint suite.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct StageEvidence {
    /// Mean bunny healthy lifetime in simulated seconds.
    pub(super) bunny_lifetime: f64,
    /// Mean fox healthy lifetime in simulated seconds.
    pub(super) fox_lifetime: f64,
    /// Fraction of bunnies that ate food.
    pub(super) bunny_food_fraction: f64,
    /// Mean seconds to first resource contact among successful bunnies.
    pub(super) bunny_mean_time_to_contact: f64,
    /// Fraction of bunnies that both ate and drank.
    pub(super) bunny_food_and_water_fraction: f64,
    /// Fraction of bunnies that sheltered before critical exposure.
    pub(super) bunny_shelter_fraction: f64,
    /// Number of stable bunny identities that both ate and drank.
    pub(super) bunny_complete_resource_consumers: u32,
    /// Largest stable bunny identity lifetime advantage.
    pub(super) bunny_identity_advantage: f64,
    /// Fraction of foxes that consumed prey and water.
    pub(super) fox_food_and_water_fraction: f64,
    /// Fraction of foxes that sheltered before critical exposure.
    pub(super) fox_shelter_fraction: f64,
    /// Median turn command while food appeared left.
    pub(super) bunny_left_food_turn: f64,
    /// Median turn command while food appeared right.
    pub(super) bunny_right_food_turn: f64,
    /// Bunny eating fraction after food perception was ablated.
    pub(super) bunny_ablated_food_fraction: f64,
    /// Ecosystem fraction containing at least one predation event.
    pub(super) predation_episode_fraction: f64,
    /// Agent-agent contact observations.
    pub(super) collision_contacts: u64,
    /// Agent-agent contacts that displaced an actor.
    pub(super) contact_displacements: u64,
    /// Solid contacts beyond the post-solver penetration tolerance.
    pub(super) unresolved_solid_penetrations: u64,
    /// Thorn damage normalized by evaluated agent-minutes.
    pub(super) thorn_damage_per_agent_minute: f64,
    /// Deaths caused by exposure.
    pub(super) exposure_deaths: u32,
}

/// One earlier-stage suite result supplied to a later-stage decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RetentionEvidence {
    /// Frozen earlier-stage environment.
    pub(super) stage: QualificationStage,
    /// Whether the candidate stayed above that suite's fixed floor.
    pub(super) passed: bool,
}

/// Complete evidence boundary for one stage qualification decision.
pub(super) struct QualificationInput<'a> {
    /// Stage being classified.
    pub(super) stage: QualificationStage,
    /// Held-out candidate metrics.
    pub(super) candidate: StageEvidence,
    /// Held-out step-zero metrics from the same stage profile.
    pub(super) step_zero: StageEvidence,
    /// Equal-budget from-scratch metrics when the stage requires a control.
    pub(super) from_scratch: Option<StageEvidence>,
    /// Worst current-to-historical cross-play score ratio.
    pub(super) minimum_cross_play_ratio: Option<f64>,
    /// Frozen earlier-stage retention decisions.
    pub(super) retention: &'a [RetentionEvidence],
}

/// One observable qualification comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct GateResult {
    /// Stable machine-facing gate name.
    pub(super) gate: String,
    /// Measured value, or absent when required evidence was not supplied.
    pub(super) observed: Option<GateObservation>,
    /// Stable human-readable comparison applied to the measured value.
    pub(super) requirement: String,
    /// Whether the supplied evidence satisfies the gate.
    pub(super) passed: bool,
}

/// Exact observed value without projecting integer counts through a float.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum GateObservation {
    /// Finite scalar fraction, duration, or rate.
    Scalar(f64),
    /// Exact event or identity count.
    Count(u64),
    /// Boolean retention status.
    Status(bool),
}

/// Complete stage decision with every constituent gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct QualificationReport {
    /// Qualified curriculum stage.
    pub(super) stage: String,
    /// Whether every required gate passed.
    pub(super) passed: bool,
    /// Ordered gate decisions.
    pub(super) gates: Vec<GateResult>,
}

/// Apply the fixed stage contract without treating missing evidence as success.
#[expect(
    clippy::too_many_lines,
    reason = "the ordered branches are the complete curriculum qualification matrix"
)]
pub(super) fn qualify(input: QualificationInput<'_>) -> QualificationReport {
    let mut gates = Vec::new();
    match input.stage {
        QualificationStage::Forage => {
            push_gate(
                &mut gates,
                "food_success_fraction",
                input.candidate.bunny_food_fraction,
                ">= 0.90",
                input.candidate.bunny_food_fraction >= 0.90,
            );
            push_gate(
                &mut gates,
                "left_food_turn",
                input.candidate.bunny_left_food_turn,
                "> 0",
                input.candidate.bunny_left_food_turn > 0.0,
            );
            push_gate(
                &mut gates,
                "right_food_turn",
                input.candidate.bunny_right_food_turn,
                "< 0",
                input.candidate.bunny_right_food_turn < 0.0,
            );
            let ablation_drop =
                input.candidate.bunny_food_fraction - input.candidate.bunny_ablated_food_fraction;
            push_gate(
                &mut gates,
                "food_ablation_drop",
                ablation_drop,
                ">= 0.50",
                input.candidate.bunny_food_fraction
                    >= input.candidate.bunny_ablated_food_fraction + 0.50,
            );
        }
        QualificationStage::Sprint => {
            push_gate(
                &mut gates,
                "food_success_fraction",
                input.candidate.bunny_food_fraction,
                ">= 0.80",
                input.candidate.bunny_food_fraction >= 0.80,
            );
            push_gate(
                &mut gates,
                "mean_time_to_contact",
                input.candidate.bunny_mean_time_to_contact,
                "<= 4.00 seconds",
                input.candidate.bunny_mean_time_to_contact <= 4.0,
            );
            push_retention_gate(&mut gates, input.retention, QualificationStage::Forage);
        }
        QualificationStage::Gorge => {
            // Every initial target is on the opposite bank, so eating proves a safe crossing.
            push_gate(
                &mut gates,
                "opposite_bank_food_fraction",
                input.candidate.bunny_food_fraction,
                ">= 0.80",
                input.candidate.bunny_food_fraction >= 0.80,
            );
            push_gate(
                &mut gates,
                "mean_time_to_contact",
                input.candidate.bunny_mean_time_to_contact,
                "<= 5.00 seconds",
                input.candidate.bunny_mean_time_to_contact <= 5.0,
            );
            push_retention_gate(&mut gates, input.retention, QualificationStage::Sprint);
        }
        QualificationStage::Survival => {
            push_gate(
                &mut gates,
                "food_and_water_fraction",
                input.candidate.bunny_food_and_water_fraction,
                ">= 0.80",
                input.candidate.bunny_food_and_water_fraction >= 0.80,
            );
            push_relative_gate(
                &mut gates,
                "bunny_lifetime_gain",
                input.candidate.bunny_lifetime,
                input.step_zero.bunny_lifetime,
                1.30,
                ">= 1.30 * step zero",
            );
            push_retention_gate(&mut gates, input.retention, QualificationStage::Gorge);
        }
        QualificationStage::Shelter => {
            push_gate(
                &mut gates,
                "early_shelter_fraction",
                input.candidate.bunny_shelter_fraction,
                ">= 0.80",
                input.candidate.bunny_shelter_fraction >= 0.80,
            );
            let exposure_passed = input.candidate.exposure_deaths.saturating_mul(2)
                <= input.step_zero.exposure_deaths;
            gates.push(GateResult {
                gate: "exposure_deaths".to_owned(),
                observed: Some(GateObservation::Count(u64::from(
                    input.candidate.exposure_deaths,
                ))),
                requirement: "<= 0.50 * step zero".to_owned(),
                passed: exposure_passed,
            });
            push_retention_gate(&mut gates, input.retention, QualificationStage::Survival);
        }
        QualificationStage::Competition => {
            push_relative_gate(
                &mut gates,
                "bunny_lifetime_gain",
                input.candidate.bunny_lifetime,
                input.step_zero.bunny_lifetime,
                1.15,
                ">= 1.15 * step zero",
            );
            push_count_gate(
                &mut gates,
                "complete_resource_consumers",
                u64::from(input.candidate.bunny_complete_resource_consumers),
                ">= 4",
                input.candidate.bunny_complete_resource_consumers >= 4,
            );
            push_gate(
                &mut gates,
                "identity_lifetime_advantage",
                input.candidate.bunny_identity_advantage,
                "<= 0.20",
                input.candidate.bunny_identity_advantage <= 0.20,
            );
            push_count_gate(
                &mut gates,
                "collision_contacts",
                input.candidate.collision_contacts,
                "> 0",
                input.candidate.collision_contacts > 0,
            );
            push_count_gate(
                &mut gates,
                "contact_displacements",
                input.candidate.contact_displacements,
                "> 0",
                input.candidate.contact_displacements > 0,
            );
            push_retention_gate(&mut gates, input.retention, QualificationStage::Shelter);
        }
        QualificationStage::PredatorPrey => {
            push_relative_gate(
                &mut gates,
                "bunny_lifetime_gain",
                input.candidate.bunny_lifetime,
                input.step_zero.bunny_lifetime,
                1.15,
                ">= 1.15 * step zero",
            );
            push_relative_gate(
                &mut gates,
                "fox_lifetime_gain",
                input.candidate.fox_lifetime,
                input.step_zero.fox_lifetime,
                1.15,
                ">= 1.15 * step zero",
            );
            push_positive_gate(
                &mut gates,
                "bunny_food_and_water",
                input.candidate.bunny_food_and_water_fraction,
            );
            push_positive_gate(
                &mut gates,
                "bunny_shelter",
                input.candidate.bunny_shelter_fraction,
            );
            push_positive_gate(
                &mut gates,
                "fox_prey_and_water",
                input.candidate.fox_food_and_water_fraction,
            );
            push_positive_gate(
                &mut gates,
                "fox_shelter",
                input.candidate.fox_shelter_fraction,
            );
            push_gate(
                &mut gates,
                "predation_episode_fraction",
                input.candidate.predation_episode_fraction,
                "in 0.10..=0.90",
                (0.10..=0.90).contains(&input.candidate.predation_episode_fraction),
            );
            push_optional_gate(
                &mut gates,
                "cross_play_ratio",
                input.minimum_cross_play_ratio,
                ">= 0.70",
                |ratio| ratio >= 0.70,
            );
            push_retention_gate(&mut gates, input.retention, QualificationStage::Competition);
            push_retention_gate(&mut gates, input.retention, QualificationStage::Shelter);
        }
        QualificationStage::Obstacles => {
            push_count_gate(
                &mut gates,
                "solid_penetrations",
                input.candidate.unresolved_solid_penetrations,
                "== 0",
                input.candidate.unresolved_solid_penetrations == 0,
            );
            push_relative_gate_at_most(
                &mut gates,
                "thorn_damage",
                input.candidate.thorn_damage_per_agent_minute,
                input.step_zero.thorn_damage_per_agent_minute,
                0.80,
                "<= 0.80 * step zero",
            );
            push_optional_gate(
                &mut gates,
                "bunny_from_scratch_control",
                input.from_scratch.map(|evidence| evidence.bunny_lifetime),
                "> from scratch",
                |lifetime| lifetime < input.candidate.bunny_lifetime,
            );
            push_optional_gate(
                &mut gates,
                "fox_from_scratch_control",
                input.from_scratch.map(|evidence| evidence.fox_lifetime),
                "> from scratch",
                |lifetime| lifetime < input.candidate.fox_lifetime,
            );
            for stage in [
                QualificationStage::Forage,
                QualificationStage::Sprint,
                QualificationStage::Gorge,
                QualificationStage::Survival,
                QualificationStage::Shelter,
                QualificationStage::Competition,
                QualificationStage::PredatorPrey,
            ] {
                push_retention_gate(&mut gates, input.retention, stage);
            }
        }
    }
    QualificationReport {
        stage: input.stage.as_key().to_owned(),
        passed: gates.iter().all(|gate| gate.passed),
        gates,
    }
}

/// Record one finite scalar comparison.
fn push_gate(
    gates: &mut Vec<GateResult>,
    gate: &'static str,
    observed: f64,
    requirement: &'static str,
    passed: bool,
) {
    gates.push(GateResult {
        gate: gate.to_owned(),
        observed: observed
            .is_finite()
            .then_some(GateObservation::Scalar(observed)),
        requirement: requirement.to_owned(),
        passed: observed.is_finite() && passed,
    });
}

/// Record one exact count comparison.
fn push_count_gate(
    gates: &mut Vec<GateResult>,
    gate: &'static str,
    observed: u64,
    requirement: &'static str,
    passed: bool,
) {
    gates.push(GateResult {
        gate: gate.to_owned(),
        observed: Some(GateObservation::Count(observed)),
        requirement: requirement.to_owned(),
        passed,
    });
}

/// Record one strictly positive fraction or count.
fn push_positive_gate(gates: &mut Vec<GateResult>, gate: &'static str, observed: f64) {
    push_gate(gates, gate, observed, "> 0", observed > 0.0);
}

/// Record a minimum multiplicative improvement from step zero.
fn push_relative_gate(
    gates: &mut Vec<GateResult>,
    gate: &'static str,
    observed: f64,
    baseline: f64,
    factor: f64,
    requirement: &'static str,
) {
    push_gate(
        gates,
        gate,
        observed,
        requirement,
        baseline.is_finite() && baseline > 0.0 && observed >= baseline * factor,
    );
}

/// Record a maximum multiplicative value relative to step zero.
fn push_relative_gate_at_most(
    gates: &mut Vec<GateResult>,
    gate: &'static str,
    observed: f64,
    baseline: f64,
    factor: f64,
    requirement: &'static str,
) {
    push_gate(
        gates,
        gate,
        observed,
        requirement,
        baseline.is_finite() && baseline > 0.0 && observed <= baseline * factor,
    );
}

/// Record optional evidence and fail when it was not supplied.
fn push_optional_gate(
    gates: &mut Vec<GateResult>,
    gate: &'static str,
    observed: Option<f64>,
    requirement: &'static str,
    predicate: impl FnOnce(f64) -> bool,
) {
    let gate_passed = observed.is_some_and(|value| value.is_finite() && predicate(value));
    gates.push(GateResult {
        gate: gate.to_owned(),
        observed: observed
            .filter(|value| value.is_finite())
            .map(GateObservation::Scalar),
        requirement: requirement.to_owned(),
        passed: gate_passed,
    });
}

/// Record one required frozen earlier-stage result and reject missing stages.
fn push_retention_gate(
    gates: &mut Vec<GateResult>,
    retention: &[RetentionEvidence],
    stage: QualificationStage,
) {
    let gate = match stage {
        QualificationStage::Forage => "retention_forage",
        QualificationStage::Sprint => "retention_sprint",
        QualificationStage::Gorge => "retention_gorge",
        QualificationStage::Survival => "retention_survival",
        QualificationStage::Shelter => "retention_shelter",
        QualificationStage::Competition => "retention_competition",
        QualificationStage::PredatorPrey => "retention_predator_prey",
        QualificationStage::Obstacles => "retention_obstacles",
    };
    let observed = retention
        .iter()
        .find(|evidence| evidence.stage == stage)
        .map(|evidence| evidence.passed);
    gates.push(GateResult {
        gate: gate.to_owned(),
        observed: observed.map(GateObservation::Status),
        requirement: "== pass".to_owned(),
        passed: observed == Some(true),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-agent lessons must use their declared behavior, gain, and retention edges.
    #[test]
    fn single_agent_stage_boundaries_match_the_contract() {
        // Exercise every single-agent boundary and its direct retention edge.
        let forage = qualify(QualificationInput {
            stage: QualificationStage::Forage,
            candidate: StageEvidence {
                bunny_food_fraction: 0.90,
                bunny_left_food_turn: 0.01,
                bunny_right_food_turn: -0.01,
                bunny_ablated_food_fraction: 0.40,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence::default(),
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[],
        });
        assert!(forage.passed);

        let sprint = qualify(QualificationInput {
            stage: QualificationStage::Sprint,
            candidate: StageEvidence {
                bunny_food_fraction: 0.80,
                bunny_mean_time_to_contact: 4.0,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence::default(),
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[RetentionEvidence {
                stage: QualificationStage::Forage,
                passed: true,
            }],
        });
        assert!(sprint.passed);

        let gorge = qualify(QualificationInput {
            stage: QualificationStage::Gorge,
            candidate: StageEvidence {
                bunny_food_fraction: 0.80,
                bunny_mean_time_to_contact: 5.0,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence::default(),
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[RetentionEvidence {
                stage: QualificationStage::Sprint,
                passed: true,
            }],
        });
        assert!(gorge.passed);

        let survival = qualify(QualificationInput {
            stage: QualificationStage::Survival,
            candidate: StageEvidence {
                bunny_lifetime: 13.0,
                bunny_food_and_water_fraction: 0.80,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence {
                bunny_lifetime: 10.0,
                ..StageEvidence::default()
            },
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[RetentionEvidence {
                stage: QualificationStage::Gorge,
                passed: true,
            }],
        });
        assert!(survival.passed);

        let shelter = qualify(QualificationInput {
            stage: QualificationStage::Shelter,
            candidate: StageEvidence {
                bunny_shelter_fraction: 0.80,
                exposure_deaths: 5,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence {
                exposure_deaths: 10,
                ..StageEvidence::default()
            },
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[RetentionEvidence {
                stage: QualificationStage::Survival,
                passed: true,
            }],
        });
        assert!(shelter.passed);
    }

    /// A complete Competition pass must include improvement, fairness, contact, and retention.
    #[test]
    fn competition_requires_every_declared_gate() {
        let candidate = StageEvidence {
            bunny_lifetime: 11.5,
            bunny_complete_resource_consumers: 4,
            bunny_identity_advantage: 0.20,
            collision_contacts: 1,
            contact_displacements: 1,
            ..StageEvidence::default()
        };
        let report = qualify(QualificationInput {
            stage: QualificationStage::Competition,
            candidate,
            step_zero: StageEvidence {
                bunny_lifetime: 10.0,
                ..StageEvidence::default()
            },
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[RetentionEvidence {
                stage: QualificationStage::Shelter,
                passed: true,
            }],
        });

        assert!(report.passed);

        let failed = qualify(QualificationInput {
            stage: QualificationStage::Competition,
            candidate: StageEvidence {
                contact_displacements: 0,
                ..candidate
            },
            step_zero: StageEvidence {
                bunny_lifetime: 10.0,
                ..StageEvidence::default()
            },
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[RetentionEvidence {
                stage: QualificationStage::Shelter,
                passed: true,
            }],
        });
        assert!(!failed.passed);
    }

    /// Predator qualification must fail closed when cross-play evidence is absent.
    #[test]
    fn predator_requires_cross_play_and_both_retention_suites() {
        let report = qualify(QualificationInput {
            stage: QualificationStage::PredatorPrey,
            candidate: StageEvidence {
                bunny_lifetime: 11.5,
                fox_lifetime: 11.5,
                bunny_food_and_water_fraction: 0.5,
                bunny_shelter_fraction: 0.5,
                fox_food_and_water_fraction: 0.5,
                fox_shelter_fraction: 0.5,
                predation_episode_fraction: 0.5,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence {
                bunny_lifetime: 10.0,
                fox_lifetime: 10.0,
                ..StageEvidence::default()
            },
            from_scratch: None,
            minimum_cross_play_ratio: None,
            retention: &[
                RetentionEvidence {
                    stage: QualificationStage::Competition,
                    passed: true,
                },
                RetentionEvidence {
                    stage: QualificationStage::Shelter,
                    passed: true,
                },
            ],
        });

        assert!(!report.passed);
        assert!(report
            .gates
            .iter()
            .any(|gate| gate.gate == "cross_play_ratio" && !gate.passed));
    }

    /// Obstacles qualification needs both scratch controls and every earlier suite.
    #[test]
    fn obstacles_requires_scratch_controls_and_all_retention() {
        let retention = [
            QualificationStage::Forage,
            QualificationStage::Sprint,
            QualificationStage::Gorge,
            QualificationStage::Survival,
            QualificationStage::Shelter,
            QualificationStage::Competition,
            QualificationStage::PredatorPrey,
        ]
        .map(|stage| RetentionEvidence {
            stage,
            passed: true,
        });
        let report = qualify(QualificationInput {
            stage: QualificationStage::Obstacles,
            candidate: StageEvidence {
                bunny_lifetime: 11.0,
                fox_lifetime: 12.0,
                thorn_damage_per_agent_minute: 0.8,
                ..StageEvidence::default()
            },
            step_zero: StageEvidence {
                thorn_damage_per_agent_minute: 1.0,
                ..StageEvidence::default()
            },
            from_scratch: Some(StageEvidence {
                bunny_lifetime: 10.0,
                fox_lifetime: 11.0,
                ..StageEvidence::default()
            }),
            minimum_cross_play_ratio: None,
            retention: &retention,
        });

        assert!(report.passed);
    }
}
