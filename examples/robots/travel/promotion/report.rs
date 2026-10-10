//! Typed promotion-validation reports; final held-out roots remain separate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(not(target_arch = "wasm32"))]
use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
use crate::travel::model;
use crate::travel::{evaluation, evaluation::Score, stage::Stage};

/// Wire report from one frozen evaluator process, validated before any transition.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Report {
    /// Closed format version for validation reports.
    schema: Schema,
    /// Lowercase SHA-256 of the exact checkpoint bytes loaded by the child.
    checkpoint_sha256: String,
    /// Every prerequisite and the proposed stage, in curriculum order.
    stages: Vec<StageScores>,
}

impl Report {
    /// Evaluate immutable policy bytes without creating a learner or exploration sampler.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn evaluate(stage: Stage, bytes: Vec<u8>) -> Result<Self, Box<dyn Error>> {
        let digest = Sha256::digest(&bytes);
        let checkpoint_sha256 = format!("{digest:x}");
        let policy = model::load_policy(bytes)?;
        let stages = stage
            .through()
            .map(|stage| {
                evaluation::evaluate(stage, &policy, &seeds())
                    .map(|episodes| StageScores { stage, episodes })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            schema: Schema::VersionOne,
            checkpoint_sha256,
            stages,
        })
    }

    /// Recompute physical gates for exactly the ordered prerequisite set.
    pub(super) fn passes(&self, stage: Stage) -> bool {
        self.stages.len() == stage.through().count()
            && self
                .stages
                .iter()
                .zip(stage.through())
                .all(|(row, expected)| {
                    row.stage == expected
                        && row.episodes.iter().all(valid_measurements)
                        && evaluation::passes(row.stage, &row.episodes, &seeds())
                })
    }

    /// Bind the child report to the bytes saved from the selection policy.
    pub(super) fn matches_digest(&self, expected: &sha2::digest::Output<Sha256>) -> bool {
        self.checkpoint_sha256 == format!("{expected:x}")
    }
}

/// Fixed high-half validation roots, excluded from training, selection and the final suite.
fn seeds() -> [u64; 32] {
    std::array::from_fn(|index| u64::MAX - 1_024 - index as u64)
}

/// Reject negative or nonfinite physical values before interpreting a child report.
fn valid_measurements(score: &Score) -> bool {
    [
        score.final_distance,
        score.final_heading_error,
        score.final_speed,
    ]
    .into_iter()
    .all(|value| value.is_finite() && value >= 0.0)
}

/// The only supported promotion-report version.
#[derive(Serialize, Deserialize)]
enum Schema {
    /// Independent validation, distinct from selection and final evaluation.
    #[serde(rename = "travel-promotion-v1")]
    VersionOne,
}

/// Physical episode results for one named task distribution.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StageScores {
    /// Closed task identity.
    stage: Stage,
    /// Thirty-two ordered validation cases with independent recurrent memories.
    episodes: Vec<Score>,
}

/// Synthetic gate boundary records are isolated tests, never qualification evidence.
#[cfg(test)]
pub(super) fn fixture(stage: Stage) -> Report {
    let digest = Sha256::digest(b"isolated checkpoint fixture");
    Report {
        schema: Schema::VersionOne,
        checkpoint_sha256: format!("{digest:x}"),
        stages: stage
            .through()
            .map(|stage| StageScores {
                stage,
                episodes: seeds()
                    .into_iter()
                    .map(|seed| Score {
                        seed,
                        steps: 1_000,
                        reward: stage.minimum_return(),
                        survived: true,
                        first_arrival: std::num::NonZeroU16::new(stage.deadline()),
                        settled_actions: 100,
                        final_distance: 0.5,
                        final_heading_error: std::f32::consts::PI / 12.0,
                        final_speed: 0.5,
                    })
                    .collect(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Corrupt checkpoints fail before producing any validation cases.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn invalid_checkpoint_bytes_cannot_enter_validation() {
        assert!(Report::evaluate(Stage::Endurance, Vec::new()).is_err());
    }

    /// Validation never reuses selection, training or final evaluation roots.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn validation_roots_have_a_separate_ordered_partition() {
        let roots = seeds();
        assert_eq!(roots.first(), Some(&(u64::MAX - 1_024)));
        assert_eq!(roots.last(), Some(&(u64::MAX - 1_055)));
        assert!(roots.windows(2).all(|pair| {
            matches!(pair, [first, second] if first.checked_sub(1) == Some(*second))
        }));
        assert!(roots.iter().all(|root| *root > u64::MAX >> 1));
        assert!(roots.iter().all(|root| {
            !crate::learning::SELECTION_SEEDS.contains(root) && *root < u64::MAX - 32
        }));
    }

    /// Every proposed stage requires exactly its prefix; earlier failures deny advancement.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn every_prerequisite_case_controls_promotion() {
        for (stage, length) in Stage::ALL.into_iter().zip(1..=4) {
            let serialized = serde_json::to_vec(&fixture(stage)).expect("complete stage prefix");
            let mut report: Report = serde_json::from_slice(&serialized).expect("known stages");
            assert_eq!(report.stages.len(), length);
            assert!(report.passes(stage));
            report
                .stages
                .first_mut()
                .expect("endurance")
                .episodes
                .first_mut()
                .expect("first root")
                .settled_actions = 99;
            assert!(!report.passes(stage));
        }
        let mut report = fixture(Stage::Near);
        report.stages.swap(0, 1);
        assert!(!report.passes(Stage::Near));
        report.stages.swap(0, 1);
        report.stages.pop();
        assert!(!report.passes(Stage::Near));
        assert!(!fixture(Stage::Fast).passes(Stage::Near));
    }

    /// Ordered roots and case cardinality are required independently of physical success.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn missing_reordered_or_final_roots_cannot_pass() {
        let mut report = fixture(Stage::Endurance);
        let episodes = &mut report.stages.first_mut().expect("endurance").episodes;
        episodes.swap(0, 1);
        assert!(!report.passes(Stage::Endurance));
        let mut report = fixture(Stage::Endurance);
        report.stages.first_mut().expect("endurance").episodes.pop();
        assert!(!report.passes(Stage::Endurance));
        let mut report = fixture(Stage::Endurance);
        report
            .stages
            .first_mut()
            .expect("endurance")
            .episodes
            .first_mut()
            .expect("first root")
            .seed = u64::MAX - 1;
        assert!(!report.passes(Stage::Endurance));
    }

    /// Negative, nonfinite and beyond-gate physical measurements cannot authorize promotion.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn physical_measurements_are_checked_after_deserialization() {
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            for field in 0..3 {
                let mut report = fixture(Stage::Endurance);
                let score = report
                    .stages
                    .first_mut()
                    .expect("endurance")
                    .episodes
                    .first_mut()
                    .expect("first root");
                match field {
                    0 => score.final_distance = invalid,
                    1 => score.final_heading_error = invalid,
                    _ => score.final_speed = invalid,
                }
                assert!(!report.passes(Stage::Endurance));
            }
        }
        let mut report = fixture(Stage::Endurance);
        report
            .stages
            .first_mut()
            .expect("endurance")
            .episodes
            .first_mut()
            .expect("first root")
            .final_heading_error = 0.262;
        assert!(!report.passes(Stage::Endurance));
    }

    /// Wire versions, closed stages, mandatory fields and unknown members fail explicitly.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn malformed_report_shapes_are_rejected() {
        let baseline = serde_json::to_value(fixture(Stage::Endurance)).expect("fixture JSON");
        let round_trip: Report = serde_json::from_value(baseline.clone()).expect("valid report");
        assert!(round_trip.passes(Stage::Endurance));
        for field in ["schema", "checkpoint_sha256", "stages"] {
            let mut value = baseline.clone();
            value.as_object_mut().expect("object").remove(field);
            assert!(serde_json::from_value::<Report>(value).is_err());
        }
        for (pointer, replacement) in [
            ("/schema", serde_json::json!("travel-promotion-v2")),
            ("/stages/0/stage", serde_json::json!("unknown")),
        ] {
            let mut value = baseline.clone();
            *value.pointer_mut(pointer).expect("fixture member") = replacement;
            assert!(serde_json::from_value::<Report>(value).is_err());
        }
        for pointer in ["", "/stages/0", "/stages/0/episodes/0"] {
            let mut value = baseline.clone();
            value
                .pointer_mut(pointer)
                .expect("object")
                .as_object_mut()
                .expect("object")
                .insert("unexpected".into(), serde_json::json!(0));
            assert!(serde_json::from_value::<Report>(value).is_err());
        }
    }

    /// Every stage and score field is mandatory; zero arrival is malformed, null is a failed case.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn stage_and_score_members_are_required() {
        let baseline = serde_json::to_value(fixture(Stage::Endurance)).expect("fixture JSON");
        for (pointer, fields) in [
            ("/stages/0", vec!["stage", "episodes"]),
            (
                "/stages/0/episodes/0",
                vec![
                    "seed",
                    "steps",
                    "reward",
                    "survived",
                    "first_arrival",
                    "settled_actions",
                    "final_distance",
                    "final_heading_error",
                    "final_speed",
                ],
            ),
        ] {
            for field in fields {
                let mut value = baseline.clone();
                value
                    .pointer_mut(pointer)
                    .expect("fixture object")
                    .as_object_mut()
                    .expect("object")
                    .remove(field);
                assert!(serde_json::from_value::<Report>(value).is_err());
            }
        }
        let mut value = baseline.clone();
        *value
            .pointer_mut("/stages/0/episodes/0/first_arrival")
            .expect("arrival") = serde_json::json!(0);
        assert!(serde_json::from_value::<Report>(value).is_err());
        let mut value = baseline;
        *value
            .pointer_mut("/stages/0/episodes/0/first_arrival")
            .expect("arrival") = serde_json::Value::Null;
        let report: Report = serde_json::from_value(value).expect("null means no arrival");
        assert!(!report.passes(Stage::Endurance));
    }

    /// Digest lexical form and value must match the selected checkpoint exactly.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn checkpoint_digest_matches_without_normalizing_unrelated_values() {
        let mut report = fixture(Stage::Endurance);
        let digest = Sha256::digest(b"isolated checkpoint fixture");
        assert!(report.matches_digest(&digest));
        assert!(!report.matches_digest(&Sha256::digest(b"different weights")));
        report.checkpoint_sha256.make_ascii_uppercase();
        assert!(!report.matches_digest(&digest));
    }
}
