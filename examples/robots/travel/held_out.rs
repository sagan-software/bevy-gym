//! Fixed held-out travel evaluation bound to the exact loaded checkpoint bytes.

use sha2::{Digest, Sha256};
use std::{error::Error, path::Path};

use super::{evaluation, model, stage::Stage};

/// Ordered final roots, disjoint from low-half training roots and selection roots.
fn seeds() -> [u64; 32] {
    std::array::from_fn(|index| u64::MAX - index as u64 - 1)
}

/// Run all stages without updates and report failures before returning a nonzero exit.
pub(crate) fn run(checkpoint: &Path) -> Result<(), Box<dyn Error>> {
    // Hash the same owned bytes moved into loading; a later file edit cannot change this evaluation.
    // SHA-256 API: https://docs.rs/sha2/0.10.9/sha2/; this report uses lowercase hex.
    let bytes = std::fs::read(checkpoint)?;
    let digest = Sha256::digest(&bytes);
    let checkpoint_sha256 = format!("{digest:x}");
    let policy = model::load_policy(bytes)?;
    let mut stages = Vec::with_capacity(Stage::ALL.len());
    for stage in Stage::ALL {
        let episodes = evaluation::evaluate(stage, &policy, &seeds())?;
        stages.push(StageEvaluation { stage, episodes });
    }
    let passed = passes(&stages);
    // Serialize only at the CLI boundary; each physical measurement retains its typed source.
    let rows: Vec<_> = stages
        .iter()
        .map(|row| {
            serde_json::json!({"stage":row.stage.name(),
            "evaluation":label(evaluation::passes(row.stage, &row.episodes, &seeds())),
            "episodes":row.episodes})
        })
        .collect();
    let report = serde_json::json!({"mode":"held-out-evaluation", "suite":"travel-v1",
        "checkpoint":checkpoint, "checkpoint_sha256":checkpoint_sha256,
        "evaluation":label(passed), "qualification":"not established by evaluation alone",
        "stages":rows});
    println!("{report}");
    require_pass(passed)
}

/// One stage's physical outcomes; success is derived from these values.
struct StageEvaluation {
    /// Closed task distribution evaluated for these episodes.
    stage: Stage,
    /// Frozen episodes in the fixed final-root order.
    episodes: Vec<evaluation::Score>,
}

/// Require every named stage exactly once in curriculum order before applying episode gates.
fn passes(stages: &[StageEvaluation]) -> bool {
    stages.len() == Stage::ALL.len()
        && stages.iter().zip(Stage::ALL).all(|(row, expected)| {
            row.stage == expected && evaluation::passes(row.stage, &row.episodes, &seeds())
        })
}

/// Spell the closed evaluation outcome at the JSON boundary.
const fn label(passed: bool) -> &'static str {
    if passed {
        "passed"
    } else {
        "failed"
    }
}

/// Preserve a complete failed report while signaling failure to command-line callers.
fn require_pass(passed: bool) -> Result<(), Box<dyn Error>> {
    if passed {
        Ok(())
    } else {
        Err("Held-out travel evaluation failed; checkpoint is not qualified.".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU16;

    /// Final roots occupy the excluded high half and retain their documented order.
    #[test]
    fn final_roots_are_ordered_unique_and_excluded_from_selection() {
        let roots = seeds();
        assert_eq!(roots.first(), Some(&(u64::MAX - 1)));
        assert_eq!(roots.last(), Some(&(u64::MAX - 32)));
        assert!(roots
            .windows(2)
            .all(|pair| matches!(pair, [first, second] if first > second)));
        assert!(roots.iter().all(|root| *root > u64::MAX >> 1));
        assert!(roots
            .iter()
            .all(|root| !crate::learning::SELECTION_SEEDS.contains(root)));
    }

    /// Synthetic boundary scores test aggregation only, never learned-policy qualification.
    #[test]
    fn every_stage_and_episode_gate_controls_report_success() {
        let mut rows: Vec<_> = Stage::ALL
            .into_iter()
            .map(|stage| StageEvaluation {
                stage,
                episodes: seeds()
                    .into_iter()
                    .map(|seed| evaluation::Score {
                        seed,
                        steps: 1_000,
                        reward: stage.minimum_return(),
                        survived: true,
                        first_arrival: NonZeroU16::new(stage.deadline()),
                        settled_actions: 100,
                        final_distance: 0.5,
                        final_heading_error: std::f32::consts::PI / 12.0,
                        final_speed: 0.5,
                    })
                    .collect(),
            })
            .collect();
        assert!(passes(&rows));
        assert_eq!(label(true), "passed");
        require_pass(true).expect("all stage gates pass");
        assert!(!passes(&[]));
        assert!(!passes(rows.get(..3).expect("three rows")));
        rows.swap(0, 1);
        assert!(!passes(&rows));
        rows.swap(0, 1);
        rows.last_mut()
            .expect("four stages")
            .episodes
            .last_mut()
            .expect("32 episodes")
            .settled_actions = 99;
        assert!(!passes(&rows));
        assert_eq!(label(false), "failed");
        assert_eq!(
            require_pass(false)
                .expect_err("failed gates fail the CLI")
                .to_string(),
            "Held-out travel evaluation failed; checkpoint is not qualified."
        );
    }
}
