//! Require independent frozen validation before advancing any travel stage.

mod report;

use report::Report;
use sha2::Sha256;
use std::{error::Error, path::Path};
use tokio::process::Command;

use super::stage::Stage;

/// Emit every prerequisite's validation cases without training or creating output files.
pub(crate) fn run(stage: Stage, checkpoint: &Path) -> Result<(), Box<dyn Error>> {
    let report = Report::evaluate(stage, std::fs::read(checkpoint)?)?;
    let passed = report.passes(stage);
    let serialized = serde_json::to_string(&report)?;
    println!("{serialized}");
    if passed {
        Ok(())
    } else {
        Err("Promotion validation failed; no stage transition is allowed.".into())
    }
}

/// Preserve a separate process's report and require matching bytes, cases and exit status.
pub(crate) fn validate(
    stage: Stage,
    checkpoint: &Path,
    expected: &sha2::digest::Output<Sha256>,
    destination: &Path,
) -> Result<bool, Box<dyn Error>> {
    // The synchronous collector waits at this boundary; process pipes use Tokio's reactor.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let output = runtime.block_on(async {
        Command::new(std::env::current_exe()?)
            .args(["--lesson", "travel", "--evaluate-promotion"])
            .arg(checkpoint)
            .args(["--travel-stage", stage.name()])
            .output()
            .await
    })?;
    // Retain complete failed validation before interpreting it; never promote from a status alone.
    std::fs::write(destination, &output.stdout)?;
    verify(stage, expected, &output.stdout, output.status.code())
}

/// Treat malformed, unrelated or contradictory child output as an operational failure.
fn verify(
    stage: Stage,
    expected: &sha2::digest::Output<Sha256>,
    bytes: &[u8],
    exit_code: Option<i32>,
) -> Result<bool, Box<dyn Error>> {
    let report: Report = serde_json::from_slice(bytes)?;
    if !report.matches_digest(expected) {
        return Err("promotion report checkpoint differs from the selection policy".into());
    }
    let passed = report.passes(stage);
    match (passed, exit_code) {
        (true, Some(0)) => Ok(true),
        (false, Some(1)) => Ok(false),
        _ => Err("promotion report gates and evaluator exit status disagree".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    /// Child status cannot override gates, checkpoint identity or malformed output.
    #[test]
    fn separate_process_output_must_agree_with_recomputed_gates() {
        let stage = Stage::Endurance;
        let digest = Sha256::digest(b"isolated checkpoint fixture");
        let passing = serde_json::to_vec(&report::fixture(stage)).expect("fixture JSON");
        assert!(verify(stage, &digest, &passing, Some(0)).expect("pass"));
        for status in [Some(1), Some(2), None] {
            assert!(verify(stage, &digest, &passing, status).is_err());
        }
        let mut failed: serde_json::Value = serde_json::from_slice(&passing).expect("JSON");
        *failed
            .pointer_mut("/stages/0/episodes/0/settled_actions")
            .expect("fixture field") = serde_json::json!(99);
        let failed = serde_json::to_vec(&failed).expect("failed JSON");
        assert!(!verify(stage, &digest, &failed, Some(1)).expect("failure"));
        for status in [Some(0), Some(2), None] {
            assert!(verify(stage, &digest, &failed, status).is_err());
        }
        assert!(verify(stage, &digest, b"invalid JSON", Some(1)).is_err());
        assert!(verify(stage, &Sha256::digest(b"other weights"), &passing, Some(0)).is_err());
    }
}
