//! Checkpoint path, recorder, and error boundary.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

/// Architecture-selected policy checkpoint recorder.
pub type PolicyRecorder = NamedMpkFileRecorder<FullPrecisionSettings>;

/// Construct the default policy recorder.
#[must_use]
pub fn policy_recorder() -> PolicyRecorder {
    PolicyRecorder::default()
}

/// Checkpoint file layout for a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointPaths {
    /// Run directory containing root `best.mpk`.
    pub run_dir: PathBuf,

    /// Directory containing latest and periodic checkpoints.
    pub checkpoints_dir: PathBuf,

    /// Root best-policy checkpoint.
    pub best: PathBuf,

    /// Latest policy checkpoint.
    pub latest: PathBuf,
}

impl CheckpointPaths {
    /// Build checkpoint paths under a run directory.
    #[must_use]
    pub fn new(run_dir: impl AsRef<Path>) -> Self {
        let run_dir = run_dir.as_ref().to_path_buf();
        let checkpoints_dir = run_dir.join("checkpoints");

        Self {
            best: run_dir.join("best.mpk"),
            latest: checkpoints_dir.join("latest.mpk"),
            checkpoints_dir,
            run_dir,
        }
    }

    /// Return the periodic step checkpoint path.
    #[must_use]
    pub fn step(&self, global_step: u64) -> PathBuf {
        self.checkpoints_dir
            .join(format!("step-{global_step:06}.mpk"))
    }
}

/// Checkpoint operation used in errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointOperation {
    /// Saving a checkpoint.
    Save,

    /// Loading a checkpoint.
    Load,
}

impl fmt::Display for CheckpointOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Save => "save",
            Self::Load => "load",
        })
    }
}

/// Checkpoint error with path and operation context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointError {
    /// Failed operation.
    pub operation: CheckpointOperation,

    /// Checkpoint path.
    pub path: PathBuf,

    /// Human-readable cause.
    pub message: String,
}

impl CheckpointError {
    /// Create a checkpoint error.
    #[must_use]
    pub fn new(
        operation: CheckpointOperation,
        path: impl Into<PathBuf>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            operation,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "checkpoint {} failed for {}: {}",
            self.operation,
            self.path.display(),
            self.message
        )
    }
}

impl Error for CheckpointError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_paths_preserve_contract_names() {
        let paths = CheckpointPaths::new("runs/cartpole-dqn/ci-smoke");

        assert_eq!(
            paths.best,
            PathBuf::from("runs/cartpole-dqn/ci-smoke/best.mpk")
        );
        assert_eq!(
            paths.latest,
            PathBuf::from("runs/cartpole-dqn/ci-smoke/checkpoints/latest.mpk")
        );
        assert_eq!(
            paths.step(10),
            PathBuf::from("runs/cartpole-dqn/ci-smoke/checkpoints/step-000010.mpk")
        );
    }

    #[test]
    fn checkpoint_error_names_operation_and_path() {
        let error = CheckpointError::new(CheckpointOperation::Load, "best.mpk", "missing file");
        let message = error.to_string();

        assert!(message.contains("load"));
        assert!(message.contains("best.mpk"));
        assert!(message.contains("missing file"));
    }

    #[test]
    fn policy_recorder_is_constructible() {
        let _recorder = policy_recorder();
    }
}
