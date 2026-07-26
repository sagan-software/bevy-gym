//! Run and algorithm configuration boundary types.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Supported trainer algorithm families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgorithmKind {
    /// Deep Q-Network for discrete actions.
    Dqn,

    /// Proximal Policy Optimization.
    Ppo,

    /// Project-specific algorithm key.
    Custom(String),
}

impl AlgorithmKind {
    /// Return the lowercase algorithm key used in paths.
    #[must_use]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "custom algorithm keys are owned strings"
    )]
    pub fn as_key(&self) -> &str {
        match self {
            Self::Dqn => "dqn",
            Self::Ppo => "ppo",
            Self::Custom(key) => key.as_str(),
        }
    }
}

impl fmt::Display for AlgorithmKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_key())
    }
}

/// Stable identifier for one training run.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(String);

impl RunId {
    /// Create a run id from caller-provided text.
    ///
    /// # Errors
    ///
    /// Returns [`RunConfigError`] when the id is empty or is not a single path
    /// segment.
    pub fn new(value: impl Into<String>) -> Result<Self, RunConfigError> {
        let value = value.into();
        validate_path_segment("run_id", &value)?;
        Ok(Self(value))
    }

    /// Borrow the run id as a path/display-safe string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Run-level trainer configuration shared by algorithms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunConfig {
    /// Environment name, for example `cartpole`.
    env_name: String,

    /// Algorithm family.
    algorithm: AlgorithmKind,

    /// Stable run id.
    run_id: RunId,

    /// Root directory containing run artifacts.
    runs_root: PathBuf,

    /// Validated `<env>-<algorithm>` path segment.
    run_name: String,
}

impl RunConfig {
    /// Construct run configuration under a root directory.
    ///
    /// # Errors
    ///
    /// Returns [`RunConfigError`] when `env_name`, `algorithm`, or `run_id`
    /// cannot produce the stable run directory contract.
    pub fn new(
        env_name: impl Into<String>,
        algorithm: AlgorithmKind,
        run_id: RunId,
        runs_root: impl Into<PathBuf>,
    ) -> Result<Self, RunConfigError> {
        let env_name = env_name.into();
        let run_name = run_name(&env_name, &algorithm)?;

        Ok(Self {
            env_name,
            algorithm,
            run_id,
            runs_root: runs_root.into(),
            run_name,
        })
    }

    /// Borrow the environment name.
    #[must_use]
    pub fn env_name(&self) -> &str {
        &self.env_name
    }

    /// Borrow the algorithm kind.
    #[must_use]
    pub const fn algorithm(&self) -> &AlgorithmKind {
        &self.algorithm
    }

    /// Borrow the run id.
    #[must_use]
    pub const fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// Borrow the artifact root path.
    #[must_use]
    pub fn runs_root(&self) -> &Path {
        &self.runs_root
    }

    /// Return the `<env>-<algorithm>` path segment.
    #[must_use]
    pub fn run_name(&self) -> &str {
        &self.run_name
    }

    /// Resolve the run artifact paths for this config.
    #[must_use]
    pub fn paths(&self) -> RunPaths {
        RunPaths::from_valid_parts(&self.runs_root, self.run_name(), &self.run_id)
    }
}

/// File layout for one training run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunPaths {
    /// Run directory, e.g. `runs/cartpole-dqn/ci-smoke`.
    pub run_dir: PathBuf,

    /// `config.json` path.
    pub config_json: PathBuf,

    /// `seeds.json` path.
    pub seeds_json: PathBuf,

    /// `metrics.jsonl` path.
    pub metrics_jsonl: PathBuf,

    /// `eval.jsonl` path.
    pub eval_jsonl: PathBuf,

    /// `summary.json` path.
    pub summary_json: PathBuf,

    /// `provenance.json` path.
    pub provenance_json: PathBuf,

    /// `proof.json` path.
    pub proof_json: PathBuf,

    /// `checkpoints/` directory path.
    pub checkpoints_dir: PathBuf,

    /// `checkpoints/latest.mpk` path.
    pub latest_checkpoint: PathBuf,

    /// Root `best.mpk` path.
    pub best_checkpoint: PathBuf,

    /// `demo/` directory path.
    pub demo_dir: PathBuf,

    /// `demo/training-timelapse.mp4` path.
    pub demo_video: PathBuf,

    /// `demo/poster.png` path.
    pub demo_poster: PathBuf,

    /// `demo/manifest.json` path.
    pub demo_manifest: PathBuf,
}

impl RunPaths {
    /// Build paths from root, environment, algorithm, and run id.
    ///
    /// # Errors
    ///
    /// Returns [`RunConfigError`] when the environment or algorithm cannot
    /// produce a non-empty stable path segment.
    pub fn new(
        runs_root: impl AsRef<Path>,
        env_name: &str,
        algorithm: &AlgorithmKind,
        run_id: &RunId,
    ) -> Result<Self, RunConfigError> {
        let run_name = run_name(env_name, algorithm)?;
        Ok(Self::from_valid_parts(runs_root, &run_name, run_id))
    }

    /// Create a new run and checkpoint directory without reusing old data.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::AlreadyExists`] when the run directory already
    /// exists, or another I/O error when its parent or checkpoint directory
    /// cannot be created.
    #[expect(
        clippy::allow_attributes,
        reason = "the repository disallows synchronous filesystem helpers by default"
    )]
    #[allow(
        clippy::disallowed_methods,
        reason = "run setup is synchronous and has no async runtime"
    )]
    pub fn create_new(&self) -> io::Result<()> {
        if let Some(parent) = self.run_dir.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(&self.run_dir)?;
        fs::create_dir(&self.checkpoints_dir)?;
        fs::create_dir(&self.demo_dir)
    }

    /// Build paths from already validated run-name and run-id segments.
    fn from_valid_parts(runs_root: impl AsRef<Path>, run_name: &str, run_id: &RunId) -> Self {
        let run_dir = runs_root.as_ref().join(run_name).join(run_id.as_str());
        let checkpoints_dir = run_dir.join("checkpoints");
        let demo_dir = run_dir.join("demo");

        Self {
            config_json: run_dir.join("config.json"),
            seeds_json: run_dir.join("seeds.json"),
            metrics_jsonl: run_dir.join("metrics.jsonl"),
            eval_jsonl: run_dir.join("eval.jsonl"),
            summary_json: run_dir.join("summary.json"),
            provenance_json: run_dir.join("provenance.json"),
            proof_json: run_dir.join("proof.json"),
            latest_checkpoint: checkpoints_dir.join("latest.mpk"),
            best_checkpoint: run_dir.join("best.mpk"),
            demo_video: demo_dir.join("training-timelapse.mp4"),
            demo_poster: demo_dir.join("poster.png"),
            demo_manifest: demo_dir.join("manifest.json"),
            demo_dir,
            checkpoints_dir,
            run_dir,
        }
    }
}

/// Return the lowercase kebab-case run name `<env>-<algorithm>`.
///
/// # Errors
///
/// Returns [`RunConfigError`] when either segment sanitizes to an empty path
/// component.
pub fn run_name(env_name: &str, algorithm: &AlgorithmKind) -> Result<String, RunConfigError> {
    let env_key = kebab(env_name);
    if env_key.is_empty() {
        return Err(RunConfigError::EmptyRunNameSegment {
            field: "env_name",
            value: env_name.to_owned(),
        });
    }

    let algorithm_value = algorithm.as_key();
    let algorithm_key = kebab(algorithm_value);
    if algorithm_key.is_empty() {
        return Err(RunConfigError::EmptyRunNameSegment {
            field: "algorithm",
            value: algorithm_value.to_owned(),
        });
    }

    Ok(format!("{env_key}-{algorithm_key}"))
}

/// Convert a human-readable path segment to lowercase kebab case.
fn kebab(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_dash = false;

    for ch in input.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            last_dash = false;
        } else if !last_dash && !out.is_empty() {
            out.push('-');
            last_dash = true;
        }
    }

    if out.ends_with('-') {
        out.pop();
    }

    out
}

/// Validate that caller-provided text is one path segment.
fn validate_path_segment(field: &'static str, value: &str) -> Result<(), RunConfigError> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
        || value.chars().any(char::is_control)
    {
        return Err(RunConfigError::InvalidPathSegment {
            field,
            value: value.to_owned(),
        });
    }

    Ok(())
}

/// Run configuration validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunConfigError {
    /// A caller-provided path segment could escape the expected run directory.
    InvalidPathSegment {
        /// Field being validated.
        field: &'static str,

        /// Rejected value.
        value: String,
    },

    /// A run-name input sanitized to an empty segment.
    EmptyRunNameSegment {
        /// Field being validated.
        field: &'static str,

        /// Rejected value.
        value: String,
    },
}

impl fmt::Display for RunConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPathSegment { field, value } => write!(
                formatter,
                "{field} must be a non-empty single path segment, got {value:?}"
            ),
            Self::EmptyRunNameSegment { field, value } => {
                write!(
                    formatter,
                    "{field} must produce a non-empty run-name segment, got {value:?}"
                )
            }
        }
    }
}

impl Error for RunConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_name_is_lowercase_kebab_case() {
        assert_eq!(
            run_name("CartPole", &AlgorithmKind::Dqn).expect("valid DQN run name"),
            "cartpole-dqn"
        );
        assert_eq!(
            run_name(
                "Inverted Double Pendulum",
                &AlgorithmKind::Custom("ppo-lite".into())
            )
            .expect("valid custom run name"),
            "inverted-double-pendulum-ppo-lite"
        );
    }

    #[test]
    fn run_paths_match_artifact_contract() {
        let paths = RunConfig::new(
            "cartpole",
            AlgorithmKind::Dqn,
            RunId::new("ci-smoke").expect("valid run id"),
            "runs",
        )
        .expect("valid run config")
        .paths();

        assert_eq!(paths.run_dir, PathBuf::from("runs/cartpole-dqn/ci-smoke"));
        assert_eq!(paths.config_json, paths.run_dir.join("config.json"));
        assert_eq!(paths.seeds_json, paths.run_dir.join("seeds.json"));
        assert_eq!(paths.metrics_jsonl, paths.run_dir.join("metrics.jsonl"));
        assert_eq!(paths.eval_jsonl, paths.run_dir.join("eval.jsonl"));
        assert_eq!(paths.summary_json, paths.run_dir.join("summary.json"));
        assert_eq!(paths.provenance_json, paths.run_dir.join("provenance.json"));
        assert_eq!(paths.proof_json, paths.run_dir.join("proof.json"));
        assert_eq!(paths.checkpoints_dir, paths.run_dir.join("checkpoints"));
        assert_eq!(
            paths.latest_checkpoint,
            paths.run_dir.join("checkpoints/latest.mpk")
        );
        assert_eq!(paths.best_checkpoint, paths.run_dir.join("best.mpk"));
        assert_eq!(paths.demo_dir, paths.run_dir.join("demo"));
        assert_eq!(
            paths.demo_video,
            paths.run_dir.join("demo/training-timelapse.mp4")
        );
        assert_eq!(paths.demo_poster, paths.run_dir.join("demo/poster.png"));
        assert_eq!(
            paths.demo_manifest,
            paths.run_dir.join("demo/manifest.json")
        );
    }

    #[test]
    fn run_id_rejects_path_escape_segments() {
        assert_eq!(
            RunId::new("../escape"),
            Err(RunConfigError::InvalidPathSegment {
                field: "run_id",
                value: "../escape".into(),
            })
        );
        assert_eq!(
            RunId::new(""),
            Err(RunConfigError::InvalidPathSegment {
                field: "run_id",
                value: String::new(),
            })
        );
    }

    #[test]
    fn run_name_rejects_empty_sanitized_segments() {
        assert_eq!(
            run_name("!!!", &AlgorithmKind::Dqn),
            Err(RunConfigError::EmptyRunNameSegment {
                field: "env_name",
                value: "!!!".into(),
            })
        );
        assert_eq!(
            run_name("cartpole", &AlgorithmKind::Custom("...".into())),
            Err(RunConfigError::EmptyRunNameSegment {
                field: "algorithm",
                value: "...".into(),
            })
        );
    }

    #[test]
    #[expect(
        clippy::allow_attributes,
        reason = "the repository disallows synchronous filesystem helpers by default"
    )]
    #[allow(
        clippy::disallowed_methods,
        reason = "the synchronous filesystem test owns its temporary directory"
    )]
    fn run_creation_is_exclusive_and_creates_checkpoint_directory() {
        let root = std::env::temp_dir().join(format!("bevy-gym-run-create-{}", std::process::id()));
        drop(fs::remove_dir_all(&root));
        let paths = RunConfig::new(
            "cartpole",
            AlgorithmKind::Dqn,
            RunId::new("exclusive").expect("valid run id"),
            &root,
        )
        .expect("valid run config")
        .paths();

        paths.create_new().expect("new run directory is created");
        assert!(paths.run_dir.is_dir());
        assert!(paths.checkpoints_dir.is_dir());
        assert!(paths.demo_dir.is_dir());

        let error = paths
            .create_new()
            .expect_err("existing run directory is rejected");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);

        drop(fs::remove_dir_all(root));
    }
}
