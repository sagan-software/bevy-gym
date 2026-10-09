//! Native checkpoint selection before the renderer starts.

#![allow(
    clippy::disallowed_methods,
    reason = "Native startup reads one local checkpoint before Bevy's event loop."
)]

use super::{
    pilot::{CheckpointKind, RecoveryPilot, MAX_CHECKPOINT_BYTES},
    session::Session,
};
use clap::Parser;
use std::{error::Error, io::Read, path::PathBuf};

/// Optional local model and its explicit observation recipe.
#[derive(Parser)]
struct Options {
    /// Local .mpk checkpoint, limited to 1 MiB.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Observation recipe used to train the selected checkpoint.
    #[arg(long, value_enum, default_value = "recovery", requires = "checkpoint")]
    policy: CheckpointKind,
}

/// Parse native arguments and validate a file before creating the window.
pub(super) fn initial_session() -> Result<Session, Box<dyn Error>> {
    session(Options::parse())
}

/// Keep the default paused scene unless the caller explicitly selected a file.
fn session(options: Options) -> Result<Session, Box<dyn Error>> {
    let mut session = Session::default();
    if let Some(path) = options.checkpoint {
        // Read one byte past the limit so oversized files fail without an unbounded read.
        let source = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        source
            .take(MAX_CHECKPOINT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        let pilot = RecoveryPilot::load_file(bytes, options.policy)?;
        session.watch_policy(pilot);
    }
    Ok(session)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Playback, StartProfile};

    #[test]
    fn default_start_remains_paused_and_an_explicit_kind_requires_a_file() {
        let default = Options::try_parse_from(["drone-flight"]).unwrap();
        let session = session(default).unwrap();
        assert_eq!(session.playback(), Playback::Paused);
        assert_eq!(session.policy_label(), None);
        assert_eq!(session.start_profile(), StartProfile::Calm);
        Options::try_parse_from(["drone-flight", "--policy", "motor-failure"])
            .err()
            .expect("missing checkpoint");
        Options::try_parse_from([
            "drone-flight",
            "--checkpoint",
            "model.mpk",
            "--policy",
            "unknown",
        ])
        .err()
        .expect("unknown recipe");
        let selected = Options::try_parse_from([
            "drone-flight",
            "--checkpoint",
            "model.mpk",
            "--policy",
            "motor-failure",
        ])
        .unwrap();
        assert_eq!(selected.policy, CheckpointKind::MotorFailure);
    }

    #[test]
    fn healthy_file_starts_inference_and_file_errors_return_before_the_window() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/robots/recovery.mpk");
        let loaded = session(Options {
            checkpoint: Some(path),
            policy: CheckpointKind::Recovery,
        })
        .unwrap();
        assert_eq!(loaded.playback(), Playback::Running);
        assert_eq!(loaded.start_profile(), StartProfile::Disturbed);
        assert_eq!(loaded.policy_label(), Some("Recovery checkpoint"));
        let missing =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/robots/missing-checkpoint.mpk");
        assert!(session(Options {
            checkpoint: Some(missing),
            policy: CheckpointKind::Recovery
        })
        .is_err());
    }
    #[cfg(unix)]
    #[test]
    fn unreadable_directory_returns_an_io_error() {
        let error = session(Options {
            checkpoint: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR"))),
            policy: CheckpointKind::Recovery,
        })
        .err()
        .unwrap();
        assert!(error.downcast_ref::<std::io::Error>().is_some());
    }
}
