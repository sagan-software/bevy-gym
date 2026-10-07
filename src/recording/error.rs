//! Recording setup failures with source paths and I/O causes.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;
use std::process::ExitStatus;

/// Failure to build or encode a training recording.
#[derive(Debug)]
pub enum RecordingError {
    /// A required artifact directory could not be created.
    CreateDirectory {
        /// Directory that could not be created.
        path: PathBuf,
        /// Underlying filesystem error.
        source: io::Error,
    },
    /// The ffmpeg process could not be started or awaited.
    RunEncoder {
        /// Destination that ffmpeg was expected to create.
        output: PathBuf,
        /// Underlying process error.
        source: io::Error,
    },
    /// The ffmpeg process returned a non-success exit status.
    EncoderFailed {
        /// Destination that ffmpeg failed to create.
        output: PathBuf,
        /// Process exit status.
        status: ExitStatus,
    },
    /// An encoded frame rate violated the 15 FPS artifact minimum.
    FramesPerSecondBelowMinimum(u16),
    /// The run has no selected `best.mpk` checkpoint.
    MissingBestCheckpoint(PathBuf),
    /// The run has no immutable numbered checkpoints.
    MissingStepCheckpoints(PathBuf),
    /// A run directory could not be inspected.
    ReadDirectory {
        /// Directory that could not be read.
        path: PathBuf,
        /// Underlying filesystem error.
        source: io::Error,
    },
}

impl fmt::Display for RecordingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        match self {
            Self::CreateDirectory { path, source } => {
                write!(formatter, "could not create {}: {source}", path.display())
            }
            Self::RunEncoder { output, source } => write!(
                formatter,
                "could not run ffmpeg for {}: {source}",
                output.display()
            ),
            Self::EncoderFailed { output, status } => write!(
                formatter,
                "ffmpeg failed for {} with {status}",
                output.display()
            ),
            Self::FramesPerSecondBelowMinimum(value) => write!(
                formatter,
                "recording frame rate must be at least 15 FPS, received {value}"
            ),
            Self::MissingBestCheckpoint(path) => {
                write!(
                    formatter,
                    "best checkpoint does not exist: {}",
                    path.display()
                )
            }
            Self::MissingStepCheckpoints(path) => write!(
                formatter,
                "checkpoint directory has no numbered step checkpoints: {}",
                path.display()
            ),
            Self::ReadDirectory { path, source } => {
                write!(formatter, "could not read {}: {source}", path.display())
            }
        }
    }
}

impl Error for RecordingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        match self {
            Self::CreateDirectory { source, .. }
            | Self::ReadDirectory { source, .. }
            | Self::RunEncoder { source, .. } => Some(source),
            Self::EncoderFailed { .. }
            | Self::FramesPerSecondBelowMinimum(_)
            | Self::MissingBestCheckpoint(_)
            | Self::MissingStepCheckpoints(_) => None,
        }
    }
}
