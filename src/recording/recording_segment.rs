//! One checkpoint segment in a rendered recording.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::CheckpointRole;

/// One continuous checkpoint playback segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingSegment {
    /// Semantic position of the checkpoint in training.
    role: CheckpointRole,
    /// Checkpoint loaded for this segment.
    checkpoint: PathBuf,
    /// Encoded playback duration.
    duration: Duration,
}

impl RecordingSegment {
    /// Construct one validated segment.
    pub(super) const fn new(role: CheckpointRole, checkpoint: PathBuf, duration: Duration) -> Self {
        Self {
            role,
            checkpoint,
            duration,
        }
    }

    /// Return the checkpoint's semantic role.
    #[must_use]
    pub const fn role(&self) -> CheckpointRole {
        self.role
    }

    /// Return the checkpoint path loaded for this segment.
    #[must_use]
    pub fn checkpoint(&self) -> &Path {
        &self.checkpoint
    }

    /// Return the encoded playback duration.
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.duration
    }
}
