//! Dynamic checkpoint quantiles and exact recording segment timing.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::runtime_io;

use super::{CheckpointRole, RecordingError, RecordingSegment};

/// Ordered checkpoint playback used to render one artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointTimeline {
    /// Playback segments in encoded order.
    segments: Vec<RecordingSegment>,
}

impl CheckpointTimeline {
    /// Build a five-second preview of one selected checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError`] when the checkpoint does not exist.
    pub fn best_preview(checkpoint: PathBuf) -> Result<Self, RecordingError> {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        if !checkpoint.is_file() {
            return Err(RecordingError::MissingBestCheckpoint(checkpoint));
        }
        Ok(Self {
            segments: vec![RecordingSegment::new(
                CheckpointRole::Best,
                checkpoint,
                Duration::from_secs(5),
            )],
        })
    }

    /// Build the required 30-second best, first, 33%, 66%, best progression.
    ///
    /// The quantiles use the available immutable checkpoints. A short run may
    /// reuse one checkpoint for multiple roles instead of assuming a fixed
    /// checkpoint count.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError`] when `best.mpk` or all numbered checkpoints
    /// are missing, or when the checkpoint directory cannot be read.
    pub fn training_progress(run_directory: &Path) -> Result<Self, RecordingError> {
        let best = run_directory.join("best.mpk");
        if !best.is_file() {
            return Err(RecordingError::MissingBestCheckpoint(best));
        }

        let checkpoint_directory = run_directory.join("checkpoints");
        let entries = runtime_io::directory_entries(&checkpoint_directory).map_err(|source| {
            RecordingError::ReadDirectory {
                path: checkpoint_directory.clone(),
                source,
            }
        })?;

        // Sort by the numeric training step so zero-padded and compact names
        // share one deterministic progression.
        let mut checkpoints = Vec::new();
        for path in entries {
            if let Some(step) = checkpoint_step(&path) {
                checkpoints.push((step, path));
            }
        }
        checkpoints.sort_by_key(|(step, _)| *step);
        if checkpoints.is_empty() {
            return Err(RecordingError::MissingStepCheckpoints(checkpoint_directory));
        }

        let last_index = checkpoints.len() - 1;
        let Some((_, first_checkpoint)) = checkpoints.first() else {
            return Err(RecordingError::MissingStepCheckpoints(checkpoint_directory));
        };
        let Some((_, checkpoint_33)) = checkpoints.get(last_index / 3) else {
            return Err(RecordingError::MissingStepCheckpoints(checkpoint_directory));
        };
        let Some((_, checkpoint_66)) = checkpoints.get((last_index * 2) / 3) else {
            return Err(RecordingError::MissingStepCheckpoints(checkpoint_directory));
        };
        let first = first_checkpoint.clone();
        let progress_33 = checkpoint_33.clone();
        let progress_66 = checkpoint_66.clone();
        let five_seconds = Duration::from_secs(5);

        Ok(Self {
            segments: vec![
                RecordingSegment::new(CheckpointRole::Best, best.clone(), five_seconds),
                RecordingSegment::new(CheckpointRole::First, first, five_seconds),
                RecordingSegment::new(CheckpointRole::Progress33, progress_33, five_seconds),
                RecordingSegment::new(CheckpointRole::Progress66, progress_66, five_seconds),
                RecordingSegment::new(CheckpointRole::Best, best, Duration::from_secs(10)),
            ],
        })
    }

    /// Return the segments in encoded playback order.
    #[must_use]
    pub fn segments(&self) -> &[RecordingSegment] {
        &self.segments
    }

    /// Return the checkpoint used by the first playback segment.
    #[must_use]
    pub fn first_checkpoint(&self) -> Option<&Path> {
        self.segments.first().map(RecordingSegment::checkpoint)
    }

    /// Return the total encoded duration.
    #[must_use]
    pub fn duration(&self) -> Duration {
        self.segments.iter().map(RecordingSegment::duration).sum()
    }
}

/// Parse `step-000100.mpk` and `step_100.mpk` without accepting other files.
fn checkpoint_step(path: &Path) -> Option<u64> {
    let file_stem = path.file_stem()?.to_str()?;
    let numeric_step = file_stem
        .strip_prefix("step-")
        .or_else(|| file_stem.strip_prefix("step_"))?;
    numeric_step.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::checkpoint_step;
    use std::path::Path;

    #[test]
    fn checkpoint_step_accepts_both_repository_filename_styles() {
        assert_eq!(checkpoint_step(Path::new("step-000120.mpk")), Some(120));
        assert_eq!(checkpoint_step(Path::new("step_25000.mpk")), Some(25_000));
        assert_eq!(checkpoint_step(Path::new("latest.mpk")), None);
        assert_eq!(checkpoint_step(Path::new("step-broken.mpk")), None);
    }
}
