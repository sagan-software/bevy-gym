//! Validated public settings for the Bevy recording plugin.

use std::path::{Path, PathBuf};

use super::{CheckpointTimeline, RecordingError, RecordingFormat};

/// Validated output and timing settings for one recording session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordingSettings {
    /// Destination artifact path.
    output: PathBuf,
    /// Checkpoint playback schedule.
    timeline: CheckpointTimeline,
    /// Encoded frames per second.
    frames_per_second: u16,
    /// Encoded container.
    format: RecordingFormat,
    /// Render frames reserved for asset and pipeline initialization.
    warmup_frames: usize,
}

impl RecordingSettings {
    /// Configure a five-second best-checkpoint GIF at 20 frames per second.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError`] when the checkpoint does not exist.
    pub fn gif(output: PathBuf, checkpoint: PathBuf) -> Result<Self, RecordingError> {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        Ok(Self {
            output,
            timeline: CheckpointTimeline::best_preview(checkpoint)?,
            frames_per_second: 20,
            format: RecordingFormat::Gif,
            warmup_frames: 30,
        })
    }

    /// Configure the standard 30-second checkpoint-progress MP4 at 30 frames per second.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError`] when the run has no best or numbered checkpoint.
    pub fn training_video(output: PathBuf, run_directory: &Path) -> Result<Self, RecordingError> {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        Ok(Self {
            output,
            timeline: CheckpointTimeline::training_progress(run_directory)?,
            frames_per_second: 30,
            format: RecordingFormat::Mp4,
            warmup_frames: 30,
        })
    }

    /// Set an encoded rate of at least 15 frames per second.
    ///
    /// # Errors
    ///
    /// Returns [`RecordingError::FramesPerSecondBelowMinimum`] below 15 FPS.
    pub fn with_frames_per_second(
        mut self,
        frames_per_second: u16,
    ) -> Result<Self, RecordingError> {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        if frames_per_second < 15 {
            return Err(RecordingError::FramesPerSecondBelowMinimum(
                frames_per_second,
            ));
        }
        self.frames_per_second = frames_per_second;
        Ok(self)
    }

    /// Set the number of render frames reserved for initialization.
    #[must_use]
    pub const fn with_warmup_frames(mut self, warmup_frames: usize) -> Self {
        self.warmup_frames = warmup_frames;
        self
    }

    /// Return the destination artifact path.
    #[must_use]
    pub fn output(&self) -> &Path {
        &self.output
    }

    /// Return the checkpoint playback schedule.
    #[must_use]
    pub const fn timeline(&self) -> &CheckpointTimeline {
        &self.timeline
    }

    /// Return the encoded frame rate.
    #[must_use]
    pub const fn frames_per_second(&self) -> u16 {
        self.frames_per_second
    }

    /// Return the encoded container.
    #[must_use]
    pub const fn format(&self) -> RecordingFormat {
        self.format
    }

    /// Return the number of render frames reserved for initialization.
    #[must_use]
    pub const fn warmup_frames(&self) -> usize {
        self.warmup_frames
    }
}
