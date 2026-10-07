//! Frame position supplied to a recorder's environment driver.

use super::CheckpointRole;

/// Position of one rendered frame inside its checkpoint segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordingFrame {
    /// Semantic checkpoint role.
    role: CheckpointRole,
    /// Zero-based segment position.
    segment_index: usize,
    /// Zero-based frame position inside the segment.
    frame_in_segment: usize,
    /// Total encoded frames in the segment.
    segment_frames: usize,
    /// Encoded frame rate.
    frames_per_second: u16,
}

impl RecordingFrame {
    /// Construct the frame position for the recorder callback.
    pub(super) const fn new(
        role: CheckpointRole,
        segment_index: usize,
        frame_in_segment: usize,
        segment_frames: usize,
        frames_per_second: u16,
    ) -> Self {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        Self {
            role,
            segment_index,
            frame_in_segment,
            segment_frames,
            frames_per_second,
        }
    }

    /// Return the checkpoint's semantic role.
    #[must_use]
    pub const fn role(self) -> CheckpointRole {
        self.role
    }

    /// Return the zero-based segment position.
    #[must_use]
    pub const fn segment_index(self) -> usize {
        self.segment_index
    }

    /// Return the zero-based frame position inside the segment.
    #[must_use]
    pub const fn frame_in_segment(self) -> usize {
        self.frame_in_segment
    }

    /// Return the total encoded frames in the segment.
    #[must_use]
    pub const fn segment_frames(self) -> usize {
        self.segment_frames
    }

    /// Return the encoded frame rate.
    #[must_use]
    pub const fn frames_per_second(self) -> u16 {
        self.frames_per_second
    }

    /// Return whether this is the first frame after a checkpoint load.
    #[must_use]
    pub const fn is_first_in_segment(self) -> bool {
        self.frame_in_segment == 0
    }
}
