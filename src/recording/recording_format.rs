//! Supported encoded recording containers.

/// Encoded artifact produced after frame capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingFormat {
    /// Palette-optimized animated GIF.
    Gif,
    /// H.264 video in an MP4 container.
    Mp4,
}
