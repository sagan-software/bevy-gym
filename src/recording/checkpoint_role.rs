//! Semantic roles for checkpoints shown in a progress recording.

/// A checkpoint's position in a training-progress recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointRole {
    /// The checkpoint selected by held-out evaluation.
    Best,
    /// The earliest immutable training checkpoint.
    First,
    /// The checkpoint nearest one third of the saved training progression.
    Progress33,
    /// The checkpoint nearest two thirds of the saved training progression.
    Progress66,
}
