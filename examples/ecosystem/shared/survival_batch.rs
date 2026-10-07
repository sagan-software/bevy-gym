//! Validated visual replay data for one survival training batch.

use std::io;

use super::domain::{CurriculumStage, VisualWorldSnapshot, SURVIVAL_BATCH_ENVIRONMENTS};

/// Why one nine-environment trace is being shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SurvivalBatchPurpose {
    /// Immediate random-policy replay while offline survival initialization runs.
    InitializationPreview,

    /// On-policy environments pooled into one numbered PPO update.
    TrainingUpdate {
        /// One-based optimizer update number.
        iteration: usize,
    },
}

/// Exact trajectories from nine survival environments shown together.
#[derive(Debug, Clone)]
pub(super) struct SurvivalBatchTrace {
    /// Presentation and optimizer role for these trajectories.
    purpose: SurvivalBatchPurpose,

    /// Simulated seconds between adjacent recorded frames.
    time_step: f32,

    /// Stable row-major environment lanes.
    lanes: [SurvivalLaneTrace; SURVIVAL_BATCH_ENVIRONMENTS],
}

impl SurvivalBatchTrace {
    /// Validate one complete 3x3 trace before it crosses into rendering.
    pub(super) fn new(
        iteration: usize,
        time_step: f32,
        lanes: Vec<SurvivalLaneTrace>,
    ) -> io::Result<Self> {
        // Reject malformed timing before the renderer derives frame cadence from it.
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "survival batch time step must be finite and positive",
            ));
        }
        if lanes
            .iter()
            .enumerate()
            .any(|(lane, trace)| trace.lane != lane)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "survival batch lanes must use stable row-major order",
            ));
        }
        let lanes = lanes.try_into().map_err(|_lanes| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "survival batch trace must contain exactly nine lanes",
            )
        })?;
        Ok(Self {
            purpose: SurvivalBatchPurpose::TrainingUpdate { iteration },
            time_step,
            lanes,
        })
    }

    /// Convert a random-policy trace into the startup preview.
    pub(super) const fn into_initialization_preview(mut self) -> Self {
        self.purpose = SurvivalBatchPurpose::InitializationPreview;
        self
    }

    /// Return why this trace is being shown.
    pub(super) const fn purpose(&self) -> SurvivalBatchPurpose {
        self.purpose
    }

    /// Return the fixed simulated duration between adjacent frames.
    pub(super) const fn time_step(&self) -> f32 {
        self.time_step
    }

    /// Return the nine contributing environment trajectories.
    pub(super) const fn lanes(&self) -> &[SurvivalLaneTrace] {
        &self.lanes
    }

    /// Return the longest recorded lane length.
    pub(super) fn frame_count(&self) -> usize {
        self.lanes
            .iter()
            .map(SurvivalLaneTrace::frame_count)
            .max()
            .unwrap_or_default()
    }
}

/// Exact snapshots and returns from one contributing training environment.
#[derive(Debug, Clone)]
pub(super) struct SurvivalLaneTrace {
    /// Stable row-major lane index.
    lane: usize,

    /// Curriculum lesson collected in this retained lane.
    stage: CurriculumStage,

    /// Initial state followed by every post-physics state.
    initial_snapshot: VisualWorldSnapshot,

    /// Post-step snapshots after the initial environment state.
    remaining_snapshots: Vec<VisualWorldSnapshot>,

    /// Cumulative bunny return aligned with each snapshot.
    initial_episode_return: f64,

    /// Cumulative returns aligned with each post-step snapshot.
    remaining_episode_returns: Vec<f64>,
}

impl SurvivalLaneTrace {
    /// Validate aligned nonempty snapshot and return sequences.
    pub(super) fn new(
        lane: usize,
        stage: CurriculumStage,
        snapshots: Vec<VisualWorldSnapshot>,
        episode_returns: Vec<f64>,
    ) -> io::Result<Self> {
        // Keep snapshots and returns aligned so reward labels describe the visible state.
        if snapshots.len() != episode_returns.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "survival lane snapshots and returns must be nonempty and aligned",
            ));
        }
        let mut snapshots = snapshots.into_iter();
        let Some(initial_snapshot) = snapshots.next() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "survival lane trace must contain an initial snapshot",
            ));
        };
        let mut episode_returns = episode_returns.into_iter();
        let Some(initial_episode_return) = episode_returns.next() else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "survival lane trace must contain an initial return",
            ));
        };
        Ok(Self {
            lane,
            stage,
            initial_snapshot,
            remaining_snapshots: snapshots.collect(),
            initial_episode_return,
            remaining_episode_returns: episode_returns.collect(),
        })
    }

    /// Return this lane's stable row-major index.
    pub(super) const fn lane(&self) -> usize {
        self.lane
    }

    /// Return the curriculum lesson collected in this lane.
    pub(super) const fn stage(&self) -> CurriculumStage {
        self.stage
    }

    /// Return this lane's number of recorded environment states.
    pub(super) const fn frame_count(&self) -> usize {
        self.remaining_snapshots.len().saturating_add(1)
    }

    /// Return the requested frame, holding the terminal frame after completion.
    pub(super) fn snapshot(&self, frame: usize) -> &VisualWorldSnapshot {
        // Frame zero has separate storage so every valid lane is structurally nonempty.
        let Some(remaining_index) = frame.checked_sub(1) else {
            return &self.initial_snapshot;
        };
        self.remaining_snapshots
            .get(remaining_index)
            .or_else(|| self.remaining_snapshots.last())
            .unwrap_or(&self.initial_snapshot)
    }

    /// Return cumulative episode reward aligned with the requested frame.
    pub(super) fn episode_return(&self, frame: usize) -> f64 {
        // Hold the terminal return while shorter lanes wait for the batch replay to end.
        let Some(remaining_index) = frame.checked_sub(1) else {
            return self.initial_episode_return;
        };
        self.remaining_episode_returns
            .get(remaining_index)
            .or_else(|| self.remaining_episode_returns.last())
            .copied()
            .unwrap_or(self.initial_episode_return)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cross-thread boundary must reject partial visual batches.
    #[test]
    fn batch_trace_rejects_fewer_than_nine_lanes() {
        assert!(SurvivalBatchTrace::new(0, 0.1, Vec::new()).is_err());
    }
}
