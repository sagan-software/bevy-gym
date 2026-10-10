//! Closed travel stages with fixed sampling bounds and promotion thresholds.

/// Increase destination distance and tighten the arrival deadline without changing actuators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    /// Learn position and heading changes near the original hover target.
    Near,
    /// Reach destinations several metres away.
    Far,
    /// Reach farther destinations under a shorter arrival deadline.
    Fast,
}

impl Stage {
    /// Ordered lessons; one learner and optimizer continue across these stages.
    pub(crate) const ALL: [Self; 3] = [Self::Near, Self::Far, Self::Fast];

    /// Stable artifact spelling.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Near => "travel-near",
            Self::Far => "travel-far",
            Self::Fast => "travel-fast",
        }
    }

    /// Horizontal-radius and world-height bounds, all in metres.
    pub(crate) const fn bounds(self) -> ((f32, f32), (f32, f32)) {
        match self {
            Self::Near => ((0.5, 1.0), (1.5, 2.5)),
            Self::Far => ((2.0, 4.0), (1.0, 4.0)),
            Self::Fast => ((4.0, 6.0), (1.0, 5.0)),
        }
    }

    /// Maximum action index of first arrival, at 20 ms per action.
    pub(crate) const fn deadline(self) -> u16 {
        match self {
            Self::Near | Self::Far => 500,
            Self::Fast => 250,
        }
    }

    /// Required mean unscaled return over the 1,000-action selection episodes.
    pub(crate) const fn minimum_return(self) -> f64 {
        match self {
            Self::Near => 600.0,
            Self::Far | Self::Fast => 500.0,
        }
    }
}
