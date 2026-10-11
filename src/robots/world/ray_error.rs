//! Closed invalid-ray boundary categories.
/// Validation fails in origin, direction, distance, then endpoint order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidRobotRay {
    /// The world-space origin contains a nonfinite component.
    Origin,
    /// The direction contains a nonfinite component or has zero length.
    Direction,
    /// The range in metres is nonfinite or is not positive.
    Distance,
    /// The normalized finite segment ends outside finite f32 coordinates.
    Endpoint,
}
impl std::fmt::Display for InvalidRobotRay {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Origin => "robot ray origin must be finite",
            Self::Direction => "robot ray direction must be finite and nonzero",
            Self::Distance => "robot ray distance must be finite and positive",
            Self::Endpoint => "robot ray endpoint must remain finite",
        })
    }
}
impl std::error::Error for InvalidRobotRay {}
