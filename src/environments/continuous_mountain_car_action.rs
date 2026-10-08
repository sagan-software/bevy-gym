//! Finite continuous `MountainCar` actions retain the raw value used by the reward.

/// Finite raw force; dynamics clip it to [-1, 1], while reward uses its square.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuousMountainCarAction {
    /// Raw force before dynamics clipping.
    value: f32,
}

/// An action containing NaN or infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidContinuousMountainCarAction;

impl TryFrom<f32> for ContinuousMountainCarAction {
    type Error = InvalidContinuousMountainCarAction;
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if value.is_finite() {
            Ok(Self { value })
        } else {
            Err(InvalidContinuousMountainCarAction)
        }
    }
}
impl From<ContinuousMountainCarAction> for f32 {
    fn from(action: ContinuousMountainCarAction) -> Self {
        action.value
    }
}
impl std::fmt::Display for InvalidContinuousMountainCarAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Continuous MountainCar action must be finite")
    }
}
impl std::error::Error for InvalidContinuousMountainCarAction {}
