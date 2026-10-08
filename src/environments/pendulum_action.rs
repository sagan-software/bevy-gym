//! Finite raw Pendulum torque at the environment input boundary.

/// Raw torque in newton metres; the environment clips it to [-2, 2].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PendulumAction {
    /// Finite torque before source-defined clipping.
    value: f32,
}

/// A torque value containing NaN or infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidPendulumAction;

impl TryFrom<f32> for PendulumAction {
    type Error = InvalidPendulumAction;
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        if value.is_finite() {
            Ok(Self { value })
        } else {
            Err(InvalidPendulumAction)
        }
    }
}
impl From<PendulumAction> for f32 {
    fn from(action: PendulumAction) -> Self {
        action.value
    }
}
impl std::fmt::Display for InvalidPendulumAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Pendulum torque must be finite")
    }
}
impl std::error::Error for InvalidPendulumAction {}
