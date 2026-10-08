//! Closed torque vocabulary for Acrobot's actuated joint.

/// Original `Discrete(3)` action order and torques in newton metres.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcrobotAction {
    /// Apply -1 newton metre at the second joint.
    Negative,
    /// Apply zero torque.
    Coast,
    /// Apply 1 newton metre at the second joint.
    Positive,
}
/// An index outside the three-action vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidAcrobotAction;
impl TryFrom<usize> for AcrobotAction {
    type Error = InvalidAcrobotAction;
    fn try_from(index: usize) -> Result<Self, Self::Error> {
        match index {
            0 => Ok(Self::Negative),
            1 => Ok(Self::Coast),
            2 => Ok(Self::Positive),
            _ => Err(InvalidAcrobotAction),
        }
    }
}
impl From<AcrobotAction> for usize {
    fn from(action: AcrobotAction) -> Self {
        match action {
            AcrobotAction::Negative => 0,
            AcrobotAction::Coast => 1,
            AcrobotAction::Positive => 2,
        }
    }
}
impl std::fmt::Display for InvalidAcrobotAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Acrobot action must be 0, 1, or 2")
    }
}
impl std::error::Error for InvalidAcrobotAction {}
