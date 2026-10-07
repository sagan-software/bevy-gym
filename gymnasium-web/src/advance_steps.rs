//! Bounded worker work budget.
/// At most 256 transitions between worker message boundaries.
#[derive(Debug, Clone, Copy)]
pub struct AdvanceSteps(u16);
impl TryFrom<u16> for AdvanceSteps {
    type Error = crate::SessionError;
    fn try_from(value: u16) -> Result<Self, Self::Error> {
        if (1..=256).contains(&value) {
            Ok(Self(value))
        } else {
            Err(crate::SessionError::InvalidBudget)
        }
    }
}
impl AdvanceSteps {
    /// Number of transitions requested in this batch.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}
