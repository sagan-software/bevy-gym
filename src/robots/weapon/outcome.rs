//! Mechanical disposition of one policy trigger request.
/// Disposition of one twenty-millisecond mechanical step, never an agent decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotWeaponOutcome {
    /// A valid current trigger request discharged one round.
    Fired,
    /// No trigger was requested while ready, or charging was cancelled.
    NoRequest,
    /// The drone has not completed its charge.
    Charging,
    /// A drone volley has not completed its inter-round delay.
    VolleyDelay,
    /// Recovery, including cancelled volleys, prevents discharge this step.
    Cooling,
    /// The droid has no remaining ammunition.
    Empty,
    /// Lifecycle shutdown prevents discharge.
    Disabled,
}
