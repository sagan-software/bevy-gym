//! Closed joint trigger and alert choices.

/// One of eight joint trigger-request and alert-state choices sampled by the actor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombatSignal {
    /// Idle alert, no trigger request.
    IdleReleased,
    /// Idle alert, trigger requested.
    IdleRequested,
    /// Suspicious alert, no trigger request.
    SuspiciousReleased,
    /// Suspicious alert, trigger requested.
    SuspiciousRequested,
    /// Pursuit alert, no trigger request.
    PursuitReleased,
    /// Pursuit alert, trigger requested.
    PursuitRequested,
    /// Search alert, no trigger request.
    SearchReleased,
    /// Search alert, trigger requested.
    SearchRequested,
}
impl CombatSignal {
    /// Stable categorical columns; checkpoint profiles use this exact order.
    pub const ALL: [Self; 8] = [
        Self::IdleReleased,
        Self::IdleRequested,
        Self::SuspiciousReleased,
        Self::SuspiciousRequested,
        Self::PursuitReleased,
        Self::PursuitRequested,
        Self::SearchReleased,
        Self::SearchRequested,
    ];

    /// Read the categorical column without permitting an invalid stored index.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::IdleReleased => 0,
            Self::IdleRequested => 1,
            Self::SuspiciousReleased => 2,
            Self::SuspiciousRequested => 3,
            Self::PursuitReleased => 4,
            Self::PursuitRequested => 5,
            Self::SearchReleased => 6,
            Self::SearchRequested => 7,
        }
    }
}
