//! Session boundary failures.
/// A rejected worker request or failed learner operation.
#[derive(Debug)]
pub enum SessionError {
    /// Requested batch was empty or above the message latency bound.
    InvalidBudget,
    /// The shared Bevy Gym learner rejected an operation.
    Learner(bevy_gym::training::DqnError),
}

impl From<bevy_gym::training::DqnError> for SessionError {
    fn from(error: bevy_gym::training::DqnError) -> Self {
        Self::Learner(error)
    }
}
impl std::fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBudget => {
                formatter.write_str("worker batch must contain 1 through 256 steps")
            }
            Self::Learner(error) => error.fmt(formatter),
        }
    }
}
impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidBudget => None,
            Self::Learner(error) => Some(error),
        }
    }
}
