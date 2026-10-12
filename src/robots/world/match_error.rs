//! Match input, lifecycle and physical failure categories.
use super::{RobotMatchResult, RobotWorldError};
use std::{error::Error, fmt};
/// Invalid configuration, denied terminal advancement or a preserved physical cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotMatchError {
    /// Duration must contain `1..=u32::MAX` complete 20 ms frames.
    InvalidDuration,
    /// A completed match cannot consume actions, triggers or time.
    Finished(RobotMatchResult),
    /// The original world error is retained without changing its category or source.
    Failed(RobotWorldError),
}
impl fmt::Display for RobotMatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration => {
                formatter.write_str("match duration requires 1..=u32::MAX complete 20 ms frames")
            }
            Self::Finished(result) => write!(formatter, "match finished: {result:?}"),
            Self::Failed(error) => error.fmt(formatter),
        }
    }
}
impl Error for RobotMatchError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Failed(error) => Some(error),
            Self::InvalidDuration | Self::Finished(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Configuration and terminal denial have no source; physical failures retain the original source.
    #[test]
    fn categories_preserve_sources_and_diagnostics() {
        let invalid = RobotMatchError::InvalidDuration;
        assert!(invalid.source().is_none());
        assert_eq!(
            invalid.to_string(),
            "match duration requires 1..=u32::MAX complete 20 ms frames"
        );
        let finished = RobotMatchError::Finished(RobotMatchResult::Timeout);
        assert!(finished.source().is_none());
        assert_eq!(finished.to_string(), "match finished: Timeout");
        let cause = RobotWorldError::Rejected(super::super::RobotWorldFailure::WrongFrame);
        let failed = RobotMatchError::Failed(cause);
        assert!(failed.source().is_some());
        assert_eq!(failed.to_string(), cause.to_string());
    }
}
