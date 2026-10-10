//! Input rejection and stopped-world diagnostics with preserved causes.

use super::RobotId;
use std::{error::Error, fmt};

/// Original reason that further physical advancement is prohibited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotWorldFailure {
    /// Actions refer to another world, reset, or previous physical boundary.
    WrongFrame,
    /// The finite frame ordinal has reached its maximum.
    ClockExhausted,
    /// A named robot's physical state is nonfinite.
    NonFinite(RobotId),
}

impl fmt::Display for RobotWorldFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongFrame => {
                formatter.write_str("actions do not belong to the current robot frame")
            }
            Self::ClockExhausted => formatter.write_str("robot frame clock exhausted"),
            Self::NonFinite(robot) => write!(formatter, "nonfinite physical state for {robot:?}"),
        }
    }
}
impl Error for RobotWorldFailure {}

/// Reject a frame or report the original cause of an already stopped world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotWorldError {
    /// This operation discovered the failure.
    Rejected(RobotWorldFailure),
    /// A previous operation stopped the world; only reset can resume it.
    Stopped(RobotWorldFailure),
}
impl fmt::Display for RobotWorldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected(cause) => write!(formatter, "robot frame rejected: {cause}"),
            Self::Stopped(cause) => write!(formatter, "robot world stopped: {cause}"),
        }
    }
}
impl Error for RobotWorldError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rejected(cause) | Self::Stopped(cause) => Some(cause),
        }
    }
}
