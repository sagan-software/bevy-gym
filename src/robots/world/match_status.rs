//! Read-only match lifecycle measurements.
use super::{RobotMatchResult, RobotWorldError};
/// Exactly one current lifecycle measurement, derived from the owning match state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotMatchStatus {
    /// Physical requests remain permitted.
    Running,
    /// A final mechanical result prohibits further physical advancement.
    Finished(RobotMatchResult),
    /// The original physical failure prohibits further physical advancement.
    Failed(RobotWorldError),
}
