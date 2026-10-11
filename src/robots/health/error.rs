//! Reject rotor locations on droid hulls.
use std::{error::Error, fmt};
/// Droids have no drone rotors; rejection changes no health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobotRotorHitRejected;
impl fmt::Display for RobotRotorHitRejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("droids have no drone rotors")
    }
}
impl Error for RobotRotorHitRejected {}
