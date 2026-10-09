//! Failure returned when damage targets an already-ended flight.

use std::{error::Error, fmt};

/// The episode has ended, so its terminal observation cannot change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DroneEpisodeEnded;

impl fmt::Display for DroneEpisodeEnded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cannot fail a motor after the drone episode has ended")
    }
}

impl Error for DroneEpisodeEnded {}
