//! Goal-conditioned collision clearance through learned motor commands and local ranges.

mod environment;
mod observation;

pub use environment::DroneClearance;
pub use observation::DroneClearanceObservation;
