//! Convert a validated flight goal into healthy-drone motor commands.

mod goal;
mod pilot;

#[path = "../learning/encoding.rs"]
mod encoding;

pub(crate) use goal::FlightGoal;
pub(crate) use pilot::FlightPilot;
