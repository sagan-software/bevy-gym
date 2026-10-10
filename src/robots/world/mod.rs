//! Shared six-agent physics and validated actuator-frame boundary.

mod actions;
mod error;
mod frame;
mod identity;
mod simulation;
mod slot;
mod snapshot;
mod team;

pub use actions::RobotActions;
pub use error::{RobotWorldError, RobotWorldFailure};
pub use frame::RobotFrame;
pub use identity::RobotId;
pub use simulation::RobotWorld;
pub use slot::RobotSlot;
pub use snapshot::RobotSnapshot;
pub use team::RobotTeam;
