//! Commanded walking mechanics over the existing articulated droid body.

mod command;
mod encoding;
mod environment;
mod observation;
mod reward;
mod status;
mod work;

pub use command::{DroidVelocityCommand, InvalidDroidVelocityCommand};
pub use environment::DroidWalking;
pub use observation::DroidWalkingObservation;
pub use status::{DroidWalkingFall, DroidWalkingStatus};
