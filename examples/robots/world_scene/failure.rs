//! Robot identity and original inference, decoding or world failure causes.

use crate::{learning::encoding::ActionDecodeError, standing::encoding::DecodeError};
use bevy_gym::{
    robots::{RobotId, RobotSlot, RobotWorldError},
    training::RecurrentPpoError,
};
use std::{error::Error, fmt};

/// First failure that permanently revokes this clip's inference capability.
#[derive(Debug)]
pub(super) enum Failure {
    /// A named robot's recurrent policy failed before physical advancement.
    Inference {
        /// Owning robot; team follows its closed identity variant.
        robot: RobotId,
        /// Original policy failure with its model or input context.
        cause: RecurrentPpoError,
    },
    /// A drone's inferred motor values failed width or fraction validation.
    DroneAction {
        /// Owning slot within the drone team.
        slot: RobotSlot,
        /// Original decoder failure.
        cause: ActionDecodeError,
    },
    /// A droid's inferred torques failed width or fraction validation.
    DroidAction {
        /// Owning slot within the droid team.
        slot: RobotSlot,
        /// Original decoder failure.
        cause: DecodeError,
    },
    /// The shared world rejected advancement or reset.
    World(RobotWorldError),
}

impl fmt::Display for Failure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference { robot, cause } => write!(formatter, "{robot:?}: {cause}"),
            Self::DroneAction { slot, cause } => {
                let robot = RobotId::Drone(*slot);
                write!(formatter, "{robot:?}: {cause}")
            }
            Self::DroidAction { slot, cause } => {
                let robot = RobotId::Droid(*slot);
                write!(formatter, "{robot:?}: {cause}")
            }
            Self::World(cause) => cause.fmt(formatter),
        }
    }
}

impl Error for Failure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Inference { cause, .. } => Some(cause),
            Self::DroneAction { cause, .. } => Some(cause),
            Self::DroidAction { cause, .. } => Some(cause),
            Self::World(cause) => Some(cause),
        }
    }
}
