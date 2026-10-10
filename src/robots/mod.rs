//! Rigid-body drone lessons for native and browser simulations.

mod action;
mod destination;
mod droid;
mod episode_ended;
mod hover;
mod impulse;
mod motor;
mod motor_state;
mod observation;
mod obstacle;
mod ranges;
mod travel;
mod travel_observation;

pub use action::{DroneAction, InvalidDroneAction};
pub use episode_ended::DroneEpisodeEnded;
pub use hover::DroneHover;
pub use impulse::{DroneImpulse, DroneImpulseRejected, InvalidDroneImpulse};
pub use motor::DroneMotor;
pub use motor_state::DroneMotorState;
pub use observation::DroneObservation;
pub use obstacle::{DroneObstacle, InvalidDroneObstacle};

pub use destination::{DroneDestination, InvalidDroneDestination};
pub use travel::DroneTravel;
pub use travel_observation::DroneTravelObservation;

pub use droid::{
    DroidAction, DroidActuator, DroidBody, DroidBodyState, DroidObservation, DroidStanding,
    InvalidDroidAction,
};

pub use ranges::{DroneRangeDirection, DroneRangeDistance, DroneRanges, InvalidDroneRangeDistance};
