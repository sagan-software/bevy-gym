//! Rigid-body drone lessons for native and browser simulations.

mod action;
mod clearance;
mod destination;
mod droid;
mod drone;
mod episode_ended;
mod health;
mod hover;
mod impulse;
mod motor;
mod motor_state;
mod observation;
mod obstacle;
mod ranges;
mod travel;
mod travel_observation;
mod weapon;
mod world;

pub use action::{DroneAction, InvalidDroneAction};
pub use clearance::{DroneClearance, DroneClearanceObservation};
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
    DroidVelocityCommand, DroidWalking, DroidWalkingFall, DroidWalkingObservation,
    DroidWalkingStatus, InvalidDroidAction, InvalidDroidVelocityCommand,
};

pub use ranges::{DroneRangeDirection, DroneRangeDistance, DroneRanges, InvalidDroneRangeDistance};

pub use world::{
    InvalidRobotRay, RobotActions, RobotBattle, RobotBattleSnapshot, RobotFrame, RobotId,
    RobotMuzzle, RobotRay, RobotRayHit, RobotRayTarget, RobotSlot, RobotSnapshot, RobotTeam,
    RobotWorld, RobotWorldError, RobotWorldFailure,
};

pub use weapon::{RobotVolleyRounds, RobotWeapon, RobotWeaponOutcome, RobotWeaponPhase};

pub use health::{
    RobotDamage, RobotHealth, RobotHitCount, RobotRotorHealth, RobotRotorHitRejected,
};
