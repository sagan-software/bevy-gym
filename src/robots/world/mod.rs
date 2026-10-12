//! Shared six-agent physics and validated actuator-frame boundary.

mod actions;
mod actuation;
mod battle;
mod battle_snapshot;
mod competition;
mod error;
mod frame;
mod identity;
mod match_error;
mod match_result;
mod match_status;
mod muzzle;
mod projectile;
mod simulation;
mod slot;
mod snapshot;
mod team;

pub use actions::RobotActions;
pub use battle::RobotBattle;
pub use battle_snapshot::RobotBattleSnapshot;
pub use competition::RobotMatch;
pub use error::{RobotWorldError, RobotWorldFailure};
pub use frame::RobotFrame;
pub use identity::RobotId;
pub use match_error::RobotMatchError;
pub use match_result::RobotMatchResult;
pub use match_status::RobotMatchStatus;
pub use muzzle::RobotMuzzle;
pub use simulation::RobotWorld;
pub use slot::RobotSlot;
pub use snapshot::RobotSnapshot;
pub use team::RobotTeam;

mod ray;
mod ray_error;
mod ray_hit;
mod ray_target;
pub use ray::RobotRay;
pub use ray_error::InvalidRobotRay;
pub use ray_hit::RobotRayHit;
pub use ray_target::RobotRayTarget;
