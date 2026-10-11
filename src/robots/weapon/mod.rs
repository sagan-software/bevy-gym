//! Mechanical trigger timing independent of sensors and agent decisions.
mod delay;
mod magazine;
mod outcome;
mod phase;
mod rounds;
mod simulation;
pub use outcome::RobotWeaponOutcome;
pub use phase::RobotWeaponPhase;
pub use rounds::RobotVolleyRounds;
pub use simulation::RobotWeapon;
