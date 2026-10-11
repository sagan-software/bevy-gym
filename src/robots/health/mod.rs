//! Shared damage rules without navigation, aiming or trigger decisions.
mod damage;
mod error;
mod hits;
mod hull;
mod rotor;
mod simulation;
pub use damage::RobotDamage;
pub use error::RobotRotorHitRejected;
pub use hits::RobotHitCount;
pub use rotor::RobotRotorHealth;
pub use simulation::RobotHealth;
