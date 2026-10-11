//! One-time mechanical damage effects.
use crate::robots::DroneMotor;
/// Collision consequences for health, actuators and presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotDamage {
    /// A destroyed part or dead hull does not change.
    Ignored,
    /// One hit reduced life without destroying the hull or rotor.
    Hit,
    /// Disable this named rotor once.
    RotorDestroyed(DroneMotor),
    /// Disable the hull and all actuators once.
    Destroyed,
}
