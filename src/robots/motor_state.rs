//! Read-only actuator health, independent of commanded thrust.

/// Whether a motor can produce thrust and reaction torque.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneMotorState {
    /// The validated command determines motor force.
    Working,
    /// Force and reaction torque remain zero until the episode is reset.
    Failed,
}
