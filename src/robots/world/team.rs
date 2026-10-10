//! The two robot teams.

/// Team identity; each team owns exactly three slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotTeam {
    /// Four-motor rigid-body drones.
    Drones,
    /// Thirteen-segment direct-torque droids.
    Droids,
}
