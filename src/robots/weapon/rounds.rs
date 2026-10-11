//! Closed remaining-round count for a drone volley.
/// Only one or two rounds can remain after the first drone discharge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotVolleyRounds {
    /// Two later rounds remain.
    Two,
    /// The final round remains.
    One,
}
