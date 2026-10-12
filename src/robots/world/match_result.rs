//! Closed mechanical match outcomes.
/// Final result of elimination or a complete configured physical time limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotMatchResult {
    /// Only drones remain alive after every launched round settles.
    DronesWin,
    /// Only droids remain alive after every launched round settles.
    DroidsWin,
    /// Neither team remains alive after every launched round settles.
    MutualElimination,
    /// The complete configured physical duration elapsed before settled elimination.
    Timeout,
}
