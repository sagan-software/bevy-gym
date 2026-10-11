//! Inspect fresh mechanical health without choosing autonomous actions.
use bevy_gym::robots::{RobotHealth, RobotTeam};
/// Print the two teams' initial hull limits.
fn main() {
    for team in [RobotTeam::Drones, RobotTeam::Droids] {
        let health = RobotHealth::from(team);
        let remaining = health.body_hits_remaining().get();
        println!("{team:?}: {remaining} hits");
    }
}
