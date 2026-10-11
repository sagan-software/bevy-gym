//! Inspect initial mechanical weapon state without running an autonomous controller.
use bevy_gym::robots::{RobotTeam, RobotWeapon};
/// Print both teams' initial weapon phases; no trigger request is supplied.
fn main() {
    let drone = RobotWeapon::from(RobotTeam::Drones);
    let droid = RobotWeapon::from(RobotTeam::Droids);
    let drone_phase = drone.phase();
    let droid_phase = droid.phase();
    println!("{drone_phase:?} {droid_phase:?}");
}
