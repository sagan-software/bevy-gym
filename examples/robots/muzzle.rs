//! Inspect physical muzzle geometry without supplying autonomous actions.
use bevy_gym::robots::{RobotId, RobotMuzzle, RobotSlot, RobotWorld};
/// Print a read-only muzzle origin from the common physical snapshot.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = RobotWorld::default().snapshot()?;
    let muzzle = RobotMuzzle::try_from((&snapshot, RobotId::Drone(RobotSlot::First)))?;
    let origin = muzzle.origin();
    println!("{origin:?}");
    Ok(())
}
