//! Reset-only inspection of the shared physical world; no autonomous decisions or actions.

use bevy_gym::robots::{DroidBody, RobotSlot, RobotWorld};

/// Inspect all six physical agents without advancing the world or choosing any action.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = RobotWorld::default();
    let snapshot = world.snapshot()?;
    for slot in RobotSlot::ALL {
        let drone = snapshot.drone(slot).position();
        let droid = snapshot.droid(slot).body(DroidBody::Pelvis).position();
        println!("{slot:?}: drone {drone:?}; droid pelvis {droid:?}");
    }
    Ok(())
}
