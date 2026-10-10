//! Inspect readonly clearance inputs without issuing an autonomous action.

use bevy_gym::robots::{DroneHover, DroneRangeDirection};

/// Print one physical snapshot's six range readings, measured in metres.
fn main() {
    let drone = DroneHover::default();
    let ranges = drone.ranges();
    for direction in DroneRangeDirection::ALL {
        let reading = ranges.distance(direction);
        println!("{direction:?}: {reading:?}");
    }
}
