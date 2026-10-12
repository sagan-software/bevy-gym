//! Inspect the shared combat environment without supplying autonomous actions.
use bevy_gym::robots::{RobotId, RobotMatch, RobotSlot};
/// Print initial physical combat health.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = RobotMatch::try_from(std::time::Duration::from_mins(2))?.snapshot()?;
    let health = snapshot.health(RobotId::Drone(RobotSlot::First));
    let hits = health.body_hits_remaining().get();
    println!("{hits}");
    Ok(())
}
