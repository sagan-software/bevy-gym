//! Inspect the shared combat environment without supplying autonomous actions.
use bevy_gym::robots::{RobotBattle, RobotId, RobotSlot};
/// Print initial physical combat health.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let snapshot = RobotBattle::default().snapshot()?;
    let health = snapshot.health(RobotId::Drone(RobotSlot::First));
    let hits = health.body_hits_remaining().get();
    println!("{hits}");
    Ok(())
}
