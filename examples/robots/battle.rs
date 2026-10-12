//! Inspect the shared combat environment without supplying autonomous actions.
use bevy_gym::robots::{RobotId, RobotMatch, RobotSlot};
/// Print initial physical combat health and the fixed camera image dimensions.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let game = RobotMatch::try_from(std::time::Duration::from_mins(2))?;
    let snapshot = game.snapshot()?;
    let health = snapshot.health(RobotId::Drone(RobotSlot::First));
    let hits = health.body_hits_remaining().get();
    println!("{hits}");
    let image = game.image(RobotId::Drone(RobotSlot::Third))?;
    let rows = image.pixels().len();
    let columns = image.pixels()[0].len();
    println!("{columns} x {rows} RGB camera image");
    Ok(())
}
