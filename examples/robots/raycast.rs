//! Read physical collision geometry without executing any autonomous controller.
use bevy::math::Vec3;
use bevy_gym::robots::{RobotRay, RobotWorld};
/// Print the shared floor's measured distance before the first physics action.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world = RobotWorld::default();
    let ray = RobotRay::try_from((Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y, 2.0))?;
    let hit = world.cast_ray(&ray, None)?;
    println!("{hit:?}");
    Ok(())
}
