//! Add a solid platform to the drone's flight world.
//!
//! Run `cargo run --features robots --example drone-obstacles`.

use bevy::math::{Quat, Vec3};
use bevy_gym::robots::{DroneAction, DroneHover, DroneObstacle};
use bevy_gym::Env;

/// Drop the unpowered drone onto a platform one metre above the floor.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Boxes use a centre, local half extents in metres, and a world rotation.
    let platform = DroneObstacle::try_from((
        Vec3::new(0.0, 0.9, 0.0),
        Vec3::new(1.0, 0.1, 1.0),
        Quat::IDENTITY,
    ))?;
    let mut drone = DroneHover::with_obstacles([platform]);
    drone.reset(Some(42));
    let power_off = DroneAction::try_from([0.0; 4])?;

    loop {
        let step = drone.step(power_off);
        if step.is_done() {
            let position = step.observation.position();
            println!("Contact ended flight at {position:?} metres");
            break;
        }
    }
    Ok(())
}
