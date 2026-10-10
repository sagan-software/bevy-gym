//! Read a destination and physical range observations without choosing an agent action.

use bevy::math::{Quat, Vec2, Vec3};
use bevy_gym::robots::{DroneClearance, DroneDestination, DroneObstacle, DroneRangeDirection};

/// Construct validated lesson geometry and inspect one disturbed physical snapshot.
fn main() {
    let destination = DroneDestination::try_from((Vec3::new(0.0, 2.0, -4.0), Vec2::NEG_Y))
        .expect("finite interior destination");
    let obstacle = DroneObstacle::try_from((
        Vec3::new(1.0, 2.0, -2.0),
        Vec3::new(0.25, 1.0, 0.25),
        Quat::IDENTITY,
    ))
    .expect("finite positive collision box");
    let environment = DroneClearance::new(destination, [obstacle]);
    let observation = environment.observation();
    let task = observation.travel().destination();
    println!("Destination: {task:?}");
    for direction in DroneRangeDirection::ALL {
        let distance = observation.ranges().distance(direction);
        println!("{direction:?}: {distance:?}");
    }
}
