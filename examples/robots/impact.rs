//! Apply an off-centre impact and inspect its physical response.

use bevy::math::Vec3;
use bevy_gym::robots::{DroneHover, DroneImpulse};

fn main() {
    let mut drone = DroneHover::default();
    // Strike 25 cm behind the body centre with 0.4 newton-seconds toward world +X.
    let impact = DroneImpulse::try_from((Vec3::Z * 0.25, Vec3::X * 0.4))
        .expect("The point and momentum are within the lesson's limits");
    drone
        .apply_impulse(impact)
        .expect("The drone is still flying");
    let velocity = drone.observation().linear_velocity();
    let spin = drone.observation().angular_velocity();
    println!("Velocity: {velocity:?} m/s; angular velocity: {spin:?} rad/s");
}
