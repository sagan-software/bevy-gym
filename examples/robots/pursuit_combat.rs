//! Collect a pistol and destroy one rotor through the shared combat rules.

#[expect(
    dead_code,
    reason = "This guide shows rotor hits; the shared helper also handles body damage and crashes."
)]
#[path = "pursuit/combat.rs"]
mod combat;

use bevy_gym::robots::DroneMotor;
use combat::{health::DroneHealth, pistol::Pistol};
use std::time::Duration;

/// Two accepted pistol shots destroy a weak point; each shot consumes ammunition.
fn main() {
    let mut pistol = Pistol::default();
    let mut drone = DroneHealth::default();
    pistol
        .pick_up(Pistol::LOCATION)
        .expect("Standing at the pickup");

    for _ in 0..2 {
        pistol
            .fire()
            .expect("Loaded pistol with no remaining cooldown");
        let damage = drone.hit_rotor(DroneMotor::FrontLeft);
        println!("Rotor hit: {damage:?}");
        pistol.advance(Duration::from_millis(250));
    }
    let rounds = pistol.rounds();
    println!("Rounds remaining: {rounds}");
}
