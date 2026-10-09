//! Collect a pistol and aim two shots at a rotor in the shared arena.
#![expect(
    dead_code,
    reason = "Walking, body hits, and pose inspection are tested separately."
)]

#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/combat.rs"]
mod combat;
#[path = "pursuit/shot.rs"]
mod shot;

use arena::Arena;
use bevy::math::{Quat, Vec3};
use bevy_gym::robots::DroneMotor;
use combat::pistol::Pistol;
use shot::{aim::Aim, target::Target};
use std::time::Duration;

/// Two clear shots destroy a weak point; walls and closer body parts would block them.
fn main() {
    let arena = Arena::default();
    let mut pistol = Pistol::default();
    pistol
        .pick_up(Pistol::LOCATION)
        .expect("Standing at the pickup");
    let mut drone =
        Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).expect("Finite target pose");
    let rotor = drone.rotor_centre(DroneMotor::FrontLeft);
    let aim = Aim::try_from((rotor + Vec3::Y, Vec3::NEG_Y)).expect("Aim down at the rotor");

    for _ in 0..2 {
        pistol.fire().expect("Loaded pistol with no cooldown");
        let shot = drone
            .sweep(&arena, aim, 1.0)
            .expect("The one-metre segment reaches the rotor");
        println!("Shot: {shot:?}");
        pistol.advance(Duration::from_millis(250));
    }
    let rounds = pistol.rounds();
    println!("Rounds remaining: {rounds}");
}
