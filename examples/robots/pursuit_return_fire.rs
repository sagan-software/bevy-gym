//! Watch a drone weapon warn before it emits moving projectiles.
//!
//! Run `cargo run --features robots --example pursuit-return-fire`.

#[expect(dead_code, reason = "This guide only queries the arena.")]
#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/enemy.rs"]
mod enemy;
#[expect(dead_code, reason = "The guide starts with a current sighting.")]
#[path = "pursuit/sight.rs"]
mod sight;

use arena::Arena;
use bevy::math::Vec3;
use enemy::{
    gun::{Gun, Phase},
    projectile::Projectiles,
    robot_health::RobotHealth,
};
use sight::Contact;

/// Each action represents twenty milliseconds of simulation.
fn main() {
    let arena = Arena::default();
    let mut gun = Gun::default();
    let muzzle = Vec3::new(0.0, 2.0, 0.0);
    let seen = Contact::Visible(arena.position());

    // A current sighting starts the warning; it cannot fire immediately.
    assert!(gun.advance(&arena, muzzle, seen).is_none());
    for _ in 0..39 {
        assert!(gun.advance(&arena, muzzle, seen).is_none());
    }
    let projectile = gun
        .advance(&arena, muzzle, seen)
        .expect("Full warning elapsed");
    let position = projectile.position();
    println!("First projectile starts at {position:?}");

    // A shot flies through the same collision world; one impact consumes one health point.
    let mut projectiles = Projectiles::default();
    let mut robot = RobotHealth::default();
    projectiles
        .launch(projectile)
        .expect("A free projectile slot");
    for _ in 0..30 {
        projectiles.advance(&arena, &mut robot);
    }
    assert_eq!(robot.hits_remaining(), 2);
    assert!(robot.is_alive());
    assert_eq!(projectiles.iter().count(), 0);

    // Losing sight cancels the remaining volley. Remembered positions cannot authorize fire.
    assert!(gun.advance(&arena, muzzle, Contact::Unknown).is_none());
    gun.disable();
    assert_eq!(gun.phase(), Phase::Disabled);
}
