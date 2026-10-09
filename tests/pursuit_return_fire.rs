//! Drone shots require a warning and current unobstructed visual contact.
#![cfg(feature = "robots")]

#[expect(
    dead_code,
    reason = "Other arena and sight operations have dedicated tests."
)]
#[path = "../examples/robots/pursuit/arena.rs"]
mod arena;
#[path = "../examples/robots/pursuit/enemy.rs"]
mod enemy;
#[expect(dead_code, reason = "This test supplies the already filtered contact.")]
#[path = "../examples/robots/pursuit/sight.rs"]
mod sight;

use arena::Arena;
use bevy::math::Vec3;
use enemy::gun::{Gun, Phase};
use sight::Contact;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// Every action before a boundary must remain quiet, not just the final one.
fn wait_without_shot(gun: &mut Gun, arena: &Arena, origin: Vec3, contact: Contact, steps: usize) {
    for _ in 0..steps {
        assert!(gun.advance(arena, origin, contact).is_none());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn warning_precedes_three_spaced_shots_and_cooldown() {
    let arena = Arena::default();
    let origin = Vec3::new(0.0, 2.0, 0.0);
    let contact = Contact::Visible(arena.position());
    let mut gun = Gun::default();
    assert_eq!(gun.phase(), Phase::Idle);
    assert!(gun.advance(&arena, origin, contact).is_none());
    assert!(matches!(gun.phase(), Phase::Charging { .. }));
    wait_without_shot(&mut gun, &arena, origin, contact, 39);
    assert!(gun.advance(&arena, origin, contact).is_some());
    for _ in 0..2 {
        wait_without_shot(&mut gun, &arena, origin, contact, 5);
        assert!(gun.advance(&arena, origin, contact).is_some());
    }
    assert!(matches!(gun.phase(), Phase::Cooling { .. }));
    wait_without_shot(&mut gun, &arena, origin, contact, 60);
    assert_eq!(gun.phase(), Phase::Idle);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn remembered_hidden_and_invalid_contacts_cannot_complete_a_warning() {
    let arena = Arena::default();
    let origin = Vec3::new(0.0, 2.0, 0.0);
    let visible = Contact::Visible(arena.position());
    for lost in [
        Contact::Unknown,
        Contact::Remembered {
            point: arena.position(),
            age: std::time::Duration::ZERO,
        },
        Contact::Visible(Vec3::splat(f32::NAN)),
        Contact::Visible(Vec3::splat(f32::INFINITY)),
        Contact::Visible(Vec3::splat(1000.01)),
        Contact::Visible(origin),
        Contact::Visible(Vec3::new(0.0, 2.0, 20.001)),
        Contact::Visible(Vec3::new(-7.0, 1.7, -3.0)),
    ] {
        let mut gun = Gun::default();
        for _ in 0..40 {
            assert!(gun.advance(&arena, origin, visible).is_none());
        }
        assert!(gun.advance(&arena, origin, lost).is_none());
        assert_eq!(gun.phase(), Phase::Idle);
        assert!(gun.advance(&arena, origin, visible).is_none());
        assert_eq!(
            gun.phase(),
            Phase::Charging {
                remaining: std::time::Duration::from_millis(800)
            }
        );
    }
}

/// Finish one warning in an independent gun to obtain a valid launch.
fn shot(arena: &Arena, from: Vec3, point: Vec3) -> enemy::projectile::Projectile {
    let mut gun = Gun::default();
    for _ in 0..40 {
        assert!(gun.advance(arena, from, Contact::Visible(point)).is_none());
    }
    gun.advance(arena, from, Contact::Visible(point))
        .expect("Validated launch")
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn moving_projectiles_damage_once_and_three_hits_are_terminal() {
    use enemy::{projectile::Projectiles, robot_health::RobotHealth};
    let arena = Arena::default();
    let target = arena.position();
    let mut projectiles = Projectiles::default();
    let mut health = RobotHealth::default();
    assert_eq!(health.hits_remaining(), 3);
    for remaining in [2, 1, 0, 0] {
        projectiles
            .launch(shot(&arena, target - Vec3::Z * 2.0, target))
            .expect("Free slot");
        for _ in 0..4 {
            projectiles.advance(&arena, &mut health);
        }
        assert_eq!(projectiles.iter().count(), 1);
        projectiles.advance(&arena, &mut health);
        assert_eq!(health.hits_remaining(), remaining);
        assert_eq!(projectiles.iter().count(), 0);
        projectiles.advance(&arena, &mut health);
        assert_eq!(health.hits_remaining(), remaining);
    }
    assert!(!health.is_alive());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn projectiles_have_bounded_capacity_and_expire_without_steering() {
    use enemy::{
        projectile::{Full, Projectiles},
        robot_health::RobotHealth,
    };
    let arena = Arena::default();
    let from = Vec3::new(0.0, 10.0, 0.0);
    let shot = shot(&arena, from, from + Vec3::Y);
    let mut projectiles = Projectiles::default();
    let mut health = RobotHealth::default();
    for _ in 0..3 {
        projectiles.launch(shot).expect("Available slot");
    }
    assert_eq!(projectiles.launch(shot), Err(Full));
    for _ in 0..99 {
        projectiles.advance(&arena, &mut health);
    }
    assert_eq!(projectiles.iter().count(), 3);
    assert!(projectiles
        .iter()
        .all(|p| (p.position().y - 45.64).abs() < 0.001));
    projectiles.advance(&arena, &mut health);
    assert_eq!(projectiles.iter().count(), 0);
    assert_eq!(health.hits_remaining(), 3);
    projectiles
        .launch(shot)
        .expect("Expired slots can be reused");
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn cancelling_a_volley_pays_cooldown_and_disabling_is_absorbing() {
    let arena = Arena::default();
    let origin = Vec3::new(0.0, 2.0, 0.0);
    let contact = Contact::Visible(arena.position());
    let mut gun = Gun::default();
    for _ in 0..40 {
        assert!(gun.advance(&arena, origin, contact).is_none());
    }
    assert!(gun.advance(&arena, origin, contact).is_some());
    assert!(gun.advance(&arena, origin, Contact::Unknown).is_none());
    assert_eq!(
        gun.phase(),
        Phase::Cooling {
            remaining: std::time::Duration::from_millis(1200)
        }
    );
    for _ in 0..60 {
        assert!(gun.advance(&arena, origin, Contact::Unknown).is_none());
    }
    assert_eq!(gun.phase(), Phase::Idle);
    gun.disable();
    for _ in 0..200 {
        assert!(gun.advance(&arena, origin, contact).is_none());
    }
    assert_eq!(gun.phase(), Phase::Disabled);
    gun = Gun::default();
    assert!(gun.advance(&arena, origin, contact).is_none());
    assert!(matches!(gun.phase(), Phase::Charging { .. }));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_muzzles_fail_closed_and_range_includes_twenty_metres() {
    let arena = Arena::default();
    for origin in [
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::NEG_INFINITY),
        Vec3::splat(-1000.01),
    ] {
        let mut gun = Gun::default();
        assert!(gun
            .advance(&arena, origin, Contact::Visible(Vec3::Y * 10.0))
            .is_none());
        assert_eq!(gun.phase(), Phase::Idle);
    }
    for origin in [
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(1000.0, 10.0, 0.0),
        Vec3::new(-1000.0, 10.0, 0.0),
    ] {
        let mut gun = Gun::default();
        assert!(gun
            .advance(&arena, origin, Contact::Visible(origin + Vec3::Z * 20.0))
            .is_none());
        assert!(matches!(gun.phase(), Phase::Charging { .. }));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn a_wall_stops_a_launched_projectile_and_moving_out_of_its_line_dodges_it() {
    use enemy::{projectile::Projectiles, robot_health::RobotHealth};
    let mut arena = Arena::default();
    let from = Vec3::new(-7.0, 1.7, 1.0);
    let mut projectiles = Projectiles::default();
    let mut health = RobotHealth::default();
    projectiles
        .launch(shot(&arena, from, from - Vec3::Z * 0.5))
        .expect("Clear launch toward wall");
    for _ in 0..3 {
        projectiles.advance(&arena, &mut health);
    }
    assert_eq!(projectiles.iter().count(), 0);
    assert_eq!(health.hits_remaining(), 3);
    let point = arena.position();
    projectiles
        .launch(shot(&arena, point - Vec3::Z * 2.0, point))
        .expect("Clear launch");
    for _ in 0..20 {
        arena.step(arena::Movement::Right);
    }
    for _ in 0..10 {
        projectiles.advance(&arena, &mut health);
    }
    assert_eq!(health.hits_remaining(), 3);
    assert_eq!(projectiles.iter().count(), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn window_and_pipe_openings_permit_warning_and_projectile_travel() {
    use enemy::{projectile::Projectiles, robot_health::RobotHealth};
    let arena = Arena::default();
    for from in [Vec3::new(-5.0, 1.7, 1.0), Vec3::new(5.0, 1.7, 5.0)] {
        let mut projectiles = Projectiles::default();
        let mut health = RobotHealth::default();
        projectiles
            .launch(shot(&arena, from, from - Vec3::Z * 4.0))
            .expect("Open firing opportunity");
        for _ in 0..6 {
            projectiles.advance(&arena, &mut health);
        }
        let position = projectiles
            .iter()
            .next()
            .expect("Passed opening")
            .position();
        assert!(position.distance(from - Vec3::Z * 2.16) < 0.001);
        assert_eq!(health.hits_remaining(), 3);
    }
}
