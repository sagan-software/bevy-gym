//! Game integration keeps camera observations tied to simulation and episode lifetime.

use super::*;
use std::time::Duration;

/// Seed a clear observation independently of the drone's current heading.
fn seed_sighting(game: &mut Game) {
    game.sight.sample(
        &game.arena,
        Vec3::new(0.0, 10.0, 0.0),
        Dir3::Z,
        Vec3::new(0.0, 9.4, 5.0),
        Duration::ZERO,
    );
}

#[test]
fn reset_and_controller_failure_clear_sight_memory() {
    let mut game = Game::default();
    seed_sighting(&mut game);
    assert!(matches!(game.sight.contact(), sight::Contact::Visible(_)));
    game.reset();
    assert_eq!(game.sight.contact(), sight::Contact::Unknown);
    seed_sighting(&mut game);
    game.flight.fail("Test controller failure".to_owned());
    game.step(Movement::Idle);
    assert_eq!(game.sight.contact(), sight::Contact::Unknown);
}

#[test]
fn body_death_clears_sight_before_the_next_physics_action() {
    let mut game = Game::default();
    seed_sighting(&mut game);
    game.act(firing::Action::PickUp);
    let aim =
        Dir3::new(game.combat.target().position() - firing::Combat::origin(game.arena.position()))
            .expect("Aim");
    for _ in 0..6 {
        game.combat.advance(Duration::from_millis(250));
        game.act(firing::Action::Fire(aim));
    }
    assert!(!game.combat.target().health().is_alive());
    assert_eq!(game.sight.contact(), sight::Contact::Unknown);
}

#[test]
fn body_camera_observes_the_actual_character_during_hover() {
    let mut game = Game::default();
    assert_eq!(game.sight.contact(), sight::Contact::Unknown);
    for step in 0..300 {
        let movement = if step < 150 {
            Movement::Forward
        } else if step < 200 {
            Movement::Right
        } else {
            Movement::Backward
        };
        game.step(movement);
        if let sight::Contact::Visible(point) = game.sight.contact() {
            assert_eq!(point.x, game.arena.position().x);
            assert_eq!(point.z, game.arena.position().z);
            game.step(Movement::Right);
            if let sight::Contact::Visible(moved) = game.sight.contact() {
                assert!(moved.x > point.x);
                assert_eq!(moved.x, game.arena.position().x);
                return;
            }
        }
    }
    panic!("The body-mounted camera never saw the moving character");
}

#[test]
fn physical_death_clears_memory_and_camera_pose_is_body_local() {
    let mut game = Game::default();
    seed_sighting(&mut game);
    game.combat.crash();
    game.step(Movement::Idle);
    assert_eq!(game.sight.contact(), sight::Contact::Unknown);
    game.reset();
    for _ in 0..100 {
        game.step(Movement::Idle);
    }
    let eye = game.eye();
    let target = game.combat.target();
    assert!(eye.rotation.abs_diff_eq(target.rotation(), 1e-6));
    let local = target.rotation().inverse() * (Vec3::from(eye.translation) - target.position());
    assert!(local.abs_diff_eq(DRONE_EYE, 1e-6));
}
