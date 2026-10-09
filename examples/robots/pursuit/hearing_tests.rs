//! Only actual grounded strides and accepted shots produce hearing observations.

use super::*;
use std::time::Duration;

#[test]
fn accepted_shots_emit_once_and_rejected_shots_do_not_refresh_hearing() {
    let mut game = Game::default();
    game.act(firing::Action::Fire(Dir3::Y));
    assert!(game.hearing.latest().is_none());
    game.act(firing::Action::PickUp);
    assert!(game.hearing.latest().is_none());
    game.act(firing::Action::Fire(Dir3::Y));
    assert_eq!(
        game.hearing.latest().expect("Accepted shot").noise(),
        hearing::Noise::Gunshot
    );
    game.hearing.advance(Duration::from_millis(100));
    let previous = game.hearing.latest();
    game.act(firing::Action::Fire(Dir3::Y));
    assert_eq!(game.hearing.latest(), previous);
}

#[test]
fn grounded_walking_is_audible_and_blocked_walking_is_silent() {
    let mut game = Game::default();
    game.act(firing::Action::PickUp);
    for _ in 0..30 {
        game.step(Movement::Forward);
    }
    assert_eq!(
        game.hearing
            .latest()
            .expect("Walking within eight metres")
            .noise(),
        hearing::Noise::Footstep
    );
    for _ in 0..95 {
        game.step(Movement::Forward);
    }
    for _ in 0..100 {
        game.step(Movement::Left);
    }
    let blocked = game.arena.position();
    assert!((-1.7..-1.3).contains(&blocked.x));
    game.act(firing::Action::Fire(Dir3::Y));
    assert_eq!(
        game.hearing
            .latest()
            .expect("Listener is in audible range")
            .noise(),
        hearing::Noise::Gunshot
    );
    for _ in 0..200 {
        game.step(Movement::Left);
    }
    let drift = game.arena.position().distance(blocked);
    assert!(drift < 0.01, "Wall drift {drift} m");
    assert!(game.hearing.latest().is_none());
    game.reset();
    assert!(game.hearing.latest().is_none());
}

#[test]
fn failure_body_death_and_reset_clear_hearing() {
    let mut game = Game::default();
    game.act(firing::Action::PickUp);
    game.act(firing::Action::Fire(Dir3::Y));
    assert!(game.hearing.latest().is_some());
    game.flight.fail("Controller failed".to_owned());
    game.step(Movement::Idle);
    assert!(game.hearing.latest().is_none());
    game.advance_combat(Duration::from_millis(250));
    game.act(firing::Action::Fire(Dir3::Y));
    assert!(game.hearing.latest().is_none());
    game.reset();
    assert!(game.hearing.latest().is_none());
    assert!(game.footstep_distance.abs() < f32::EPSILON);
    game.act(firing::Action::PickUp);
    let aim =
        Dir3::new(game.combat.target().position() - firing::Combat::origin(game.arena.position()))
            .expect("Aim");
    for _ in 0..6 {
        game.advance_combat(Duration::from_millis(250));
        game.act(firing::Action::Fire(aim));
        game.advance_combat(Duration::from_millis(100));
    }
    assert!(!game.combat.target().health().is_alive());
    assert!(game.hearing.latest().is_none());
    for _ in 0..20 {
        game.step(Movement::Forward);
    }
    assert!(game.hearing.latest().is_none());
}

#[test]
fn an_empty_magazine_cannot_refresh_the_last_noise() {
    let mut game = Game::default();
    game.act(firing::Action::PickUp);
    for _ in 0..12 {
        game.advance_combat(Duration::from_millis(250));
        game.act(firing::Action::Fire(Dir3::Y));
    }
    assert_eq!(game.combat.rounds(), 0);
    game.hearing.advance(Duration::from_millis(100));
    let previous = game.hearing.latest();
    game.advance_combat(Duration::from_millis(250));
    game.act(firing::Action::Fire(Dir3::Y));
    assert_eq!(game.hearing.latest(), previous);
}

#[test]
fn airborne_travel_does_not_count_as_footsteps() {
    let mut game = Game {
        arena: Arena::airborne(),
        footstep_distance: 0.79,
        ..Game::default()
    };
    let initial = game.arena.position();
    game.step(Movement::Forward);
    assert!(!game.arena.is_grounded());
    assert!(game.arena.position().z < initial.z);
    assert!(game.arena.position().y < initial.y);
    assert!((game.footstep_distance - 0.79).abs() < f32::EPSILON);
    assert!(game.hearing.latest().is_none());
    for _ in 0..100 {
        game.step(Movement::Forward);
    }
    assert!(game.arena.is_grounded());
    assert_eq!(
        game.hearing
            .latest()
            .expect("Walking resumes after landing")
            .noise(),
        hearing::Noise::Footstep
    );
}
