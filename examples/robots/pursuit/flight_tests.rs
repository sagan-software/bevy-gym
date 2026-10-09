//! Exercise live flight through the same game actions as the player.

use super::*;
use bevy_gym::robots::{DroneMotor, DroneMotorState};

#[test]
fn sight_driven_navigation_finds_the_robot_and_survives_a_minute() {
    let mut game = Game::default();
    // Keep the target alive to measure tracking throughout the full flight.
    game.gun.disable();
    let mut visible = 0;
    for _ in 0..3000 {
        game.step(Movement::Idle);
        assert!(game.flight.error().is_none());
        assert!(game.combat.target().health().is_alive());
        visible += usize::from(matches!(game.sight.contact(), sight::Contact::Visible(_)));
    }
    assert!(
        visible >= 1800,
        "At least 60% visible samples, received {visible}"
    );
    assert!(
        game.flight
            .observation()
            .position()
            .distance(Vec3::new(0.0, 2.0, 0.0))
            > 1.0
    );
}

#[test]
fn body_death_stops_flight_before_another_policy_action() {
    let mut game = Game::default();
    game.act(firing::Action::PickUp);
    let point = game.combat.target().position();
    let aim = Dir3::new(point - firing::Combat::origin(game.arena.position())).expect("Aim");
    for _ in 0..6 {
        game.combat.advance(std::time::Duration::from_millis(250));
        game.act(firing::Action::Fire(aim));
    }
    assert!(!game.combat.target().health().is_alive());
    let observation = game.flight.observation();
    game.step(Movement::Idle);
    assert_eq!(game.flight.observation(), observation);
    while game.combat.pop_effect().is_some() {}
    game.combat.crash();
    assert_eq!(game.combat.pop_effect(), None);
}

#[test]
fn rotor_hits_disable_motor_forces_and_a_crash_ends_combat_once() {
    let mut game = Game::default();
    game.act(firing::Action::PickUp);
    while game.combat.pop_effect().is_some() {}
    let point = game.combat.target().rotor_centre(DroneMotor::RearRight);
    let aim = Dir3::new(point - firing::Combat::origin(game.arena.position())).expect("Aim");
    for _ in 0..2 {
        game.combat.advance(std::time::Duration::from_millis(250));
        game.act(firing::Action::Fire(aim));
    }
    game.step(Movement::Idle);
    assert_eq!(
        game.flight.observation().motor_state(DroneMotor::RearRight),
        DroneMotorState::Failed
    );
    for _ in 0..500 {
        game.step(Movement::Idle);
        if !game.combat.target().health().is_alive() {
            break;
        }
    }
    assert!(!game.combat.target().health().is_alive());
    assert_eq!(game.sight.contact(), sight::Contact::Unknown);
    assert!(game.hearing.latest().is_none());
    let terminal = game.flight.observation();
    assert_eq!(game.combat.target().position(), terminal.position());
    let mut deaths = 0;
    while let Some(event) = game.combat.pop_effect() {
        if event == firing::Effect::Destroyed {
            deaths += 1;
        }
    }
    assert_eq!(deaths, 1);
    game.step(Movement::Idle);
    assert_eq!(game.flight.observation(), terminal);
    assert_eq!(game.combat.pop_effect(), None);
    game.reset();
    assert!(game.combat.target().health().is_alive());
    assert_eq!(
        game.flight.observation().motor_state(DroneMotor::RearRight),
        DroneMotorState::Working
    );
}

#[test]
fn learned_hover_moves_the_hitboxes_and_reset_replays_the_same_episode() {
    let mut game = Game::default();
    let initial = game.combat.target().position();
    for _ in 0..25 {
        game.step(Movement::Idle);
    }
    let observation = game.flight.observation();
    assert_ne!(observation.position(), initial);
    assert_eq!(game.combat.target().position(), observation.position());
    // Hitbox construction normalizes the solver quaternion once more.
    assert!(game
        .combat
        .target()
        .rotation()
        .abs_diff_eq(observation.orientation(), 1e-6));
    game.reset();
    for _ in 0..25 {
        game.step(Movement::Idle);
    }
    assert_eq!(game.flight.observation(), observation);
}
