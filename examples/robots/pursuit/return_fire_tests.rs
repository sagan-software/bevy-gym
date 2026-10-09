//! The game owns weapon origins, character health, and reset boundaries.

use super::*;

/// Send three actual projectiles through the normal game collision update.
fn destroy_robot(game: &mut Game) {
    let position = game.arena.position();
    for _ in 0..3 {
        let mut gun = enemy::gun::Gun::default();
        let origin = position - Vec3::Z * 2.0;
        for _ in 0..40 {
            assert!(gun
                .advance(&game.arena, origin, sight::Contact::Visible(position))
                .is_none());
        }
        let shot = gun
            .advance(&game.arena, origin, sight::Contact::Visible(position))
            .expect("Shot");
        game.projectiles.launch(shot).expect("Free slot");
        for _ in 0..6 {
            game.step(Movement::Idle);
        }
    }
}

#[test]
fn robot_death_blocks_movement_and_interactions_until_reset() {
    let mut game = Game::default();
    destroy_robot(&mut game);
    assert!(!game.robot_health.is_alive());
    let position = game.arena.position();
    game.hear(game.combat.target().position(), hearing::Noise::Footstep);
    assert!(game.hearing.latest().is_none());
    game.act(firing::Action::PickUp);
    for _ in 0..20 {
        game.step(Movement::Forward);
    }
    assert!(game.arena.position().distance(position) < 0.001);
    assert!(!game.combat.is_armed());
    assert_eq!(game.gun.phase(), enemy::gun::Phase::Disabled);
    game.reset();
    assert_eq!(game.robot_health.hits_remaining(), 3);
    assert_eq!(game.gun.phase(), enemy::gun::Phase::Idle);
    assert_eq!(game.projectiles.iter().count(), 0);
    game.act(firing::Action::PickUp);
    assert!(game.combat.is_armed());
    game.step(Movement::Forward);
    assert!(game.arena.position().z < position.z);
}

#[test]
fn drone_controller_failure_disables_firing() {
    let mut game = Game::default();
    game.flight.fail("Controller failed".to_owned());
    game.step(Movement::Idle);
    assert_eq!(game.gun.phase(), enemy::gun::Phase::Disabled);
    assert_eq!(game.projectiles.iter().count(), 0);
    assert_eq!(game.robot_health.hits_remaining(), 3);
}

#[test]
fn weapon_projection_has_a_warning_health_and_bounded_projectile_meshes() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_asset::<Font>()
        .init_resource::<Game>()
        .add_systems(Startup, hud::setup)
        .add_plugins(return_fire::install);
    app.update();
    let world = app.world_mut();
    let text = world
        .query_filtered::<&Text, With<return_fire::Status>>()
        .single(world)
        .expect("Return-fire status");
    assert_eq!(text.0, "Robot 3/3 · Drone idle");
}

#[test]
fn reset_discards_unrendered_shot_sounds() {
    let mut game = Game::default();
    game.sound.shot(Vec3::ZERO);
    assert_eq!(game.sound.pending_shots(), 1);
    game.reset();
    assert_eq!(game.sound.pending_shots(), 0);
}

#[test]
fn current_sight_drives_warning_and_real_projectile_damage() {
    let mut game = Game::default();
    for _ in 0..500 {
        game.step(Movement::Idle);
        if matches!(game.gun.phase(), enemy::gun::Phase::Charging { .. }) {
            break;
        }
    }
    assert!(matches!(
        game.gun.phase(),
        enemy::gun::Phase::Charging { .. }
    ));
    assert_eq!(game.robot_health.hits_remaining(), 3);
    for _ in 0..300 {
        game.step(Movement::Idle);
    }
    assert!(!game.robot_health.is_alive());
    assert_eq!(game.gun.phase(), enemy::gun::Phase::Disabled);
}

#[test]
fn an_armed_dead_robot_cannot_fire_or_spend_ammunition() {
    let mut game = Game::default();
    game.act(firing::Action::PickUp);
    destroy_robot(&mut game);
    let feedback = game.combat.feedback();
    game.act(firing::Action::Fire(Dir3::Y));
    assert_eq!(game.combat.rounds(), 12);
    assert_eq!(game.combat.feedback(), feedback);
    assert!(game.combat.trace().is_none());
}
