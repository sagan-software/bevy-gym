//! Public drone action, dynamics, and episode contracts.

#![cfg(feature = "robots")]

use bevy::math::{Quat, Vec3};
use bevy::prelude::{App, FixedUpdate, Messages, MinimalPlugins};
use bevy_gym::robots::{DroneAction, DroneHover, InvalidDroneAction};
use bevy_gym::{ActionRequest, ActionResponse, BevyGymPlugin, Env, EpisodeStatus, TransitionEvent};

/// Compare scalar physics values with an explicit absolute tolerance.
fn close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}, tolerance {tolerance}"
    );
}

#[test]
fn motor_commands_validate_each_boundary_before_simulation() {
    for value in [0.0, -0.0, 0.5, 1.0] {
        let action = DroneAction::try_from([value; 4]).expect("valid motor fraction");
        assert_eq!(
            action.fractions().map(f32::to_bits),
            [value; 4].map(f32::to_bits)
        );
    }
    for motor in 0..4 {
        for invalid in [
            -f32::from_bits(1),
            f32::from_bits(1.0_f32.to_bits() + 1),
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut fractions = [0.5; 4];
            *fractions.get_mut(motor).expect("one of four motors") = invalid;
            assert_eq!(DroneAction::try_from(fractions), Err(InvalidDroneAction));
        }
    }
    assert_eq!(
        InvalidDroneAction.to_string(),
        "drone motor commands must be four finite fractions between 0 and 1"
    );
    assert!(std::error::Error::source(&InvalidDroneAction).is_none());
}

#[test]
fn equal_seeds_reproduce_resets_and_rollouts() {
    let mut left = DroneHover::default();
    let mut right = DroneHover::default();
    let initial = left.reset(Some(42));
    assert_eq!(initial, right.reset(Some(42)));
    assert_eq!(initial.observation.orientation(), Quat::IDENTITY);
    assert_eq!(initial.observation.linear_velocity(), Vec3::ZERO);
    assert_eq!(initial.observation.angular_velocity(), Vec3::ZERO);
    for fractions in [[0.5; 4], [0.4, 0.6, 0.6, 0.4], [0.0; 4]] {
        let action = DroneAction::try_from(fractions).expect("valid fractions");
        assert_eq!(left.step(action), right.step(action));
    }
    assert_eq!(initial, left.reset(Some(42)));
    let next = left.reset(None);
    assert_ne!(initial, next);
    assert_eq!(initial, right.reset(Some(42)));
    assert_eq!(next, right.reset(None));
}

#[test]
fn power_off_accelerates_downward_at_gravity() {
    let mut drone = DroneHover::default();
    let initial = drone.reset(Some(42)).observation;
    let off = DroneAction::try_from([0.0; 4]).expect("motors off");
    let step = drone.step(off);
    close(step.observation.linear_velocity().y, -9.81 * 0.020, 1e-5);
    close(step.observation.linear_velocity().x, 0.0, 1e-6);
    close(step.observation.linear_velocity().z, 0.0, 1e-6);
    assert!(step.observation.position().y < initial.position().y);
    assert_eq!(step.observation.angular_velocity(), Vec3::ZERO);
    assert_eq!(step.status, EpisodeStatus::Continuing);
}

#[test]
fn symmetric_hover_balances_gravity_and_all_moments() {
    let mut drone = DroneHover::default();
    let initial = drone.reset(Some(42)).observation;
    let hover = DroneAction::try_from([0.5; 4]).expect("balanced motors");
    for _ in 0..500 {
        let step = drone.step(hover);
        assert_eq!(step.status, EpisodeStatus::Continuing);
        assert!(step.observation.position().distance(initial.position()) < 1e-4);
        assert!(step.observation.linear_velocity().length() < 1e-5);
        assert!(step.observation.angular_velocity().length() < 1e-5);
        close(
            step.observation.orientation().dot(Quat::IDENTITY),
            1.0,
            1e-6,
        );
        assert!((0.0..=1.0).contains(&step.reward));
    }
}

#[test]
fn each_motor_produces_the_expected_roll_pitch_and_yaw_direction() {
    // A body's moment is offset cross force, plus alternating rotor reaction.
    for (fractions, direction) in [
        ([1.0, 0.0, 0.0, 0.0], Vec3::new(1.0, 1.0, -1.0)),
        ([0.0, 1.0, 0.0, 0.0], Vec3::new(1.0, -1.0, 1.0)),
        ([0.0, 0.0, 1.0, 0.0], Vec3::new(-1.0, 1.0, 1.0)),
        ([0.0, 0.0, 0.0, 1.0], Vec3::new(-1.0, -1.0, -1.0)),
    ] {
        let mut drone = DroneHover::default();
        drone.reset(Some(42));
        let action = DroneAction::try_from(fractions).expect("one powered motor");
        let step = drone.step(action);
        assert!((step.observation.angular_velocity() * direction)
            .cmpgt(Vec3::ZERO)
            .all());
    }
}

#[test]
fn tilted_thrust_changes_horizontal_velocity() {
    let mut drone = DroneHover::default();
    drone.reset(Some(42));
    let tilt = DroneAction::try_from([0.7, 0.3, 0.3, 0.7]).expect("left motors stronger");
    let mut step = drone.step(tilt);
    for _ in 0..9 {
        step = drone.step(tilt);
    }
    let before = step.observation;
    assert!((before.orientation() * Vec3::Y).x > 0.1);
    let hover = DroneAction::try_from([0.5; 4]).expect("balanced motors");
    let after = drone.step(hover).observation;
    assert!(after.linear_velocity().x > before.linear_velocity().x);
}

#[test]
fn ground_contact_and_region_exit_end_until_reset() {
    for (fractions, terminal_height) in [([0.0; 4], 0.0..=0.2), ([1.0; 4], 10.0..=10.2)] {
        let mut drone = DroneHover::default();
        let initial = drone.reset(Some(42));
        let action = DroneAction::try_from(fractions).expect("valid motor commands");
        let terminal = (0..500)
            .map(|_| drone.step(action))
            .find(bevy_gym::Step::is_done)
            .expect("flight ends at ground or ceiling");
        assert_eq!(terminal.status, EpisodeStatus::Terminated);
        assert!(terminal_height.contains(&terminal.observation.position().y));
        close(terminal.reward as f32, 0.0, f32::EPSILON);
        for _ in 0..3 {
            let repeated = drone.step(action);
            assert_eq!(repeated, terminal);
        }
        assert_eq!(initial, drone.reset(Some(42)));
        let mut fresh = DroneHover::default();
        fresh.reset(Some(42));
        let first = drone.step(action);
        assert_eq!(first.status, EpisodeStatus::Continuing);
        assert_eq!(first, fresh.step(action));
    }
}

#[test]
fn bevy_runs_two_drone_environments_through_typed_messages() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(BevyGymPlugin::new(|_| DroneHover::default()).with_envs(2));
    app.update();
    let requests: Vec<_> = app
        .world_mut()
        .resource_mut::<Messages<ActionRequest<DroneHover>>>()
        .drain()
        .collect();
    assert_eq!(requests.len(), 2);
    let hover = DroneAction::try_from([0.5; 4]).expect("balanced motors");
    for request in requests {
        app.world_mut()
            .write_message(ActionResponse::<DroneHover> {
                entity: request.entity,
                action: hover,
            })
            .expect("plugin registered its action message");
    }
    app.world_mut().run_schedule(FixedUpdate);
    let transitions: Vec<_> = app
        .world_mut()
        .resource_mut::<Messages<TransitionEvent<DroneHover>>>()
        .drain()
        .collect();
    assert_eq!(transitions.len(), 2);
    for event in transitions {
        assert_eq!(event.transition.status, EpisodeStatus::Continuing);
        let before = event.transition.observation.position();
        let after = event.transition.next_observation.position();
        assert!(after.distance(before) < 1e-5);
    }
}
