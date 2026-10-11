//! Walking-task mechanics through the existing public environment boundary.
#![cfg(feature = "robots")]

use bevy::math::Vec2;
use bevy_gym::robots::{
    DroidAction, DroidActuator, DroidBody, DroidStanding, DroidVelocityCommand, DroidWalking,
    DroidWalkingStatus, InvalidDroidVelocityCommand,
};
use bevy_gym::{Env, EpisodeStatus};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// A walking reset retains its command and the existing articulated body model.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reset_retains_the_validated_command_and_original_body() {
    assert_eq!(DroidWalkingStatus::default(), DroidWalkingStatus::Running);
    let command = DroidVelocityCommand::try_from(Vec2::new(0.5, 0.25)).expect("walking command");
    let mut walking = DroidWalking::new(command);
    let observation = walking.reset(Some(42)).observation;
    let standing = DroidStanding::default().reset(Some(42)).observation;
    assert_eq!(observation.command(), command);
    assert_eq!(observation.physical(), &standing);
    assert_eq!(
        DroidWalking::POLICY_INTERVAL,
        DroidStanding::POLICY_INTERVAL
    );
    assert_eq!(walking.reset(Some(42)).observation, observation);
    assert_ne!(walking.reset(None).observation, observation);
    assert!(format!("{walking:?}").starts_with("DroidWalking"));
}

/// A velocity request cannot replace torques; an unpowered body falls and freezes.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn passive_torques_fall_and_terminal_actions_freeze_until_reset() {
    let mut walking = DroidWalking::default();
    let initial = walking.reset(Some(42));
    assert_eq!(initial.info, DroidWalkingStatus::Running);
    let passive = DroidAction::try_from([0.0; 26]).expect("isolated passive fixture");
    let mut last = walking.step(passive);
    assert_eq!(last.status, EpisodeStatus::Continuing);
    assert_eq!(last.info, DroidWalkingStatus::Running);
    assert!(
        last.observation
            .physical()
            .body(DroidBody::Pelvis)
            .linear_velocity()
            .y
            < -0.1
    );
    assert!((0.0..=1.0).contains(&last.reward));
    for _ in 1..1_000 {
        if last.is_done() {
            break;
        }
        last = walking.step(passive);
    }
    assert_eq!(last.status, EpisodeStatus::Terminated);
    assert!(matches!(last.info, DroidWalkingStatus::Fallen(_)));
    assert_eq!(last.reward.to_bits(), 0.0_f64.to_bits());
    for fraction in [-1.0, 0.0, 1.0] {
        let action = DroidAction::try_from([fraction; 26]).expect("isolated bounded fixture");
        assert_eq!(walking.step(action), last);
    }
    assert_eq!(walking.reset(Some(42)), initial);
}

/// Valid commands retain exact components, including both signed-zero representations.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn velocity_commands_validate_without_normalization() {
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    for velocity in [
        Vec2::ZERO,
        Vec2::new(-0.0, 0.0),
        Vec2::new(0.0, -0.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(-1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(0.0, -1.0),
        Vec2::new(0.3, 0.4),
        Vec2::new(-diagonal, diagonal),
    ] {
        let command = DroidVelocityCommand::try_from(velocity).expect("bounded velocity");
        assert_eq!(
            command.velocity().to_array().map(f32::to_bits),
            velocity.to_array().map(f32::to_bits)
        );
    }
}

/// Invalid components and represented speeds fail without normalization or an error source.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_velocity_commands_preserve_the_validation_error() {
    let diagonal = std::f32::consts::FRAC_1_SQRT_2;
    for component in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        1.000_000_1,
        -1.000_000_1,
    ] {
        for velocity in [Vec2::new(component, 0.0), Vec2::new(0.0, component)] {
            assert_eq!(
                DroidVelocityCommand::try_from(velocity),
                Err(InvalidDroidVelocityCommand)
            );
        }
    }
    let outside_diagonal = f32::from_bits(diagonal.to_bits() + 1);
    for velocity in [
        Vec2::new(outside_diagonal, outside_diagonal),
        Vec2::new(0.6, 0.8),
        Vec2::new(1.0, f32::from_bits(1)),
        Vec2::new(1.0, 1e-20),
        Vec2::new(-1.0, -1e-20),
    ] {
        assert_eq!(
            DroidVelocityCommand::try_from(velocity),
            Err(InvalidDroidVelocityCommand)
        );
    }
    assert_eq!(
        DroidVelocityCommand::default().velocity(),
        Vec2::new(0.5, 0.0)
    );
    assert_eq!(
        InvalidDroidVelocityCommand.to_string(),
        "droid walking commands must be finite X/Z velocities with speed at most 1 m/s"
    );
    assert!(std::error::Error::source(&InvalidDroidVelocityCommand).is_none());
}

/// Opposing lesson commands cannot alter physics under the same isolated action fixture.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn commands_do_not_select_or_modify_actuation() {
    let command = DroidVelocityCommand::try_from(Vec2::new(-0.5, 0.0)).expect("reverse command");
    let mut forward = DroidWalking::default();
    let mut reverse = DroidWalking::new(command);
    forward.reset(Some(42));
    reverse.reset(Some(42));
    let action = DroidAction::try_from([0.1; 26]).expect("isolated torque fixture");
    for _ in 0..128 {
        let forward = forward.step(action);
        let reverse = reverse.step(action);
        assert_eq!(
            forward.observation.physical(),
            reverse.observation.physical()
        );
        assert_eq!(forward.status, reverse.status);
        assert_eq!(forward.info, reverse.info);
        assert_ne!(forward.observation.command(), reverse.observation.command());
    }
}

/// All signed actuator requests use the unchanged standing body and force operator.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn every_walking_actuator_reuses_the_original_physical_operator() {
    let mut walking = DroidWalking::default();
    let mut standing = DroidStanding::default();
    for actuator in DroidActuator::ALL {
        for fraction in [-1.0, 1.0] {
            let mut fractions = [0.0; 26];
            *fractions
                .get_mut(actuator as usize)
                .expect("closed actuator identity") = fraction;
            let action = DroidAction::try_from(fractions).expect("isolated bounded fixture");
            walking.reset(Some(42));
            standing.reset(Some(42));
            let result = walking.step(action);
            let reference = standing.step(action);
            assert_eq!(result.observation.physical(), &reference.observation);
            assert_eq!(result.status, reference.status);
        }
    }
}
