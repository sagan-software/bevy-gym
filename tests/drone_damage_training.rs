//! Damage-aware observations and training through typed drone actions.
#![cfg(feature = "robots")]

#[path = "../examples/robots/damage/training.rs"]
mod damage_training;
#[expect(
    unused_imports,
    dead_code,
    reason = "This test reuses only part of the shared tutorial facade."
)]
#[path = "../examples/robots/learning/mod.rs"]
mod learning;

#[path = "../examples/robots/damage/assessment.rs"]
mod assessment;

use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor, DroneMotorState};
use bevy_gym::Env;
use damage_training::{encode, new_agent, DamageTask};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn health_inputs_follow_the_existing_motor_order_without_changing_motion_inputs() {
    let mut drone = DroneHover::default();
    let healthy = drone.reset(Some(42)).observation;
    assert_eq!(&encode(healthy)[..12], &learning::encode(healthy));
    assert_eq!(&encode(healthy)[12..], &[1.0; 4]);
    for (index, motor) in DroneMotor::ALL.into_iter().enumerate() {
        drone.reset(Some(42));
        drone.fail_motor(motor).unwrap();
        let mut expected = [1.0; 4];
        expected[index] = 0.0;
        let damaged = encode(drone.observation());
        assert_eq!(&damaged[..12], &learning::encode(healthy));
        assert_eq!(&damaged[12..], &expected);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn fixed_failure_is_observed_before_the_first_command_and_restored_on_reset() {
    let mut task = DamageTask::front_left();
    let initial = task.reset(Some(42));
    assert_eq!(
        initial.observation.motor_state(DroneMotor::FrontLeft),
        DroneMotorState::Failed
    );
    let action = DroneAction::try_from([0.5; 4]).unwrap();
    task.step(action);
    assert_eq!(task.reset(Some(42)), initial);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn real_damage_batches_update_a_separate_sixteen_input_policy() {
    let mut agent = new_agent(7).unwrap();
    let policy = agent.policy();
    let mut batch = learning::RecoveryBatch::with_task(
        7,
        &policy,
        DamageTask::front_left,
        |observation| encode(*observation),
        std::num::NonZeroU16::new(500).unwrap(),
    );
    let samples = batch.collect(&policy).unwrap();
    assert_eq!(
        samples.iter().map(|s| s.observations.len()).sum::<usize>(),
        512
    );
    assert!(samples
        .iter()
        .flat_map(|s| &s.observations)
        .all(|o| o.len() == 16 && o[12..] == [0.0, 1.0, 1.0, 1.0]));
    let observation = samples[0].observations[0].clone();
    let before = policy
        .mean_action(&observation, &policy.initial_memory())
        .unwrap();
    assert_eq!(agent.update(&samples).unwrap().valid_samples, 512);
    let after = agent
        .policy()
        .mean_action(&observation, &policy.initial_memory())
        .unwrap();
    assert_ne!(before.action, after.action);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn scheduled_failures_cover_every_corner_and_both_times_without_moving_at_failure() {
    let mut seen = [[false; 2]; 4];
    let half = DroneAction::try_from([0.5; 4]).unwrap();
    for seed in 0..64 {
        let mut task = DamageTask::scheduled();
        let mut intact = DroneHover::default();
        assert_eq!(task.reset(Some(seed)), intact.reset(Some(seed)));
        for step in 1..=250 {
            let result = task.step(half);
            let reference = intact.step(half);
            assert_eq!(
                result.observation.position(),
                reference.observation.position()
            );
            assert_eq!(
                result.observation.orientation(),
                reference.observation.orientation()
            );
            if let Some(index) = DroneMotor::ALL
                .iter()
                .position(|motor| result.observation.motor_state(*motor) == DroneMotorState::Failed)
            {
                assert!(step == 100 || step == 250);
                seen[index][usize::from(step == 250)] = true;
                break;
            }
            assert!(step < 250, "every schedule must fail by the last boundary");
        }
    }
    assert_eq!(seen, [[true; 2]; 4]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn seeded_and_continuing_reset_streams_repeat_without_cross_episode_damage() {
    let mut left = DamageTask::scheduled();
    let mut right = DamageTask::scheduled();
    let half = DroneAction::try_from([0.5; 4]).unwrap();
    for seed in [Some(42), None, None, Some(42)] {
        assert_eq!(left.reset(seed), right.reset(seed));
        for _ in 0..300 {
            assert_eq!(left.step(half), right.step(half));
        }
    }
    let mut hover = DamageTask::hover();
    assert!(DroneMotor::ALL.into_iter().all(|motor| hover
        .reset(Some(42))
        .observation
        .motor_state(motor)
        == DroneMotorState::Working));
    for _ in 0..500 {
        let step = hover.step(half);
        assert!(!step.is_done());
        assert!(DroneMotor::ALL
            .into_iter()
            .all(|motor| step.observation.motor_state(motor) == DroneMotorState::Working));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn termination_prevents_a_pending_failure_and_reset_restores_the_schedule() {
    let mut task = DamageTask::scheduled();
    let initial = task.reset(Some(42));
    let off = DroneAction::try_from([0.0; 4]).unwrap();
    let terminal = loop {
        let step = task.step(off);
        if step.is_done() {
            break step;
        }
    };
    for _ in 0..300 {
        let step = task.step(off);
        assert_eq!(step.observation, terminal.observation);
        assert!(step.is_done());
        assert_eq!(step.reward.to_bits(), 0.0_f64.to_bits());
    }
    assert_eq!(task.reset(Some(42)), initial);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn each_lesson_collects_and_evaluates_without_changing_the_frozen_policy() {
    let policy = new_agent(7).unwrap().policy();
    let bytes = policy.to_bytes().unwrap();
    let restored = damage_training::load_policy(bytes).unwrap();
    let restored_bytes = restored.to_bytes().unwrap();
    damage_training::load_policy(include_bytes!("../assets/robots/recovery.mpk").to_vec())
        .expect_err("healthy checkpoint has only twelve input features");
    for (lesson, name, count) in [
        (damage_training::Lesson::Hover, "hover", 0),
        (damage_training::Lesson::FrontLeft, "front-left", 5),
        (damage_training::Lesson::Scheduled, "scheduled", 40),
    ] {
        assert_eq!(lesson.name(), name);
        let mut batch = lesson.batch(7, &restored);
        assert_eq!(
            batch
                .collect(&restored)
                .unwrap()
                .iter()
                .map(|s| s.observations.len())
                .sum::<usize>(),
            512
        );
        let scores = lesson.evaluate(&restored).unwrap();
        assert!(!scores.passes());
        let report = serde_json::to_value(scores).unwrap();
        assert_eq!(report["healthy"].as_array().unwrap().len(), 5);
        assert_eq!(report["damaged"].as_array().unwrap().len(), count);
    }
    assert_eq!(restored.to_bytes().unwrap(), restored_bytes);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn incompatible_evaluation_models_fail_before_the_first_physics_command() {
    let policy =
        learning::load_policy(include_bytes!("../assets/robots/recovery.mpk").to_vec()).unwrap();
    let error = damage_training::Lesson::Hover
        .evaluate(&policy)
        .err()
        .expect("input width mismatch");
    assert!(matches!(
        error.downcast_ref::<bevy_gym::training::RecurrentPpoError>(),
        Some(bevy_gym::training::RecurrentPpoError::DimensionMismatch {
            expected: 12,
            actual: 16,
            ..
        })
    ));
}
