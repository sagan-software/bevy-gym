//! Browser continuous-action training and frozen-inference contracts.
use bevy_gym as _;
use serde as _;
use serde_json as _;
use tokio as _;

use bevy_gym_browser::{AdvanceSteps, Session, Task};

#[test]
fn continuous_training_updates_both_optimizers_and_exports_frozen_inference() {
    let task = Task::MountainCarContinuous;
    let mut training = Session::train_task(task, 42).expect("continuous learner");
    let before = training.export_policy().expect("initial parameters");
    let steps = AdvanceSteps::try_from(256).expect("bounded batch");
    let first = training.advance(steps).expect("half rollout");
    assert_eq!(first.transitions, 256);
    assert_eq!(first.optimizer_steps, 0);
    assert_eq!(first.parallel_environments, 8);
    let updated = training.advance(steps).expect("complete rollout");
    assert!(updated.optimizer_steps > 0);
    assert_eq!(updated.learning_rate, Some(0.003));
    assert_eq!(updated.critic_learning_rate, Some(0.001));
    assert!(updated.loss.expect("actor loss").is_finite());
    assert!(updated.critic_loss.expect("critic loss").is_finite());
    let bytes = training.export_policy().expect("trained parameters");
    assert_ne!(bytes, before);
    assert_frozen_inference(bytes, steps);
}

/// An imported model retains its parameters across optimizer-free episode resets.
fn assert_frozen_inference(bytes: Vec<u8>, steps: AdvanceSteps) {
    let mut inference =
        Session::inference_task(Task::MountainCarContinuous, bytes, 17).expect("frozen policy");
    let frozen = inference.advance(steps).expect("inference steps");
    assert_eq!(frozen.optimizer_steps, 0);
    assert_eq!(frozen.learning_rate, None);
    assert_eq!(frozen.critic_learning_rate, None);
    assert_eq!(frozen.parallel_environments, 1);
    assert!(frozen.state.iter().all(|value| value.is_finite()));
    let frozen_parameters = inference.export_policy().expect("frozen parameters");
    for _ in 0..4 {
        inference
            .advance(steps)
            .expect("complete an inference episode");
    }
    assert_eq!(
        inference.export_policy().expect("unchanged parameters"),
        frozen_parameters
    );
}

#[test]
fn continuous_batch_sizes_preserve_rollouts_and_optimizer_results() {
    let mut small = Session::train_task(Task::MountainCarContinuous, 73).expect("small batches");
    let mut large = Session::train_task(Task::MountainCarContinuous, 73).expect("large batches");
    for _ in 0..1023 {
        small
            .advance(AdvanceSteps::try_from(1).expect("one transition"))
            .expect("advance");
    }
    let small_result = small
        .advance(AdvanceSteps::try_from(1).expect("one transition"))
        .expect("rollout boundary");
    for _ in 0..3 {
        large
            .advance(AdvanceSteps::try_from(256).expect("batch"))
            .expect("advance");
    }
    let large_result = large
        .advance(AdvanceSteps::try_from(256).expect("batch"))
        .expect("rollout boundary");
    assert_eq!(small_result.transitions, large_result.transitions);
    assert_eq!(small_result.optimizer_steps, large_result.optimizer_steps);
    assert_eq!(
        small_result.state.map(f64::to_bits),
        large_result.state.map(f64::to_bits)
    );
    let small_bytes = small.export_policy().expect("small-batch policy");
    let large_bytes = large.export_policy().expect("large-batch policy");
    let mut small_inference = Session::inference_task(Task::MountainCarContinuous, small_bytes, 23)
        .expect("small inference");
    let mut large_inference = Session::inference_task(Task::MountainCarContinuous, large_bytes, 23)
        .expect("large inference");
    let steps = AdvanceSteps::try_from(256).expect("batch");
    assert_eq!(
        small_inference
            .advance(steps)
            .expect("small policy")
            .state
            .map(f64::to_bits),
        large_inference
            .advance(steps)
            .expect("large policy")
            .state
            .map(f64::to_bits),
    );
}

#[test]
fn continuous_errors_retain_the_shared_learner_source() {
    use std::error::Error;
    let error = Session::inference_task(Task::MountainCarContinuous, vec![0], 42)
        .expect_err("corrupt record");
    assert!(matches!(
        error,
        bevy_gym_browser::SessionError::RecurrentLearner(_)
    ));
    assert!(error.source().is_some());
    assert!(error.to_string().contains("<memory>"));
}

#[test]
fn every_training_lane_reports_completed_episodes_with_original_rewards() {
    let mut session = Session::train_task(Task::MountainCarContinuous, 42).expect("eight lanes");
    let steps = AdvanceSteps::try_from(256).expect("batch");
    let mut completed = 0;
    let mut last_transition = 0;
    for _ in 0..32 {
        let snapshot = session.advance(steps).expect("complete rollout");
        for episode in snapshot.completed {
            assert!(episode.transition > last_transition);
            assert!((-99.9..=100.0).contains(&episode.reward));
            last_transition = episode.transition;
            completed += 1;
        }
        assert_eq!(snapshot.episode_count, completed);
    }
    assert!(
        completed >= 8,
        "each of eight lanes has a 999-transition cap"
    );
}

#[test]
fn continuous_policy_cannot_load_as_a_discrete_policy() {
    let policy = Session::train_task(Task::MountainCarContinuous, 42)
        .expect("continuous learner")
        .export_policy()
        .expect("policy bytes");
    for task in [Task::CartPole, Task::MountainCar] {
        Session::inference_task(task, policy.clone(), 42).expect_err("incompatible learner");
    }
    let discrete = Session::train_task(Task::MountainCar, 42)
        .expect("discrete learner")
        .export_policy()
        .expect("discrete parameters");
    Session::inference_task(Task::MountainCarContinuous, discrete, 42)
        .expect_err("continuous task requires a recurrent PPO record");
}
