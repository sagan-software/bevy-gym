//! Pendulum training and frozen inference through the browser's public session API.
use bevy_gym::environments::{Pendulum, PendulumAction};
use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoPolicy};
use bevy_gym::{Env, TimeLimit};
use bevy_gym_browser::{AdvanceSteps, Session, Task};
use serde as _;
use serde_json as _;
use tokio as _;

#[test]
fn pendulum_training_updates_both_networks_and_exports_a_compatible_record() {
    assert_eq!(
        "pendulum".parse::<Task>().expect("task name"),
        Task::Pendulum
    );
    let mut training = Session::train_task(Task::Pendulum, 42).expect("fresh PPO");
    let initial = training.export_policy().expect("initial model");
    let before = policy_outputs(&load_policy(initial));
    training
        .advance(AdvanceSteps::try_from(256).expect("batch"))
        .expect("collect");
    let snapshot = training
        .advance(AdvanceSteps::try_from(256).expect("batch"))
        .expect("update");
    assert!(snapshot.optimizer_steps > 0);
    assert_eq!(snapshot.learning_rate, Some(0.003));
    assert_eq!(snapshot.critic_learning_rate, Some(0.001));
    assert!(snapshot.loss.expect("actor loss").is_finite());
    assert!(snapshot.critic_loss.expect("critic loss").is_finite());
    assert_eq!(snapshot.parallel_environments, 8);
    assert_eq!(snapshot.epsilon, None);
    assert!(snapshot.episode_return <= 0.0);
    assert!((-2.0..=2.0).contains(&snapshot.state[2]));
    let bytes = training.export_policy().expect("updated model");
    let after = policy_outputs(&load_policy(bytes.clone()));
    assert_ne!(before.0.to_bits(), after.0.to_bits());
    assert_ne!(before.1.to_bits(), after.1.to_bits());
    Session::inference_task(Task::Pendulum, bytes, 42).expect("compatible frozen record");
}

#[test]
fn frozen_inference_matches_native_actions_rewards_and_the_two_hundred_step_limit() {
    let bytes = Session::train_task(Task::Pendulum, 42)
        .expect("fresh PPO")
        .export_policy()
        .expect("record");
    let policy = load_policy(bytes.clone());
    let mut memory = policy.initial_memory();
    let mut native = TimeLimit::new(Pendulum::default(), 200).expect("positive cap");
    let mut observation = native.reset(Some(17)).observation;
    let mut session = Session::inference_task(Task::Pendulum, bytes, 17).expect("frozen session");
    let mut total = 0.0;
    for index in 1..=200 {
        let encoded = [observation[0], observation[1], observation[2] / 8.0];
        let sampled = policy.mean_action(&encoded, &memory).expect("mean torque");
        memory = sampled.next_memory;
        let torque = PendulumAction::try_from(sampled.action[0]).expect("finite torque");
        let result = native.step(torque);
        observation = result.observation;
        total += result.reward;
        let snapshot = session
            .advance(AdvanceSteps::try_from(1).expect("one step"))
            .expect("inference");
        assert_eq!(snapshot.transitions, index);
        assert_eq!(snapshot.optimizer_steps, 0);
        assert_eq!(snapshot.learning_rate, None);
        assert_eq!(snapshot.critic_learning_rate, None);
        if index < 200 {
            assert!(snapshot.completed.is_empty());
            assert_eq!(snapshot.episode_return.to_bits(), total.to_bits());
            let [angle, velocity] = native.inner().state();
            assert_eq!(
                snapshot.state.map(f64::to_bits),
                [angle, velocity, f64::from(f32::from(torque)), 0.0].map(f64::to_bits)
            );
        } else {
            assert_eq!(snapshot.completed.len(), 1);
            assert_eq!(snapshot.completed[0].reward.to_bits(), total.to_bits());
            assert_eq!(snapshot.episode_count, 1);
        }
    }
}

/// Read the shared actor and critic architecture independently of the session facade.
fn load_policy(bytes: Vec<u8>) -> RecurrentPpoPolicy {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 32,
        critic_hidden_sizes: vec![64, 32],
        ..RecurrentPpoConfig::default()
    };
    RecurrentPpoPolicy::load_bytes(bytes, 3, 3, 1, &[-2.0], &[2.0], &config)
        .expect("native policy record")
}

/// A fixed valid observation exposes parameter changes in both networks.
fn policy_outputs(policy: &RecurrentPpoPolicy) -> (f32, f32) {
    let observation = [0.0, 1.0, 0.25];
    let action = policy
        .mean_action(&observation, &policy.initial_memory())
        .expect("mean torque");
    (
        *action.action.first().expect("one torque coordinate"),
        policy.value(&observation, 0).expect("critic value"),
    )
}

#[test]
fn pendulum_rejects_corrupt_and_wrong_task_architectures() {
    for bytes in [
        vec![1, 2, 3],
        include_bytes!("../models/mountain-car-continuous.mpk").to_vec(),
    ] {
        Session::inference_task(Task::Pendulum, bytes, 42).expect_err("incompatible policy");
    }
}

#[test]
fn inference_batching_preserves_resets_memory_and_the_frozen_policy() {
    let bytes = Session::train_task(Task::Pendulum, 43)
        .expect("fresh PPO")
        .export_policy()
        .expect("record");
    let mut batched =
        Session::inference_task(Task::Pendulum, bytes.clone(), 18).expect("batched inference");
    let mut single =
        Session::inference_task(Task::Pendulum, bytes.clone(), 18).expect("single-step inference");
    let mut single_returns = Vec::new();
    let mut batched_returns = Vec::new();
    for _ in 0..3 {
        let expected = batched
            .advance(AdvanceSteps::try_from(200).expect("episode batch"))
            .expect("batched transitions");
        batched_returns.extend(
            expected
                .completed
                .iter()
                .map(|episode| episode.reward.to_bits()),
        );
        for index in 1..=200 {
            let actual = single
                .advance(AdvanceSteps::try_from(1).expect("one transition"))
                .expect("single transition");
            single_returns.extend(
                actual
                    .completed
                    .iter()
                    .map(|episode| episode.reward.to_bits()),
            );
            if index == 200 {
                assert_eq!(
                    actual.state.map(f64::to_bits),
                    expected.state.map(f64::to_bits)
                );
                assert_eq!(actual.transitions, expected.transitions);
                assert_eq!(actual.episode_count, expected.episode_count);
                assert_eq!(actual.episode_return.to_bits(), 0.0_f64.to_bits());
            }
        }
    }
    assert_eq!(single_returns, batched_returns);
    assert_eq!(single_returns.len(), 3);
    assert_ne!(single_returns[0], single_returns[1]);
    assert_eq!(single.export_policy().expect("frozen export"), bytes);
    assert_eq!(batched.export_policy().expect("frozen export"), bytes);
}
