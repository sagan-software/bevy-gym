//! Browser-compatible recurrent PPO parameter records and optimizer updates.

#![cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
#[cfg(not(target_arch = "wasm32"))]
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde as _;
use serde_json as _;
use shakmaty as _;
#[cfg(not(target_arch = "wasm32"))]
use tokio as _;

use bevy_gym::training::{
    RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoPolicy, RecurrentPpoSequence,
    RecurrentSampler, SeedConfig,
};

/// Small architecture keeps the external recorder and update contracts focused.
fn config() -> RecurrentPpoConfig {
    RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        epochs: 1,
        minibatch_sequences: 1,
        ..RecurrentPpoConfig::default()
    }
}

/// Two contiguous samples keep advantage normalization nontrivial.
fn sampled_sequence(policy: &RecurrentPpoPolicy, observation: &[f32]) -> RecurrentPpoSequence {
    let memory = policy.initial_memory();
    let mut sampler = RecurrentSampler::new(43);
    let first = policy
        .sample_action(observation, &memory, &mut sampler)
        .expect("first sample");
    let second = policy
        .sample_action(observation, &first.next_memory, &mut sampler)
        .expect("second sample");
    RecurrentPpoSequence {
        observations: vec![observation.to_vec(); 2],
        global_states: vec![observation.to_vec(); 2],
        pre_tanh_actions: vec![first.pre_tanh_action, second.pre_tanh_action],
        old_log_probabilities: vec![first.log_probability, second.log_probability],
        advantages: vec![1.0, -1.0],
        returns: vec![2.0, 1.0],
        value_index: 0,
        initial_memory: memory,
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn optimizer_update_and_byte_round_trip_preserve_actor_and_critic() {
    let config = config();
    let mut agent = RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config.clone(),
        SeedConfig::from_root(42),
    )
    .expect("valid learner");
    let before = agent.policy();
    let observation = vec![0.3, -0.2];
    let memory = before.initial_memory();
    let sequence = sampled_sequence(&before, &observation);
    let update = agent.update(&[sequence]).expect("optimizer update");
    assert_eq!(update.optimizer_updates, 1);
    assert!(update.actor_loss.is_finite() && update.critic_loss.is_finite());
    let after = agent.policy();
    assert!(
        (before.value(&observation, 0).expect("initial value")
            - after.value(&observation, 0).expect("updated value"))
        .abs()
            > 1e-8
    );
    assert_ne!(
        before
            .mean_action(&observation, &memory)
            .expect("initial action"),
        after
            .mean_action(&observation, &memory)
            .expect("updated action")
    );
    let bytes = after.to_bytes().expect("encode without filesystem");
    let restored = RecurrentPpoPolicy::load_bytes(bytes, 2, 2, 1, &[-1.0], &[1.0], &config)
        .expect("restore without filesystem");
    assert_eq!(
        after
            .mean_action(&observation, &memory)
            .expect("trained action"),
        restored
            .mean_action(&observation, &memory)
            .expect("restored action")
    );
    assert_eq!(
        after
            .value(&observation, 0)
            .expect("trained value")
            .to_bits(),
        restored
            .value(&observation, 0)
            .expect("restored value")
            .to_bits()
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn byte_loading_rejects_corruption_and_each_architecture_mismatch() {
    let config = config();
    let agent = RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config.clone(),
        SeedConfig::from_root(42),
    )
    .expect("valid learner");
    let bytes = agent.policy().to_bytes().expect("policy bytes");
    RecurrentPpoPolicy::load_bytes(vec![0, 1, 2], 2, 2, 1, &[-1.0], &[1.0], &config)
        .expect_err("corrupt bytes are rejected");
    RecurrentPpoAgent::load_bytes(
        vec![0, 1, 2],
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config.clone(),
        SeedConfig::from_root(99),
    )
    .expect_err("corrupt training record is rejected");
    for (observations, global, values, low, high, mismatched) in [
        (3, 2, 1, vec![-1.0], vec![1.0], config.clone()),
        (2, 3, 1, vec![-1.0], vec![1.0], config.clone()),
        (2, 2, 2, vec![-1.0], vec![1.0], config.clone()),
        (2, 2, 1, vec![-1.0; 2], vec![1.0; 2], config.clone()),
        (
            2,
            2,
            1,
            vec![-1.0],
            vec![1.0],
            RecurrentPpoConfig {
                actor_hidden_size: 8,
                ..config.clone()
            },
        ),
        (
            2,
            2,
            1,
            vec![-1.0],
            vec![1.0],
            RecurrentPpoConfig {
                critic_hidden_sizes: vec![8],
                ..config.clone()
            },
        ),
        (
            2,
            2,
            1,
            vec![-1.0],
            vec![1.0],
            RecurrentPpoConfig {
                critic_hidden_sizes: vec![4, 4],
                ..config
            },
        ),
    ] {
        RecurrentPpoAgent::load_bytes(
            bytes.clone(),
            observations,
            global,
            values,
            &low,
            &high,
            mismatched.clone(),
            SeedConfig::from_root(99),
        )
        .expect_err("training architecture mismatch is rejected");
        RecurrentPpoPolicy::load_bytes(
            bytes.clone(),
            observations,
            global,
            values,
            &low,
            &high,
            &mismatched,
        )
        .expect_err("architecture mismatch is rejected");
    }
}

/// Restoring RL weights retains actor/critic behavior while a fresh optimizer learns again.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn byte_loaded_agent_preserves_weights_and_restarts_optimizer() {
    let config = config();
    let mut original = RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config.clone(),
        SeedConfig::from_root(42),
    )
    .expect("original learner");
    let observation = [0.3, -0.2];
    let sequence = sampled_sequence(&original.policy(), &observation);
    let first = original.update(&[sequence]).expect("RL update");
    assert_eq!(first.optimizer_steps, 1);
    let trained = original.policy();
    let mut restored = RecurrentPpoAgent::load_bytes(
        trained.to_bytes().expect("trained record"),
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config,
        SeedConfig::from_root(99),
    )
    .expect("checkpoint-start learner");
    let before = restored.policy();
    let mut expected_memory = trained.initial_memory();
    let mut actual_memory = before.initial_memory();
    for observation in [[0.3, -0.2], [-0.8, 0.4], [0.0, 0.0]] {
        let expected = trained
            .mean_action(&observation, &expected_memory)
            .expect("source action");
        let actual = before
            .mean_action(&observation, &actual_memory)
            .expect("restored action");
        assert_eq!(actual, expected);
        expected_memory = expected.next_memory;
        actual_memory = actual.next_memory;
        assert_eq!(
            before
                .value(&observation, 0)
                .expect("restored critic")
                .to_bits(),
            trained
                .value(&observation, 0)
                .expect("source critic")
                .to_bits()
        );
    }
    let sequence = sampled_sequence(&before, &observation);
    let update = restored
        .update(&[sequence])
        .expect("fresh optimizer update");
    assert_eq!(update.optimizer_steps, 1);
    assert_eq!(update.optimizer_updates, 1);
    assert_eq!(update.valid_samples, 2);
    assert_ne!(
        before
            .value(&observation, 0)
            .expect("critic before")
            .to_bits(),
        restored
            .policy()
            .value(&observation, 0)
            .expect("critic after")
            .to_bits()
    );
    assert_ne!(
        before
            .mean_action(&observation, &before.initial_memory())
            .expect("before"),
        restored
            .policy()
            .mean_action(&observation, &before.initial_memory())
            .expect("after")
    );
}
