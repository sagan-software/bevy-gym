//! Browser-compatible recurrent PPO parameter records and optimizer updates.

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde as _;
use serde_json as _;
use shakmaty as _;
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

#[test]
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

#[test]
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
