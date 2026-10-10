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

/// Mixed-length unrolls retain independent memory, clipping, both critic outputs and PPO updates.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn mixed_length_minibatch_matches_retained_scalar_optimizer() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 8,
        critic_hidden_sizes: vec![8],
        epochs: 1,
        minibatch_sequences: 8,
        ..RecurrentPpoConfig::default()
    };
    let mut agent = RecurrentPpoAgent::new(
        2,
        3,
        2,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        config,
        SeedConfig::from_root(123),
    )
    .expect("mixed-length learner");
    let sequences = scalar_minibatch_sequences(&agent.policy());
    let mut metrics = Vec::new();
    for _ in 0..4 {
        let update = agent.update(&sequences).expect("update");
        assert_eq!(update.valid_samples, 10);
        assert_eq!(update.optimizer_updates, 1);
        metrics.push(vec![
            update.actor_loss,
            update.critic_loss,
            update.entropy,
            update.approximate_kl,
        ]);
    }
    let outputs = scalar_minibatch_outputs(&agent.policy(), &sequences);
    assert_scalar_optimizer_baseline(&metrics, &outputs);
}

/// Construct synthetic regression sequences with distinct memories and critic indices.
fn scalar_minibatch_sequences(policy: &RecurrentPpoPolicy) -> Vec<RecurrentPpoSequence> {
    let mut sampler = RecurrentSampler::new(321);
    let mut sequences = Vec::new();
    for (lane, length) in [1, 4, 2, 3].into_iter().enumerate() {
        let mut sequence = empty_scalar_sequence(policy, lane);
        let mut memory = sequence.initial_memory.clone();
        for step in 0..length {
            let observation = [
                (step as f32).mul_add(0.01, lane as f32 * 0.2),
                (step as f32).mul_add(0.03, -0.3),
            ];
            let action = policy
                .sample_action(&observation, &memory, &mut sampler)
                .expect("sample");
            memory = action.next_memory;
            sequence
                .global_states
                .push(vec![observation[0], observation[1], lane as f32 * -0.2]);
            sequence.observations.push(observation.to_vec());
            sequence.pre_tanh_actions.push(action.pre_tanh_action);
            sequence
                .old_log_probabilities
                .push(action.log_probability + if step % 2 == 0 { -0.3 } else { 0.3 });
            sequence.advantages.push(if step % 2 == 0 {
                (lane as f32).mul_add(0.1, 1.0)
            } else {
                -0.5
            });
            sequence
                .returns
                .push((step as f32).mul_add(-0.1, (lane as f32).mul_add(0.3, 0.5)));
        }
        sequences.push(sequence);
    }
    sequences
}

/// Set each fixture lane's initial memory independently before adding samples.
fn empty_scalar_sequence(policy: &RecurrentPpoPolicy, lane: usize) -> RecurrentPpoSequence {
    let mut initial_memory = policy.initial_memory();
    initial_memory.cell.fill(lane as f32 * 0.1);
    initial_memory.hidden.fill(lane as f32 * -0.04);
    RecurrentPpoSequence {
        observations: Vec::new(),
        global_states: Vec::new(),
        pre_tanh_actions: Vec::new(),
        old_log_probabilities: Vec::new(),
        advantages: Vec::new(),
        returns: Vec::new(),
        value_index: lane % 2,
        initial_memory,
    }
}

/// Inspect actions, recurrent memory and both critic values after the updates.
fn scalar_minibatch_outputs(
    policy: &RecurrentPpoPolicy,
    sequences: &[RecurrentPpoSequence],
) -> Vec<Vec<f64>> {
    let mut outputs = Vec::new();
    for (lane, observation) in [[0.4, -0.2], [-0.3, 0.1], [0.0, 0.0], [1.0, -1.0]]
        .into_iter()
        .enumerate()
    {
        let sequence = sequences.get(lane).expect("one sequence per fixture probe");
        let action = policy
            .mean_action(&observation, &sequence.initial_memory)
            .expect("probe inference");
        let mut output = action.action;
        output.extend(action.pre_tanh_action);
        output.push(action.log_probability);
        output.extend(action.next_memory.cell);
        output.extend(action.next_memory.hidden);
        let state = [observation[0], observation[1], lane as f32 * -0.2];
        output.push(policy.value(&state, 0).expect("first value"));
        output.push(policy.value(&state, 1).expect("second value"));
        outputs.push(output.into_iter().map(f64::from).collect::<Vec<_>>());
    }
    outputs
}

/// Compare complete diagnostic and inference rows with the retained scalar optimizer.
fn assert_scalar_optimizer_baseline(metrics: &[Vec<f64>], outputs: &[Vec<f64>]) {
    // The fixture comes from the scalar optimizer at dc7ede6 before batching.
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/recurrent_ppo_scalar_minibatch.json"))
            .expect("retained scalar optimizer outputs");
    for (key, actual) in [("metrics", metrics), ("outputs", outputs)] {
        let rows = expected
            .get(key)
            .expect("baseline key")
            .as_array()
            .expect("baseline matrix");
        assert_eq!(actual.len(), rows.len());
        for (actual_row, expected_row) in actual.iter().zip(rows) {
            let expected_row = expected_row.as_array().expect("baseline row");
            assert_eq!(actual_row.len(), expected_row.len());
            for (actual, expected) in actual_row.iter().zip(expected_row) {
                let expected = expected.as_f64().expect("finite baseline scalar");
                assert!(
                    (actual - expected).abs() < 3e-5,
                    "{key}: {actual} differs from scalar baseline {expected}"
                );
            }
        }
    }
}

/// Padded outputs must not create overflow when the valid Gaussian log density exceeds 100.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn padding_is_excluded_before_probability_ratios() {
    let configuration = RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        epochs: 1,
        minibatch_sequences: 2,
        initial_log_std: -5.0,
        ..RecurrentPpoConfig::default()
    };
    let mut agent = RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        configuration,
        SeedConfig::from_root(127),
    )
    .expect("narrow Gaussian");
    let policy = agent.policy();
    let memory = policy.initial_memory();
    let mean = policy
        .mean_action(&[0.0, 0.0], &memory)
        .expect("zero observation");
    assert!(mean.log_probability > 100.0);
    let sequences = [1, 3].map(|length| RecurrentPpoSequence {
        observations: vec![vec![0.0, 0.0]; length],
        global_states: vec![vec![0.0, 0.0]; length],
        pre_tanh_actions: vec![mean.pre_tanh_action.clone(); length],
        old_log_probabilities: vec![mean.log_probability; length],
        advantages: vec![if length == 1 { 1.0 } else { -1.0 }; length],
        returns: vec![1.0; length],
        value_index: 0,
        initial_memory: memory.clone(),
    });
    let update = agent
        .update(&sequences)
        .expect("only four valid density ratios");
    assert_eq!(update.valid_samples, 4);
    assert_eq!(update.optimizer_updates, 1);
    assert!(update.actor_loss.is_finite());
    assert!(update.critic_loss.is_finite());
    assert!(update.entropy.is_finite());
    assert!(update.approximate_kl.is_finite());
}
