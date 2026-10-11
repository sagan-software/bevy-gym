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
    RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy,
    RecurrentPpoSequence, RecurrentSampler, SeedConfig,
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

/// Reset exploration after loading without changing learned mean actions or values.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reset_exploration_preserves_learned_mean_and_scales_sample_noise() {
    let (mut agent, target) = imported_agent_with_reduced_exploration();
    let before = agent.policy();
    let observation = [0.2, -0.1];
    let memory = before.initial_memory();
    let mean = before.mean_action(&observation, &memory).expect("mean");
    let value = before.value(&observation, 0).expect("value");
    let old_sample = before
        .sample_action(&observation, &memory, &mut RecurrentSampler::new(7))
        .expect("sample");
    agent.reset_exploration();
    let after = agent.policy();
    assert_eq!(
        after
            .mean_action(&observation, &memory)
            .expect("mean")
            .action,
        mean.action
    );
    assert_eq!(
        after.value(&observation, 0).expect("value").to_bits(),
        value.to_bits()
    );
    let sample = after
        .sample_action(&observation, &memory, &mut RecurrentSampler::new(7))
        .expect("sample");
    let ratio = (sample.pre_tanh_action[0] - mean.pre_tanh_action[0])
        / (old_sample.pre_tanh_action[0] - mean.pre_tanh_action[0]);
    assert!(
        (ratio - 0.082_085).abs() < 1e-6,
        "exp(-3.5 - -1) noise ratio: {ratio}"
    );
    let reloaded = RecurrentPpoPolicy::load_bytes(
        after.to_bytes().expect("record"),
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        &target,
    )
    .expect("reload");
    assert_eq!(
        reloaded
            .sample_action(&observation, &memory, &mut RecurrentSampler::new(7))
            .expect("sample")
            .pre_tanh_action,
        sample.pre_tanh_action
    );
}

/// Import original parameters under a lower configured exploration scale.
fn imported_agent_with_reduced_exploration() -> (RecurrentPpoAgent, RecurrentPpoConfig) {
    let original = RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(19),
    )
    .expect("agent");
    let bytes = original.policy().to_bytes().expect("record");
    let mut target = config();
    target.initial_log_std = -3.5;
    let agent = RecurrentPpoAgent::load_bytes(
        bytes,
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        target.clone(),
        SeedConfig::from_root(23),
    )
    .expect("load");
    (agent, target)
}

/// Retain the seeded reference before the explicit mean-head initialization experiment.
fn reduced_mean_agent() -> (RecurrentPpoAgent, RecurrentPpoPolicy) {
    let mut agent = RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(101),
    )
    .expect("seeded learner fixture");
    let before = agent.policy();
    agent.reduce_action_mean_head();
    (agent, before)
}

/// Parameter scaling changes only action means; it cannot change memory or critic state.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reduced_mean_head_preserves_recurrent_memory_and_critic() {
    let (agent, before) = reduced_mean_agent();
    let after = agent.policy();
    let mut previous_memory = before.initial_memory();
    let mut reduced_memory = after.initial_memory();
    for observation in [[0.3, -0.2], [-0.8, 0.4], [0.0, 0.0]] {
        let previous = before
            .mean_action(&observation, &previous_memory)
            .expect("original mean");
        let reduced = after
            .mean_action(&observation, &reduced_memory)
            .expect("reduced mean");
        let previous_mean = previous.action.first().expect("one action").atanh();
        let expected = (previous_mean * 0.01).tanh();
        assert!((reduced.action.first().expect("one action") - expected).abs() < 1e-6);
        assert_eq!(previous.next_memory, reduced.next_memory);
        assert_eq!(
            before
                .value(&observation, 0)
                .expect("original critic")
                .to_bits(),
            after
                .value(&observation, 0)
                .expect("retained critic")
                .to_bits()
        );
        previous_memory = previous.next_memory;
        reduced_memory = reduced.next_memory;
    }
}

/// Reduced parameters remain trainable and survive an inference-only byte reload.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reduced_mean_head_learns_with_actual_ppo_and_reloads() {
    let (mut agent, _) = reduced_mean_agent();
    let observation = [0.3, -0.2];
    let before = agent.policy();
    let sequence = sampled_sequence(&before, &observation);
    let metrics = agent.update(&[sequence]).expect("actual PPO update");
    assert_eq!(metrics.valid_samples, 2);
    assert_eq!(metrics.optimizer_updates, 1);
    let trained = agent.policy();
    let expected = trained
        .mean_action(&observation, &trained.initial_memory())
        .expect("learned mean");
    assert_ne!(
        expected.action,
        before
            .mean_action(&observation, &before.initial_memory())
            .expect("initial mean")
            .action
    );
    let loaded = RecurrentPpoPolicy::load_bytes(
        trained.to_bytes().expect("record"),
        2,
        2,
        1,
        &[-1.0],
        &[1.0],
        &config(),
    )
    .expect("inference reload");
    assert_eq!(
        loaded
            .mean_action(&observation, &loaded.initial_memory())
            .expect("reloaded mean"),
        expected
    );
}

/// Compare equal seeded Gaussian draws after subtracting each policy's own mean.
fn sampled_noise(policy: &RecurrentPpoPolicy) -> f32 {
    let observation = [0.3, -0.2];
    let memory = policy.initial_memory();
    let mean = policy.mean_action(&observation, &memory).expect("mean");
    let sample = policy
        .sample_action(&observation, &memory, &mut RecurrentSampler::new(43))
        .expect("sample");
    sample.pre_tanh_action.first().expect("one sample")
        - mean.action.first().expect("one mean").atanh()
}

/// Parameter scaling cannot silently replace the separately learned exploration distribution.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn reduced_mean_head_retains_gaussian_exploration() {
    let (agent, before) = reduced_mean_agent();
    assert!((sampled_noise(&before) - sampled_noise(&agent.policy())).abs() < 1e-6);
}

/// Sample real behavior probabilities while rewarding larger or smaller Gaussian residuals.
fn variance_preference_sequence(
    policy: &RecurrentPpoPolicy,
    preference: f32,
    seed: u64,
) -> RecurrentPpoSequence {
    let initial_memory = policy.initial_memory();
    let mut memory = initial_memory.clone();
    let mut sampler = RecurrentSampler::new(seed);
    let observation = [0.3, -0.2];
    let mut actions = Vec::with_capacity(32);
    let mut probabilities = Vec::with_capacity(32);
    let mut advantages = Vec::with_capacity(32);
    for _ in 0..32 {
        let mean = policy
            .mean_action(&observation, &memory)
            .expect("valid mean");
        let sample = policy
            .sample_action(&observation, &memory, &mut sampler)
            .expect("valid sample");
        let residual = sample.pre_tanh_action.first().expect("one action")
            - mean.pre_tanh_action.first().expect("one mean");
        advantages.push(preference * residual * residual);
        actions.push(sample.pre_tanh_action);
        probabilities.push(sample.log_probability);
        memory = sample.next_memory;
    }
    RecurrentPpoSequence {
        observations: vec![observation.to_vec(); 32],
        global_states: vec![observation.to_vec(); 32],
        pre_tanh_actions: actions,
        old_log_probabilities: probabilities,
        advantages,
        returns: vec![0.0; 32],
        value_index: 0,
        initial_memory,
    }
}

/// Boundaries constrain variance without permanently preventing inward learning.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn bounded_exploration_recovers_after_variance_preferences_reverse() {
    for (initial, outward, inward) in [(-5.0, -1.0, 1.0), (1.0, 1.0, -1.0)] {
        let mut configuration = config();
        configuration.initial_log_std = initial;
        configuration.entropy_coefficient = 0.0;
        let mut agent = RecurrentPpoAgent::new(
            2,
            2,
            1,
            &[-1.0],
            &[1.0],
            configuration,
            SeedConfig::from_root(42),
        )
        .expect("valid bounded learner");
        let first = agent
            .update(&[variance_preference_sequence(&agent.policy(), outward, 43)])
            .expect("outward preference update");
        assert_eq!(first.optimizer_updates, 1);
        assert_eq!(first.valid_samples, 32);
        let first_entropy = first.entropy;
        let mut last_entropy = first_entropy;
        for seed in 100..200 {
            let update = agent
                .update(&[variance_preference_sequence(&agent.policy(), inward, seed)])
                .expect("inward preference update");
            last_entropy = update.entropy;
        }
        let recovered = if initial < 0.0 {
            last_entropy > first_entropy + 0.00001
        } else {
            last_entropy < first_entropy - 0.00001
        };
        assert!(
            recovered,
            "initial log std {initial}: entropy {first_entropy} remained at {last_entropy}"
        );
    }
}

/// Build the shared three-action architecture for exploration regressions.
fn three_action_agent(seed: u64, config: RecurrentPpoConfig) -> RecurrentPpoAgent {
    RecurrentPpoAgent::new(
        2,
        2,
        1,
        &[-1.0; 3],
        &[1.0; 3],
        config,
        SeedConfig::from_root(seed),
    )
    .expect("three-action learner")
}

/// Compare trained bytes and the complete seeded decision after reloading.
fn assert_three_action_reload(
    policy: &RecurrentPpoPolicy,
    observation: &[f32],
    config: &RecurrentPpoConfig,
) {
    let bytes = policy.to_bytes().expect("trained record");
    let restored =
        RecurrentPpoPolicy::load_bytes(bytes.clone(), 2, 2, 1, &[-1.0; 3], &[1.0; 3], config)
            .expect("exact reload");
    assert_eq!(restored.to_bytes().expect("unchanged record"), bytes);
    let memory = policy.initial_memory();
    assert_eq!(
        restored
            .sample_action(observation, &memory, &mut RecurrentSampler::new(7))
            .expect("restored sample"),
        policy
            .sample_action(observation, &memory, &mut RecurrentSampler::new(7))
            .expect("trained sample")
    );
}

/// Preserve the complete f32 critic representation in comparisons.
fn critic_value_bits(policy: &RecurrentPpoPolicy, observation: &[f32]) -> u32 {
    policy
        .value(observation, 0)
        .expect("critic value")
        .to_bits()
}

/// Check dimensionless Gaussian noise ratios under the same seeded draws.
fn assert_noise_ratios(before: &[f32], after: &[f32], mean: &[f32], log_std_differences: [f64; 3]) {
    assert_eq!(before.len(), 3);
    assert_eq!(after.len(), 3);
    assert_eq!(mean.len(), 3);
    for (index, (((before, after), mean), difference)) in before
        .iter()
        .zip(after)
        .zip(mean)
        .zip(log_std_differences)
        .enumerate()
    {
        let ratio = f64::from(*after - *mean) / f64::from(*before - *mean);
        assert!(
            (ratio - difference.exp()).abs() < 1e-5,
            "channel {index}: {ratio}"
        );
    }
}

/// Per-channel exploration changes noise without changing learned mean, memory or value.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn per_action_exploration_preserves_mean_and_learns_after_reset() {
    let config = RecurrentPpoConfig {
        log_std_min: -9.0,
        log_std_max: 0.0,
        ..config()
    };
    let mut agent = three_action_agent(227, config.clone());
    let before = agent.policy();
    let observation = [0.2, -0.1];
    let memory = before.initial_memory();
    let mean = before.mean_action(&observation, &memory).expect("mean");
    let value = before.value(&observation, 0).expect("value");
    let old = before
        .sample_action(&observation, &memory, &mut RecurrentSampler::new(7))
        .expect("old sample");
    agent
        .reset_exploration_with_log_std(&[-9.0, -2.0, 0.0])
        .expect("inclusive boundaries and interior");
    let policy = agent.policy();
    let after = policy
        .mean_action(&observation, &memory)
        .expect("same mean");
    assert_eq!(after.action, mean.action);
    assert_eq!(after.next_memory, mean.next_memory);
    assert_eq!(
        policy.value(&observation, 0).expect("same value").to_bits(),
        value.to_bits()
    );
    let sample = policy
        .sample_action(&observation, &memory, &mut RecurrentSampler::new(7))
        .expect("new sample");
    assert_noise_ratios(
        &old.pre_tanh_action,
        &sample.pre_tanh_action,
        &mean.pre_tanh_action,
        [-8.0, -1.0, 1.0],
    );
    let sequences = [sampled_sequence(&policy, &observation)];
    let metrics = agent.update(&sequences).expect("actual PPO after reset");
    assert_eq!(metrics.optimizer_updates, 1);
    assert_eq!(metrics.valid_samples, 2);
    let trained = agent.policy();
    let trained_mean = trained
        .mean_action(&observation, &memory)
        .expect("trained mean");
    assert!(trained_mean
        .action
        .iter()
        .zip(&after.action)
        .any(|(a, b)| a.to_bits() != b.to_bits()));
    assert_three_action_reload(&trained, &observation, &config);
}

/// Width, finiteness and bounds reject before parameters or optimizer state can change.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn per_action_exploration_rejects_invalid_values_before_mutation() {
    use bevy_gym::training::RecurrentPpoError;
    let mut agent = three_action_agent(229, config());
    let before = agent.policy().to_bytes().expect("record");
    for values in [vec![], vec![f32::NAN; 2], vec![0.0; 4]] {
        assert_eq!(
            agent.reset_exploration_with_log_std(&values),
            Err(RecurrentPpoError::DimensionMismatch {
                field: "initial_action_log_std",
                expected: 3,
                actual: values.len()
            })
        );
    }
    for dimension in 0..3 {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [0.0; 3];
            values[dimension] = invalid;
            assert_eq!(
                agent.reset_exploration_with_log_std(&values),
                Err(RecurrentPpoError::NonFiniteValue {
                    field: "initial_action_log_std",
                    dimension
                })
            );
        }
        for invalid in [-5.000_001, 1.000_001] {
            let mut values = [0.0; 3];
            values[dimension] = invalid;
            assert!(matches!(
                agent.reset_exploration_with_log_std(&values),
                Err(RecurrentPpoError::InvalidConfig {
                    field: "initial_action_log_std",
                    ..
                })
            ));
        }
    }
    assert_eq!(
        agent.reset_exploration_with_log_std(&[-6.0, f32::NAN, 0.0]),
        Err(RecurrentPpoError::NonFiniteValue {
            field: "initial_action_log_std",
            dimension: 1
        })
    );
    assert_eq!(agent.policy().to_bytes().expect("unchanged record"), before);
}

/// Invalid scales retain trained Adam moments; a valid reset retains critic Adam and counters.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn per_action_exploration_preserves_optimizer_history_at_its_boundaries() {
    let make = || {
        three_action_agent(
            233,
            RecurrentPpoConfig {
                entropy_coefficient: 0.0,
                ..config()
            },
        )
    };
    let mut reference = make();
    let mut candidate = make();
    let observation = [0.2, -0.1];
    let first = [sampled_sequence(&reference.policy(), &observation)];
    reference.update(&first).expect("reference first update");
    candidate.update(&first).expect("candidate first update");
    candidate
        .reset_exploration_with_log_std(&[0.0, f32::NAN, 0.0])
        .expect_err("invalid reset");
    let second = [sampled_sequence(&reference.policy(), &observation)];
    let a = reference.update(&second).expect("reference second update");
    let b = candidate.update(&second).expect("candidate second update");
    assert_eq!(a.optimizer_steps, b.optimizer_steps);
    let policy = reference.policy();
    let expected = policy
        .mean_action(&observation, &policy.initial_memory())
        .expect("reference decision");
    let policy = candidate.policy();
    assert_eq!(
        policy
            .mean_action(&observation, &policy.initial_memory())
            .expect("retained actor moments"),
        expected
    );
    candidate
        .reset_exploration_with_log_std(&[-2.0; 3])
        .expect("valid reset");
    let third = [sampled_sequence(&reference.policy(), &observation)];
    let a = reference.update(&third).expect("reference third update");
    let b = candidate.update(&third).expect("candidate third update");
    assert_eq!(a.optimizer_steps, b.optimizer_steps);
    assert_eq!(
        critic_value_bits(&candidate.policy(), &observation),
        critic_value_bits(&reference.policy(), &observation)
    );
}

/// Mean-head update scaling preserves inference until an actual PPO update.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn action_mean_update_scales_preserve_frozen_policy_and_unscaled_channels() {
    let settings = config();
    let mut baseline = three_action_agent(313, settings.clone());
    let mut scaled = three_action_agent(313, settings.clone());
    let before = scaled.policy().to_bytes().expect("initial record");
    scaled
        .set_action_mean_update_scales(&[1.0, 0.5, 0.25])
        .expect("validated update factors");
    assert_eq!(
        scaled.policy().to_bytes().expect("unchanged inference"),
        before
    );
    let observation = [0.3, -0.2];
    let sequences = [sampled_sequence(&baseline.policy(), &observation)];
    let reference = baseline.update(&sequences).expect("baseline PPO");
    let actual = scaled.update(&sequences).expect("scaled PPO");
    assert_eq!(actual, reference);
    let decision = |agent: &RecurrentPpoAgent| {
        let policy = agent.policy();
        policy
            .mean_action(&observation, &policy.initial_memory())
            .expect("trained inference")
    };
    let full = decision(&baseline);
    let partial = decision(&scaled);
    assert_eq!(full.next_memory, partial.next_memory);
    assert_eq!(full.action[0].to_bits(), partial.action[0].to_bits());
    assert_ne!(full.action[1].to_bits(), partial.action[1].to_bits());
    assert_ne!(full.action[2].to_bits(), partial.action[2].to_bits());
    assert_eq!(
        critic_value_bits(&baseline.policy(), &observation),
        critic_value_bits(&scaled.policy(), &observation)
    );
    assert_three_action_reload(&scaled.policy(), &observation, &settings);
}

/// Width, nonfinite and range errors cannot mutate parameters or optimizer history.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn action_mean_update_scales_validate_before_mutation() {
    use bevy_gym::training::RecurrentPpoError;
    let mut agent = three_action_agent(317, config());
    let before = agent.policy().to_bytes().expect("record");
    for values in [vec![], vec![f32::NAN; 2], vec![1.0; 4]] {
        assert_eq!(
            agent.set_action_mean_update_scales(&values),
            Err(RecurrentPpoError::DimensionMismatch {
                field: "action_mean_update_scales",
                expected: 3,
                actual: values.len()
            })
        );
    }
    for dimension in 0..3 {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut values = [1.0; 3];
            values[dimension] = invalid;
            assert_eq!(
                agent.set_action_mean_update_scales(&values),
                Err(RecurrentPpoError::NonFiniteValue {
                    field: "action_mean_update_scales",
                    dimension
                })
            );
        }
        for invalid in [0.0, -0.0, -0.1, 1.000_001] {
            let mut values = [1.0; 3];
            values[dimension] = invalid;
            assert_eq!(
                agent.set_action_mean_update_scales(&values),
                Err(RecurrentPpoError::InvalidConfig {
                    field: "action_mean_update_scales",
                    reason: "values must be greater than zero and at most one"
                })
            );
        }
    }
    assert_eq!(
        agent.set_action_mean_update_scales(&[0.0, f32::NAN, 1.0]),
        Err(RecurrentPpoError::NonFiniteValue {
            field: "action_mean_update_scales",
            dimension: 1
        })
    );
    assert_eq!(agent.policy().to_bytes().expect("unchanged record"), before);
    agent
        .set_action_mean_update_scales(&[f32::MIN_POSITIVE, 1.0, 0.5])
        .expect("inclusive unit endpoint and smallest positive normal");
}

/// Reconfiguration and rejection retain Adam history, counters and minibatch order.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn action_mean_update_scales_retain_optimizer_history_and_restore_default() {
    let mut reference = three_action_agent(331, config());
    let mut candidate = three_action_agent(331, config());
    let observation = [0.3, -0.2];
    for turn in 0..3 {
        if turn == 1 {
            candidate
                .set_action_mean_update_scales(&[0.25, 0.5, 1.0])
                .expect("enable factors");
            candidate
                .set_action_mean_update_scales(&[1.0; 3])
                .expect("restore exact default");
        }
        if turn == 2 {
            for agent in [&mut reference, &mut candidate] {
                agent
                    .set_action_mean_update_scales(&[1.0, 0.5, 0.25])
                    .expect("matching factors");
            }
            assert!(candidate
                .set_action_mean_update_scales(&[1.0, f32::NAN, 0.25])
                .is_err());
        }
        let sequences = [sampled_sequence(&reference.policy(), &observation)];
        assert_eq!(
            candidate.update(&sequences).expect("candidate step"),
            reference.update(&sequences).expect("reference step")
        );
        let mean = |agent: &RecurrentPpoAgent| {
            let policy = agent.policy();
            policy
                .mean_action(&observation, &policy.initial_memory())
                .expect("mean")
        };
        assert_eq!(mean(&candidate), mean(&reference));
        assert_eq!(
            critic_value_bits(&candidate.policy(), &observation),
            critic_value_bits(&reference.policy(), &observation)
        );
    }
}

/// A policy's conditional pre-tanh Gaussian has zero divergence from itself.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_is_read_only_and_zero_for_identical_policies() {
    let agent = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap();
    let policy = agent.policy();
    let memory = policy.initial_memory();
    let before = memory.clone();
    let value = policy
        .pre_tanh_kl_from(&policy, &[0.25], &memory, &memory)
        .unwrap();
    assert_eq!(value.to_bits(), 0.0_f64.to_bits());
    assert_eq!(memory, before);
}

/// Mean and scale changes have directional analytic Gaussian divergence.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_matches_mean_and_variance_changes() {
    let settings = config();
    let mut agent = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        settings.clone(),
        SeedConfig::from_root(71),
    )
    .unwrap();
    let previous = agent.policy();
    let previous_memory = previous.initial_memory();
    agent.shift_actor_mean_bias(0, 0.25).unwrap();
    let current = agent.policy();
    let memory = current.initial_memory();
    let old = previous.mean_action(&[0.25], &previous_memory).unwrap();
    let new = current.mean_action(&[0.25], &memory).unwrap();
    let difference = f64::from(old.pre_tanh_action[0]) - f64::from(new.pre_tanh_action[0]);
    let expected = 0.5 * difference.powi(2) * (-2.0 * f64::from(settings.initial_log_std)).exp();
    let actual = current
        .pre_tanh_kl_from(&previous, &[0.25], &previous_memory, &memory)
        .unwrap();
    assert!((actual - expected).abs() < 1e-10);
}

/// Gaussian variance changes retain both divergence directions.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_matches_variance_changes() {
    let settings = config();
    let previous = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        settings.clone(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    let previous_memory = previous.initial_memory();
    let mut scaled = RecurrentPpoAgent::load_bytes(
        previous.to_bytes().unwrap(),
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        settings.clone(),
        SeedConfig::from_root(71),
    )
    .unwrap();
    scaled
        .reset_exploration_with_log_std(&[settings.initial_log_std - 1.0])
        .unwrap();
    let scaled = scaled.policy();
    let scaled_memory = scaled.initial_memory();
    let forward = scaled
        .pre_tanh_kl_from(&previous, &[0.25], &previous_memory, &scaled_memory)
        .unwrap();
    let reverse = previous
        .pre_tanh_kl_from(&scaled, &[0.25], &scaled_memory, &previous_memory)
        .unwrap();
    assert!((forward - (2.0_f64.exp_m1().mul_add(0.5, -1.0))).abs() < 1e-12);
    assert!((reverse - ((-2.0_f64).exp_m1().mul_add(0.5, 1.0))).abs() < 1e-12);
}

/// Action scaling does not alter the latent Gaussian distribution.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_ignores_action_scaling() {
    let settings = config();
    let previous = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        settings.clone(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    let previous_memory = previous.initial_memory();
    let differently_scaled = RecurrentPpoPolicy::load_bytes(
        previous.to_bytes().unwrap(),
        1,
        1,
        1,
        &[-2.0],
        &[2.0],
        &settings,
    )
    .unwrap();
    assert_eq!(
        differently_scaled
            .pre_tanh_kl_from(
                &previous,
                &[0.25],
                &previous_memory,
                &differently_scaled.initial_memory()
            )
            .unwrap()
            .to_bits(),
        0_f64.to_bits()
    );
}

/// Profile width mismatches precede observation validation.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_rejects_invalid_profiles_and_inputs() {
    let policy = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    let memory = policy.initial_memory();
    let wider_observation = RecurrentPpoAgent::new(
        2,
        1,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    assert!(matches!(
        wider_observation.pre_tanh_kl_from(&policy, &[f32::NAN], &memory, &memory),
        Err(RecurrentPpoError::DimensionMismatch {
            field: "KL observation",
            expected: 1,
            actual: 2
        })
    ));
    let wider_action = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0; 2],
        &[1.0; 2],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    assert!(matches!(
        wider_action.pre_tanh_kl_from(&policy, &[f32::NAN], &memory, &memory),
        Err(RecurrentPpoError::DimensionMismatch {
            field: "KL action",
            expected: 1,
            actual: 2
        })
    ));
}

/// Observation validation precedes recurrent-memory validation.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_rejects_invalid_observations() {
    let policy = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    let memory = policy.initial_memory();
    assert!(matches!(
        policy.pre_tanh_kl_from(&policy, &[], &memory, &memory),
        Err(RecurrentPpoError::DimensionMismatch {
            field: "observation",
            expected: 1,
            actual: 0
        })
    ));
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            policy.pre_tanh_kl_from(&policy, &[invalid], &memory, &memory),
            Err(RecurrentPpoError::NonFiniteValue {
                field: "observation",
                dimension: 0
            })
        ));
    }
}

/// Both policies reject invalid recurrent memories.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_rejects_invalid_memories() {
    let policy = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    let memory = policy.initial_memory();
    let mut bad = memory.clone();
    bad.cell.clear();
    assert!(matches!(
        policy.pre_tanh_kl_from(&policy, &[0.25], &bad, &memory),
        Err(RecurrentPpoError::DimensionMismatch {
            field: "memory cell",
            expected: 4,
            actual: 0
        })
    ));
    assert!(matches!(
        policy.pre_tanh_kl_from(&policy, &[0.25], &memory, &bad),
        Err(RecurrentPpoError::DimensionMismatch {
            field: "memory cell",
            expected: 4,
            actual: 0
        })
    ));
    bad = memory.clone();
    bad.hidden[0] = f32::NAN;
    assert!(matches!(
        policy.pre_tanh_kl_from(&policy, &[0.25], &bad, &memory),
        Err(RecurrentPpoError::NonFiniteValue {
            field: "memory hidden",
            dimension: 0
        })
    ));
    assert!(matches!(
        policy.pre_tanh_kl_from(&policy, &[0.25], &memory, &bad),
        Err(RecurrentPpoError::NonFiniteValue {
            field: "memory hidden",
            dimension: 0
        })
    ));
}

/// Each policy uses its own recurrent architecture and history during comparison.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_accepts_independent_recurrent_histories() {
    let settings = config();
    let previous = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        settings.clone(),
        SeedConfig::from_root(71),
    )
    .unwrap()
    .policy();
    let different = RecurrentPpoConfig {
        actor_hidden_size: 8,
        ..settings.clone()
    };
    let current = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        different,
        SeedConfig::from_root(72),
    )
    .unwrap()
    .policy();
    let old_memory = previous
        .mean_action(&[0.25], &previous.initial_memory())
        .unwrap()
        .next_memory;
    let memory = current
        .mean_action(&[-0.25], &current.initial_memory())
        .unwrap()
        .next_memory;
    let old_before = old_memory.clone();
    let before = memory.clone();
    let old = previous.mean_action(&[0.5], &old_memory).unwrap();
    let new = current.mean_action(&[0.5], &memory).unwrap();
    let old_mean = f64::from(*old.pre_tanh_action.first().unwrap());
    let mean = f64::from(*new.pre_tanh_action.first().unwrap());
    let expected =
        0.5 * (old_mean - mean).powi(2) * (-2.0 * f64::from(settings.initial_log_std)).exp();
    let actual = current
        .pre_tanh_kl_from(&previous, &[0.5], &old_memory, &memory)
        .unwrap();
    assert!((actual - expected).abs() < 1e-10);
    assert_eq!(old_memory, old_before);
    assert_eq!(memory, before);
}

/// Nonfinite inferred means cannot become a reported finite policy divergence.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn conditional_gaussian_kl_rejects_nonfinite_inferred_means() {
    let mut agent = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        config(),
        SeedConfig::from_root(71),
    )
    .unwrap();
    let valid = agent.policy();
    agent.shift_actor_mean_bias(0, f32::MAX).unwrap();
    agent.shift_actor_mean_bias(0, f32::MAX).unwrap();
    let invalid = agent.policy();
    let memory = valid.initial_memory();
    assert!(matches!(
        valid.pre_tanh_kl_from(&invalid, &[0.25], &memory, &memory),
        Err(RecurrentPpoError::NonFiniteValue {
            field: "previous Gaussian mean",
            dimension: 0
        })
    ));
    assert!(matches!(
        invalid.pre_tanh_kl_from(&valid, &[0.25], &memory, &memory),
        Err(RecurrentPpoError::NonFiniteValue {
            field: "Gaussian mean",
            dimension: 0
        })
    ));
}
