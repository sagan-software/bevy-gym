//! External contracts for learning actions from earlier observations.
#![cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
use bevy_gym::training::{
    RecurrentBehaviorSample, RecurrentPpoAgent, RecurrentPpoConfig, SeedConfig,
};

/// A small recurrent actor makes the delayed-cue learning check inexpensive.
fn learner() -> RecurrentPpoAgent {
    RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-1.0],
        &[1.0],
        RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![4],
            actor_learning_rate: 0.02,
            ..RecurrentPpoConfig::default()
        },
        SeedConfig::from_root(73),
    )
    .expect("valid architecture")
}

/// Only the first observation identifies the desired later action.
fn demonstration(cue: f32) -> Vec<RecurrentBehaviorSample> {
    (0..4)
        .map(|step| RecurrentBehaviorSample {
            observation: vec![if step == 0 { cue } else { 0.0 }],
            action: vec![if step == 0 { 0.0 } else { cue * 0.8 }],
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
/// Opposite histories require opposite actions from the same blank observation.
fn earlier_cue_controls_actions_after_the_cue_disappears() {
    let mut agent = learner();
    let memory = agent.policy().initial_memory();
    let negative = demonstration(-1.0);
    let positive = demonstration(1.0);
    for _ in 0..250 {
        agent
            .behavior_clone_sequence(&negative, &memory)
            .expect("negative sequence");
        agent
            .behavior_clone_sequence(&positive, &memory)
            .expect("positive sequence");
    }
    let policy = agent.policy();
    for cue in [-1.0, 1.0] {
        let mut memory = policy.initial_memory();
        for sample in demonstration(cue) {
            let action = policy
                .mean_action(&sample.observation, &memory)
                .expect("inference");
            if sample.observation == [0.0] {
                assert!(
                    action.action.first().expect("one action") * cue > 0.6,
                    "cue={cue}, action={action:?}"
                );
            }
            memory = action.next_memory;
        }
    }
}

/// Invalid input must leave both parameters and the next optimizer step unchanged.
fn rejected_without_mutation(
    samples: &[RecurrentBehaviorSample],
    memory: &bevy_gym::training::RecurrentMemory,
    expected: bevy_gym::training::RecurrentPpoError,
) {
    let mut agent = learner();
    let mut control = learner();
    let before = agent.policy().to_bytes().expect("record before rejection");
    assert_eq!(
        agent.behavior_clone_sequence(samples, memory),
        Err(expected)
    );
    assert_eq!(
        agent.policy().to_bytes().expect("record after rejection"),
        before
    );
    let valid = demonstration(1.0);
    let initial = control.policy().initial_memory();
    agent
        .behavior_clone_sequence(&valid, &initial)
        .expect("next actor update");
    control
        .behavior_clone_sequence(&valid, &initial)
        .expect("control actor update");
    assert_eq!(
        agent
            .policy()
            .mean_action(&[0.5], &initial)
            .expect("updated actor"),
        control
            .policy()
            .mean_action(&[0.5], &initial)
            .expect("control actor"),
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
/// Empty demonstrations take precedence over invalid initial memory.
fn empty_samples_fail_before_malformed_memory() {
    use bevy_gym::training::{RecurrentMemory, RecurrentPpoError};
    rejected_without_mutation(
        &[],
        &RecurrentMemory::zeros(0),
        RecurrentPpoError::InvalidSequence("at least one behavior sample is required"),
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
/// Every input vector validates its width and every nonfinite representation.
fn every_vector_rejects_incorrect_dimensions_and_nonfinite_values() {
    use bevy_gym::training::RecurrentPpoError;
    for field in ["observation", "action", "memory cell", "memory hidden"] {
        let expected = if field.starts_with("memory") { 8 } else { 1 };
        for actual in [0, expected + 1] {
            let mut samples = demonstration(1.0);
            let mut memory = learner().policy().initial_memory();
            let values = match field {
                "observation" => &mut samples.get_mut(1).expect("second sample").observation,
                "action" => &mut samples.get_mut(1).expect("second sample").action,
                "memory cell" => &mut memory.cell,
                _ => &mut memory.hidden,
            };
            *values = vec![0.0; actual];
            rejected_without_mutation(
                &samples,
                &memory,
                RecurrentPpoError::DimensionMismatch {
                    field,
                    expected,
                    actual,
                },
            );
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut samples = demonstration(1.0);
            let mut memory = learner().policy().initial_memory();
            let values = match field {
                "observation" => &mut samples.get_mut(1).expect("second sample").observation,
                "action" => &mut samples.get_mut(1).expect("second sample").action,
                "memory cell" => &mut memory.cell,
                _ => &mut memory.hidden,
            };
            *values.first_mut().expect("nonempty vector") = invalid;
            rejected_without_mutation(
                &samples,
                &memory,
                RecurrentPpoError::NonFiniteValue {
                    field,
                    dimension: 0,
                },
            );
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
/// Both action endpoints are valid; values beyond either endpoint fail first.
fn action_bounds_are_inclusive_and_rejected_before_memory() {
    use bevy_gym::training::{RecurrentMemory, RecurrentPpoError};
    for action in [-1.0001, 1.0001] {
        let sample = RecurrentBehaviorSample {
            observation: vec![0.0],
            action: vec![action],
        };
        rejected_without_mutation(
            &[sample],
            &RecurrentMemory::zeros(0),
            RecurrentPpoError::InvalidSequence(
                "behavior actions must remain inside the configured bounds",
            ),
        );
    }
    for action in [-1.0, 1.0] {
        let mut agent = learner();
        let memory = agent.policy().initial_memory();
        let sample = RecurrentBehaviorSample {
            observation: vec![0.0],
            action: vec![action],
        };
        assert!(agent
            .behavior_clone_sequence(&[sample], &memory)
            .expect("inclusive endpoint")
            .is_finite());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
/// One temporal sample retains explicit-memory semantics without critic updates.
fn singleton_matches_explicit_memory_cloning_and_preserves_critic() {
    use bevy_gym::training::RecurrentBehaviorMemorySample;
    let mut sequence_agent = learner();
    let mut sample_agent = learner();
    let memory = sequence_agent
        .policy()
        .mean_action(&[0.9], &sequence_agent.policy().initial_memory())
        .expect("earlier cue")
        .next_memory;
    let original_memory = memory.clone();
    let before_value = sequence_agent
        .policy()
        .value(&[0.2], 0)
        .expect("critic before");
    let target = RecurrentBehaviorSample {
        observation: vec![0.2],
        action: vec![-0.7],
    };
    let sequence_loss = sequence_agent
        .behavior_clone_sequence(std::slice::from_ref(&target), &memory)
        .expect("one-step sequence");
    let sample_loss = sample_agent
        .behavior_clone_with_memory(&[RecurrentBehaviorMemorySample {
            observation: target.observation,
            action: target.action,
            initial_memory: memory.clone(),
        }])
        .expect("explicit-memory sample");
    assert!((sequence_loss - sample_loss).abs() < 1e-8);
    assert_eq!(
        sequence_agent
            .policy()
            .mean_action(&[0.2], &memory)
            .expect("sequence actor"),
        sample_agent
            .policy()
            .mean_action(&[0.2], &memory)
            .expect("sample actor")
    );
    assert_eq!(memory, original_memory);
    assert!(
        (sequence_agent
            .policy()
            .value(&[0.2], 0)
            .expect("critic after")
            - before_value)
            .abs()
            < f32::EPSILON
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
/// Different physical action ranges contribute equal normalized units to loss.
fn loss_averages_time_and_action_dimensions_in_normalized_units() {
    let mut agent = RecurrentPpoAgent::new(
        1,
        1,
        1,
        &[-2.0, 10.0],
        &[4.0, 14.0],
        RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![4],
            ..RecurrentPpoConfig::default()
        },
        SeedConfig::from_root(73),
    )
    .expect("asymmetric action bounds");
    let policy = agent.policy();
    let initial = policy.initial_memory();
    let samples = [
        RecurrentBehaviorSample {
            observation: vec![1.0],
            action: vec![-2.0, 14.0],
        },
        RecurrentBehaviorSample {
            observation: vec![0.0],
            action: vec![4.0, 10.0],
        },
    ];
    let mut memory = initial.clone();
    let mut expected_loss = 0.0_f64;
    for sample in &samples {
        let prediction = policy
            .mean_action(&sample.observation, &memory)
            .expect("pre-update action");
        // Physical half-spans are three and two action units; normalize each separately.
        for ((predicted, target), half_span) in
            prediction.action.iter().zip(&sample.action).zip([3.0, 2.0])
        {
            let residual = f64::from((predicted - target) / half_span);
            expected_loss += residual * residual / 4.0;
        }
        memory = prediction.next_memory;
    }
    let measured_loss = agent
        .behavior_clone_sequence(&samples, &initial)
        .expect("sequence loss");
    assert!((measured_loss - expected_loss).abs() < 1e-6);
}
