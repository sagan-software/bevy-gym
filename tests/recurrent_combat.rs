//! The combat actor must learn actual categorical requests alongside continuous actions.
use bevy_gym::training::{
    CombatSignal, RecurrentCombatAgent, RecurrentCombatPolicy, RecurrentCombatSequence,
    RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoSequence, RecurrentSampler, SeedConfig,
};

/// PPO must increase the rewarded signal through sampled joint probabilities.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn joint_policy_learns_requests_without_thresholding_continuous_actions() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 8,
        critic_hidden_sizes: vec![8],
        gamma: 0.0,
        actor_learning_rate: 0.01,
        epochs: 1,
        ..RecurrentPpoConfig::default()
    };
    let mut agent = RecurrentCombatAgent::new(
        2,
        3,
        1,
        &[-1.0],
        &[1.0],
        config.clone(),
        SeedConfig::from_root(0),
    )
    .expect("valid joint actor");
    let mut sampler = RecurrentSampler::new(42);
    let observation = [0.1, 0.2];
    let initial_probability = agent
        .policy()
        .signal_probabilities(&observation, &agent.policy().initial_memory())
        .expect("categorical distribution");
    for _ in 0..200 {
        let policy = agent.policy();
        let sequence = bandit_sequence(&policy, &mut sampler);
        let metrics = agent.update(&[sequence]).expect("joint PPO update");
        assert_eq!(metrics.valid_samples, 32);
    }
    let policy = agent.policy();
    let learned = policy
        .signal_probabilities(&observation, &policy.initial_memory())
        .expect("learned distribution");
    let target = CombatSignal::SearchRequested;
    assert!(learned[target.index()] > initial_probability[target.index()] + 0.3);
    assert_trained_round_trip(&policy, learned, &config);
}

/// Serialize trained parameters and retain their complete decisions after frozen loading.
fn assert_trained_round_trip(
    policy: &RecurrentCombatPolicy,
    probabilities: [f32; 8],
    config: &RecurrentPpoConfig,
) {
    let bytes = policy.to_bytes().expect("trained checkpoint");
    let loaded = RecurrentCombatPolicy::load_bytes(bytes.clone(), 2, 3, 1, &[-1.0], &[1.0], config)
        .expect("trained frozen policy");
    assert_eq!(
        probabilities.map(f32::to_bits),
        loaded
            .signal_probabilities(&[0.1, 0.2], &loaded.initial_memory())
            .expect("same learned probabilities")
            .map(f32::to_bits)
    );
    assert_joint_decisions(policy, &loaded);
    assert_eq!(
        bytes,
        loaded.to_bytes().expect("unchanged trained parameters")
    );
}

/// Frozen bytes retain both heads, memory and joint probabilities; Gaussian bytes are incompatible.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frozen_checkpoint_preserves_complete_joint_decisions() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        ..RecurrentPpoConfig::default()
    };
    let agent = RecurrentCombatAgent::new(
        2,
        3,
        2,
        &[-2.0],
        &[3.0],
        config.clone(),
        SeedConfig::from_root(5),
    )
    .expect("valid actor");
    let policy = agent.policy();
    let bytes = policy.to_bytes().expect("serializable joint record");
    let loaded =
        RecurrentCombatPolicy::load_bytes(bytes.clone(), 2, 3, 2, &[-2.0], &[3.0], &config)
            .expect("validated joint record");
    assert_joint_decisions(&policy, &loaded);
    for slot in 0..2 {
        assert_eq!(
            policy
                .value(&[0.2, 0.3, 0.4], slot)
                .expect("value")
                .to_bits(),
            loaded
                .value(&[0.2, 0.3, 0.4], slot)
                .expect("same value")
                .to_bits()
        );
    }
    assert_eq!(bytes, loaded.to_bytes().expect("frozen bytes unchanged"));
    reject_other_checkpoints(bytes, &config);
}

/// Invalid input stops before sampling or updating any parameters.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_boundaries_leave_policy_and_sampler_unchanged() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        ..RecurrentPpoConfig::default()
    };
    let mut agent = RecurrentCombatAgent::new(
        2,
        3,
        1,
        &[-1.0],
        &[1.0],
        config.clone(),
        SeedConfig::from_root(7),
    )
    .expect("valid actor");
    let policy = agent.policy();
    let bytes = policy.to_bytes().expect("bytes");
    reject_invalid_configurations(&bytes, &config);
    reject_invalid_inference(&policy);
    assert!(matches!(
        agent.update(&[]),
        Err(RecurrentPpoError::InvalidSequence(_))
    ));
    let initial = policy.initial_memory();
    let base = RecurrentPpoSequence {
        observations: vec![vec![0.0; 2]],
        global_states: vec![vec![0.0; 3]],
        pre_tanh_actions: vec![vec![0.0]],
        old_log_probabilities: vec![0.0],
        advantages: vec![1.0],
        returns: vec![1.0],
        value_index: 0,
        initial_memory: initial,
    };
    let invalid = RecurrentCombatSequence {
        continuous: base,
        signals: vec![],
    };
    assert!(matches!(
        agent.update(&[invalid]),
        Err(RecurrentPpoError::InvalidSequence(_))
    ));
    assert_eq!(
        bytes,
        agent.policy().to_bytes().expect("parameters unchanged")
    );
}

/// Build a contiguous real-policy trajectory with one sampled signal per action.
fn sampled_sequence(
    policy: &RecurrentCombatPolicy,
    length: usize,
    sampler: &mut RecurrentSampler,
) -> RecurrentCombatSequence {
    let initial_memory = policy.initial_memory();
    let mut memory = initial_memory.clone();
    let mut sequence = RecurrentCombatSequence {
        continuous: RecurrentPpoSequence {
            observations: Vec::new(),
            global_states: Vec::new(),
            pre_tanh_actions: Vec::new(),
            old_log_probabilities: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
            value_index: 0,
            initial_memory,
        },
        signals: Vec::new(),
    };
    for step in 0..length {
        let observation = [0.1, step as f32 / 4.0];
        let decision = policy
            .sample_action(&observation, &memory, sampler)
            .expect("joint sample");
        memory = decision.continuous.next_memory;
        sequence.continuous.observations.push(observation.to_vec());
        sequence.continuous.global_states.push(vec![0.0; 3]);
        sequence
            .continuous
            .pre_tanh_actions
            .push(decision.continuous.pre_tanh_action);
        sequence
            .continuous
            .old_log_probabilities
            .push(decision.log_probability);
        sequence.continuous.advantages.push(step as f32 - 1.0);
        sequence.continuous.returns.push(step as f32 / 2.0);
        sequence.signals.push(decision.signal);
    }
    sequence
}

/// Multiple unequal lanes update both heads while reporting only valid transitions.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn unequal_lane_updates_count_valid_samples_and_optimizer_steps() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        epochs: 2,
        minibatch_sequences: 2,
        ..RecurrentPpoConfig::default()
    };
    let mut agent =
        RecurrentCombatAgent::new(2, 3, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(17))
            .expect("actor");
    let mut sampler = RecurrentSampler::new(19);
    for update in 1..=2 {
        let policy = agent.policy();
        let before = policy.to_bytes().expect("bytes");
        let sequences = [1, 3].map(|length| sampled_sequence(&policy, length, &mut sampler));
        let metrics = agent.update(&sequences).expect("valid unequal lanes");
        assert_eq!(metrics.valid_samples, 4);
        assert_eq!(metrics.optimizer_updates, 2);
        assert_eq!(metrics.optimizer_steps, 2 * update);
        assert!([
            metrics.actor_loss,
            metrics.critic_loss,
            metrics.entropy,
            metrics.approximate_kl
        ]
        .iter()
        .all(|x| x.is_finite()));
        assert_ne!(before, agent.policy().to_bytes().expect("updated bytes"));
    }
}

/// Finite inputs that overflow normalization or objectives fail before the first optimizer step.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn overflowing_first_batch_leaves_parameters_unchanged() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        ..RecurrentPpoConfig::default()
    };
    let mut agent =
        RecurrentCombatAgent::new(2, 3, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(23))
            .expect("actor");
    let policy = agent.policy();
    let before = policy.to_bytes().expect("bytes");
    let base = sampled_sequence(&policy, 2, &mut RecurrentSampler::new(29));
    let mut normalization = base.clone();
    normalization.continuous.advantages.fill(f32::MAX);
    let mut critic = base.clone();
    critic.continuous.returns.fill(f32::MAX);
    let mut actor = base;
    actor.continuous.old_log_probabilities.fill(-f32::MAX);
    for (sequence, field) in [
        (normalization, "advantage normalization"),
        (critic, "critic objective"),
        (actor, "actor objective"),
    ] {
        assert!(
            matches!(agent.update(&[sequence]),Err(RecurrentPpoError::NonFiniteValue {field:actual,..}) if actual==field)
        );
        assert_eq!(
            before,
            agent.policy().to_bytes().expect("unchanged parameters")
        );
    }
}

/// Collect the actual joint policy's contextual-bandit actions and rewards.
fn bandit_sequence(
    policy: &RecurrentCombatPolicy,
    sampler: &mut RecurrentSampler,
) -> RecurrentCombatSequence {
    let initial_memory = policy.initial_memory();
    let mut memory = initial_memory.clone();
    let mut sequence = RecurrentCombatSequence {
        continuous: RecurrentPpoSequence {
            observations: Vec::new(),
            global_states: Vec::new(),
            pre_tanh_actions: Vec::new(),
            old_log_probabilities: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
            value_index: 0,
            initial_memory,
        },
        signals: Vec::new(),
    };
    let observation = [0.1, 0.2];
    let global_state = [0.3, 0.4, 0.5];
    for _ in 0..32 {
        let decision = policy
            .sample_action(&observation, &memory, sampler)
            .expect("actual joint sample");
        let reward = if decision.signal == CombatSignal::SearchRequested {
            1.0
        } else {
            -1.0
        };
        let value = policy.value(&global_state, 0).expect("centralized critic");
        memory = decision.continuous.next_memory;
        sequence.continuous.observations.push(observation.to_vec());
        sequence
            .continuous
            .global_states
            .push(global_state.to_vec());
        sequence
            .continuous
            .pre_tanh_actions
            .push(decision.continuous.pre_tanh_action);
        sequence
            .continuous
            .old_log_probabilities
            .push(decision.log_probability);
        sequence.continuous.advantages.push(reward - value);
        sequence.continuous.returns.push(reward);
        sequence.signals.push(decision.signal);
    }
    sequence
}

/// Compare seeded samples, deterministic means, recurrence and density sums after import.
fn assert_joint_decisions(policy: &RecurrentCombatPolicy, loaded: &RecurrentCombatPolicy) {
    let mut memory = policy.initial_memory();
    let mut sampler = RecurrentSampler::new(11);
    let mut other_sampler = RecurrentSampler::new(11);
    for observation in [[0.1, 0.2], [0.4, -0.5], [-0.3, 0.2]] {
        let action = policy
            .sample_action(&observation, &memory, &mut sampler)
            .expect("sample");
        assert_eq!(
            action,
            loaded
                .sample_action(&observation, &memory, &mut other_sampler)
                .expect("same sample")
        );
        assert_eq!(
            policy.mean_action(&observation, &memory).expect("mean"),
            loaded
                .mean_action(&observation, &memory)
                .expect("same mean")
        );
        let probabilities = policy
            .signal_probabilities(&observation, &memory)
            .expect("distribution");
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
        let joint = action.continuous.log_probability
            + probabilities
                .get(action.signal.index())
                .copied()
                .expect("closed signal column")
                .ln();
        assert!((joint - action.log_probability).abs() < 1e-5);
        assert!((-2.0..=3.0).contains(action.continuous.action.first().expect("one action")));
        memory = action.continuous.next_memory;
    }
}

/// Reject missing, malformed, Gaussian-only and wrong-shape checkpoint records.
fn reject_other_checkpoints(bytes: Vec<u8>, config: &RecurrentPpoConfig) {
    use bevy_gym::training::RecurrentPpoAgent;
    for invalid in [
        Vec::new(),
        vec![0, 1, 2],
        RecurrentPpoAgent::new(
            2,
            3,
            2,
            &[-2.0],
            &[3.0],
            config.clone(),
            SeedConfig::from_root(5),
        )
        .expect("Gaussian actor")
        .policy()
        .to_bytes()
        .expect("Gaussian record"),
    ] {
        assert!(matches!(
            RecurrentCombatPolicy::load_bytes(invalid, 2, 3, 2, &[-2.0], &[3.0], config),
            Err(RecurrentPpoError::Checkpoint(_))
        ));
    }
    for shapes in [(3, 3, 2), (2, 4, 2), (2, 3, 3)] {
        assert!(matches!(
            RecurrentCombatPolicy::load_bytes(
                bytes.clone(),
                shapes.0,
                shapes.1,
                shapes.2,
                &[-2.0],
                &[3.0],
                config
            ),
            Err(RecurrentPpoError::InvalidConfig { .. })
        ));
    }
}

/// Both construction and decoding reject invalid dimensions and actuator bounds.
fn reject_invalid_configurations(bytes: &[u8], config: &RecurrentPpoConfig) {
    for (obs, global, slots, low, high) in [
        (0, 3, 1, vec![-1.0], vec![1.0]),
        (2, 0, 1, vec![-1.0], vec![1.0]),
        (2, 3, 0, vec![-1.0], vec![1.0]),
        (2, 3, 1, vec![], vec![]),
        (2, 3, 1, vec![0.0], vec![0.0]),
        (2, 3, 1, vec![f32::NAN], vec![1.0]),
    ] {
        assert!(matches!(
            RecurrentCombatAgent::new(
                obs,
                global,
                slots,
                &low,
                &high,
                config.clone(),
                SeedConfig::from_root(7)
            ),
            Err(RecurrentPpoError::InvalidConfig { .. }
                | RecurrentPpoError::NonFiniteValue {
                    field: "action bound",
                    ..
                })
        ));
        assert!(matches!(
            RecurrentCombatPolicy::load_bytes(
                bytes.to_vec(),
                obs,
                global,
                slots,
                &low,
                &high,
                config
            ),
            Err(RecurrentPpoError::InvalidConfig { .. }
                | RecurrentPpoError::NonFiniteValue {
                    field: "action bound",
                    ..
                })
        ));
    }
}

/// Invalid local observations and memories leave the action sampler untouched.
fn reject_invalid_inference(policy: &RecurrentCombatPolicy) {
    let mut sampler = RecurrentSampler::new(9);
    let mut untouched = RecurrentSampler::new(9);
    for observation in [
        vec![],
        vec![0.0],
        vec![f32::NAN, 0.0],
        vec![0.0, f32::INFINITY],
    ] {
        assert!(matches!(
            policy.sample_action(&observation, &policy.initial_memory(), &mut sampler),
            Err(RecurrentPpoError::DimensionMismatch { .. }
                | RecurrentPpoError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            policy.mean_action(&observation, &policy.initial_memory()),
            Err(RecurrentPpoError::DimensionMismatch { .. }
                | RecurrentPpoError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            policy.signal_probabilities(&observation, &policy.initial_memory()),
            Err(RecurrentPpoError::DimensionMismatch { .. }
                | RecurrentPpoError::NonFiniteValue { .. })
        ));
    }
    reject_invalid_memories(policy, &mut sampler);
    assert_eq!(
        policy
            .sample_action(&[0.0, 0.0], &policy.initial_memory(), &mut sampler)
            .expect("unchanged sample"),
        policy
            .sample_action(&[0.0, 0.0], &policy.initial_memory(), &mut untouched)
            .expect("original sample")
    );
    for (state, slot) in [(vec![], 0), (vec![0.0; 3], 1), (vec![f32::NAN; 3], 0)] {
        assert!(matches!(
            policy.value(&state, slot),
            Err(RecurrentPpoError::DimensionMismatch { .. }
                | RecurrentPpoError::NonFiniteValue { .. }
                | RecurrentPpoError::ValueIndex { .. })
        ));
    }
}

/// Wrong-width and nonfinite memories cannot consume any random action draw.
fn reject_invalid_memories(policy: &RecurrentCombatPolicy, sampler: &mut RecurrentSampler) {
    use bevy_gym::training::RecurrentMemory;
    for memory in [
        RecurrentMemory::zeros(3),
        RecurrentMemory {
            cell: vec![f32::NAN; 4],
            hidden: vec![0.0; 4],
        },
    ] {
        assert!(matches!(
            policy.sample_action(&[0.0, 0.0], &memory, sampler),
            Err(RecurrentPpoError::DimensionMismatch { .. }
                | RecurrentPpoError::NonFiniteValue { .. })
        ));
    }
}

/// Continuous validation precedes categorical cardinality and preserves every parameter.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn continuous_contract_errors_precede_signal_errors() {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 4,
        critic_hidden_sizes: vec![4],
        ..RecurrentPpoConfig::default()
    };
    let mut agent =
        RecurrentCombatAgent::new(2, 3, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(31))
            .expect("actor");
    let debug = format!("{agent:?}");
    assert!(debug.contains("RecurrentCombatAgent"));
    assert!(debug.contains("optimizer_steps: 0"));
    let policy = agent.policy();
    let before = policy.to_bytes().expect("bytes");
    let mut sequence = sampled_sequence(&policy, 1, &mut RecurrentSampler::new(37));
    sequence.continuous.returns.clear();
    sequence.signals.clear();
    assert_eq!(
        agent
            .update(&[sequence])
            .expect_err("continuous length fails first"),
        RecurrentPpoError::InvalidSequence("all temporal fields must have the same nonzero length")
    );
    assert_eq!(before, agent.policy().to_bytes().expect("unchanged bytes"));
}
