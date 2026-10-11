//! Train the joint PPO API on a small bandit; this does not qualify physical combat.
use bevy_gym::training::{
    CombatSignal, RecurrentCombatAgent, RecurrentCombatPolicy, RecurrentCombatSequence,
    RecurrentPpoConfig, RecurrentPpoSequence, RecurrentSampler, SeedConfig,
};

/// Train with actual sampled choices before printing the frozen policy's decision.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 8,
        critic_hidden_sizes: vec![8],
        gamma: 0.0,
        actor_learning_rate: 0.01,
        epochs: 1,
        ..RecurrentPpoConfig::default()
    };
    let mut agent =
        RecurrentCombatAgent::new(2, 3, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(0))?;
    let mut sampler = RecurrentSampler::new(42);
    for _ in 0..200 {
        let sequence = collect(&agent.policy(), &mut sampler)?;
        agent.update(&[sequence])?;
    }
    let policy = agent.policy();
    let decision = policy.mean_action(&[0.1, 0.2], &policy.initial_memory())?;
    let signal = decision.signal;
    let bytes = policy.to_bytes()?.len();
    println!("Joint PPO bandit: {signal:?}; 6400 transitions; {bytes} checkpoint bytes; physical combat unqualified");
    Ok(())
}

/// Start an intact trajectory with this agent's episode-boundary memory.
fn empty_sequence(policy: &RecurrentCombatPolicy) -> RecurrentCombatSequence {
    RecurrentCombatSequence {
        continuous: RecurrentPpoSequence {
            observations: Vec::new(),
            global_states: Vec::new(),
            pre_tanh_actions: Vec::new(),
            old_log_probabilities: Vec::new(),
            advantages: Vec::new(),
            returns: Vec::new(),
            value_index: 0,
            initial_memory: policy.initial_memory(),
        },
        signals: Vec::new(),
    }
}

/// The reward grades sampled signals; it never supplies the policy's choice.
fn collect(
    policy: &RecurrentCombatPolicy,
    sampler: &mut RecurrentSampler,
) -> Result<RecurrentCombatSequence, bevy_gym::training::RecurrentPpoError> {
    let mut sequence = empty_sequence(policy);
    let mut memory = sequence.continuous.initial_memory.clone();
    for _ in 0..32 {
        let decision = policy.sample_action(&[0.1, 0.2], &memory, sampler)?;
        let reward = if decision.signal == CombatSignal::SearchRequested {
            1.0
        } else {
            -1.0
        };
        let value = policy.value(&[0.3, 0.4, 0.5], 0)?;
        memory = decision.continuous.next_memory;
        sequence.continuous.observations.push(vec![0.1, 0.2]);
        sequence.continuous.global_states.push(vec![0.3, 0.4, 0.5]);
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
    Ok(sequence)
}
