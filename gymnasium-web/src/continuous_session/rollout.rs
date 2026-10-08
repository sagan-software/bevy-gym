//! Bounded contiguous trajectories and time-limit-aware advantage estimates.
use crate::observation::Observation;
use bevy_gym::training::{RecurrentMemory, RecurrentPpoConfig, RecurrentPpoSequence};
use bevy_gym::EpisodeStatus;

/// Fields of one sampled transition stay together until PPO sequence construction.
#[derive(Debug)]
pub(super) struct RolloutStep {
    /// Task-specific normalized observation before the action.
    pub observation: Observation,
    /// Gaussian action sample before tanh squashing and physical scaling.
    pub pre_tanh_action: Vec<f32>,
    /// Behavior-policy density including the tanh correction.
    pub log_probability: f32,
    /// Shaped or scaled reward, used only for optimization.
    pub reward: f32,
    /// Critic estimate before the action.
    pub value: f32,
    /// Critic estimate of the actual next observation, before any reset.
    pub next_value: f32,
    /// Termination suppresses bootstrap; either boundary ends the GAE trace.
    pub status: EpisodeStatus,
}

/// Consume one nonempty trajectory segment in O(length) time and space.
pub(super) fn finish(steps: Vec<RolloutStep>, config: &RecurrentPpoConfig) -> RecurrentPpoSequence {
    let mut advantages = Vec::with_capacity(steps.len());
    let mut next_advantage = 0.0;
    // A time limit retains the final value estimate without leaking the next episode's trace.
    for step in steps.iter().rev() {
        let bootstrap = step.status.bootstrap_mask() as f32;
        let trace = f32::from(step.status == EpisodeStatus::Continuing);
        let delta = (config.gamma * bootstrap).mul_add(step.next_value, step.reward - step.value);
        let advantage = (config.gamma * config.gae_lambda * trace).mul_add(next_advantage, delta);
        advantages.push(advantage);
        next_advantage = advantage;
    }
    advantages.reverse();
    let length = steps.len();
    let mut sequence = RecurrentPpoSequence {
        observations: Vec::with_capacity(length),
        global_states: Vec::with_capacity(length),
        pre_tanh_actions: Vec::with_capacity(length),
        old_log_probabilities: Vec::with_capacity(length),
        returns: Vec::with_capacity(length),
        advantages,
        value_index: 0,
        initial_memory: RecurrentMemory::zeros(config.actor_hidden_size),
    };
    // The shared API owns actor and critic inputs separately; copy only the task's coordinates.
    for (step, advantage) in steps.into_iter().zip(&sequence.advantages) {
        sequence
            .observations
            .push(step.observation.as_ref().to_vec());
        sequence
            .global_states
            .push(step.observation.as_ref().to_vec());
        sequence.pre_tanh_actions.push(step.pre_tanh_action);
        sequence.old_log_probabilities.push(step.log_probability);
        sequence.returns.push(step.value + advantage);
    }
    sequence
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A literal two-state value example distinguishes termination from a time limit.
    fn transition(status: EpisodeStatus) -> RolloutStep {
        RolloutStep {
            observation: Observation::MountainCar([0.0, 0.0]),
            pre_tanh_action: vec![0.0],
            log_probability: 0.0,
            reward: 2.0,
            value: 1.0,
            next_value: 3.0,
            status,
        }
    }

    #[test]
    fn terminal_and_truncated_segments_have_distinct_bootstrap_values() {
        let config = super::super::config();
        for (status, expected) in [
            (EpisodeStatus::Terminated, 1.0),
            (EpisodeStatus::Truncated, 3.97),
        ] {
            let sequence = finish(vec![transition(status)], &config);
            assert!((sequence.advantages[0] - expected).abs() < 1e-6);
            assert!((sequence.returns[0] - (expected + 1.0)).abs() < 1e-6);
            assert_eq!(sequence.observations, sequence.global_states);
        }
        let sequence = finish(
            vec![
                transition(EpisodeStatus::Continuing),
                transition(EpisodeStatus::Truncated),
            ],
            &config,
        );
        assert!((sequence.advantages[0] - 7.703_785).abs() < 1e-5);
        assert!((sequence.advantages[1] - 3.97).abs() < 1e-6);
    }
}
