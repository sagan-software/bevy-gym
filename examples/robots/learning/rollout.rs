//! Bounded on-policy collection with separate reset streams and recurrent memory.

use std::{error::Error, num::NonZeroU16};

use bevy_gym::robots::{DroneAction, DroneHover};
use bevy_gym::training::{
    RecurrentMemory, RecurrentPpoPolicy, RecurrentPpoSequence, RecurrentSampler, SeedConfig,
};
use bevy_gym::{Env, EpisodeStatus, TimeLimit};

use super::{decode_action, encode, GAE_LAMBDA, GAMMA};

/// Eight independent training lanes, each contributing 64 actions per update.
pub(crate) struct RecoveryBatch<E = DroneHover, const N: usize = 12>
where
    E: Env<Action = DroneAction>,
{
    /// Persistent environments and recurrent state, never shared with evaluation.
    lanes: [Lane<E>; 8],
    /// Root streams used to derive each lane's next episode seed.
    seeds: SeedConfig,
    /// Fixed-width task encoder shared by policy and value observations.
    encode: fn(E::Observation) -> [f32; N],
}

/// One continuing episode and its independent policy-sampling stream.
struct Lane<E: Env<Action = DroneAction>> {
    /// Private environment behind the lesson's positive action limit.
    environment: TimeLimit<E>,
    /// Most recent observation, replaced atomically after each action or reset.
    observation: E::Observation,
    /// Actor context retained across consecutive rollout batches.
    memory: RecurrentMemory,
    /// Independent Gaussian exploration stream.
    sampler: RecurrentSampler,
    /// Completed episodes in this lane.
    episode: u64,
}

/// One transition before advantage construction consumes the owned tensors.
struct Transition<const N: usize = 12> {
    /// Dimensionless body-frame policy input before the action.
    observation: [f32; N],
    /// Gaussian motor samples before tanh, consumed by the optimizer.
    pre_tanh_action: Vec<f32>,
    /// Behavior-policy log probability of those motor samples.
    log_probability: f32,
    /// Critic prediction before the transition.
    value: f32,
    /// Unscaled environment reward.
    reward: f32,
    /// Critic prediction at the final observation, before any reset.
    next_value: f32,
    /// Boundary state controlling bootstrapping and advantage propagation.
    status: EpisodeStatus,
}

impl RecoveryBatch {
    /// Initialize disturbed episodes and independent samplers from one root seed.
    pub(crate) fn new(seed: u64, policy: &RecurrentPpoPolicy) -> Self {
        Self::with_environment(seed, policy, DroneHover::disturbed)
    }

    /// Start fresh lanes for one lesson without replacing the caller's optimizer.
    pub(crate) fn with_environment(
        seed: u64,
        policy: &RecurrentPpoPolicy,
        make_environment: fn() -> DroneHover,
    ) -> Self {
        Self::with_task(
            seed,
            policy,
            make_environment,
            encode,
            NonZeroU16::new(500).expect("positive task limit"),
        )
    }
}

impl<E, const N: usize> RecoveryBatch<E, N>
where
    E: Env<Action = DroneAction>,
    E::Observation: Copy,
{
    /// Reuse bounded collection for another typed drone task and observation width.
    ///
    /// The encoder receives a copied snapshot with the environment's observation type.
    /// Collection retains eight independent lanes and their recurrent memories.
    pub(crate) fn with_task(
        seed: u64,
        policy: &RecurrentPpoPolicy,
        make_environment: fn() -> E,
        encode: fn(E::Observation) -> [f32; N],
        limit: NonZeroU16,
    ) -> Self {
        let seeds = SeedConfig::from_root(seed);
        let lanes = std::array::from_fn(|index| {
            let mut environment = TimeLimit::new(make_environment(), usize::from(limit.get()))
                .expect("positive task limit");
            let observation = environment
                .reset(Some(training_seed(seeds, index, 0)))
                .observation;
            Lane {
                environment,
                observation,
                memory: policy.initial_memory(),
                sampler: RecurrentSampler::new(seeds.action.wrapping_add(index as u64)),
                episode: 0,
            }
        });
        Self {
            lanes,
            seeds,
            encode,
        }
    }

    /// Collect exactly 512 transitions, splitting sequences at episode boundaries.
    pub(crate) fn collect(
        &mut self,
        policy: &RecurrentPpoPolicy,
    ) -> Result<Vec<RecurrentPpoSequence>, Box<dyn Error>> {
        let mut sequences = Vec::with_capacity(self.lanes.len());
        for (index, lane) in self.lanes.iter_mut().enumerate() {
            let mut initial_memory = lane.memory.clone();
            let mut transitions = Vec::with_capacity(64);
            for step in 0..64 {
                let observation = (self.encode)(lane.observation);
                let action = policy.sample_action(&observation, &lane.memory, &mut lane.sampler)?;
                let value = policy.value(&observation, 0)?;
                let result = lane.environment.step(decode_action(&action.action)?);
                // Read the final state's value before a reset can replace that state.
                let next_value = policy.value(&(self.encode)(result.observation), 0)?;
                lane.memory = action.next_memory;
                lane.observation = result.observation;
                transitions.push(Transition {
                    observation,
                    pre_tanh_action: action.pre_tanh_action,
                    log_probability: action.log_probability,
                    value,
                    reward: result.reward as f32,
                    next_value,
                    status: result.status,
                });

                if result.status.is_done() || step == 63 {
                    sequences.push(finish(std::mem::take(&mut transitions), initial_memory));
                    if result.status.is_done() {
                        lane.episode = lane.episode.wrapping_add(1);
                        lane.observation = lane
                            .environment
                            .reset(Some(training_seed(self.seeds, index, lane.episode)))
                            .observation;
                        lane.memory = policy.initial_memory();
                    }
                    initial_memory = lane.memory.clone();
                }
            }
        }
        Ok(sequences)
    }
}

/// Keep training below the high-bit evaluation partition and exclude small demo seeds.
const fn training_seed(seeds: SeedConfig, lane: usize, episode: u64) -> u64 {
    (seeds.environment_episode(lane, episode) & (u64::MAX >> 1)) | (1 << 10)
}

/// Construct one contiguous sequence; no transition after reset enters this buffer.
fn finish<const N: usize>(
    transitions: Vec<Transition<N>>,
    initial_memory: RecurrentMemory,
) -> RecurrentPpoSequence {
    let mut advantages = vec![0.0; transitions.len()];
    let mut next_advantage = 0.0;
    for (transition, advantage) in transitions.iter().zip(&mut advantages).rev() {
        let bootstrap = transition.status.bootstrap_mask() as f32;
        let trace = f32::from(!transition.status.is_done());
        let delta = (GAMMA * bootstrap)
            .mul_add(transition.next_value, transition.reward - transition.value);
        *advantage = (GAMMA * GAE_LAMBDA * trace).mul_add(next_advantage, delta);
        next_advantage = *advantage;
    }
    let length = transitions.len();
    let mut sequence = RecurrentPpoSequence {
        observations: Vec::with_capacity(length),
        global_states: Vec::with_capacity(length),
        pre_tanh_actions: Vec::with_capacity(length),
        old_log_probabilities: Vec::with_capacity(length),
        advantages,
        returns: Vec::with_capacity(length),
        value_index: 0,
        initial_memory,
    };
    for (transition, advantage) in transitions.into_iter().zip(&sequence.advantages) {
        // The existing optimizer owns separate actor and critic input matrices.
        sequence.observations.push(transition.observation.to_vec());
        sequence.global_states.push(transition.observation.to_vec());
        sequence.pre_tanh_actions.push(transition.pre_tanh_action);
        sequence
            .old_log_probabilities
            .push(transition.log_probability);
        sequence.returns.push(transition.value + advantage);
    }
    sequence
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::{new_agent, SELECTION_SEEDS};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn incompatible_policies_cannot_advance_physics() {
        use crate::learning::{evaluate, learning_config};
        use bevy_gym::training::RecurrentPpoAgent;

        let valid = new_agent(7).expect("valid recipe").policy();
        for (width, low, high) in [
            (11, vec![0.0; 4], vec![1.0; 4]),
            (12, vec![0.0; 3], vec![1.0; 3]),
            (12, vec![2.0; 4], vec![3.0; 4]),
        ] {
            let wrong = RecurrentPpoAgent::new(
                width,
                width,
                1,
                &low,
                &high,
                learning_config(),
                SeedConfig::from_root(9),
            )
            .expect("valid network for a different task")
            .policy();
            let mut batch = RecoveryBatch::new(7, &valid);
            let observations = batch.lanes.each_ref().map(|lane| lane.observation);
            batch
                .collect(&wrong)
                .expect_err("wrong input or motor domain must fail");
            assert_eq!(
                observations,
                batch.lanes.each_ref().map(|lane| lane.observation)
            );
            assert!(batch.lanes.iter().all(|lane| lane.episode == 0));
            evaluate(&wrong, &[42]).expect_err("evaluation rejects the same mismatch");
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn training_seeds_exclude_selection_and_high_bit_evaluation() {
        for root in [0, 7, u64::MAX] {
            let seeds = SeedConfig::from_root(root);
            for lane in 0..8 {
                for episode in 0..100 {
                    let seed = training_seed(seeds, lane, episode);
                    assert_eq!(seed >> 63, 0);
                    assert_ne!(seed & (1 << 10), 0);
                    assert!(!SELECTION_SEEDS.contains(&seed));
                    assert_eq!(seed, training_seed(seeds, lane, episode));
                }
            }
        }
    }

    /// Build one algebraic GAE case without physics or learned value estimates.
    fn transition(status: EpisodeStatus) -> Transition {
        Transition {
            observation: [0.25; 12],
            pre_tanh_action: vec![0.0; 4],
            log_probability: -2.0,
            value: 3.0,
            reward: 2.0,
            next_value: 5.0,
            status,
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn gae_bootstraps_truncation_and_stops_traces_at_both_endings() {
        for status in [
            EpisodeStatus::Continuing,
            EpisodeStatus::Terminated,
            EpisodeStatus::Truncated,
        ] {
            let memory = RecurrentMemory {
                cell: vec![0.2],
                hidden: vec![0.3],
            };
            let sequence = finish(
                vec![transition(status), transition(EpisodeStatus::Terminated)],
                memory.clone(),
            );
            let expected = match status {
                EpisodeStatus::Continuing => GAMMA.mul_add(-GAE_LAMBDA, 3.975),
                EpisodeStatus::Terminated => -1.0,
                EpisodeStatus::Truncated => 3.975,
            };
            assert!((sequence.advantages[0] - expected).abs() < 1e-5);
            assert_eq!(sequence.advantages[1].to_bits(), (-1.0_f32).to_bits());
            assert!((sequence.returns[0] - (3.0 + expected)).abs() < 1e-5);
            assert_eq!(sequence.initial_memory, memory);
            assert_eq!(sequence.observations, sequence.global_states);
            assert_eq!(sequence.pre_tanh_actions, vec![vec![0.0; 4]; 2]);
            assert_eq!(sequence.old_log_probabilities, [-2.0; 2]);
        }
        let cutoff = finish(
            vec![transition(EpisodeStatus::Continuing)],
            RecurrentMemory::zeros(1),
        );
        assert!((cutoff.advantages[0] - 3.975).abs() < 1e-5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn a_rollout_boundary_preserves_memory_and_episode_boundaries_reset_it() {
        let policy = new_agent(7).unwrap().policy();
        let mut batch = RecoveryBatch::new(7, &policy);
        batch.collect(&policy).unwrap();
        let memory = batch.lanes[0].memory.clone();
        assert_ne!(memory, policy.initial_memory());
        let next = batch.collect(&policy).unwrap();
        assert_eq!(next[0].initial_memory, memory);

        // One-step truncations force a reset before every subsequent sample.
        batch.lanes[0].environment = TimeLimit::new(DroneHover::disturbed(), 1).unwrap();
        batch.lanes[0].observation = batch.lanes[0].environment.reset(Some(1024)).observation;
        batch.lanes[0].memory = policy.initial_memory();
        let episode = batch.lanes[0].episode;
        let truncated = batch.collect(&policy).unwrap();
        assert_eq!(batch.lanes[0].episode, episode + 64);
        for sequence in &truncated[..64] {
            assert_eq!(sequence.observations.len(), 1);
            assert_eq!(sequence.initial_memory, policy.initial_memory());
            let observation: [f32; 12] = sequence.observations[0].clone().try_into().unwrap();
            let reward_plus_bootstrap = sequence.returns[0];
            assert!(reward_plus_bootstrap.is_finite());
            assert!(observation.iter().all(|value| value.is_finite()));
        }
    }
}
