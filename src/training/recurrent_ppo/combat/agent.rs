//! Joint clipped PPO with independent recurrent lanes and a centralized critic.
use super::super::{
    encode_matrix, tensor_log_probabilities, training_device, validate_finite, CentralCritic,
    LstmState, MinibatchRng, RecurrentPpoConfig, RecurrentPpoError, RecurrentSampler, SeedConfig,
    TrainingBackend, TrainingDevice, UpdateTotals, LOG_TWO_PI, MODEL_INITIALIZATION_LOCK,
};
use super::{actor::CombatActor, policy::RecurrentCombatPolicy, sequence::RecurrentCombatSequence};
use burn::module::AutodiffModule;
use burn::optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion};
use burn::tensor::{activation::log_softmax, Tensor};
use std::num::NonZeroUsize;

/// Learner for one parameter-sharing combat team; agent memories remain in trajectories.
pub struct RecurrentCombatAgent {
    /// Autodiff local actor.
    actor: CombatActor<TrainingBackend>,

    /// Autodiff centralized critic.
    critic: CentralCritic<TrainingBackend>,

    /// Actor Adam state.
    actor_optimizer: OptimizerAdaptor<Adam, CombatActor<TrainingBackend>, TrainingBackend>,

    /// Critic Adam state.
    critic_optimizer: OptimizerAdaptor<Adam, CentralCritic<TrainingBackend>, TrainingBackend>,

    /// Training backend device.
    device: TrainingDevice,

    /// Local observation width.
    observation_dim: NonZeroUsize,

    /// Global state width.
    global_state_dim: NonZeroUsize,

    /// Stable possible-agent value count.
    value_count: NonZeroUsize,

    /// Recurrent width derived once from the validated architecture.
    hidden_size: NonZeroUsize,

    /// Inclusive action lower bounds.
    action_low: Vec<f32>,

    /// Inclusive action upper bounds.
    action_high: Vec<f32>,

    /// Validated hyperparameters.
    config: RecurrentPpoConfig,

    /// Completed actor and critic optimizer steps.
    optimizer_steps: u64,

    /// Dedicated deterministic recurrent-minibatch ordering stream.
    minibatch_rng: MinibatchRng,

    /// Independent fresh Gaussian draws for the bounded continuous entropy estimate.
    entropy_sampler: RecurrentSampler,
}

impl RecurrentCombatAgent {
    /// Validate shapes and bounds before seeded initialization of the joint actor and critic.
    /// The config entropy coefficient weights bounded continuous and categorical joint entropy.
    ///
    /// # Errors
    /// Returns the existing configuration errors before constructing any network.
    pub fn new(
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: RecurrentPpoConfig,
        seeds: SeedConfig,
    ) -> Result<Self, RecurrentPpoError> {
        config.validate(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
        )?;
        let device = training_device();
        let (actor, critic) = initialize_networks(
            observation_dim,
            global_state_dim,
            value_count,
            action_low.len(),
            &config,
            seeds,
            device,
        );
        let actor_optimizer = AdamConfig::new()
            .with_epsilon(config.optimizer_epsilon)
            .init::<TrainingBackend, CombatActor<TrainingBackend>>();
        let critic_optimizer = AdamConfig::new()
            .with_epsilon(config.optimizer_epsilon)
            .init::<TrainingBackend, CentralCritic<TrainingBackend>>();
        Ok(Self {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            device,
            observation_dim: super::positive_width(observation_dim)?,
            global_state_dim: super::positive_width(global_state_dim)?,
            value_count: super::positive_width(value_count)?,
            action_low: action_low.to_vec(),
            action_high: action_high.to_vec(),
            hidden_size: super::positive_width(config.actor_hidden_size)?,
            config,
            optimizer_steps: 0,
            minibatch_rng: MinibatchRng::new(seeds.rollout),
            entropy_sampler: RecurrentSampler::new(seeds.action ^ 0x636f_6d62_6174_0001),
        })
    }

    /// Snapshot actor and critic parameters for collection and evaluation.
    #[must_use]
    pub fn policy(&self) -> RecurrentCombatPolicy {
        RecurrentCombatPolicy {
            actor: self.actor.valid(),
            critic: self.critic.valid(),
            observation_dim: self.observation_dim,
            global_state_dim: self.global_state_dim,
            value_count: self.value_count,
            hidden_size: self.hidden_size,
            action_low: self.action_low.clone(),
            action_high: self.action_high.clone(),
            log_std_min: self.config.log_std_min,
            log_std_max: self.config.log_std_max,
            log_probability_epsilon: self.config.log_probability_epsilon,
        }
    }

    /// Update both heads using one joint ratio, retaining intact sequences and private memories.
    ///
    /// # Errors
    /// Rejects empty batches, malformed trajectories and non-finite objectives.
    pub fn update(
        &mut self,
        sequences: &[RecurrentCombatSequence],
    ) -> Result<super::RecurrentCombatUpdate, RecurrentPpoError> {
        if sequences.is_empty() {
            return Err(RecurrentPpoError::invalid_sequence(
                "at least one sequence is required",
            ));
        }
        for sequence in sequences {
            sequence.validate(
                self.observation_dim.get(),
                self.global_state_dim.get(),
                self.action_low.len(),
                self.config.actor_hidden_size,
                self.value_count.get(),
            )?;
        }
        let (count, mean, scale) = normalization(sequences, self.config.advantage_epsilon)?;
        let mut totals = UpdateTotals::default();
        let mut optimizer_updates = 0_u64;
        let mut order = sequences.iter().collect::<Vec<_>>();
        for _ in 0..self.config.epochs {
            self.minibatch_rng.shuffle(&mut order);
            for minibatch in order.chunks(self.config.minibatch_sequences) {
                self.update_minibatch(minibatch, mean, scale, &mut totals)?;
                optimizer_updates = optimizer_updates.saturating_add(1);
            }
        }
        let divisor = totals.valid_timesteps as f64;
        Ok(super::RecurrentCombatUpdate {
            optimizer_steps: self.optimizer_steps,
            optimizer_updates,
            valid_samples: count as u64,
            actor_loss: totals.actor_loss / divisor,
            critic_loss: totals.critic_loss / divisor,
            entropy: totals.entropy / divisor,
            approximate_kl: totals.approximate_kl / divisor,
            actor_learning_rate: self.config.actor_learning_rate,
            critic_learning_rate: self.config.critic_learning_rate,
        })
    }

    /// Check one complete minibatch before applying its joint actor and critic gradients.
    fn update_minibatch(
        &mut self,
        minibatch: &[&RecurrentCombatSequence],
        mean: f32,
        scale: f32,
        totals: &mut UpdateTotals,
    ) -> Result<(), RecurrentPpoError> {
        let timesteps = minibatch
            .iter()
            .map(|sequence| sequence.continuous.observations.len())
            .sum::<usize>();
        let (actor_loss, metrics) = self.actor_loss_batch(minibatch, mean, scale);
        let (critic_loss, critic_value) = self.critic_batch_loss(minibatch);
        validate_finite(&[critic_value as f32], "critic objective")?;
        validate_finite(
            &[
                metrics.loss as f32,
                metrics.entropy as f32,
                metrics.approximate_kl as f32,
            ],
            "actor objective",
        )?;
        self.apply_gradients(actor_loss, critic_loss);
        totals.critic_loss = critic_value.mul_add(timesteps as f64, totals.critic_loss);
        totals.actor_loss = metrics.loss.mul_add(timesteps as f64, totals.actor_loss);
        totals.entropy = metrics.entropy.mul_add(timesteps as f64, totals.entropy);
        totals.approximate_kl = metrics
            .approximate_kl
            .mul_add(timesteps as f64, totals.approximate_kl);
        totals.valid_timesteps = totals.valid_timesteps.saturating_add(timesteps as u64);
        Ok(())
    }

    /// Apply gradients after objective validation and preserve exploration parameter identity.
    fn apply_gradients(
        &mut self,
        actor_loss: Tensor<TrainingBackend, 1>,
        critic_loss: Tensor<TrainingBackend, 1>,
    ) {
        // Both heads and the shared recurrent representation receive the same clipped joint loss.
        let gradients = GradientsParams::from_grads(actor_loss.backward(), &self.actor);
        self.actor = self.actor_optimizer.step(
            self.config.actor_learning_rate,
            self.actor.clone(),
            gradients,
        );
        // Preserve parameter identity and inward gradients at both exploration boundaries.
        self.actor.continuous.log_std = self.actor.continuous.log_std.clone().map(|tensor| {
            tensor
                .clamp(self.config.log_std_min, self.config.log_std_max)
                .detach()
                .require_grad()
        });
        let gradients = GradientsParams::from_grads(critic_loss.backward(), &self.critic);
        self.critic = self.critic_optimizer.step(
            self.config.critic_learning_rate,
            self.critic.clone(),
            gradients,
        );
        self.optimizer_steps = self.optimizer_steps.saturating_add(1);
    }

    /// Sum Gaussian and categorical log probabilities before the single PPO ratio and clip.
    fn actor_loss_batch(
        &mut self,
        sequences: &[&RecurrentCombatSequence],
        advantage_mean: f32,
        advantage_scale: f32,
    ) -> (Tensor<TrainingBackend, 1>, JointMetrics) {
        let (mean, log_std, logits) = self.actor_batch_distribution(sequences);
        let (samples, old_log_probabilities, advantages) =
            self.actor_batch_targets(sequences, advantage_mean, advantage_scale);
        let gaussian = tensor_log_probabilities(
            mean.clone(),
            log_std.clone(),
            samples,
            &self.action_low,
            &self.action_high,
            self.config.log_probability_epsilon,
            &self.device,
        );
        let (categorical, categorical_logs) = self.categorical_log_probabilities(logits, sequences);
        let joint = gaussian + categorical;
        let policy_loss = clipped_joint_loss(
            joint.clone(),
            old_log_probabilities.clone(),
            advantages,
            self.config.clip_ratio,
        );
        let continuous_entropy = self.continuous_entropy(mean, log_std);
        // Categorical Shannon entropy is exact for the eight-column distribution.
        let categorical_entropy = (categorical_logs.clone().exp() * categorical_logs)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
            .neg();
        let entropy = (continuous_entropy + categorical_entropy).mean();
        let loss = policy_loss - entropy.clone() * self.config.entropy_coefficient;
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let entropy_value = entropy.into_scalar().elem::<f64>();
        let approximate_kl = (old_log_probabilities - joint)
            .mean()
            .into_scalar()
            .elem::<f64>();
        (
            loss,
            JointMetrics {
                loss: loss_value,
                entropy: entropy_value,
                approximate_kl,
            },
        )
    }

    /// Gather only each trajectory's actual categorical choice, retaining logits for entropy.
    fn categorical_log_probabilities(
        &self,
        logits: Tensor<TrainingBackend, 2>,
        sequences: &[&RecurrentCombatSequence],
    ) -> (Tensor<TrainingBackend, 1>, Tensor<TrainingBackend, 2>) {
        let categorical_logs = log_softmax(logits, 1);
        let choices = sequences
            .iter()
            .flat_map(|sequence| sequence.signals.iter().map(|signal| signal.index() as i64))
            .collect::<Vec<_>>();
        let choices = Tensor::<TrainingBackend, 1, burn::tensor::Int>::from_ints(
            choices.as_slice(),
            &self.device,
        )
        .reshape([choices.len(), 1]);
        let categorical = categorical_logs
            .clone()
            .gather(1, choices)
            .squeeze_dim::<1>(1);
        (categorical, categorical_logs)
    }

    /// Estimate bounded Gaussian entropy with fresh reparameterized samples and the density's Jacobian floor.
    fn continuous_entropy(
        &mut self,
        mean: Tensor<TrainingBackend, 2>,
        log_std: Tensor<TrainingBackend, 2>,
    ) -> Tensor<TrainingBackend, 1> {
        // Estimate bounded continuous entropy with fresh reparameterized Gaussian draws.
        let [rows, columns] = mean.dims();
        let noise = (0..rows * columns)
            .map(|_| self.entropy_sampler.normal())
            .collect::<Vec<_>>();
        let noise = Tensor::<TrainingBackend, 1>::from_floats(noise.as_slice(), &self.device)
            .reshape([rows, columns]);
        let entropy_samples = mean + log_std.clone().exp() * noise;
        let squashed = entropy_samples.tanh();
        let log_jacobian = (squashed.clone() * squashed)
            .neg()
            .add_scalar(1.0)
            .clamp_min(self.config.log_probability_epsilon)
            .log();
        let affine = self
            .action_low
            .iter()
            .zip(&self.action_high)
            .map(|(low, high)| ((high - low) * 0.5).ln())
            .collect::<Vec<_>>();
        let affine = Tensor::<TrainingBackend, 1>::from_floats(affine.as_slice(), &self.device)
            .reshape([1, columns]);
        (log_std + f32::midpoint(1.0, LOG_TWO_PI) + log_jacobian + affine)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }

    /// Encode padded observations and independent initial memories without cloning host rows.
    ///
    /// Burn 0.21 LSTM uses batch-first `[lanes, time, features]` inputs and
    /// `[lanes, hidden]` states:
    /// <https://docs.rs/burn/0.21.0/burn/nn/struct.Lstm.html#method.forward>.
    /// Tensor/graph size is proportional to minibatch lanes times maximum unroll length.
    fn actor_minibatch_inputs(
        &self,
        sequences: &[&RecurrentCombatSequence],
    ) -> (
        Tensor<TrainingBackend, 3>,
        LstmState<TrainingBackend, 2>,
        usize,
    ) {
        let sequence_length = sequences
            .iter()
            .map(|sequence| sequence.continuous.observations.len())
            .max()
            .expect("validated nonempty minibatch");
        let zero_row = vec![0.0; self.observation_dim.get()];
        let observations = sequences
            .iter()
            .flat_map(|sequence| {
                sequence
                    .continuous
                    .observations
                    .iter()
                    .map(Vec::as_slice)
                    .chain(std::iter::repeat_n(
                        zero_row.as_slice(),
                        sequence_length - sequence.continuous.observations.len(),
                    ))
            })
            .collect::<Vec<_>>();
        let observations = encode_matrix::<TrainingBackend>(
            &observations,
            self.observation_dim.get(),
            &self.device,
        )
        .reshape([sequences.len(), sequence_length, self.observation_dim.get()]);
        let cells = sequences
            .iter()
            .map(|sequence| sequence.continuous.initial_memory.cell.as_slice())
            .collect::<Vec<_>>();
        let hidden = sequences
            .iter()
            .map(|sequence| sequence.continuous.initial_memory.hidden.as_slice())
            .collect::<Vec<_>>();
        let state = LstmState::new(
            encode_matrix::<TrainingBackend>(&cells, self.config.actor_hidden_size, &self.device),
            encode_matrix::<TrainingBackend>(&hidden, self.config.actor_hidden_size, &self.device),
        );
        (observations, state, sequence_length)
    }

    /// Evaluate one recurrent batch and retain valid output rows in sequence order.
    fn actor_batch_distribution(
        &self,
        sequences: &[&RecurrentCombatSequence],
    ) -> (
        Tensor<TrainingBackend, 2>,
        Tensor<TrainingBackend, 2>,
        Tensor<TrainingBackend, 2>,
    ) {
        let (observations, state, sequence_length) = self.actor_minibatch_inputs(sequences);
        let distribution = self.actor.forward(
            observations,
            Some(state),
            self.action_low.len(),
            self.config.log_std_min,
            self.config.log_std_max,
        );
        // Padding occurs after each intact sequence. Selecting valid rows before
        // density evaluation prevents padded ratios or advantages from affecting gradients.
        let indices = sequences
            .iter()
            .enumerate()
            .flat_map(|(lane, sequence)| {
                (0..sequence.continuous.observations.len())
                    .map(move |step| (lane * sequence_length + step) as i64)
            })
            .collect::<Vec<_>>();
        let indices = Tensor::<TrainingBackend, 1, burn::tensor::Int>::from_ints(
            indices.as_slice(),
            &self.device,
        );
        (
            distribution.mean.select(0, indices.clone()),
            distribution.log_std.select(0, indices.clone()),
            distribution.logits.select(0, indices),
        )
    }

    /// Encode stored actions, behavior probabilities and normalized advantages for valid rows.
    fn actor_batch_targets(
        &self,
        sequences: &[&RecurrentCombatSequence],
        advantage_mean: f32,
        advantage_scale: f32,
    ) -> (
        Tensor<TrainingBackend, 2>,
        Tensor<TrainingBackend, 1>,
        Tensor<TrainingBackend, 1>,
    ) {
        let samples = sequences
            .iter()
            .flat_map(|sequence| {
                sequence
                    .continuous
                    .pre_tanh_actions
                    .iter()
                    .map(Vec::as_slice)
            })
            .collect::<Vec<_>>();
        let samples =
            encode_matrix::<TrainingBackend>(&samples, self.action_low.len(), &self.device);
        let old_log_probabilities = sequences
            .iter()
            .flat_map(|sequence| sequence.continuous.old_log_probabilities.iter().copied())
            .collect::<Vec<_>>();
        let old_log_probabilities = Tensor::<TrainingBackend, 1>::from_floats(
            old_log_probabilities.as_slice(),
            &self.device,
        );
        let normalized_advantages = sequences
            .iter()
            .flat_map(|sequence| {
                sequence
                    .continuous
                    .advantages
                    .iter()
                    .map(|value| (value - advantage_mean) / advantage_scale)
            })
            .collect::<Vec<_>>();
        let advantages = Tensor::<TrainingBackend, 1>::from_floats(
            normalized_advantages.as_slice(),
            &self.device,
        );
        (samples, old_log_probabilities, advantages)
    }

    /// Evaluate centralized values for valid rows, preserving each lane's selected value slot.
    fn critic_batch_loss(
        &self,
        sequences: &[&RecurrentCombatSequence],
    ) -> (Tensor<TrainingBackend, 1>, f64) {
        let states = sequences
            .iter()
            .flat_map(|sequence| sequence.continuous.global_states.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let states =
            encode_matrix::<TrainingBackend>(&states, self.global_state_dim.get(), &self.device);
        let slots = sequences
            .iter()
            .flat_map(|sequence| {
                std::iter::repeat_n(
                    sequence.continuous.value_index as i64,
                    sequence.continuous.returns.len(),
                )
            })
            .collect::<Vec<_>>();
        let slots = Tensor::<TrainingBackend, 1, burn::tensor::Int>::from_ints(
            slots.as_slice(),
            &self.device,
        )
        .reshape([slots.len(), 1]);
        let values = self
            .critic
            .forward(states)
            .gather(1, slots)
            .squeeze_dim::<1>(1);
        let returns = sequences
            .iter()
            .flat_map(|sequence| sequence.continuous.returns.iter().copied())
            .collect::<Vec<_>>();
        let returns = Tensor::<TrainingBackend, 1>::from_floats(returns.as_slice(), &self.device);
        let residual = values - returns;
        let loss = residual.clone().mul(residual).mean() * 0.5;
        let value = loss.clone().into_scalar().elem::<f64>();
        (loss, value)
    }
}

impl std::fmt::Debug for RecurrentCombatAgent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RecurrentCombatAgent")
            .field("observation_dim", &self.observation_dim)
            .field("global_state_dim", &self.global_state_dim)
            .field("value_count", &self.value_count)
            .field("action_dim", &self.action_low.len())
            .field("optimizer_steps", &self.optimizer_steps)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Clip one joint probability ratio after the action-head log densities have been summed.
fn clipped_joint_loss(
    joint: Tensor<TrainingBackend, 1>,
    old: Tensor<TrainingBackend, 1>,
    advantages: Tensor<TrainingBackend, 1>,
    clip: f32,
) -> Tensor<TrainingBackend, 1> {
    let ratios = (joint - old).exp();
    let unclipped = ratios.clone() * advantages.clone();
    let clipped = ratios.clamp(1.0 - clip, 1.0 + clip) * advantages;
    unclipped.min_pair(clipped).mean().neg()
}

/// Materialize seeded parameters while the initialization lock protects lazy backend draws.
fn initialize_networks(
    observation_dim: usize,
    global_state_dim: usize,
    value_count: usize,
    actions: usize,
    config: &RecurrentPpoConfig,
    seeds: SeedConfig,
    device: TrainingDevice,
) -> (CombatActor<TrainingBackend>, CentralCritic<TrainingBackend>) {
    let _initialization_guard = MODEL_INITIALIZATION_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    TrainingBackend::seed(&device, seeds.model);
    let actor = CombatActor::new(
        observation_dim,
        config.actor_hidden_size,
        actions,
        config.initial_log_std,
        &device,
    );
    let critic = CentralCritic::new(
        global_state_dim,
        &config.critic_hidden_sizes,
        value_count,
        &device,
    );

    // Burn may initialize parameters lazily. Materialize both networks
    // while the model seed lock still protects the backend RNG.
    let zero_local = Tensor::zeros([1, 1, observation_dim], &device);
    let distribution = actor.forward(
        zero_local,
        None,
        actions,
        config.log_std_min,
        config.log_std_max,
    );
    let _mean_materialized = (distribution.mean + distribution.log_std).into_data();
    let _signal_materialized = distribution.logits.into_data();
    let _critic_materialized = critic
        .forward(Tensor::zeros([1, global_state_dim], &device))
        .into_data();

    (actor, critic)
}

/// Normalize only validated real advantages, rejecting overflow before optimization.
fn normalization(
    sequences: &[RecurrentCombatSequence],
    epsilon: f32,
) -> Result<(usize, f32, f32), RecurrentPpoError> {
    let count = sequences
        .iter()
        .map(|sequence| sequence.continuous.advantages.len())
        .sum::<usize>();
    let mean = sequences
        .iter()
        .flat_map(|sequence| &sequence.continuous.advantages)
        .copied()
        .sum::<f32>()
        / count as f32;
    let variance = sequences
        .iter()
        .flat_map(|sequence| &sequence.continuous.advantages)
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / count as f32;
    let scale = (variance + epsilon).sqrt();
    validate_finite(&[mean, scale], "advantage normalization")?;
    Ok((count, mean, scale))
}
/// Actor measurements for the complete joint density rather than a Gaussian-only policy.
struct JointMetrics {
    /// Mean clipped joint actor objective.
    loss: f64,
    /// Mean bounded-continuous estimate plus categorical Shannon entropy.
    entropy: f64,
    /// Mean behavior-to-current joint log-density difference.
    approximate_kl: f64,
}

#[cfg(test)]
mod tests {
    use super::super::super::{memory_to_state, tensor_vec, RecurrentMemory, RecurrentPpoSequence};
    use super::*;
    /// The complete density is clipped once, including positive and negative advantage boundaries.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn joint_ratio_precedes_clipping_for_both_advantage_signs() {
        let device = training_device();
        for (gaussian, categorical, advantage, expected) in [
            (2.0_f32, 0.5_f32, 1.0_f32, -1.0_f32),
            (3.0, 1.0, 1.0, -1.2),
            (0.1, 1.0, -1.0, 0.8),
            (3.0, 1.0, -1.0, 3.0),
            (0.1, 1.0, 1.0, -0.1),
        ] {
            let joint = Tensor::from_floats([gaussian.ln() + categorical.ln()], &device);
            let old = Tensor::from_floats([0.0], &device);
            let advantages = Tensor::from_floats([advantage], &device);
            let loss = clipped_joint_loss(joint, old, advantages, 0.2)
                .into_scalar()
                .elem::<f32>();
            assert!((loss - expected).abs() < 1e-6);
        }
    }
    /// Give each fixture a distinct initial memory and intact valid temporal rows.
    fn lane_fixture(length: usize) -> RecurrentCombatSequence {
        RecurrentCombatSequence {
            continuous: RecurrentPpoSequence {
                observations: vec![vec![length as f32 / 4.0, -0.2]; length],
                global_states: vec![vec![0.0; 3]; length],
                pre_tanh_actions: vec![vec![0.0]; length],
                old_log_probabilities: vec![0.0; length],
                advantages: vec![1.0; length],
                returns: vec![1.0; length],
                value_index: 0,
                initial_memory: RecurrentMemory {
                    cell: vec![length as f32 / 8.0; 4],
                    hidden: vec![-(length as f32) / 8.0; 4],
                },
            },
            signals: vec![super::super::CombatSignal::SearchRequested; length],
        }
    }

    /// Padded recurrent lanes produce exactly their independent valid steps in both heads.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn unequal_lanes_keep_initial_memories_and_mask_padded_rows() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 4,
            critic_hidden_sizes: vec![4],
            ..RecurrentPpoConfig::default()
        };
        let agent =
            RecurrentCombatAgent::new(2, 3, 2, &[-1.0], &[1.0], config, SeedConfig::from_root(11))
                .expect("actor");
        let sequences = [1, 3].map(lane_fixture);
        let references = sequences.iter().collect::<Vec<_>>();
        let (mean, std, logits) = agent.actor_batch_distribution(&references);
        let actual = [
            tensor_vec(mean).expect("means"),
            tensor_vec(std).expect("scales"),
            tensor_vec(logits).expect("signals"),
        ];
        let mut expected: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        for sequence in &sequences {
            let observation = encode_matrix::<TrainingBackend>(
                &sequence.continuous.observations,
                2,
                &agent.device,
            )
            .reshape([1, sequence.continuous.observations.len(), 2]);
            let memory = memory_to_state(&sequence.continuous.initial_memory, &agent.device, 4);
            let distribution = agent.actor.forward(
                observation,
                Some(memory),
                1,
                agent.config.log_std_min,
                agent.config.log_std_max,
            );
            for (target, values) in expected.iter_mut().zip([
                tensor_vec(distribution.mean).expect("means"),
                tensor_vec(distribution.log_std).expect("scales"),
                tensor_vec(distribution.logits).expect("signals"),
            ]) {
                target.extend(values);
            }
        }
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.len(), expected.len());
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }
}
