//! Recurrent PPO with a decentralized LSTM actor and centralized critic.
//!
//! The actor API accepts only local observations and its own recurrent memory.
//! Global state is accepted by a separate training-only critic API, which keeps
//! centralized training data unreachable from decentralized execution.

mod checkpoint;

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use burn::module::AutodiffModule;
use burn::module::{Initializer, Module, Param};
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, LstmState, Relu};
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use burn::optim::adaptor::OptimizerAdaptor;
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::Tensor;

use super::backend::{inference_device, InferenceBackend, MODEL_INITIALIZATION_LOCK};
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use super::backend::{training_device, TrainingBackend, TrainingDevice};
use super::checkpoint::{policy_recorder, CheckpointError, CheckpointOperation};
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
use super::rng::SeedConfig;

/// Constant in the diagonal Gaussian log-density formula.
const LOG_TWO_PI: f32 = 1.837_877;

/// Environment-neutral recurrent PPO hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentPpoConfig {
    /// LSTM hidden-state width used as learned actor memory.
    pub actor_hidden_size: usize,

    /// Hidden widths used by the centralized critic MLP.
    pub critic_hidden_sizes: Vec<usize>,

    /// Discount factor used by rollout GAE construction.
    pub gamma: f32,

    /// Lambda used by rollout GAE construction.
    pub gae_lambda: f32,

    /// Adam learning rate for the actor and LSTM.
    pub actor_learning_rate: f64,

    /// Adam learning rate for the centralized critic.
    pub critic_learning_rate: f64,

    /// Numerical-stability epsilon used by both Adam optimizers.
    pub optimizer_epsilon: f32,

    /// Symmetric PPO ratio clipping distance around one.
    pub clip_ratio: f32,

    /// Gaussian entropy coefficient in the actor objective.
    pub entropy_coefficient: f32,

    /// PPO passes over each collected set of contiguous sequences.
    pub epochs: usize,

    /// Contiguous agent sequences accumulated into one optimizer step.
    pub minibatch_sequences: usize,

    /// Initial state-independent actor log standard deviation.
    pub initial_log_std: f32,

    /// Minimum emitted actor log standard deviation.
    pub log_std_min: f32,

    /// Maximum emitted actor log standard deviation.
    pub log_std_max: f32,

    /// Epsilon inside the tanh Jacobian logarithm.
    pub log_probability_epsilon: f32,

    /// Epsilon used while normalizing advantages.
    pub advantage_epsilon: f32,
}

impl Default for RecurrentPpoConfig {
    fn default() -> Self {
        Self {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![128, 64],
            gamma: 0.99,
            gae_lambda: 0.95,
            actor_learning_rate: 3e-4,
            critic_learning_rate: 3e-4,
            optimizer_epsilon: 1e-8,
            clip_ratio: 0.2,
            entropy_coefficient: 0.005,
            epochs: 4,
            minibatch_sequences: 8,
            initial_log_std: -1.0,
            log_std_min: -5.0,
            log_std_max: 1.0,
            log_probability_epsilon: 1e-6,
            advantage_epsilon: 1e-8,
        }
    }
}

impl RecurrentPpoConfig {
    /// Validate all runtime dimensions, action bounds, and hyperparameters.
    fn validate(
        &self,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
    ) -> Result<(), RecurrentPpoError> {
        self.validate_model_shape(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
        )?;
        self.validate_optimization()
    }

    /// Validate tensor dimensions, network widths, and action bounds.
    fn validate_model_shape(
        &self,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
    ) -> Result<(), RecurrentPpoError> {
        if observation_dim == 0 || global_state_dim == 0 || value_count == 0 {
            return Err(RecurrentPpoError::invalid_config(
                "dimensions",
                "observation, global state, and value count must be nonzero",
            ));
        }
        if action_low.is_empty() || action_low.len() != action_high.len() {
            return Err(RecurrentPpoError::invalid_config(
                "action_bounds",
                "equal nonempty lower and upper bounds are required",
            ));
        }
        for (dimension, (low, high)) in action_low.iter().zip(action_high).enumerate() {
            if !low.is_finite() || !high.is_finite() {
                return Err(RecurrentPpoError::NonFiniteValue {
                    field: "action bound",
                    dimension,
                });
            }
            if low >= high {
                return Err(RecurrentPpoError::invalid_config(
                    "action_bounds",
                    "every lower bound must be less than its upper bound",
                ));
            }
        }
        if self.actor_hidden_size == 0 || self.critic_hidden_sizes.contains(&0) {
            return Err(RecurrentPpoError::invalid_config(
                "hidden_sizes",
                "all hidden widths must be nonzero",
            ));
        }
        Ok(())
    }

    /// Validate optimizer, PPO, exploration, and normalization settings.
    fn validate_optimization(&self) -> Result<(), RecurrentPpoError> {
        if !self.gamma.is_finite() || !(0.0..=1.0).contains(&self.gamma) {
            return Err(RecurrentPpoError::invalid_config(
                "gamma",
                "must be finite and in 0..=1",
            ));
        }
        if !self.gae_lambda.is_finite() || !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err(RecurrentPpoError::invalid_config(
                "gae_lambda",
                "must be finite and in 0..=1",
            ));
        }
        if !self.actor_learning_rate.is_finite() || self.actor_learning_rate <= 0.0 {
            return Err(RecurrentPpoError::invalid_config(
                "actor_learning_rate",
                "must be finite and positive",
            ));
        }
        if !self.critic_learning_rate.is_finite() || self.critic_learning_rate <= 0.0 {
            return Err(RecurrentPpoError::invalid_config(
                "critic_learning_rate",
                "must be finite and positive",
            ));
        }
        if !self.optimizer_epsilon.is_finite() || self.optimizer_epsilon <= 0.0 {
            return Err(RecurrentPpoError::invalid_config(
                "optimizer_epsilon",
                "must be finite and positive",
            ));
        }
        if !self.clip_ratio.is_finite() || !(0.0..1.0).contains(&self.clip_ratio) {
            return Err(RecurrentPpoError::invalid_config(
                "clip_ratio",
                "must be finite and in 0..1",
            ));
        }
        if !self.entropy_coefficient.is_finite() || self.entropy_coefficient < 0.0 {
            return Err(RecurrentPpoError::invalid_config(
                "entropy_coefficient",
                "must be finite and non-negative",
            ));
        }
        if self.epochs == 0 {
            return Err(RecurrentPpoError::invalid_config(
                "epochs",
                "must be nonzero",
            ));
        }
        if self.minibatch_sequences == 0 {
            return Err(RecurrentPpoError::invalid_config(
                "minibatch_sequences",
                "must be nonzero",
            ));
        }
        if !self.log_std_min.is_finite()
            || !self.log_std_max.is_finite()
            || self.log_std_min >= self.log_std_max
        {
            return Err(RecurrentPpoError::invalid_config(
                "log_std",
                "finite minimum must be less than finite maximum",
            ));
        }
        if !self.initial_log_std.is_finite()
            || !(self.log_std_min..=self.log_std_max).contains(&self.initial_log_std)
        {
            return Err(RecurrentPpoError::invalid_config(
                "initial_log_std",
                "must be finite and within the configured log standard deviation bounds",
            ));
        }
        if !self.log_probability_epsilon.is_finite()
            || self.log_probability_epsilon <= 0.0
            || !self.advantage_epsilon.is_finite()
            || self.advantage_epsilon <= 0.0
        {
            return Err(RecurrentPpoError::invalid_config(
                "epsilon",
                "probability and advantage epsilons must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// LSTM actor emitting diagonal Gaussian parameters at every sequence step.
#[derive(Module, Debug)]
struct RecurrentActor<B: Backend> {
    /// Learned recurrent memory over local observation histories.
    memory: Lstm<B>,

    /// Observation-dependent action means.
    mean_head: Linear<B>,

    /// Learned state-independent exploration scale per action dimension.
    log_std: Param<Tensor<B, 1>>,
}

impl<B: Backend> RecurrentActor<B> {
    /// Initialize the local-observation actor.
    fn new(
        observation_dim: usize,
        hidden_size: usize,
        action_dim: usize,
        initial_log_std: f32,
        device: &B::Device,
    ) -> Self {
        // Begin with moderate exploration that cannot depend on observations.
        Self {
            memory: LstmConfig::new(observation_dim, hidden_size, true).init(device),
            mean_head: LinearConfig::new(hidden_size, action_dim).init(device),
            log_std: Initializer::Constant {
                value: f64::from(initial_log_std),
            }
            .init([action_dim], device),
        }
    }

    /// Copy one observation feature across every recurrent input gate.
    #[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
    fn copy_input_feature(&mut self, source_feature: usize, target_feature: usize) {
        let copy_weight = |weight: Param<Tensor<B, 2>>| {
            weight.map(|tensor| {
                let hidden_size = tensor.dims()[1];
                let source = tensor
                    .clone()
                    .slice([source_feature..source_feature + 1, 0..hidden_size]);
                tensor.slice_assign([target_feature..target_feature + 1, 0..hidden_size], source)
            })
        };

        self.memory.input_gate.input_transform.weight =
            copy_weight(self.memory.input_gate.input_transform.weight.clone());
        self.memory.forget_gate.input_transform.weight =
            copy_weight(self.memory.forget_gate.input_transform.weight.clone());
        self.memory.output_gate.input_transform.weight =
            copy_weight(self.memory.output_gate.input_transform.weight.clone());
        self.memory.cell_gate.input_transform.weight =
            copy_weight(self.memory.cell_gate.input_transform.weight.clone());
    }

    /// Insert one zero-weight observation row into every recurrent input gate.
    #[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
    fn insert_zero_input_feature(&mut self, feature: usize, device: &B::Device) {
        let insert_weight = |weight: Param<Tensor<B, 2>>| {
            weight.map(|tensor| {
                let [observation_dim, hidden_size] = tensor.dims();
                let mut expanded = Tensor::zeros([observation_dim + 1, hidden_size], device);
                if feature > 0 {
                    let prefix = tensor.clone().slice([0..feature, 0..hidden_size]);
                    expanded = expanded.slice_assign([0..feature, 0..hidden_size], prefix);
                }
                if feature < observation_dim {
                    let suffix = tensor.slice([feature..observation_dim, 0..hidden_size]);
                    expanded = expanded
                        .slice_assign([feature + 1..observation_dim + 1, 0..hidden_size], suffix);
                }
                expanded
            })
        };

        self.memory.input_gate.input_transform.weight =
            insert_weight(self.memory.input_gate.input_transform.weight.clone());
        self.memory.forget_gate.input_transform.weight =
            insert_weight(self.memory.forget_gate.input_transform.weight.clone());
        self.memory.output_gate.input_transform.weight =
            insert_weight(self.memory.output_gate.input_transform.weight.clone());
        self.memory.cell_gate.input_transform.weight =
            insert_weight(self.memory.cell_gate.input_transform.weight.clone());
    }

    /// Add one finite offset to one pre-tanh action-mean bias.
    #[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
    fn shift_mean_bias(&mut self, action_dimension: usize, delta: f32) {
        if let Some(bias) = self.mean_head.bias.take() {
            self.mean_head.bias = Some(bias.map(|tensor| {
                let shifted = tensor
                    .clone()
                    .slice(action_dimension..action_dimension + 1)
                    .add_scalar(delta);
                tensor.slice_assign(action_dimension..action_dimension + 1, shifted)
            }));
        }
    }

    /// Evaluate one batch-first contiguous sequence and return its final memory.
    fn forward(
        &self,
        observations: Tensor<B, 3>,
        initial_state: Option<LstmState<B, 2>>,
        action_dim: usize,
        log_std_min: f32,
        log_std_max: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, LstmState<B, 2>) {
        let (encoded, final_state) = self.memory.forward(observations, initial_state);
        let [batch_size, sequence_length, hidden_size] = encoded.dims();
        let mean = self
            .mean_head
            .forward(encoded.reshape([batch_size * sequence_length, hidden_size]));
        // Share exploration across observations while retaining one learned
        // scale for each continuous action dimension.
        let log_std = self
            .log_std
            .val()
            .reshape([1, action_dim])
            .repeat_dim(0, batch_size * sequence_length)
            .clamp(log_std_min, log_std_max);
        (mean, log_std, final_state)
    }
}

/// Runtime-shaped feed-forward centralized critic.
#[derive(Module, Debug)]
struct CentralCritic<B: Backend> {
    /// Ordered affine layers.
    layers: Vec<Linear<B>>,

    /// Hidden-layer activation.
    activation: Relu,
}

/// Single-record actor and critic checkpoint payload.
#[derive(Module, Debug)]
struct RecurrentNetworks<B: Backend> {
    /// Decentralized recurrent actor parameters.
    actor: RecurrentActor<B>,

    /// Centralized critic parameters.
    critic: CentralCritic<B>,
}

impl<B: Backend> CentralCritic<B> {
    /// Initialize one critic that emits a stable possible-agent value vector.
    fn new(
        global_state_dim: usize,
        hidden_sizes: &[usize],
        value_count: usize,
        device: &B::Device,
    ) -> Self {
        let widths: Vec<_> = std::iter::once(global_state_dim)
            .chain(hidden_sizes.iter().copied())
            .chain(std::iter::once(value_count))
            .collect();
        let layers = widths
            .windows(2)
            .map(|window| {
                let input = *window.first().expect("critic width window has input");
                let output = *window.get(1).expect("critic width window has output");
                LinearConfig::new(input, output).init(device)
            })
            .collect();
        Self {
            layers,
            activation: Relu::new(),
        }
    }

    /// Evaluate possible-agent values for a batch of global states.
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let last_layer = self.layers.len() - 1;
        let mut output = input;
        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.forward(output);
            if index < last_layer {
                output = self.activation.forward(output);
            }
        }
        output
    }
}

/// Host-owned LSTM state for one environment/species/agent trajectory.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentMemory {
    /// LSTM cell state.
    pub cell: Vec<f32>,

    /// LSTM hidden state.
    pub hidden: Vec<f32>,
}

impl RecurrentMemory {
    /// Construct a zero state for an episode boundary.
    #[must_use]
    pub fn zeros(hidden_size: usize) -> Self {
        Self {
            cell: vec![0.0; hidden_size],
            hidden: vec![0.0; hidden_size],
        }
    }
}

/// One sampled bounded action and the memory produced by its observation.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentAction {
    /// Action scaled into the configured environment bounds.
    pub action: Vec<f32>,

    /// Gaussian sample before tanh squashing.
    pub pre_tanh_action: Vec<f32>,

    /// Corrected behavior-policy log probability.
    pub log_probability: f32,

    /// Final LSTM state after consuming the current observation.
    pub next_memory: RecurrentMemory,
}

/// Deterministic independent stream used for policy action sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentSampler {
    /// `SplitMix64` stream state.
    state: u64,

    /// Cached second Box-Muller normal encoded as bits.
    cached_normal: Option<u64>,
}

impl RecurrentSampler {
    /// Construct a non-global action sampler.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    /// Draw one standard normal without touching backend or environment RNGs.
    fn normal(&mut self) -> f32 {
        if let Some(bits) = self.cached_normal.take() {
            return f64::from_bits(bits) as f32;
        }
        let first = self.unit_open();
        let second = self.unit_open();
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f64::consts::TAU * second;
        let (sin, cos) = angle.sin_cos();
        self.cached_normal = Some((radius * sin).to_bits());
        (radius * cos) as f32
    }

    /// Draw one value strictly between zero and one.
    fn unit_open(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^= value >> 31;
        let mantissa = (value >> 11).max(1);
        mantissa as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

/// One intact agent trajectory segment used by recurrent PPO updates.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentPpoSequence {
    /// Local observations in temporal order.
    pub observations: Vec<Vec<f32>>,

    /// Training-only global states in the same temporal order.
    pub global_states: Vec<Vec<f32>>,

    /// Gaussian samples before tanh squashing.
    pub pre_tanh_actions: Vec<Vec<f32>>,

    /// Behavior-policy corrected log probabilities.
    pub old_log_probabilities: Vec<f32>,

    /// Precomputed and normalized-later generalized advantages.
    pub advantages: Vec<f32>,

    /// Critic regression targets.
    pub returns: Vec<f32>,

    /// Stable centralized value-vector slot for this agent.
    pub value_index: usize,

    /// LSTM state immediately before the first observation.
    pub initial_memory: RecurrentMemory,
}

/// One expert observation and bounded environment action for actor pretraining.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentBehaviorSample {
    /// Encoded local observation; sequence cloning preserves its temporal position.
    pub observation: Vec<f32>,

    /// Expert action inside the configured environment bounds.
    pub action: Vec<f32>,
}

/// One bounded actor target paired with an explicit recurrent state.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentBehaviorMemorySample {
    /// Encoded local actor observation.
    pub observation: Vec<f32>,

    /// LSTM state immediately before the observation.
    pub initial_memory: RecurrentMemory,

    /// Target action inside the configured environment bounds.
    pub action: Vec<f32>,
}

impl RecurrentPpoSequence {
    /// Validate dimensions without permitting episode-boundary padding.
    #[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
    fn validate(
        &self,
        observation_dim: usize,
        global_state_dim: usize,
        action_dim: usize,
        hidden_size: usize,
        value_count: usize,
    ) -> Result<(), RecurrentPpoError> {
        let length = self.observations.len();
        if length == 0
            || self.global_states.len() != length
            || self.pre_tanh_actions.len() != length
            || self.old_log_probabilities.len() != length
            || self.advantages.len() != length
            || self.returns.len() != length
        {
            return Err(RecurrentPpoError::invalid_sequence(
                "all temporal fields must have the same nonzero length",
            ));
        }
        if self.value_index >= value_count {
            return Err(RecurrentPpoError::invalid_sequence(
                "value index must fit the centralized critic output",
            ));
        }
        validate_memory(&self.initial_memory, hidden_size)?;
        validate_matrix(&self.observations, observation_dim, "observation")?;
        validate_matrix(&self.global_states, global_state_dim, "global state")?;
        validate_matrix(&self.pre_tanh_actions, action_dim, "pre-tanh action")?;
        validate_finite(&self.old_log_probabilities, "old log probability")?;
        validate_finite(&self.advantages, "advantage")?;
        validate_finite(&self.returns, "return")?;
        Ok(())
    }
}

/// Metrics from one recurrent PPO update over contiguous sequences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecurrentPpoUpdate {
    /// Number of optimizer steps completed by each optimizer.
    pub optimizer_steps: u64,

    /// Number of minibatch optimizer updates completed by this call.
    pub optimizer_updates: u64,

    /// Number of unique valid rollout samples consumed by this call.
    pub valid_samples: u64,

    /// Mean clipped actor loss across valid recurrent timesteps.
    pub actor_loss: f64,

    /// Mean centralized value loss across valid recurrent timesteps.
    pub critic_loss: f64,

    /// Mean diagonal Gaussian entropy before tanh transformation.
    pub entropy: f64,

    /// Mean behavior-to-current log-probability difference.
    pub approximate_kl: f64,

    /// Actor learning rate used for this update.
    pub actor_learning_rate: f64,

    /// Critic learning rate used for this update.
    pub critic_learning_rate: f64,
}

/// Inference snapshot with a decentralized actor and separate critic method.
#[derive(Debug, Clone)]
pub struct RecurrentPpoPolicy {
    /// Inference-only LSTM actor.
    actor: RecurrentActor<InferenceBackend>,

    /// Inference-only centralized critic.
    critic: CentralCritic<InferenceBackend>,

    /// Encoded local observation width.
    observation_dim: usize,

    /// Encoded global state width.
    global_state_dim: usize,

    /// Stable possible-agent critic output count.
    value_count: usize,

    /// LSTM hidden width.
    hidden_size: usize,

    /// Inclusive action lower bounds.
    action_low: Vec<f32>,

    /// Inclusive action upper bounds.
    action_high: Vec<f32>,

    /// Minimum actor log standard deviation.
    log_std_min: f32,

    /// Maximum actor log standard deviation.
    log_std_max: f32,

    /// Tanh Jacobian epsilon.
    log_probability_epsilon: f32,
}

impl RecurrentPpoPolicy {
    /// Encode actor and critic parameters without filesystem access.
    ///
    /// # Errors
    /// Returns a checkpoint error when encoding fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RecurrentPpoError> {
        let networks = RecurrentNetworks {
            actor: self.actor.clone(),
            critic: self.critic.clone(),
        };
        NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .record(networks.into_record(), ())
            .map_err(|error| {
                RecurrentPpoError::Checkpoint(CheckpointError::new(
                    CheckpointOperation::Save,
                    "<memory>",
                    error.to_string(),
                ))
            })
    }

    /// Return a clean episode-boundary memory state.
    #[must_use]
    pub fn initial_memory(&self) -> RecurrentMemory {
        RecurrentMemory::zeros(self.hidden_size)
    }

    /// Sample one bounded action using only local observation and own memory.
    ///
    /// # Errors
    ///
    /// Returns a dimension, non-finite, or tensor conversion error.
    pub fn sample_action(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
        sampler: &mut RecurrentSampler,
    ) -> Result<RecurrentAction, RecurrentPpoError> {
        let (mean, log_std, next_memory) = self.distribution(observation, memory)?;
        let pre_tanh_action: Vec<_> = mean
            .iter()
            .zip(&log_std)
            .map(|(location, log_scale)| location + log_scale.exp() * sampler.normal())
            .collect();
        let action = scale_action(&pre_tanh_action, &self.action_low, &self.action_high);
        let log_probability = host_log_probability(
            &mean,
            &log_std,
            &pre_tanh_action,
            &self.action_low,
            &self.action_high,
            self.log_probability_epsilon,
        );
        Ok(RecurrentAction {
            action,
            pre_tanh_action,
            log_probability,
            next_memory,
        })
    }

    /// Return the deterministic mean action without sampling.
    ///
    /// # Errors
    ///
    /// Returns a dimension, non-finite, or tensor conversion error.
    pub fn mean_action(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
    ) -> Result<RecurrentAction, RecurrentPpoError> {
        let (mean, log_std, next_memory) = self.distribution(observation, memory)?;
        let action = scale_action(&mean, &self.action_low, &self.action_high);
        let log_probability = host_log_probability(
            &mean,
            &log_std,
            &mean,
            &self.action_low,
            &self.action_high,
            self.log_probability_epsilon,
        );
        Ok(RecurrentAction {
            action,
            pre_tanh_action: mean,
            log_probability,
            next_memory,
        })
    }

    /// Evaluate one focal agent's training-only centralized value.
    ///
    /// # Errors
    ///
    /// Returns a dimension, value-index, or tensor conversion error.
    pub fn value(
        &self,
        global_state: &[f32],
        value_index: usize,
    ) -> Result<f32, RecurrentPpoError> {
        validate_vector(global_state, self.global_state_dim, "global state")?;
        if value_index >= self.value_count {
            return Err(RecurrentPpoError::ValueIndex {
                index: value_index,
                count: self.value_count,
            });
        }
        let device = inference_device();
        let input = Tensor::<InferenceBackend, 1>::from_floats(global_state, &device)
            .reshape([1, self.global_state_dim]);
        let value = self.critic.forward(input).narrow(1, value_index, 1);
        Ok(value.into_scalar().elem::<f32>())
    }

    /// Save actor and critic parameters in one named `MessagePack` record.
    ///
    /// # Errors
    ///
    /// Returns a checkpoint error with save-path context.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf, RecurrentPpoError> {
        let path = path.as_ref().to_path_buf();
        RecurrentNetworks {
            actor: self.actor.clone(),
            critic: self.critic.clone(),
        }
        .save_file(path.clone(), &policy_recorder())
        .map_err(|error| {
            RecurrentPpoError::Checkpoint(CheckpointError::new(
                CheckpointOperation::Save,
                path.clone(),
                error.to_string(),
            ))
        })?;
        Ok(path)
    }

    /// Load actor and critic parameters into a matching inference architecture.
    ///
    /// Burn records store parameters rather than application architecture, so
    /// callers provide the exact runtime shapes and config used at creation.
    ///
    /// # Errors
    ///
    /// Returns a config or checkpoint error with load-path context.
    pub fn load(
        path: impl AsRef<Path>,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: &RecurrentPpoConfig,
    ) -> Result<Self, RecurrentPpoError> {
        config.validate(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
        )?;
        let path = path.as_ref().to_path_buf();
        let _initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = inference_device();
        let networks = RecurrentNetworks {
            actor: RecurrentActor::new(
                observation_dim,
                config.actor_hidden_size,
                action_low.len(),
                config.initial_log_std,
                &device,
            ),
            critic: CentralCritic::new(
                global_state_dim,
                &config.critic_hidden_sizes,
                value_count,
                &device,
            ),
        }
        .load_file(path.clone(), &policy_recorder(), &device)
        .map_err(|error| {
            RecurrentPpoError::Checkpoint(CheckpointError::new(
                CheckpointOperation::Load,
                path,
                error.to_string(),
            ))
        })?;
        Ok(Self {
            actor: networks.actor,
            critic: networks.critic,
            observation_dim,
            global_state_dim,
            value_count,
            hidden_size: config.actor_hidden_size,
            action_low: action_low.to_vec(),
            action_high: action_high.to_vec(),
            log_std_min: config.log_std_min,
            log_std_max: config.log_std_max,
            log_probability_epsilon: config.log_probability_epsilon,
        })
    }

    /// Load actor and critic parameters from named `MessagePack` bytes.
    ///
    /// This is the browser-compatible equivalent of [`Self::load`]. Burn
    /// records store parameters rather than application architecture, so the
    /// caller still supplies the exact runtime shapes and configuration.
    ///
    /// # Errors
    ///
    /// Returns a config or checkpoint error with an in-memory source marker.
    pub fn load_bytes(
        bytes: Vec<u8>,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: &RecurrentPpoConfig,
    ) -> Result<Self, RecurrentPpoError> {
        config.validate(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
        )?;
        let _initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = inference_device();
        let networks = RecurrentNetworks {
            actor: RecurrentActor::new(
                observation_dim,
                config.actor_hidden_size,
                action_low.len(),
                config.initial_log_std,
                &device,
            ),
            critic: CentralCritic::new(
                global_state_dim,
                &config.critic_hidden_sizes,
                value_count,
                &device,
            ),
        };
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let record = recorder.load(bytes, &device).map_err(|error| {
            RecurrentPpoError::Checkpoint(CheckpointError::new(
                CheckpointOperation::Load,
                "<memory>",
                error.to_string(),
            ))
        })?;
        checkpoint::validate(
            &record,
            observation_dim,
            global_state_dim,
            value_count,
            action_low.len(),
            config,
        )?;
        let networks = networks.load_record(record);
        Ok(Self {
            actor: networks.actor,
            critic: networks.critic,
            observation_dim,
            global_state_dim,
            value_count,
            hidden_size: config.actor_hidden_size,
            action_low: action_low.to_vec(),
            action_high: action_high.to_vec(),
            log_std_min: config.log_std_min,
            log_std_max: config.log_std_max,
            log_probability_epsilon: config.log_probability_epsilon,
        })
    }

    /// Evaluate Gaussian parameters and advance one local recurrent state.
    fn distribution(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
    ) -> Result<(Vec<f32>, Vec<f32>, RecurrentMemory), RecurrentPpoError> {
        validate_vector(observation, self.observation_dim, "observation")?;
        validate_memory(memory, self.hidden_size)?;
        let device = inference_device();
        let observations = Tensor::<InferenceBackend, 1>::from_floats(observation, &device)
            .reshape([1, 1, self.observation_dim]);
        let state = memory_to_state::<InferenceBackend>(memory, &device, self.hidden_size);
        let (mean, log_std, next_state) = self.actor.forward(
            observations,
            Some(state),
            self.action_low.len(),
            self.log_std_min,
            self.log_std_max,
        );
        Ok((
            tensor_vec(mean)?,
            tensor_vec(log_std)?,
            state_to_memory(next_state)?,
        ))
    }
}

/// Stateful recurrent PPO learner for one parameter-sharing group.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
pub struct RecurrentPpoAgent {
    /// Autodiff local actor.
    actor: RecurrentActor<TrainingBackend>,

    /// Autodiff centralized critic.
    critic: CentralCritic<TrainingBackend>,

    /// Actor Adam state.
    actor_optimizer: OptimizerAdaptor<Adam, RecurrentActor<TrainingBackend>, TrainingBackend>,

    /// Critic Adam state.
    critic_optimizer: OptimizerAdaptor<Adam, CentralCritic<TrainingBackend>, TrainingBackend>,

    /// Training backend device.
    device: TrainingDevice,

    /// Local observation width.
    observation_dim: usize,

    /// Global state width.
    global_state_dim: usize,

    /// Stable possible-agent value count.
    value_count: usize,

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
}

/// Small independent stream used only to shuffle intact recurrent chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
struct MinibatchRng {
    /// Current `SplitMix64` state.
    state: u64,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
impl MinibatchRng {
    /// Construct the stream from the caller's dedicated rollout seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Shuffle elements without changing any sequence's internal time order.
    fn shuffle<T>(&mut self, values: &mut [T]) {
        for upper in (1..values.len()).rev() {
            let index = (self.next_u64() as usize) % (upper + 1);
            values.swap(upper, index);
        }
    }

    /// Draw one `SplitMix64` value without touching model or action RNGs.
    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
impl fmt::Debug for RecurrentPpoAgent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RecurrentPpoAgent")
            .field("observation_dim", &self.observation_dim)
            .field("global_state_dim", &self.global_state_dim)
            .field("value_count", &self.value_count)
            .field("action_dim", &self.action_low.len())
            .field("optimizer_steps", &self.optimizer_steps)
            .field("minibatch_rng", &self.minibatch_rng)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
impl RecurrentPpoAgent {
    /// Initialize one species-specific actor and critic pair.
    ///
    /// # Errors
    ///
    /// Returns [`RecurrentPpoError::InvalidConfig`] for invalid shapes,
    /// bounds, or hyperparameters.
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
        let _initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = training_device();
        TrainingBackend::seed(&device, seeds.model);
        let actor = RecurrentActor::new(
            observation_dim,
            config.actor_hidden_size,
            action_low.len(),
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
        let (mean, log_std, _) = actor.forward(
            zero_local,
            None,
            action_low.len(),
            config.log_std_min,
            config.log_std_max,
        );
        let _actor_materialized = (mean + log_std).into_data();
        let _critic_materialized = critic
            .forward(Tensor::zeros([1, global_state_dim], &device))
            .into_data();

        let actor_optimizer = AdamConfig::new()
            .with_epsilon(config.optimizer_epsilon)
            .init::<TrainingBackend, RecurrentActor<TrainingBackend>>();
        let critic_optimizer = AdamConfig::new()
            .with_epsilon(config.optimizer_epsilon)
            .init::<TrainingBackend, CentralCritic<TrainingBackend>>();
        Ok(Self {
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            device,
            observation_dim,
            global_state_dim,
            value_count,
            action_low: action_low.to_vec(),
            action_high: action_high.to_vec(),
            config,
            optimizer_steps: 0,
            minibatch_rng: MinibatchRng::new(seeds.rollout),
        })
    }

    /// Snapshot actor and critic parameters for collection and evaluation.
    #[must_use]
    pub fn policy(&self) -> RecurrentPpoPolicy {
        RecurrentPpoPolicy {
            actor: self.actor.valid(),
            critic: self.critic.valid(),
            observation_dim: self.observation_dim,
            global_state_dim: self.global_state_dim,
            value_count: self.value_count,
            hidden_size: self.config.actor_hidden_size,
            action_low: self.action_low.clone(),
            action_high: self.action_high.clone(),
            log_std_min: self.config.log_std_min,
            log_std_max: self.config.log_std_max,
            log_probability_epsilon: self.config.log_probability_epsilon,
        }
    }

    /// Load a complete actor/critic record for curriculum continuation.
    ///
    /// Optimizer state starts fresh for the new lesson while all learned
    /// network parameters transfer directly.
    ///
    /// # Errors
    ///
    /// Returns a configuration or checkpoint error.
    pub fn load(
        path: impl AsRef<Path>,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: RecurrentPpoConfig,
        seeds: SeedConfig,
    ) -> Result<Self, RecurrentPpoError> {
        let path = path.as_ref().to_path_buf();
        let mut agent = Self::new(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
            config,
            seeds,
        )?;
        let networks = agent.load_networks(path)?;
        agent.actor = networks.actor;
        agent.critic = networks.critic;
        Ok(agent)
    }

    /// Load validated actor and critic bytes with fresh optimizer and rollout RNG state.
    ///
    /// Every parameter transfers unchanged. This starts a new optimization run;
    /// it does not restore Adam moments, optimizer counters, or episode memory.
    /// Import takes O(parameters) time and space and consumes the supplied bytes.
    /// Burn 0.21 wraps the validated inference modules through `from_inner`:
    /// <https://docs.rs/burn/0.21.0/burn/module/trait.AutodiffModule.html>.
    ///
    /// # Errors
    ///
    /// Returns a configuration or checkpoint error before constructing a usable learner
    /// when bytes are corrupt, parameter shapes differ, or parameters are nonfinite.
    pub fn load_bytes(
        bytes: Vec<u8>,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: RecurrentPpoConfig,
        seeds: SeedConfig,
    ) -> Result<Self, RecurrentPpoError> {
        // Validate the complete inference record before converting any parameter to autodiff.
        let policy = RecurrentPpoPolicy::load_bytes(
            bytes,
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
            &config,
        )?;
        // New Adam instances and the supplied RNG seed define a fresh optimization run.
        let mut agent = Self::new(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
            config,
            seeds,
        )?;
        // Loading into initialized training modules retains their enabled gradient flags.
        agent.actor = agent
            .actor
            .load_record(RecurrentActor::from_inner(policy.actor).into_record());
        agent.critic = agent
            .critic
            .load_record(CentralCritic::from_inner(policy.critic).into_record());
        Ok(agent)
    }

    /// Load a recurrent actor while retaining a newly initialized critic.
    ///
    /// This transfer mode preserves decentralized behavior across curriculum
    /// stages and starts value learning from the destination stage's seed.
    /// Both optimizer states also start fresh.
    ///
    /// # Errors
    ///
    /// Returns a configuration or checkpoint error.
    pub fn load_actor_with_fresh_critic(
        path: impl AsRef<Path>,
        observation_dim: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: RecurrentPpoConfig,
        seeds: SeedConfig,
    ) -> Result<Self, RecurrentPpoError> {
        let path = path.as_ref().to_path_buf();
        let mut agent = Self::new(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
            config,
            seeds,
        )?;
        let networks = agent.load_networks(path)?;
        agent.actor = networks.actor;
        Ok(agent)
    }

    /// Load a legacy actor after inserting one zero-weight observation feature.
    ///
    /// Every existing input row keeps its relative order. The inserted feature
    /// cannot affect the migrated actor until later training changes its zero
    /// weights. The destination critic and both optimizer states start fresh.
    ///
    /// # Errors
    ///
    /// Returns a configuration or checkpoint error if the source shape, insert
    /// position, destination shape, or saved record is invalid.
    #[expect(
        clippy::too_many_arguments,
        reason = "checkpoint migration requires explicit source and destination contracts"
    )]
    pub fn load_actor_with_inserted_input_feature_and_fresh_critic(
        path: impl AsRef<Path>,
        source_observation_dim: usize,
        inserted_feature: usize,
        global_state_dim: usize,
        value_count: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: RecurrentPpoConfig,
        seeds: SeedConfig,
    ) -> Result<Self, RecurrentPpoError> {
        let observation_dim = source_observation_dim.checked_add(1).ok_or_else(|| {
            RecurrentPpoError::invalid_config(
                "source_observation_dim",
                "source width must allow one inserted feature",
            )
        })?;
        if inserted_feature > source_observation_dim {
            return Err(RecurrentPpoError::invalid_config(
                "inserted_feature",
                "insert position must be at or before the source width",
            ));
        }

        let path = path.as_ref().to_path_buf();
        let mut agent = Self::new(
            observation_dim,
            global_state_dim,
            value_count,
            action_low,
            action_high,
            config,
            seeds,
        )?;
        // Migration constructs a temporary actor and must not consume the
        // backend RNG while another agent initializes from its model seed.
        let _initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let networks = RecurrentNetworks {
            actor: RecurrentActor::new(
                source_observation_dim,
                agent.config.actor_hidden_size,
                action_low.len(),
                agent.config.initial_log_std,
                &agent.device,
            ),
            critic: agent.critic.clone(),
        }
        .load_file(path.clone(), &policy_recorder(), &agent.device)
        .map_err(|error| {
            RecurrentPpoError::Checkpoint(CheckpointError::new(
                CheckpointOperation::Load,
                path,
                error.to_string(),
            ))
        })?;
        agent.actor = networks.actor;
        agent
            .actor
            .insert_zero_input_feature(inserted_feature, &agent.device);
        Ok(agent)
    }

    /// Copy one actor observation feature's LSTM input weights to another feature.
    ///
    /// This preserves learned behavior when two typed input channels represent
    /// the same curriculum context before and after a stage transition. All
    /// recurrent, output-head, exploration, and critic parameters remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns a configuration error if either feature index is outside the
    /// actor observation width.
    pub fn copy_actor_input_feature(
        &mut self,
        source_feature: usize,
        target_feature: usize,
    ) -> Result<(), RecurrentPpoError> {
        if source_feature >= self.observation_dim || target_feature >= self.observation_dim {
            return Err(RecurrentPpoError::invalid_config(
                "actor_input_feature",
                "source and target must be inside the observation width",
            ));
        }

        // Copy the input row for every LSTM gate so the target channel has the
        // exact same recurrent effect as the learned source channel.
        self.actor
            .copy_input_feature(source_feature, target_feature);
        Ok(())
    }

    /// Add a finite offset to one actor action-mean bias.
    ///
    /// The offset applies before `tanh` action bounding. Recurrent memory,
    /// exploration scale, every other action dimension, and the critic remain
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns a configuration error for an invalid action dimension or a
    /// non-finite offset.
    pub fn shift_actor_mean_bias(
        &mut self,
        action_dimension: usize,
        delta: f32,
    ) -> Result<(), RecurrentPpoError> {
        if action_dimension >= self.action_low.len() || !delta.is_finite() {
            return Err(RecurrentPpoError::invalid_config(
                "actor_mean_bias",
                "action dimension must be valid and offset must be finite",
            ));
        }
        self.actor.shift_mean_bias(action_dimension, delta);
        Ok(())
    }

    /// Read one complete network record into this agent's architecture.
    fn load_networks(
        &self,
        path: PathBuf,
    ) -> Result<RecurrentNetworks<TrainingBackend>, RecurrentPpoError> {
        RecurrentNetworks {
            actor: self.actor.clone(),
            critic: self.critic.clone(),
        }
        .load_file(path.clone(), &policy_recorder(), &self.device)
        .map_err(|error| {
            RecurrentPpoError::Checkpoint(CheckpointError::new(
                CheckpointOperation::Load,
                path,
                error.to_string(),
            ))
        })
    }

    /// Fit the deterministic actor mean to one batch of expert actions.
    ///
    /// Each sample starts with zero recurrent memory. This makes the method fit
    /// Markov demonstrations without exposing the training-only critic.
    ///
    /// # Errors
    ///
    /// Returns a dimension, non-finite, action-bound, or empty-batch error.
    pub fn behavior_clone(
        &mut self,
        samples: &[RecurrentBehaviorSample],
    ) -> Result<f64, RecurrentPpoError> {
        self.validate_behavior_samples(samples)?;

        let observations = samples
            .iter()
            .map(|sample| sample.observation.clone())
            .collect::<Vec<_>>();
        let target_actions = samples
            .iter()
            .map(|sample| normalize_action(&sample.action, &self.action_low, &self.action_high))
            .collect::<Vec<_>>();
        let observations =
            encode_matrix::<TrainingBackend>(&observations, self.observation_dim, &self.device)
                .reshape([samples.len(), 1, self.observation_dim]);
        let targets =
            encode_matrix::<TrainingBackend>(&target_actions, self.action_low.len(), &self.device);
        let (mean, _, _) = self.actor.forward(
            observations,
            None,
            self.action_low.len(),
            self.config.log_std_min,
            self.config.log_std_max,
        );
        let residual = mean.tanh() - targets;
        let loss = residual.clone().mul(residual).mean();
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let gradients = GradientsParams::from_grads(loss.backward(), &self.actor);
        self.actor = self.actor_optimizer.step(
            self.config.actor_learning_rate,
            self.actor.clone(),
            gradients,
        );
        Ok(loss_value)
    }

    /// Fit one contiguous demonstration, propagating gradients through its history.
    ///
    /// Samples must stay within one episode. Initial memory is detached from any
    /// earlier computation; subsequent states remain connected within this call.
    /// The returned loss is mean squared error over time and normalized action
    /// dimensions. This updates only the actor and borrows the initial memory.
    /// Activation storage grows with the number of samples.
    ///
    /// # Errors
    ///
    /// Returns the existing sample validation errors before validating memory.
    /// Invalid input cannot change the actor or its optimizer.
    pub fn behavior_clone_sequence(
        &mut self,
        samples: &[RecurrentBehaviorSample],
        initial_memory: &RecurrentMemory,
    ) -> Result<f64, RecurrentPpoError> {
        self.validate_behavior_samples(samples)?;
        validate_memory(initial_memory, self.config.actor_hidden_size)?;
        // Burn 0.21.0 Lstm uses [batch, time, feature] with batch_first enabled.
        // Preserve one temporal lane so later losses reach earlier observations.
        let observations = samples
            .iter()
            .map(|sample| sample.observation.as_slice())
            .collect::<Vec<_>>();
        let targets = samples
            .iter()
            .map(|sample| normalize_action(&sample.action, &self.action_low, &self.action_high))
            .collect::<Vec<_>>();
        let observations =
            encode_matrix::<TrainingBackend>(&observations, self.observation_dim, &self.device)
                .reshape([1, samples.len(), self.observation_dim]);
        let targets =
            encode_matrix::<TrainingBackend>(&targets, self.action_low.len(), &self.device);
        let state = memory_to_state(initial_memory, &self.device, self.config.actor_hidden_size);
        let (mean, _, _) = self.actor.forward(
            observations,
            Some(state),
            self.action_low.len(),
            self.config.log_std_min,
            self.config.log_std_max,
        );
        // Normalize action units exactly as the independent-sample cloning methods do.
        let residual = mean.tanh() - targets;
        let loss = residual.clone().mul(residual).mean();
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let gradients = GradientsParams::from_grads(loss.backward(), &self.actor);
        self.actor = self.actor_optimizer.step(
            self.config.actor_learning_rate,
            self.actor.clone(),
            gradients,
        );
        Ok(loss_value)
    }

    /// Fit bounded actor targets at explicit recurrent states.
    ///
    /// Each sample is one step, so callers can regularize recurrent behavior
    /// without backpropagating through the actions that produced its memory.
    ///
    /// # Errors
    ///
    /// Returns a dimension, non-finite, action-bound, memory, or empty-batch error.
    pub fn behavior_clone_with_memory(
        &mut self,
        samples: &[RecurrentBehaviorMemorySample],
    ) -> Result<f64, RecurrentPpoError> {
        self.validate_memory_behavior_samples(samples)?;

        let observations = samples
            .iter()
            .map(|sample| sample.observation.clone())
            .collect::<Vec<_>>();
        let target_actions = samples
            .iter()
            .map(|sample| normalize_action(&sample.action, &self.action_low, &self.action_high))
            .collect::<Vec<_>>();
        let observations =
            encode_matrix::<TrainingBackend>(&observations, self.observation_dim, &self.device)
                .reshape([samples.len(), 1, self.observation_dim]);
        let memories = samples
            .iter()
            .map(|sample| sample.initial_memory.clone())
            .collect::<Vec<_>>();
        let state = memories_to_state::<TrainingBackend>(
            &memories,
            &self.device,
            self.config.actor_hidden_size,
        );
        let targets =
            encode_matrix::<TrainingBackend>(&target_actions, self.action_low.len(), &self.device);
        let (mean, _, _) = self.actor.forward(
            observations,
            Some(state),
            self.action_low.len(),
            self.config.log_std_min,
            self.config.log_std_max,
        );
        let residual = mean.tanh() - targets;
        let loss = residual.clone().mul(residual).mean();
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let gradients = GradientsParams::from_grads(loss.backward(), &self.actor);
        self.actor = self.actor_optimizer.step(
            self.config.actor_learning_rate,
            self.actor.clone(),
            gradients,
        );
        Ok(loss_value)
    }

    /// Validate one Markov behavior-cloning batch at the public boundary.
    fn validate_behavior_samples(
        &self,
        samples: &[RecurrentBehaviorSample],
    ) -> Result<(), RecurrentPpoError> {
        if samples.is_empty() {
            return Err(RecurrentPpoError::invalid_sequence(
                "at least one behavior sample is required",
            ));
        }
        for sample in samples {
            validate_vector(&sample.observation, self.observation_dim, "observation")?;
            validate_vector(&sample.action, self.action_low.len(), "action")?;
            if sample
                .action
                .iter()
                .zip(self.action_low.iter().zip(&self.action_high))
                .any(|(action, (low, high))| action < low || action > high)
            {
                return Err(RecurrentPpoError::invalid_sequence(
                    "behavior actions must remain inside the configured bounds",
                ));
            }
        }
        Ok(())
    }

    /// Validate one explicit-memory behavior-cloning batch.
    fn validate_memory_behavior_samples(
        &self,
        samples: &[RecurrentBehaviorMemorySample],
    ) -> Result<(), RecurrentPpoError> {
        if samples.is_empty() {
            return Err(RecurrentPpoError::invalid_sequence(
                "at least one explicit-memory behavior sample is required",
            ));
        }
        for sample in samples {
            validate_vector(&sample.observation, self.observation_dim, "observation")?;
            validate_vector(&sample.action, self.action_low.len(), "action")?;
            validate_memory(&sample.initial_memory, self.config.actor_hidden_size)?;
            if sample
                .action
                .iter()
                .zip(self.action_low.iter().zip(&self.action_high))
                .any(|(action, (low, high))| action < low || action > high)
            {
                return Err(RecurrentPpoError::invalid_sequence(
                    "explicit-memory behavior actions must remain inside the configured bounds",
                ));
            }
        }
        Ok(())
    }

    /// Apply clipped PPO actor updates and centralized critic regression.
    ///
    /// Every input sequence must be contiguous and must not cross an episode
    /// boundary. The stored initial memory reconstructs the behavior-policy
    /// recurrent context for every optimization epoch.
    ///
    /// # Errors
    ///
    /// Returns a sequence validation or tensor conversion error.
    pub fn update(
        &mut self,
        sequences: &[RecurrentPpoSequence],
    ) -> Result<RecurrentPpoUpdate, RecurrentPpoError> {
        if sequences.is_empty() {
            return Err(RecurrentPpoError::invalid_sequence(
                "at least one sequence is required",
            ));
        }
        for sequence in sequences {
            sequence.validate(
                self.observation_dim,
                self.global_state_dim,
                self.action_low.len(),
                self.config.actor_hidden_size,
                self.value_count,
            )?;
        }

        let (advantage_mean, advantage_scale) =
            advantage_normalization(sequences, self.config.advantage_epsilon);
        let valid_samples = sequences.iter().fold(0_u64, |total, sequence| {
            total.saturating_add(sequence.observations.len() as u64)
        });
        let mut totals = UpdateTotals::default();
        let mut optimizer_updates = 0_u64;
        let mut minibatch_order = sequences.iter().collect::<Vec<_>>();
        for _ in 0..self.config.epochs {
            // Preserve time inside each recurrent chunk while removing the
            // systematic last-update bias of trajectory-ordered chunks.
            self.minibatch_rng.shuffle(&mut minibatch_order);
            for minibatch in minibatch_order.chunks(self.config.minibatch_sequences) {
                let mut actor_loss = None;
                let mut critic_loss = None;
                let mut minibatch_valid_timesteps = 0_usize;
                for sequence in minibatch {
                    let (sequence_actor_loss, actor_metrics) =
                        self.actor_loss(sequence, advantage_mean, advantage_scale);
                    let (sequence_critic_loss, critic_loss_value) = self.critic_loss(sequence);
                    let valid_timesteps = sequence.observations.len();
                    let tensor_weight = valid_timesteps as f32;
                    let metric_weight = valid_timesteps as f64;
                    actor_loss = Some(match actor_loss {
                        Some(total) => total + sequence_actor_loss * tensor_weight,
                        None => sequence_actor_loss * tensor_weight,
                    });
                    critic_loss = Some(match critic_loss {
                        Some(total) => total + sequence_critic_loss * tensor_weight,
                        None => sequence_critic_loss * tensor_weight,
                    });
                    totals.actor_loss =
                        actor_metrics.loss.mul_add(metric_weight, totals.actor_loss);
                    totals.critic_loss =
                        critic_loss_value.mul_add(metric_weight, totals.critic_loss);
                    totals.entropy = actor_metrics.entropy.mul_add(metric_weight, totals.entropy);
                    totals.approximate_kl = actor_metrics
                        .approximate_kl
                        .mul_add(metric_weight, totals.approximate_kl);
                    totals.valid_timesteps = totals
                        .valid_timesteps
                        .saturating_add(valid_timesteps as u64);
                    minibatch_valid_timesteps =
                        minibatch_valid_timesteps.saturating_add(valid_timesteps);
                }
                let Some(actor_loss) = actor_loss else {
                    return Err(RecurrentPpoError::invalid_sequence(
                        "optimizer minibatches must be nonempty",
                    ));
                };
                let Some(critic_loss) = critic_loss else {
                    return Err(RecurrentPpoError::invalid_sequence(
                        "optimizer minibatches must be nonempty",
                    ));
                };
                // Convert per-sequence means back into one valid-timestep mean
                // so short terminal chunks cannot dominate full unrolls.
                let divisor = minibatch_valid_timesteps as f32;
                let actor_loss = actor_loss / divisor;
                let critic_loss = critic_loss / divisor;
                let actor_gradients =
                    GradientsParams::from_grads(actor_loss.backward(), &self.actor);
                self.actor = self.actor_optimizer.step(
                    self.config.actor_learning_rate,
                    self.actor.clone(),
                    actor_gradients,
                );
                let critic_gradients =
                    GradientsParams::from_grads(critic_loss.backward(), &self.critic);
                self.critic = self.critic_optimizer.step(
                    self.config.critic_learning_rate,
                    self.critic.clone(),
                    critic_gradients,
                );
                self.optimizer_steps = self.optimizer_steps.saturating_add(1);
                optimizer_updates = optimizer_updates.saturating_add(1);
            }
        }
        let divisor = totals.valid_timesteps as f64;
        Ok(RecurrentPpoUpdate {
            optimizer_steps: self.optimizer_steps,
            optimizer_updates,
            valid_samples,
            actor_loss: totals.actor_loss / divisor,
            critic_loss: totals.critic_loss / divisor,
            entropy: totals.entropy / divisor,
            approximate_kl: totals.approximate_kl / divisor,
            actor_learning_rate: self.config.actor_learning_rate,
            critic_learning_rate: self.config.critic_learning_rate,
        })
    }

    /// Regress the centralized critic while leaving the actor unchanged.
    ///
    /// This supports bounded value warmup after actor-only curriculum transfer.
    /// Every sequence must remain contiguous and inside one episode.
    ///
    /// # Errors
    ///
    /// Returns a sequence validation or tensor conversion error.
    pub fn update_critic(
        &mut self,
        sequences: &[RecurrentPpoSequence],
    ) -> Result<RecurrentPpoUpdate, RecurrentPpoError> {
        if sequences.is_empty() {
            return Err(RecurrentPpoError::invalid_sequence(
                "at least one sequence is required",
            ));
        }
        for sequence in sequences {
            sequence.validate(
                self.observation_dim,
                self.global_state_dim,
                self.action_low.len(),
                self.config.actor_hidden_size,
                self.value_count,
            )?;
        }

        let valid_samples = sequences.iter().fold(0_u64, |total, sequence| {
            total.saturating_add(sequence.observations.len() as u64)
        });
        let mut critic_loss_total = 0.0;
        let mut valid_timestep_total = 0_u64;
        let mut optimizer_updates = 0_u64;
        let mut minibatch_order = sequences.iter().collect::<Vec<_>>();
        for _ in 0..self.config.epochs {
            self.minibatch_rng.shuffle(&mut minibatch_order);
            for minibatch in minibatch_order.chunks(self.config.minibatch_sequences) {
                let mut critic_loss = None;
                let mut minibatch_valid_timesteps = 0_usize;
                for sequence in minibatch {
                    let (sequence_loss, sequence_loss_value) = self.critic_loss(sequence);
                    let valid_timesteps = sequence.observations.len();
                    let tensor_weight = valid_timesteps as f32;
                    let metric_weight = valid_timesteps as f64;
                    critic_loss = Some(match critic_loss {
                        Some(total) => total + sequence_loss * tensor_weight,
                        None => sequence_loss * tensor_weight,
                    });
                    critic_loss_total =
                        sequence_loss_value.mul_add(metric_weight, critic_loss_total);
                    valid_timestep_total =
                        valid_timestep_total.saturating_add(valid_timesteps as u64);
                    minibatch_valid_timesteps =
                        minibatch_valid_timesteps.saturating_add(valid_timesteps);
                }
                let Some(critic_loss) = critic_loss else {
                    return Err(RecurrentPpoError::invalid_sequence(
                        "optimizer minibatches must be nonempty",
                    ));
                };
                let critic_loss = critic_loss / minibatch_valid_timesteps as f32;
                let gradients = GradientsParams::from_grads(critic_loss.backward(), &self.critic);
                self.critic = self.critic_optimizer.step(
                    self.config.critic_learning_rate,
                    self.critic.clone(),
                    gradients,
                );
                optimizer_updates = optimizer_updates.saturating_add(1);
            }
        }
        let divisor = valid_timestep_total as f64;
        Ok(RecurrentPpoUpdate {
            optimizer_steps: self.optimizer_steps,
            optimizer_updates,
            valid_samples,
            actor_loss: 0.0,
            critic_loss: critic_loss_total / divisor,
            entropy: 0.0,
            approximate_kl: 0.0,
            actor_learning_rate: self.config.actor_learning_rate,
            critic_learning_rate: self.config.critic_learning_rate,
        })
    }

    /// Build one intact sequence's differentiable actor objective.
    fn actor_loss(
        &self,
        sequence: &RecurrentPpoSequence,
        advantage_mean: f32,
        advantage_scale: f32,
    ) -> (Tensor<TrainingBackend, 1>, ActorMetrics) {
        let sequence_length = sequence.observations.len();
        let observations = encode_matrix::<TrainingBackend>(
            &sequence.observations,
            self.observation_dim,
            &self.device,
        )
        .reshape([1, sequence_length, self.observation_dim]);
        let state = memory_to_state::<TrainingBackend>(
            &sequence.initial_memory,
            &self.device,
            self.config.actor_hidden_size,
        );
        let (mean, log_std, _) = self.actor.forward(
            observations,
            Some(state),
            self.action_low.len(),
            self.config.log_std_min,
            self.config.log_std_max,
        );
        let samples = encode_matrix::<TrainingBackend>(
            &sequence.pre_tanh_actions,
            self.action_low.len(),
            &self.device,
        );
        let new_log_probabilities = tensor_log_probabilities(
            mean,
            log_std.clone(),
            samples,
            &self.action_low,
            &self.action_high,
            self.config.log_probability_epsilon,
            &self.device,
        );
        let old_log_probabilities = Tensor::<TrainingBackend, 1>::from_floats(
            sequence.old_log_probabilities.as_slice(),
            &self.device,
        );
        let normalized_advantages: Vec<_> = sequence
            .advantages
            .iter()
            .map(|value| (value - advantage_mean) / advantage_scale)
            .collect();
        let advantages = Tensor::<TrainingBackend, 1>::from_floats(
            normalized_advantages.as_slice(),
            &self.device,
        );
        let ratios = (new_log_probabilities.clone() - old_log_probabilities.clone()).exp();
        let unclipped = ratios.clone() * advantages.clone();
        let clipped =
            ratios.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages;
        let policy_loss = unclipped.min_pair(clipped).mean().neg();
        let entropy = (log_std + (0.5 * (1.0 + LOG_TWO_PI))).sum_dim(1).mean();
        let loss = policy_loss - entropy.clone() * self.config.entropy_coefficient;
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let entropy_value = entropy.into_scalar().elem::<f64>();
        let approximate_kl = (old_log_probabilities - new_log_probabilities)
            .mean()
            .into_scalar()
            .elem::<f64>();
        (
            loss,
            ActorMetrics {
                loss: loss_value,
                entropy: entropy_value,
                approximate_kl,
            },
        )
    }

    /// Build one sequence's differentiable centralized critic objective.
    fn critic_loss(&self, sequence: &RecurrentPpoSequence) -> (Tensor<TrainingBackend, 1>, f64) {
        let states = encode_matrix::<TrainingBackend>(
            &sequence.global_states,
            self.global_state_dim,
            &self.device,
        );
        let values = self
            .critic
            .forward(states)
            .narrow(1, sequence.value_index, 1)
            .squeeze_dim::<1>(1);
        let returns =
            Tensor::<TrainingBackend, 1>::from_floats(sequence.returns.as_slice(), &self.device);
        let residual = values - returns;
        let loss = residual.clone().mul(residual).mean() * 0.5;
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        (loss, loss_value)
    }
}

/// Accumulated update diagnostics before averaging.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
struct UpdateTotals {
    /// Actor loss sum weighted by valid recurrent timesteps.
    actor_loss: f64,

    /// Critic loss sum weighted by valid recurrent timesteps.
    critic_loss: f64,

    /// Entropy sum weighted by valid recurrent timesteps.
    entropy: f64,

    /// Approximate KL sum weighted by valid recurrent timesteps.
    approximate_kl: f64,

    /// Number of valid recurrent timesteps evaluated across all epochs.
    valid_timesteps: u64,
}

/// Diagnostics returned by one actor sequence update.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
struct ActorMetrics {
    /// Combined actor loss.
    loss: f64,

    /// Pre-tanh diagonal Gaussian entropy.
    entropy: f64,

    /// Behavior-to-current log-probability difference.
    approximate_kl: f64,
}

/// Recurrent PPO configuration, sequence, or tensor failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecurrentPpoError {
    /// One runtime setting violates its invariant.
    InvalidConfig {
        /// Setting name.
        field: &'static str,

        /// Static constraint description.
        reason: &'static str,
    },

    /// One sequence violates contiguity-shape requirements.
    InvalidSequence(&'static str),

    /// One vector has an unexpected width.
    DimensionMismatch {
        /// Vector role.
        field: &'static str,

        /// Expected scalar count.
        expected: usize,

        /// Received scalar count.
        actual: usize,
    },

    /// One scalar is NaN or infinite.
    NonFiniteValue {
        /// Vector role.
        field: &'static str,

        /// Invalid scalar position.
        dimension: usize,
    },

    /// Centralized critic slot is outside its fixed output.
    ValueIndex {
        /// Invalid requested slot.
        index: usize,

        /// Stable output count.
        count: usize,
    },

    /// Burn tensor data could not convert to host values.
    TensorConversion(String),

    /// Actor and critic checkpoint recording failed.
    Checkpoint(CheckpointError),
}

impl RecurrentPpoError {
    /// Construct a static configuration error.
    const fn invalid_config(field: &'static str, reason: &'static str) -> Self {
        Self::InvalidConfig { field, reason }
    }

    /// Construct a static sequence error.
    #[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
    const fn invalid_sequence(reason: &'static str) -> Self {
        Self::InvalidSequence(reason)
    }
}

impl fmt::Display for RecurrentPpoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(
                    formatter,
                    "invalid recurrent PPO config `{field}`: {reason}"
                )
            }
            Self::InvalidSequence(reason) => {
                write!(formatter, "invalid recurrent PPO sequence: {reason}")
            }
            Self::DimensionMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} width mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteValue { field, dimension } => {
                write!(formatter, "{field} scalar {dimension} is not finite")
            }
            Self::ValueIndex { index, count } => {
                write!(
                    formatter,
                    "critic value index {index} is outside 0..{count}"
                )
            }
            Self::TensorConversion(message) => {
                write!(
                    formatter,
                    "recurrent PPO tensor conversion failed: {message}"
                )
            }
            Self::Checkpoint(error) => error.fmt(formatter),
        }
    }
}

impl Error for RecurrentPpoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            _ => None,
        }
    }
}

/// Convert a host recurrent state into a one-lane backend state.
fn memory_to_state<B: Backend>(
    memory: &RecurrentMemory,
    device: &B::Device,
    hidden_size: usize,
) -> LstmState<B, 2> {
    let cell =
        Tensor::<B, 1>::from_floats(memory.cell.as_slice(), device).reshape([1, hidden_size]);
    let hidden =
        Tensor::<B, 1>::from_floats(memory.hidden.as_slice(), device).reshape([1, hidden_size]);
    LstmState::new(cell, hidden)
}

/// Convert host recurrent states into one batched backend state.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
fn memories_to_state<B: Backend>(
    memories: &[RecurrentMemory],
    device: &B::Device,
    hidden_size: usize,
) -> LstmState<B, 2> {
    let cells = memories
        .iter()
        .map(|memory| memory.cell.clone())
        .collect::<Vec<_>>();
    let hidden = memories
        .iter()
        .map(|memory| memory.hidden.clone())
        .collect::<Vec<_>>();
    LstmState::new(
        encode_matrix::<B>(&cells, hidden_size, device),
        encode_matrix::<B>(&hidden, hidden_size, device),
    )
}

/// Convert a one-lane backend state into detached host memory.
fn state_to_memory<B: Backend>(
    state: LstmState<B, 2>,
) -> Result<RecurrentMemory, RecurrentPpoError> {
    Ok(RecurrentMemory {
        cell: tensor_vec(state.cell)?,
        hidden: tensor_vec(state.hidden)?,
    })
}

/// Convert one tensor into host floats with operation context.
fn tensor_vec<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, RecurrentPpoError> {
    tensor
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| RecurrentPpoError::TensorConversion(error.to_string()))
}

/// Flatten a validated row-major matrix into a backend tensor.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
fn encode_matrix<B: Backend>(
    rows: &[impl AsRef<[f32]>],
    width: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let values: Vec<_> = rows.iter().flat_map(AsRef::as_ref).copied().collect();
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([rows.len(), width])
}

/// Evaluate corrected tanh-and-affine diagonal Gaussian log probabilities.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
fn tensor_log_probabilities<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
    samples: Tensor<B, 2>,
    action_low: &[f32],
    action_high: &[f32],
    epsilon: f32,
    device: &B::Device,
) -> Tensor<B, 1> {
    let standard_deviation = log_std.clone().exp();
    let normalized = (samples.clone() - mean) / standard_deviation;
    let gaussian = normalized.clone().mul(normalized) * -0.5 - log_std - 0.5 * LOG_TWO_PI;
    let squashed = samples.tanh();
    let jacobian = (squashed.clone().mul(squashed).neg() + 1.0)
        .clamp_min(epsilon)
        .log();
    let log_scales: Vec<_> = action_low
        .iter()
        .zip(action_high)
        .map(|(low, high)| ((high - low) * 0.5).ln())
        .collect();
    let log_scales =
        Tensor::<B, 1>::from_floats(log_scales.as_slice(), device).reshape([1, action_low.len()]);
    (gaussian - jacobian - log_scales)
        .sum_dim(1)
        .squeeze_dim::<1>(1)
}

/// Scale pre-tanh samples into configured environment bounds.
fn scale_action(samples: &[f32], action_low: &[f32], action_high: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .zip(action_low.iter().zip(action_high))
        .map(|(sample, (low, high))| {
            let scale = (high - low) * 0.5;
            let bias = (high + low) * 0.5;
            sample.tanh().mul_add(scale, bias)
        })
        .collect()
}

/// Normalize bounded environment actions into the actor's `[-1, 1]` range.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
fn normalize_action(actions: &[f32], action_low: &[f32], action_high: &[f32]) -> Vec<f32> {
    actions
        .iter()
        .zip(action_low.iter().zip(action_high))
        .map(|(action, (low, high))| 2.0 * (action - low) / (high - low) - 1.0)
        .collect()
}

/// Host equivalent of [`tensor_log_probabilities`] for behavior collection.
fn host_log_probability(
    mean: &[f32],
    log_std: &[f32],
    samples: &[f32],
    action_low: &[f32],
    action_high: &[f32],
    epsilon: f32,
) -> f32 {
    mean.iter()
        .zip(log_std)
        .zip(samples)
        .zip(action_low.iter().zip(action_high))
        .map(|(((location, log_scale), sample), (low, high))| {
            let deviation = log_scale.exp();
            let normalized = (sample - location) / deviation;
            let gaussian =
                (-0.5_f32).mul_add(normalized.mul_add(normalized, LOG_TWO_PI), -*log_scale);
            let squashed = sample.tanh();
            let jacobian = squashed.mul_add(-squashed, 1.0).max(epsilon).ln();
            let affine = ((high - low) * 0.5).ln();
            gaussian - jacobian - affine
        })
        .sum()
}

/// Compute shared advantage normalization statistics.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
fn advantage_normalization(sequences: &[RecurrentPpoSequence], epsilon: f32) -> (f32, f32) {
    let count: usize = sequences
        .iter()
        .map(|sequence| sequence.advantages.len())
        .sum();
    let mean = sequences
        .iter()
        .flat_map(|sequence| &sequence.advantages)
        .copied()
        .sum::<f32>()
        / count as f32;
    let variance = sequences
        .iter()
        .flat_map(|sequence| &sequence.advantages)
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / count as f32;
    (mean, (variance + epsilon).sqrt())
}

/// Validate one recurrent state width and finite content.
fn validate_memory(memory: &RecurrentMemory, hidden_size: usize) -> Result<(), RecurrentPpoError> {
    validate_vector(&memory.cell, hidden_size, "memory cell")?;
    validate_vector(&memory.hidden, hidden_size, "memory hidden")
}

/// Validate every row of one matrix.
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
fn validate_matrix(
    rows: &[Vec<f32>],
    width: usize,
    field: &'static str,
) -> Result<(), RecurrentPpoError> {
    for row in rows {
        validate_vector(row, width, field)?;
    }
    Ok(())
}

/// Validate one finite fixed-width vector.
fn validate_vector(
    values: &[f32],
    width: usize,
    field: &'static str,
) -> Result<(), RecurrentPpoError> {
    if values.len() != width {
        return Err(RecurrentPpoError::DimensionMismatch {
            field,
            expected: width,
            actual: values.len(),
        });
    }
    validate_finite(values, field)
}

/// Validate finite scalar content.
fn validate_finite(values: &[f32], field: &'static str) -> Result<(), RecurrentPpoError> {
    if let Some((dimension, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        Err(RecurrentPpoError::NonFiniteValue { field, dimension })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Borrowed rows preserve row order without cloning their owned vectors.
    #[test]
    fn matrix_encoding_accepts_borrowed_and_owned_rows() {
        let owned = [vec![1.0, 2.0], vec![-3.0, 4.0]];
        let borrowed = owned.each_ref().map(Vec::as_slice);
        let device = inference_device();
        let owned_tensor = encode_matrix::<InferenceBackend>(&owned, 2, &device);
        let borrowed_tensor = encode_matrix::<InferenceBackend>(&borrowed, 2, &device);
        assert_eq!(borrowed_tensor.dims(), [2, 2]);
        assert_eq!(
            tensor_vec(owned_tensor).expect("owned rows"),
            [1.0, 2.0, -3.0, 4.0]
        );
        assert_eq!(
            tensor_vec(borrowed_tensor).expect("borrowed rows"),
            [1.0, 2.0, -3.0, 4.0]
        );
    }

    /// Equal model seeds and observations must reproduce recurrent inference.
    #[test]
    fn same_seed_reproduces_actor_and_memory() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            ..RecurrentPpoConfig::default()
        };
        let seeds = SeedConfig::from_root(71);
        let first = RecurrentPpoAgent::new(3, 5, 2, &[-1.0], &[1.0], config.clone(), seeds)
            .expect("first agent initializes");
        let second = RecurrentPpoAgent::new(3, 5, 2, &[-1.0], &[1.0], config, seeds)
            .expect("second agent initializes");
        let memory = first.policy().initial_memory();
        let first_action = first
            .policy()
            .mean_action(&[0.2, -0.3, 0.4], &memory)
            .expect("first inference succeeds");
        let second_action = second
            .policy()
            .mean_action(&[0.2, -0.3, 0.4], &memory)
            .expect("second inference succeeds");
        assert_eq!(first_action, second_action);
    }

    /// Exploration begins inside its bounds and is shared across observations.
    #[test]
    fn exploration_scale_starts_unsaturated_and_state_independent() {
        // Match the ecosystem profile while retaining room below the clamp.
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            initial_log_std: -0.5,
            log_std_min: -5.0,
            log_std_max: 0.0,
            ..RecurrentPpoConfig::default()
        };
        let agent = RecurrentPpoAgent::new(
            3,
            5,
            2,
            &[-1.0, -1.0],
            &[1.0, 1.0],
            config.clone(),
            SeedConfig::from_root(72),
        )
        .expect("agent initializes");
        let policy = agent.policy();
        let memory = policy.initial_memory();
        let (_, first, _) = policy
            .distribution(&[0.0, 0.0, 0.0], &memory)
            .expect("first distribution succeeds");
        let (_, second, _) = policy
            .distribution(&[0.8, -0.4, 0.2], &memory)
            .expect("second distribution succeeds");

        assert_eq!(first, second);
        assert!(first
            .iter()
            .all(|value| config.log_std_min < *value && *value < config.log_std_max));
        assert!(first.iter().all(|value| (*value + 0.5).abs() < 1e-5));
    }

    /// Initial exploration must remain inside the emitted distribution bounds.
    #[test]
    fn initial_exploration_scale_outside_bounds_is_rejected() {
        // Validate construction before an invalid Gaussian reaches sampling.
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            initial_log_std: 2.0,
            ..RecurrentPpoConfig::default()
        };

        let error =
            RecurrentPpoAgent::new(3, 5, 2, &[-1.0], &[1.0], config, SeedConfig::from_root(74))
                .expect_err("out-of-bounds exploration is rejected");

        assert!(matches!(
            error,
            RecurrentPpoError::InvalidConfig {
                field: "initial_log_std",
                ..
            }
        ));
    }

    /// Recurrent memory must change and affect a later partially observed step.
    #[test]
    fn local_history_changes_recurrent_memory() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            ..RecurrentPpoConfig::default()
        };
        let agent =
            RecurrentPpoAgent::new(2, 4, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(73))
                .expect("agent initializes");
        let policy = agent.policy();
        let initial = policy.initial_memory();
        let remembered = policy
            .mean_action(&[1.0, 0.0], &initial)
            .expect("cue inference succeeds")
            .next_memory;
        let with_memory = policy
            .mean_action(&[0.0, 0.0], &remembered)
            .expect("remembered inference succeeds");
        let without_memory = policy
            .mean_action(&[0.0, 0.0], &initial)
            .expect("zero-memory inference succeeds");
        assert_ne!(remembered, initial);
        assert_ne!(with_memory.next_memory, without_memory.next_memory);
    }

    /// One valid sequence must update both actor and centralized critic.
    #[test]
    fn contiguous_sequence_updates_both_optimizers() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            epochs: 1,
            ..RecurrentPpoConfig::default()
        };
        let mut agent =
            RecurrentPpoAgent::new(2, 3, 2, &[-1.0], &[1.0], config, SeedConfig::from_root(79))
                .expect("agent initializes");
        let policy = agent.policy();
        let memory = policy.initial_memory();
        let first = policy
            .mean_action(&[0.1, 0.2], &memory)
            .expect("behavior inference succeeds");
        let sequence = RecurrentPpoSequence {
            observations: vec![vec![0.1, 0.2], vec![0.2, 0.3]],
            global_states: vec![vec![0.0, 0.1, 0.2], vec![0.1, 0.2, 0.3]],
            pre_tanh_actions: vec![first.pre_tanh_action.clone(), first.pre_tanh_action],
            old_log_probabilities: vec![first.log_probability, first.log_probability],
            advantages: vec![1.0, -0.25],
            returns: vec![1.5, 0.5],
            value_index: 0,
            initial_memory: memory,
        };
        let update = agent.update(&[sequence]).expect("sequence update succeeds");
        assert_eq!(update.optimizer_steps, 1);
        assert_eq!(update.optimizer_updates, 1);
        assert_eq!(update.valid_samples, 2);
        assert!(update.actor_loss.is_finite());
        assert!(update.critic_loss.is_finite());
        assert!(update.entropy.is_finite());
        assert!(update.approximate_kl.is_finite());
    }

    /// Critic warmup must leave recurrent actor behavior unchanged.
    #[test]
    fn critic_only_update_preserves_actor_and_changes_value() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            epochs: 2,
            ..RecurrentPpoConfig::default()
        };
        let mut agent =
            RecurrentPpoAgent::new(2, 3, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(80))
                .expect("agent initializes");
        let observation = [0.1, 0.2];
        let global_state = [0.0, 0.1, 0.2];
        let policy = agent.policy();
        let memory = policy.initial_memory();
        let before_action = policy
            .mean_action(&observation, &memory)
            .expect("actor evaluates before warmup");
        let before_value = policy
            .value(&global_state, 0)
            .expect("critic evaluates before warmup");
        let sequence = RecurrentPpoSequence {
            observations: vec![observation.to_vec(); 4],
            global_states: vec![global_state.to_vec(); 4],
            pre_tanh_actions: vec![before_action.pre_tanh_action.clone(); 4],
            old_log_probabilities: vec![before_action.log_probability; 4],
            advantages: vec![1.0; 4],
            returns: vec![2.0; 4],
            value_index: 0,
            initial_memory: memory,
        };

        let update = agent
            .update_critic(std::slice::from_ref(&sequence))
            .expect("critic warmup succeeds");
        let warmed = agent.policy();

        assert_eq!(
            before_action,
            warmed
                .mean_action(&observation, &warmed.initial_memory())
                .expect("actor evaluates after warmup")
        );
        assert_ne!(
            before_value.to_bits(),
            warmed
                .value(&global_state, 0)
                .expect("critic evaluates after warmup")
                .to_bits()
        );
        assert!(update.actor_loss.abs() < f64::EPSILON);
        assert!(update.critic_loss.is_finite());
        assert_eq!(update.optimizer_updates, 2);
    }

    /// Demonstrations must directly move the deterministic actor toward expert actions.
    #[test]
    fn behavior_cloning_reduces_mean_action_error() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            actor_learning_rate: 0.01,
            ..RecurrentPpoConfig::default()
        };
        let mut agent =
            RecurrentPpoAgent::new(2, 2, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(80))
                .expect("agent initializes");
        let demonstrations = [
            RecurrentBehaviorSample {
                observation: vec![1.0, 0.0],
                action: vec![0.8],
            },
            RecurrentBehaviorSample {
                observation: vec![-1.0, 0.0],
                action: vec![-0.8],
            },
        ];
        let mean_error = |agent: &RecurrentPpoAgent| {
            let policy = agent.policy();
            demonstrations
                .iter()
                .map(|sample| {
                    let predicted = policy
                        .mean_action(&sample.observation, &policy.initial_memory())
                        .expect("mean action succeeds")
                        .action[0];
                    (predicted - sample.action[0]).powi(2)
                })
                .sum::<f32>()
                / demonstrations.len() as f32
        };
        let initial_error = mean_error(&agent);
        for _ in 0..200 {
            agent
                .behavior_clone(&demonstrations)
                .expect("behavior update succeeds");
        }

        assert!(mean_error(&agent) < initial_error * 0.1);
    }

    /// Explicit-memory behavior cloning must fit targets at nonzero LSTM states.
    #[test]
    fn explicit_memory_behavior_cloning_reduces_mean_action_error() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            actor_learning_rate: 0.01,
            ..RecurrentPpoConfig::default()
        };
        let mut agent =
            RecurrentPpoAgent::new(2, 2, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(81))
                .expect("agent initializes");
        let policy = agent.policy();
        let memory = policy
            .mean_action(&[1.0, 0.0], &policy.initial_memory())
            .expect("warmup succeeds")
            .next_memory;
        let demonstrations = [RecurrentBehaviorMemorySample {
            observation: vec![0.0, 1.0],
            initial_memory: memory,
            action: vec![0.8],
        }];
        let mean_error = |agent: &RecurrentPpoAgent| {
            let policy = agent.policy();
            let sample = &demonstrations[0];
            let predicted = policy
                .mean_action(&sample.observation, &sample.initial_memory)
                .expect("explicit-memory mean action succeeds")
                .action[0];
            (predicted - sample.action[0]).powi(2)
        };
        let initial_error = mean_error(&agent);
        for _ in 0..200 {
            agent
                .behavior_clone_with_memory(&demonstrations)
                .expect("explicit-memory behavior update succeeds");
        }

        assert!(mean_error(&agent) < initial_error * 0.1);
    }

    /// Mixed recurrent chunks must contribute in proportion to valid timesteps.
    #[test]
    fn mixed_length_sequences_weight_update_metrics_by_valid_timesteps() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            entropy_coefficient: 0.1,
            epochs: 1,
            minibatch_sequences: 2,
            ..RecurrentPpoConfig::default()
        };
        let seeds = SeedConfig::from_root(81);
        let make_agent = || {
            RecurrentPpoAgent::new(2, 3, 1, &[-1.0], &[1.0], config.clone(), seeds)
                .expect("agent initializes")
        };
        let memory = make_agent().policy().initial_memory();
        let sequence = |length: usize, observation: [f32; 2], old_log_probability: f32, target| {
            RecurrentPpoSequence {
                observations: vec![observation.to_vec(); length],
                global_states: vec![vec![observation[0], observation[1], 0.25]; length],
                pre_tanh_actions: vec![vec![0.0]; length],
                old_log_probabilities: vec![old_log_probability; length],
                advantages: vec![1.0; length],
                returns: vec![target; length],
                value_index: 0,
                initial_memory: memory.clone(),
            }
        };
        let short = sequence(1, [0.1, 0.2], 0.3, 4.0);
        let long = sequence(32, [-0.4, 0.7], -0.2, -1.5);
        let short_update = make_agent()
            .update(std::slice::from_ref(&short))
            .expect("short update succeeds");
        let long_update = make_agent()
            .update(std::slice::from_ref(&long))
            .expect("long update succeeds");
        let mixed_update = make_agent()
            .update(&[short, long])
            .expect("mixed update succeeds");

        // The worked 1:32 weighting is independent of optimizer batching.
        let valid_timestep_mean =
            |short_value: f64, long_value: f64| 32.0_f64.mul_add(long_value, short_value) / 33.0;
        for (actual, expected) in [
            (
                mixed_update.actor_loss,
                valid_timestep_mean(short_update.actor_loss, long_update.actor_loss),
            ),
            (
                mixed_update.critic_loss,
                valid_timestep_mean(short_update.critic_loss, long_update.critic_loss),
            ),
            (
                mixed_update.entropy,
                valid_timestep_mean(short_update.entropy, long_update.entropy),
            ),
            (
                mixed_update.approximate_kl,
                valid_timestep_mean(short_update.approximate_kl, long_update.approximate_kl),
            ),
        ] {
            assert!(
                (actual - expected).abs() < 1.0e-6,
                "actual {actual} differs from valid-timestep mean {expected}"
            );
        }
    }

    /// Episode-boundary memory dimensions are checked before tensorization.
    #[test]
    fn invalid_memory_width_is_rejected() {
        let agent = RecurrentPpoAgent::new(
            2,
            3,
            1,
            &[-1.0],
            &[1.0],
            RecurrentPpoConfig::default(),
            SeedConfig::from_root(83),
        )
        .expect("agent initializes");
        let error = agent
            .policy()
            .mean_action(
                &[0.0, 0.0],
                &RecurrentMemory {
                    cell: vec![0.0],
                    hidden: vec![0.0],
                },
            )
            .expect_err("invalid memory is rejected");
        assert!(matches!(
            error,
            RecurrentPpoError::DimensionMismatch {
                field: "memory cell",
                ..
            }
        ));
    }

    /// Recurrent chunks are shuffled reproducibly without loss or duplication.
    #[test]
    fn minibatch_shuffle_is_seeded_and_permutational() {
        let mut first_rng = MinibatchRng::new(97);
        let mut second_rng = MinibatchRng::new(97);
        let mut first = (0..32).collect::<Vec<_>>();
        let mut second = first.clone();
        first_rng.shuffle(&mut first);
        second_rng.shuffle(&mut second);

        assert_eq!(first, second);
        assert_ne!(first, (0..32).collect::<Vec<_>>());
        first.sort_unstable();
        assert_eq!(first, (0..32).collect::<Vec<_>>());
    }

    /// A single record must preserve actor memory evolution and critic output.
    #[test]
    #[expect(
        clippy::allow_attributes,
        reason = "the repository disallows synchronous filesystem helpers by default"
    )]
    #[allow(
        clippy::disallowed_methods,
        reason = "bounded synchronous filesystem setup keeps the recorder round-trip test dependency-free"
    )]
    fn checkpoint_round_trip_preserves_recurrent_policy() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            ..RecurrentPpoConfig::default()
        };
        let agent = RecurrentPpoAgent::new(
            2,
            3,
            2,
            &[-1.0],
            &[1.0],
            config.clone(),
            SeedConfig::from_root(89),
        )
        .expect("agent initializes");
        let policy = agent.policy();
        let observation = [0.25, -0.5];
        let global_state = [0.1, 0.2, 0.3];
        let expected_action = policy
            .mean_action(&observation, &policy.initial_memory())
            .expect("source actor inference succeeds");
        let expected_value = policy
            .value(&global_state, 1)
            .expect("source critic inference succeeds");
        let root = std::env::temp_dir().join(format!(
            "bevy-gym-recurrent-ppo-checkpoint-{}",
            std::process::id()
        ));
        drop(std::fs::remove_dir_all(&root));
        std::fs::create_dir_all(&root).expect("temporary checkpoint directory creates");
        let path = root.join("policy.mpk");
        policy.save(&path).expect("policy checkpoint saves");
        let loaded = RecurrentPpoPolicy::load(&path, 2, 3, 2, &[-1.0], &[1.0], &config)
            .expect("policy checkpoint loads");
        let checkpoint_bytes = std::fs::read(&path).expect("checkpoint bytes read");
        let loaded_from_bytes =
            RecurrentPpoPolicy::load_bytes(checkpoint_bytes, 2, 3, 2, &[-1.0], &[1.0], &config)
                .expect("policy checkpoint bytes load");

        assert_eq!(
            expected_action,
            loaded
                .mean_action(&observation, &loaded.initial_memory())
                .expect("loaded actor inference succeeds")
        );
        assert_eq!(
            expected_value.to_bits(),
            loaded
                .value(&global_state, 1)
                .expect("loaded critic inference succeeds")
                .to_bits()
        );
        assert_eq!(
            expected_action,
            loaded_from_bytes
                .mean_action(&observation, &loaded_from_bytes.initial_memory())
                .expect("bytes-loaded actor inference succeeds")
        );
        assert_eq!(
            expected_value.to_bits(),
            loaded_from_bytes
                .value(&global_state, 1)
                .expect("bytes-loaded critic inference succeeds")
                .to_bits()
        );
        drop(std::fs::remove_dir_all(root));
    }

    /// Stage transfer must preserve the actor while reinitializing the critic.
    #[test]
    #[expect(
        clippy::allow_attributes,
        reason = "the repository disallows synchronous filesystem helpers by default"
    )]
    #[allow(
        clippy::disallowed_methods,
        reason = "bounded synchronous filesystem setup keeps the transfer test dependency-free"
    )]
    fn actor_only_checkpoint_load_keeps_a_fresh_critic() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            ..RecurrentPpoConfig::default()
        };
        let source = RecurrentPpoAgent::new(
            2,
            3,
            2,
            &[-1.0],
            &[1.0],
            config.clone(),
            SeedConfig::from_root(101),
        )
        .expect("source agent initializes");
        let root = std::env::temp_dir().join(format!(
            "bevy-gym-recurrent-ppo-actor-transfer-{}",
            std::process::id()
        ));
        drop(std::fs::remove_dir_all(&root));
        std::fs::create_dir_all(&root).expect("temporary checkpoint directory creates");
        let path = root.join("policy.mpk");
        source.policy().save(&path).expect("source policy saves");
        let transfer_seeds = SeedConfig::from_root(103);
        let fresh =
            RecurrentPpoAgent::new(2, 3, 2, &[-1.0], &[1.0], config.clone(), transfer_seeds)
                .expect("fresh reference agent initializes");

        let transferred = RecurrentPpoAgent::load_actor_with_fresh_critic(
            &path,
            2,
            3,
            2,
            &[-1.0],
            &[1.0],
            config,
            transfer_seeds,
        )
        .expect("actor-only transfer loads");
        let observation = [0.25, -0.5];
        let global_state = [0.1, 0.2, 0.3];
        let source_policy = source.policy();
        let fresh_policy = fresh.policy();
        let transferred_policy = transferred.policy();

        assert_eq!(
            source_policy
                .mean_action(&observation, &source_policy.initial_memory())
                .expect("source actor evaluates"),
            transferred_policy
                .mean_action(&observation, &transferred_policy.initial_memory())
                .expect("transferred actor evaluates")
        );
        assert_eq!(
            fresh_policy
                .value(&global_state, 1)
                .expect("fresh critic evaluates")
                .to_bits(),
            transferred_policy
                .value(&global_state, 1)
                .expect("transferred critic evaluates")
                .to_bits()
        );
        assert_ne!(
            source_policy
                .value(&global_state, 1)
                .expect("source critic evaluates")
                .to_bits(),
            transferred_policy
                .value(&global_state, 1)
                .expect("transferred critic evaluates")
                .to_bits()
        );
        drop(std::fs::remove_dir_all(root));
    }

    /// Input-feature transfer must preserve the source actor response exactly.
    #[test]
    fn actor_input_feature_copy_preserves_behavior_and_validates_indices() {
        let mut agent = RecurrentPpoAgent::new(
            2,
            3,
            2,
            &[-1.0],
            &[1.0],
            RecurrentPpoConfig::default(),
            SeedConfig::from_root(107),
        )
        .expect("agent initializes");
        let source_policy = agent.policy();
        let expected = source_policy
            .mean_action(&[1.0, 0.0], &source_policy.initial_memory())
            .expect("source feature evaluates");

        agent
            .copy_actor_input_feature(0, 1)
            .expect("bounded feature copy succeeds");
        let transferred_policy = agent.policy();

        assert_eq!(
            expected,
            transferred_policy
                .mean_action(&[0.0, 1.0], &transferred_policy.initial_memory())
                .expect("target feature evaluates")
        );
        assert!(agent.copy_actor_input_feature(2, 0).is_err());
        assert!(agent.copy_actor_input_feature(0, 2).is_err());
    }

    /// Adding one zero-weight observation feature must preserve legacy actor behavior exactly.
    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded synchronous filesystem setup keeps the migration test dependency-free"
    )]
    fn actor_input_insertion_migrates_a_legacy_checkpoint() {
        let directory = std::env::temp_dir().join(format!(
            "bevy-gym-recurrent-ppo-input-migration-{}",
            std::process::id()
        ));
        if directory.exists() {
            std::fs::remove_dir_all(&directory).expect("stale test directory is removable");
        }
        std::fs::create_dir_all(&directory).expect("test directory is creatable");
        let checkpoint = directory.join("legacy.mpk");
        let config = RecurrentPpoConfig {
            actor_hidden_size: 8,
            critic_hidden_sizes: vec![8],
            ..RecurrentPpoConfig::default()
        };
        let source = RecurrentPpoAgent::new(
            2,
            3,
            2,
            &[-1.0],
            &[1.0],
            config.clone(),
            SeedConfig::from_root(127),
        )
        .expect("source learner initializes");
        source
            .policy()
            .save(&checkpoint)
            .expect("legacy checkpoint saves");
        let source_policy = source.policy();
        let migrated = RecurrentPpoAgent::load_actor_with_inserted_input_feature_and_fresh_critic(
            &checkpoint,
            2,
            1,
            3,
            2,
            &[-1.0],
            &[1.0],
            config,
            SeedConfig::from_root(131),
        )
        .expect("legacy actor migrates");
        let migrated_policy = migrated.policy();
        let source_action = source_policy
            .mean_action(&[0.25, -0.5], &source_policy.initial_memory())
            .expect("source actor evaluates");
        let migrated_action = migrated_policy
            .mean_action(&[0.25, 1.0, -0.5], &migrated_policy.initial_memory())
            .expect("migrated actor evaluates");

        assert_eq!(source_action, migrated_action);
        std::fs::remove_dir_all(directory).expect("test directory is removable");
    }

    /// Mean-bias calibration must change one action without changing memory.
    #[test]
    fn actor_mean_bias_shift_is_scoped_and_validated() {
        let mut agent = RecurrentPpoAgent::new(
            2,
            3,
            2,
            &[-1.0, -1.0],
            &[1.0, 1.0],
            RecurrentPpoConfig::default(),
            SeedConfig::from_root(109),
        )
        .expect("agent initializes");
        let observation = [0.25, -0.5];
        let before_policy = agent.policy();
        let before = before_policy
            .mean_action(&observation, &before_policy.initial_memory())
            .expect("actor evaluates before calibration");

        agent
            .shift_actor_mean_bias(0, 0.10)
            .expect("bounded bias shift succeeds");
        let after_policy = agent.policy();
        let after = after_policy
            .mean_action(&observation, &after_policy.initial_memory())
            .expect("actor evaluates after calibration");

        assert!(after.action[0] > before.action[0]);
        assert_eq!(after.action[1].to_bits(), before.action[1].to_bits());
        assert_eq!(after.next_memory, before.next_memory);
        assert!(agent.shift_actor_mean_bias(2, 0.10).is_err());
        assert!(agent.shift_actor_mean_bias(0, f32::NAN).is_err());
    }
}
