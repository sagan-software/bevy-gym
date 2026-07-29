//! Recurrent PPO with a decentralized LSTM actor and centralized critic.
//!
//! The actor API accepts only local observations and its own recurrent memory.
//! Global state is accepted by a separate training-only critic API, which keeps
//! centralized training data unreachable from decentralized execution.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};

use burn::module::{AutodiffModule, Initializer, Module, Param};
use burn::nn::{Linear, LinearConfig, Lstm, LstmConfig, LstmState, Relu};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion};
use burn::tensor::Tensor;

use super::backend::{
    inference_device, training_device, InferenceBackend, TrainingBackend, TrainingDevice,
    MODEL_INITIALIZATION_LOCK,
};
use super::checkpoint::{policy_recorder, CheckpointError, CheckpointOperation};
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

impl RecurrentPpoSequence {
    /// Validate dimensions without permitting episode-boundary padding.
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
struct MinibatchRng {
    /// Current `SplitMix64` state.
    state: u64,
}

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
        let networks = RecurrentNetworks {
            actor: agent.actor.clone(),
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
        agent.critic = networks.critic;
        Ok(agent)
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
fn encode_matrix<B: Backend>(rows: &[Vec<f32>], width: usize, device: &B::Device) -> Tensor<B, 2> {
    let values: Vec<_> = rows.iter().flatten().copied().collect();
    Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([rows.len(), width])
}

/// Evaluate corrected tanh-and-affine diagonal Gaussian log probabilities.
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
        drop(std::fs::remove_dir_all(root));
    }
}
