//! Reusable Burn-backed Deep Q-Network training core.

use std::error::Error;
use std::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion};
use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
use burn::tensor::{Int, Tensor};

#[cfg(not(target_arch = "wasm32"))]
use crate::Env;
use crate::EpisodeStatus;

use super::backend::{
    inference_device, training_device, InferenceBackend, TrainingBackend, MODEL_INITIALIZATION_LOCK,
};
use super::checkpoint::{policy_recorder, CheckpointError, CheckpointOperation};
#[cfg(not(target_arch = "wasm32"))]
use super::config::{RunConfig, RunPaths};
use super::rng::SeedConfig;

/// Environment-neutral configuration for the DQN learner.
#[derive(Debug, Clone, PartialEq)]
pub struct DqnConfig {
    /// Hidden layer widths for the Q-network MLP.
    pub hidden_sizes: Vec<usize>,

    /// Discount factor.
    pub gamma: f32,

    /// Adam learning rate.
    pub learning_rate: f64,

    /// Numerical-stability epsilon used by Adam.
    pub optimizer_epsilon: f32,

    /// Replay buffer capacity.
    pub replay_capacity: usize,

    /// Number of stored transitions required before optimization starts.
    pub min_replay_size: usize,

    /// Minibatch size.
    pub batch_size: usize,

    /// Target-network update interval in optimizer steps.
    pub target_update_interval: usize,

    /// Exploration probability at environment step zero.
    pub epsilon_start: f64,

    /// Exploration probability after the decay interval.
    pub epsilon_end: f64,

    /// Environment steps over which epsilon decays linearly.
    pub epsilon_decay_steps: u64,

    /// Delta separating quadratic and linear regions of the Huber TD loss.
    pub huber_delta: f32,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            hidden_sizes: vec![64, 64],
            gamma: 0.99,
            learning_rate: 3e-4,
            optimizer_epsilon: 1e-8,
            replay_capacity: 50_000,
            min_replay_size: 1_000,
            batch_size: 64,
            target_update_interval: 500,
            epsilon_start: 0.15,
            epsilon_end: 0.01,
            epsilon_decay_steps: 10_000,
            huber_delta: 1.0,
        }
    }
}

impl DqnConfig {
    /// Validate generic DQN dimensions and hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`DqnError::InvalidConfig`] for zero dimensions, invalid layer
    /// widths, or hyperparameters outside their supported ranges.
    pub fn validate(&self, observation_dim: usize, action_dim: usize) -> Result<(), DqnError> {
        validate_architecture(observation_dim, action_dim, &self.hidden_sizes)?;
        if !self.gamma.is_finite() || !(0.0..=1.0).contains(&self.gamma) {
            return Err(DqnError::invalid_config(
                "gamma",
                "must be finite and in 0..=1",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(DqnError::invalid_config(
                "learning_rate",
                "must be finite and greater than zero",
            ));
        }
        if !self.optimizer_epsilon.is_finite() || self.optimizer_epsilon <= 0.0 {
            return Err(DqnError::invalid_config(
                "optimizer_epsilon",
                "must be finite and greater than zero",
            ));
        }
        if self.replay_capacity == 0 {
            return Err(DqnError::invalid_config(
                "replay_capacity",
                "must be greater than zero",
            ));
        }
        if self.min_replay_size == 0 || self.min_replay_size > self.replay_capacity {
            return Err(DqnError::invalid_config(
                "min_replay_size",
                "must be in 1..=replay_capacity",
            ));
        }
        if self.batch_size == 0 || self.batch_size > self.replay_capacity {
            return Err(DqnError::invalid_config(
                "batch_size",
                "must be in 1..=replay_capacity",
            ));
        }
        if self.target_update_interval == 0 {
            return Err(DqnError::invalid_config(
                "target_update_interval",
                "must be greater than zero",
            ));
        }
        if !self.epsilon_start.is_finite()
            || !self.epsilon_end.is_finite()
            || !(0.0..=1.0).contains(&self.epsilon_start)
            || !(0.0..=1.0).contains(&self.epsilon_end)
            || self.epsilon_start < self.epsilon_end
        {
            return Err(DqnError::invalid_config(
                "epsilon",
                "start and end must be finite in 0..=1 with start >= end",
            ));
        }
        if self.epsilon_decay_steps == 0 {
            return Err(DqnError::invalid_config(
                "epsilon_decay_steps",
                "must be greater than zero",
            ));
        }
        if !self.huber_delta.is_finite() || self.huber_delta <= 0.0 {
            return Err(DqnError::invalid_config(
                "huber_delta",
                "must be finite and greater than zero",
            ));
        }

        Ok(())
    }
}

/// Runtime-shaped multilayer perceptron that emits one Q-value per action.
#[derive(Module, Debug)]
struct QNetwork<B: Backend> {
    /// Ordered linear layers from encoded observation to action values.
    layers: Vec<Linear<B>>,

    /// Nonlinearity applied after every hidden layer.
    activation: Relu,
}

impl<B: Backend> QNetwork<B> {
    /// Initialize a Q-network with caller-provided runtime dimensions.
    fn new(
        observation_dim: usize,
        action_dim: usize,
        hidden_sizes: &[usize],
        device: &B::Device,
    ) -> Self {
        let widths: Vec<_> = std::iter::once(observation_dim)
            .chain(hidden_sizes.iter().copied())
            .chain(std::iter::once(action_dim))
            .collect();
        let layers = widths
            .windows(2)
            .map(|window| {
                let input = *window.first().expect("MLP width window has an input");
                let output = *window.get(1).expect("MLP width window has an output");
                LinearConfig::new(input, output).init(device)
            })
            .collect();

        Self {
            layers,
            activation: Relu::new(),
        }
    }

    /// Evaluate a batch of encoded observations.
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

/// Validate checkpoint shapes and finite parameters before activating a loaded network.
fn validate_policy_record(
    record: &QNetworkRecord<InferenceBackend>,
    observation_dim: usize,
    action_dim: usize,
    hidden_sizes: &[usize],
) -> Result<(), DqnError> {
    // Reject incompatible shapes before Burn can execute a mismatched matrix product.
    if record.layers.len() != hidden_sizes.len() + 1 {
        return Err(DqnError::invalid_config(
            "checkpoint",
            "layer count differs from the declared architecture",
        ));
    }
    let inputs = std::iter::once(observation_dim).chain(hidden_sizes.iter().copied());
    let outputs = hidden_sizes
        .iter()
        .copied()
        .chain(std::iter::once(action_dim));
    for (layer, (input, output)) in record.layers.iter().zip(inputs.zip(outputs)) {
        let bias = layer.bias.as_ref().ok_or_else(|| {
            DqnError::invalid_config("checkpoint", "every layer must contain a bias")
        })?;
        if layer.weight.val().dims() != [input, output] || bias.val().dims() != [output] {
            return Err(DqnError::invalid_config(
                "checkpoint",
                "layer shape differs from the declared architecture",
            ));
        }
        let weights = layer
            .weight
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| DqnError::TensorConversion(error.to_string()))?;
        let bias = bias
            .val()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| DqnError::TensorConversion(error.to_string()))?;
        if !weights.iter().chain(&bias).all(|value| value.is_finite()) {
            return Err(DqnError::invalid_config(
                "checkpoint",
                "parameters must be finite",
            ));
        }
    }
    Ok(())
}

/// Deterministic inference-only DQN policy.
#[derive(Debug, Clone)]
pub struct DqnPolicy {
    /// Inference-backend Q-network parameters.
    network: QNetwork<InferenceBackend>,

    /// Encoded observation width.
    observation_dim: usize,

    /// Discrete action count.
    action_dim: usize,

    /// Hidden widths needed to rebuild the record shape.
    hidden_sizes: Vec<usize>,
}

impl DqnPolicy {
    /// Serialize a policy record without accessing the filesystem.
    ///
    /// # Errors
    /// Returns a checkpoint error if the record cannot be encoded.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DqnError> {
        NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .record(self.network.clone().into_record(), ())
            .map_err(|error| {
                DqnError::Checkpoint(CheckpointError::new(
                    CheckpointOperation::Save,
                    "<memory>",
                    error.to_string(),
                ))
            })
    }

    /// Load a policy from bytes with its declared architecture.
    ///
    /// # Errors
    /// Returns a configuration or checkpoint error for invalid input.
    pub fn load_bytes(
        bytes: Vec<u8>,
        observation_dim: usize,
        action_dim: usize,
        hidden_sizes: &[usize],
    ) -> Result<Self, DqnError> {
        validate_architecture(observation_dim, action_dim, hidden_sizes)?;
        let device = inference_device();
        let record: QNetworkRecord<InferenceBackend> =
            NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
                .load(bytes, &device)
                .map_err(|error| {
                    DqnError::Checkpoint(CheckpointError::new(
                        CheckpointOperation::Load,
                        "<memory>",
                        error.to_string(),
                    ))
                })?;
        validate_policy_record(&record, observation_dim, action_dim, hidden_sizes)?;
        let _guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let network =
            QNetwork::new(observation_dim, action_dim, hidden_sizes, &device).load_record(record);
        Ok(Self {
            network,
            observation_dim,
            action_dim,
            hidden_sizes: hidden_sizes.to_vec(),
        })
    }

    /// Return Q-values for one encoded observation.
    ///
    /// # Errors
    ///
    /// Returns [`DqnError::ObservationDimension`] when the observation length
    /// does not match the runtime dimension used to build the policy.
    pub fn q_values(&self, observation: &[f32]) -> Result<Vec<f32>, DqnError> {
        validate_observation(observation, self.observation_dim)?;
        let device = inference_device();
        let input = Tensor::<InferenceBackend, 1>::from_floats(observation, &device)
            .reshape([1, self.observation_dim]);
        self.network
            .forward(input)
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| DqnError::TensorConversion(error.to_string()))
    }

    /// Select the highest-valued discrete action without exploration.
    ///
    /// # Errors
    ///
    /// Returns an observation error from [`Self::q_values`].
    pub fn greedy_action(&self, observation: &[f32]) -> Result<usize, DqnError> {
        let q_values = self.q_values(observation)?;
        greedy_legal_action(&q_values, &vec![true; self.action_dim])
    }

    /// Select the highest-valued action allowed by an environment action mask.
    ///
    /// # Errors
    ///
    /// Returns an observation or action-mask validation error.
    pub fn greedy_action_masked(
        &self,
        observation: &[f32],
        action_mask: &[bool],
    ) -> Result<usize, DqnError> {
        validate_action_mask(action_mask, self.action_dim)?;
        greedy_legal_action(&self.q_values(observation)?, action_mask)
    }

    /// Runtime observation width expected by this policy.
    #[must_use]
    pub const fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    /// Number of discrete actions emitted by this policy.
    #[must_use]
    pub const fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Hidden MLP widths required when recreating this policy for loading.
    #[must_use]
    pub fn hidden_sizes(&self) -> &[usize] {
        &self.hidden_sizes
    }

    /// Number of scalar trainable parameters in the policy.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.network.num_params()
    }

    /// Save this policy with the architecture-selected named `MessagePack`
    /// recorder.
    ///
    /// # Errors
    ///
    /// Returns [`DqnError::Checkpoint`] with save-path context when recording
    /// fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf, DqnError> {
        let path = path.as_ref().to_path_buf();
        self.network
            .clone()
            .save_file(path.clone(), &policy_recorder())
            .map_err(|error| {
                DqnError::Checkpoint(CheckpointError::new(
                    CheckpointOperation::Save,
                    path.clone(),
                    error.to_string(),
                ))
            })?;
        Ok(path)
    }

    /// Save this policy to the run's root `best.mpk` contract path.
    ///
    /// # Errors
    ///
    /// Returns the same checkpoint error as [`Self::save`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_best(&self, paths: &RunPaths) -> Result<PathBuf, DqnError> {
        self.save(&paths.best_checkpoint)
    }

    /// Save this policy to the run's `checkpoints/latest.mpk` contract path.
    ///
    /// # Errors
    ///
    /// Returns the same checkpoint error as [`Self::save`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_latest(&self, paths: &RunPaths) -> Result<PathBuf, DqnError> {
        self.save(&paths.latest_checkpoint)
    }

    /// Load a policy record into a freshly initialized runtime-shaped MLP.
    ///
    /// Because Burn policy records contain parameters rather than application
    /// architecture metadata, callers provide the same dimensions and hidden
    /// widths used for training.
    ///
    /// # Errors
    ///
    /// Returns an invalid-config error for unusable dimensions or a checkpoint
    /// error with load-path context when decoding fails.
    pub fn load(
        path: impl AsRef<Path>,
        observation_dim: usize,
        action_dim: usize,
        hidden_sizes: &[usize],
    ) -> Result<Self, DqnError> {
        validate_architecture(observation_dim, action_dim, hidden_sizes)?;
        let path = path.as_ref().to_path_buf();
        let _model_initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = inference_device();
        let network =
            QNetwork::<InferenceBackend>::new(observation_dim, action_dim, hidden_sizes, &device)
                .load_file(path.clone(), &policy_recorder(), &device)
                .map_err(|error| {
                    DqnError::Checkpoint(CheckpointError::new(
                        CheckpointOperation::Load,
                        path,
                        error.to_string(),
                    ))
                })?;

        Ok(Self {
            network,
            observation_dim,
            action_dim,
            hidden_sizes: hidden_sizes.to_vec(),
        })
    }

    /// Load the run's root `best.mpk` policy record.
    ///
    /// # Errors
    ///
    /// Returns the same validation or checkpoint error as [`Self::load`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_best(
        paths: &RunPaths,
        observation_dim: usize,
        action_dim: usize,
        hidden_sizes: &[usize],
    ) -> Result<Self, DqnError> {
        Self::load(
            &paths.best_checkpoint,
            observation_dim,
            action_dim,
            hidden_sizes,
        )
    }
}

/// Replay-owned encoded transition.
#[derive(Debug, Clone)]
struct Experience {
    /// Encoded observation before the action.
    observation: Vec<f32>,

    /// Discrete action applied to the environment.
    action_index: usize,

    /// Scalar environment reward.
    reward: f64,

    /// Encoded observation after the action.
    next_observation: Vec<f32>,

    /// Legal actions in the next state for masked value bootstrapping.
    next_action_mask: Vec<bool>,

    /// Sign or scale applied to the discounted next-state value.
    bootstrap_multiplier: f32,

    /// Episode boundary semantics for target masking.
    status: EpisodeStatus,
}

/// Fixed-capacity cyclic replay storage with seeded sampling.
#[derive(Debug)]
struct ReplayBuffer {
    /// Retained transitions.
    data: Vec<Experience>,

    /// Maximum retained transition count.
    capacity: usize,

    /// Slot replaced by the next push after the buffer fills.
    next_index: usize,
}

impl ReplayBuffer {
    /// Allocate empty replay storage.
    fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            next_index: 0,
        }
    }

    /// Append or cyclically replace one transition.
    fn push(&mut self, experience: Experience) {
        if self.data.len() < self.capacity {
            self.data.push(experience);
        } else {
            *self
                .data
                .get_mut(self.next_index)
                .expect("replay replacement index stays within capacity") = experience;
            self.next_index = (self.next_index + 1) % self.capacity;
        }
    }

    /// Return the number of retained transitions.
    const fn len(&self) -> usize {
        self.data.len()
    }

    /// Sample a minibatch with replacement from a dedicated RNG stream.
    fn sample(&self, batch_size: usize, rng: &mut SplitMix64) -> Vec<Experience> {
        (0..batch_size)
            .map(|_| {
                self.data
                    .get(rng.usize_below(self.data.len()))
                    .expect("sampled replay index is in bounds")
                    .clone()
            })
            .collect()
    }
}

/// Small deterministic RNG used to isolate action and replay streams.
#[derive(Debug, Clone)]
struct SplitMix64 {
    /// Current `SplitMix64` state.
    state: u64,
}

impl SplitMix64 {
    /// Initialize the stream from one derived seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate the next mixed integer.
    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Generate a floating-point value in the half-open interval `[0, 1)`.
    fn f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Select one index below a nonzero upper bound.
    const fn usize_below(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

/// Evidence returned by epsilon-greedy action selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DqnActionSelection {
    /// Selected discrete action index.
    pub action_index: usize,

    /// Whether this choice came from the exploration RNG.
    pub explored: bool,

    /// Epsilon used for this choice.
    pub epsilon: f64,
}

/// Evidence emitted by one real DQN optimizer update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DqnUpdate {
    /// One-based optimizer step after applying this update.
    pub optimizer_step: u64,

    /// Mean Huber temporal-difference loss.
    pub loss: f64,

    /// Number of scalar trainable parameters updated by the optimizer.
    pub parameter_count: usize,

    /// L1 change in the online policy's Q-values for a sampled probe
    /// observation before and after the update.
    pub policy_output_delta_l1: f64,

    /// Whether this update copied online parameters to the target network.
    pub target_updated: bool,

    /// Number of sampled transitions whose next-state value was bootstrapped.
    pub bootstrapped_samples: usize,

    /// Number of sampled natural terminal transitions.
    pub terminated_samples: usize,

    /// Number of sampled time-limit or externally truncated transitions.
    pub truncated_samples: usize,
}

/// Stateful DQN learner with runtime observation and action dimensions.
pub struct DqnAgent {
    /// Autodiff network updated by Adam.
    online_network: QNetwork<TrainingBackend>,

    /// Frozen inference network used to build TD targets.
    target_network: QNetwork<InferenceBackend>,

    /// Adam state tied to the online network parameters.
    optimizer: OptimizerAdaptor<Adam, QNetwork<TrainingBackend>, TrainingBackend>,

    /// Cyclic off-policy transition storage.
    replay: ReplayBuffer,

    /// Device used for training tensors.
    device: super::backend::TrainingDevice,

    /// Encoded observation width.
    observation_dim: usize,

    /// Discrete action count.
    action_dim: usize,

    /// Validated algorithm hyperparameters.
    config: DqnConfig,

    /// Dedicated epsilon/action sampling stream.
    action_rng: SplitMix64,

    /// Dedicated replay sampling stream.
    replay_rng: SplitMix64,

    /// Number of stored environment transitions.
    global_steps: u64,

    /// Number of applied optimizer updates.
    optimizer_steps: u64,
}

impl fmt::Debug for DqnAgent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DqnAgent")
            .field("observation_dim", &self.observation_dim)
            .field("action_dim", &self.action_dim)
            .field("config", &self.config)
            .field("parameter_count", &self.online_network.num_params())
            .field("replay_len", &self.replay.len())
            .field("global_steps", &self.global_steps)
            .field("optimizer_steps", &self.optimizer_steps)
            .finish_non_exhaustive()
    }
}

impl DqnAgent {
    /// Initialize a generic DQN learner and apply the dedicated model seed
    /// before creating any parameters.
    ///
    /// # Errors
    ///
    /// Returns [`DqnError::InvalidConfig`] when dimensions or hyperparameters
    /// are invalid.
    pub fn new(
        observation_dim: usize,
        action_dim: usize,
        config: DqnConfig,
        seeds: SeedConfig,
    ) -> Result<Self, DqnError> {
        config.validate(observation_dim, action_dim)?;
        let _model_initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = training_device();
        TrainingBackend::seed(&device, seeds.model);
        let online_network =
            QNetwork::new(observation_dim, action_dim, &config.hidden_sizes, &device);
        // Burn parameters can initialize lazily. Materialize them before any
        // later agent reseeds the process-global backend RNG.
        let _materialized = online_network
            .forward(Tensor::zeros([1, observation_dim], &device))
            .into_data();
        let target_network = online_network.valid();
        let optimizer = AdamConfig::new()
            .with_epsilon(config.optimizer_epsilon)
            .init::<TrainingBackend, QNetwork<TrainingBackend>>();
        let replay_capacity = config.replay_capacity;

        Ok(Self {
            online_network,
            target_network,
            optimizer,
            replay: ReplayBuffer::new(replay_capacity),
            device,
            observation_dim,
            action_dim,
            config,
            action_rng: SplitMix64::new(seeds.action),
            replay_rng: SplitMix64::new(seeds.replay),
            global_steps: 0,
            optimizer_steps: 0,
        })
    }

    /// Snapshot the current online network as an inference-only policy.
    #[must_use]
    pub fn policy(&self) -> DqnPolicy {
        DqnPolicy {
            network: self.online_network.valid(),
            observation_dim: self.observation_dim,
            action_dim: self.action_dim,
            hidden_sizes: self.config.hidden_sizes.clone(),
        }
    }

    /// Return Q-values from the current online policy.
    ///
    /// # Errors
    ///
    /// Returns an observation error from [`DqnPolicy::q_values`].
    pub fn q_values(&self, observation: &[f32]) -> Result<Vec<f32>, DqnError> {
        self.policy().q_values(observation)
    }

    /// Select the current greedy action without exploration.
    ///
    /// # Errors
    ///
    /// Returns an observation error from [`DqnPolicy::greedy_action`].
    pub fn greedy_action(&self, observation: &[f32]) -> Result<usize, DqnError> {
        self.policy().greedy_action(observation)
    }

    /// Select the current highest-valued legal action without exploration.
    ///
    /// # Errors
    ///
    /// Returns observation or action-mask validation errors.
    pub fn greedy_action_masked(
        &self,
        observation: &[f32],
        action_mask: &[bool],
    ) -> Result<usize, DqnError> {
        self.policy().greedy_action_masked(observation, action_mask)
    }

    /// Number of scalar trainable parameters in the online network.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.online_network.num_params()
    }

    /// Save the current online policy to the run's root `best.mpk` path.
    ///
    /// # Errors
    ///
    /// Returns the same checkpoint error as [`DqnPolicy::save_best`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_best(&self, paths: &RunPaths) -> Result<PathBuf, DqnError> {
        self.policy().save_best(paths)
    }

    /// Save the current online policy to `checkpoints/latest.mpk`.
    ///
    /// # Errors
    ///
    /// Returns the same checkpoint error as [`DqnPolicy::save_latest`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_latest(&self, paths: &RunPaths) -> Result<PathBuf, DqnError> {
        self.policy().save_latest(paths)
    }

    /// Borrow the validated hyperparameters used by this learner.
    #[must_use]
    pub const fn config(&self) -> &DqnConfig {
        &self.config
    }

    /// Runtime observation width accepted by the learner.
    #[must_use]
    pub const fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    /// Runtime number of discrete actions.
    #[must_use]
    pub const fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Number of environment transitions observed so far.
    #[must_use]
    pub const fn global_steps(&self) -> u64 {
        self.global_steps
    }

    /// Number of optimizer updates applied so far.
    #[must_use]
    pub const fn optimizer_steps(&self) -> u64 {
        self.optimizer_steps
    }

    /// Current number of transitions retained by replay storage.
    #[must_use]
    pub const fn replay_len(&self) -> usize {
        self.replay.len()
    }

    /// Current linearly decayed epsilon value.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        let progress =
            (self.global_steps as f64 / self.config.epsilon_decay_steps as f64).clamp(0.0, 1.0);
        self.config
            .epsilon_start
            .mul_add(1.0 - progress, self.config.epsilon_end * progress)
    }

    /// Select a discrete action using the dedicated deterministic action RNG
    /// and the configured epsilon schedule.
    ///
    /// # Errors
    ///
    /// Returns an observation error when `observation` has the wrong width.
    pub fn select_action(&mut self, observation: &[f32]) -> Result<DqnActionSelection, DqnError> {
        self.select_action_masked(observation, &vec![true; self.action_dim])
    }

    /// Select an action from the legal subset using epsilon-greedy exploration.
    ///
    /// # Errors
    ///
    /// Returns an observation or action-mask validation error.
    pub fn select_action_masked(
        &mut self,
        observation: &[f32],
        action_mask: &[bool],
    ) -> Result<DqnActionSelection, DqnError> {
        validate_observation(observation, self.observation_dim)?;
        let legal_actions = legal_action_indices(action_mask, self.action_dim)?;
        let epsilon = self.epsilon();
        let explored = self.action_rng.f64() < epsilon;
        let action_index = if explored {
            legal_actions
                .get(self.action_rng.usize_below(legal_actions.len()))
                .copied()
                .ok_or(DqnError::NoLegalActions)?
        } else {
            self.greedy_action_masked(observation, action_mask)?
        };

        Ok(DqnActionSelection {
            action_index,
            explored,
            epsilon,
        })
    }

    /// Store one encoded environment transition and optimize when replay is
    /// ready. Truncations retain value bootstrapping; only natural terminal
    /// transitions mask the next-state target.
    ///
    /// # Errors
    ///
    /// Returns an error for observation-width mismatches, invalid action
    /// indices, non-finite rewards, or tensor conversion failures.
    pub fn observe(
        &mut self,
        observation: &[f32],
        action_index: usize,
        reward: f64,
        next_observation: &[f32],
        status: EpisodeStatus,
    ) -> Result<Option<DqnUpdate>, DqnError> {
        self.observe_masked(
            observation,
            action_index,
            reward,
            next_observation,
            &vec![true; self.action_dim],
            status,
        )
    }

    /// Store a transition with its next state's legal-action mask.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid observations, actions, masks, rewards, or
    /// tensor conversions.
    pub fn observe_masked(
        &mut self,
        observation: &[f32],
        action_index: usize,
        reward: f64,
        next_observation: &[f32],
        next_action_mask: &[bool],
        status: EpisodeStatus,
    ) -> Result<Option<DqnUpdate>, DqnError> {
        self.observe_masked_with_bootstrap_multiplier(
            observation,
            action_index,
            reward,
            next_observation,
            next_action_mask,
            status,
            1.0,
        )
    }

    /// Store a masked transition with an explicit next-state value multiplier.
    ///
    /// Use `-1.0` when a zero-sum transition changes the acting perspective,
    /// and `1.0` when the next observation keeps the same objective.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid transition data or a non-finite bootstrap
    /// multiplier.
    pub fn observe_masked_with_bootstrap_multiplier(
        &mut self,
        observation: &[f32],
        action_index: usize,
        reward: f64,
        next_observation: &[f32],
        next_action_mask: &[bool],
        status: EpisodeStatus,
        bootstrap_multiplier: f32,
    ) -> Result<Option<DqnUpdate>, DqnError> {
        validate_observation(observation, self.observation_dim)?;
        validate_observation(next_observation, self.observation_dim)?;
        if status.is_terminal() {
            if next_action_mask.len() != self.action_dim {
                return Err(DqnError::ActionMaskDimension {
                    expected: self.action_dim,
                    actual: next_action_mask.len(),
                });
            }
        } else {
            validate_action_mask(next_action_mask, self.action_dim)?;
        }
        if action_index >= self.action_dim {
            return Err(DqnError::ActionOutOfBounds {
                action_index,
                action_dim: self.action_dim,
            });
        }
        if !reward.is_finite() || !(reward as f32).is_finite() {
            return Err(DqnError::NonFiniteReward);
        }
        if !bootstrap_multiplier.is_finite() {
            return Err(DqnError::NonFiniteBootstrapMultiplier);
        }

        self.replay.push(Experience {
            observation: observation.to_vec(),
            action_index,
            reward,
            next_observation: next_observation.to_vec(),
            next_action_mask: next_action_mask.to_vec(),
            bootstrap_multiplier,
            status,
        });
        self.global_steps += 1;

        if self.replay.len() < self.config.min_replay_size {
            return Ok(None);
        }

        self.train_step().map(Some)
    }

    /// Sample replay and apply one Huber TD optimizer update.
    fn train_step(&mut self) -> Result<DqnUpdate, DqnError> {
        let batch = self
            .replay
            .sample(self.config.batch_size, &mut self.replay_rng);
        let batch_size = batch.len();
        let observations: Vec<_> = batch.iter().map(|item| item.observation.clone()).collect();
        let next_observations: Vec<_> = batch
            .iter()
            .map(|item| item.next_observation.clone())
            .collect();
        let rewards: Vec<f32> = batch.iter().map(|item| item.reward as f32).collect();
        let masks: Vec<f32> = batch
            .iter()
            .map(|item| item.status.bootstrap_mask() as f32)
            .collect();
        let bootstrap_multipliers: Vec<f32> =
            batch.iter().map(|item| item.bootstrap_multiplier).collect();
        let action_indices: Vec<i32> = batch.iter().map(|item| item.action_index as i32).collect();
        let probe_observation = observations
            .first()
            .expect("validated nonzero batch size produces a probe")
            .clone();
        let before_q_values = self.q_values(&probe_observation)?;

        let rewards_tensor =
            Tensor::<TrainingBackend, 1>::from_floats(rewards.as_slice(), &self.device);
        let masks_tensor =
            Tensor::<TrainingBackend, 1>::from_floats(masks.as_slice(), &self.device);
        let bootstrap_multipliers_tensor = Tensor::<TrainingBackend, 1>::from_floats(
            bootstrap_multipliers.as_slice(),
            &self.device,
        );
        let next_q_values = self.target_network.forward(encode_batch_inference(
            &next_observations,
            self.observation_dim,
        ));
        // Apply every stored mask before the Bellman maximum so illegal moves
        // cannot inflate an otherwise valid predecessor state's target.
        let max_next_q = masked_bootstrap_values(&batch, next_q_values, self.action_dim)?;
        let max_next_q =
            Tensor::<TrainingBackend, 1>::from_floats(max_next_q.as_slice(), &self.device);
        let targets = rewards_tensor
            + masks_tensor * bootstrap_multipliers_tensor * max_next_q * self.config.gamma;

        let q_values = self.online_network.forward(encode_batch_training(
            &observations,
            self.observation_dim,
            self.device,
        ));
        let action_indices_tensor =
            Tensor::<TrainingBackend, 1, Int>::from_ints(action_indices.as_slice(), &self.device);
        let selected_q_values = q_values
            .gather(1, action_indices_tensor.reshape([batch_size, 1]))
            .squeeze_dim::<1>(1);
        let loss = HuberLossConfig::new(self.config.huber_delta)
            .init()
            .forward(selected_q_values, targets.detach(), Reduction::Mean);
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let gradients = GradientsParams::from_grads(loss.backward(), &self.online_network);
        self.online_network = self.optimizer.step(
            self.config.learning_rate,
            self.online_network.clone(),
            gradients,
        );
        self.optimizer_steps += 1;

        let target_updated = self
            .optimizer_steps
            .is_multiple_of(self.config.target_update_interval as u64);
        if target_updated {
            self.target_network = self.online_network.valid();
        }

        let after_q_values = self.q_values(&probe_observation)?;
        let policy_output_delta_l1 = before_q_values
            .iter()
            .zip(after_q_values.iter())
            .map(|(before, after)| f64::from((after - before).abs()))
            .sum();
        let terminated_samples = batch
            .iter()
            .filter(|item| item.status.is_terminal())
            .count();
        let truncated_samples = batch
            .iter()
            .filter(|item| item.status.is_truncated())
            .count();
        let bootstrapped_samples = batch_size - terminated_samples;

        Ok(DqnUpdate {
            optimizer_step: self.optimizer_steps,
            loss: loss_value,
            parameter_count: self.parameter_count(),
            policy_output_delta_l1,
            target_updated,
            bootstrapped_samples,
            terminated_samples,
            truncated_samples,
        })
    }
}

/// Convert a target-network batch into one legal Bellman maximum per sample.
fn masked_bootstrap_values(
    batch: &[Experience],
    next_q_values: Tensor<InferenceBackend, 2>,
    action_dim: usize,
) -> Result<Vec<f32>, DqnError> {
    let flat_next_q = next_q_values
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| DqnError::TensorConversion(error.to_string()))?;
    batch
        .iter()
        .zip(flat_next_q.chunks_exact(action_dim))
        .map(|(experience, q_values)| {
            if experience.status.is_terminal() {
                return Ok(0.0);
            }
            let index = greedy_legal_action(q_values, &experience.next_action_mask)?;
            q_values
                .get(index)
                .copied()
                .ok_or(DqnError::ActionOutOfBounds {
                    action_index: index,
                    action_dim: q_values.len(),
                })
        })
        .collect()
}

/// Errors produced by the reusable DQN core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DqnError {
    /// A runtime dimension or hyperparameter is invalid.
    InvalidConfig {
        /// Configuration field name.
        field: &'static str,

        /// Human-readable constraint.
        reason: &'static str,
    },

    /// Encoded observation width does not match the network input width.
    ObservationDimension {
        /// Expected encoded width.
        expected: usize,

        /// Received encoded width.
        actual: usize,
    },

    /// Discrete action index is outside the configured action space.
    ActionOutOfBounds {
        /// Invalid action index.
        action_index: usize,

        /// Configured action count.
        action_dim: usize,
    },

    /// Action-mask width does not match the configured action space.
    ActionMaskDimension {
        /// Expected action count.
        expected: usize,

        /// Received mask width.
        actual: usize,
    },

    /// An action mask marks every action as illegal.
    NoLegalActions,

    /// Burn tensor data could not be converted to host `f32` values.
    TensorConversion(String),

    /// A transition reward was NaN or infinite.
    NonFiniteReward,

    /// A transition bootstrap multiplier was NaN or infinite.
    NonFiniteBootstrapMultiplier,

    /// Burn policy recording failed with operation and path context.
    Checkpoint(CheckpointError),
}

impl DqnError {
    /// Build a static configuration validation error.
    const fn invalid_config(field: &'static str, reason: &'static str) -> Self {
        Self::InvalidConfig { field, reason }
    }
}

impl fmt::Display for DqnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(formatter, "invalid DQN config `{field}`: {reason}")
            }
            Self::ObservationDimension { expected, actual } => write!(
                formatter,
                "DQN observation width mismatch: expected {expected}, got {actual}"
            ),
            Self::ActionOutOfBounds {
                action_index,
                action_dim,
            } => write!(
                formatter,
                "DQN action index {action_index} is outside 0..{action_dim}"
            ),
            Self::ActionMaskDimension { expected, actual } => write!(
                formatter,
                "DQN action-mask width mismatch: expected {expected}, got {actual}"
            ),
            Self::NoLegalActions => formatter.write_str("DQN action mask has no legal actions"),
            Self::TensorConversion(message) => {
                write!(formatter, "DQN tensor conversion failed: {message}")
            }
            Self::NonFiniteReward => formatter.write_str("DQN transition reward must be finite"),
            Self::NonFiniteBootstrapMultiplier => {
                formatter.write_str("DQN bootstrap multiplier must be finite")
            }
            Self::Checkpoint(error) => error.fmt(formatter),
        }
    }
}

impl Error for DqnError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            _ => None,
        }
    }
}

/// Validate runtime network dimensions independently of optimizer settings.
fn validate_architecture(
    observation_dim: usize,
    action_dim: usize,
    hidden_sizes: &[usize],
) -> Result<(), DqnError> {
    if observation_dim == 0 {
        return Err(DqnError::invalid_config(
            "observation_dim",
            "must be greater than zero",
        ));
    }
    if action_dim == 0 {
        return Err(DqnError::invalid_config(
            "action_dim",
            "must be greater than zero",
        ));
    }
    if action_dim > i32::MAX as usize {
        return Err(DqnError::invalid_config(
            "action_dim",
            "must fit the backend's i32 index tensor",
        ));
    }
    if hidden_sizes.contains(&0) {
        return Err(DqnError::invalid_config(
            "hidden_sizes",
            "all hidden widths must be greater than zero",
        ));
    }

    Ok(())
}

/// Validate one encoded observation against the runtime network width.
const fn validate_observation(observation: &[f32], expected: usize) -> Result<(), DqnError> {
    if observation.len() != expected {
        return Err(DqnError::ObservationDimension {
            expected,
            actual: observation.len(),
        });
    }

    Ok(())
}

/// Validate an environment-provided legal-action mask.
fn validate_action_mask(action_mask: &[bool], expected: usize) -> Result<(), DqnError> {
    if action_mask.len() != expected {
        return Err(DqnError::ActionMaskDimension {
            expected,
            actual: action_mask.len(),
        });
    }
    if !action_mask.iter().any(|is_legal| *is_legal) {
        return Err(DqnError::NoLegalActions);
    }
    Ok(())
}

/// Return the stable action indices admitted by a validated mask.
fn legal_action_indices(action_mask: &[bool], expected: usize) -> Result<Vec<usize>, DqnError> {
    validate_action_mask(action_mask, expected)?;
    Ok(action_mask
        .iter()
        .enumerate()
        .filter_map(|(index, is_legal)| is_legal.then_some(index))
        .collect())
}

/// Select the highest-valued legal action with lowest-index tie breaking.
fn greedy_legal_action(q_values: &[f32], action_mask: &[bool]) -> Result<usize, DqnError> {
    validate_action_mask(action_mask, q_values.len())?;
    q_values
        .iter()
        .zip(action_mask)
        .enumerate()
        .filter(|(_, (_, is_legal))| **is_legal)
        .max_by(
            |(left_index, (left_value, _)), (right_index, (right_value, _))| {
                left_value
                    .total_cmp(right_value)
                    .then_with(|| right_index.cmp(left_index))
            },
        )
        .map(|(index, _)| index)
        .ok_or(DqnError::NoLegalActions)
}

/// Flatten encoded observations into an inference-backend batch tensor.
fn encode_batch_inference(
    observations: &[Vec<f32>],
    observation_dim: usize,
) -> Tensor<InferenceBackend, 2> {
    let device = inference_device();
    let flat: Vec<f32> = observations.iter().flatten().copied().collect();
    Tensor::<InferenceBackend, 1>::from_floats(flat.as_slice(), &device)
        .reshape([observations.len(), observation_dim])
}

/// Flatten encoded observations into an autodiff-backend batch tensor.
fn encode_batch_training(
    observations: &[Vec<f32>],
    observation_dim: usize,
    device: super::backend::TrainingDevice,
) -> Tensor<TrainingBackend, 2> {
    let flat: Vec<f32> = observations.iter().flatten().copied().collect();
    Tensor::<TrainingBackend, 1>::from_floats(flat.as_slice(), &device)
        .reshape([observations.len(), observation_dim])
}

/// Run-oriented DQN trainer boundary type.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct DqnTrainer<E: Env> {
    /// Shared run configuration.
    pub run: RunConfig,

    /// DQN-specific configuration.
    pub dqn: DqnConfig,

    /// Retains the environment type without storing an environment instance.
    _env: PhantomData<E>,
}

#[cfg(not(target_arch = "wasm32"))]
impl<E: Env> DqnTrainer<E> {
    /// Create a DQN trainer boundary.
    #[must_use]
    pub const fn new(run: RunConfig, dqn: DqnConfig) -> Self {
        Self {
            run,
            dqn,
            _env: PhantomData,
        }
    }

    /// Resolve the run paths this trainer will use.
    #[must_use]
    pub fn run_paths(&self) -> RunPaths {
        self.run.paths()
    }

    /// Build a reusable learner for caller-provided encoded dimensions.
    ///
    /// Environment adapters remain responsible only for encoding their typed
    /// observations and discrete actions at this boundary.
    ///
    /// # Errors
    ///
    /// Returns the same dimension or hyperparameter error as
    /// [`DqnAgent::new`].
    pub fn build_agent(
        &self,
        observation_dim: usize,
        action_dim: usize,
        seeds: SeedConfig,
    ) -> Result<DqnAgent, DqnError> {
        DqnAgent::new(observation_dim, action_dim, self.dqn.clone(), seeds)
    }
}

/// Summary returned by DQN training/evaluation orchestration.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq)]
pub struct DqnReport {
    /// Run paths used for artifacts.
    pub paths: RunPaths,

    /// Number of environment steps consumed.
    pub global_steps: u64,

    /// Best deterministic eval reward if one has been selected.
    pub best_eval_reward: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::config::{AlgorithmKind, RunId};

    #[test]
    fn byte_loader_rejects_missing_bias_wrong_bias_shape_and_nonfinite_parameters() {
        let device = inference_device();
        for variant in 0..3 {
            let policy = DqnAgent::new(
                4,
                2,
                DqnConfig {
                    hidden_sizes: vec![8],
                    ..DqnConfig::default()
                },
                SeedConfig::from_root(42),
            )
            .expect("valid learner")
            .policy();
            let mut record = policy.network.into_record();
            let layer = record.layers.first_mut().expect("first layer");
            match variant {
                0 => layer.bias = None,
                1 => {
                    layer.bias = Some(burn::module::Param::from_tensor(Tensor::zeros(
                        [9],
                        &device,
                    )));
                }
                _ => {
                    layer.weight =
                        burn::module::Param::from_tensor(Tensor::full([4, 8], f32::NAN, &device));
                }
            }
            let bytes = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
                .record(record, ())
                .expect("encode invalid architecture");
            assert!(matches!(
                DqnPolicy::load_bytes(bytes, 4, 2, &[8]),
                Err(DqnError::InvalidConfig { .. })
            ));
        }
    }

    #[test]
    fn same_model_seed_initializes_the_same_policy() {
        let config = DqnConfig {
            hidden_sizes: vec![8],
            ..DqnConfig::default()
        };
        let seeds = SeedConfig::from_root(73);
        let observation = [0.25, -0.5, 0.75];

        let first = DqnAgent::new(3, 2, config.clone(), seeds).expect("valid first agent");
        let second = DqnAgent::new(3, 2, config, seeds).expect("valid second agent");

        assert_eq!(
            first.q_values(&observation).expect("first Q values"),
            second.q_values(&observation).expect("second Q values")
        );
    }

    #[test]
    fn optimizer_update_changes_policy_output_and_reports_parameters() {
        let config = DqnConfig {
            hidden_sizes: vec![8],
            gamma: 0.0,
            learning_rate: 5e-2,
            replay_capacity: 8,
            min_replay_size: 1,
            batch_size: 1,
            target_update_interval: 1,
            ..DqnConfig::default()
        };
        let mut agent =
            DqnAgent::new(3, 2, config, SeedConfig::from_root(91)).expect("valid agent");
        let observation = [1.0, -0.5, 0.25];
        let next_observation = [0.0, 0.0, 0.0];
        let before = agent
            .q_values(&observation)
            .expect("Q values before update");

        let update = agent
            .observe(
                &observation,
                0,
                10.0,
                &next_observation,
                EpisodeStatus::Terminated,
            )
            .expect("valid transition")
            .expect("replay is ready for an update");
        let after = agent.q_values(&observation).expect("Q values after update");

        assert_ne!(before, after);
        assert!(update.loss.is_finite() && update.loss > 0.0);
        assert!(update.policy_output_delta_l1 > 0.0);
        assert_eq!(update.parameter_count, agent.parameter_count());
        assert_eq!(update.optimizer_step, 1);
        assert!(update.target_updated);
        assert_eq!(update.terminated_samples, 1);
        assert_eq!(update.truncated_samples, 0);
        assert_eq!(update.bootstrapped_samples, 0);
    }

    #[test]
    #[expect(
        clippy::allow_attributes,
        reason = "the repository disallows synchronous filesystem helpers by default"
    )]
    #[allow(
        clippy::disallowed_methods,
        reason = "bounded synchronous filesystem setup keeps the recorder round-trip test dependency-free"
    )]
    fn policy_checkpoint_round_trip_preserves_q_values() {
        let config = DqnConfig {
            hidden_sizes: vec![8],
            ..DqnConfig::default()
        };
        let agent = DqnAgent::new(3, 2, config, SeedConfig::from_root(17)).expect("valid agent");
        let observation = [0.125, -0.25, 0.5];
        let expected = agent.q_values(&observation).expect("source Q values");
        let root =
            std::env::temp_dir().join(format!("bevy-gym-dqn-checkpoint-{}", std::process::id()));
        drop(std::fs::remove_dir_all(&root));
        let paths = RunConfig::new(
            "dqn-unit",
            AlgorithmKind::Dqn,
            RunId::new("checkpoint-round-trip").expect("valid run id"),
            &root,
        )
        .expect("valid run config")
        .paths();
        paths.create_new().expect("new run directories");

        agent
            .policy()
            .save_best(&paths)
            .expect("policy checkpoint saves");
        let loaded = DqnPolicy::load_best(&paths, 3, 2, &[8]).expect("policy checkpoint loads");

        assert!(paths.best_checkpoint.is_file());
        assert_eq!(
            expected,
            loaded.q_values(&observation).expect("loaded Q values")
        );

        drop(std::fs::remove_dir_all(root));
    }

    #[test]
    fn truncated_transition_keeps_the_bootstrap_target_enabled() {
        let config = DqnConfig {
            hidden_sizes: vec![8],
            replay_capacity: 8,
            min_replay_size: 1,
            batch_size: 1,
            ..DqnConfig::default()
        };
        let mut agent =
            DqnAgent::new(2, 2, config, SeedConfig::from_root(101)).expect("valid agent");

        let update = agent
            .observe(
                &[0.5, -0.25],
                1,
                0.0,
                &[0.25, 0.75],
                EpisodeStatus::Truncated,
            )
            .expect("valid transition")
            .expect("replay is ready");

        assert_eq!(update.terminated_samples, 0);
        assert_eq!(update.truncated_samples, 1);
        assert_eq!(update.bootstrapped_samples, 1);
    }

    #[test]
    fn zero_sum_transition_records_negative_opponent_bootstrap() {
        let config = DqnConfig {
            hidden_sizes: vec![8],
            replay_capacity: 8,
            min_replay_size: 2,
            batch_size: 1,
            ..DqnConfig::default()
        };
        let mut agent =
            DqnAgent::new(2, 2, config, SeedConfig::from_root(103)).expect("valid agent");

        let update = agent
            .observe_masked_with_bootstrap_multiplier(
                &[0.5, -0.25],
                1,
                0.0,
                &[0.25, 0.75],
                &[true, true],
                EpisodeStatus::Continuing,
                -1.0,
            )
            .expect("valid zero-sum transition");

        assert!(update.is_none());
        let multiplier = agent.replay.data[0].bootstrap_multiplier;
        assert!((multiplier - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn non_finite_bootstrap_multiplier_is_rejected() {
        let mut agent = DqnAgent::new(2, 2, DqnConfig::default(), SeedConfig::from_root(107))
            .expect("valid agent");

        let error = agent
            .observe_masked_with_bootstrap_multiplier(
                &[0.5, -0.25],
                1,
                0.0,
                &[0.25, 0.75],
                &[true, true],
                EpisodeStatus::Continuing,
                f32::NAN,
            )
            .expect_err("non-finite bootstrap multiplier must fail");

        assert!(matches!(error, DqnError::NonFiniteBootstrapMultiplier));
    }

    #[test]
    fn epsilon_greedy_actions_use_the_dedicated_action_seed() {
        let config = DqnConfig {
            hidden_sizes: vec![4],
            epsilon_start: 1.0,
            epsilon_end: 1.0,
            ..DqnConfig::default()
        };
        let seeds = SeedConfig::from_root(211);
        let mut first = DqnAgent::new(2, 3, config.clone(), seeds).expect("first agent");
        let mut second = DqnAgent::new(2, 3, config, seeds).expect("second agent");

        let first_actions: Vec<_> = (0..8)
            .map(|_| first.select_action(&[0.0, 0.0]).expect("first action"))
            .collect();
        let second_actions: Vec<_> = (0..8)
            .map(|_| second.select_action(&[0.0, 0.0]).expect("second action"))
            .collect();

        assert!(first_actions.iter().all(|selection| selection.explored));
        assert_eq!(first_actions, second_actions);
    }

    #[test]
    fn masked_epsilon_greedy_never_selects_an_illegal_action() {
        let config = DqnConfig {
            hidden_sizes: vec![4],
            epsilon_start: 1.0,
            epsilon_end: 1.0,
            ..DqnConfig::default()
        };
        let mut agent =
            DqnAgent::new(2, 4, config, SeedConfig::from_root(307)).expect("valid agent");

        let selections: Vec<_> = (0..64)
            .map(|_| {
                agent
                    .select_action_masked(&[0.0, 0.0], &[false, true, false, true])
                    .expect("masked action")
            })
            .collect();

        assert!(selections
            .iter()
            .all(|selection| matches!(selection.action_index, 1 | 3)));
    }

    #[test]
    fn action_masks_reject_wrong_width_and_empty_legal_sets() {
        let config = DqnConfig {
            hidden_sizes: vec![4],
            ..DqnConfig::default()
        };
        let mut agent =
            DqnAgent::new(2, 3, config, SeedConfig::from_root(311)).expect("valid agent");

        assert_eq!(
            agent.select_action_masked(&[0.0, 0.0], &[true, false]),
            Err(DqnError::ActionMaskDimension {
                expected: 3,
                actual: 2,
            })
        );
        assert_eq!(
            agent.select_action_masked(&[0.0, 0.0], &[false; 3]),
            Err(DqnError::NoLegalActions)
        );
    }
}
