//! Reusable Burn-backed PPO core for bounded continuous actions.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Mutex;

use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::{Backend, ElementConversion};
use burn::tensor::Tensor;

use crate::{Env, EpisodeStatus};

use super::backend::{inference_device, training_device, InferenceBackend, TrainingBackend};
use super::checkpoint::CheckpointError;
use super::config::{RunConfig, RunPaths};
use super::rng::SeedConfig;

/// Serializes Flex seeding and lazy PPO parameter materialization.
static MODEL_INITIALIZATION_LOCK: Mutex<()> = Mutex::new(());

/// Configuration for the PPO trainer boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoConfig {
    /// Hidden widths used independently by actor and critic MLPs.
    pub hidden_sizes: Vec<usize>,

    /// Discount factor.
    pub gamma: f32,

    /// Generalized advantage estimation lambda.
    pub gae_lambda: f32,

    /// Number of rollout steps before an update.
    pub rollout_steps: usize,

    /// Minibatch size for PPO updates.
    pub minibatch_size: usize,

    /// Optimization epochs per rollout.
    pub epochs: usize,

    /// Adam learning rate.
    pub learning_rate: f64,

    /// Numerical-stability epsilon used by Adam.
    pub optimizer_epsilon: f32,

    /// Symmetric PPO probability-ratio clipping distance around one.
    pub clip_ratio: f32,

    /// Value loss coefficient in the combined objective.
    pub value_loss_coefficient: f32,

    /// Gaussian entropy coefficient in the combined objective.
    pub entropy_coefficient: f32,

    /// Minimum actor log standard deviation.
    pub log_std_min: f32,

    /// Maximum actor log standard deviation.
    pub log_std_max: f32,

    /// Epsilon inside the tanh Jacobian logarithm.
    pub log_probability_epsilon: f32,

    /// Epsilon used while normalizing rollout advantages.
    pub advantage_epsilon: f64,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            hidden_sizes: vec![64, 64],
            gamma: 0.99,
            gae_lambda: 0.95,
            rollout_steps: 128,
            minibatch_size: 64,
            epochs: 4,
            learning_rate: 3e-4,
            optimizer_epsilon: 1e-8,
            clip_ratio: 0.2,
            value_loss_coefficient: 0.5,
            entropy_coefficient: 0.01,
            log_std_min: -5.0,
            log_std_max: 2.0,
            log_probability_epsilon: 1e-6,
            advantage_epsilon: 1e-8,
        }
    }
}

impl PpoConfig {
    /// Validate environment-neutral PPO hyperparameters and runtime shapes.
    ///
    /// # Errors
    ///
    /// Returns [`PpoError::InvalidConfig`] when any value cannot support a
    /// bounded PPO update.
    pub fn validate(
        &self,
        observation_dim: usize,
        action_low: &[f32],
        action_high: &[f32],
    ) -> Result<(), PpoError> {
        validate_architecture(observation_dim, action_low, action_high, &self.hidden_sizes)?;
        if !self.gamma.is_finite() || !(0.0..=1.0).contains(&self.gamma) {
            return Err(PpoError::invalid_config(
                "gamma",
                "must be finite and in 0..=1",
            ));
        }
        if !self.gae_lambda.is_finite() || !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err(PpoError::invalid_config(
                "gae_lambda",
                "must be finite and in 0..=1",
            ));
        }
        if self.rollout_steps == 0 {
            return Err(PpoError::invalid_config(
                "rollout_steps",
                "must be greater than zero",
            ));
        }
        if self.minibatch_size == 0 || self.minibatch_size > self.rollout_steps {
            return Err(PpoError::invalid_config(
                "minibatch_size",
                "must be in 1..=rollout_steps",
            ));
        }
        if self.epochs == 0 {
            return Err(PpoError::invalid_config(
                "epochs",
                "must be greater than zero",
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(PpoError::invalid_config(
                "learning_rate",
                "must be finite and greater than zero",
            ));
        }
        if !self.optimizer_epsilon.is_finite() || self.optimizer_epsilon <= 0.0 {
            return Err(PpoError::invalid_config(
                "optimizer_epsilon",
                "must be finite and greater than zero",
            ));
        }
        if !self.clip_ratio.is_finite() || !(0.0..1.0).contains(&self.clip_ratio) {
            return Err(PpoError::invalid_config(
                "clip_ratio",
                "must be finite and in 0..1",
            ));
        }
        if !self.value_loss_coefficient.is_finite() || self.value_loss_coefficient < 0.0 {
            return Err(PpoError::invalid_config(
                "value_loss_coefficient",
                "must be finite and non-negative",
            ));
        }
        if !self.entropy_coefficient.is_finite() || self.entropy_coefficient < 0.0 {
            return Err(PpoError::invalid_config(
                "entropy_coefficient",
                "must be finite and non-negative",
            ));
        }
        if !self.log_std_min.is_finite()
            || !self.log_std_max.is_finite()
            || self.log_std_min >= self.log_std_max
        {
            return Err(PpoError::invalid_config(
                "log_std",
                "finite minimum must be less than finite maximum",
            ));
        }
        if !self.log_probability_epsilon.is_finite() || self.log_probability_epsilon <= 0.0 {
            return Err(PpoError::invalid_config(
                "log_probability_epsilon",
                "must be finite and greater than zero",
            ));
        }
        if !self.advantage_epsilon.is_finite() || self.advantage_epsilon <= 0.0 {
            return Err(PpoError::invalid_config(
                "advantage_epsilon",
                "must be finite and greater than zero",
            ));
        }

        Ok(())
    }
}

/// Runtime-shaped feed-forward network.
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    /// Ordered affine layers.
    layers: Vec<Linear<B>>,

    /// Hidden-layer nonlinearity.
    activation: Relu,
}

impl<B: Backend> Mlp<B> {
    /// Initialize one MLP from explicit widths.
    fn new(
        input_dim: usize,
        hidden_sizes: &[usize],
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        let widths: Vec<_> = std::iter::once(input_dim)
            .chain(hidden_sizes.iter().copied())
            .chain(std::iter::once(output_dim))
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

    /// Evaluate a batch through every affine and hidden activation.
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

/// Independent Gaussian actor and scalar critic networks.
#[derive(Module, Debug)]
struct ActorCritic<B: Backend> {
    /// Actor emitting means followed by log standard deviations.
    actor: Mlp<B>,

    /// Critic emitting one state value.
    critic: Mlp<B>,
}

impl<B: Backend> ActorCritic<B> {
    /// Initialize runtime-shaped actor and critic networks.
    fn new(
        observation_dim: usize,
        action_dim: usize,
        hidden_sizes: &[usize],
        device: &B::Device,
    ) -> Self {
        Self {
            actor: Mlp::new(observation_dim, hidden_sizes, action_dim * 2, device),
            critic: Mlp::new(observation_dim, hidden_sizes, 1, device),
        }
    }

    /// Evaluate Gaussian parameters with bounded log standard deviations.
    fn distribution(
        &self,
        observations: Tensor<B, 2>,
        action_dim: usize,
        log_std_min: f32,
        log_std_max: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let parameters = self.actor.forward(observations);
        let mean = parameters.clone().narrow(1, 0, action_dim);
        let log_std = parameters
            .narrow(1, action_dim, action_dim)
            .clamp(log_std_min, log_std_max);
        (mean, log_std)
    }

    /// Evaluate scalar state values.
    fn values(&self, observations: Tensor<B, 2>) -> Tensor<B, 1> {
        self.critic.forward(observations).squeeze_dim::<1>(1)
    }
}

/// Host-side Gaussian parameters for one observation.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoDistribution {
    /// Unsquashed Gaussian means.
    pub mean: Vec<f32>,

    /// Clamped Gaussian log standard deviations.
    pub log_std: Vec<f32>,
}

/// One on-policy transition retained until a PPO update.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoRolloutStep {
    /// Stable environment lane identifier for interleaved rollouts.
    pub env_id: usize,

    /// Encoded observation before the action.
    pub observation: Vec<f32>,

    /// Scaled action sent to the environment.
    pub action: Vec<f32>,

    /// Gaussian sample before tanh squashing.
    pub pre_tanh_action: Vec<f32>,

    /// Scalar reward received from the environment.
    pub reward: f64,

    /// Critic value recorded before the action.
    pub value: f64,

    /// Corrected transformed-action log probability under the behavior policy.
    pub log_probability: f64,

    /// Critic value for the actual next observation.
    pub next_value: f64,

    /// Episode boundary semantics for bootstrapping and trace continuation.
    pub status: EpisodeStatus,
}

/// GAE and return target corresponding to one rollout step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpoAdvantage {
    /// Generalized advantage estimate.
    pub advantage: f64,

    /// Critic regression target `value + advantage`.
    pub return_value: f64,
}

/// Compute GAE for potentially interleaved environment lanes.
///
/// Natural termination masks the next value. Truncation retains that value in
/// the TD residual but stops the trace before a reset observation can leak into
/// the previous episode.
///
/// # Errors
///
/// Returns [`PpoError::InvalidConfig`] when `gamma` or `gae_lambda` is not
/// finite and contained in `0..=1`.
pub fn compute_gae(
    rollout: &[PpoRolloutStep],
    gamma: f64,
    gae_lambda: f64,
) -> Result<Vec<PpoAdvantage>, PpoError> {
    if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
        return Err(PpoError::invalid_config(
            "gamma",
            "must be finite and in 0..=1",
        ));
    }
    if !gae_lambda.is_finite() || !(0.0..=1.0).contains(&gae_lambda) {
        return Err(PpoError::invalid_config(
            "gae_lambda",
            "must be finite and in 0..=1",
        ));
    }

    let mut output = Vec::with_capacity(rollout.len());
    let mut next_advantage_by_environment = HashMap::<usize, f64>::new();

    for step in rollout.iter().rev() {
        let next_advantage = next_advantage_by_environment
            .get(&step.env_id)
            .copied()
            .unwrap_or(0.0);
        let bootstrap_mask = step.status.bootstrap_mask();
        let trace_mask = if step.status == EpisodeStatus::Continuing {
            1.0
        } else {
            0.0
        };
        let delta = (gamma * bootstrap_mask).mul_add(step.next_value, step.reward) - step.value;
        let advantage = (gamma * gae_lambda * trace_mask).mul_add(next_advantage, delta);
        output.push(PpoAdvantage {
            advantage,
            return_value: step.value + advantage,
        });
        next_advantage_by_environment.insert(step.env_id, advantage);
    }
    output.reverse();

    Ok(output)
}

/// Deterministic inference and checkpoint surface for a PPO policy.
#[derive(Debug, Clone)]
pub struct PpoPolicy {
    /// Inference-backend actor and critic.
    network: ActorCritic<InferenceBackend>,

    /// Encoded observation width.
    observation_dim: usize,

    /// Inclusive environment action lower bounds.
    action_low: Vec<f32>,

    /// Inclusive environment action upper bounds.
    action_high: Vec<f32>,

    /// Minimum actor log standard deviation.
    log_std_min: f32,

    /// Maximum actor log standard deviation.
    log_std_max: f32,

    /// Numerical epsilon used by the tanh Jacobian correction.
    log_probability_epsilon: f32,
}

impl PpoPolicy {
    /// Number of bounded continuous action dimensions.
    #[must_use]
    pub const fn action_dim(&self) -> usize {
        self.action_low.len()
    }

    /// Encoded observation width expected by the actor and critic.
    #[must_use]
    pub const fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    /// Evaluate unsquashed Gaussian parameters for one observation.
    ///
    /// # Errors
    ///
    /// Returns an observation-width or tensor-conversion error.
    pub fn distribution(&self, observation: &[f32]) -> Result<PpoDistribution, PpoError> {
        validate_observation(observation, self.observation_dim)?;
        let device = inference_device();
        let input = Tensor::<InferenceBackend, 1>::from_floats(observation, &device)
            .reshape([1, self.observation_dim]);
        let (mean, log_std) =
            self.network
                .distribution(input, self.action_dim(), self.log_std_min, self.log_std_max);
        let mean = mean
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| PpoError::TensorConversion(error.to_string()))?;
        let log_std = log_std
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| PpoError::TensorConversion(error.to_string()))?;
        Ok(PpoDistribution { mean, log_std })
    }

    /// Evaluate the scalar critic value for one observation.
    ///
    /// # Errors
    ///
    /// Returns an observation-width error.
    pub fn value(&self, observation: &[f32]) -> Result<f64, PpoError> {
        validate_observation(observation, self.observation_dim)?;
        let device = inference_device();
        let input = Tensor::<InferenceBackend, 1>::from_floats(observation, &device)
            .reshape([1, self.observation_dim]);
        Ok(self.network.values(input).into_scalar().elem::<f64>())
    }

    /// Return the deterministic environment action obtained by squashing and
    /// scaling the Gaussian mean.
    ///
    /// # Errors
    ///
    /// Returns an observation or tensor-conversion error.
    pub fn mean_action(&self, observation: &[f32]) -> Result<Vec<f32>, PpoError> {
        let distribution = self.distribution(observation)?;
        let squashed: Vec<_> = distribution.mean.iter().map(|value| value.tanh()).collect();
        self.scale_squashed_action(&squashed)
    }

    /// Evaluate the transformed environment-action log probability of one
    /// pre-tanh sample. This includes both the tanh Jacobian and affine action
    /// scaling corrections.
    ///
    /// # Errors
    ///
    /// Returns an observation, action-dimension, non-finite, or tensor error.
    pub fn log_probability(
        &self,
        observation: &[f32],
        pre_tanh_action: &[f32],
    ) -> Result<f64, PpoError> {
        validate_action(pre_tanh_action, self.action_dim(), false)?;
        let distribution = self.distribution(observation)?;
        Ok(squashed_gaussian_log_probability(
            &distribution.mean,
            &distribution.log_std,
            pre_tanh_action,
            &self.action_low,
            &self.action_high,
            self.log_probability_epsilon,
        ))
    }

    /// Scale one tanh-space action from `[-1, 1]` into environment bounds.
    ///
    /// # Errors
    ///
    /// Returns a dimension or non-finite/action-range error for invalid input.
    pub fn scale_squashed_action(&self, squashed: &[f32]) -> Result<Vec<f32>, PpoError> {
        validate_action(squashed, self.action_dim(), true)?;
        Ok(squashed
            .iter()
            .zip(self.action_low.iter().zip(&self.action_high))
            .map(|(unit, (low, high))| {
                let scale = (high - low) * 0.5;
                let bias = (high + low) * 0.5;
                unit.mul_add(scale, bias)
            })
            .collect())
    }

    /// Number of scalar trainable actor and critic parameters.
    #[must_use]
    pub fn parameter_count(&self) -> usize {
        self.network.num_params()
    }
}

/// Stateful bounded-continuous PPO learner.
pub struct PpoAgent {
    /// Autodiff actor and critic updated by Adam.
    network: ActorCritic<TrainingBackend>,

    /// Encoded observation width.
    observation_dim: usize,

    /// Environment action lower bounds.
    action_low: Vec<f32>,

    /// Environment action upper bounds.
    action_high: Vec<f32>,

    /// Validated PPO hyperparameters.
    config: PpoConfig,
}

impl fmt::Debug for PpoAgent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PpoAgent")
            .field("observation_dim", &self.observation_dim)
            .field("action_dim", &self.action_low.len())
            .field("parameter_count", &self.network.num_params())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl PpoAgent {
    /// Initialize a runtime-shaped Gaussian actor and critic under the model
    /// seed stream.
    ///
    /// # Errors
    ///
    /// Returns [`PpoError::InvalidConfig`] for unusable dimensions, bounds, or
    /// hyperparameters.
    pub fn new(
        observation_dim: usize,
        action_low: &[f32],
        action_high: &[f32],
        config: PpoConfig,
        seeds: SeedConfig,
    ) -> Result<Self, PpoError> {
        config.validate(observation_dim, action_low, action_high)?;
        let _initialization_guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = training_device();
        TrainingBackend::seed(&device, seeds.model);
        let network = ActorCritic::new(
            observation_dim,
            action_low.len(),
            &config.hidden_sizes,
            &device,
        );
        let zero_observation = Tensor::zeros([1, observation_dim], &device);
        let (mean, log_std) = network.distribution(
            zero_observation.clone(),
            action_low.len(),
            config.log_std_min,
            config.log_std_max,
        );
        let _actor_materialized = (mean + log_std).into_data();
        let _critic_materialized = network.values(zero_observation).into_data();

        Ok(Self {
            network,
            observation_dim,
            action_low: action_low.to_vec(),
            action_high: action_high.to_vec(),
            config,
        })
    }

    /// Snapshot the current actor and critic for deterministic inference.
    #[must_use]
    pub fn policy(&self) -> PpoPolicy {
        PpoPolicy {
            network: self.network.valid(),
            observation_dim: self.observation_dim,
            action_low: self.action_low.clone(),
            action_high: self.action_high.clone(),
            log_std_min: self.config.log_std_min,
            log_std_max: self.config.log_std_max,
            log_probability_epsilon: self.config.log_probability_epsilon,
        }
    }
}

/// Errors produced by the bounded-continuous PPO core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PpoError {
    /// A runtime dimension, bound, or hyperparameter is invalid.
    InvalidConfig {
        /// Configuration field name.
        field: &'static str,

        /// Human-readable constraint.
        reason: &'static str,
    },

    /// One encoded vector has the wrong width.
    DimensionMismatch {
        /// Vector role.
        field: &'static str,

        /// Expected width.
        expected: usize,

        /// Received width.
        actual: usize,
    },

    /// A provided observation or action contains NaN or infinity.
    NonFiniteValue {
        /// Vector role.
        field: &'static str,

        /// Invalid element position.
        dimension: usize,
    },

    /// A purported tanh-space action lies outside `[-1, 1]`.
    SquashedActionOutOfRange {
        /// Invalid element position.
        dimension: usize,
    },

    /// Burn tensor data could not convert to host values.
    TensorConversion(String),

    /// Policy checkpoint recording failed.
    Checkpoint(CheckpointError),
}

impl PpoError {
    /// Build a static config validation error.
    const fn invalid_config(field: &'static str, reason: &'static str) -> Self {
        Self::InvalidConfig { field, reason }
    }
}

impl fmt::Display for PpoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(formatter, "invalid PPO config `{field}`: {reason}")
            }
            Self::DimensionMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "PPO {field} width mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteValue { field, dimension } => {
                write!(
                    formatter,
                    "PPO {field} at dimension {dimension} is not finite"
                )
            }
            Self::SquashedActionOutOfRange { dimension } => write!(
                formatter,
                "PPO squashed action at dimension {dimension} is outside [-1, 1]"
            ),
            Self::TensorConversion(message) => {
                write!(formatter, "PPO tensor conversion failed: {message}")
            }
            Self::Checkpoint(error) => error.fmt(formatter),
        }
    }
}

impl Error for PpoError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Checkpoint(error) => Some(error),
            _ => None,
        }
    }
}

/// Validate runtime dimensions and finite ordered action bounds.
fn validate_architecture(
    observation_dim: usize,
    action_low: &[f32],
    action_high: &[f32],
    hidden_sizes: &[usize],
) -> Result<(), PpoError> {
    if observation_dim == 0 {
        return Err(PpoError::invalid_config(
            "observation_dim",
            "must be greater than zero",
        ));
    }
    if action_low.is_empty() || action_low.len().checked_mul(2).is_none() {
        return Err(PpoError::invalid_config(
            "action_bounds",
            "must contain a representable nonzero action dimension",
        ));
    }
    if action_low.len() != action_high.len() {
        return Err(PpoError::invalid_config(
            "action_bounds",
            "lower and upper bounds must have equal lengths",
        ));
    }
    if hidden_sizes.contains(&0) {
        return Err(PpoError::invalid_config(
            "hidden_sizes",
            "all hidden widths must be greater than zero",
        ));
    }
    for (dimension, (low, high)) in action_low.iter().zip(action_high).enumerate() {
        if !low.is_finite() || !high.is_finite() {
            return Err(PpoError::NonFiniteValue {
                field: "action bound",
                dimension,
            });
        }
        if low >= high {
            return Err(PpoError::invalid_config(
                "action_bounds",
                "every lower bound must be less than its upper bound",
            ));
        }
    }
    Ok(())
}

/// Validate one encoded vector and optional tanh-space range.
fn validate_action(action: &[f32], expected: usize, squashed: bool) -> Result<(), PpoError> {
    if action.len() != expected {
        return Err(PpoError::DimensionMismatch {
            field: "action",
            expected,
            actual: action.len(),
        });
    }
    for (dimension, value) in action.iter().enumerate() {
        if !value.is_finite() {
            return Err(PpoError::NonFiniteValue {
                field: "action",
                dimension,
            });
        }
        if squashed && !(-1.0..=1.0).contains(value) {
            return Err(PpoError::SquashedActionOutOfRange { dimension });
        }
    }
    Ok(())
}

/// Validate a finite encoded observation.
fn validate_observation(observation: &[f32], expected: usize) -> Result<(), PpoError> {
    if observation.len() != expected {
        return Err(PpoError::DimensionMismatch {
            field: "observation",
            expected,
            actual: observation.len(),
        });
    }
    if let Some((dimension, _)) = observation
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(PpoError::NonFiniteValue {
            field: "observation",
            dimension,
        });
    }
    Ok(())
}

/// Compute a diagonal Gaussian log probability after tanh and affine scaling.
fn squashed_gaussian_log_probability(
    mean: &[f32],
    log_std: &[f32],
    pre_tanh_action: &[f32],
    action_low: &[f32],
    action_high: &[f32],
    epsilon: f32,
) -> f64 {
    const LOG_TWO_PI: f64 = 1.837_877_066_409_345_3;
    mean.iter()
        .zip(log_std)
        .zip(pre_tanh_action)
        .zip(action_low.iter().zip(action_high))
        .map(|(((mean, log_std), sample), (low, high))| {
            let mean = f64::from(*mean);
            let log_std = f64::from(*log_std);
            let sample = f64::from(*sample);
            let standard_deviation = log_std.exp();
            let normalized = (sample - mean) / standard_deviation;
            let gaussian = (-0.5_f64).mul_add(normalized.mul_add(normalized, LOG_TWO_PI), -log_std);
            let squashed = sample.tanh();
            let jacobian = squashed.mul_add(-squashed, 1.0).max(f64::from(epsilon));
            let scale = f64::from((high - low) * 0.5);
            gaussian - jacobian.ln() - scale.ln()
        })
        .sum()
}

/// PPO trainer boundary type.
#[derive(Debug, Clone)]
pub struct PpoTrainer<E: Env> {
    /// Shared run configuration.
    pub run: RunConfig,

    /// PPO-specific configuration.
    pub ppo: PpoConfig,

    /// Retains the environment type without storing an environment instance.
    _env: PhantomData<E>,
}

impl<E: Env> PpoTrainer<E> {
    /// Create a PPO trainer boundary.
    #[must_use]
    pub const fn new(run: RunConfig, ppo: PpoConfig) -> Self {
        Self {
            run,
            ppo,
            _env: PhantomData,
        }
    }

    /// Resolve the run paths this trainer will use.
    #[must_use]
    pub fn run_paths(&self) -> RunPaths {
        self.run.paths()
    }
}

/// Summary returned by future PPO training/eval slices.
#[derive(Debug, Clone, PartialEq)]
pub struct PpoReport {
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

    #[test]
    fn squashed_actions_scale_to_asymmetric_runtime_bounds() {
        let agent = PpoAgent::new(
            3,
            &[-2.0, 10.0],
            &[4.0, 14.0],
            PpoConfig::default(),
            SeedConfig::from_root(31),
        )
        .expect("valid PPO agent");
        let policy = agent.policy();

        assert_eq!(
            policy
                .scale_squashed_action(&[-1.0, 1.0])
                .expect("valid endpoint action"),
            vec![-2.0, 14.0]
        );
        assert_eq!(
            policy
                .scale_squashed_action(&[0.0, 0.0])
                .expect("valid midpoint action"),
            vec![1.0, 12.0]
        );
    }

    #[test]
    fn corrected_log_probability_stays_finite_near_tanh_limits() {
        let agent = PpoAgent::new(
            2,
            &[-2.0, -0.5],
            &[3.0, 2.5],
            PpoConfig::default(),
            SeedConfig::from_root(37),
        )
        .expect("valid PPO agent");

        let log_probability = agent
            .policy()
            .log_probability(&[0.25, -0.75], &[10.0, -10.0])
            .expect("valid transformed log probability");

        assert!(log_probability.is_finite());
    }

    #[test]
    fn gae_bootstraps_truncation_but_not_natural_termination() {
        let step = |status| PpoRolloutStep {
            env_id: 0,
            observation: vec![0.0],
            action: vec![0.0],
            pre_tanh_action: vec![0.0],
            reward: 2.0,
            value: 1.0,
            log_probability: -0.5,
            next_value: 4.0,
            status,
        };

        let terminated = compute_gae(&[step(EpisodeStatus::Terminated)], 0.5, 1.0)
            .expect("valid terminated GAE");
        let truncated =
            compute_gae(&[step(EpisodeStatus::Truncated)], 0.5, 1.0).expect("valid truncated GAE");

        assert!((terminated[0].advantage - 1.0).abs() < f64::EPSILON);
        assert!((truncated[0].advantage - 3.0).abs() < f64::EPSILON);
        assert!((terminated[0].return_value - 2.0).abs() < f64::EPSILON);
        assert!((truncated[0].return_value - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn same_seed_produces_identical_deterministic_inference() {
        let config = PpoConfig {
            hidden_sizes: vec![8],
            ..PpoConfig::default()
        };
        let seeds = SeedConfig::from_root(43);
        let first =
            PpoAgent::new(3, &[-1.0], &[2.0], config.clone(), seeds).expect("first PPO agent");
        let second = PpoAgent::new(3, &[-1.0], &[2.0], config, seeds).expect("second PPO agent");
        let observation = [0.25, -0.5, 0.75];

        assert_eq!(
            first
                .policy()
                .mean_action(&observation)
                .expect("first mean action"),
            second
                .policy()
                .mean_action(&observation)
                .expect("second mean action")
        );
        assert_eq!(
            first
                .policy()
                .value(&observation)
                .expect("first value")
                .to_bits(),
            second
                .policy()
                .value(&observation)
                .expect("second value")
                .to_bits()
        );
    }
}
