//! Frozen joint inference with one local recurrent-state update per action.
use super::super::{
    host_log_probability, inference_device, memory_to_state, scale_action, state_to_memory,
    tensor_vec, validate_finite, validate_memory, validate_vector, CentralCritic, InferenceBackend,
    RecurrentAction, RecurrentMemory, RecurrentPpoError, RecurrentSampler,
};
use super::{actor::CombatActor, signal::CombatSignal, RecurrentCombatAction};
use burn::prelude::ElementConversion;
use burn::tensor::Tensor;
use std::num::NonZeroUsize;

/// Frozen decentralized actor with a separately callable centralized training critic.
#[derive(Debug, Clone)]
pub struct RecurrentCombatPolicy {
    /// One recurrent actor shared by the two action heads.
    pub(super) actor: CombatActor<InferenceBackend>,
    /// Training-only state-value network, never queried by action inference.
    pub(super) critic: CentralCritic<InferenceBackend>,
    /// Validated local observation width.
    pub(super) observation_dim: NonZeroUsize,
    /// Validated centralized training-state width.
    pub(super) global_state_dim: NonZeroUsize,
    /// Validated centralized value-vector size.
    pub(super) value_count: NonZeroUsize,
    /// Validated recurrent-state width.
    pub(super) hidden_size: NonZeroUsize,
    /// Continuous actuator lower bounds validated before construction.
    pub(super) action_low: Vec<f32>,
    /// Continuous actuator upper bounds validated before construction.
    pub(super) action_high: Vec<f32>,
    /// Finite lower exploration boundary.
    pub(super) log_std_min: f32,
    /// Finite upper exploration boundary.
    pub(super) log_std_max: f32,
    /// Positive tanh Jacobian floor.
    pub(super) log_probability_epsilon: f32,
}

/// Host distributions from one validated recurrent forward evaluation.
struct HostDistribution {
    /// Continuous pre-tanh means.
    mean: Vec<f32>,
    /// Continuous log standard deviations.
    log_std: Vec<f32>,
    /// Categorical probabilities in the fixed signal order.
    probabilities: [f32; 8],
    /// Categorical log probabilities before exponentiation can underflow.
    log_probabilities: [f32; 8],
    /// Recurrent state advanced exactly once.
    memory: RecurrentMemory,
}
impl RecurrentCombatPolicy {
    /// Start one independent agent episode with zero recurrent state.
    #[must_use]
    pub fn initial_memory(&self) -> RecurrentMemory {
        RecurrentMemory::zeros(self.hidden_size.get())
    }

    /// Read the categorical distribution from local observation and own memory only.
    ///
    /// # Errors
    /// Returns a shape, non-finite or tensor conversion error before choosing any action.
    pub fn signal_probabilities(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
    ) -> Result<[f32; 8], RecurrentPpoError> {
        Ok(self.distribution(observation, memory)?.probabilities)
    }

    /// Sample the joint choice without any continuous-action threshold for the signal.
    ///
    /// # Errors
    /// Returns a shape, non-finite or tensor conversion error.
    pub fn sample_action(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
        sampler: &mut RecurrentSampler,
    ) -> Result<RecurrentCombatAction, RecurrentPpoError> {
        let distribution = self.distribution(observation, memory)?;
        let pre_tanh = distribution
            .mean
            .iter()
            .zip(&distribution.log_std)
            .map(|(mean, log_std)| mean + log_std.exp() * sampler.normal())
            .collect::<Vec<_>>();
        // Normalize finite probability weights in f64 before drawing the categorical column.
        let sum = distribution
            .probabilities
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        let draw = sampler.unit_open() * sum;
        let mut cumulative = 0.0;
        let signal = CombatSignal::ALL
            .into_iter()
            .zip(distribution.probabilities)
            .find_map(|(signal, probability)| {
                cumulative += f64::from(probability);
                (draw < cumulative).then_some(signal)
            })
            .ok_or_else(|| {
                RecurrentPpoError::invalid_config(
                    "categorical distribution",
                    "sample must lie inside the finite probability mass",
                )
            })?;
        self.decision(distribution, Some(pre_tanh), signal)
    }

    /// Use the continuous mean and categorical mode; ties choose the first fixed column.
    ///
    /// # Errors
    /// Returns a shape, non-finite or tensor conversion error.
    pub fn mean_action(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
    ) -> Result<RecurrentCombatAction, RecurrentPpoError> {
        let distribution = self.distribution(observation, memory)?;
        let mut signal = CombatSignal::IdleReleased;
        let mut maximum = f32::NEG_INFINITY;
        for (candidate, probability) in CombatSignal::ALL
            .into_iter()
            .zip(distribution.probabilities)
        {
            if probability > maximum {
                maximum = probability;
                signal = candidate;
            }
        }
        self.decision(distribution, None, signal)
    }

    /// Evaluate one centralized value slot without exposing global state to the actor.
    ///
    /// # Errors
    /// Returns a shape, value-index or non-finite error.
    pub fn value(&self, state: &[f32], index: usize) -> Result<f32, RecurrentPpoError> {
        validate_vector(state, self.global_state_dim.get(), "global state")?;
        if index >= self.value_count.get() {
            return Err(RecurrentPpoError::ValueIndex {
                index,
                count: self.value_count.get(),
            });
        }
        let device = inference_device();
        let input = Tensor::<InferenceBackend, 1>::from_floats(state, &device)
            .reshape([1, self.global_state_dim.get()]);
        let value = self
            .critic
            .forward(input)
            .narrow(1, index, 1)
            .into_scalar()
            .elem::<f32>();
        validate_finite(&[value], "centralized value")?;
        Ok(value)
    }

    /// Construct and validate both log-probability contributions before exposing an action.
    fn decision(
        &self,
        distribution: HostDistribution,
        samples: Option<Vec<f32>>,
        signal: CombatSignal,
    ) -> Result<RecurrentCombatAction, RecurrentPpoError> {
        let values = samples.as_deref().unwrap_or(&distribution.mean);
        validate_finite(values, "pre-tanh action")?;
        let action = scale_action(values, &self.action_low, &self.action_high);
        validate_finite(&action, "continuous action")?;
        let gaussian = host_log_probability(
            &distribution.mean,
            &distribution.log_std,
            values,
            &self.action_low,
            &self.action_high,
            self.log_probability_epsilon,
        );
        let log_probability = gaussian
            + distribution
                .log_probabilities
                .get(signal.index())
                .copied()
                .ok_or_else(|| {
                    RecurrentPpoError::invalid_config(
                        "categorical signal",
                        "column must be in the fixed signal vocabulary",
                    )
                })?;
        validate_finite(&[gaussian, log_probability], "joint log probability")?;
        Ok(RecurrentCombatAction {
            continuous: RecurrentAction {
                action,
                pre_tanh_action: samples.unwrap_or(distribution.mean),
                log_probability: gaussian,
                next_memory: distribution.memory,
            },
            signal,
            log_probability,
        })
    }

    /// Validate boundary vectors, advance memory once and normalize the categorical logits.
    fn distribution(
        &self,
        observation: &[f32],
        memory: &RecurrentMemory,
    ) -> Result<HostDistribution, RecurrentPpoError> {
        validate_vector(observation, self.observation_dim.get(), "observation")?;
        validate_memory(memory, self.hidden_size.get())?;
        let device = inference_device();
        let input = Tensor::<InferenceBackend, 1>::from_floats(observation, &device).reshape([
            1,
            1,
            self.observation_dim.get(),
        ]);
        let state = memory_to_state(memory, &device, self.hidden_size.get());
        let distribution = self.actor.forward(
            input,
            Some(state),
            self.action_low.len(),
            self.log_std_min,
            self.log_std_max,
        );
        let mean = tensor_vec(distribution.mean)?;
        let log_std = tensor_vec(distribution.log_std)?;
        let logits = tensor_vec(distribution.logits)?;
        validate_finite(&mean, "Gaussian means")?;
        validate_finite(&log_std, "Gaussian log standard deviations")?;
        validate_vector(&logits, 8, "categorical logits")?;
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer = logits
            .iter()
            .map(|value| (f64::from(*value) - f64::from(maximum)).exp())
            .sum::<f64>()
            .ln();
        let mut probabilities = [0.0; 8];
        let mut log_probabilities = [0.0; 8];
        for ((probability, log_probability), logit) in probabilities
            .iter_mut()
            .zip(&mut log_probabilities)
            .zip(logits)
        {
            *log_probability = (f64::from(logit) - f64::from(maximum) - normalizer) as f32;
            *probability = log_probability.exp();
        }
        validate_finite(&log_probabilities, "categorical log probabilities")?;
        let memory = state_to_memory(distribution.memory)?;
        validate_memory(&memory, self.hidden_size.get())?;
        Ok(HostDistribution {
            mean,
            log_std,
            probabilities,
            log_probabilities,
            memory,
        })
    }
}

impl RecurrentCombatPolicy {
    /// Serialize both heads and the critic through the locked Burn 0.21 named record.
    ///
    /// # Errors
    /// Returns a checkpoint encoding error with the in-memory source marker.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RecurrentPpoError> {
        use super::super::{CheckpointError, CheckpointOperation};
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
        let networks = super::actor::CombatNetworks {
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

    /// Validate configuration and every tensor before activating a distinct joint checkpoint.
    ///
    /// # Errors
    /// Rejects malformed bytes, Gaussian-only records, incompatible shapes and non-finite tensors.
    pub fn load_bytes(
        bytes: Vec<u8>,
        observations: usize,
        global: usize,
        values: usize,
        low: &[f32],
        high: &[f32],
        config: &super::super::RecurrentPpoConfig,
    ) -> Result<Self, RecurrentPpoError> {
        use super::super::{CheckpointError, CheckpointOperation, MODEL_INITIALIZATION_LOCK};
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};
        config.validate(observations, global, values, low, high)?;
        let _guard = MODEL_INITIALIZATION_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let device = inference_device();
        let networks = super::actor::CombatNetworks {
            actor: CombatActor::new(
                observations,
                config.actor_hidden_size,
                low.len(),
                config.initial_log_std,
                &device,
            ),
            critic: CentralCritic::new(global, &config.critic_hidden_sizes, values, &device),
        };
        // The record's nested continuous and categorical fields distinguish this profile.
        let record = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .load(bytes, &device)
            .map_err(|error| {
                RecurrentPpoError::Checkpoint(CheckpointError::new(
                    CheckpointOperation::Load,
                    "<memory>",
                    error.to_string(),
                ))
            })?;
        super::checkpoint::validate(&record, observations, global, values, low.len(), config)?;
        let networks = networks.load_record(record);
        Ok(Self {
            actor: networks.actor,
            critic: networks.critic,
            observation_dim: super::positive_width(observations)?,
            global_state_dim: super::positive_width(global)?,
            value_count: super::positive_width(values)?,
            hidden_size: super::positive_width(config.actor_hidden_size)?,
            action_low: low.to_vec(),
            action_high: high.to_vec(),
            log_std_min: config.log_std_min,
            log_std_max: config.log_std_max,
            log_probability_epsilon: config.log_probability_epsilon,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{RecurrentPpoConfig, SeedConfig};
    use super::super::RecurrentCombatAgent;
    use super::*;
    /// Construct a frozen joint fixture before injecting simulated optimizer failures.
    fn fixture() -> RecurrentCombatPolicy {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 4,
            critic_hidden_sizes: vec![4],
            ..RecurrentPpoConfig::default()
        };
        RecurrentCombatAgent::new(2, 3, 1, &[-1.0], &[1.0], config, SeedConfig::from_root(41))
            .expect("actor")
            .policy()
    }
    /// Inject one simulated optimizer failure into valid parameter shapes.
    fn invalid_policy(case: usize) -> RecurrentCombatPolicy {
        let device = inference_device();
        let mut policy = fixture();
        match case {
            0 => {
                policy.actor.continuous.mean_head.bias = policy
                    .actor
                    .continuous
                    .mean_head
                    .bias
                    .map(|bias| bias.map(|_| Tensor::full([1], f32::NAN, &device)));
            }
            1 => {
                policy.actor.continuous.log_std = policy
                    .actor
                    .continuous
                    .log_std
                    .map(|_| Tensor::full([1], f32::NAN, &device));
            }
            2 => {
                policy.actor.signals.bias = policy
                    .actor
                    .signals
                    .bias
                    .map(|bias| bias.map(|_| Tensor::full([8], f32::NAN, &device)));
            }
            _ => {
                policy.actor.signals.bias = policy.actor.signals.bias.map(|bias| {
                    bias.map(|_| {
                        Tensor::from_floats(
                            [f32::MAX, -f32::MAX, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            &device,
                        )
                    })
                });
            }
        }
        policy
    }

    /// Nonfinite learned means, scales, logits and overflowing log probabilities fail visibly.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn invalid_learned_parameters_stop_before_sampling() {
        let mut sampler = RecurrentSampler::new(43);
        for (case, field) in [
            (0, "Gaussian means"),
            (1, "Gaussian log standard deviations"),
            (2, "categorical logits"),
            (3, "categorical log probabilities"),
        ] {
            let policy = invalid_policy(case);
            assert!(
                matches!(policy.sample_action(&[0.1,0.2],&policy.initial_memory(),&mut sampler),Err(RecurrentPpoError::NonFiniteValue {field:actual,..}) if actual==field)
            );
        }
        let policy = fixture();
        assert_eq!(
            policy
                .sample_action(&[0.1, 0.2], &policy.initial_memory(), &mut sampler)
                .expect("untouched sampler"),
            policy
                .sample_action(
                    &[0.1, 0.2],
                    &policy.initial_memory(),
                    &mut RecurrentSampler::new(43)
                )
                .expect("original sample")
        );
    }
    /// Decision validation rejects nonfinite samples, action scaling and density overflow.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn malformed_decision_values_and_critic_outputs_fail() {
        let mut policy = fixture();
        let distribution = || HostDistribution {
            mean: vec![0.0],
            log_std: vec![0.0],
            probabilities: [0.125; 8],
            log_probabilities: [0.125_f32.ln(); 8],
            memory: policy.initial_memory(),
        };
        for (value, field) in [
            (f32::NAN, "pre-tanh action"),
            (f32::MAX, "joint log probability"),
        ] {
            assert!(
                matches!(policy.decision(distribution(),Some(vec![value]),CombatSignal::IdleReleased),Err(RecurrentPpoError::NonFiniteValue {field:actual,..}) if actual==field)
            );
        }
        let device = inference_device();
        let layer = policy.critic.layers.last_mut().expect("nonempty critic");
        layer.bias = layer
            .bias
            .take()
            .map(|bias| bias.map(|_| Tensor::full([1], f32::NAN, &device)));
        assert!(matches!(
            policy.value(&[0.0; 3], 0),
            Err(RecurrentPpoError::NonFiniteValue {
                field: "centralized value",
                ..
            })
        ));
        policy.action_low = vec![-f32::MAX];
        policy.action_high = vec![f32::MAX];
        assert!(matches!(
            policy.mean_action(&[0.1, 0.2], &policy.initial_memory()),
            Err(RecurrentPpoError::NonFiniteValue {
                field: "continuous action",
                ..
            })
        ));
    }
}
