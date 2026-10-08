//! Validate the locked Burn 0.21 named `MessagePack` record before activation.
use super::{
    tensor_vec, InferenceBackend, RecurrentNetworksRecord, RecurrentPpoConfig, RecurrentPpoError,
};
use burn::module::Param;
use burn::nn::LinearRecord;
use burn::tensor::Tensor;

/// Check every actor and critic tensor against the requested architecture.
///
/// Import costs O(parameters) time and at most one tensor's host copy at once.
/// The caller retains the decoded record; validation does not copy the model.
pub(super) fn validate(
    record: &RecurrentNetworksRecord<InferenceBackend>,
    observations: usize,
    global: usize,
    values: usize,
    actions: usize,
    config: &RecurrentPpoConfig,
) -> Result<(), RecurrentPpoError> {
    // Validate all four LSTM gates before any tensor reaches a matrix product.
    let hidden = config.actor_hidden_size;
    let memory = &record.actor.memory;
    for gate in [
        &memory.input_gate,
        &memory.forget_gate,
        &memory.output_gate,
        &memory.cell_gate,
    ] {
        linear(&gate.input_transform, observations, hidden)?;
        linear(&gate.hidden_transform, hidden, hidden)?;
    }
    linear(&record.actor.mean_head, hidden, actions)?;
    parameter(&record.actor.log_std, [actions])?;

    // A record must supply exactly one layer for every declared critic edge.
    if record.critic.layers.len() != config.critic_hidden_sizes.len() + 1 {
        return Err(RecurrentPpoError::invalid_config(
            "checkpoint",
            "critic layer count differs from the declared architecture",
        ));
    }
    let inputs = std::iter::once(global).chain(config.critic_hidden_sizes.iter().copied());
    let outputs = config
        .critic_hidden_sizes
        .iter()
        .copied()
        .chain(std::iter::once(values));
    for (layer, (input, output)) in record.critic.layers.iter().zip(inputs.zip(outputs)) {
        linear(layer, input, output)?;
    }
    Ok(())
}

/// Require a biased affine layer with finite parameters of the declared shape.
fn linear(
    record: &LinearRecord<InferenceBackend>,
    input: usize,
    output: usize,
) -> Result<(), RecurrentPpoError> {
    let bias = record.bias.as_ref().ok_or_else(|| {
        RecurrentPpoError::invalid_config("checkpoint", "every layer must contain a bias")
    })?;
    parameter(&record.weight, [input, output])?;
    parameter(bias, [output])
}

/// Reject shape differences and nonfinite values before activating a parameter.
fn parameter<const D: usize>(
    parameter: &Param<Tensor<InferenceBackend, D>>,
    expected: [usize; D],
) -> Result<(), RecurrentPpoError> {
    let tensor = parameter.val();
    if tensor.dims() != expected {
        return Err(RecurrentPpoError::invalid_config(
            "checkpoint",
            "parameter shape differs from the declared architecture",
        ));
    }
    if !tensor_vec(tensor)?.iter().all(|value| value.is_finite()) {
        return Err(RecurrentPpoError::invalid_config(
            "checkpoint",
            "parameters must be finite",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::recurrent_ppo::{
        RecurrentNetworks, RecurrentPpoAgent, RecurrentPpoPolicy,
    };
    use crate::training::{inference_device, SeedConfig};
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder};

    /// Make a valid record through the same learner boundary as browser exports.
    fn record() -> (
        RecurrentNetworksRecord<InferenceBackend>,
        RecurrentPpoConfig,
    ) {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 4,
            critic_hidden_sizes: vec![4],
            ..RecurrentPpoConfig::default()
        };
        let policy = RecurrentPpoAgent::new(
            2,
            2,
            1,
            &[-1.0],
            &[1.0],
            config.clone(),
            SeedConfig::from_root(1),
        )
        .expect("valid learner")
        .policy();
        (
            RecurrentNetworks {
                actor: policy.actor,
                critic: policy.critic,
            }
            .into_record(),
            config,
        )
    }

    /// Encode deliberately invalid data before exercising the public import seam.
    fn reject(record: RecurrentNetworksRecord<InferenceBackend>, config: &RecurrentPpoConfig) {
        let bytes = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
            .record(record, ())
            .expect("serializable record");
        let error = RecurrentPpoPolicy::load_bytes(bytes, 2, 2, 1, &[-1.0], &[1.0], config)
            .expect_err("invalid record is rejected before activation");
        assert!(matches!(
            error,
            RecurrentPpoError::InvalidConfig {
                field: "checkpoint",
                ..
            }
        ));
    }

    #[test]
    fn missing_bias_and_wrong_bias_width_are_rejected() {
        let (mut missing, config) = record();
        missing.actor.mean_head.bias = None;
        reject(missing, &config);
        let (mut wrong, config) = record();
        wrong.actor.mean_head.bias =
            Some(Param::from_tensor(Tensor::zeros([2], &inference_device())));
        reject(wrong, &config);
    }

    #[test]
    fn recurrent_hidden_transform_shape_is_checked_independently() {
        let (mut record, config) = record();
        record.actor.memory.forget_gate.hidden_transform.weight =
            Param::from_tensor(Tensor::zeros([2, 4], &inference_device()));
        reject(record, &config);
    }

    #[test]
    fn nonfinite_log_standard_deviations_are_rejected() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let (mut record, config) = record();
            record.actor.log_std =
                Param::from_tensor(Tensor::from_floats([value], &inference_device()));
            reject(record, &config);
        }
    }
}
