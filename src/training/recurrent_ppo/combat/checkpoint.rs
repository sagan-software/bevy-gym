//! Validate the distinct joint record before activating either action head.
use super::super::{checkpoint, InferenceBackend, RecurrentPpoConfig, RecurrentPpoError};
use super::actor::CombatNetworksRecord;
/// Validate shared Gaussian and critic tensors, then the eight-column categorical head.
pub(super) fn validate(
    record: &CombatNetworksRecord<InferenceBackend>,
    observations: usize,
    global: usize,
    values: usize,
    actions: usize,
    config: &RecurrentPpoConfig,
) -> Result<(), RecurrentPpoError> {
    checkpoint::validate_parts(
        &record.actor.continuous,
        &record.critic,
        observations,
        global,
        values,
        actions,
        config,
    )?;
    checkpoint::linear(&record.actor.signals, config.actor_hidden_size, 8)
}

#[cfg(test)]
mod tests {
    use super::super::super::{inference_device, SeedConfig};
    use super::super::{RecurrentCombatAgent, RecurrentCombatPolicy};
    use super::*;
    use burn::{
        module::Module,
        record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
        tensor::Tensor,
    };
    /// Produce a valid joint record through the same public learner used by collection.
    fn record(config: &RecurrentPpoConfig) -> CombatNetworksRecord<InferenceBackend> {
        let policy = RecurrentCombatAgent::new(
            2,
            3,
            1,
            &[-1.0],
            &[1.0],
            config.clone(),
            SeedConfig::from_root(4),
        )
        .expect("actor")
        .policy();
        super::super::actor::CombatNetworks {
            actor: policy.actor,
            critic: policy.critic,
        }
        .into_record()
    }

    /// Every malformed categorical tensor fails before either head can act.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn missing_wrong_width_and_nonfinite_signal_parameters_are_rejected() {
        let config = RecurrentPpoConfig {
            actor_hidden_size: 4,
            critic_hidden_sizes: vec![4],
            ..RecurrentPpoConfig::default()
        };
        for case in 0..4 {
            let mut record = record(&config);
            let device = inference_device();
            match case {
                0 => record.actor.signals.bias = None,
                1 => {
                    record.actor.signals.bias = record
                        .actor
                        .signals
                        .bias
                        .map(|bias| bias.map(|_| Tensor::zeros([7], &device)));
                }
                2 => {
                    record.actor.signals.weight = record
                        .actor
                        .signals
                        .weight
                        .map(|_| Tensor::full([4, 8], f32::NAN, &device));
                }
                _ => {
                    record.actor.signals.weight = record
                        .actor
                        .signals
                        .weight
                        .map(|_| Tensor::zeros([3, 8], &device));
                }
            }
            let bytes = NamedMpkBytesRecorder::<FullPrecisionSettings>::default()
                .record(record, ())
                .expect("encoded fixture");
            assert!(matches!(
                RecurrentCombatPolicy::load_bytes(bytes, 2, 3, 1, &[-1.0], &[1.0], &config),
                Err(RecurrentPpoError::InvalidConfig {
                    field: "checkpoint",
                    ..
                })
            ));
        }
    }
}
