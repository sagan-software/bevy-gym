//! One recurrent representation for the continuous and categorical heads.
use super::super::{CentralCritic, RecurrentActor};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, LstmState};
use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Distinct network record; legacy Gaussian checkpoints have no joint categorical head.
#[derive(Module, Debug)]
pub(super) struct CombatNetworks<B: Backend> {
    /// Shared recurrent actor and its two action heads.
    pub(super) actor: CombatActor<B>,
    /// Centralized state available only to training value queries.
    pub(super) critic: CentralCritic<B>,
}

/// Shared temporal features preserve one memory update for the complete joint action.
#[derive(Module, Debug)]
pub(super) struct CombatActor<B: Backend> {
    /// Existing recurrent representation, Gaussian head and trainable exploration scales.
    pub(super) continuous: RecurrentActor<B>,
    /// Eight logits for the complete trigger and alert Cartesian product.
    pub(super) signals: Linear<B>,
}

/// Differentiable distributions and the final state from one contiguous sequence.
pub(super) struct CombatDistribution<B: Backend> {
    /// Pre-tanh Gaussian means in flattened temporal order.
    pub(super) mean: Tensor<B, 2>,
    /// Bounded log standard deviations in the same order.
    pub(super) log_std: Tensor<B, 2>,
    /// Joint trigger and alert logits in the same order.
    pub(super) logits: Tensor<B, 2>,
    /// Recurrent state after consuming the sequence exactly once.
    pub(super) memory: LstmState<B, 2>,
}
impl<B: Backend> CombatActor<B> {
    /// Initialize both heads from the same seeded recurrent representation.
    pub(super) fn new(
        observation_dim: usize,
        hidden_size: usize,
        action_dim: usize,
        initial_log_std: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            continuous: RecurrentActor::new(
                observation_dim,
                hidden_size,
                action_dim,
                initial_log_std,
                device,
            ),
            signals: LinearConfig::new(hidden_size, 8).init(device),
        }
    }

    /// Evaluate both action distributions without advancing recurrent memory twice.
    pub(super) fn forward(
        &self,
        observations: Tensor<B, 3>,
        initial_state: Option<LstmState<B, 2>>,
        action_dim: usize,
        log_std_min: f32,
        log_std_max: f32,
    ) -> CombatDistribution<B> {
        let (encoded, memory) = self.continuous.memory.forward(observations, initial_state);
        let [batch_size, sequence_length, hidden_size] = encoded.dims();
        let encoded = encoded.reshape([batch_size * sequence_length, hidden_size]);
        // Tensor clones share storage; both heads retain gradients into the same recurrent features.
        let mean = self.continuous.mean_head.forward(encoded.clone());
        let logits = self.signals.forward(encoded);
        let log_std = self
            .continuous
            .log_std
            .val()
            .reshape([1, action_dim])
            .repeat_dim(0, batch_size * sequence_length)
            .clamp(log_std_min, log_std_max);
        CombatDistribution {
            mean,
            log_std,
            logits,
            memory,
        }
    }
}
