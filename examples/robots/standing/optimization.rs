//! Explicit optimizer recipes; architecture and frozen mean actions remain unchanged.

use bevy_gym::training::RecurrentPpoConfig;
use clap::ValueEnum;
use serde::Serialize;

/// Closed training recipes; the default retains the historical PPO recipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub(crate) enum OptimizerProfile {
    /// Four passes at the original learning rates and entropy weight.
    #[default]
    Original,
    /// One pass with a tenfold smaller actor rate and no entropy bonus.
    ConservativeV1,
    /// Conservative optimization with freshly reduced Gaussian exploration.
    LowNoiseV1,
}

impl OptimizerProfile {
    /// Configure learning and exploration; shapes, action bounds and mean inference stay identical.
    pub(crate) const fn configure(self, mut config: RecurrentPpoConfig) -> RecurrentPpoConfig {
        match self {
            Self::Original => config,
            Self::ConservativeV1 | Self::LowNoiseV1 => {
                config.actor_learning_rate = 0.00003;
                config.entropy_coefficient = 0.0;
                config.epochs = 1;
                if matches!(self, Self::LowNoiseV1) {
                    config.initial_log_std = -3.5;
                }
                config
            }
        }
    }

    /// Preserve default artifacts; emit the complete selected optimizer recipe otherwise.
    pub(crate) const fn record(self) -> Option<OptimizerRecord> {
        match self {
            Self::Original => None,
            Self::ConservativeV1 => Some(OptimizerRecord {
                schema: OptimizerSchema::ConservativeV1,
                actor_learning_rate: 0.00003,
                critic_learning_rate: 0.001,
                entropy_coefficient: 0.0,
                epochs: 1,
                reset_initial_log_std: None,
            }),
            Self::LowNoiseV1 => Some(OptimizerRecord {
                schema: OptimizerSchema::LowNoiseV1,
                actor_learning_rate: 0.00003,
                critic_learning_rate: 0.001,
                entropy_coefficient: 0.0,
                epochs: 1,
                reset_initial_log_std: Some(-3.5),
            }),
        }
    }
}

/// Fixed emitted optimizer-profile vocabulary; there is no input parser.
#[derive(Clone, Copy, Debug, Serialize)]
enum OptimizerSchema {
    /// The conservative optimizer's first version.
    #[serde(rename = "droid-standing-conservative-optimizer-v1")]
    ConservativeV1,
    /// The low-noise optimizer's first version.
    #[serde(rename = "droid-standing-low-noise-optimizer-v1")]
    LowNoiseV1,
}

/// Complete optimization metadata written before collecting transitions.
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct OptimizerRecord {
    /// Fixed schema identity.
    schema: OptimizerSchema,
    /// Adam actor rate, dimensionless per optimizer step.
    actor_learning_rate: f64,
    /// Adam critic rate, dimensionless per optimizer step.
    critic_learning_rate: f64,
    /// Dimensionless multiplier of the Gaussian entropy term.
    entropy_coefficient: f32,
    /// Complete optimizer passes over each rollout.
    epochs: usize,
    /// Natural logarithm of dimensionless pre-tanh Gaussian standard deviation.
    #[serde(skip_serializing_if = "Option::is_none")]
    reset_initial_log_std: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both recipes retain architecture and inference settings; only selected optimization changes.
    #[test]
    fn profiles_preserve_shapes_and_record_selected_optimization() {
        let base = RecurrentPpoConfig {
            critic_learning_rate: 0.001,
            ..Default::default()
        };
        for profile in [
            OptimizerProfile::Original,
            OptimizerProfile::ConservativeV1,
            OptimizerProfile::LowNoiseV1,
        ] {
            let config = profile.configure(base.clone());
            assert_eq!(config.actor_hidden_size, base.actor_hidden_size);
            assert_eq!(config.critic_hidden_sizes, base.critic_hidden_sizes);
            assert_eq!(config.gamma.to_bits(), base.gamma.to_bits());
            assert_eq!(config.gae_lambda.to_bits(), base.gae_lambda.to_bits());
            assert_eq!(
                config.initial_log_std.to_bits(),
                if profile == OptimizerProfile::LowNoiseV1 {
                    (-3.5_f32).to_bits()
                } else {
                    base.initial_log_std.to_bits()
                }
            );
            assert_eq!(config.log_std_min.to_bits(), base.log_std_min.to_bits());
            assert_eq!(config.log_std_max.to_bits(), base.log_std_max.to_bits());
            assert_eq!(config.critic_learning_rate.to_bits(), 0.001_f64.to_bits());
            match profile {
                OptimizerProfile::Original => {
                    assert_eq!(config.actor_learning_rate.to_bits(), 0.0003_f64.to_bits());
                    assert_eq!(config.entropy_coefficient.to_bits(), 0.005_f32.to_bits());
                    assert_eq!(config.epochs, 4);
                    assert!(profile.record().is_none());
                }
                OptimizerProfile::ConservativeV1 | OptimizerProfile::LowNoiseV1 => {
                    assert_eq!(config.actor_learning_rate.to_bits(), 0.00003_f64.to_bits());
                    assert_eq!(config.entropy_coefficient.to_bits(), 0.0_f32.to_bits());
                    assert_eq!(config.epochs, 1);
                    let record = serde_json::to_value(profile.record().expect("explicit recipe"))
                        .expect("finite fixed fields");
                    assert_eq!(record["epochs"], 1);
                    assert_eq!(record["actor_learning_rate"], 0.00003);
                }
            }
        }
    }
}
