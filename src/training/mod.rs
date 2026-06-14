//! Burn-backed trainer boundary.
//!
//! The minimal [`crate::Env`] contract remains algorithm-agnostic. Training
//! modules own Burn backend selection, run metadata, tensorization requirements,
//! metrics, checkpoints, and algorithm-specific trainer entry points.

pub mod backend;
pub mod checkpoint;
pub mod config;
pub mod dqn;
pub mod metrics;
pub mod ppo;
pub mod rng;
pub mod tensor;

pub use backend::{inference_device, training_device, InferenceBackend, TrainingBackend};
pub use checkpoint::{
    policy_recorder, CheckpointError, CheckpointOperation, CheckpointPaths, PolicyRecorder,
};
pub use config::{run_name, AlgorithmKind, RunConfig, RunConfigError, RunId, RunPaths};
pub use dqn::{DqnConfig, DqnReport, DqnTrainer};
pub use metrics::{MetricRecord, MetricValue, MetricsError, MetricsWriter};
pub use ppo::{PpoConfig, PpoReport, PpoTrainer};
pub use rng::SeedConfig;
pub use tensor::{
    ActionSpec, DiscreteActionSpec, ObservationSpec, TensorDType, TensorizationError,
};
