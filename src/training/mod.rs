//! Burn-backed trainer boundary.
//!
//! The minimal [`crate::Env`] contract remains algorithm-agnostic. Training
//! modules own Burn backend selection, run metadata, tensorization requirements,
//! metrics, checkpoints, and algorithm-specific trainer entry points.

pub mod backend;
pub mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
pub mod collector;
#[cfg(not(target_arch = "wasm32"))]
pub mod config;
#[cfg(not(target_arch = "wasm32"))]
pub mod continuous_workflow;
#[cfg(not(target_arch = "wasm32"))]
pub mod discrete_workflow;
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
pub mod dqn;
#[cfg(not(target_arch = "wasm32"))]
pub mod evaluation;
#[cfg(not(target_arch = "wasm32"))]
pub mod humanoid_model;
#[cfg(not(target_arch = "wasm32"))]
pub mod metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod ppo;
pub mod recurrent_ppo;
pub mod rng;
pub mod split_mix64;
#[cfg(not(target_arch = "wasm32"))]
pub mod tabular_q;
#[cfg(not(target_arch = "wasm32"))]
pub mod tabular_workflow;
#[cfg(not(target_arch = "wasm32"))]
pub mod tensor;

pub use backend::{inference_device, InferenceBackend};
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
pub use backend::{training_device, TrainingBackend};
pub use checkpoint::{
    policy_recorder, CheckpointError, CheckpointOperation, CheckpointPaths, PolicyRecorder,
};
#[cfg(not(target_arch = "wasm32"))]
pub use collector::{BevyTransitionCollector, CollectionError, TransitionBatch};
#[cfg(not(target_arch = "wasm32"))]
pub use config::{run_name, AlgorithmKind, RunConfig, RunConfigError, RunId, RunPaths};
#[cfg(not(target_arch = "wasm32"))]
pub use continuous_workflow::{
    run_continuous_workflow, ContinuousEvaluation, ContinuousPpoExample,
};
#[cfg(not(target_arch = "wasm32"))]
pub use discrete_workflow::{
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction,
};
#[cfg(any(not(target_arch = "wasm32"), feature = "browser-training"))]
pub use dqn::{DqnActionSelection, DqnAgent, DqnConfig, DqnError, DqnPolicy, DqnUpdate};
#[cfg(not(target_arch = "wasm32"))]
pub use dqn::{DqnReport, DqnTrainer};
#[cfg(not(target_arch = "wasm32"))]
pub use evaluation::{
    ConfidenceInterval, EpisodeOutcome, EvaluationError, EvaluationGate, EvaluationSuite,
    EvaluationSuiteKind, EvaluationSummary, GateAssessment,
};
#[cfg(not(target_arch = "wasm32"))]
pub use humanoid_model::{
    HumanoidModel, HumanoidMotion, HUMANOID_ACTIONS, HUMANOID_ACTION_LIMIT, HUMANOID_OBSERVATIONS,
};
#[cfg(not(target_arch = "wasm32"))]
pub use metrics::{MetricRecord, MetricValue, MetricsError, MetricsWriter};
#[cfg(not(target_arch = "wasm32"))]
pub use ppo::{PpoConfig, PpoReport, PpoTrainer};
#[cfg(not(target_arch = "wasm32"))]
pub use recurrent_ppo::RecurrentPpoAgent;
pub use recurrent_ppo::{
    RecurrentAction, RecurrentBehaviorMemorySample, RecurrentBehaviorSample, RecurrentMemory,
    RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy, RecurrentPpoSequence,
    RecurrentPpoUpdate, RecurrentSampler,
};
pub use rng::SeedConfig;
pub use split_mix64::SplitMix64;
#[cfg(not(target_arch = "wasm32"))]
pub use tabular_q::{
    TabularQConfig, TabularQError, TabularQEvidence, TabularQPolicy, TabularQTrainer,
    TabularQUpdate, TabularTransition, TABULAR_Q_CHECKPOINT_VERSION,
};
#[cfg(not(target_arch = "wasm32"))]
pub use tabular_workflow::{
    run_tabular_workflow, IndexedAction, TabularEvaluation, TabularExample,
};
#[cfg(not(target_arch = "wasm32"))]
pub use tensor::{
    ActionSpec, ContinuousActionSpec, DiscreteActionSpec, ObservationSpec, TensorDType,
    TensorizationError,
};
