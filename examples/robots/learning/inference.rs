//! Shared frozen-policy mechanics without the training-agent factory.

#[path = "encoding.rs"]
/// Shared observation encoding and validated motor decoding.
pub(crate) mod encoding;
#[path = "episode.rs"]
/// Frozen evaluation episodes using the physical task.
mod episode;
#[path = "evaluation.rs"]
/// Selection criteria retained by the shared episode implementation.
pub(crate) mod evaluation;
#[path = "model.rs"]
/// Recurrent architecture and discount constants shared with training.
mod model;
#[path = "rollout.rs"]
/// Collection type used by the standing module's shared task definitions.
mod rollout;

pub(crate) use encoding::{decode_action, encode};
pub(crate) use model::{GAE_LAMBDA, GAMMA};
pub(crate) use rollout::RecoveryBatch;
