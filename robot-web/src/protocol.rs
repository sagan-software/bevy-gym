//! Strict JSON commands shared by native contract tests and the browser worker.

use serde::Deserialize;
use serde_json::{json, Value};

use crate::learning::{baseline, evaluate, learning_config, load_policy, SELECTION_SEEDS};
use crate::session::{Failure, FailureKind, Session, UPDATE_LIMIT};

/// Bound decoding allocation before parsing a command or checkpoint array.
const MAX_MESSAGE_BYTES: usize = 1_048_576;

/// One host request. Unknown fields and repeated fields are rejected by Serde.
#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
enum Command {
    /// Start a fresh fixed-budget run, replacing any previous run.
    Start {
        /// Unsigned 32-bit root seed, exactly representable in JavaScript.
        seed: u32,
    },
    /// Start the native two-lesson curriculum with a 600-update budget per lesson.
    StartCurriculum {
        /// Unsigned root seed with the same boundary as direct recovery.
        seed: u32,
    },
    /// Collect and optimize one batch; the host controls pacing.
    Advance {},
    /// Copy the current inference checkpoint without modifying the run.
    Export {},
    /// Score independent frozen weights; never replace the active training run.
    Evaluate {
        /// Burn `MessagePack` actor and critic weights for this lesson's architecture.
        bytes: Vec<u8>,
    },
}

/// Validate before mutation, then emit one response for this one request.
pub(crate) fn respond(session: &mut Session, text: &str) -> Value {
    if text.len() > MAX_MESSAGE_BYTES {
        return failure(Failure::new(
            FailureKind::InvalidRequest,
            "Worker message exceeds 1 MiB.",
        ));
    }
    serde_json::from_str::<Command>(text)
        .map_err(|error| Failure::new(FailureKind::InvalidRequest, error))
        .and_then(|command| execute(session, command))
        .unwrap_or_else(failure)
}

/// Keep protocol categories stable while retaining useful diagnostics.
pub(crate) fn failure(error: Failure) -> Value {
    json!({"event": "error", "code": error.kind, "message": error.message})
}

/// Dispatch already-validated input to the worker-owned state.
fn execute(session: &mut Session, command: Command) -> Result<Value, Failure> {
    match command {
        Command::Start { seed } => {
            session.start(seed)?;
            let config = learning_config();
            Ok(json!({"event": "started", "seed": seed, "updates": 0,
                "transitions": 0, "update_limit": UPDATE_LIMIT,
                "actor_learning_rate": config.actor_learning_rate,
                "critic_learning_rate": config.critic_learning_rate}))
        }
        Command::StartCurriculum { seed } => {
            session.start_curriculum(seed)?;
            let config = learning_config();
            Ok(
                json!({"event":"started", "seed":seed, "updates":0, "transitions":0,
                "update_limit": 2 * crate::curriculum::LESSON_LIMIT,
                "curriculum": session.progress()?.1,
                "actor_learning_rate":config.actor_learning_rate,
                "critic_learning_rate":config.critic_learning_rate}),
            )
        }
        Command::Advance {} => {
            let (updates, metrics) = session.advance()?;
            let (status, curriculum) = session.progress()?;
            let mut result = json!({"event": "progress", "status": status, "updates": updates,
                "transitions": u64::from(updates) * 512,
                "optimizer_steps": metrics.optimizer_steps,
                "actor_loss": metrics.actor_loss, "critic_loss": metrics.critic_loss,
                "entropy": metrics.entropy, "approximate_kl": metrics.approximate_kl,
                "actor_learning_rate": metrics.actor_learning_rate,
                "critic_learning_rate": metrics.critic_learning_rate});
            if let Some(curriculum) = curriculum {
                result
                    .as_object_mut()
                    .expect("progress is a JSON object")
                    .insert("curriculum".into(), curriculum);
            }
            Ok(result)
        }
        Command::Export {} => Ok(json!({"event": "policy", "bytes": session.export()?})),
        Command::Evaluate { bytes } => {
            score(bytes).map_err(|error| Failure::new(FailureKind::Checkpoint, error))
        }
    }
}

/// Evaluate in fresh environments so scores cannot change training state.
fn score(bytes: Vec<u8>) -> Result<Value, Box<dyn std::error::Error>> {
    let policy = load_policy(bytes)?;
    let episodes = evaluate(&policy, &SELECTION_SEEDS)?;
    let constant = baseline(&SELECTION_SEEDS)?;
    // Evaluation uses u64::MAX. Decimal strings preserve every seed bit in JavaScript.
    let [episodes, constant] = [episodes, constant].map(|scores| {
        scores
            .into_iter()
            .map(|episode| episode_json(&episode))
            .collect::<Vec<_>>()
    });
    Ok(json!({"event": "evaluation", "episodes": episodes, "baseline": constant}))
}

/// Preserve every seed bit at the JavaScript boundary.
pub(crate) fn episode_json(episode: &crate::learning::evaluation::EpisodeScore) -> Value {
    json!({"seed": episode.seed.to_string(), "steps": episode.steps,
        "reward": episode.reward, "final_distance": episode.final_distance,
        "survived": episode.survived})
}
