//! A bounded training run owned exclusively by one worker.

use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoUpdate};
use serde::Serialize;

use crate::learning::{new_agent, RecoveryBatch};

/// Fixed lesson budget: 133,120 transitions, without a claim of convergence.
pub(crate) const UPDATE_LIMIT: u16 = 260;

/// Mutually exclusive worker states; failed runs cannot export partial weights.
#[derive(Default)]
pub(crate) enum Session {
    /// No training run has been requested.
    #[default]
    Idle,
    /// A valid run, including one that has reached its update limit.
    Ready(Box<Run>),
    /// An update failed after it could have changed optimizer or rollout state.
    Failed(String),
}

/// Model, optimizer, and continuing rollout state for one seeded run.
pub(crate) struct Run {
    /// Owns both optimizers and the learned parameters.
    agent: RecurrentPpoAgent,
    /// Eight environments and recurrent memories retained across updates.
    batch: RecoveryBatch,
    /// Successful updates, bounded by the fixed lesson budget.
    updates: CompletedUpdates,
}

/// Only successful updates can increase this private bounded count.
#[derive(Clone, Copy, Default)]
struct CompletedUpdates(u16);

/// Stable wire categories distinguish malformed requests from run failures.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FailureKind {
    /// Input type, JSON syntax, schema, or size failed validation.
    InvalidRequest,
    /// The requested operation needs a training run.
    NotStarted,
    /// This run has consumed its update budget.
    Complete,
    /// A previous update failed; the user must start another run.
    Failed,
    /// Model initialization, collection, or optimization failed.
    Training,
    /// An imported or exported checkpoint could not be used.
    Checkpoint,
}

/// Diagnostic text is separate from the machine-facing failure category.
#[derive(Debug)]
pub(crate) struct Failure {
    /// Stable category used by the browser controller.
    pub(crate) kind: FailureKind,
    /// Human-readable detail from validation or the underlying operation.
    pub(crate) message: String,
}

impl Failure {
    /// Preserve the operation's diagnostic when crossing the JSON boundary.
    pub(crate) fn new(kind: FailureKind, message: impl ToString) -> Self {
        Self {
            kind,
            message: message.to_string(),
        }
    }
}

impl Session {
    /// Replace the current run only after its model and rollout exist.
    pub(crate) fn start(&mut self, seed: u32) -> Result<(), Failure> {
        let agent = new_agent(u64::from(seed))
            .map_err(|error| Failure::new(FailureKind::Training, error))?;
        let batch = RecoveryBatch::new(u64::from(seed), &agent.policy());
        *self = Self::Ready(Box::new(Run {
            agent,
            batch,
            updates: CompletedUpdates::default(),
        }));
        Ok(())
    }

    /// Perform exactly one collection and optimizer update, then yield to the host.
    pub(crate) fn advance(&mut self) -> Result<(u16, RecurrentPpoUpdate), Failure> {
        let run = match self {
            Self::Idle => {
                return Err(Failure::new(
                    FailureKind::NotStarted,
                    "Start training first.",
                ))
            }
            Self::Failed(message) => return Err(Failure::new(FailureKind::Failed, message)),
            Self::Ready(run) => run,
        };
        if run.updates.0 == UPDATE_LIMIT {
            return Err(Failure::new(
                FailureKind::Complete,
                "The training budget is complete.",
            ));
        }

        // A failed collection or update may have mutated state. Discard that run.
        match run.update() {
            Ok(metrics) => {
                run.updates.0 += 1;
                Ok((run.updates.0, metrics))
            }
            Err(error) => {
                let message = error.to_string();
                *self = Self::Failed(message.clone());
                Err(Failure::new(FailureKind::Training, message))
            }
        }
    }

    /// Export current actor and critic weights without advancing any environment.
    pub(crate) fn export(&self) -> Result<Vec<u8>, Failure> {
        match self {
            Self::Idle => Err(Failure::new(
                FailureKind::NotStarted,
                "Start training first.",
            )),
            Self::Failed(message) => Err(Failure::new(FailureKind::Failed, message)),
            Self::Ready(run) => run
                .agent
                .policy()
                .to_bytes()
                .map_err(|error| Failure::new(FailureKind::Checkpoint, error)),
        }
    }
}

impl Run {
    /// Bound transient allocation to one 512-transition batch and optimizer update.
    fn update(&mut self) -> Result<RecurrentPpoUpdate, Box<dyn std::error::Error>> {
        let sequences = self.batch.collect(&self.agent.policy())?;
        validate_metrics(self.agent.update(&sequences)?)
    }
}

/// Reject invalid numerical results before JSON could replace them with null.
fn validate_metrics(
    metrics: RecurrentPpoUpdate,
) -> Result<RecurrentPpoUpdate, Box<dyn std::error::Error>> {
    if [
        metrics.actor_loss,
        metrics.critic_loss,
        metrics.entropy,
        metrics.approximate_kl,
        metrics.actor_learning_rate,
        metrics.critic_learning_rate,
    ]
    .into_iter()
    .any(|value| !value.is_finite())
    {
        return Err("Training returned non-finite metrics.".into());
    }
    if metrics.valid_samples != 512 {
        return Err("Training did not consume the 512-transition batch.".into());
    }
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoUpdate, SeedConfig};

    use super::{validate_metrics, CompletedUpdates, Run, Session, UPDATE_LIMIT};
    use crate::learning::{learning_config, new_agent, RecoveryBatch};
    use crate::protocol::respond;

    #[test]
    fn numerical_failures_cannot_be_reported_as_successful_progress() {
        let valid = RecurrentPpoUpdate {
            optimizer_steps: 4,
            optimizer_updates: 4,
            valid_samples: 512,
            actor_loss: 0.0,
            critic_loss: 0.0,
            entropy: 0.0,
            approximate_kl: 0.0,
            actor_learning_rate: 0.0003,
            critic_learning_rate: 0.001,
        };
        assert_eq!(validate_metrics(valid).expect("finite batch"), valid);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in 0..6 {
                let mut metrics = valid;
                let fields = [
                    &mut metrics.actor_loss,
                    &mut metrics.critic_loss,
                    &mut metrics.entropy,
                    &mut metrics.approximate_kl,
                    &mut metrics.actor_learning_rate,
                    &mut metrics.critic_learning_rate,
                ];
                *fields.into_iter().nth(field).expect("known metric") = invalid;
                assert_eq!(
                    validate_metrics(metrics)
                        .expect_err("non-finite metric")
                        .to_string(),
                    "Training returned non-finite metrics."
                );
            }
        }
        for samples in [0, 511, 513, u64::MAX] {
            let metrics = RecurrentPpoUpdate {
                valid_samples: samples,
                ..valid
            };
            assert_eq!(
                validate_metrics(metrics)
                    .expect_err("wrong batch size")
                    .to_string(),
                "Training did not consume the 512-transition batch."
            );
        }
    }

    #[test]
    fn the_last_update_completes_and_preserves_exportable_weights() {
        let agent = new_agent(7).expect("valid lesson recipe");
        let batch = RecoveryBatch::new(7, &agent.policy());
        let mut session = Session::Ready(Box::new(Run {
            agent,
            batch,
            updates: CompletedUpdates(UPDATE_LIMIT - 1),
        }));
        let result = respond(&mut session, r#"{"command":"advance"}"#);
        assert_eq!(result["status"], "complete");
        assert_eq!(result["updates"], 260);
        assert_eq!(result["transitions"], 133_120);
        let checkpoint = respond(&mut session, r#"{"command":"export"}"#);
        assert_eq!(
            respond(&mut session, r#"{"command":"advance"}"#)["code"],
            "complete"
        );
        assert_eq!(respond(&mut session, r#"{"command":"export"}"#), checkpoint);
        assert_eq!(
            respond(&mut session, r#"{"command":"start","seed":7}"#)["updates"],
            0
        );
        assert_eq!(
            respond(&mut session, r#"{"command":"advance"}"#)["updates"],
            1
        );
    }

    #[test]
    fn failed_collection_discards_the_run_until_explicit_restart() {
        // Deliberately violate the private recipe to exercise an actual collection error.
        let agent = RecurrentPpoAgent::new(
            11,
            12,
            1,
            &[0.0; 4],
            &[1.0; 4],
            learning_config(),
            SeedConfig::from_root(7),
        )
        .expect("constructible but incompatible actor");
        let batch = RecoveryBatch::new(7, &agent.policy());
        let mut session = Session::Ready(Box::new(Run {
            agent,
            batch,
            updates: CompletedUpdates::default(),
        }));
        let first = respond(&mut session, r#"{"command":"advance"}"#);
        assert_eq!(first["code"], "training");
        for command in ["advance", "export"] {
            let request = serde_json::json!({"command": command}).to_string();
            let failed = respond(&mut session, &request);
            assert_eq!(failed["code"], "failed");
            assert_eq!(failed["message"], first["message"]);
        }
        assert_eq!(
            respond(&mut session, r#"{"command":"start","seed":7}"#)["event"],
            "started"
        );
        assert_eq!(
            respond(&mut session, r#"{"command":"advance"}"#)["event"],
            "progress"
        );
    }
}
