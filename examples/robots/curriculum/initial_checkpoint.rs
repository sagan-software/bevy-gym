//! Transfer only a recorded RL checkpoint whose complete network identity is known.

use std::{error::Error, path::Path};

use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};
use clap::ValueEnum;

use crate::learning::{learning_config, load_policy};

/// Closed catalog of qualified prerequisite weights for standalone training.
#[derive(Clone, Copy, ValueEnum)]
pub(crate) enum InitialCheckpoint {
    /// PPO seed 7, hover update 280; independently qualified before recovery.
    QualifiedHover,
}

impl InitialCheckpoint {
    /// Load actor and critic weights with a fresh optimizer and sampling streams.
    pub(crate) fn load(self, seed: u64) -> Result<RecurrentPpoAgent, Box<dyn Error>> {
        match self {
            Self::QualifiedHover => load_verified_hover(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/progress/drone-hover.mpk"),
                seed,
            ),
        }
    }

    /// Record source identity and the actual transfer semantics beside new checkpoints.
    pub(crate) fn record(self, seed: u64) -> serde_json::Value {
        match self {
            Self::QualifiedHover => serde_json::json!({
                "event": "checkpoint-transfer",
                "source": "qualified-hover",
                "source_checkpoint": "docs/progress/drone-hover.mpk",
                "qualified_record_sha256": "5aa47c6941b1aa243c4eafcb0fafaf8c6fa9ee816944dc5a5ca0d3f95712403b",
                "source_training": "PPO, seed 7, hover update 280",
                "destination": "recovery",
                "optimizer": "fresh",
                "optimizer_steps": 0,
                "seed": seed,
            }),
        }
    }
}

/// Check the loaded networks themselves, so a file change between reads cannot bypass identity.
fn load_verified_hover(path: &Path, seed: u64) -> Result<RecurrentPpoAgent, Box<dyn Error>> {
    // Use the existing architecture and transfer API; no optimizer state is serialized.
    let agent = RecurrentPpoAgent::load(
        path,
        12,
        12,
        1,
        &[0.0; 4],
        &[1.0; 4],
        learning_config(),
        SeedConfig::from_root(seed),
    )?;
    // Canonical record comparison covers both actor and critic, not only sampled actions.
    let expected = load_policy(include_bytes!("../../../docs/progress/drone-hover.mpk").to_vec())?;
    if agent.policy().to_bytes()? != expected.to_bytes()? {
        return Err("qualified-hover checkpoint identity mismatch".into());
    }
    Ok(agent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::RecoveryBatch;

    /// Preserve both networks and start the optimizer counter from this new rollout.
    #[test]
    fn transfer_preserves_networks_and_starts_a_fresh_optimizer() {
        let mut agent = InitialCheckpoint::QualifiedHover
            .load(11)
            .expect("qualified source");
        let expected =
            load_policy(include_bytes!("../../../docs/progress/drone-hover.mpk").to_vec())
                .expect("recorded source");
        let before = agent
            .policy()
            .to_bytes()
            .expect("serialize transferred networks");
        assert_eq!(
            before,
            expected.to_bytes().expect("serialize source networks")
        );
        let mut batch = RecoveryBatch::new(11, &agent.policy());
        let sequences = batch
            .collect(&agent.policy())
            .expect("collect independent rollout");
        let config = learning_config();
        let maximum_updates = config.epochs * sequences.len().div_ceil(config.minibatch_sequences);
        let metrics = agent.update(&sequences).expect("real PPO update");
        assert!(metrics.optimizer_steps > 0);
        assert_eq!(metrics.optimizer_steps, metrics.optimizer_updates);
        assert!(metrics.optimizer_steps <= maximum_updates as u64);
        assert_ne!(
            before,
            agent
                .policy()
                .to_bytes()
                .expect("serialize updated networks")
        );
    }

    /// Missing and compatible-but-different records cannot become accepted source weights.
    #[test]
    fn source_identity_rejects_missing_and_different_checkpoints() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let missing = root.join("docs/progress/absent-qualified-hover.mpk");
        assert!(load_verified_hover(&missing, 11).is_err());
        let different = root.join("docs/progress/drone-curriculum.mpk");
        let error = load_verified_hover(&different, 11).expect_err("different network identity");
        assert_eq!(
            error.to_string(),
            "qualified-hover checkpoint identity mismatch"
        );
    }
}
