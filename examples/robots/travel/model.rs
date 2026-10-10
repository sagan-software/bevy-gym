//! Initialize travel PPO from the qualified RL recovery actor with one extra input.

use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};
use std::{error::Error, fs::File, io::Read, path::Path};

use crate::learning::learning_config;

/// Transfer the recorded recovery actor; initialize heading weights, critic and optimizers fresh.
pub(crate) fn new_agent(seed: u64) -> Result<RecurrentPpoAgent, Box<dyn Error>> {
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/progress/drone-recovery-transfer.mpk");
    from_source(&source, seed)
}

/// Reject missing or changed prerequisite records before constructing a travel learner.
fn from_source(source: &Path, seed: u64) -> Result<RecurrentPpoAgent, Box<dyn Error>> {
    // Refuse an altered prerequisite instead of silently relabeling arbitrary weights as qualified.
    let reference = include_bytes!("../../../docs/progress/drone-recovery-transfer.mpk");
    let mut bytes = Vec::with_capacity(reference.len() + 1);
    File::open(source)?
        .take(reference.len() as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes != reference {
        return Err("qualified recovery source differs from its recorded checkpoint".into());
    }
    RecurrentPpoAgent::load_actor_with_inserted_input_feature_and_fresh_critic(
        source,
        12,
        12,
        13,
        1,
        &[0.0; 4],
        &[1.0; 4],
        learning_config(),
        SeedConfig::from_root(seed),
    )
    .map_err(Into::into)
}

/// Validate and load a frozen travel checkpoint with thirteen actor and critic inputs.
pub(crate) fn load_policy(
    bytes: Vec<u8>,
) -> Result<bevy_gym::training::RecurrentPpoPolicy, bevy_gym::training::RecurrentPpoError> {
    bevy_gym::training::RecurrentPpoPolicy::load_bytes(
        bytes,
        13,
        13,
        1,
        &[0.0; 4],
        &[1.0; 4],
        &learning_config(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Neither missing files nor other compatible RL weights may acquire the prerequisite label.
    #[test]
    fn missing_and_different_sources_are_rejected() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        from_source(&root.join("docs/progress/absent-travel-source.mpk"), 11)
            .expect_err("missing source fails");
        let error = from_source(&root.join("docs/progress/drone-hover.mpk"), 11)
            .expect_err("different qualified policy is not this prerequisite");
        assert_eq!(
            error.to_string(),
            "qualified recovery source differs from its recorded checkpoint"
        );
    }
}
