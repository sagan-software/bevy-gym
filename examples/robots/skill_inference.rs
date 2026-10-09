//! Run one frozen RL checkpoint without a training loop or fallback controller.

#[path = "learning/encoding.rs"]
mod encoding;
#[path = "learning/episode.rs"]
mod episode;
#[path = "learning/model.rs"]
mod model;

use std::{error::Error, io::Read, path::PathBuf};

use bevy_gym::robots::DroneHover;
use clap::Parser;
use encoding::{decode_action, encode};

/// Select frozen weights and an independent episode reset.
#[derive(Parser)]
struct Options {
    /// Checkpoint matching the twelve-input, four-motor recovery architecture.
    #[arg(long, default_value = "docs/progress/drone-curriculum.mpk")]
    checkpoint: PathBuf,
    /// Environment reset seed; inference retains memory until the episode ends.
    #[arg(long, default_value = "42")]
    seed: u64,
}

/// Load the policy before constructing or advancing the selected lesson.
pub(crate) fn run(make_environment: fn() -> DroneHover) -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    // Reject missing or incompatible weights before the first environment action.
    let mut bytes = Vec::new();
    std::fs::File::open(&options.checkpoint)?.read_to_end(&mut bytes)?;
    let policy = model::load_policy(bytes)?;
    let episodes = episode::evaluate_with(&policy, &[options.seed], make_environment)?;
    let record = serde_json::json!({
        "checkpoint": options.checkpoint,
        "mode": "frozen-policy-inference",
        "episodes": episodes,
    });
    println!("{record}");
    Ok(())
}
