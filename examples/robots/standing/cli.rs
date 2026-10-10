//! Separate standing training and frozen evaluation commands.

use super::{checkpoint, evaluation, reward::Recipe, trace, training};
use clap::{Parser, Subcommand};
use std::{
    error::Error,
    num::NonZeroU32,
    path::{Path, PathBuf},
};

/// Run physical standing training or frozen checkpoint evaluation.
#[derive(Parser)]
struct Options {
    /// Training and inference are distinct commands with disjoint options.
    #[command(subcommand)]
    command: Command,
}

/// Closed standalone workflow modes.
#[derive(Subcommand)]
enum Command {
    /// Stream one frozen physical episode; completion is not standing qualification.
    Trace {
        /// Weight file and its required .json PPO provenance record.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Independent physical reset root; default matches the standing viewer.
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Train with PPO; a failed budget returns an error and retains its candidate.
    Train {
        /// Training-only reward; frozen evaluation always retains the original profile.
        #[arg(long, value_enum, default_value_t = Recipe::Original)]
        reward_profile: Recipe,
        /// Maximum 512-transition updates; exhaustion never grants qualification.
        #[arg(long, default_value = "600")]
        updates: NonZeroU32,
        /// Root for initialization and independent training streams.
        #[arg(long, default_value = "7")]
        seed: u64,
        /// New output directory; an existing path is rejected.
        #[arg(long, default_value = "runs/droid-standing/seed7")]
        output: PathBuf,
    },
    /// Start from RL actor/critic weights with fresh optimizer and episode state.
    WarmStart {
        /// Valid standing checkpoint and its required seven-member provenance sidecar.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Training-only reward; frozen evaluation always retains the original profile.
        #[arg(long, value_enum, default_value_t = Recipe::Original)]
        reward_profile: Recipe,
        /// Maximum new 512-transition batches; prior updates are recorded separately.
        #[arg(long, default_value = "600")]
        updates: NonZeroU32,
        /// New root for minibatch ordering and independent training streams.
        #[arg(long, default_value = "13")]
        seed: u64,
        /// New output directory; an existing path is rejected.
        #[arg(long, default_value = "runs/droid-standing/checkpoint-start-seed13")]
        output: PathBuf,
    },
    /// Evaluate frozen weights without training or writing files.
    Evaluate {
        /// Weight file accompanied by its required .json PPO provenance record.
        #[arg(long)]
        checkpoint: PathBuf,
        /// Inspect one seed; omission runs all 32 fixed held-out cases.
        #[arg(long)]
        seed: Option<u64>,
    },
}

/// Parse process arguments with clap and preserve every mode's failure exit.
pub(super) fn run() -> Result<(), Box<dyn Error>> {
    let command = Options::parse().command;
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(execute(command))
}

/// Await file operations while keeping policy updates between complete rollout batches.
async fn execute(command: Command) -> Result<(), Box<dyn Error>> {
    match command {
        Command::Trace { checkpoint, seed } => {
            let bytes = tokio::fs::read(&checkpoint).await?;
            let metadata = tokio::fs::read(checkpoint.with_extension("json")).await?;
            trace::write(bytes, &metadata, seed, &mut std::io::stdout().lock())
        }
        Command::Train {
            reward_profile,
            updates,
            seed,
            output,
        } => Box::pin(train_selected(seed, updates, &output, None, reward_profile)).await,
        Command::WarmStart {
            reward_profile,
            checkpoint,
            updates,
            seed,
            output,
        } => {
            Box::pin(train_selected(
                seed,
                updates,
                &output,
                Some(&checkpoint),
                reward_profile,
            ))
            .await
        }
        Command::Evaluate { checkpoint, seed } => evaluate_checkpoint(&checkpoint, seed).await,
    }
}

/// Preserve existing default entry points and use shaping only when explicitly selected.
async fn train_selected(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
    checkpoint: Option<&Path>,
    recipe: Recipe,
) -> Result<(), Box<dyn Error>> {
    match (recipe, checkpoint) {
        (Recipe::Original, None) => Box::pin(training::train(seed, updates, output)).await,
        (Recipe::Original, Some(checkpoint)) => {
            Box::pin(training::warm_start(seed, updates, output, checkpoint)).await
        }
        (Recipe::PostureV1, checkpoint) => {
            Box::pin(training::train_profiled(
                seed, updates, output, checkpoint, recipe,
            ))
            .await
        }
    }
}

/// Reload immutable weights before measuring either one episode or the original held-out suite.
async fn evaluate_checkpoint(checkpoint: &Path, seed: Option<u64>) -> Result<(), Box<dyn Error>> {
    // Identity validation precedes construction of any environment.
    let bytes = tokio::fs::read(checkpoint).await?;
    let metadata = tokio::fs::read(checkpoint.with_extension("json")).await?;
    let (policy, record) = checkpoint::from_bytes(bytes, &metadata)?;
    let held_out = evaluation::held_out_seeds();
    let seeds = seed
        .as_ref()
        .map_or(held_out.as_slice(), std::slice::from_ref);
    let episodes = evaluation::evaluate(&policy, seeds)?;
    let passed = evaluation::passes(&episodes, seeds);
    let suite = if seed.is_some() {
        "single-episode"
    } else {
        "held-out"
    };
    let report = serde_json::json!({"mode":"frozen-inference", "suite":suite,
        "checkpoint":checkpoint, "provenance":record, "passed":passed,
        "qualification":"requires independent run evidence; a single episode cannot qualify",
        "episodes":episodes});
    println!("{report}");
    if passed {
        Ok(())
    } else {
        Err("Standing evaluation failed; checkpoint is not qualified.".into())
    }
}
