//! Separate standing training and frozen evaluation commands.

use super::{checkpoint, evaluation, training};
use clap::{Parser, Subcommand};
use std::{error::Error, num::NonZeroU32, path::PathBuf};

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
    /// Train with PPO; a failed budget returns an error and retains its candidate.
    Train {
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
        Command::Train {
            updates,
            seed,
            output,
        } => Box::pin(training::train(seed, updates, &output)).await,
        Command::WarmStart {
            checkpoint,
            updates,
            seed,
            output,
        } => Box::pin(training::warm_start(seed, updates, &output, &checkpoint)).await,
        Command::Evaluate { checkpoint, seed } => {
            // Read each file once; identity validation precedes construction of any environment.
            let bytes = tokio::fs::read(&checkpoint).await?;
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
    }
}
