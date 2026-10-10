//! Train increasingly distant and faster arrivals through direct RL motor policies.

pub(crate) mod encoding;
pub(crate) mod environment;
pub(crate) mod evaluation;
pub(crate) mod held_out;
pub(crate) mod model;
mod progress;
pub(crate) mod promotion;
pub(crate) mod sampling;
pub(crate) mod stage;

use crate::learning::{encoding as motor_encoding, RecoveryBatch, SELECTION_SEEDS};
use sampling::Recipe;
use sha2::{Digest, Sha256};
use stage::Stage;
use std::{
    error::Error,
    fs::File,
    io::BufWriter,
    num::{NonZeroU16, NonZeroU32},
    path::Path,
};

/// Transfer qualified recovery weights and retain the learner across all travel stages.
pub(crate) fn train(seed: u64, updates: NonZeroU32, output: &Path) -> Result<(), Box<dyn Error>> {
    train_with_recipe(seed, updates, output, Recipe::Original)
}

/// Train existing tasks with prerequisite rehearsal while retaining every evaluation gate.
pub(crate) fn train_rehearsed(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    train_with_recipe(seed, updates, output, Recipe::Rehearsal)
}

/// Keep learner transfer and promotion identical while choosing only the reset recipe.
fn train_with_recipe(
    seed: u64,
    updates: NonZeroU32,
    output: &Path,
    recipe: Recipe,
) -> Result<(), Box<dyn Error>> {
    let mut agent = model::new_agent(seed)?;
    let mut progress = initialize_progress(seed, output, recipe)?;
    for stage in Stage::ALL {
        let mut batch = RecoveryBatch::with_task(
            seed,
            &agent.policy(),
            recipe.factory(stage),
            |observation| encoding::encode(*observation),
            NonZeroU16::new(1_000).expect("positive travel horizon"),
        );
        let name = stage.name();
        for update in 1..=updates.get() {
            let update = NonZeroU32::new(update).expect("update loop starts at one");
            // Every lane uses this frozen snapshot; both optimizers update only afterward.
            let sequences = batch.collect(&agent.policy())?;
            let metrics = agent.update(&sequences)?;
            // Preserve every update, including those between checkpoint selections.
            progress::record(&mut progress, stage, update, &metrics)?;
            if !update.get().is_multiple_of(20) && update != updates {
                continue;
            }
            let bytes = agent.policy().to_bytes()?;
            let checkpoint = output.join(format!("{name}-{update}.mpk"));
            let digest = Sha256::digest(&bytes);
            std::fs::write(&checkpoint, &bytes)?;
            let policy = model::load_policy(bytes)?;
            let episodes = evaluation::evaluate(stage, &policy, &SELECTION_SEEDS)?;
            let passed =
                record_selection(output, stage, update, &episodes, metrics.optimizer_steps)?;
            if passed
                && promotion::validate(
                    stage,
                    &checkpoint,
                    &digest,
                    &output.join(format!("{name}-{update}.promotion.json")),
                )?
            {
                break;
            }
            if update == updates {
                return Err(
                    format!("Lesson {name} exhausted its update budget without passing.").into(),
                );
            }
        }
    }
    Ok(())
}

/// Run a frozen checkpoint through the same stage environment without training or output files.
pub(crate) fn infer(stage: Stage, seed: u64, checkpoint: &Path) -> Result<(), Box<dyn Error>> {
    // Validate the complete network before an environment or action is created.
    let policy = model::load_policy(std::fs::read(checkpoint)?)?;
    let episodes = evaluation::evaluate(stage, &policy, &[seed])?;
    let record = serde_json::json!({"mode":"frozen-inference", "checkpoint":checkpoint,
        "stage":stage.name(), "qualification":"not established by this episode", "episodes":episodes});
    println!("{record}");
    Ok(())
}

/// Record transfer and the opt-in recipe before collecting any training samples.
fn initialize_progress(
    seed: u64,
    output: &Path,
    recipe: Recipe,
) -> Result<BufWriter<File>, Box<dyn Error>> {
    std::fs::create_dir_all(output)?;
    if matches!(recipe, Recipe::Rehearsal) {
        let record = serde_json::json!({"recipe":"travel-prerequisite-rehearsal-v1",
            "task_stream_channel":2, "current_stage_fraction":0.5,
            "prerequisites":"remaining half split among earlier stages; endurance unchanged",
            "evaluation":"original-single-stage", "seed":seed});
        std::fs::write(output.join("training-recipe.json"), record.to_string())?;
    }
    let progress = BufWriter::new(File::create(output.join("optimization.jsonl"))?);
    let transfer = serde_json::json!({"event":"checkpoint-transfer", "source":"qualified-recovery",
        "source_checkpoint":"docs/progress/drone-recovery-transfer.mpk",
        "qualified_record_sha256":"7c2b9a6f2a0288faa27676d1514848710f4d5c24cd77505e75cf4990576a12b1",
        "destination":"travel-endurance", "optimizer":"fresh", "critic":"fresh", "optimizer_steps":0,
        "actor_observations":13, "inserted_heading_weights":"zero", "seed":seed});
    std::fs::write(output.join("transfer.json"), transfer.to_string())?;
    println!("{transfer}");
    Ok(progress)
}

/// Derive selection status from episode evidence before writing the existing artifact.
fn record_selection(
    output: &Path,
    stage: Stage,
    update: NonZeroU32,
    episodes: &[evaluation::Score],
    optimizer_steps: u64,
) -> Result<bool, Box<dyn Error>> {
    let passed = evaluation::passes(stage, episodes, &SELECTION_SEEDS);
    let name = stage.name();
    let record = serde_json::json!({"lesson":name, "update":update, "passed":passed,
        "episodes":episodes, "optimizer_steps":optimizer_steps});
    std::fs::write(
        output.join(format!("{name}-{update}.json")),
        record.to_string(),
    )?;
    println!("{record}");
    Ok(passed)
}
