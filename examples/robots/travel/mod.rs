//! Train increasingly distant and faster arrivals through direct RL motor policies.

pub(crate) mod encoding;
pub(crate) mod environment;
pub(crate) mod evaluation;
pub(crate) mod held_out;
pub(crate) mod model;
pub(crate) mod stage;

use crate::learning::{encoding as motor_encoding, RecoveryBatch, SELECTION_SEEDS};
use environment::TravelTask;
use stage::Stage;
use std::{
    error::Error,
    num::{NonZeroU16, NonZeroU32},
    path::Path,
};

/// Transfer qualified recovery weights and retain the learner across all travel stages.
pub(crate) fn train(seed: u64, updates: NonZeroU32, output: &Path) -> Result<(), Box<dyn Error>> {
    let mut agent = model::new_agent(seed)?;
    std::fs::create_dir_all(output)?;
    let transfer = serde_json::json!({"event":"checkpoint-transfer", "source":"qualified-recovery",
        "source_checkpoint":"docs/progress/drone-recovery-transfer.mpk",
        "qualified_record_sha256":"7c2b9a6f2a0288faa27676d1514848710f4d5c24cd77505e75cf4990576a12b1",
        "destination":"travel-endurance", "optimizer":"fresh", "critic":"fresh", "optimizer_steps":0,
        "actor_observations":13, "inserted_heading_weights":"zero", "seed":seed});
    std::fs::write(output.join("transfer.json"), transfer.to_string())?;
    println!("{transfer}");
    for stage in Stage::ALL {
        let mut batch = RecoveryBatch::with_task(
            seed,
            &agent.policy(),
            TravelTask::factory(stage),
            encoding::encode,
            NonZeroU16::new(1_000).expect("positive travel horizon"),
        );
        let name = stage.name();
        for update in 1..=updates.get() {
            // Every lane uses this frozen snapshot; both optimizers update only afterward.
            let sequences = batch.collect(&agent.policy())?;
            let metrics = agent.update(&sequences)?;
            if update % 20 != 0 && update != updates.get() {
                continue;
            }
            let bytes = agent.policy().to_bytes()?;
            std::fs::write(output.join(format!("{name}-{update}.mpk")), &bytes)?;
            let policy = model::load_policy(bytes)?;
            let episodes = evaluation::evaluate(stage, &policy, &SELECTION_SEEDS)?;
            let passed = evaluation::passes(stage, &episodes, &SELECTION_SEEDS);
            let record = serde_json::json!({"lesson":name, "update":update, "passed":passed,
                "episodes":episodes, "optimizer_steps":metrics.optimizer_steps});
            std::fs::write(
                output.join(format!("{name}-{update}.json")),
                record.to_string(),
            )?;
            println!("{record}");
            if passed {
                break;
            }
            if update == updates.get() {
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
