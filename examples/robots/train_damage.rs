//! Train intact hover, a fixed motor failure, then randomized failures.
//!
//! Run `cargo run --no-default-features --features robots --example drone-train-damage`.
//! Each lesson must pass its selection cases before the next lesson starts.

#![allow(
    clippy::disallowed_methods,
    reason = "This synchronous tutorial writes local checkpoints without an async runtime."
)]

#[path = "damage/assessment.rs"]
mod assessment;
#[path = "damage/training.rs"]
mod damage_training;
#[expect(
    unused_imports,
    dead_code,
    reason = "This guide reuses only part of the shared healthy-flight tutorial facade."
)]
mod learning;

use clap::Parser;
use damage_training::{load_policy, new_agent, Lesson};
use std::{error::Error, num::NonZeroU32, path::PathBuf};

/// Bound training and retain every evaluated checkpoint for inspection.
#[derive(Parser)]
struct Options {
    /// Maximum 512-transition updates per lesson.
    #[arg(long, default_value = "600")]
    updates: NonZeroU32,
    /// Root seed for policy initialization and independent training streams.
    #[arg(long, default_value = "7")]
    seed: u64,
    /// Directory for lesson-named weights and selection results.
    #[arg(long, default_value = "runs/drone-damage-training")]
    output: PathBuf,
}

/// Retain the optimizer across lessons and evaluate only reloaded frozen weights.
fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    std::fs::create_dir_all(&options.output)?;
    let mut agent = new_agent(options.seed)?;
    for lesson in [Lesson::Hover, Lesson::FrontLeft, Lesson::Scheduled] {
        let name = lesson.name();
        let mut batch = lesson.batch(options.seed, &agent.policy());
        for update in 1..=options.updates.get() {
            let metrics = agent.update(&batch.collect(&agent.policy())?)?;
            if update % 20 != 0 && update != options.updates.get() {
                continue;
            }
            let bytes = agent.policy().to_bytes()?;
            std::fs::write(options.output.join(format!("{name}-{update}.mpk")), &bytes)?;
            let assessment = lesson.evaluate(&load_policy(bytes)?)?;
            let passed = assessment.passes();
            let record = serde_json::json!({"lesson": name, "update": update, "passed": passed,
                "assessment": assessment, "optimizer_steps": metrics.optimizer_steps});
            println!("{record}");
            std::fs::write(
                options.output.join(format!("{name}-{update}.json")),
                record.to_string(),
            )?;
            if passed {
                break;
            }
            if update == options.updates.get() {
                return Err(
                    format!("Lesson {name} exhausted its update budget without passing.").into(),
                );
            }
        }
    }
    Ok(())
}
