//! Train calm hover first, then retain the optimizer while learning disturbed recovery.

#![allow(
    clippy::disallowed_methods,
    reason = "This synchronous tutorial writes checkpoints without an async runtime."
)]

mod learning;
#[path = "curriculum/lesson.rs"]
mod lesson;

use std::{error::Error, num::NonZeroU32, path::PathBuf};

use clap::{Parser, ValueEnum};
use learning::{baseline, load_policy, new_agent, SELECTION_SEEDS};
use lesson::Lesson;

/// Bound the work and choose where to save each evaluated checkpoint.
#[derive(Parser)]
struct Options {
    /// Train one lesson independently; omission runs hover followed by recovery.
    #[arg(long, value_enum)]
    lesson: Option<Lesson>,
    /// Maximum 512-transition updates per lesson; exhaustion does not pass a lesson.
    #[arg(long, default_value = "600")]
    updates: NonZeroU32,
    /// Seed shared by initialization and each lesson's independent reset streams.
    #[arg(long, default_value = "7")]
    seed: u64,
    /// Directory for lesson-named checkpoints and scores.
    #[arg(long, default_value = "runs/drone-curriculum")]
    output: PathBuf,
}

// Keep CLI parsing outside the lesson module shared with the browser worker.
impl ValueEnum for Lesson {
    /// Offer the same closed lessons and order as curriculum execution.
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Hover, Self::Recovery]
    }

    /// Reuse the lesson's artifact name as its command-line spelling.
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.name()))
    }
}

/// Retain the agent across lessons and replace only the episode collection state.
fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    std::fs::create_dir_all(&options.output)?;
    let mut agent = new_agent(options.seed)?;
    // Borrow the selected lesson or the fixed curriculum without duplicating its runner.
    let lessons = match &options.lesson {
        Some(lesson) => std::slice::from_ref(lesson),
        None => &[Lesson::Hover, Lesson::Recovery],
    };
    for &lesson in lessons {
        let name = lesson.name();
        let mut batch = lesson.batch(options.seed, &agent.policy());
        for update in 1..=options.updates.get() {
            let metrics = agent.update(&batch.collect(&agent.policy())?)?;
            if update % 20 != 0 && update != options.updates.get() {
                continue;
            }
            // Score a reloaded frozen model; evaluation never becomes optimizer input.
            let bytes = agent.policy().to_bytes()?;
            std::fs::write(options.output.join(format!("{name}-{update}.mpk")), &bytes)?;
            let policy = load_policy(bytes)?;
            let episodes = lesson.evaluate(&policy)?;
            let passed = lesson::passes(&episodes);
            let record = serde_json::json!({"lesson": name, "update": update, "passed": passed,
                "episodes": episodes, "optimizer_steps": metrics.optimizer_steps});
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
    let constant_thrust = baseline(&SELECTION_SEEDS)?;
    let comparison = serde_json::to_string(&constant_thrust)?;
    println!("Constant-thrust recovery baseline: {comparison}");
    Ok(())
}
