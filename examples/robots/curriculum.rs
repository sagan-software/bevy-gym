//! Train calm hover first, then retain the optimizer while learning disturbed recovery.

#![allow(
    clippy::disallowed_methods,
    reason = "This synchronous tutorial writes checkpoints without an async runtime."
)]

#[path = "curriculum/initial_checkpoint.rs"]
mod initial_checkpoint;
mod learning;
#[path = "curriculum/lesson.rs"]
mod lesson;
mod travel;

use std::{error::Error, num::NonZeroU32, path::PathBuf};

use clap::{Parser, ValueEnum};
use initial_checkpoint::InitialCheckpoint;
use learning::{baseline, load_policy, new_agent, SELECTION_SEEDS};
use lesson::Lesson;

/// Bound the work and choose where to save each evaluated checkpoint.
#[derive(Parser)]
struct Options {
    /// Train one lesson independently; omission runs hover followed by recovery.
    #[arg(long, value_enum)]
    lesson: Option<SelectedLesson>,
    /// Initialize standalone recovery from a recorded, qualified RL hover checkpoint.
    #[arg(long, value_enum)]
    initialize_from: Option<InitialCheckpoint>,
    /// Run frozen travel inference without collecting training samples or updating weights.
    #[arg(long, requires = "lesson", conflicts_with_all = ["updates", "initialize_from"])]
    evaluate_checkpoint: Option<PathBuf>,
    /// Evaluate every travel stage on the fixed held-out seed suite without training.
    #[arg(long, requires = "lesson", conflicts_with_all = ["updates", "initialize_from", "evaluate_checkpoint", "travel_stage", "seed"])]
    evaluate_held_out: Option<PathBuf>,
    /// Select the task distribution for frozen travel inference.
    #[arg(long, value_enum, requires = "evaluate_checkpoint")]
    travel_stage: Option<travel::stage::Stage>,
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

/// Select either a twelve-input control lesson or the goal-conditioned travel curriculum.
#[derive(Clone, Copy)]
enum SelectedLesson {
    /// Existing hover/recovery contract.
    Control(Lesson),
    /// Thirteen-input position and heading curriculum.
    Travel,
}

impl ValueEnum for SelectedLesson {
    /// Expose the closed lesson vocabulary without changing existing spellings.
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Control(Lesson::Hover),
            Self::Control(Lesson::Recovery),
            Self::Travel,
        ]
    }

    /// Keep artifact and CLI lesson names consistent.
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        let name = match self {
            Self::Control(lesson) => lesson.name(),
            Self::Travel => "travel",
        };
        Some(clap::builder::PossibleValue::new(name))
    }
}

impl ValueEnum for travel::stage::Stage {
    /// Keep stage spellings identical to saved checkpoint prefixes.
    fn value_variants<'a>() -> &'a [Self] {
        &Self::ALL
    }

    /// Translate the closed stage into a CLI value.
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.name()))
    }
}

/// Retain the agent across lessons and replace only the episode collection state.
fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    // Reject incompatible inference modes before loading weights or creating output.
    if options.evaluate_checkpoint.is_some()
        && !matches!(options.lesson, Some(SelectedLesson::Travel))
    {
        return Err("--evaluate-checkpoint requires --lesson travel".into());
    }
    if options.evaluate_held_out.is_some()
        && !matches!(options.lesson, Some(SelectedLesson::Travel))
    {
        return Err("--evaluate-held-out requires --lesson travel".into());
    }
    // Reject unsupported transitions before loading weights or creating artifacts.
    if options.initialize_from.is_some()
        && !matches!(
            options.lesson,
            Some(SelectedLesson::Control(Lesson::Recovery))
        )
    {
        return Err("qualified-hover initialization requires --lesson recovery".into());
    }
    // Travel has its own actor input width and qualified prerequisite; dispatch before construction.
    let lessons = match &options.lesson {
        Some(SelectedLesson::Control(lesson)) => std::slice::from_ref(lesson),
        Some(SelectedLesson::Travel) => {
            if let Some(path) = &options.evaluate_held_out {
                return travel::held_out::run(path);
            }
            if let Some(path) = &options.evaluate_checkpoint {
                return travel::infer(
                    options.travel_stage.unwrap_or(travel::stage::Stage::Near),
                    options.seed,
                    path,
                );
            }
            return travel::train(options.seed, options.updates, &options.output);
        }
        None => &[Lesson::Hover, Lesson::Recovery],
    };
    let mut agent = match options.initialize_from {
        Some(source) => source.load(options.seed)?,
        None => new_agent(options.seed)?,
    };
    std::fs::create_dir_all(&options.output)?;
    if let Some(source) = options.initialize_from {
        let record = source.record(options.seed);
        std::fs::write(options.output.join("transfer.json"), record.to_string())?;
        println!("{record}");
    }
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
