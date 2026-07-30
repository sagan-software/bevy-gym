//! Shared implementation for every ecosystem curriculum example.

#[cfg(feature = "render")]
mod demo;
mod domain;
#[cfg(feature = "render")]
mod rendering;
mod rng;
mod simulation;
mod training;
#[cfg(feature = "render")]
mod video;

use std::error::Error;
use std::io::{self, Write as _};

// The recurrent trainer slice will use Burn directly. Retain the example-level
// dependency signal while the deterministic environment slice lands first.
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;

pub(crate) use domain::CurriculumStage;
use domain::{LocomotionAction, SimulationConfig};
use rng::SplitMix64;
use simulation::Ecosystem;

/// Closed command modes shared by every ecosystem example.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Interactive visual training and policy playback.
    Demo,
    /// Short deterministic environment check.
    Smoke,
    /// Headless recurrent PPO training.
    Train,
    /// Fixed-seed checkpoint evaluation.
    Eval,
    /// Interactive checkpoint playback without training.
    Watch,
    /// Deterministic checkpoint video generation.
    Video,
    /// Command help.
    Help,
}

impl Mode {
    /// Resolve the visual-first default and explicit command vocabulary.
    fn from_first_argument(value: Option<&str>, has_render: bool) -> Result<Self, String> {
        // The compiled feature controls only the absent-argument default.
        match value {
            None if has_render => Ok(Self::Demo),
            None | Some("headless" | "--headless" | "train") => Ok(Self::Train),
            Some("demo") => Ok(Self::Demo),
            Some("smoke" | "--smoke") => Ok(Self::Smoke),
            Some("eval") => Ok(Self::Eval),
            Some("watch") => Ok(Self::Watch),
            Some("video") => Ok(Self::Video),
            Some("--help" | "-h") => Ok(Self::Help),
            Some(value) => Err(format!("unknown ecosystem mode {value:?}; use --help")),
        }
    }
}

/// Run the selected curriculum-stage command.
pub(crate) fn run(stage: CurriculumStage) -> Result<(), Box<dyn Error>> {
    debug_assert!(CurriculumStage::ALL.contains(&stage));
    // Cargo arguments after `--` belong to this closed example command.
    let mut arguments = std::env::args().skip(1);
    let first_argument = arguments.next();
    let mode = Mode::from_first_argument(first_argument.as_deref(), cfg!(feature = "render"))?;
    match mode {
        #[cfg(feature = "render")]
        Mode::Demo => demo::run_demo(stage, arguments),
        #[cfg(not(feature = "render"))]
        Mode::Demo => Err("demo mode requires the default `render` feature".into()),
        Mode::Smoke => run_smoke(stage),
        Mode::Train => training::run_training(stage, arguments),
        Mode::Eval => training::run_evaluation(stage, arguments),
        #[cfg(feature = "render")]
        Mode::Watch => rendering::run_watch(stage, arguments),
        #[cfg(not(feature = "render"))]
        Mode::Watch => Err("watch mode requires the default `render` feature".into()),
        #[cfg(feature = "render")]
        Mode::Video => video::run_video(stage, arguments),
        #[cfg(not(feature = "render"))]
        Mode::Video => Err("video mode requires the default `render` feature".into()),
        Mode::Help => write_help(stage),
    }
}

/// Exercise deterministic reset, Avian physics, needs, resources, and local observations.
fn run_smoke(stage: CurriculumStage) -> Result<(), Box<dyn Error>> {
    let config = SimulationConfig::for_stage(stage)?;
    let configured_agents = config.agent_count();
    let mut ecosystem = Ecosystem::new(config, 42)?;
    let mut action_rng = SplitMix64::new(7);

    // A short joint rollout is the nearest runtime check before trainer work.
    for _ in 0..32 {
        let actions = ecosystem
            .living_agents()
            .into_iter()
            .map(|agent| {
                let action = LocomotionAction::new(
                    action_rng.f32_between(-1.0, 1.0),
                    action_rng.f32_between(-1.0, 1.0),
                    action_rng.f32_between(-1.0, 1.0),
                )?;
                Ok((agent, action))
            })
            .collect::<Result<Vec<_>, domain::ActionError>>()?;
        if ecosystem.step(&actions)?.is_done {
            break;
        }
    }

    let snapshot = ecosystem.snapshot();
    writeln!(
        io::stdout().lock(),
        "stage={} step={} living={}/{} food={} well={:.2} observations={} global_state={}",
        stage.as_key(),
        snapshot.step,
        snapshot.living_agents,
        configured_agents,
        snapshot.food_count,
        snapshot.well_water,
        domain::LOCAL_OBSERVATION_SIZE,
        domain::GLOBAL_STATE_SIZE,
    )?;
    Ok(())
}

/// Write the stage command surface.
fn write_help(stage: CurriculumStage) -> Result<(), Box<dyn Error>> {
    // Repeat the concrete target so every line is ready to paste unchanged.
    writeln!(
        io::stdout().lock(),
        "{}\n\nUsage:\n  cargo run --example ecosystem-{}\n  cargo run --example ecosystem-{} -- demo [--iterations N] [--episode-seconds N] [--seed N]\n  cargo run --no-default-features --release --example ecosystem-{} -- train [--iterations N] [--episode-seconds N] [--seed N]\n  cargo run --example ecosystem-{} -- eval --checkpoint <run-or-mpk>\n  cargo run --example ecosystem-{} -- watch --checkpoint <run-or-mpk> [--seed N] [--speed N]\n  cargo run --example ecosystem-{} -- video --checkpoint <run-dir> --output <video.mp4>\n  cargo run --example ecosystem-{} -- smoke\n\nDemo controls are in the Inspector-egui panel. Press F1 for the Bevy world inspector.",
        stage.title(),
        stage.as_key(),
        stage.as_key(),
        stage.as_key(),
        stage.as_key(),
        stage.as_key(),
        stage.as_key(),
        stage.as_key(),
    )?;
    Ok(())
}

#[cfg(test)]
mod command_tests {
    use super::*;

    /// Default routing must preserve the visual-first example contract.
    #[test]
    fn visual_build_defaults_to_demo_and_headless_build_defaults_to_training() {
        assert_eq!(Mode::from_first_argument(None, true), Ok(Mode::Demo));
        assert_eq!(Mode::from_first_argument(None, false), Ok(Mode::Train));
    }

    /// Headless training remains an explicit command in a visual build.
    #[test]
    fn headless_alias_selects_training() {
        assert_eq!(
            Mode::from_first_argument(Some("headless"), true),
            Ok(Mode::Train)
        );
    }
}
