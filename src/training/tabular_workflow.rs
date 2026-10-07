//! Shared train, evaluate, checkpoint, and visual command workflow for finite examples.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::runtime_io;
use crate::training::{
    AlgorithmKind, MetricRecord, MetricValue, MetricsWriter, RunConfig, RunId, RunPaths,
    TabularQConfig, TabularQPolicy, TabularQTrainer, TabularTransition,
};
use crate::{Env, EpisodeStatus};
use clap::{Parser, Subcommand};

/// Discrete action that maps to and from one table column.
pub trait IndexedAction: Copy {
    /// Convert one table column into an environment action.
    fn from_index(index: usize) -> Self;

    // Implementors must preserve the stable table-column mapping in checkpoints.
}

/// Environment-specific constants and visual callbacks used by the shared workflow.
pub trait TabularExample:
    Env<Observation = usize> + Default + Clone + Send + Sync + 'static
where
    Self::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    /// Stable lowercase artifact key.
    const ENV_NAME: &'static str;
    /// Gymnasium registry identifier written into run metadata.
    const GYMNASIUM_ID: &'static str;
    /// Encoded finite state count.
    const STATE_COUNT: usize;
    /// Encoded finite action count.
    const ACTION_COUNT: usize;
    /// Maximum transitions in one episode.
    const MAX_STEPS: usize;
    /// Safe default training episode budget.
    const DEFAULT_EPISODES: usize;
    /// Safe default held-out evaluation interval.
    const DEFAULT_EVAL_INTERVAL: usize;
    /// Safe default held-out episode count.
    const DEFAULT_EVAL_EPISODES: usize;
    /// Tuned default Q-value interpolation factor.
    const DEFAULT_LEARNING_RATE: f64;
    /// Discount factor.
    const GAMMA: f64;
    /// Exploration probability before decay.
    const EPSILON_START: f64 = 0.2;
    /// Exploration probability after decay.
    const EPSILON_END: f64 = 0.02;
    /// Expected action decisions per episode used to size linear decay.
    const EPSILON_DECAY_STEPS_PER_EPISODE: usize = 20;
    /// Held-out score required for example completion.
    const SOLVED_SCORE: f64;
    /// Default GIF output path.
    const GIF_PATH: &'static str;

    /// Return fixed legal-action rows when the environment publishes action masks.
    #[must_use]
    fn action_masks() -> Option<Vec<Vec<bool>>> {
        None
    }

    /// Return the selection score from held-out aggregate evidence.
    fn selection_score(evaluation: &TabularEvaluation) -> f64;

    /// Return whether one completed episode achieved the environment objective.
    fn is_success(final_observation: usize, total_reward: f64, status: EpisodeStatus) -> bool;

    /// Start the environment-specific rendered watch scene.
    ///
    /// # Errors
    ///
    /// Returns an error when the checkpoint or render scene cannot be loaded.
    fn watch(checkpoint: &Path) -> Result<(), Box<dyn Error>>;

    /// Render the environment-specific five-second GIF.
    ///
    /// # Errors
    ///
    /// Returns an error when checkpoint loading, capture, or encoding fails.
    fn gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>>;

    /// Render the standard dynamic-checkpoint 30-second training video.
    ///
    /// # Errors
    ///
    /// Returns an error when checkpoint loading, capture, or encoding fails.
    fn video(run_directory: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        // Keep this shared workflow stage consistent for every example that uses the public training boundary.
        #[cfg(feature = "render")]
        return crate::recording::record_training_video_from_gif_command(run_directory, output);
        #[cfg(not(feature = "render"))]
        {
            let _ = (run_directory, output);
            Err("video mode requires the `render` feature".into())
        }
    }
}

/// Standard command line shared by finite discrete examples.
#[derive(Debug, Parser)]
#[command(about = "Train, evaluate, watch, and record a finite Gymnasium example")]
struct Args {
    /// Operation to run. Visual watch is the default with rendering enabled.
    #[command(subcommand)]
    command: Option<Mode>,
}

/// Supported finite-example workflows.
#[derive(Debug, Subcommand)]
enum Mode {
    /// Train batched environments and publish checkpoints.
    Train {
        /// Total completed training episodes.
        #[arg(long)]
        episodes: Option<usize>,
        /// Independent environment lanes processed per batch.
        #[arg(long, default_value_t = 16)]
        num_envs: usize,
        /// Episodes between held-out evaluation and checkpoint publication.
        #[arg(long)]
        eval_interval: Option<usize>,
        /// Held-out episodes per evaluation.
        #[arg(long)]
        eval_episodes: Option<usize>,
        /// Tabular Q interpolation factor.
        #[arg(long)]
        learning_rate: Option<f64>,
        /// Root deterministic seed.
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional durable run identifier.
        #[arg(long)]
        run_id: Option<String>,
        /// Root artifact directory.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Evaluate a greedy checkpoint on held-out episodes.
    Eval {
        /// Checkpoint or run directory. The latest best checkpoint is the default.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Held-out episodes.
        #[arg(long)]
        episodes: Option<usize>,
        /// Held-out environment seed.
        #[arg(long, default_value_t = 9_001)]
        seed: u64,
        /// Root artifact directory used when no checkpoint is supplied.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Watch the best checkpoint in the environment-specific Bevy scene.
    Watch {
        /// Checkpoint or run directory. Missing checkpoints trigger default training.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Root artifact directory used when no checkpoint is supplied.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Render five seconds from the best checkpoint into an animated GIF.
    Gif {
        /// Checkpoint or run directory. Missing checkpoints trigger default training.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Output GIF path.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Root artifact directory used when no checkpoint is supplied.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Render a 30-second best, first, 33%, 66%, best checkpoint progression.
    Video {
        /// Checkpoint or run directory. Missing checkpoints trigger default training.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Output MP4 path.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Root artifact directory used when no checkpoint is supplied.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
}

/// Held-out aggregate policy evidence.
#[derive(Debug, Clone, Copy)]
pub struct TabularEvaluation {
    /// Evaluated episode count.
    pub episodes: usize,
    /// Fraction of episodes reaching the objective.
    pub success_rate: f64,
    /// Mean episode return.
    pub mean_reward: f64,
    /// Mean transitions per episode.
    pub mean_length: f64,
}

/// Completed training evidence.
#[derive(Debug)]
struct TrainReport {
    /// Durable run paths.
    paths: RunPaths,
    /// Random-table held-out score.
    initial: TabularEvaluation,
    /// Validation-selected held-out score.
    best: TabularEvaluation,
    /// Completed training episodes.
    episodes: usize,
}

/// Parse and execute the standard finite-example workflow.
///
/// # Errors
///
/// Returns an error when argument handling, training, evaluation, or rendering fails.
pub fn run_tabular_workflow<E>() -> Result<(), Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let args = Args::parse();
    let mode = args.command.unwrap_or_else(|| default_mode::<E>());
    match mode {
        Mode::Train {
            episodes,
            num_envs,
            eval_interval,
            eval_episodes,
            learning_rate,
            seed,
            run_id,
            runs_root,
        } => {
            let report = train::<E>(TrainOptions {
                episodes: episodes.unwrap_or(E::DEFAULT_EPISODES),
                num_envs,
                eval_interval: eval_interval.unwrap_or(E::DEFAULT_EVAL_INTERVAL),
                eval_episodes: eval_episodes.unwrap_or(E::DEFAULT_EVAL_EPISODES),
                learning_rate: learning_rate.unwrap_or(E::DEFAULT_LEARNING_RATE),
                seed,
                run_id,
                runs_root,
            })?;
            println!(
                "run={} episodes={} initial_score={:.3} best_score={:.3} checkpoint={}",
                report.paths.run_dir.display(),
                report.episodes,
                E::selection_score(&report.initial),
                E::selection_score(&report.best),
                report.paths.best_checkpoint.display()
            );
        }
        Mode::Eval {
            checkpoint,
            episodes,
            seed,
            runs_root,
        } => {
            let checkpoint = resolve_checkpoint::<E>(checkpoint.as_deref(), &runs_root)?;
            let trainer = TabularQTrainer::load(&checkpoint)?;
            let summary = evaluate::<E>(
                &trainer.policy(),
                episodes.unwrap_or(E::DEFAULT_EVAL_EPISODES),
                seed,
            )?;
            println!(
                "checkpoint={} episodes={} score={:.3} success_rate={:.3} mean_reward={:.3} mean_length={:.3}",
                checkpoint.display(),
                summary.episodes,
                E::selection_score(&summary),
                summary.success_rate,
                summary.mean_reward,
                summary.mean_length
            );
        }
        Mode::Watch {
            checkpoint,
            runs_root,
        } => {
            let checkpoint = resolve_or_train_checkpoint::<E>(checkpoint.as_deref(), &runs_root)?;
            E::watch(&checkpoint)?;
        }
        Mode::Gif {
            checkpoint,
            output,
            runs_root,
        } => {
            let checkpoint = resolve_or_train_checkpoint::<E>(checkpoint.as_deref(), &runs_root)?;
            let output = output.unwrap_or_else(|| PathBuf::from(E::GIF_PATH));
            E::gif(&checkpoint, &output)?;
            println!("gif={}", output.display());
        }
        Mode::Video {
            checkpoint,
            output,
            runs_root,
        } => {
            let checkpoint = resolve_or_train_checkpoint::<E>(checkpoint.as_deref(), &runs_root)?;
            let run_directory = checkpoint
                .parent()
                .ok_or("checkpoint has no run directory")?;
            let output = output.unwrap_or_else(|| {
                PathBuf::from("docs/videos").join(format!("{}.mp4", E::ENV_NAME))
            });
            E::video(run_directory, &output)?;
            println!("video={}", output.display());
        }
    }
    Ok(())
}

/// Choose visual mode for normal builds and explicit training for headless builds.
fn default_mode<E: TabularExample>() -> Mode
where
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if cfg!(feature = "render") {
        Mode::Watch {
            checkpoint: None,
            runs_root: PathBuf::from("runs"),
        }
    } else {
        Mode::Train {
            episodes: Some(E::DEFAULT_EPISODES),
            num_envs: 16,
            eval_interval: Some(E::DEFAULT_EVAL_INTERVAL),
            eval_episodes: Some(E::DEFAULT_EVAL_EPISODES),
            learning_rate: Some(E::DEFAULT_LEARNING_RATE),
            seed: 42,
            run_id: None,
            runs_root: PathBuf::from("runs"),
        }
    }
}

/// Complete train-mode configuration after example defaults are resolved.
#[derive(Debug)]
struct TrainOptions {
    /// Total completed episodes.
    episodes: usize,
    /// Environment lanes per outer batch.
    num_envs: usize,
    /// Evaluation interval in completed episodes.
    eval_interval: usize,
    /// Episodes per held-out evaluation.
    eval_episodes: usize,
    /// Q-value interpolation factor.
    learning_rate: f64,
    /// Root seed.
    seed: u64,
    /// Optional run identifier.
    run_id: Option<String>,
    /// Artifact root.
    runs_root: PathBuf,
}

/// Train batched environments with tabular Q-learning.
fn train<E>(options: TrainOptions) -> Result<TrainReport, Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if options.num_envs == 0 || options.eval_interval == 0 || options.eval_episodes == 0 {
        return Err("num-envs, eval-interval, and eval-episodes must be greater than zero".into());
    }
    let run_id = match options.run_id.as_deref() {
        Some(value) => RunId::new(value.to_owned())?,
        None => RunId::new(default_run_id()?)?,
    };
    let run = RunConfig::new(
        E::ENV_NAME,
        AlgorithmKind::Custom("tabular-q".into()),
        run_id,
        &options.runs_root,
    )?;
    let paths = run.paths();
    paths.create_new()?;
    write_run_config::<E>(&paths, &options)?;

    let config = TabularQConfig {
        learning_rate: options.learning_rate,
        gamma: E::GAMMA,
        epsilon_start: E::EPSILON_START,
        epsilon_end: E::EPSILON_END,
        epsilon_decay_steps: u64::try_from(
            options
                .episodes
                .saturating_mul(E::EPSILON_DECAY_STEPS_PER_EPISODE),
        )?,
        seed: options.seed,
    };
    let mut trainer =
        TabularQTrainer::new(E::STATE_COUNT, E::ACTION_COUNT, config, E::action_masks())?;
    let validation_seed = options.seed ^ 0xa5a5_a5a5;
    let initial = evaluate::<E>(&trainer.policy(), options.eval_episodes, validation_seed)?;
    let mut best = initial;
    trainer.save_best(&paths)?;
    trainer.save_latest(&paths)?;
    trainer.save_step(&paths, 0)?;
    let mut metrics = MetricsWriter::append(&paths.metrics_jsonl)?;
    write_evaluation::<E>(&mut metrics, 0, initial, true)?;

    let mut completed = 0usize;
    let mut next_eval = options.eval_interval;
    while completed < options.episodes {
        let batch = options.num_envs.min(options.episodes - completed);
        for lane in 0..batch {
            let episode = completed + lane;
            train_episode::<E>(
                &mut trainer,
                options.seed.wrapping_add(u64::try_from(episode)?),
            )?;
        }
        completed += batch;

        if completed >= next_eval || completed == options.episodes {
            let summary = evaluate::<E>(&trainer.policy(), options.eval_episodes, validation_seed)?;
            let is_best = E::selection_score(&summary) > E::selection_score(&best);
            trainer.save_latest(&paths)?;
            trainer.save_step(&paths, trainer.update_count())?;
            if is_best {
                best = summary;
                trainer.save_best(&paths)?;
            }
            write_evaluation::<E>(&mut metrics, completed, summary, is_best)?;
            println!(
                "episodes={completed} updates={} epsilon={:.3} score={:.3} success_rate={:.3} mean_reward={:.3} mean_length={:.2} best={:.3}",
                trainer.update_count(),
                trainer.current_epsilon(),
                E::selection_score(&summary),
                summary.success_rate,
                summary.mean_reward,
                summary.mean_length,
                E::selection_score(&best)
            );
            if E::selection_score(&best) >= E::SOLVED_SCORE {
                break;
            }
            next_eval = next_eval.saturating_add(options.eval_interval);
        }
    }

    write_summary::<E>(&paths, completed, initial, best)?;
    Ok(TrainReport {
        paths,
        initial,
        best,
        episodes: completed,
    })
}

/// Apply one complete environment episode to the shared table.
fn train_episode<E>(trainer: &mut TabularQTrainer, seed: u64) -> Result<(), Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mut env = E::default();
    let mut state = env.reset(Some(seed)).observation;
    for step_index in 0..E::MAX_STEPS {
        let action_index = trainer.select_action(state)?;
        let transition = env.step(E::Action::from_index(action_index));
        let truncated =
            step_index + 1 == E::MAX_STEPS && transition.status == EpisodeStatus::Continuing;
        trainer.update(TabularTransition {
            state,
            action: action_index,
            reward: transition.reward,
            next_state: transition.observation,
            terminated: transition.status == EpisodeStatus::Terminated,
            truncated,
        })?;
        state = transition.observation;
        if transition.status != EpisodeStatus::Continuing || truncated {
            break;
        }
    }
    Ok(())
}

/// Evaluate a greedy policy on a fixed held-out seed schedule.
fn evaluate<E>(
    policy: &TabularQPolicy,
    episodes: usize,
    seed: u64,
) -> Result<TabularEvaluation, Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mut successes = 0usize;
    let mut total_reward = 0.0;
    let mut total_length = 0usize;
    for episode in 0..episodes {
        let mut env = E::default();
        let mut state = env
            .reset(Some(seed.wrapping_add(u64::try_from(episode)?)))
            .observation;
        let mut episode_reward = 0.0;
        let mut final_status = EpisodeStatus::Continuing;
        for step_index in 0..E::MAX_STEPS {
            let action = E::Action::from_index(policy.greedy_action(state)?);
            let transition = env.step(action);
            state = transition.observation;
            episode_reward += transition.reward;
            total_length += 1;
            final_status = transition.status;
            if transition.status != EpisodeStatus::Continuing || step_index + 1 == E::MAX_STEPS {
                break;
            }
        }
        successes += usize::from(E::is_success(state, episode_reward, final_status));
        total_reward += episode_reward;
    }
    Ok(TabularEvaluation {
        episodes,
        success_rate: successes as f64 / episodes as f64,
        mean_reward: total_reward / episodes as f64,
        mean_length: total_length as f64 / episodes as f64,
    })
}

/// Write one evaluation row through the shared metric contract.
fn write_evaluation<E>(
    metrics: &mut MetricsWriter,
    episode: usize,
    summary: TabularEvaluation,
    is_best: bool,
) -> Result<(), Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    metrics.write_record(
        &MetricRecord::new(u64::try_from(episode)?)
            .with_field(
                "eval/score",
                MetricValue::Number(E::selection_score(&summary)),
            )
            .with_field(
                "eval/success_rate",
                MetricValue::Number(summary.success_rate),
            )
            .with_field("eval/mean_reward", MetricValue::Number(summary.mean_reward))
            .with_field("eval/mean_length", MetricValue::Number(summary.mean_length))
            .with_field("checkpoint/best", MetricValue::Bool(is_best)),
    )?;
    Ok(())
}

/// Persist the exact training profile beside its checkpoints.
fn write_run_config<E>(paths: &RunPaths, options: &TrainOptions) -> Result<(), Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let config = format!(
        "{{\"environment\":\"{}\",\"episodes\":{},\"num_envs\":{},\"eval_interval\":{},\"eval_episodes\":{},\"learning_rate\":{},\"gamma\":{},\"seed\":{}}}\n",
        E::GYMNASIUM_ID,
        options.episodes,
        options.num_envs,
        options.eval_interval,
        options.eval_episodes,
        options.learning_rate,
        E::GAMMA,
        options.seed
    );
    runtime_io::write(&paths.config_json, config)?;
    runtime_io::write(
        &paths.seeds_json,
        format!(
            "{{\"training\":{},\"validation\":{}}}\n",
            options.seed,
            options.seed ^ 0xa5a5_a5a5
        ),
    )?;
    Ok(())
}

/// Persist final baseline and selected-checkpoint evidence.
fn write_summary<E>(
    paths: &RunPaths,
    episodes: usize,
    initial: TabularEvaluation,
    best: TabularEvaluation,
) -> Result<(), Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    runtime_io::write(
        &paths.summary_json,
        format!(
            "{{\"episodes\":{episodes},\"initial_score\":{},\"best_score\":{},\"solved\":{}}}\n",
            E::selection_score(&initial),
            E::selection_score(&best),
            E::selection_score(&best) >= E::SOLVED_SCORE
        ),
    )?;
    Ok(())
}

/// Build a sortable run identifier without external state.
fn default_run_id() -> Result<String, Box<dyn Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(format!("run-{timestamp}"))
}

/// Resolve an explicit checkpoint or the newest validation-selected checkpoint.
fn resolve_checkpoint<E>(
    explicit: Option<&Path>,
    runs_root: &Path,
) -> Result<PathBuf, Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if let Some(path) = explicit {
        let checkpoint = if path.is_dir() {
            path.join("best.mpk")
        } else {
            path.to_path_buf()
        };
        if checkpoint.exists() {
            return Ok(checkpoint);
        }
        return Err(format!("checkpoint not found: {}", checkpoint.display()).into());
    }

    let root = runs_root.join(format!("{}-tabular-q", E::ENV_NAME));
    runtime_io::directory_entries(&root)?
        .into_iter()
        .filter_map(|entry| {
            let path = entry.join("best.mpk");
            let modified = runtime_io::modified(&path).ok()?;
            Some((modified, path))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
        .ok_or_else(|| format!("no best.mpk found under {}", root.display()).into())
}

/// Train with safe defaults when visual mode has no existing checkpoint.
fn resolve_or_train_checkpoint<E>(
    explicit: Option<&Path>,
    runs_root: &Path,
) -> Result<PathBuf, Box<dyn Error>>
where
    E: TabularExample,
    E::Action: IndexedAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if let Ok(checkpoint) = resolve_checkpoint::<E>(explicit, runs_root) {
        return Ok(checkpoint);
    }
    let report = train::<E>(TrainOptions {
        episodes: E::DEFAULT_EPISODES,
        num_envs: 16,
        eval_interval: E::DEFAULT_EVAL_INTERVAL,
        eval_episodes: E::DEFAULT_EVAL_EPISODES,
        learning_rate: E::DEFAULT_LEARNING_RATE,
        seed: 42,
        run_id: None,
        runs_root: runs_root.to_path_buf(),
    })?;
    Ok(report.paths.best_checkpoint)
}
