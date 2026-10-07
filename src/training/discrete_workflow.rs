//! Shared batched DQN workflow for Gymnasium examples with discrete actions.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::runtime_io;
use crate::training::{
    AlgorithmKind, BevyTransitionCollector, DqnAgent, DqnConfig, DqnError, DqnPolicy, DqnUpdate,
    MetricRecord, MetricValue, MetricsWriter, RunConfig, RunId, RunPaths, SeedConfig,
};
use crate::{Env, EpisodeStatus};
use clap::{Parser, Subcommand};

/// Discrete action that maps from a DQN output column.
pub trait DqnAction: Copy {
    /// Convert one output column into an environment action.
    fn from_index(index: usize) -> Self;

    // Implementors must preserve the stable action-column mapping in checkpoints.

    /// Convert an environment action into one output column.
    fn as_index(self) -> usize;
}

/// Environment profile consumed by the shared DQN workflow.
pub trait DiscreteDqnExample:
    Env<Observation = Vec<f32>> + Default + Clone + Send + Sync + 'static
where
    Self::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    /// Stable lowercase artifact key.
    const ENV_NAME: &'static str;
    /// Gymnasium registry identifier.
    const GYMNASIUM_ID: &'static str;
    /// Exact environment observation width.
    const OBSERVATION_DIM: usize;
    /// Discrete action count.
    const ACTION_COUNT: usize;
    /// Default environment transition budget.
    const DEFAULT_TRAIN_STEPS: usize;
    /// Default held-out evaluation interval.
    const DEFAULT_EVAL_INTERVAL: usize;
    /// Default held-out episode count.
    const DEFAULT_EVAL_EPISODES: usize;
    /// Tuned default Adam learning rate.
    const DEFAULT_LEARNING_RATE: f64 = 3e-4;
    /// Default parallel environment count.
    const DEFAULT_NUM_ENVS: usize = 16;
    /// Default potential-based training reward scale.
    const DEFAULT_REWARD_SCALE: f64 = 0.0;
    /// Whether held-out rewards are zero-sum and admit the `PettingZoo` reward share.
    const ZERO_SUM_COMPARISON: bool = false;
    /// Multiplier for the discounted next-state value in the Bellman target.
    ///
    /// Alternating zero-sum games use `-1.0` because the next observation is
    /// from the opponent's perspective. Cooperative and single-agent examples
    /// keep the default `1.0`.
    const BOOTSTRAP_MULTIPLIER: f32 = 1.0;
    /// Initial transitions that may use environment-provided demonstrations.
    const GUIDED_TRANSITIONS: usize = 0;
    /// One in this many lanes uses a demonstration during the guided phase.
    const GUIDED_ACTION_DIVISOR: usize = 1;
    /// Minimum transitions collected before reporting the target reached.
    const MIN_TRAIN_STEPS: usize = 0;
    /// Held-out selection score required for completion.
    const SOLVED_MEAN_REWARD: f64;
    /// Default GIF output path.
    const GIF_PATH: &'static str;

    /// Return the example's tuned DQN profile.
    fn dqn_config(learning_rate: f64) -> DqnConfig;

    /// Encode an exact environment observation for the neural policy.
    #[must_use]
    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation.to_vec()
    }

    /// Return the legal-action mask for one exact environment observation.
    #[must_use]
    fn action_mask(_observation: &[f32]) -> Vec<bool> {
        vec![true; Self::ACTION_COUNT]
    }

    /// Configure one fresh environment for held-out evaluation.
    ///
    /// Multi-agent examples use this hook to keep self-play training separate
    /// from the documented trained-policy-versus-random evaluation protocol.
    fn prepare_evaluation(&mut self) {}

    /// Select one held-out action for the environment's current AEC agent.
    ///
    /// The default controls every agent with the learned policy. Multi-agent
    /// examples override this method to reproduce a documented comparison
    /// policy for agents that are not controlled by the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`DqnError`] when the observation or legal-action mask does not
    /// match the policy contract.
    fn evaluation_action(
        &mut self,
        policy: &DqnPolicy,
        observation: &[f32],
    ) -> Result<Self::Action, DqnError> {
        Self::deployment_action(policy, observation)
    }

    /// Return the learner reward while preserving the exact environment reward for metrics.
    #[must_use]
    fn training_reward(
        _observation: &[f32],
        _action: Self::Action,
        environment_reward: f64,
        _next_observation: &[f32],
        _status: EpisodeStatus,
        _reward_scale: f64,
    ) -> f64 {
        // Keep this shared workflow stage consistent for every example that uses the public training boundary.
        environment_reward
    }

    /// Return an optional demonstration action for early replay warm-up.
    #[must_use]
    fn guided_action(_observation: &[f32]) -> Option<Self::Action> {
        None
    }

    /// Select the deployed action used by held-out evaluation.
    ///
    /// # Errors
    ///
    /// Returns policy observation or action-mask validation errors.
    fn deployment_action(
        policy: &DqnPolicy,
        observation: &[f32],
    ) -> Result<Self::Action, DqnError> {
        let encoded = Self::encode_observation(observation);
        let action_mask = Self::action_mask(observation);
        policy
            .greedy_action_masked(&encoded, &action_mask)
            .map(Self::Action::from_index)
    }

    /// Return the scalar used for checkpoint selection and solve detection.
    #[must_use]
    fn selection_score(evaluation: &DiscreteEvaluation) -> f64 {
        evaluation.mean_reward
    }

    /// Return whether one completed episode achieved the environment objective.
    fn is_success(final_observation: &[f32], total_reward: f64, status: EpisodeStatus) -> bool;

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

/// Standard DQN example command line.
#[derive(Debug, Parser)]
#[command(about = "Train, evaluate, watch, and record a Gymnasium DQN example")]
struct Args {
    /// Operation to run. Visual watch is the default with rendering enabled.
    #[command(subcommand)]
    command: Option<Mode>,
}

/// Supported DQN example workflows.
#[derive(Debug, Subcommand)]
enum Mode {
    /// Train batched environments and publish checkpoints.
    Train {
        /// Total environment transitions.
        #[arg(long)]
        steps: Option<usize>,
        /// Independent environment lanes stepped per batch.
        #[arg(long)]
        num_envs: Option<usize>,
        /// Transitions between held-out evaluations.
        #[arg(long)]
        eval_interval: Option<usize>,
        /// Held-out episodes per evaluation.
        #[arg(long)]
        eval_episodes: Option<usize>,
        /// Adam learning rate.
        #[arg(long)]
        learning_rate: Option<f64>,
        /// Environment-specific potential-based reward scale.
        #[arg(long)]
        reward_scale: Option<f64>,
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
        /// Checkpoint or run directory.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Held-out episodes.
        #[arg(long)]
        episodes: Option<usize>,
        /// Held-out environment seed.
        #[arg(long, default_value_t = 9_001)]
        seed: u64,
        /// Root artifact directory.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Watch the best checkpoint in the environment-specific scene.
    Watch {
        /// Checkpoint or run directory. Missing checkpoints trigger training.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Root artifact directory.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Render five seconds from the best checkpoint into an animated GIF.
    Gif {
        /// Checkpoint or run directory. Missing checkpoints trigger training.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Output GIF path.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Root artifact directory.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
    /// Render a 30-second best, first, 33%, 66%, best checkpoint progression.
    Video {
        /// Checkpoint or run directory. Missing checkpoints trigger training.
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Output MP4 path.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Root artifact directory.
        #[arg(long, default_value = "runs")]
        runs_root: PathBuf,
    },
}

/// Held-out aggregate policy evidence.
#[derive(Debug, Clone, Copy)]
pub struct DiscreteEvaluation {
    /// Evaluated episode count.
    pub episodes: usize,
    /// Fraction of episodes reaching the objective.
    pub success_rate: f64,
    /// Trained agent's share of absolute decisive reward in zero-sum games.
    pub comparison_win_rate: f64,
    /// Mean episode return.
    pub mean_reward: f64,
    /// Mean transitions per episode.
    pub mean_length: f64,
    /// Lowest episode return.
    pub min_reward: f64,
    /// Highest episode return.
    pub max_reward: f64,
}

/// Resolved train-mode values.
#[derive(Debug)]
struct TrainOptions {
    /// Total environment transitions.
    steps: usize,
    /// Parallel environment lanes.
    num_envs: usize,
    /// Evaluation interval.
    eval_interval: usize,
    /// Evaluation episode count.
    eval_episodes: usize,
    /// Adam learning rate.
    learning_rate: f64,
    /// Potential-based training reward scale.
    reward_scale: f64,
    /// Root seed.
    seed: u64,
    /// Optional run identifier.
    run_id: Option<String>,
    /// Artifact root.
    runs_root: PathBuf,
}

/// Completed training evidence.
#[derive(Debug)]
struct TrainReport {
    /// Durable artifact paths.
    paths: RunPaths,
    /// Random-policy held-out score.
    initial: DiscreteEvaluation,
    /// Validation-selected held-out score.
    best: DiscreteEvaluation,
    /// Completed environment transitions.
    steps: usize,
}

/// Parse and execute the standard DQN workflow.
///
/// # Errors
///
/// Returns an error when argument handling, training, evaluation, or rendering fails.
pub fn run_discrete_workflow<E>() -> Result<(), Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mode = Args::parse().command.unwrap_or_else(|| default_mode::<E>());
    match mode {
        Mode::Train {
            steps,
            num_envs,
            eval_interval,
            eval_episodes,
            learning_rate,
            reward_scale,
            seed,
            run_id,
            runs_root,
        } => {
            let report = train::<E>(TrainOptions {
                steps: steps.unwrap_or(E::DEFAULT_TRAIN_STEPS),
                num_envs: num_envs.unwrap_or(E::DEFAULT_NUM_ENVS),
                eval_interval: eval_interval.unwrap_or(E::DEFAULT_EVAL_INTERVAL),
                eval_episodes: eval_episodes.unwrap_or(E::DEFAULT_EVAL_EPISODES),
                learning_rate: learning_rate.unwrap_or(E::DEFAULT_LEARNING_RATE),
                reward_scale: reward_scale.unwrap_or(E::DEFAULT_REWARD_SCALE),
                seed,
                run_id,
                runs_root,
            })?;
            println!(
                "run={} steps={} initial_mean_reward={:.3} best_mean_reward={:.3} checkpoint={}",
                report.paths.run_dir.display(),
                report.steps,
                report.initial.mean_reward,
                report.best.mean_reward,
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
            let policy = load_policy::<E>(&checkpoint)?;
            let summary =
                evaluate::<E>(&policy, episodes.unwrap_or(E::DEFAULT_EVAL_EPISODES), seed)?;
            println!(
                "checkpoint={} episodes={} success_rate={:.3} comparison_win_rate={:.3} mean_reward={:.3} mean_length={:.3} min_reward={:.3} max_reward={:.3}",
                checkpoint.display(),
                summary.episodes,
                summary.success_rate,
                summary.comparison_win_rate,
                summary.mean_reward,
                summary.mean_length,
                summary.min_reward,
                summary.max_reward
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

/// Select visual mode for render builds and training for headless builds.
fn default_mode<E: DiscreteDqnExample>() -> Mode
where
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if cfg!(feature = "render") {
        Mode::Watch {
            checkpoint: None,
            runs_root: PathBuf::from("runs"),
        }
    } else {
        Mode::Train {
            steps: Some(E::DEFAULT_TRAIN_STEPS),
            num_envs: Some(E::DEFAULT_NUM_ENVS),
            eval_interval: Some(E::DEFAULT_EVAL_INTERVAL),
            eval_episodes: Some(E::DEFAULT_EVAL_EPISODES),
            learning_rate: Some(E::DEFAULT_LEARNING_RATE),
            reward_scale: Some(E::DEFAULT_REWARD_SCALE),
            seed: 42,
            run_id: None,
            runs_root: PathBuf::from("runs"),
        }
    }
}

/// Train one DQN through the repository's synchronous Bevy collector.
fn train<E>(options: TrainOptions) -> Result<TrainReport, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let DiscreteTrainingState {
        paths,
        seeds,
        hidden_sizes,
        mut agent,
        initial,
        mut best,
        mut metrics,
    } = initialize_training::<E>(&options)?;
    let reset_seeds = seeds;
    let mut collector = BevyTransitionCollector::new(
        |_| E::default(),
        options.num_envs,
        move |env_id, episode| Some(reset_seeds.environment_episode(env_id, episode)),
    )?;
    let mut next_eval = options.eval_interval;
    let mut completed_episodes = 0usize;
    let mut latest_update = None;
    while agent.global_steps() < u64::try_from(options.steps)? {
        let guided_step = usize::try_from(agent.global_steps())?;
        let within_guided_phase = guided_step < E::GUIDED_TRANSITIONS;
        let actions = collector
            .requests()
            .iter()
            .enumerate()
            .map(|(env_id, request)| {
                let encoded = E::encode_observation(&request.observation);
                let action_mask = E::action_mask(&request.observation);
                let selected = agent
                    .select_action_masked(&encoded, &action_mask)
                    .map(|selection| E::Action::from_index(selection.action_index))?;
                let guided_lane = E::GUIDED_ACTION_DIVISOR > 0
                    && (guided_step + env_id).is_multiple_of(E::GUIDED_ACTION_DIVISOR);
                Ok::<E::Action, DqnError>(if within_guided_phase && guided_lane {
                    E::guided_action(&request.observation).unwrap_or(selected)
                } else {
                    selected
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batch = collector.step(actions)?;
        for event in &batch.transitions {
            let transition = &event.transition;
            let observation = E::encode_observation(&transition.observation);
            let next_observation = E::encode_observation(&transition.next_observation);
            let next_action_mask = E::action_mask(&transition.next_observation);
            let learner_reward = E::training_reward(
                &transition.observation,
                transition.action,
                transition.reward,
                &transition.next_observation,
                transition.status,
                options.reward_scale,
            );
            if let Some(update) = agent.observe_masked_with_bootstrap_multiplier(
                &observation,
                transition.action.as_index(),
                learner_reward,
                &next_observation,
                &next_action_mask,
                transition.status,
                E::BOOTSTRAP_MULTIPLIER,
            )? {
                latest_update = Some(update);
            }
        }
        completed_episodes += batch.episode_ends.len();
        let step = usize::try_from(agent.global_steps())?;

        if step >= next_eval || step == options.steps {
            let solved = evaluate_training_checkpoint::<E>(
                &agent,
                &paths,
                &mut metrics,
                &options,
                seeds.validation,
                step,
                completed_episodes,
                latest_update,
                &mut best,
            );
            if solved? {
                break;
            }
            next_eval = next_eval.saturating_add(options.eval_interval);
        }
    }

    let steps = usize::try_from(agent.global_steps())?;
    write_summary(&paths, steps, initial, best, &hidden_sizes)?;
    Ok(TrainReport {
        paths,
        initial,
        best,
        steps,
    })
}

/// Evaluate, persist, and report one periodic training checkpoint.
fn evaluate_training_checkpoint<E>(
    agent: &DqnAgent,
    paths: &RunPaths,
    metrics: &mut MetricsWriter,
    options: &TrainOptions,
    validation_seed: u64,
    step: usize,
    completed_episodes: usize,
    latest_update: Option<DqnUpdate>,
    best: &mut DiscreteEvaluation,
) -> Result<bool, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    let summary = evaluate::<E>(&agent.policy(), options.eval_episodes, validation_seed)?;
    let is_best = E::selection_score(&summary) > E::selection_score(best);
    agent.save_latest(paths)?;
    agent
        .policy()
        .save(paths.checkpoints_dir.join(format!("step-{step}.mpk")))?;
    if is_best {
        *best = summary;
        agent.save_best(paths)?;
    }
    write_evaluation(metrics, step, summary, is_best)?;
    let mut record = MetricRecord::new(u64::try_from(step)?)
        .with_field(
            "train/episodes",
            MetricValue::Number(completed_episodes as f64),
        )
        .with_field("train/epsilon", MetricValue::Number(agent.epsilon()))
        .with_field(
            "train/optimizer_updates",
            MetricValue::Number(agent.optimizer_steps() as f64),
        );
    if let Some(update) = latest_update {
        record = record.with_field("train/loss", MetricValue::Number(update.loss));
    }
    metrics.write_record(&record)?;
    print_training_progress(
        agent,
        step,
        completed_episodes,
        summary,
        *best,
        latest_update.map(|update| update.loss),
    );
    Ok(step >= E::MIN_TRAIN_STEPS && E::selection_score(best) >= E::SOLVED_MEAN_REWARD)
}

/// Print one compact validation record for an interactive training run.
fn print_training_progress(
    agent: &DqnAgent,
    step: usize,
    completed_episodes: usize,
    summary: DiscreteEvaluation,
    best: DiscreteEvaluation,
    latest_loss: Option<f64>,
) {
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    println!(
        "steps={step} episodes={completed_episodes} updates={} epsilon={:.3} loss={:.6} success_rate={:.3} comparison_win_rate={:.3} mean_reward={:.3} mean_length={:.2} best={:.3}",
        agent.optimizer_steps(),
        agent.epsilon(),
        latest_loss.unwrap_or(f64::NAN),
        summary.success_rate,
        summary.comparison_win_rate,
        summary.mean_reward,
        summary.mean_length,
        best.mean_reward
    );
}

/// Initialized DQN learner state and durable outputs.
struct DiscreteTrainingState {
    /// Durable run paths.
    paths: RunPaths,
    /// Derived deterministic seeds.
    seeds: SeedConfig,
    /// Hidden-layer widths persisted in the summary.
    hidden_sizes: Vec<usize>,
    /// Mutable DQN learner.
    agent: DqnAgent,
    /// Random-policy held-out baseline.
    initial: DiscreteEvaluation,
    /// Current validation-selected result.
    best: DiscreteEvaluation,
    /// Append-only metric writer.
    metrics: MetricsWriter,
}

/// Validate options and initialize checkpoints, metrics, and learner state.
fn initialize_training<E>(options: &TrainOptions) -> Result<DiscreteTrainingState, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if options.num_envs == 0 || options.eval_interval == 0 || options.eval_episodes == 0 {
        return Err("num-envs, eval-interval, and eval-episodes must be greater than zero".into());
    }
    if !options.steps.is_multiple_of(options.num_envs) {
        return Err("steps must be divisible by num-envs".into());
    }
    let run_id = match options.run_id.as_deref() {
        Some(value) => RunId::new(value.to_owned())?,
        None => RunId::new(default_run_id()?)?,
    };
    let run = RunConfig::new(E::ENV_NAME, AlgorithmKind::Dqn, run_id, &options.runs_root)?;
    let paths = run.paths();
    paths.create_new()?;
    let seeds = SeedConfig::from_root(options.seed);
    write_run_config::<E>(&paths, options, seeds)?;

    let learner_config = E::dqn_config(options.learning_rate);
    let hidden_sizes = learner_config.hidden_sizes.clone();
    let agent = DqnAgent::new(E::OBSERVATION_DIM, E::ACTION_COUNT, learner_config, seeds)?;
    let initial = evaluate::<E>(&agent.policy(), options.eval_episodes, seeds.validation)?;
    let best = initial;
    agent.save_best(&paths)?;
    agent.save_latest(&paths)?;
    agent
        .policy()
        .save(paths.checkpoints_dir.join("step-0.mpk"))?;
    let mut metrics = MetricsWriter::append(&paths.metrics_jsonl)?;
    write_evaluation(&mut metrics, 0, initial, true)?;
    Ok(DiscreteTrainingState {
        paths,
        seeds,
        hidden_sizes,
        agent,
        initial,
        best,
        metrics,
    })
}

/// Evaluate a greedy DQN on a fixed held-out seed schedule.
fn evaluate<E>(
    policy: &DqnPolicy,
    episodes: usize,
    seed: u64,
) -> Result<DiscreteEvaluation, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mut successes = 0usize;
    let mut positive_decisive_reward = 0.0;
    let mut absolute_decisive_reward = 0.0;
    let mut total_reward = 0.0;
    let mut total_length = 0usize;
    let mut min_reward = f64::INFINITY;
    let mut max_reward = f64::NEG_INFINITY;
    for episode in 0..episodes {
        let mut env = E::default();
        env.prepare_evaluation();
        let mut observation = env
            .reset(Some(seed.wrapping_add(u64::try_from(episode)?)))
            .observation;
        let mut episode_reward = 0.0;
        let mut episode_length = 0usize;
        let final_status = loop {
            let action = env.evaluation_action(policy, &observation)?;
            let transition = env.step(action);
            episode_reward += transition.reward;
            episode_length += 1;
            observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                break transition.status;
            }
        };
        successes += usize::from(E::is_success(&observation, episode_reward, final_status));
        if E::ZERO_SUM_COMPARISON {
            positive_decisive_reward += episode_reward.max(0.0);
            absolute_decisive_reward += episode_reward.abs();
        }
        total_reward += episode_reward;
        total_length += episode_length;
        min_reward = min_reward.min(episode_reward);
        max_reward = max_reward.max(episode_reward);
    }
    Ok(DiscreteEvaluation {
        episodes,
        success_rate: successes as f64 / episodes as f64,
        comparison_win_rate: if absolute_decisive_reward > 0.0 {
            positive_decisive_reward / absolute_decisive_reward
        } else {
            0.0
        },
        mean_reward: total_reward / episodes as f64,
        mean_length: total_length as f64 / episodes as f64,
        min_reward,
        max_reward,
    })
}

/// Write one held-out evaluation row.
fn write_evaluation(
    metrics: &mut MetricsWriter,
    step: usize,
    summary: DiscreteEvaluation,
    is_best: bool,
) -> Result<(), Box<dyn Error>> {
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    metrics.write_record(
        &MetricRecord::new(u64::try_from(step)?)
            .with_field(
                "eval/success_rate",
                MetricValue::Number(summary.success_rate),
            )
            .with_field(
                "eval/comparison_win_rate",
                MetricValue::Number(summary.comparison_win_rate),
            )
            .with_field("eval/mean_reward", MetricValue::Number(summary.mean_reward))
            .with_field("eval/mean_length", MetricValue::Number(summary.mean_length))
            .with_field("checkpoint/best", MetricValue::Bool(is_best)),
    )?;
    Ok(())
}

/// Resolve an explicit checkpoint or the newest validation-selected checkpoint.
fn resolve_checkpoint<E>(
    explicit: Option<&Path>,
    runs_root: &Path,
) -> Result<PathBuf, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
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
    let root = runs_root.join(format!("{}-dqn", E::ENV_NAME));
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

/// Train with defaults when visual mode has no checkpoint.
fn resolve_or_train_checkpoint<E>(
    explicit: Option<&Path>,
    runs_root: &Path,
) -> Result<PathBuf, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if let Ok(checkpoint) = resolve_checkpoint::<E>(explicit, runs_root) {
        return Ok(checkpoint);
    }
    let report = train::<E>(TrainOptions {
        steps: E::DEFAULT_TRAIN_STEPS,
        num_envs: E::DEFAULT_NUM_ENVS,
        eval_interval: E::DEFAULT_EVAL_INTERVAL,
        eval_episodes: E::DEFAULT_EVAL_EPISODES,
        learning_rate: E::DEFAULT_LEARNING_RATE,
        reward_scale: E::DEFAULT_REWARD_SCALE,
        seed: 42,
        run_id: None,
        runs_root: runs_root.to_path_buf(),
    })?;
    Ok(report.paths.best_checkpoint)
}

/// Load one policy using the example's architecture profile.
fn load_policy<E>(path: &Path) -> Result<DqnPolicy, Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let config = E::dqn_config(E::DEFAULT_LEARNING_RATE);
    Ok(DqnPolicy::load(
        path,
        E::OBSERVATION_DIM,
        E::ACTION_COUNT,
        &config.hidden_sizes,
    )?)
}

/// Persist the exact training profile.
fn write_run_config<E>(
    paths: &RunPaths,
    options: &TrainOptions,
    seeds: SeedConfig,
) -> Result<(), Box<dyn Error>>
where
    E: DiscreteDqnExample,
    E::Action: DqnAction,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    runtime_io::write(
        &paths.config_json,
        format!(
            "{{\"algorithm\":\"dqn\",\"environment\":\"{}\",\"steps\":{},\"num_envs\":{},\"eval_interval\":{},\"eval_episodes\":{},\"learning_rate\":{},\"reward_scale\":{}}}\n",
            E::GYMNASIUM_ID,
            options.steps,
            options.num_envs,
            options.eval_interval,
            options.eval_episodes,
            options.learning_rate,
            options.reward_scale
        ),
    )?;
    runtime_io::write(
        &paths.seeds_json,
        format!(
            "{{\"root\":{},\"validation\":{},\"test\":{},\"demo\":{}}}\n",
            options.seed, seeds.validation, seeds.test, seeds.demo
        ),
    )?;
    Ok(())
}

/// Persist final baseline and selected-checkpoint evidence.
fn write_summary(
    paths: &RunPaths,
    steps: usize,
    initial: DiscreteEvaluation,
    best: DiscreteEvaluation,
    hidden_sizes: &[usize],
) -> Result<(), Box<dyn Error>> {
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    runtime_io::write(
        &paths.summary_json,
        format!(
            "{{\"steps\":{steps},\"initial_mean_reward\":{},\"best_mean_reward\":{},\"success_rate\":{},\"comparison_win_rate\":{},\"hidden_sizes\":{:?}}}\n",
            initial.mean_reward,
            best.mean_reward,
            best.success_rate,
            best.comparison_win_rate,
            hidden_sizes
        ),
    )?;
    Ok(())
}

/// Build a sortable run identifier.
fn default_run_id() -> Result<String, Box<dyn Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(format!("run-{timestamp}"))
}
