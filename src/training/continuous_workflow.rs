//! Shared batched recurrent PPO workflow for bounded continuous examples.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::runtime_io;
use crate::training::{
    AlgorithmKind, MetricRecord, MetricValue, MetricsWriter, RecurrentBehaviorSample,
    RecurrentMemory, RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoPolicy,
    RecurrentPpoSequence, RecurrentSampler, RunConfig, RunId, RunPaths, SeedConfig,
};
use crate::{Env, EpisodeStatus};
use clap::{Parser, Subcommand};

/// Environment profile consumed by the shared bounded-action PPO workflow.
pub trait ContinuousPpoExample:
    Env<Observation = Vec<f32>, Action = Vec<f32>> + Default + Send + Sync + 'static
{
    /// Stable lowercase artifact key.
    const ENV_NAME: &'static str;
    /// Gymnasium registry identifier.
    const GYMNASIUM_ID: &'static str;
    /// Exact observation width.
    const OBSERVATION_DIM: usize;
    /// Inclusive action lower bounds.
    const ACTION_LOW: &'static [f32];
    /// Inclusive action upper bounds.
    const ACTION_HIGH: &'static [f32];
    /// Default total transition budget.
    const DEFAULT_TRAIN_STEPS: usize;
    /// Transitions collected per environment before one PPO update.
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    /// Default parallel environment count.
    const DEFAULT_NUM_ENVS: usize = 16;
    /// Evaluation interval in collected transitions.
    const DEFAULT_EVAL_INTERVAL: usize;
    /// Default held-out episode count.
    const DEFAULT_EVAL_EPISODES: usize;
    /// Tuned actor learning rate.
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 3e-4;
    /// Tuned critic learning rate.
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 3e-4;
    /// Training-only reward multiplier.
    const DEFAULT_REWARD_SCALE: f64 = 1.0;
    /// Mean held-out return required for completion.
    const SOLVED_MEAN_REWARD: f64;
    /// Default GIF output path.
    const GIF_PATH: &'static str;
    /// Expert samples used to initialize difficult continuous actors.
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 0;
    /// Passes over the expert sample set before PPO collection.
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 0;
    /// Expert samples consumed by one actor-only optimizer update.
    const BEHAVIOR_PRETRAINING_BATCH_SIZE: usize = 256;
    /// Whether actor memory persists across fully observed environment steps.
    const RETAIN_RECURRENT_MEMORY: bool = true;

    /// Return the example's PPO model and optimizer profile.
    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig;

    /// Encode one exact environment observation for actor and critic input.
    #[must_use]
    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation.to_vec()
    }

    /// Build deterministic expert samples for optional actor pretraining.
    #[must_use]
    fn behavior_demonstrations(_sample_count: usize, _seed: u64) -> Vec<RecurrentBehaviorSample> {
        Vec::new()
    }

    /// Return the learner reward while retaining the environment reward for evaluation.
    #[must_use]
    fn training_reward(
        _observation: &[f32],
        _action: &[f32],
        environment_reward: f64,
        _next_observation: &[f32],
        _status: EpisodeStatus,
        reward_scale: f64,
    ) -> f64 {
        // Keep this shared workflow stage consistent for every example that uses the public training boundary.
        environment_reward * reward_scale
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

/// Standard PPO example command line.
#[derive(Debug, Parser)]
#[command(about = "Train, evaluate, watch, and record a continuous Gymnasium example")]
struct Args {
    /// Operation to run. Visual watch is the default with rendering enabled.
    #[command(subcommand)]
    command: Option<Mode>,
}

/// Supported bounded-action workflows.
#[derive(Debug, Subcommand)]
enum Mode {
    /// Train batched environments and publish checkpoints.
    Train {
        /// Approximate total environment transitions.
        #[arg(long)]
        steps: Option<usize>,
        /// Independent environment lanes collected per PPO rollout.
        #[arg(long)]
        num_envs: Option<usize>,
        /// Transitions between held-out evaluations.
        #[arg(long)]
        eval_interval: Option<usize>,
        /// Held-out episodes per evaluation.
        #[arg(long)]
        eval_episodes: Option<usize>,
        /// Actor Adam learning rate.
        #[arg(long)]
        actor_learning_rate: Option<f64>,
        /// Critic Adam learning rate.
        #[arg(long)]
        critic_learning_rate: Option<f64>,
        /// Training-only reward multiplier.
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
    /// Evaluate a deterministic checkpoint on held-out episodes.
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
pub struct ContinuousEvaluation {
    /// Evaluated episode count.
    pub episodes: usize,
    /// Fraction of episodes reaching the objective.
    pub success_rate: f64,
    /// Mean episode return.
    pub mean_reward: f64,
    /// Mean transitions per episode.
    pub mean_length: f64,
    /// Lowest episode return.
    pub min_reward: f64,
    /// Highest episode return.
    pub max_reward: f64,
}

/// Resolved training values.
#[derive(Debug)]
struct TrainOptions {
    /// Approximate transition budget.
    steps: usize,
    /// Parallel environment lanes.
    num_envs: usize,
    /// Evaluation interval.
    eval_interval: usize,
    /// Evaluation episode count.
    eval_episodes: usize,
    /// Actor learning rate.
    actor_learning_rate: f64,
    /// Critic learning rate.
    critic_learning_rate: f64,
    /// Training reward multiplier.
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
    /// Durable run paths.
    paths: RunPaths,
    /// Initial deterministic evaluation.
    initial: ContinuousEvaluation,
    /// Validation-selected evaluation.
    best: ContinuousEvaluation,
    /// Collected transitions.
    steps: usize,
}

/// One persistent environment lane.
struct Lane<E: ContinuousPpoExample> {
    /// Environment instance.
    env: E,
    /// Current exact observation.
    observation: Vec<f32>,
    /// Independent policy sampling stream.
    sampler: RecurrentSampler,
    /// Completed episode count used for reset seeds.
    episode: u64,
}

/// One collected recurrent sequence before GAE construction.
struct SequenceBuffer {
    /// Encoded observations.
    observations: Vec<Vec<f32>>,
    /// Samples before tanh.
    pre_tanh_actions: Vec<Vec<f32>>,
    /// Behavior log probabilities.
    log_probabilities: Vec<f32>,
    /// Environment rewards.
    rewards: Vec<f32>,
    /// Critic values.
    values: Vec<f32>,
    /// Next-state critic values.
    next_values: Vec<f32>,
    /// Boundary states.
    statuses: Vec<EpisodeStatus>,
    /// Actor memory before the first observation.
    initial_memory: RecurrentMemory,
}

impl SequenceBuffer {
    /// Allocate an empty contiguous sequence.
    const fn new(initial_memory: RecurrentMemory) -> Self {
        // Keep this shared workflow stage consistent for every example that uses the public training boundary.
        Self {
            observations: Vec::new(),
            pre_tanh_actions: Vec::new(),
            log_probabilities: Vec::new(),
            rewards: Vec::new(),
            values: Vec::new(),
            next_values: Vec::new(),
            statuses: Vec::new(),
            initial_memory,
        }
    }

    /// Return whether no transition has been appended.
    const fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Convert this contiguous sequence into one PPO update record.
    #[expect(
        clippy::indexing_slicing,
        reason = "all transition vectors are appended together and therefore have equal lengths"
    )]
    fn finish(self, gamma: f32, gae_lambda: f32) -> RecurrentPpoSequence {
        // Keep this shared workflow stage consistent for every example that uses the public training boundary.
        let mut advantages = Vec::with_capacity(self.rewards.len());
        let mut next_advantage = 0.0_f32;
        for index in (0..self.rewards.len()).rev() {
            let status = self.statuses[index];
            let bootstrap = status.bootstrap_mask() as f32;
            let trace = f32::from(status == EpisodeStatus::Continuing);
            let delta = (gamma * bootstrap).mul_add(
                self.next_values[index],
                self.rewards[index] - self.values[index],
            );
            let advantage = (gamma * gae_lambda * trace).mul_add(next_advantage, delta);
            advantages.push(advantage);
            next_advantage = advantage;
        }
        advantages.reverse();
        let returns = self
            .values
            .iter()
            .zip(&advantages)
            .map(|(value, advantage)| value + advantage)
            .collect();
        RecurrentPpoSequence {
            global_states: self.observations.clone(),
            observations: self.observations,
            pre_tanh_actions: self.pre_tanh_actions,
            old_log_probabilities: self.log_probabilities,
            advantages,
            returns,
            value_index: 0,
            initial_memory: self.initial_memory,
        }
    }
}

/// Parse and execute the standard continuous workflow.
///
/// # Errors
///
/// Returns an error when argument handling, training, evaluation, or rendering fails.
pub fn run_continuous_workflow<E>() -> Result<(), Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mode = Args::parse().command.unwrap_or_else(|| default_mode::<E>());
    match mode {
        Mode::Train {
            steps,
            num_envs,
            eval_interval,
            eval_episodes,
            actor_learning_rate,
            critic_learning_rate,
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
                actor_learning_rate: actor_learning_rate.unwrap_or(E::DEFAULT_ACTOR_LEARNING_RATE),
                critic_learning_rate: critic_learning_rate
                    .unwrap_or(E::DEFAULT_CRITIC_LEARNING_RATE),
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
                "checkpoint={} episodes={} success_rate={:.3} mean_reward={:.3} mean_length={:.3} min_reward={:.3} max_reward={:.3}",
                checkpoint.display(),
                summary.episodes,
                summary.success_rate,
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
fn default_mode<E: ContinuousPpoExample>() -> Mode {
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
            actor_learning_rate: Some(E::DEFAULT_ACTOR_LEARNING_RATE),
            critic_learning_rate: Some(E::DEFAULT_CRITIC_LEARNING_RATE),
            reward_scale: Some(E::DEFAULT_REWARD_SCALE),
            seed: 42,
            run_id: None,
            runs_root: PathBuf::from("runs"),
        }
    }
}

/// Train one recurrent PPO policy from batched on-policy sequences.
fn train<E>(options: TrainOptions) -> Result<TrainReport, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let ContinuousTrainingState {
        paths,
        seeds,
        config,
        mut agent,
        initial,
        mut best,
        mut metrics,
    } = initialize_training::<E>(&options)?;
    if best.mean_reward >= E::SOLVED_MEAN_REWARD {
        write_summary(&paths, 0, initial, best)?;
        return Ok(TrainReport {
            paths,
            initial,
            best,
            steps: 0,
        });
    }

    let mut lanes = make_lanes::<E>(options.num_envs, seeds)?;
    let batch_steps = options
        .num_envs
        .checked_mul(E::ROLLOUT_STEPS_PER_ENV)
        .ok_or("rollout batch size overflow")?;
    let mut global_steps = 0usize;
    let mut next_eval = options.eval_interval;
    while global_steps < options.steps {
        let policy = agent.policy();
        let sequences = collect_sequences::<E>(
            &policy,
            &mut lanes,
            E::ROLLOUT_STEPS_PER_ENV,
            config.gamma,
            config.gae_lambda,
            options.reward_scale,
            seeds,
        )?;
        let update = agent.update(&sequences)?;
        global_steps = global_steps.saturating_add(batch_steps);

        if global_steps >= next_eval || global_steps >= options.steps {
            let summary = evaluate::<E>(&agent.policy(), options.eval_episodes, seeds.validation)?;
            let is_best = summary.mean_reward > best.mean_reward;
            agent.policy().save(&paths.latest_checkpoint)?;
            agent.policy().save(
                paths
                    .checkpoints_dir
                    .join(format!("step-{global_steps}.mpk")),
            )?;
            if is_best {
                best = summary;
                agent.policy().save(&paths.best_checkpoint)?;
            }
            write_evaluation(&mut metrics, global_steps, summary, is_best)?;
            metrics.write_record(
                &MetricRecord::new(u64::try_from(global_steps)?)
                    .with_field("train/actor_loss", MetricValue::Number(update.actor_loss))
                    .with_field("train/critic_loss", MetricValue::Number(update.critic_loss))
                    .with_field("train/entropy", MetricValue::Number(update.entropy))
                    .with_field(
                        "train/approximate_kl",
                        MetricValue::Number(update.approximate_kl),
                    )
                    .with_field(
                        "train/optimizer_updates",
                        MetricValue::Number(update.optimizer_steps as f64),
                    ),
            )?;
            println!(
                "steps={global_steps} updates={} actor_loss={:.4} critic_loss={:.4} kl={:.4} success_rate={:.3} mean_reward={:.3} mean_length={:.2} best={:.3}",
                update.optimizer_steps,
                update.actor_loss,
                update.critic_loss,
                update.approximate_kl,
                summary.success_rate,
                summary.mean_reward,
                summary.mean_length,
                best.mean_reward
            );
            if best.mean_reward >= E::SOLVED_MEAN_REWARD {
                break;
            }
            while next_eval <= global_steps {
                next_eval = next_eval.saturating_add(options.eval_interval);
            }
        }
    }

    write_summary(&paths, global_steps, initial, best)?;
    Ok(TrainReport {
        paths,
        initial,
        best,
        steps: global_steps,
    })
}

/// Initialized recurrent trainer state and durable outputs.
struct ContinuousTrainingState {
    /// Durable run paths.
    paths: RunPaths,
    /// Derived deterministic seeds.
    seeds: SeedConfig,
    /// Exact optimizer and model profile.
    config: RecurrentPpoConfig,
    /// Mutable recurrent PPO learner.
    agent: RecurrentPpoAgent,
    /// Random-policy held-out baseline.
    initial: ContinuousEvaluation,
    /// Current validation-selected result.
    best: ContinuousEvaluation,
    /// Append-only metric writer.
    metrics: MetricsWriter,
}

/// Create checkpoints, metrics, learner state, and optional actor pretraining.
fn initialize_training<E>(options: &TrainOptions) -> Result<ContinuousTrainingState, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if options.num_envs == 0
        || options.eval_interval == 0
        || options.eval_episodes == 0
        || E::ROLLOUT_STEPS_PER_ENV == 0
    {
        return Err(
            "num-envs, rollout steps, eval-interval, and eval-episodes must be positive".into(),
        );
    }
    let run_id = match options.run_id.as_deref() {
        Some(value) => RunId::new(value.to_owned())?,
        None => RunId::new(default_run_id()?)?,
    };
    let run = RunConfig::new(E::ENV_NAME, AlgorithmKind::Ppo, run_id, &options.runs_root)?;
    let paths = run.paths();
    paths.create_new()?;
    let seeds = SeedConfig::from_root(options.seed);
    let config = E::ppo_config(options.actor_learning_rate, options.critic_learning_rate);
    write_run_config::<E>(&paths, options, &config, seeds)?;
    let mut agent = RecurrentPpoAgent::new(
        E::OBSERVATION_DIM,
        E::OBSERVATION_DIM,
        1,
        E::ACTION_LOW,
        E::ACTION_HIGH,
        config.clone(),
        seeds,
    )?;
    let initial = evaluate::<E>(&agent.policy(), options.eval_episodes, seeds.validation)?;
    let mut best = initial;
    agent.policy().save(&paths.best_checkpoint)?;
    agent.policy().save(&paths.latest_checkpoint)?;
    agent
        .policy()
        .save(paths.checkpoints_dir.join("step-0.mpk"))?;
    let mut metrics = MetricsWriter::append(&paths.metrics_jsonl)?;
    write_evaluation(&mut metrics, 0, initial, true)?;
    let pretrained = pretrain_actor::<E>(&mut agent, seeds.model, &paths)?;
    if pretrained {
        let summary = evaluate::<E>(&agent.policy(), options.eval_episodes, seeds.validation)?;
        let is_best = summary.mean_reward > best.mean_reward;
        agent.policy().save(&paths.latest_checkpoint)?;
        if is_best {
            best = summary;
            agent.policy().save(&paths.best_checkpoint)?;
        }
        write_evaluation(
            &mut metrics,
            E::BEHAVIOR_PRETRAINING_EPOCHS,
            summary,
            is_best,
        )?;
    }
    Ok(ContinuousTrainingState {
        paths,
        seeds,
        config,
        agent,
        initial,
        best,
        metrics,
    })
}

/// Initialize one actor from bounded expert demonstrations before on-policy PPO.
fn pretrain_actor<E>(
    agent: &mut RecurrentPpoAgent,
    seed: u64,
    paths: &RunPaths,
) -> Result<bool, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    if E::BEHAVIOR_PRETRAINING_SAMPLES == 0 || E::BEHAVIOR_PRETRAINING_EPOCHS == 0 {
        return Ok(false);
    }
    if E::BEHAVIOR_PRETRAINING_BATCH_SIZE == 0 {
        return Err("behavior pretraining batch size must be positive".into());
    }
    let mut demonstrations = E::behavior_demonstrations(E::BEHAVIOR_PRETRAINING_SAMPLES, seed);
    if demonstrations.len() != E::BEHAVIOR_PRETRAINING_SAMPLES {
        return Err(format!(
            "expected {} behavior samples, received {}",
            E::BEHAVIOR_PRETRAINING_SAMPLES,
            demonstrations.len()
        )
        .into());
    }
    let mut final_loss = 0.0;
    let mut shuffle_state = seed ^ 0xa076_1d64_78bd_642f;
    let mut checkpoint_epochs = [
        E::BEHAVIOR_PRETRAINING_EPOCHS / 3,
        E::BEHAVIOR_PRETRAINING_EPOCHS * 2 / 3,
        E::BEHAVIOR_PRETRAINING_EPOCHS,
    ];
    checkpoint_epochs.sort_unstable();
    for epoch in 1..=E::BEHAVIOR_PRETRAINING_EPOCHS {
        shuffle_behavior_samples(&mut demonstrations, &mut shuffle_state);
        for batch in demonstrations.chunks(E::BEHAVIOR_PRETRAINING_BATCH_SIZE) {
            final_loss = agent.behavior_clone(batch)?;
        }
        if checkpoint_epochs.contains(&epoch) {
            agent
                .policy()
                .save(paths.checkpoints_dir.join(format!("step-{epoch}.mpk")))?;
        }
    }
    println!(
        "behavior_samples={} behavior_epochs={} behavior_loss={final_loss:.6}",
        demonstrations.len(),
        E::BEHAVIOR_PRETRAINING_EPOCHS
    );
    Ok(true)
}

/// Shuffle expert samples reproducibly without adding an environment RNG dependency.
fn shuffle_behavior_samples(samples: &mut [RecurrentBehaviorSample], state: &mut u64) {
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    for upper in (1..samples.len()).rev() {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        samples.swap(upper, value as usize % (upper + 1));
    }
}

/// Initialize persistent environment lanes and independent samplers.
fn make_lanes<E>(count: usize, seeds: SeedConfig) -> Result<Vec<Lane<E>>, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    (0..count)
        .map(|lane_id| {
            let mut env = E::default();
            let observation = env
                .reset(Some(seeds.environment_episode(lane_id, 0)))
                .observation;
            Ok(Lane {
                env,
                observation,
                sampler: RecurrentSampler::new(
                    seeds.action ^ u64::try_from(lane_id)?.wrapping_mul(0x9e37_79b9),
                ),
                episode: 0,
            })
        })
        .collect()
}

/// Collect one fixed on-policy rollout from every lane.
fn collect_sequences<E>(
    policy: &RecurrentPpoPolicy,
    lanes: &mut [Lane<E>],
    steps_per_env: usize,
    gamma: f32,
    gae_lambda: f32,
    reward_scale: f64,
    seeds: SeedConfig,
) -> Result<Vec<RecurrentPpoSequence>, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mut sequences = Vec::new();
    for (lane_id, lane) in lanes.iter_mut().enumerate() {
        let mut memory = policy.initial_memory();
        let mut buffer = SequenceBuffer::new(memory.clone());
        for _ in 0..steps_per_env {
            let observation = E::encode_observation(&lane.observation);
            let sampled = policy.sample_action(&observation, &memory, &mut lane.sampler)?;
            let value = policy.value(&observation, 0)?;
            let environment_action = sampled.action.clone();
            let transition = lane.env.step(sampled.action);
            let next_observation = E::encode_observation(&transition.observation);
            let next_value = policy.value(&next_observation, 0)?;
            buffer.observations.push(observation);
            buffer.pre_tanh_actions.push(sampled.pre_tanh_action);
            buffer.log_probabilities.push(sampled.log_probability);
            buffer.rewards.push(E::training_reward(
                &lane.observation,
                &environment_action,
                transition.reward,
                &transition.observation,
                transition.status,
                reward_scale,
            ) as f32);
            buffer.values.push(value);
            buffer.next_values.push(next_value);
            buffer.statuses.push(transition.status);
            lane.observation = transition.observation;
            memory = if E::RETAIN_RECURRENT_MEMORY {
                sampled.next_memory
            } else {
                policy.initial_memory()
            };

            if transition.status != EpisodeStatus::Continuing {
                sequences.push(buffer.finish(gamma, gae_lambda));
                lane.episode = lane.episode.saturating_add(1);
                lane.observation = lane
                    .env
                    .reset(Some(seeds.environment_episode(lane_id, lane.episode)))
                    .observation;
                memory = policy.initial_memory();
                buffer = SequenceBuffer::new(memory.clone());
            } else if !E::RETAIN_RECURRENT_MEMORY {
                sequences.push(buffer.finish(gamma, gae_lambda));
                buffer = SequenceBuffer::new(memory.clone());
            }
        }
        if !buffer.is_empty() {
            sequences.push(buffer.finish(gamma, gae_lambda));
        }
    }
    Ok(sequences)
}

/// Evaluate the deterministic actor on a fixed held-out seed schedule.
fn evaluate<E>(
    policy: &RecurrentPpoPolicy,
    episodes: usize,
    seed: u64,
) -> Result<ContinuousEvaluation, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let mut successes = 0usize;
    let mut total_reward = 0.0;
    let mut total_length = 0usize;
    let mut min_reward = f64::INFINITY;
    let mut max_reward = f64::NEG_INFINITY;
    for episode in 0..episodes {
        let mut env = E::default();
        let mut observation = env
            .reset(Some(seed.wrapping_add(u64::try_from(episode)?)))
            .observation;
        let mut memory = policy.initial_memory();
        let mut episode_reward = 0.0;
        let mut episode_length = 0usize;
        let final_status = loop {
            let encoded = E::encode_observation(&observation);
            let action = policy.mean_action(&encoded, &memory)?;
            memory = if E::RETAIN_RECURRENT_MEMORY {
                action.next_memory
            } else {
                policy.initial_memory()
            };
            let transition = env.step(action.action);
            observation = transition.observation;
            episode_reward += transition.reward;
            episode_length += 1;
            if transition.status != EpisodeStatus::Continuing {
                break transition.status;
            }
        };
        successes += usize::from(E::is_success(&observation, episode_reward, final_status));
        total_reward += episode_reward;
        total_length += episode_length;
        min_reward = min_reward.min(episode_reward);
        max_reward = max_reward.max(episode_reward);
    }
    Ok(ContinuousEvaluation {
        episodes,
        success_rate: successes as f64 / episodes as f64,
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
    summary: ContinuousEvaluation,
    is_best: bool,
) -> Result<(), Box<dyn Error>> {
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    metrics.write_record(
        &MetricRecord::new(u64::try_from(step)?)
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

/// Resolve an explicit checkpoint or newest selected checkpoint.
fn resolve_checkpoint<E>(
    explicit: Option<&Path>,
    runs_root: &Path,
) -> Result<PathBuf, Box<dyn Error>>
where
    E: ContinuousPpoExample,
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
    let root = runs_root.join(format!("{}-ppo", E::ENV_NAME));
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
    E: ContinuousPpoExample,
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
        actor_learning_rate: E::DEFAULT_ACTOR_LEARNING_RATE,
        critic_learning_rate: E::DEFAULT_CRITIC_LEARNING_RATE,
        reward_scale: E::DEFAULT_REWARD_SCALE,
        seed: 42,
        run_id: None,
        runs_root: runs_root.to_path_buf(),
    })?;
    Ok(report.paths.best_checkpoint)
}

/// Load one policy using the example's exact runtime profile.
fn load_policy<E>(path: &Path) -> Result<RecurrentPpoPolicy, Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    let config = E::ppo_config(
        E::DEFAULT_ACTOR_LEARNING_RATE,
        E::DEFAULT_CRITIC_LEARNING_RATE,
    );
    Ok(RecurrentPpoPolicy::load(
        path,
        E::OBSERVATION_DIM,
        E::OBSERVATION_DIM,
        1,
        E::ACTION_LOW,
        E::ACTION_HIGH,
        &config,
    )?)
}

/// Persist the exact training profile.
fn write_run_config<E>(
    paths: &RunPaths,
    options: &TrainOptions,
    config: &RecurrentPpoConfig,
    seeds: SeedConfig,
) -> Result<(), Box<dyn Error>>
where
    E: ContinuousPpoExample,
{
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    runtime_io::write(
        &paths.config_json,
        format!(
            "{{\"algorithm\":\"recurrent-ppo\",\"environment\":\"{}\",\"steps\":{},\"num_envs\":{},\"rollout_steps_per_env\":{},\"actor_learning_rate\":{},\"critic_learning_rate\":{},\"reward_scale\":{},\"actor_hidden_size\":{}}}\n",
            E::GYMNASIUM_ID,
            options.steps,
            options.num_envs,
            E::ROLLOUT_STEPS_PER_ENV,
            options.actor_learning_rate,
            options.critic_learning_rate,
            options.reward_scale,
            config.actor_hidden_size
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
    initial: ContinuousEvaluation,
    best: ContinuousEvaluation,
) -> Result<(), Box<dyn Error>> {
    // Keep this shared workflow stage consistent for every example that uses the public training boundary.
    runtime_io::write(
        &paths.summary_json,
        format!(
            "{{\"steps\":{steps},\"initial_mean_reward\":{},\"best_mean_reward\":{},\"success_rate\":{}}}\n",
            initial.mean_reward, best.mean_reward, best.success_rate
        ),
    )?;
    Ok(())
}

/// Build a sortable run identifier.
fn default_run_id() -> Result<String, Box<dyn Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    Ok(format!("run-{timestamp}"))
}
