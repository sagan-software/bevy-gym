//! On-policy ecosystem rollout collection and curriculum-stage training loop.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use bevy_gym::training::{
    AlgorithmKind, MetricRecord, MetricValue, MetricsWriter, RecurrentMemory, RecurrentPpoAgent,
    RecurrentPpoConfig, RecurrentPpoPolicy, RecurrentPpoSequence, RecurrentPpoUpdate,
    RecurrentSampler, RunConfig, RunId, RunPaths, SeedConfig,
};
use bevy_gym::EpisodeStatus;

use super::domain::{
    AgentEpisodeMetrics, AgentId, CurriculumStage, ExperimentTuning, GlobalState, LocalObservation,
    LocomotionAction, SimulationConfig, Species, GLOBAL_STATE_SIZE, LOCAL_OBSERVATION_SIZE,
    MAX_AGENTS,
};
use super::rng::SplitMix64;
use super::simulation::Ecosystem;

/// Number of temporal steps retained in one truncated-backpropagation chunk.
const RECURRENT_CHUNK_LENGTH: usize = 32;

/// Durable ecosystem mechanics and tensor contract written beside checkpoints.
const CHECKPOINT_PROFILE: u64 = 3;

/// Validated command-line controls for one local training experiment.
#[derive(Debug, Clone, PartialEq)]
struct TrainOptions {
    /// Complete ecosystem rollouts collected before stopping.
    iterations: usize,

    /// Independent ecosystem episodes pooled into each PPO update.
    rollout_episodes: usize,

    /// Per-episode simulated-second horizon.
    episode_seconds: u16,

    /// Root for every independent random stream.
    seed: u64,

    /// Stable artifact directory leaf.
    run_id: Option<String>,

    /// Root directory for ignored run artifacts.
    runs_root: PathBuf,

    /// Fixed-seed ecosystem episodes per validation pass.
    eval_episodes: usize,

    /// Training iterations between validation passes.
    eval_interval: usize,

    /// Gaussian entropy coefficient used by PPO actor updates.
    entropy_coefficient: f32,

    /// Maximum Gaussian log standard deviation emitted by the actor.
    log_std_max: f32,

    /// Discount factor used for delayed survival credit.
    gamma: f32,

    /// GAE trace factor used for delayed survival credit.
    gae_lambda: f32,

    /// Optional predecessor bunny checkpoint or run directory.
    resume: Option<PathBuf>,

    /// Optional predecessor fox checkpoint.
    fox_resume: Option<PathBuf>,
}

/// Fixed-seed fresh-process evaluation arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EvalOptions {
    /// Bunny checkpoint or run directory containing `best.mpk`.
    checkpoint: PathBuf,

    /// Optional explicit fox checkpoint.
    fox_checkpoint: Option<PathBuf>,

    /// Evaluation ecosystem episodes.
    episodes: usize,

    /// Optional per-episode simulated-second horizon.
    episode_seconds: Option<u16>,

    /// Root of the evaluation-only seed stream.
    seed: u64,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            iterations: 32,
            rollout_episodes: 4,
            episode_seconds: ExperimentTuning::default().episode_seconds,
            seed: 42,
            run_id: None,
            runs_root: PathBuf::from("runs"),
            eval_episodes: 8,
            eval_interval: 4,
            entropy_coefficient: 0.005,
            log_std_max: 1.0,
            gamma: 0.999,
            gae_lambda: 1.0,
            resume: None,
            fox_resume: None,
        }
    }
}

impl TrainOptions {
    /// Select stage-specific defaults without changing explicit CLI overrides.
    fn for_stage(stage: CurriculumStage) -> Self {
        let mut options = Self::default();
        if stage == CurriculumStage::Survival {
            // This profile improved on a disjoint evaluation stream in six updates.
            options.iterations = 6;
            options.rollout_episodes = 16;
            options.seed = 157;
            options.eval_episodes = 16;
            options.eval_interval = 1;
            options.log_std_max = -0.5;
        }
        options
    }
}

/// Fixed-seed deterministic evaluation means.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct EvaluationSummary {
    /// Mean bunny undiscounted episode return.
    bunny_mean_return: f32,

    /// Mean fox undiscounted episode return, or zero when absent.
    fox_mean_return: f32,

    /// Mean bunny lifetime in simulated seconds.
    bunny_mean_lifetime: f32,

    /// Mean fox lifetime in simulated seconds, or zero when absent.
    fox_mean_lifetime: f32,

    /// Bootstrap 95% lower bound for bunny mean lifetime.
    bunny_lifetime_lower: f32,

    /// Bootstrap 95% upper bound for bunny mean lifetime.
    bunny_lifetime_upper: f32,

    /// Bootstrap 95% lower bound for fox mean lifetime.
    fox_lifetime_lower: f32,

    /// Bootstrap 95% upper bound for fox mean lifetime.
    fox_lifetime_upper: f32,

    /// Bunny fraction that consumed both compatible food and water.
    bunny_food_and_water_fraction: f32,

    /// Bunny identities that consumed food or water in evaluation.
    bunny_resource_consumer_count: u32,

    /// Largest bunny identity mean advantage above the population mean.
    bunny_index_lifetime_advantage_fraction: f32,

    /// Fox fraction that consumed both bunny prey and water.
    fox_food_and_water_fraction: f32,

    /// Agent-agent contact observations across all evaluation episodes.
    collision_contacts: u64,

    /// Total thorn hit-point damage across all evaluation agents.
    thorn_damage: f32,

    /// Total overconsumption hit-point damage across evaluation agents.
    overconsumption_damage: f32,

    /// Compatible food or prey events across evaluation agents.
    food_events: u64,

    /// Bunny predation events across evaluation foxes.
    predation_events: u64,

    /// Well-water units consumed across evaluation agents.
    water_consumed: f32,

    /// Evaluation episodes containing at least one predation event.
    episodes_with_predation: u32,

    /// Evaluation episodes in which every bunny avoided predation.
    episodes_without_predation: u32,

    /// Terminal-cause counts across all evaluated agents.
    deaths: EvaluationDeathCounts,
}

/// Raw terminal-cause counts for one fixed-seed evaluation suite.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct EvaluationDeathCounts {
    /// Satiation-only deaths.
    starvation: u32,

    /// Hydration-only deaths.
    dehydration: u32,

    /// Combined satiation-and-hydration deaths.
    deprivation: u32,

    /// Bunnies consumed by foxes.
    predation: u32,

    /// Thorn hazard deaths.
    thorns: u32,

    /// Deaths caused by concurrent deprivation and thorn damage.
    combined_damage: u32,

    /// Excess food or water deaths.
    overconsumption: u32,

    /// Invalid physics-state deaths.
    invalid_physics: u32,
}

/// Valid-sample-weighted optimizer diagnostics for one ecosystem iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
struct UpdateSummary {
    /// Unique valid rollout samples consumed across species.
    valid_samples: u64,

    /// Minibatch optimizer updates completed across species.
    optimizer_updates: u64,

    /// Mean actor loss.
    actor_loss: f64,

    /// Mean critic loss.
    critic_loss: f64,

    /// Mean Gaussian entropy.
    entropy: f64,

    /// Mean approximate KL divergence.
    approximate_kl: f64,

    /// Effective actor learning rate.
    actor_learning_rate: f64,

    /// Effective critic learning rate.
    critic_learning_rate: f64,
}

/// Typed policy and metric snapshot consumed by the visual trainer.
#[cfg(feature = "render")]
#[derive(Debug, Clone)]
pub(super) struct TrainingProgress {
    /// Completed optimizer iteration.
    pub(super) iteration: usize,

    /// Configured optimizer iteration limit.
    pub(super) total_iterations: usize,

    /// Joint environment steps collected so far.
    pub(super) global_steps: u64,

    /// Mean bunny episodic return from the latest training batch.
    pub(super) bunny_train_return: f32,

    /// Mean fox episodic return from the latest training batch.
    pub(super) fox_train_return: f32,

    /// Mean bunny episodic return from fixed-seed evaluation.
    pub(super) bunny_eval_return: f32,

    /// Mean fox episodic return from fixed-seed evaluation.
    pub(super) fox_eval_return: f32,

    /// Mean bunny lifetime from fixed-seed evaluation.
    pub(super) bunny_eval_lifetime: f32,

    /// Mean fox lifetime from fixed-seed evaluation.
    pub(super) fox_eval_lifetime: f32,

    /// Mean actor objective from the latest update.
    pub(super) actor_loss: f64,

    /// Mean critic objective from the latest update.
    pub(super) critic_loss: f64,

    /// Mean Gaussian policy entropy from the latest update.
    pub(super) entropy: f64,

    /// Effective actor learning rate from the latest update.
    pub(super) actor_learning_rate: f64,

    /// Effective critic learning rate from the latest update.
    pub(super) critic_learning_rate: f64,

    /// Whether fixed-seed selection accepted this iteration.
    pub(super) is_best: bool,

    /// Whether return fields contain a completed fixed-seed evaluation.
    pub(super) has_evaluation: bool,

    /// Current bunny inference policy.
    pub(super) bunny: RecurrentPpoPolicy,

    /// Current fox inference policy when the stage has predators.
    pub(super) fox: Option<RecurrentPpoPolicy>,

    /// Durable training artifact directory.
    pub(super) run_dir: PathBuf,

    /// Reward, physiology, and horizon settings used by this policy update.
    pub(super) experiment_tuning: ExperimentTuning,
}

impl EvaluationSummary {
    /// Score used only for best-checkpoint selection.
    const fn selection_score(self, has_foxes: bool) -> f32 {
        if has_foxes {
            self.bunny_mean_return.min(self.fox_mean_return)
        } else {
            self.bunny_mean_return
        }
    }
}

impl UpdateSummary {
    /// Build the pre-update metric baseline for the visual trainer.
    #[cfg(feature = "render")]
    const fn empty(algorithm: &RecurrentPpoConfig) -> Self {
        // Keep configured learning rates visible before any loss exists.
        Self {
            valid_samples: 0,
            optimizer_updates: 0,
            actor_loss: 0.0,
            critic_loss: 0.0,
            entropy: 0.0,
            approximate_kl: 0.0,
            actor_learning_rate: algorithm.actor_learning_rate,
            critic_learning_rate: algorithm.critic_learning_rate,
        }
    }
}

/// Convert one trainer instant into the visual boundary type.
#[cfg(feature = "render")]
fn training_progress(
    learners: &Learners,
    config: &SimulationConfig,
    paths: &RunPaths,
    options: &TrainOptions,
    iteration: usize,
    global_steps: u64,
    bunny_train_return: f32,
    fox_train_return: f32,
    evaluation: EvaluationSummary,
    update: UpdateSummary,
    is_best: bool,
    has_evaluation: bool,
) -> TrainingProgress {
    // Freeze inference networks at the same instant as their displayed metrics.
    TrainingProgress {
        iteration,
        total_iterations: options.iterations,
        global_steps,
        bunny_train_return,
        fox_train_return,
        bunny_eval_return: evaluation.bunny_mean_return,
        fox_eval_return: evaluation.fox_mean_return,
        bunny_eval_lifetime: evaluation.bunny_mean_lifetime,
        fox_eval_lifetime: evaluation.fox_mean_lifetime,
        actor_loss: update.actor_loss,
        critic_loss: update.critic_loss,
        entropy: update.entropy,
        actor_learning_rate: update.actor_learning_rate,
        critic_learning_rate: update.critic_learning_rate,
        is_best,
        has_evaluation,
        bunny: learners.bunny.policy(),
        fox: learners.fox.as_ref().map(RecurrentPpoAgent::policy),
        run_dir: paths.run_dir.clone(),
        experiment_tuning: config.experiment_tuning(),
    }
}

/// One behavior-policy transition before GAE and chunking.
#[derive(Debug, Clone, PartialEq)]
struct RolloutTransition {
    /// Local observation before action selection.
    observation: LocalObservation,

    /// Centralized state from the same instant.
    global_state: GlobalState,

    /// Gaussian sample before tanh squashing.
    pre_tanh_action: Vec<f32>,

    /// Corrected behavior-policy log probability.
    old_log_probability: f32,

    /// Survival reward earned by this transition.
    reward: f32,

    /// Centralized value before the action.
    value: f32,

    /// Centralized value for the actual next state.
    next_value: f32,

    /// Natural termination, horizon truncation, or continuation.
    status: EpisodeStatus,

    /// Actor memory immediately before consuming the observation.
    initial_memory: RecurrentMemory,
}

/// Completed transitions and episode-level survival diagnostics.
#[derive(Debug, Default)]
struct CollectedEpisode {
    /// Per-agent transitions in stable identity order.
    trajectories: BTreeMap<AgentId, (Species, Vec<RolloutTransition>)>,

    /// Joint environment steps completed.
    joint_steps: u32,
}

/// Pooled same-policy sequences and lifetimes for one optimizer iteration.
#[derive(Debug, Default)]
struct CollectedBatch {
    /// All bunny trajectory chunks across environment lanes.
    bunny_sequences: Vec<RecurrentPpoSequence>,

    /// All fox trajectory chunks across environment lanes.
    fox_sequences: Vec<RecurrentPpoSequence>,

    /// Joint environment steps summed across lanes.
    joint_steps: u64,

    /// Per-episode bunny mean lifetimes.
    bunny_lifetimes: Vec<f32>,

    /// Per-episode fox mean lifetimes.
    fox_lifetimes: Vec<f32>,

    /// Per-episode bunny mean undiscounted returns.
    bunny_returns: Vec<f32>,

    /// Per-episode fox mean undiscounted returns.
    fox_returns: Vec<f32>,
}

/// Species-specific training objects without parameter sharing across roles.
struct Learners {
    /// Shared bunny actor and centralized critic.
    bunny: RecurrentPpoAgent,

    /// Shared fox actor and centralized critic when the stage has foxes.
    fox: Option<RecurrentPpoAgent>,
}

/// Owned trainer state prepared before the first visual progress event.
struct TrainingSetup {
    /// Validated command-line experiment controls.
    options: TrainOptions,

    /// Independent deterministic random streams.
    seeds: SeedConfig,

    /// PPO settings after command-line overrides.
    algorithm: RecurrentPpoConfig,

    /// Mutable ecosystem settings shared by collection and evaluation.
    config: SimulationConfig,

    /// Species-specific trainable policies.
    learners: Learners,

    /// Durable artifact paths for this independent run.
    paths: RunPaths,

    /// Append-only metrics stream owned by the trainer.
    metrics: MetricsWriter,
}

/// Stable resources needed to persist one live tuning transition.
#[cfg(feature = "render")]
struct TuningPersistence<'a> {
    /// Current policy learners saved as the new same-profile baseline.
    learners: &'a Learners,
    /// Durable artifact paths for config and checkpoint replacement.
    paths: &'a RunPaths,
    /// Evaluation budget and run settings retained in the config.
    options: &'a TrainOptions,
    /// Optimizer settings retained in the config.
    algorithm: &'a RecurrentPpoConfig,
    /// Independent random streams used by evaluation.
    seeds: SeedConfig,
    /// Curriculum stage retained in the config.
    stage: CurriculumStage,
}

/// Stable resources used by one scheduled fixed-seed evaluation.
struct EvaluationPass<'a> {
    /// Curriculum stage required by the saved policy profile.
    stage: CurriculumStage,
    /// Current trainable policies under evaluation.
    learners: &'a Learners,
    /// Current stationary simulation profile.
    config: &'a SimulationConfig,
    /// Durable checkpoint paths.
    paths: &'a RunPaths,
    /// Evaluation episode count and interval settings.
    options: &'a TrainOptions,
    /// Durable metrics writer.
    metrics: &'a mut MetricsWriter,
    /// Validation random stream.
    validation_seed: u64,
    /// Whether checkpoint selection balances both species.
    has_foxes: bool,
}

impl std::fmt::Debug for Learners {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Learners")
            .field("bunny", &self.bunny)
            .field("fox", &self.fox)
            .finish()
    }
}

/// Immutable collection snapshots and independent action streams.
struct CollectionPolicies {
    /// Frozen bunny behavior policy.
    bunny: RecurrentPpoPolicy,

    /// Frozen fox behavior policy when present.
    fox: Option<RecurrentPpoPolicy>,

    /// Bunny action-sampling stream.
    bunny_sampler: RecurrentSampler,

    /// Fox action-sampling stream.
    fox_sampler: RecurrentSampler,
}

impl std::fmt::Debug for CollectionPolicies {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CollectionPolicies")
            .field("bunny", &self.bunny)
            .field("fox", &self.fox)
            .field("bunny_sampler", &self.bunny_sampler)
            .field("fox_sampler", &self.fox_sampler)
            .finish()
    }
}

/// Train one stage and write optimizer and lifetime evidence to stdout.
pub(super) fn run_training(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    // Headless training pays no policy-clone cost for visual progress events.
    #[cfg(feature = "render")]
    {
        run_training_inner(stage, arguments, None, None)
    }
    #[cfg(not(feature = "render"))]
    {
        run_training_inner(stage, arguments)
    }
}

/// Train one stage while publishing policy snapshots to an interactive demo.
#[cfg(feature = "render")]
pub(super) fn run_training_with_progress(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
    mut tuning_provider: impl FnMut() -> ExperimentTuning,
    mut reporter: impl FnMut(TrainingProgress) -> bool,
) -> Result<(), Box<dyn Error>> {
    run_training_inner(
        stage,
        arguments,
        Some(&mut tuning_provider),
        Some(&mut reporter),
    )
}

/// Publish policies before fixed-seed baseline evaluation begins.
#[cfg(feature = "render")]
fn report_training_start(
    reporter: &mut Option<&mut dyn FnMut(TrainingProgress) -> bool>,
    learners: &Learners,
    config: &SimulationConfig,
    paths: &RunPaths,
    options: &TrainOptions,
    algorithm: &RecurrentPpoConfig,
) -> bool {
    reporter.as_mut().is_none_or(|callback| {
        callback(training_progress(
            learners,
            config,
            paths,
            options,
            0,
            0,
            0.0,
            0.0,
            EvaluationSummary::default(),
            UpdateSummary::empty(algorithm),
            false,
            false,
        ))
    })
}

/// Publish the random-policy return used as the graph baseline.
#[cfg(feature = "render")]
fn report_initial_evaluation(
    reporter: &mut Option<&mut dyn FnMut(TrainingProgress) -> bool>,
    learners: &Learners,
    config: &SimulationConfig,
    paths: &RunPaths,
    options: &TrainOptions,
    algorithm: &RecurrentPpoConfig,
    evaluation: EvaluationSummary,
) -> bool {
    reporter.as_mut().is_none_or(|callback| {
        callback(training_progress(
            learners,
            config,
            paths,
            options,
            0,
            0,
            0.0,
            0.0,
            evaluation,
            UpdateSummary::empty(algorithm),
            false,
            true,
        ))
    })
}

/// Publish one completed PPO update and its current inference policies.
#[cfg(feature = "render")]
fn report_training_update(
    reporter: &mut Option<&mut dyn FnMut(TrainingProgress) -> bool>,
    learners: &Learners,
    config: &SimulationConfig,
    paths: &RunPaths,
    options: &TrainOptions,
    iteration: usize,
    global_steps: u64,
    bunny_train_return: f32,
    fox_train_return: f32,
    evaluation: EvaluationSummary,
    update: UpdateSummary,
    is_best: bool,
    has_evaluation: bool,
) -> bool {
    reporter.as_mut().is_none_or(|callback| {
        callback(training_progress(
            learners,
            config,
            paths,
            options,
            iteration,
            global_steps,
            bunny_train_return,
            fox_train_return,
            evaluation,
            update,
            is_best,
            has_evaluation,
        ))
    })
}

/// Validate arguments and create every durable trainer resource.
fn prepare_training(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
    initial_tuning: Option<ExperimentTuning>,
) -> Result<TrainingSetup, Box<dyn Error>> {
    // Resolve command-line values before creating a directory so invalid input
    // cannot leave a partial run behind.
    let options = parse_options(stage, arguments)?;
    let seeds = SeedConfig::from_root(options.seed);
    let mut algorithm = ecosystem_algorithm();
    algorithm.entropy_coefficient = options.entropy_coefficient;
    algorithm.log_std_max = options.log_std_max;
    algorithm.gamma = options.gamma;
    algorithm.gae_lambda = options.gae_lambda;
    let config = training_config(stage, &options, initial_tuning)?;
    let learners = create_learners(stage, &config, &algorithm, seeds, &options)?;
    let run_id = RunId::new(
        options
            .run_id
            .clone()
            .unwrap_or_else(|| format!("seed-{}", options.seed)),
    )?;
    let run = RunConfig::new(
        format!("ecosystem-{}", stage.as_key()),
        AlgorithmKind::Ppo,
        run_id,
        options.runs_root.clone(),
    )?;
    let paths = run.paths();
    paths.create_new()?;
    write_run_config(&paths, stage, &options, &algorithm, seeds, &config)?;
    let metrics = MetricsWriter::append(&paths.metrics_jsonl)?;
    Ok(TrainingSetup {
        options,
        seeds,
        algorithm,
        config,
        learners,
        paths,
        metrics,
    })
}

/// Resolve live HUD tuning while keeping explicit trainer arguments authoritative.
fn training_config(
    stage: CurriculumStage,
    options: &TrainOptions,
    initial_tuning: Option<ExperimentTuning>,
) -> Result<SimulationConfig, Box<dyn Error>> {
    let mut config = SimulationConfig::for_stage(stage)?;
    if let Some(tuning) = initial_tuning {
        config.apply_experiment_tuning(tuning)?;
    }
    config.episode_seconds = options.episode_seconds;
    Ok(config)
}

/// Evaluate the random policy and make its checkpoint baseline durable.
fn initialize_baseline(
    stage: CurriculumStage,
    learners: &Learners,
    config: &SimulationConfig,
    seeds: SeedConfig,
    options: &TrainOptions,
    paths: &RunPaths,
    metrics: &mut MetricsWriter,
) -> Result<EvaluationSummary, Box<dyn Error>> {
    // Persist all baseline artifacts before an optimizer update can replace
    // policy weights or change the graph's comparison point.
    let evaluation = evaluate_learners(
        learners,
        config.clone(),
        seeds.validation,
        options.eval_episodes,
    )?;
    save_checkpoint_set(stage, learners, paths, CheckpointKind::Best, config)?;
    save_checkpoint_set(stage, learners, paths, CheckpointKind::Latest, config)?;
    save_checkpoint_set(stage, learners, paths, CheckpointKind::Step(0), config)?;
    write_evaluation_metric(metrics, 0, evaluation, true, config.experiment_tuning())?;
    Ok(evaluation)
}

/// Execute the shared trainer with an optional visual progress boundary.
#[expect(
    clippy::too_many_lines,
    reason = "the coordinator keeps batch, evaluation, checkpoint, and UI publication order explicit"
)]
fn run_training_inner(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
    #[cfg(feature = "render")] mut tuning_provider: Option<&mut dyn FnMut() -> ExperimentTuning>,
    #[cfg(feature = "render")] mut reporter: Option<&mut dyn FnMut(TrainingProgress) -> bool>,
) -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "render")]
    let initial_tuning = tuning_provider.as_mut().map(|provider| provider());
    #[cfg(not(feature = "render"))]
    let initial_tuning = None;
    let TrainingSetup {
        options,
        seeds,
        algorithm,
        config: prepared_config,
        mut learners,
        paths,
        mut metrics,
    } = prepare_training(stage, arguments, initial_tuning)?;
    #[cfg(feature = "render")]
    let mut config = prepared_config;
    #[cfg(not(feature = "render"))]
    let config = prepared_config;
    // Open the visual world before evaluation to keep startup responsive.
    #[cfg(feature = "render")]
    if !report_training_start(
        &mut reporter,
        &learners,
        &config,
        &paths,
        &options,
        &algorithm,
    ) {
        return Ok(());
    }
    let has_foxes = config.fox_count > 0;
    let initial_evaluation = initialize_baseline(
        stage,
        &learners,
        &config,
        seeds,
        &options,
        &paths,
        &mut metrics,
    )?;
    let mut best_evaluation = initial_evaluation;
    #[cfg(feature = "render")]
    let mut profile_baseline = initial_evaluation;
    #[cfg(not(feature = "render"))]
    let profile_baseline = initial_evaluation;
    let mut output = io::BufWriter::new(io::stdout().lock());
    write_initial_output(&mut output, stage, initial_evaluation, &algorithm)?;
    // Publish the random initialization before the first optimizer update so
    // the visual demo can show a truthful learning baseline.
    #[cfg(feature = "render")]
    if !report_initial_evaluation(
        &mut reporter,
        &learners,
        &config,
        &paths,
        &options,
        &algorithm,
        initial_evaluation,
    ) {
        write_summary(&paths, stage, 0, initial_evaluation, best_evaluation)?;
        writeln!(output, "run_dir={}", paths.run_dir.display())?;
        return Ok(());
    }
    let mut global_steps = 0_u64;
    for iteration in 0..options.iterations {
        // Apply UI changes only between complete rollout batches so every PPO
        // update retains one stationary reward and dynamics definition.
        #[cfg(feature = "render")]
        if let Some(provider) = tuning_provider.as_mut() {
            if let Some(baseline) = apply_training_tuning(
                provider,
                &mut config,
                TuningPersistence {
                    learners: &learners,
                    paths: &paths,
                    options: &options,
                    algorithm: &algorithm,
                    seeds,
                    stage,
                },
            )? {
                best_evaluation = baseline;
                profile_baseline = baseline;
            }
        }
        let batch = collect_batch(
            &learners,
            config.clone(),
            seeds,
            iteration,
            options.rollout_episodes,
            &algorithm,
        )?;
        global_steps = global_steps.saturating_add(batch.joint_steps);
        let bunny_update = learners.bunny.update(&batch.bunny_sequences)?;
        let fox_update = if let Some(fox) = learners.fox.as_mut() {
            Some(fox.update(&batch.fox_sequences)?)
        } else {
            None
        };
        let bunny_lifetime = mean(&batch.bunny_lifetimes);
        let fox_lifetime = mean(&batch.fox_lifetimes);
        let bunny_return = mean(&batch.bunny_returns);
        let fox_return = mean(&batch.fox_returns);
        let update = combine_updates(bunny_update, fox_update);
        write_training_metric(
            &mut metrics,
            global_steps,
            iteration + 1,
            bunny_return,
            fox_return,
            bunny_lifetime,
            fox_lifetime,
            update,
            config.experiment_tuning(),
        )?;
        let should_evaluate = (iteration + 1).is_multiple_of(options.eval_interval)
            || iteration + 1 == options.iterations;
        let mut evaluation = None;
        let mut is_best = false;
        if should_evaluate {
            let result = evaluate_training_iteration(
                EvaluationPass {
                    stage,
                    learners: &learners,
                    config: &config,
                    paths: &paths,
                    options: &options,
                    metrics: &mut metrics,
                    validation_seed: seeds.validation,
                    has_foxes,
                },
                global_steps,
                best_evaluation,
            )?;
            is_best = result.1;
            if result.1 {
                let summary = result.0;
                best_evaluation = summary;
            }
            evaluation = Some(result.0);
        }
        let displayed_evaluation = evaluation.unwrap_or(best_evaluation);
        write_iteration_output(
            &mut output,
            stage,
            iteration + 1,
            global_steps,
            bunny_return,
            fox_return,
            bunny_lifetime,
            fox_lifetime,
            displayed_evaluation,
            update,
            is_best,
        )?;
        // Send the current policy after its metrics and checkpoint are durable.
        #[cfg(feature = "render")]
        if !report_training_update(
            &mut reporter,
            &learners,
            &config,
            &paths,
            &options,
            iteration + 1,
            global_steps,
            bunny_return,
            fox_return,
            displayed_evaluation,
            update,
            is_best,
            evaluation.is_some(),
        ) {
            break;
        }
    }
    write_summary(
        &paths,
        stage,
        global_steps,
        profile_baseline,
        best_evaluation,
    )?;
    writeln!(output, "run_dir={}", paths.run_dir.display())?;
    Ok(())
}

/// Evaluate, persist, and classify one scheduled policy snapshot.
fn evaluate_training_iteration(
    pass: EvaluationPass<'_>,
    global_steps: u64,
    prior_best: EvaluationSummary,
) -> Result<(EvaluationSummary, bool), Box<dyn Error>> {
    let summary = evaluate_learners(
        pass.learners,
        pass.config.clone(),
        pass.validation_seed,
        pass.options.eval_episodes,
    )?;
    save_checkpoint_set(
        pass.stage,
        pass.learners,
        pass.paths,
        CheckpointKind::Latest,
        pass.config,
    )?;
    save_checkpoint_set(
        pass.stage,
        pass.learners,
        pass.paths,
        CheckpointKind::Step(global_steps),
        pass.config,
    )?;
    let is_best =
        summary.selection_score(pass.has_foxes) > prior_best.selection_score(pass.has_foxes);
    if is_best {
        save_checkpoint_set(
            pass.stage,
            pass.learners,
            pass.paths,
            CheckpointKind::Best,
            pass.config,
        )?;
    }
    write_evaluation_metric(
        pass.metrics,
        global_steps,
        summary,
        is_best,
        pass.config.experiment_tuning(),
    )?;
    Ok((summary, is_best))
}

/// Apply a changed HUD profile and restart checkpoint selection on that profile.
#[cfg(feature = "render")]
fn apply_training_tuning(
    provider: &mut dyn FnMut() -> ExperimentTuning,
    config: &mut SimulationConfig,
    persistence: TuningPersistence<'_>,
) -> Result<Option<EvaluationSummary>, Box<dyn Error>> {
    let next_tuning = provider();
    if next_tuning == config.experiment_tuning() {
        return Ok(None);
    }
    config.apply_experiment_tuning(next_tuning)?;
    write_run_config(
        persistence.paths,
        persistence.stage,
        persistence.options,
        persistence.algorithm,
        persistence.seeds,
        config,
    )?;
    let baseline = evaluate_learners(
        persistence.learners,
        config.clone(),
        persistence.seeds.validation,
        persistence.options.eval_episodes,
    )?;
    save_checkpoint_set(
        persistence.stage,
        persistence.learners,
        persistence.paths,
        CheckpointKind::Best,
        config,
    )?;
    Ok(Some(baseline))
}

/// Write the column names and random-policy evaluation row.
fn write_initial_output(
    output: &mut impl Write,
    stage: CurriculumStage,
    evaluation: EvaluationSummary,
    algorithm: &RecurrentPpoConfig,
) -> Result<(), Box<dyn Error>> {
    // Keep the text stream machine-readable for terminal users and scripts.
    writeln!(
        output,
        "stage,iteration,global_steps,bunny_train_return,fox_train_return,bunny_train_lifetime,fox_train_lifetime,bunny_eval_return,fox_eval_return,bunny_eval_lifetime,fox_eval_lifetime,actor_loss,critic_loss,entropy,approx_kl,valid_samples,optimizer_updates,actor_lr,critic_lr,best"
    )?;
    writeln!(
        output,
        "{},0,0,0.000,0.000,0.000,0.000,{:.3},{:.3},{:.3},{:.3},0.000000,0.000000,0.000000,0.000000,0,0,{:.8},{:.8},true",
        stage.as_key(),
        evaluation.bunny_mean_return,
        evaluation.fox_mean_return,
        evaluation.bunny_mean_lifetime,
        evaluation.fox_mean_lifetime,
        algorithm.actor_learning_rate,
        algorithm.critic_learning_rate,
    )?;
    output.flush()?;
    Ok(())
}

/// Write one compact human-readable trainer progress row.
fn write_iteration_output(
    output: &mut impl Write,
    stage: CurriculumStage,
    iteration: usize,
    global_steps: u64,
    bunny_return: f32,
    fox_return: f32,
    bunny_lifetime: f32,
    fox_lifetime: f32,
    evaluation: EvaluationSummary,
    update: UpdateSummary,
    is_best: bool,
) -> Result<(), Box<dyn Error>> {
    // Keep stdout synchronized with the durable JSONL row and policy event.
    writeln!(
        output,
        "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{},{},{:.8},{:.8},{}",
        stage.as_key(),
        iteration,
        global_steps,
        bunny_return,
        fox_return,
        bunny_lifetime,
        fox_lifetime,
        evaluation.bunny_mean_return,
        evaluation.fox_mean_return,
        evaluation.bunny_mean_lifetime,
        evaluation.fox_mean_lifetime,
        update.actor_loss,
        update.critic_loss,
        update.entropy,
        update.approximate_kl,
        update.valid_samples,
        update.optimizer_updates,
        update.actor_learning_rate,
        update.critic_learning_rate,
        is_best,
    )?;
    output.flush()?;
    Ok(())
}

/// Reload saved networks and run deterministic evaluation in this process.
pub(super) fn run_evaluation(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let options = parse_eval_options(arguments)?;
    let bunny_path = resolve_bunny_checkpoint(&options.checkpoint);
    validate_checkpoint_stage(&bunny_path, stage)?;
    let mut config = SimulationConfig::for_stage(stage)?;
    let bunny_tuning = checkpoint_experiment_tuning(&bunny_path)?
        .ok_or("bunny checkpoint lacks experiment tuning")?;
    config.apply_experiment_tuning(bunny_tuning)?;
    if let Some(episode_seconds) = options.episode_seconds {
        config.episode_seconds = episode_seconds;
    }
    let algorithm = ecosystem_algorithm();
    let bunny = RecurrentPpoPolicy::load(
        &bunny_path,
        LOCAL_OBSERVATION_SIZE,
        GLOBAL_STATE_SIZE,
        MAX_AGENTS,
        &[-1.0, -1.0, -1.0],
        &[1.0, 1.0, 1.0],
        &algorithm,
    )?;
    let fox = if config.fox_count > 0 {
        let fox_path = options
            .fox_checkpoint
            .unwrap_or_else(|| sibling_fox_checkpoint(&bunny_path));
        validate_checkpoint_tuning_match(&bunny_path, &fox_path)?;
        Some(RecurrentPpoPolicy::load(
            &fox_path,
            LOCAL_OBSERVATION_SIZE,
            GLOBAL_STATE_SIZE,
            MAX_AGENTS,
            &[-1.0, -1.0, -1.0],
            &[1.0, 1.0, 1.0],
            &algorithm,
        )?)
    } else {
        None
    };
    let summary = evaluate_policies(&bunny, fox.as_ref(), config, options.seed, options.episodes)?;
    writeln!(
        io::stdout().lock(),
        "stage={} episodes={} bunny_mean_return={:.3} bunny_mean_lifetime={:.3} bunny_ci95=[{:.3},{:.3}] bunny_food_and_water={:.3} bunny_resource_consumers={} bunny_index_advantage={:.3} fox_mean_return={:.3} fox_mean_lifetime={:.3} fox_ci95=[{:.3},{:.3}] fox_prey_and_water={:.3} food_events={} predation_events={} predation_episodes={}/{} water_consumed={:.3} collision_contacts={} thorn_damage={:.3} overconsumption_damage={:.3} deaths=[starvation:{},dehydration:{},deprivation:{},predation:{},thorns:{},combined_damage:{},overconsumption:{},invalid_physics:{}] checkpoint={}",
        stage.as_key(),
        options.episodes,
        summary.bunny_mean_return,
        summary.bunny_mean_lifetime,
        summary.bunny_lifetime_lower,
        summary.bunny_lifetime_upper,
        summary.bunny_food_and_water_fraction,
        summary.bunny_resource_consumer_count,
        summary.bunny_index_lifetime_advantage_fraction,
        summary.fox_mean_return,
        summary.fox_mean_lifetime,
        summary.fox_lifetime_lower,
        summary.fox_lifetime_upper,
        summary.fox_food_and_water_fraction,
        summary.food_events,
        summary.predation_events,
        summary.episodes_with_predation,
        options.episodes,
        summary.water_consumed,
        summary.collision_contacts,
        summary.thorn_damage,
        summary.overconsumption_damage,
        summary.deaths.starvation,
        summary.deaths.dehydration,
        summary.deaths.deprivation,
        summary.deaths.predation,
        summary.deaths.thorns,
        summary.deaths.combined_damage,
        summary.deaths.overconsumption,
        summary.deaths.invalid_physics,
        bunny_path.display(),
    )?;
    Ok(())
}

/// Recurrent PPO settings matched to delayed satiation and hydration effects.
pub(super) fn ecosystem_algorithm() -> RecurrentPpoConfig {
    // Preserve the exploration scale that produced the measured survival gain.
    RecurrentPpoConfig {
        gamma: 0.999,
        gae_lambda: 1.0,
        actor_learning_rate: 3e-4,
        critic_learning_rate: 3e-4,
        entropy_coefficient: 0.005,
        epochs: 3,
        minibatch_sequences: 8,
        initial_log_std: -0.5,
        ..RecurrentPpoConfig::default()
    }
}

/// Parse the deliberately small reproducible training command surface.
fn parse_options(
    stage: CurriculumStage,
    mut arguments: impl Iterator<Item = String>,
) -> Result<TrainOptions, Box<dyn Error>> {
    let mut options = TrainOptions::for_stage(stage);
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--iterations" => options.iterations = value.parse()?,
            "--rollout-episodes" => options.rollout_episodes = value.parse()?,
            "--episode-seconds" => options.episode_seconds = value.parse()?,
            "--seed" => options.seed = value.parse()?,
            "--run-id" => options.run_id = Some(value),
            "--runs-root" => options.runs_root = PathBuf::from(value),
            "--eval-episodes" => options.eval_episodes = value.parse()?,
            "--eval-interval" => options.eval_interval = value.parse()?,
            "--entropy-coefficient" => options.entropy_coefficient = value.parse()?,
            "--log-std-max" => options.log_std_max = value.parse()?,
            "--gamma" => options.gamma = value.parse()?,
            "--gae-lambda" => options.gae_lambda = value.parse()?,
            "--resume" => options.resume = Some(PathBuf::from(value)),
            "--fox-resume" => options.fox_resume = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown train option {flag:?}").into()),
        }
    }
    if options.iterations == 0
        || options.rollout_episodes == 0
        || !(5..=300).contains(&options.episode_seconds)
        || options.eval_episodes == 0
        || options.eval_interval == 0
        || !options.entropy_coefficient.is_finite()
        || options.entropy_coefficient < 0.0
        || !options.log_std_max.is_finite()
        || options.log_std_max <= -5.0
        || !options.gamma.is_finite()
        || !(0.0..=1.0).contains(&options.gamma)
        || !options.gae_lambda.is_finite()
        || !(0.0..=1.0).contains(&options.gae_lambda)
    {
        return Err(
            "count options must be positive; --episode-seconds must be in 5..=300; entropy, gamma, and GAE lambda must be finite and bounded; --log-std-max must be finite and greater than -5"
                .into(),
        );
    }
    Ok(options)
}

/// Parse evaluation arguments without inheriting training-only options.
fn parse_eval_options(
    mut arguments: impl Iterator<Item = String>,
) -> Result<EvalOptions, Box<dyn Error>> {
    let mut checkpoint = None;
    let mut fox_checkpoint = None;
    let mut episodes = 32;
    let mut episode_seconds = None;
    let mut seed = 101;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--fox-checkpoint" => fox_checkpoint = Some(PathBuf::from(value)),
            "--episodes" => episodes = value.parse()?,
            "--episode-seconds" => episode_seconds = Some(value.parse()?),
            "--seed" => seed = value.parse()?,
            _ => return Err(format!("unknown eval option {flag:?}").into()),
        }
    }
    if episodes == 0 || episode_seconds.is_some_and(|seconds| !(5..=300).contains(&seconds)) {
        return Err("--episodes must exceed zero and --episode-seconds must be in 5..=300".into());
    }
    Ok(EvalOptions {
        checkpoint: checkpoint.ok_or("eval requires --checkpoint <run-dir-or-mpk>")?,
        fox_checkpoint,
        episodes,
        episode_seconds,
        seed,
    })
}

/// Artifact checkpoint role and optional global step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckpointKind {
    /// Validation-selected checkpoint.
    Best,

    /// Most recently evaluated checkpoint.
    Latest,

    /// Immutable chronological checkpoint.
    Step(u64),
}

/// Save bunny and optional fox networks as one synchronized checkpoint set.
fn save_checkpoint_set(
    stage: CurriculumStage,
    learners: &Learners,
    paths: &RunPaths,
    kind: CheckpointKind,
    config: &SimulationConfig,
) -> Result<(), Box<dyn Error>> {
    let (bunny_path, fox_path) = checkpoint_paths(paths, kind);
    invalidate_checkpoint_profile(&bunny_path)?;
    if learners.fox.is_some() {
        invalidate_checkpoint_profile(&fox_path)?;
    }
    learners.bunny.policy().save(&bunny_path)?;
    write_checkpoint_profile(&bunny_path, stage, config.experiment_tuning())?;
    if let Some(fox) = &learners.fox {
        fox.policy().save(&fox_path)?;
        write_checkpoint_profile(&fox_path, stage, config.experiment_tuning())?;
    }
    Ok(())
}

/// Remove prior metadata before a policy overwrite so failures remain closed.
#[expect(
    clippy::allow_attributes,
    reason = "the repository disallows synchronous filesystem helpers by default"
)]
#[allow(
    clippy::disallowed_methods,
    reason = "checkpoint metadata is invalidated before synchronous policy replacement"
)]
fn invalidate_checkpoint_profile(checkpoint: &Path) -> Result<(), Box<dyn Error>> {
    let profile = checkpoint_config_path(checkpoint);
    match std::fs::remove_file(profile) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Persist the exact environment profile paired with one immutable network.
#[expect(
    clippy::allow_attributes,
    reason = "the repository disallows synchronous filesystem helpers by default"
)]
#[allow(
    clippy::disallowed_methods,
    reason = "bounded checkpoint metadata writes accompany synchronous policy saves"
)]
pub(super) fn write_checkpoint_profile(
    checkpoint: &Path,
    stage: CurriculumStage,
    tuning: ExperimentTuning,
) -> Result<(), Box<dyn Error>> {
    let path = checkpoint.with_extension("config.json");
    let mut file = File::create(path)?;
    writeln!(
        file,
        "{}",
        serde_json::json!({
            "checkpoint_profile": CHECKPOINT_PROFILE,
            "stage": stage.as_key(),
            "experiment_tuning": experiment_tuning_json(tuning),
        })
    )?;
    Ok(())
}

/// Resolve synchronized bunny and fox checkpoint paths.
fn checkpoint_paths(paths: &RunPaths, kind: CheckpointKind) -> (PathBuf, PathBuf) {
    match kind {
        CheckpointKind::Best => (
            paths.best_checkpoint.clone(),
            paths.run_dir.join("best-fox.mpk"),
        ),
        CheckpointKind::Latest => (
            paths.latest_checkpoint.clone(),
            paths.checkpoints_dir.join("latest-fox.mpk"),
        ),
        CheckpointKind::Step(step) => (
            paths.checkpoints_dir.join(format!("step-{step:06}.mpk")),
            paths
                .checkpoints_dir
                .join(format!("step-{step:06}-fox.mpk")),
        ),
    }
}

/// Resolve a run directory to its root best checkpoint.
pub(super) fn resolve_bunny_checkpoint(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("best.mpk")
    } else {
        path.to_path_buf()
    }
}

/// Infer the synchronized fox checkpoint name beside a bunny checkpoint.
pub(super) fn sibling_fox_checkpoint(bunny_path: &Path) -> PathBuf {
    let stem = bunny_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("best");
    bunny_path.with_file_name(format!("{stem}-fox.mpk"))
}

/// Evaluate frozen mean actions on the same declared environment settings.
fn evaluate_learners(
    learners: &Learners,
    config: SimulationConfig,
    validation_seed: u64,
    episodes: usize,
) -> Result<EvaluationSummary, Box<dyn Error>> {
    let bunny = learners.bunny.policy();
    let fox = learners.fox.as_ref().map(RecurrentPpoAgent::policy);
    evaluate_policies(&bunny, fox.as_ref(), config, validation_seed, episodes)
}

/// Evaluate already loaded policy snapshots on fixed environment seeds.
fn evaluate_policies(
    bunny: &RecurrentPpoPolicy,
    fox: Option<&RecurrentPpoPolicy>,
    config: SimulationConfig,
    validation_seed: u64,
    episodes: usize,
) -> Result<EvaluationSummary, Box<dyn Error>> {
    let validation_seeds = SeedConfig::from_root(validation_seed);
    let mut bunny_lifetimes = Vec::new();
    let mut fox_lifetimes = Vec::new();
    let mut bunny_returns = Vec::new();
    let mut fox_returns = Vec::new();
    let mut bunny_resource_successes = 0_u32;
    let mut bunny_resource_consumers = [false; MAX_AGENTS];
    let mut bunny_lifetimes_by_id: [Vec<f32>; MAX_AGENTS] = std::array::from_fn(|_| Vec::new());
    let mut bunny_count = 0_u32;
    let mut fox_resource_successes = 0_u32;
    let mut fox_count = 0_u32;
    let mut collision_contacts = 0_u64;
    let mut thorn_damage = 0.0;
    let mut overconsumption_damage = 0.0;
    let mut food_events = 0_u64;
    let mut predation_events = 0_u64;
    let mut water_consumed = 0.0;
    let mut episodes_with_predation = 0_u32;
    let mut episodes_without_predation = 0_u32;
    let mut deaths = EvaluationDeathCounts::default();
    for episode in 0..episodes {
        let seed = validation_seeds.environment_episode(0, episode as u64);
        let metrics = deterministic_episode(config.clone(), seed, episode, bunny, fox)?;
        let had_predation = metrics
            .iter()
            .any(|agent| agent.death_cause == Some(super::domain::DeathCause::Predation));
        episodes_with_predation = episodes_with_predation.saturating_add(u32::from(had_predation));
        episodes_without_predation =
            episodes_without_predation.saturating_add(u32::from(!had_predation));
        for agent in metrics {
            collision_contacts =
                collision_contacts.saturating_add(u64::from(agent.collision_contacts));
            thorn_damage += agent.thorn_damage;
            overconsumption_damage += agent.overconsumption_damage;
            food_events = food_events.saturating_add(u64::from(agent.food_eaten));
            predation_events = predation_events.saturating_add(u64::from(agent.kills));
            water_consumed += agent.water_consumed;
            match agent.death_cause {
                Some(super::domain::DeathCause::Starvation) => {
                    deaths.starvation = deaths.starvation.saturating_add(1);
                }
                Some(super::domain::DeathCause::Dehydration) => {
                    deaths.dehydration = deaths.dehydration.saturating_add(1);
                }
                Some(super::domain::DeathCause::Deprivation) => {
                    deaths.deprivation = deaths.deprivation.saturating_add(1);
                }
                Some(super::domain::DeathCause::Predation) => {
                    deaths.predation = deaths.predation.saturating_add(1);
                }
                Some(super::domain::DeathCause::Thorns) => {
                    deaths.thorns = deaths.thorns.saturating_add(1);
                }
                Some(super::domain::DeathCause::CombinedDamage) => {
                    deaths.combined_damage = deaths.combined_damage.saturating_add(1);
                }
                Some(super::domain::DeathCause::InvalidPhysics) => {
                    deaths.invalid_physics = deaths.invalid_physics.saturating_add(1);
                }
                None => {}
            }
            match agent.species {
                Species::Bunny => {
                    bunny_lifetimes.push(agent.lifetime_seconds);
                    bunny_returns.push(agent.episode_return);
                    let identity = usize::from(agent.id.0);
                    let Some((identity_lifetimes, consumed_resource)) = bunny_lifetimes_by_id
                        .get_mut(identity)
                        .zip(bunny_resource_consumers.get_mut(identity))
                    else {
                        return Err("bunny identity exceeds fixed evaluation capacity".into());
                    };
                    identity_lifetimes.push(agent.lifetime_seconds);
                    bunny_count = bunny_count.saturating_add(1);
                    *consumed_resource |= agent.food_eaten > 0 || agent.water_consumed > 0.0;
                    bunny_resource_successes = bunny_resource_successes.saturating_add(u32::from(
                        agent.food_eaten > 0 && agent.water_consumed > 0.0,
                    ));
                }
                Species::Fox => {
                    fox_lifetimes.push(agent.lifetime_seconds);
                    fox_returns.push(agent.episode_return);
                    fox_count = fox_count.saturating_add(1);
                    fox_resource_successes = fox_resource_successes
                        .saturating_add(u32::from(agent.kills > 0 && agent.water_consumed > 0.0));
                }
            }
        }
    }
    let (bunny_lifetime_lower, bunny_lifetime_upper) =
        bootstrap_mean_interval(&bunny_lifetimes, validation_seed ^ 0x6275_6e6e_795f_6369);
    let (fox_lifetime_lower, fox_lifetime_upper) =
        bootstrap_mean_interval(&fox_lifetimes, validation_seed ^ 0x666f_785f_6369_0001);
    let bunny_identity_lifetimes = bunny_lifetimes_by_id
        .get(..config.bunny_count)
        .ok_or("bunny count exceeds fixed evaluation capacity")?;
    Ok(EvaluationSummary {
        bunny_mean_return: mean(&bunny_returns),
        fox_mean_return: mean(&fox_returns),
        bunny_mean_lifetime: mean(&bunny_lifetimes),
        fox_mean_lifetime: mean(&fox_lifetimes),
        bunny_lifetime_lower,
        bunny_lifetime_upper,
        fox_lifetime_lower,
        fox_lifetime_upper,
        bunny_food_and_water_fraction: fraction(bunny_resource_successes, bunny_count),
        bunny_resource_consumer_count: u32::try_from(
            bunny_resource_consumers
                .into_iter()
                .filter(|consumed| *consumed)
                .count(),
        )
        .unwrap_or(u32::MAX),
        bunny_index_lifetime_advantage_fraction: index_lifetime_advantage(bunny_identity_lifetimes),
        fox_food_and_water_fraction: fraction(fox_resource_successes, fox_count),
        collision_contacts,
        thorn_damage,
        overconsumption_damage,
        food_events,
        predation_events,
        water_consumed,
        episodes_with_predation,
        episodes_without_predation,
        deaths,
    })
}

/// Run one evaluation episode without exploration or centralized actor input.
fn deterministic_episode(
    config: SimulationConfig,
    seed: u64,
    spawn_rotation: usize,
    bunny: &RecurrentPpoPolicy,
    fox: Option<&RecurrentPpoPolicy>,
) -> Result<Vec<AgentEpisodeMetrics>, Box<dyn Error>> {
    let mut ecosystem = Ecosystem::new_rotated(config, seed, spawn_rotation)?;
    let mut current = ecosystem.state();
    let mut memories = BTreeMap::<AgentId, RecurrentMemory>::new();
    loop {
        let mut actions = Vec::with_capacity(current.agents.len());
        let mut next_memories = BTreeMap::new();
        for (id, species, observation) in &current.agents {
            let policy = match species {
                Species::Bunny => bunny,
                Species::Fox => fox.ok_or("fox agent exists without an evaluation policy")?,
            };
            let memory = memories
                .entry(*id)
                .or_insert_with(|| policy.initial_memory());
            let action = policy.mean_action(observation, memory)?;
            let [forward, turn, gaze] = action.action.as_slice() else {
                return Err("ecosystem policy must emit exactly three action axes".into());
            };
            actions.push((*id, LocomotionAction::new(*forward, *turn, *gaze)?));
            next_memories.insert(*id, action.next_memory);
        }
        let result = ecosystem.step(&actions)?;
        let mut next_agents = Vec::new();
        for agent_step in result.agents {
            if agent_step.status == EpisodeStatus::Continuing {
                if let Some(memory) = next_memories.remove(&agent_step.id) {
                    memories.insert(agent_step.id, memory);
                }
                next_agents.push((agent_step.id, agent_step.species, agent_step.observation));
            } else {
                memories.remove(&agent_step.id);
            }
        }
        if result.is_done {
            break;
        }
        current.agents = next_agents;
        current.global_state = result.global_state;
    }
    Ok(ecosystem.episode_metrics())
}

/// Weight update diagnostics by each species' valid agent-time contribution.
fn combine_updates(bunny: RecurrentPpoUpdate, fox: Option<RecurrentPpoUpdate>) -> UpdateSummary {
    let fox_valid_samples = fox.map_or(0, |update| update.valid_samples);
    let valid_samples = bunny.valid_samples.saturating_add(fox_valid_samples);
    let divisor = valid_samples as f64;
    let bunny_weight = bunny.valid_samples as f64 / divisor;
    let fox_weight = fox_valid_samples as f64 / divisor;

    // Policies retain separate optimizers, but their diagnostics describe one
    // joint batch and therefore share the agent-time denominator.
    UpdateSummary {
        valid_samples,
        optimizer_updates: bunny
            .optimizer_updates
            .saturating_add(fox.map_or(0, |update| update.optimizer_updates)),
        actor_loss: bunny.actor_loss.mul_add(
            bunny_weight,
            fox.map_or(0.0, |update| update.actor_loss) * fox_weight,
        ),
        critic_loss: bunny.critic_loss.mul_add(
            bunny_weight,
            fox.map_or(0.0, |update| update.critic_loss) * fox_weight,
        ),
        entropy: bunny.entropy.mul_add(
            bunny_weight,
            fox.map_or(0.0, |update| update.entropy) * fox_weight,
        ),
        approximate_kl: bunny.approximate_kl.mul_add(
            bunny_weight,
            fox.map_or(0.0, |update| update.approximate_kl) * fox_weight,
        ),
        actor_learning_rate: bunny.actor_learning_rate,
        critic_learning_rate: bunny.critic_learning_rate,
    }
}

/// Attach the complete playground profile to one durable metric row.
fn with_experiment_metrics(record: MetricRecord, tuning: ExperimentTuning) -> MetricRecord {
    // Use the same field set for training and evaluation rows so offline tools
    // can reconstruct each applied profile at any global step.
    record
        .with_field(
            "experiment/perception_ray_count",
            MetricValue::Number(f64::from(tuning.perception_ray_count.get())),
        )
        .with_field(
            "experiment/food_reward",
            MetricValue::Number(f64::from(tuning.food_reward)),
        )
        .with_field(
            "experiment/drink_reward",
            MetricValue::Number(f64::from(tuning.drink_reward)),
        )
        .with_field(
            "experiment/maximum_hit_points",
            MetricValue::Number(f64::from(tuning.maximum_hit_points)),
        )
        .with_field(
            "experiment/initial_hit_points",
            MetricValue::Number(f64::from(tuning.initial_hit_points)),
        )
        .with_field(
            "experiment/maximum_satiation",
            MetricValue::Number(f64::from(tuning.maximum_satiation)),
        )
        .with_field(
            "experiment/initial_satiation",
            MetricValue::Number(f64::from(tuning.initial_satiation)),
        )
        .with_field(
            "experiment/maximum_hydration",
            MetricValue::Number(f64::from(tuning.maximum_hydration)),
        )
        .with_field(
            "experiment/initial_hydration",
            MetricValue::Number(f64::from(tuning.initial_hydration)),
        )
        .with_field(
            "experiment/need_loss_interval_seconds",
            MetricValue::Number(f64::from(tuning.need_loss_interval_seconds)),
        )
        .with_field(
            "experiment/starvation_damage_interval_seconds",
            MetricValue::Number(f64::from(tuning.starvation_damage_interval_seconds)),
        )
        .with_field(
            "experiment/dehydration_damage_interval_seconds",
            MetricValue::Number(f64::from(tuning.dehydration_damage_interval_seconds)),
        )
        .with_field(
            "experiment/movement_need_cost_percent",
            MetricValue::Number(f64::from(tuning.movement_need_cost_percent)),
        )
        .with_field(
            "experiment/movement_speed_multiplier",
            MetricValue::Number(f64::from(tuning.movement_speed_multiplier)),
        )
        .with_field(
            "experiment/gaze_yaw_limit_degrees",
            MetricValue::Number(f64::from(tuning.gaze_yaw_limit_degrees)),
        )
        .with_field(
            "experiment/episode_seconds",
            MetricValue::Number(f64::from(tuning.episode_seconds)),
        )
}

/// Write one optimizer and training-lifetime metric row.
fn write_training_metric(
    metrics: &mut MetricsWriter,
    global_steps: u64,
    iteration: usize,
    bunny_return: f32,
    fox_return: f32,
    bunny_lifetime: f32,
    fox_lifetime: f32,
    update: UpdateSummary,
    tuning: ExperimentTuning,
) -> Result<(), Box<dyn Error>> {
    // Build the optimizer row first, then attach the shared experiment fields.
    let record = MetricRecord::new(global_steps)
        .with_field("train/iteration", MetricValue::Number(iteration as f64))
        .with_field(
            "train/bunny_mean_return",
            MetricValue::Number(f64::from(bunny_return)),
        )
        .with_field(
            "train/fox_mean_return",
            MetricValue::Number(f64::from(fox_return)),
        )
        .with_field(
            "train/bunny_mean_lifetime",
            MetricValue::Number(f64::from(bunny_lifetime)),
        )
        .with_field(
            "train/fox_mean_lifetime",
            MetricValue::Number(f64::from(fox_lifetime)),
        )
        .with_field("train/actor_loss", MetricValue::Number(update.actor_loss))
        .with_field("train/critic_loss", MetricValue::Number(update.critic_loss))
        .with_field("train/entropy", MetricValue::Number(update.entropy))
        .with_field(
            "train/approximate_kl",
            MetricValue::Number(update.approximate_kl),
        )
        .with_field(
            "train/valid_samples",
            MetricValue::Number(update.valid_samples as f64),
        )
        .with_field(
            "train/optimizer_updates",
            MetricValue::Number(update.optimizer_updates as f64),
        )
        .with_field(
            "train/actor_learning_rate",
            MetricValue::Number(update.actor_learning_rate),
        )
        .with_field(
            "train/critic_learning_rate",
            MetricValue::Number(update.critic_learning_rate),
        );
    metrics.write_record(&with_experiment_metrics(record, tuning))?;
    Ok(())
}

/// Write one fixed-seed evaluation row to the run metrics stream.
fn write_evaluation_metric(
    metrics: &mut MetricsWriter,
    global_steps: u64,
    summary: EvaluationSummary,
    is_best: bool,
    tuning: ExperimentTuning,
) -> Result<(), Box<dyn Error>> {
    // Evaluation retains its outcome fields beside the profile that produced
    // them, including rows written immediately after a UI change.
    let record = MetricRecord::new(global_steps)
        .with_field(
            "eval/bunny_mean_return",
            MetricValue::Number(f64::from(summary.bunny_mean_return)),
        )
        .with_field(
            "eval/fox_mean_return",
            MetricValue::Number(f64::from(summary.fox_mean_return)),
        )
        .with_field(
            "eval/bunny_mean_lifetime",
            MetricValue::Number(f64::from(summary.bunny_mean_lifetime)),
        )
        .with_field(
            "eval/fox_mean_lifetime",
            MetricValue::Number(f64::from(summary.fox_mean_lifetime)),
        )
        .with_field(
            "eval/bunny_lifetime_ci95_lower",
            MetricValue::Number(f64::from(summary.bunny_lifetime_lower)),
        )
        .with_field(
            "eval/bunny_lifetime_ci95_upper",
            MetricValue::Number(f64::from(summary.bunny_lifetime_upper)),
        )
        .with_field(
            "eval/fox_lifetime_ci95_lower",
            MetricValue::Number(f64::from(summary.fox_lifetime_lower)),
        )
        .with_field(
            "eval/fox_lifetime_ci95_upper",
            MetricValue::Number(f64::from(summary.fox_lifetime_upper)),
        )
        .with_field(
            "eval/bunny_food_and_water_fraction",
            MetricValue::Number(f64::from(summary.bunny_food_and_water_fraction)),
        )
        .with_field(
            "eval/bunny_resource_consumer_count",
            MetricValue::Number(f64::from(summary.bunny_resource_consumer_count)),
        )
        .with_field(
            "eval/bunny_index_lifetime_advantage_fraction",
            MetricValue::Number(f64::from(summary.bunny_index_lifetime_advantage_fraction)),
        )
        .with_field(
            "eval/fox_prey_and_water_fraction",
            MetricValue::Number(f64::from(summary.fox_food_and_water_fraction)),
        )
        .with_field(
            "eval/collision_contacts",
            MetricValue::Number(summary.collision_contacts as f64),
        )
        .with_field(
            "eval/thorn_damage",
            MetricValue::Number(f64::from(summary.thorn_damage)),
        )
        .with_field(
            "eval/overconsumption_damage",
            MetricValue::Number(f64::from(summary.overconsumption_damage)),
        )
        .with_field(
            "eval/food_events",
            MetricValue::Number(summary.food_events as f64),
        )
        .with_field(
            "eval/predation_events",
            MetricValue::Number(summary.predation_events as f64),
        )
        .with_field(
            "eval/episodes_with_predation",
            MetricValue::Number(f64::from(summary.episodes_with_predation)),
        )
        .with_field(
            "eval/episodes_without_predation",
            MetricValue::Number(f64::from(summary.episodes_without_predation)),
        )
        .with_field(
            "eval/water_consumed",
            MetricValue::Number(f64::from(summary.water_consumed)),
        )
        .with_field("checkpoint/best", MetricValue::Bool(is_best));
    let record = with_evaluation_death_metrics(record, summary.deaths);
    metrics.write_record(&with_experiment_metrics(record, tuning))?;
    Ok(())
}

/// Attach terminal-cause counts to one fixed-seed evaluation row.
fn with_evaluation_death_metrics(
    record: MetricRecord,
    deaths: EvaluationDeathCounts,
) -> MetricRecord {
    record
        .with_field(
            "eval/deaths_starvation",
            MetricValue::Number(f64::from(deaths.starvation)),
        )
        .with_field(
            "eval/deaths_dehydration",
            MetricValue::Number(f64::from(deaths.dehydration)),
        )
        .with_field(
            "eval/deaths_deprivation",
            MetricValue::Number(f64::from(deaths.deprivation)),
        )
        .with_field(
            "eval/deaths_predation",
            MetricValue::Number(f64::from(deaths.predation)),
        )
        .with_field(
            "eval/deaths_thorns",
            MetricValue::Number(f64::from(deaths.thorns)),
        )
        .with_field(
            "eval/deaths_combined_damage",
            MetricValue::Number(f64::from(deaths.combined_damage)),
        )
        .with_field(
            "eval/deaths_overconsumption",
            MetricValue::Number(f64::from(deaths.overconsumption)),
        )
        .with_field(
            "eval/deaths_invalid_physics",
            MetricValue::Number(f64::from(deaths.invalid_physics)),
        )
}

/// Write the latest reproducibility settings at a stationary batch boundary.
#[expect(
    clippy::allow_attributes,
    reason = "the repository disallows synchronous filesystem helpers by default"
)]
#[allow(
    clippy::disallowed_methods,
    reason = "bounded startup writes occur before the synchronous training loop"
)]
fn write_run_config(
    paths: &RunPaths,
    stage: CurriculumStage,
    options: &TrainOptions,
    algorithm: &RecurrentPpoConfig,
    seeds: SeedConfig,
    config: &SimulationConfig,
) -> Result<(), Box<dyn Error>> {
    let mut config_file = File::create(&paths.config_json)?;
    let resume = json_optional_path(options.resume.as_deref());
    let fox_resume = json_optional_path(options.fox_resume.as_deref());
    writeln!(
        config_file,
        "{{\"checkpoint_profile\":{CHECKPOINT_PROFILE},\"stage\":\"{}\",\"iterations\":{},\"rollout_episodes\":{},\"episode_seconds\":{},\"eval_episodes\":{},\"eval_interval\":{},\"actor_hidden_size\":{},\"actor_learning_rate\":{},\"critic_learning_rate\":{},\"entropy_coefficient\":{},\"initial_log_std\":{},\"log_std_max\":{},\"gamma\":{},\"gae_lambda\":{},\"epochs\":{},\"minibatch_sequences\":{},\"resume\":{},\"fox_resume\":{},\"experiment_tuning\":{}}}",
        stage.as_key(),
        options.iterations,
        options.rollout_episodes,
        config.episode_seconds,
        options.eval_episodes,
        options.eval_interval,
        algorithm.actor_hidden_size,
        algorithm.actor_learning_rate,
        algorithm.critic_learning_rate,
        algorithm.entropy_coefficient,
        algorithm.initial_log_std,
        algorithm.log_std_max,
        algorithm.gamma,
        algorithm.gae_lambda,
        algorithm.epochs,
        algorithm.minibatch_sequences,
        resume,
        fox_resume,
        experiment_tuning_json(config.experiment_tuning()),
    )?;
    let mut seed_file = File::create(&paths.seeds_json)?;
    writeln!(
        seed_file,
        "{{\"root\":{},\"environment\":{},\"action\":{},\"model\":{},\"validation\":{},\"test\":{},\"demo\":{}}}",
        seeds.root,
        seeds.env_reset,
        seeds.action,
        seeds.model,
        seeds.validation,
        seeds.test,
        seeds.demo,
    )?;
    Ok(())
}

/// Encode an optional path as one valid JSON string or null value.
fn json_optional_path(path: Option<&Path>) -> String {
    path.map_or_else(
        || "null".to_owned(),
        |path| {
            let escaped = path
                .to_string_lossy()
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r");
            format!("\"{escaped}\"")
        },
    )
}

/// Encode the complete live playground profile in the durable run config.
pub(super) fn experiment_tuning_json(tuning: ExperimentTuning) -> serde_json::Value {
    serde_json::json!({
        "perception_ray_count": tuning.perception_ray_count.get(),
        "food_reward": tuning.food_reward,
        "drink_reward": tuning.drink_reward,
        "maximum_hit_points": tuning.maximum_hit_points,
        "initial_hit_points": tuning.initial_hit_points,
        "maximum_satiation": tuning.maximum_satiation,
        "initial_satiation": tuning.initial_satiation,
        "maximum_hydration": tuning.maximum_hydration,
        "initial_hydration": tuning.initial_hydration,
        "need_loss_interval_seconds": tuning.need_loss_interval_seconds,
        "starvation_damage_interval_seconds": tuning.starvation_damage_interval_seconds,
        "dehydration_damage_interval_seconds": tuning.dehydration_damage_interval_seconds,
        "movement_need_cost_percent": tuning.movement_need_cost_percent,
        "movement_speed_multiplier": tuning.movement_speed_multiplier,
        "gaze_yaw_limit_degrees": tuning.gaze_yaw_limit_degrees,
        "episode_seconds": tuning.episode_seconds,
    })
}

/// Load the saved playground profile beside a run checkpoint when available.
pub(super) fn checkpoint_experiment_tuning(
    checkpoint: &Path,
) -> Result<Option<ExperimentTuning>, Box<dyn Error>> {
    checkpoint_profile(checkpoint).map(|(_, tuning)| Some(tuning))
}

/// Read and validate one exact checkpoint's stage and tuning profile.
fn checkpoint_profile(checkpoint: &Path) -> Result<(String, ExperimentTuning), Box<dyn Error>> {
    let config_path = checkpoint_config_path(checkpoint);
    if !config_path.exists() {
        return Err(format!(
            "checkpoint requires a companion profile-{} config: {}",
            CHECKPOINT_PROFILE,
            config_path.display()
        )
        .into());
    }
    let value: serde_json::Value = serde_json::from_reader(File::open(config_path)?)?;
    validate_checkpoint_profile(&value)?;
    let stage = value
        .get("stage")
        .and_then(serde_json::Value::as_str)
        .ok_or("checkpoint config lacks stage")?
        .to_owned();
    let Some(tuning) = value.get("experiment_tuning") else {
        return Err("run config lacks experiment_tuning".into());
    };
    Ok((stage, parse_experiment_tuning(tuning)?))
}

/// Reject a checkpoint created for a different curriculum stage.
pub(super) fn validate_checkpoint_stage(
    checkpoint: &Path,
    expected: CurriculumStage,
) -> Result<(), Box<dyn Error>> {
    let (actual, _) = checkpoint_profile(checkpoint)?;
    if actual == expected.as_key() {
        Ok(())
    } else {
        Err(format!(
            "checkpoint stage {actual} is incompatible with required stage {}: {}",
            expected.as_key(),
            checkpoint.display()
        )
        .into())
    }
}

/// Accept a current-stage or direct-predecessor policy for curriculum transfer.
fn validate_resume_checkpoint_stage(
    checkpoint: &Path,
    target: CurriculumStage,
) -> Result<(), Box<dyn Error>> {
    let (actual, _) = checkpoint_profile(checkpoint)?;
    let predecessor = match target {
        CurriculumStage::Survival => None,
        CurriculumStage::Competition => Some(CurriculumStage::Survival),
        CurriculumStage::PredatorPrey => Some(CurriculumStage::Competition),
        CurriculumStage::Obstacles => Some(CurriculumStage::PredatorPrey),
    };
    if actual == target.as_key() || predecessor.is_some_and(|stage| actual == stage.as_key()) {
        Ok(())
    } else {
        Err(format!(
            "checkpoint stage {actual} cannot resume target stage {}: {}",
            target.as_key(),
            checkpoint.display()
        )
        .into())
    }
}

/// Reject run configs created for incompatible mechanics or tensor contracts.
fn validate_checkpoint_profile(value: &serde_json::Value) -> Result<(), Box<dyn Error>> {
    let profile = value
        .get("checkpoint_profile")
        .and_then(serde_json::Value::as_u64)
        .ok_or("run config lacks checkpoint_profile")?;
    if profile == CHECKPOINT_PROFILE {
        Ok(())
    } else {
        Err(format!(
            "checkpoint profile {profile} is incompatible with required profile {CHECKPOINT_PROFILE}"
        )
        .into())
    }
}

/// Decode and validate one saved playground profile.
fn parse_experiment_tuning(tuning: &serde_json::Value) -> Result<ExperimentTuning, Box<dyn Error>> {
    let ray_count = tuning
        .get("perception_ray_count")
        .and_then(serde_json::Value::as_u64)
        .ok_or("run config lacks experiment_tuning.perception_ray_count")?;
    let ray_count = u8::try_from(ray_count)?;
    let number = |key| json_f32(tuning, key);
    let integer = |key| {
        tuning
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("run config lacks experiment_tuning.{key}"))
    };
    let tuning = ExperimentTuning {
        perception_ray_count: super::domain::PerceptionRayCount::try_from(ray_count)?,
        food_reward: number("food_reward")?,
        drink_reward: number("drink_reward")?,
        maximum_hit_points: u8::try_from(integer("maximum_hit_points")?)?,
        initial_hit_points: u8::try_from(integer("initial_hit_points")?)?,
        maximum_satiation: u8::try_from(integer("maximum_satiation")?)?,
        initial_satiation: u8::try_from(integer("initial_satiation")?)?,
        maximum_hydration: u8::try_from(integer("maximum_hydration")?)?,
        initial_hydration: u8::try_from(integer("initial_hydration")?)?,
        need_loss_interval_seconds: u16::try_from(integer("need_loss_interval_seconds")?)?,
        starvation_damage_interval_seconds: u16::try_from(integer(
            "starvation_damage_interval_seconds",
        )?)?,
        dehydration_damage_interval_seconds: u16::try_from(integer(
            "dehydration_damage_interval_seconds",
        )?)?,
        movement_need_cost_percent: u8::try_from(integer("movement_need_cost_percent")?)?,
        movement_speed_multiplier: number("movement_speed_multiplier")?,
        gaze_yaw_limit_degrees: number("gaze_yaw_limit_degrees")?,
        episode_seconds: u16::try_from(integer("episode_seconds")?)?,
    };
    tuning.validate()?;
    Ok(tuning)
}

/// Resolve the immutable profile stored beside one exact checkpoint.
fn checkpoint_config_path(checkpoint: &Path) -> PathBuf {
    checkpoint.with_extension("config.json")
}

/// Reject a synchronized policy pair trained under different mechanics.
pub(super) fn validate_checkpoint_tuning_match(
    bunny_checkpoint: &Path,
    fox_checkpoint: &Path,
) -> Result<(), Box<dyn Error>> {
    let bunny = checkpoint_profile(bunny_checkpoint)?;
    let fox = checkpoint_profile(fox_checkpoint)?;
    if bunny == fox {
        Ok(())
    } else {
        Err(format!(
            "checkpoint profile mismatch: {} and {} were trained under different mechanics",
            bunny_checkpoint.display(),
            fox_checkpoint.display()
        )
        .into())
    }
}

/// Read one finite number from the saved playground profile.
fn json_f32(value: &serde_json::Value, key: &str) -> Result<f32, Box<dyn Error>> {
    let number = value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("run config lacks experiment_tuning.{key}"))?;
    let number = number as f32;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(format!("run config has non-finite experiment_tuning.{key}").into())
    }
}

/// Write the current evidence boundary at the end of training.
#[expect(
    clippy::allow_attributes,
    reason = "the repository disallows synchronous filesystem helpers by default"
)]
#[allow(
    clippy::disallowed_methods,
    reason = "bounded summary output occurs after the synchronous training loop"
)]
fn write_summary(
    paths: &RunPaths,
    stage: CurriculumStage,
    global_steps: u64,
    initial: EvaluationSummary,
    best: EvaluationSummary,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(&paths.summary_json)?;
    writeln!(
        file,
        "{{\"stage\":\"{}\",\"global_steps\":{},\"initial_bunny_mean_return\":{},\"best_bunny_mean_return\":{},\"initial_bunny_mean_lifetime\":{},\"best_bunny_mean_lifetime\":{},\"best_bunny_ci95_lower\":{},\"best_bunny_ci95_upper\":{},\"best_bunny_food_and_water_fraction\":{},\"best_bunny_resource_consumer_count\":{},\"best_bunny_index_lifetime_advantage_fraction\":{},\"initial_fox_mean_return\":{},\"best_fox_mean_return\":{},\"initial_fox_mean_lifetime\":{},\"best_fox_mean_lifetime\":{},\"best_fox_ci95_lower\":{},\"best_fox_ci95_upper\":{},\"best_fox_prey_and_water_fraction\":{},\"best_collision_contacts\":{},\"best_thorn_damage\":{},\"best_overconsumption_damage\":{},\"qualification\":\"fixed validation-seed demo evidence; disjoint test-seed qualification is reported separately\"}}",
        stage.as_key(),
        global_steps,
        initial.bunny_mean_return,
        best.bunny_mean_return,
        initial.bunny_mean_lifetime,
        best.bunny_mean_lifetime,
        best.bunny_lifetime_lower,
        best.bunny_lifetime_upper,
        best.bunny_food_and_water_fraction,
        best.bunny_resource_consumer_count,
        best.bunny_index_lifetime_advantage_fraction,
        initial.fox_mean_return,
        best.fox_mean_return,
        initial.fox_mean_lifetime,
        best.fox_mean_lifetime,
        best.fox_lifetime_lower,
        best.fox_lifetime_upper,
        best.fox_food_and_water_fraction,
        best.collision_contacts,
        best.thorn_damage,
        best.overconsumption_damage,
    )?;
    Ok(())
}

/// Return zero for an absent species or the arithmetic lifetime mean.
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Largest relative identity advantage above the stable population mean.
fn index_lifetime_advantage(lifetimes_by_id: &[Vec<f32>]) -> f32 {
    let means = lifetimes_by_id
        .iter()
        .filter(|values| !values.is_empty())
        .map(|values| mean(values))
        .collect::<Vec<_>>();
    let population_mean = mean(&means);
    if population_mean <= f32::EPSILON {
        return 0.0;
    }
    let maximum = means.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (maximum - population_mean) / population_mean
}

/// Initialize one parameter-sharing learner per species present.
fn create_learners(
    stage: CurriculumStage,
    config: &SimulationConfig,
    algorithm: &RecurrentPpoConfig,
    seeds: SeedConfig,
    options: &TrainOptions,
) -> Result<Learners, Box<dyn Error>> {
    let create = |seed| {
        RecurrentPpoAgent::new(
            LOCAL_OBSERVATION_SIZE,
            GLOBAL_STATE_SIZE,
            MAX_AGENTS,
            &[-1.0, -1.0, -1.0],
            &[1.0, 1.0, 1.0],
            algorithm.clone(),
            SeedConfig::from_root(seed),
        )
    };
    let load = |path: &Path, seed| -> Result<RecurrentPpoAgent, Box<dyn Error>> {
        validate_resume_checkpoint_stage(path, stage)?;
        let _tuning = checkpoint_experiment_tuning(path)?;
        Ok(RecurrentPpoAgent::load(
            path,
            LOCAL_OBSERVATION_SIZE,
            GLOBAL_STATE_SIZE,
            MAX_AGENTS,
            &[-1.0, -1.0, -1.0],
            &[1.0, 1.0, 1.0],
            algorithm.clone(),
            SeedConfig::from_root(seed),
        )?)
    };
    let bunny_resume = options.resume.as_deref().map(resolve_bunny_checkpoint);
    let bunny = if let Some(path) = &bunny_resume {
        load(path, seeds.model)?
    } else {
        create(seeds.model)?
    };
    let fox_seed = seeds.model ^ 0x666f_785f_706f_6c69;
    let fox = if config.fox_count == 0 {
        None
    } else if let Some(path) = &options.fox_resume {
        if let Some(bunny_path) = &bunny_resume {
            validate_checkpoint_tuning_match(bunny_path, path)?;
        }
        Some(load(path, fox_seed)?)
    } else if let Some(bunny_path) = &bunny_resume {
        let inferred = sibling_fox_checkpoint(bunny_path);
        if inferred.is_file() {
            validate_checkpoint_tuning_match(bunny_path, &inferred)?;
            Some(load(&inferred, fox_seed)?)
        } else {
            Some(create(fox_seed)?)
        }
    } else {
        Some(create(fox_seed)?)
    };
    Ok(Learners { bunny, fox })
}

/// Freeze one behavior policy per species for an on-policy episode.
fn collection_policies(
    learners: &Learners,
    seeds: SeedConfig,
    iteration: u64,
) -> CollectionPolicies {
    CollectionPolicies {
        bunny: learners.bunny.policy(),
        fox: learners.fox.as_ref().map(RecurrentPpoAgent::policy),
        bunny_sampler: RecurrentSampler::new(seeds.action ^ iteration.rotate_left(17)),
        fox_sampler: RecurrentSampler::new(
            seeds.action ^ 0x666f_785f_6163_746e ^ iteration.rotate_left(29),
        ),
    }
}

/// Collect independent worlds under one unchanged behavior policy snapshot.
fn collect_batch(
    learners: &Learners,
    config: SimulationConfig,
    seeds: SeedConfig,
    iteration: usize,
    rollout_episodes: usize,
    algorithm: &RecurrentPpoConfig,
) -> Result<CollectedBatch, Box<dyn Error>> {
    let mut batch = CollectedBatch::default();
    for lane in 0..rollout_episodes {
        let episode_index = iteration
            .saturating_mul(rollout_episodes)
            .saturating_add(lane);
        let seed = seeds.environment_episode(lane, episode_index as u64);
        let policies = collection_policies(learners, seeds, episode_index as u64);
        let episode = collect_episode(config.clone(), seed, episode_index, policies)?;
        batch.joint_steps = batch
            .joint_steps
            .saturating_add(u64::from(episode.joint_steps));
        batch
            .bunny_lifetimes
            .push(mean_lifetime(&episode, Species::Bunny, config.time_step));
        batch
            .bunny_returns
            .push(mean_episode_return(&episode, Species::Bunny));
        if config.fox_count > 0 {
            batch
                .fox_lifetimes
                .push(mean_lifetime(&episode, Species::Fox, config.time_step));
            batch
                .fox_returns
                .push(mean_episode_return(&episode, Species::Fox));
        }
        batch.bunny_sequences.extend(build_sequences(
            &episode.trajectories,
            Species::Bunny,
            algorithm.gamma,
            algorithm.gae_lambda,
        ));
        batch.fox_sequences.extend(build_sequences(
            &episode.trajectories,
            Species::Fox,
            algorithm.gamma,
            algorithm.gae_lambda,
        ));
    }
    Ok(batch)
}

/// Collect one simultaneous multi-agent episode under frozen policies.
fn collect_episode(
    config: SimulationConfig,
    seed: u64,
    spawn_rotation: usize,
    mut policies: CollectionPolicies,
) -> Result<CollectedEpisode, Box<dyn Error>> {
    let mut ecosystem = Ecosystem::new_rotated(config, seed, spawn_rotation)?;
    let mut current = ecosystem.state();
    let mut memories = BTreeMap::<AgentId, RecurrentMemory>::new();
    let mut episode = CollectedEpisode::default();

    loop {
        let mut actions = Vec::with_capacity(current.agents.len());
        let mut selected = BTreeMap::new();
        for (id, species, observation) in &current.agents {
            let (policy, sampler) = policy_and_sampler(&mut policies, *species)?;
            let memory = memories
                .entry(*id)
                .or_insert_with(|| policy.initial_memory())
                .clone();
            let action = policy.sample_action(observation, &memory, sampler)?;
            let value = policy.value(&current.global_state, usize::from(id.0))?;
            let [forward, turn, gaze] = action.action.as_slice() else {
                return Err("ecosystem policy must emit exactly three action axes".into());
            };
            let locomotion = LocomotionAction::new(*forward, *turn, *gaze)?;
            actions.push((*id, locomotion));
            selected.insert(*id, (*species, observation, memory, action, value));
        }

        let result = ecosystem.step(&actions)?;
        episode.joint_steps = episode.joint_steps.saturating_add(1);
        let mut next_agents = Vec::new();
        for agent_step in &result.agents {
            let Some((species, observation, initial_memory, action, value)) =
                selected.remove(&agent_step.id)
            else {
                return Err("joint step returned an agent absent from its action set".into());
            };
            let policy = policy_for_species(&policies, species)?;
            let next_value = if should_bootstrap_collected_transition(agent_step.status) {
                policy.value(&result.global_state, usize::from(agent_step.id.0))?
            } else {
                0.0
            };
            episode
                .trajectories
                .entry(agent_step.id)
                .or_insert_with(|| (species, Vec::new()))
                .1
                .push(RolloutTransition {
                    observation: *observation,
                    global_state: current.global_state,
                    pre_tanh_action: action.pre_tanh_action,
                    old_log_probability: action.log_probability,
                    reward: agent_step.reward as f32,
                    value,
                    next_value,
                    status: agent_step.status,
                    initial_memory,
                });
            if agent_step.status == EpisodeStatus::Continuing {
                memories.insert(agent_step.id, action.next_memory);
                next_agents.push((agent_step.id, species, agent_step.observation));
            } else {
                memories.remove(&agent_step.id);
            }
        }
        if result.is_done {
            break;
        }
        current.agents = next_agents;
        current.global_state = result.global_state;
    }
    Ok(episode)
}

/// Bootstrap only transitions whose episode continues after the current step.
fn should_bootstrap_collected_transition(status: EpisodeStatus) -> bool {
    status == EpisodeStatus::Continuing
}

/// Resolve the actor and mutable action stream for one biological role.
fn policy_and_sampler(
    policies: &mut CollectionPolicies,
    species: Species,
) -> Result<(&RecurrentPpoPolicy, &mut RecurrentSampler), Box<dyn Error>> {
    match species {
        Species::Bunny => Ok((&policies.bunny, &mut policies.bunny_sampler)),
        Species::Fox => policies
            .fox
            .as_ref()
            .map(|policy| (policy, &mut policies.fox_sampler))
            .ok_or_else(|| "fox agent exists without a fox policy".into()),
    }
}

/// Resolve the immutable species policy for value bootstrapping.
fn policy_for_species(
    policies: &CollectionPolicies,
    species: Species,
) -> Result<&RecurrentPpoPolicy, Box<dyn Error>> {
    match species {
        Species::Bunny => Ok(&policies.bunny),
        Species::Fox => policies
            .fox
            .as_ref()
            .ok_or_else(|| "fox agent exists without a fox policy".into()),
    }
}

/// Compute per-agent GAE, then split only at contiguous chunk boundaries.
fn build_sequences(
    trajectories: &BTreeMap<AgentId, (Species, Vec<RolloutTransition>)>,
    species: Species,
    gamma: f32,
    gae_lambda: f32,
) -> Vec<RecurrentPpoSequence> {
    trajectories
        .iter()
        .filter(|(_, (trajectory_species, _))| *trajectory_species == species)
        .flat_map(|(id, (_, transitions))| {
            let advantages = generalized_advantages(transitions, gamma, gae_lambda);
            transitions
                .chunks(RECURRENT_CHUNK_LENGTH)
                .zip(advantages.chunks(RECURRENT_CHUNK_LENGTH))
                .filter_map(move |(chunk, chunk_advantages)| {
                    let initial_memory = chunk.first()?.initial_memory.clone();
                    Some(RecurrentPpoSequence {
                        observations: chunk
                            .iter()
                            .map(|transition| transition.observation.to_vec())
                            .collect(),
                        global_states: chunk
                            .iter()
                            .map(|transition| transition.global_state.to_vec())
                            .collect(),
                        pre_tanh_actions: chunk
                            .iter()
                            .map(|transition| transition.pre_tanh_action.clone())
                            .collect(),
                        old_log_probabilities: chunk
                            .iter()
                            .map(|transition| transition.old_log_probability)
                            .collect(),
                        advantages: chunk_advantages.to_vec(),
                        returns: chunk
                            .iter()
                            .zip(chunk_advantages)
                            .map(|(transition, advantage)| transition.value + advantage)
                            .collect(),
                        value_index: usize::from(id.0),
                        initial_memory,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Compute masked GAE without leaking a reset state across an episode boundary.
fn generalized_advantages(
    transitions: &[RolloutTransition],
    gamma: f32,
    gae_lambda: f32,
) -> Vec<f32> {
    let mut reversed = Vec::with_capacity(transitions.len());
    let mut next_advantage = 0.0;
    for transition in transitions.iter().rev() {
        let bootstrap = transition.status.bootstrap_mask() as f32;
        let trace = f32::from(transition.status == EpisodeStatus::Continuing);
        let delta = (gamma * bootstrap)
            .mul_add(transition.next_value, transition.reward - transition.value);
        let advantage = (gamma * gae_lambda * trace).mul_add(next_advantage, delta);
        reversed.push(advantage);
        next_advantage = advantage;
    }
    reversed.reverse();
    reversed
}

/// Return mean per-agent lifetime for one species.
fn mean_lifetime(episode: &CollectedEpisode, species: Species, time_step: f32) -> f32 {
    let lifetimes: Vec<_> = episode
        .trajectories
        .values()
        .filter(|(trajectory_species, _)| *trajectory_species == species)
        .map(|(_, transitions)| transitions.len() as f32 * time_step)
        .collect();
    if lifetimes.is_empty() {
        0.0
    } else {
        lifetimes.iter().sum::<f32>() / lifetimes.len() as f32
    }
}

/// Return mean undiscounted per-agent episode reward for one species.
fn mean_episode_return(episode: &CollectedEpisode, species: Species) -> f32 {
    let returns = episode
        .trajectories
        .values()
        .filter(|(trajectory_species, _)| *trajectory_species == species)
        .map(|(_, transitions)| {
            transitions
                .iter()
                .map(|transition| transition.reward)
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    mean(&returns)
}

/// Return a deterministic 1,000-resample percentile interval for a mean.
fn bootstrap_mean_interval(values: &[f32], seed: u64) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = SplitMix64::new(seed);
    let mut means = Vec::with_capacity(1_000);
    for _ in 0..1_000 {
        let mut total = 0.0;
        for _ in values {
            let index = (rng.next_u64() % values.len() as u64) as usize;
            if let Some(value) = values.get(index) {
                total += value;
            }
        }
        means.push(total / values.len() as f32);
    }
    means.sort_by(f32::total_cmp);
    let fallback = mean(values);
    (
        means.get(25).copied().unwrap_or(fallback),
        means.get(974).copied().unwrap_or(fallback),
    )
}

/// Divide a success count by its population without producing NaN.
fn fraction(successes: u32, count: u32) -> f32 {
    if count == 0 {
        0.0
    } else {
        successes as f32 / count as f32
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;

    use super::*;

    /// Fixed-horizon survival must select checkpoints by earned return.
    #[test]
    fn selection_score_uses_reward_when_lifetime_is_tied() {
        let baseline = EvaluationSummary {
            bunny_mean_lifetime: 20.0,
            bunny_mean_return: 20.0,
            ..EvaluationSummary::default()
        };
        let improved = EvaluationSummary {
            bunny_mean_lifetime: 20.0,
            bunny_mean_return: 28.0,
            ..EvaluationSummary::default()
        };

        assert!(improved.selection_score(false) > baseline.selection_score(false));
    }

    /// Headless defaults must retain the same delayed-survival horizon as the demo.
    #[test]
    fn headless_default_retains_the_declared_episode_horizon() {
        let options = TrainOptions::default();

        assert_eq!(
            options.episode_seconds,
            ExperimentTuning::default().episode_seconds
        );
    }

    /// An explicit trainer horizon must override the visual HUD's initial value.
    #[test]
    fn explicit_training_horizon_overrides_initial_hud_tuning() {
        let mut options = TrainOptions::for_stage(CurriculumStage::Survival);
        options.episode_seconds = 37;
        let mut tuning = ExperimentTuning::default();
        tuning.episode_seconds = 20;

        let config = training_config(CurriculumStage::Survival, &options, Some(tuning))
            .expect("training config resolves");

        assert_eq!(config.episode_seconds, 37);
    }

    /// Survival defaults must reproduce the verified fast-learning run.
    #[test]
    fn survival_defaults_match_the_verified_learning_profile() {
        // Lock every work-budget and exploration input used by the proof run.
        let options = TrainOptions::for_stage(CurriculumStage::Survival);

        assert_eq!(options.iterations, 6);
        assert_eq!(options.rollout_episodes, 16);
        assert_eq!(options.episode_seconds, 20);
        assert_eq!(options.seed, 157);
        assert_eq!(options.eval_episodes, 16);
        assert_eq!(options.eval_interval, 1);
        assert!((options.log_std_max + 0.5).abs() < f32::EPSILON);
    }

    /// Durable tuning JSON must preserve every runtime-adjustable field.
    #[test]
    fn experiment_tuning_json_round_trips_the_complete_profile() {
        let expected = ExperimentTuning::default();
        let value = experiment_tuning_json(expected);
        let actual = parse_experiment_tuning(&value).expect("saved tuning parses");
        assert_eq!(actual, expected);
    }

    /// Run configs must identify the integer-physiology checkpoint contract.
    #[test]
    fn checkpoint_profile_rejects_prior_mechanics() {
        assert!(validate_checkpoint_profile(&serde_json::json!({
            "checkpoint_profile": CHECKPOINT_PROFILE,
        }))
        .is_ok());
        assert!(validate_checkpoint_profile(&serde_json::json!({
            "checkpoint_profile": CHECKPOINT_PROFILE - 1,
        }))
        .is_err());
        assert!(
            checkpoint_experiment_tuning(Path::new("/tmp/missing-ecosystem-profile/best.mpk"))
                .is_err()
        );
    }

    /// Synchronized policies must retain and agree on immutable checkpoint tuning.
    #[test]
    fn checkpoint_sidecars_preserve_and_validate_pair_tuning() {
        let directory = std::env::temp_dir().join(format!(
            "bevy-gym-checkpoint-profile-{}",
            std::process::id()
        ));
        if directory.exists() {
            fs::remove_dir_all(&directory).expect("stale test directory is removable");
        }
        fs::create_dir_all(&directory).expect("test directory is creatable");
        let bunny = directory.join("step-000100.mpk");
        let fox = directory.join("step-000100-fox.mpk");
        let tuning = ExperimentTuning::default();
        write_checkpoint_profile(&bunny, CurriculumStage::Survival, tuning)
            .expect("bunny profile writes");
        write_checkpoint_profile(&fox, CurriculumStage::Survival, tuning)
            .expect("fox profile writes");

        assert_eq!(
            checkpoint_config_path(&bunny),
            bunny.with_extension("config.json")
        );
        validate_checkpoint_stage(&bunny, CurriculumStage::Survival)
            .expect("matching stage validates");
        assert!(validate_checkpoint_stage(&bunny, CurriculumStage::Competition).is_err());
        validate_resume_checkpoint_stage(&bunny, CurriculumStage::Competition)
            .expect("direct predecessor can seed the next stage");
        assert!(validate_resume_checkpoint_stage(&bunny, CurriculumStage::PredatorPrey).is_err());
        validate_checkpoint_tuning_match(&bunny, &fox).expect("matching profiles validate");

        let mut changed = tuning;
        changed.episode_seconds += 1;
        write_checkpoint_profile(&fox, CurriculumStage::Survival, changed)
            .expect("changed fox profile writes");
        assert!(validate_checkpoint_tuning_match(&bunny, &fox).is_err());
        fs::remove_dir_all(directory).expect("test directory is removable");
    }

    /// Every command-line horizon must stay inside the live tuning bounds.
    #[test]
    fn training_and_evaluation_reject_out_of_range_episode_seconds() {
        for value in ["1", "301"] {
            assert!(parse_options(
                CurriculumStage::Survival,
                ["--episode-seconds", value].map(str::to_owned).into_iter(),
            )
            .is_err());
            assert!(parse_eval_options(
                ["--checkpoint", "missing.mpk", "--episode-seconds", value]
                    .map(str::to_owned)
                    .into_iter(),
            )
            .is_err());
        }
    }

    /// Ecosystem exploration must start at the scale used by qualifying runs.
    #[test]
    fn ecosystem_actor_starts_with_the_qualified_exploration_scale() {
        assert!((ecosystem_algorithm().initial_log_std + 0.5).abs() < f32::EPSILON);
    }

    /// Exploration variance caps stay inside the recurrent PPO clamp range.
    #[test]
    fn training_options_validate_log_standard_deviation_cap() {
        let valid = parse_options(
            CurriculumStage::Survival,
            ["--log-std-max", "-1.0"].map(str::to_owned).into_iter(),
        )
        .expect("bounded exploration cap parses");
        assert_eq!(valid.log_std_max, -1.0);
        assert!(parse_options(
            CurriculumStage::Survival,
            ["--log-std-max", "-5.0"].map(str::to_owned).into_iter()
        )
        .is_err());
    }

    /// Long-horizon credit parameters remain configurable within probability bounds.
    #[test]
    fn training_options_validate_discount_and_trace_parameters() {
        let defaults = parse_options(CurriculumStage::Survival, std::iter::empty())
            .expect("default long-horizon credit options validate");
        assert_eq!(defaults.gamma, 0.999);
        assert_eq!(defaults.gae_lambda, 1.0);

        let valid = parse_options(
            CurriculumStage::Survival,
            ["--gamma", "0.9995", "--gae-lambda", "1.0"]
                .map(str::to_owned)
                .into_iter(),
        )
        .expect("long-horizon credit options parse");
        assert_eq!(valid.gamma, 0.9995);
        assert_eq!(valid.gae_lambda, 1.0);
        assert!(parse_options(
            CurriculumStage::Survival,
            ["--gamma", "1.1"].map(str::to_owned).into_iter()
        )
        .is_err());
        assert!(parse_options(
            CurriculumStage::Survival,
            ["--gae-lambda", "-0.1"].map(str::to_owned).into_iter()
        )
        .is_err());
    }

    /// Identity advantage compares the strongest stable mean to the population.
    #[test]
    fn identity_lifetime_advantage_uses_population_mean() {
        let balanced = [vec![10.0, 12.0], vec![12.0, 10.0]];
        assert_eq!(index_lifetime_advantage(&balanced), 0.0);
        let biased = [vec![12.0, 12.0], vec![8.0, 8.0]];
        assert!((index_lifetime_advantage(&biased) - 0.2).abs() < f32::EPSILON);
    }

    /// Every same-species trajectory contributes a distinct optimizer sequence.
    #[test]
    fn shared_policy_sequences_include_every_agent_identity() {
        let trajectories = (0_u16..4)
            .map(|identity| {
                (
                    AgentId(identity),
                    (
                        Species::Bunny,
                        vec![test_transition(EpisodeStatus::Truncated)],
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let sequences = build_sequences(&trajectories, Species::Bunny, 0.99, 0.95);
        let identities = sequences
            .iter()
            .map(|sequence| sequence.value_index)
            .collect::<BTreeSet<_>>();
        assert_eq!(sequences.len(), 4);
        assert_eq!(identities, BTreeSet::from([0, 1, 2, 3]));
        assert!(build_sequences(&trajectories, Species::Fox, 0.99, 0.95).is_empty());
    }

    /// Cross-species diagnostics must retain each policy's valid sample weight.
    #[test]
    fn shared_update_metrics_weight_species_by_valid_samples() {
        let update = |valid_samples, value| RecurrentPpoUpdate {
            optimizer_steps: 7,
            optimizer_updates: 2,
            valid_samples,
            actor_loss: value,
            critic_loss: value,
            entropy: value,
            approximate_kl: value,
            actor_learning_rate: 3.0e-4,
            critic_learning_rate: 1.0e-3,
        };
        let summary = combine_updates(update(1, 1.0), Some(update(9, 3.0)));

        assert_eq!(summary.valid_samples, 10);
        assert_eq!(summary.optimizer_updates, 4);
        assert!((summary.actor_loss - 2.8).abs() < 1.0e-12);
        assert!((summary.critic_loss - 2.8).abs() < 1.0e-12);
        assert!((summary.entropy - 2.8).abs() < 1.0e-12);
        assert!((summary.approximate_kl - 2.8).abs() < 1.0e-12);
    }

    /// Natural termination masks both bootstrap and trace continuation.
    #[test]
    fn gae_masks_natural_termination() {
        let transition = RolloutTransition {
            observation: [0.0; LOCAL_OBSERVATION_SIZE],
            global_state: [0.0; GLOBAL_STATE_SIZE],
            pre_tanh_action: vec![0.0, 0.0, 0.0],
            old_log_probability: 0.0,
            reward: 2.0,
            value: 1.0,
            next_value: 10.0,
            status: EpisodeStatus::Terminated,
            initial_memory: RecurrentMemory::zeros(4),
        };
        assert_eq!(generalized_advantages(&[transition], 0.5, 1.0), vec![1.0]);
    }

    /// Horizon truncation bootstraps its actual final observation but stops trace carry.
    #[test]
    fn gae_bootstraps_truncation() {
        let mut transition = test_transition(EpisodeStatus::Truncated);
        transition.next_value = 4.0;
        assert_eq!(generalized_advantages(&[transition], 0.5, 1.0), vec![3.0]);
    }

    /// Collected episode ends already contain their complete terminal reward.
    #[test]
    fn collected_transitions_bootstrap_only_while_continuing() {
        assert!(should_bootstrap_collected_transition(
            EpisodeStatus::Continuing
        ));
        assert!(!should_bootstrap_collected_transition(
            EpisodeStatus::Truncated
        ));
        assert!(!should_bootstrap_collected_transition(
            EpisodeStatus::Terminated
        ));
    }

    /// Construct one finite transition for sequence and GAE tests.
    fn test_transition(status: EpisodeStatus) -> RolloutTransition {
        RolloutTransition {
            observation: [0.0; LOCAL_OBSERVATION_SIZE],
            global_state: [0.0; GLOBAL_STATE_SIZE],
            pre_tanh_action: vec![0.0, 0.0, 0.0],
            old_log_probability: 0.0,
            reward: 2.0,
            value: 1.0,
            next_value: 0.0,
            status,
            initial_memory: RecurrentMemory::zeros(4),
        }
    }
}
