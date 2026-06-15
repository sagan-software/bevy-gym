//! CartPole-v1 with a Burn-trained policy and optional realtime Bevy visualizer.
//!
//! Headless training/eval works without cargo features:
//!
//! ```text
//! cargo run --example cartpole --release -- train
//! cargo run --example cartpole --release -- eval
//! ```
//!
//! Realtime visualization and Bevy MCP/BRP inspection are feature-gated:
//!
//! ```text
//! cargo run --example cartpole --features render --release
//! cargo run --example cartpole --features bevy_remote --release -- watch
//! ```
#![allow(
    clippy::missing_docs_in_private_items,
    clippy::too_many_lines,
    unused_crate_dependencies,
    reason = "examples optimize for readability and demonstrate the full workflow in one target"
)]

use std::env;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use bevy_gym::training::backend::{InferenceDevice, TrainingDevice};
use bevy_gym::training::{
    inference_device, policy_recorder, training_device, AlgorithmKind, InferenceBackend,
    MetricRecord, MetricValue, MetricsWriter, RunConfig, RunId, RunPaths, SeedConfig,
    TrainingBackend,
};
#[cfg(feature = "render")]
use bevy_gym::{
    ActionRequest, ActionResponse, BevyGymPlugin, CurrentObservation, EnvComponent, EnvStats,
    GymSet,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CrossEntropyLossConfig, HuberLossConfig, Reduction};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, ElementConversion};
use burn::tensor::{Int, Tensor};

#[cfg(all(feature = "render", not(feature = "bevy-mcp")))]
use bevy::remote::{http::RemoteHttpPlugin, RemotePlugin};
#[cfg(feature = "render")]
use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
#[cfg(feature = "render")]
use bevy::render::{
    settings::{Backends, RenderCreation, WgpuSettings},
    RenderPlugin,
};
#[cfg(feature = "render")]
use bevy::window::{PresentMode, WindowResolution};
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras::BrpExtrasPlugin;

const OBS_SIZE: usize = 4;
const NUM_ACTIONS: usize = 2;
const MAX_STEPS_PER_EPISODE: usize = 500;

const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
const LENGTH: f32 = 0.5;
const POLE_MASS_LENGTH: f32 = MASS_POLE * LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;
const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD_RADIANS: f32 = 12.0 * std::f32::consts::PI / 180.0;

const DEFAULT_TRAIN_STEPS: usize = 100_000;
const DEFAULT_EVAL_EPISODES: usize = 12;
const DEFAULT_EVAL_INTERVAL: usize = 5_000;
const DEFAULT_ROOT_SEED: u64 = 42;
const SOLVED_MEAN_REWARD: f64 = 475.0;
const DEFAULT_VIDEO_SECONDS: usize = 30;
const DEFAULT_VIDEO_FINAL_SECONDS: usize = 10;
const DEFAULT_VIDEO_FPS: usize = 50;
const DEFAULT_VIDEO_WIDTH: usize = 600;
const DEFAULT_VIDEO_HEIGHT: usize = 400;

#[cfg(feature = "render")]
const CART_SCREEN_SCALE: f32 = 145.0;
#[cfg(feature = "render")]
const TRACK_Y: f32 = -140.0;
#[cfg(feature = "render")]
const CART_Y: f32 = TRACK_Y + 26.0;
#[cfg(feature = "render")]
const CART_W: f32 = 78.0;
#[cfg(feature = "render")]
const CART_H: f32 = 34.0;
#[cfg(feature = "render")]
const POLE_LEN: f32 = 190.0;
#[cfg(feature = "render")]
const POLE_W: f32 = 12.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CartAction {
    Left,
    Right,
}

impl CartAction {
    const fn as_index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }

    const fn from_index(index: usize) -> Self {
        if index == 0 {
            Self::Left
        } else {
            Self::Right
        }
    }
}

#[derive(Debug, Clone)]
struct CartPole {
    state: [f32; OBS_SIZE],
    steps: usize,
}

impl Default for CartPole {
    fn default() -> Self {
        Self {
            state: [0.0; OBS_SIZE],
            steps: 0,
        }
    }
}

impl Env for CartPole {
    type Observation = [f32; OBS_SIZE];
    type Action = CartAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        self.steps = 0;
        self.state = [
            jitter(seed, 0),
            jitter(seed, 1),
            jitter(seed, 2),
            jitter(seed, 3),
        ];

        Reset {
            observation: self.state,
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let [x, x_dot, theta, theta_dot] = self.state;
        let force = match action {
            CartAction::Left => -FORCE_MAG,
            CartAction::Right => FORCE_MAG,
        };

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let temp = (POLE_MASS_LENGTH * theta_dot.powi(2)).mul_add(sin_theta, force) / TOTAL_MASS;
        let theta_acc = cos_theta.mul_add(-temp, GRAVITY * sin_theta)
            / (LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta.powi(2) / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        self.state = [
            TAU.mul_add(x_dot, x),
            TAU.mul_add(x_acc, x_dot),
            TAU.mul_add(theta_dot, theta),
            TAU.mul_add(theta_acc, theta_dot),
        ];
        self.steps += 1;

        let [x, _, theta, _] = self.state;
        let status = if !(-X_THRESHOLD..=X_THRESHOLD).contains(&x)
            || !(-THETA_THRESHOLD_RADIANS..=THETA_THRESHOLD_RADIANS).contains(&theta)
        {
            EpisodeStatus::Terminated
        } else if self.steps >= MAX_STEPS_PER_EPISODE {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };

        Step {
            observation: self.state,
            reward: 1.0,
            status,
            info: (),
        }
    }
}

#[derive(Module, Debug)]
struct QNetwork<B: Backend> {
    layers: Vec<Linear<B>>,
    activation: Relu,
}

impl<B: Backend> QNetwork<B> {
    fn new(device: &B::Device) -> Self {
        let layers = [OBS_SIZE, 64, 64, NUM_ACTIONS]
            .windows(2)
            .map(|window| LinearConfig::new(window[0], window[1]).init(device))
            .collect();

        Self {
            layers,
            activation: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let last_layer = self.layers.len() - 1;
        let mut output = input;

        for (index, layer) in self.layers.iter().enumerate() {
            output = layer.forward(output);
            if index < last_layer {
                output = self.activation.forward(output);
            }
        }

        output
    }
}

#[derive(Debug, Clone)]
struct CartPoleDqnConfig {
    train_steps: usize,
    eval_interval: usize,
    eval_episodes: usize,
    batch_size: usize,
    replay_capacity: usize,
    min_replay_size: usize,
    target_update_interval: usize,
    gamma: f32,
    learning_rate: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_steps: usize,
    warmup_batches: usize,
    warmup_batch_size: usize,
}

impl Default for CartPoleDqnConfig {
    fn default() -> Self {
        Self {
            train_steps: DEFAULT_TRAIN_STEPS,
            eval_interval: DEFAULT_EVAL_INTERVAL,
            eval_episodes: DEFAULT_EVAL_EPISODES,
            batch_size: 64,
            replay_capacity: 50_000,
            min_replay_size: 1_000,
            target_update_interval: 500,
            gamma: 0.99,
            learning_rate: 3e-4,
            epsilon_start: 0.15,
            epsilon_end: 0.01,
            epsilon_decay_steps: 10_000,
            warmup_batches: 0,
            warmup_batch_size: 128,
        }
    }
}

#[derive(Debug, Clone)]
struct Experience {
    observation: [f32; OBS_SIZE],
    action_index: usize,
    reward: f64,
    next_observation: [f32; OBS_SIZE],
    status: EpisodeStatus,
}

#[derive(Debug)]
struct ReplayBuffer {
    data: Vec<Experience>,
    capacity: usize,
    next_index: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            capacity,
            next_index: 0,
        }
    }

    fn push(&mut self, experience: Experience) {
        if self.data.len() < self.capacity {
            self.data.push(experience);
        } else {
            self.data[self.next_index] = experience;
            self.next_index = (self.next_index + 1) % self.capacity;
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn sample(&self, batch_size: usize, rng: &mut SplitMix64) -> Vec<Experience> {
        (0..batch_size)
            .map(|_| {
                let index = rng.usize_below(self.data.len());
                self.data[index].clone()
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }

    fn f32_between(&mut self, low: f32, high: f32) -> f32 {
        let unit = self.f64() as f32;
        low + unit * (high - low)
    }

    fn usize_below(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

struct DqnAgent {
    online_net: QNetwork<TrainingBackend>,
    target_net: QNetwork<InferenceBackend>,
    optimizer: OptimizerAdaptor<Adam, QNetwork<TrainingBackend>, TrainingBackend>,
    replay: ReplayBuffer,
    device: TrainingDevice,
    config: CartPoleDqnConfig,
    action_rng: SplitMix64,
    replay_rng: SplitMix64,
    warmup_rng: SplitMix64,
    total_steps: usize,
    last_loss: Option<f64>,
    last_warmup_loss: Option<f64>,
}

impl DqnAgent {
    fn new(config: CartPoleDqnConfig, seeds: SeedConfig) -> Self {
        let device = training_device();
        let online_net = QNetwork::<TrainingBackend>::new(&device);
        let target_net = online_net.valid();
        let optimizer = AdamConfig::new()
            .with_epsilon(1e-8)
            .init::<TrainingBackend, QNetwork<TrainingBackend>>();

        Self {
            online_net,
            target_net,
            optimizer,
            replay: ReplayBuffer::new(config.replay_capacity),
            device,
            config,
            action_rng: SplitMix64::new(seeds.action),
            replay_rng: SplitMix64::new(seeds.replay),
            warmup_rng: SplitMix64::new(seeds.model),
            total_steps: 0,
            last_loss: None,
            last_warmup_loss: None,
        }
    }

    fn epsilon(&self) -> f64 {
        let progress =
            (self.total_steps as f64 / self.config.epsilon_decay_steps as f64).clamp(0.0, 1.0);
        self.config
            .epsilon_start
            .mul_add(1.0 - progress, self.config.epsilon_end * progress)
    }

    fn act_explore(&mut self, observation: &[f32; OBS_SIZE]) -> CartAction {
        if self.action_rng.f64() < self.epsilon() {
            return CartAction::from_index(self.action_rng.usize_below(NUM_ACTIONS));
        }

        greedy_action(&self.online_net.valid(), observation, &inference_device())
    }

    fn observe(&mut self, experience: Experience) -> Result<(), Box<dyn Error>> {
        self.replay.push(experience);
        self.total_steps += 1;

        if self.total_steps % self.config.target_update_interval == 0 {
            self.target_net = self.online_net.valid();
        }

        if self.replay.len() >= self.config.min_replay_size {
            self.last_loss = Some(self.train_step()?);
        }

        Ok(())
    }

    fn behavior_clone_warmup(&mut self) -> Result<(), Box<dyn Error>> {
        let loss = CrossEntropyLossConfig::new().init(&self.device);

        for _ in 0..self.config.warmup_batches {
            let mut observations = Vec::with_capacity(self.config.warmup_batch_size);
            let mut labels = Vec::with_capacity(self.config.warmup_batch_size);

            for _ in 0..self.config.warmup_batch_size {
                let observation = [
                    self.warmup_rng.f32_between(-X_THRESHOLD, X_THRESHOLD),
                    self.warmup_rng.f32_between(-2.0, 2.0),
                    self.warmup_rng
                        .f32_between(-THETA_THRESHOLD_RADIANS, THETA_THRESHOLD_RADIANS),
                    self.warmup_rng.f32_between(-2.0, 2.0),
                ];
                observations.push(observation);
                labels.push(heuristic_action(&observation).as_index() as i32);
            }

            let logits = self
                .online_net
                .forward(encode_batch_train(&observations, &self.device));
            let targets =
                Tensor::<TrainingBackend, 1, Int>::from_ints(labels.as_slice(), &self.device);
            let batch_loss = loss.forward(logits, targets);

            let loss_value = batch_loss.clone().into_scalar().elem::<f64>();
            let grads = batch_loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.online_net);
            self.online_net =
                self.optimizer
                    .step(self.config.learning_rate, self.online_net.clone(), grads);
            self.last_warmup_loss = Some(loss_value);
        }

        self.target_net = self.online_net.valid();
        Ok(())
    }

    fn train_step(&mut self) -> Result<f64, Box<dyn Error>> {
        let batch = self
            .replay
            .sample(self.config.batch_size, &mut self.replay_rng);
        let batch_size = batch.len();

        let observations: Vec<_> = batch.iter().map(|item| item.observation).collect();
        let next_observations: Vec<_> = batch.iter().map(|item| item.next_observation).collect();
        let rewards: Vec<f32> = batch.iter().map(|item| item.reward as f32).collect();
        let masks: Vec<f32> = batch
            .iter()
            .map(|item| item.status.bootstrap_mask() as f32)
            .collect();
        let action_indices: Vec<i32> = batch.iter().map(|item| item.action_index as i32).collect();

        let rewards_t = Tensor::<TrainingBackend, 1>::from_floats(rewards.as_slice(), &self.device);
        let masks_t = Tensor::<TrainingBackend, 1>::from_floats(masks.as_slice(), &self.device);

        let next_q_values = self
            .target_net
            .forward(encode_batch_infer(&next_observations, &inference_device()));
        let max_next_q = next_q_values.max_dim(1).squeeze::<1>();
        let max_next_q = Tensor::<TrainingBackend, 1>::from_inner(max_next_q);
        let targets = rewards_t + masks_t * max_next_q * self.config.gamma;

        let q_values = self
            .online_net
            .forward(encode_batch_train(&observations, &self.device));
        let action_indices_t =
            Tensor::<TrainingBackend, 1, Int>::from_ints(action_indices.as_slice(), &self.device);
        let q_taken = q_values
            .gather(1, action_indices_t.reshape([batch_size, 1]))
            .squeeze::<1>();

        let loss =
            HuberLossConfig::new(1.0)
                .init()
                .forward(q_taken, targets.detach(), Reduction::Mean);
        let loss_value = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &self.online_net);
        self.online_net =
            self.optimizer
                .step(self.config.learning_rate, self.online_net.clone(), grads);

        Ok(loss_value)
    }

    fn policy(&self) -> QNetwork<InferenceBackend> {
        self.online_net.valid()
    }
}

#[derive(Debug, Clone)]
struct EvalSummary {
    episodes: usize,
    mean_reward: f64,
    mean_length: f64,
    min_reward: f64,
    max_reward: f64,
}

#[derive(Debug, Clone)]
struct TrainReport {
    paths: RunPaths,
    initial: EvalSummary,
    best: EvalSummary,
    global_steps: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Train,
    Eval,
    Video,
    Watch,
    TrainWatch,
}

#[derive(Debug, Clone)]
struct Args {
    mode: Mode,
    config: CartPoleDqnConfig,
    runs_root: PathBuf,
    run_id: Option<String>,
    checkpoint: Option<PathBuf>,
    seed: u64,
    screenshot: Option<PathBuf>,
    screenshot_frames: u32,
    video_output: Option<PathBuf>,
    video_seconds: usize,
    video_final_seconds: usize,
}

impl Args {
    fn parse() -> Result<Self, CliError> {
        let mut raw = env::args().skip(1);
        let mut mode = None;
        let mut config = CartPoleDqnConfig::default();
        let mut runs_root = PathBuf::from("runs");
        let mut run_id = None;
        let mut checkpoint = None;
        let mut seed = DEFAULT_ROOT_SEED;
        let mut screenshot = None;
        let mut screenshot_frames = 90;
        let mut video_output = None;
        let mut video_seconds = DEFAULT_VIDEO_SECONDS;
        let mut video_final_seconds = DEFAULT_VIDEO_FINAL_SECONDS;

        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "train" => mode = Some(Mode::Train),
                "eval" => mode = Some(Mode::Eval),
                "video" => mode = Some(Mode::Video),
                "watch" => mode = Some(Mode::Watch),
                "train-watch" => mode = Some(Mode::TrainWatch),
                "--steps" => config.train_steps = parse_next(&mut raw, "--steps")?,
                "--eval-episodes" => {
                    config.eval_episodes = parse_next(&mut raw, "--eval-episodes")?;
                }
                "--eval-interval" => {
                    config.eval_interval = parse_next(&mut raw, "--eval-interval")?;
                }
                "--warmup-batches" => {
                    config.warmup_batches = parse_next(&mut raw, "--warmup-batches")?;
                }
                "--run-id" => run_id = Some(parse_next::<String>(&mut raw, "--run-id")?),
                "--runs-root" => {
                    runs_root = PathBuf::from(parse_next::<String>(&mut raw, "--runs-root")?)
                }
                "--checkpoint" => {
                    checkpoint = Some(PathBuf::from(parse_next::<String>(
                        &mut raw,
                        "--checkpoint",
                    )?))
                }
                "--seed" => seed = parse_next(&mut raw, "--seed")?,
                "--screenshot" => {
                    screenshot = Some(PathBuf::from(parse_next::<String>(
                        &mut raw,
                        "--screenshot",
                    )?))
                }
                "--screenshot-frames" => {
                    screenshot_frames = parse_next(&mut raw, "--screenshot-frames")?;
                }
                "--output" | "--video-output" => {
                    video_output = Some(PathBuf::from(parse_next::<String>(&mut raw, "--output")?))
                }
                "--video-seconds" => video_seconds = parse_next(&mut raw, "--video-seconds")?,
                "--final-seconds" => {
                    video_final_seconds = parse_next(&mut raw, "--final-seconds")?;
                }
                "--smoke" => {
                    config.train_steps = 512;
                    config.eval_interval = 256;
                    config.eval_episodes = 2;
                    config.warmup_batches = 4;
                    config.min_replay_size = 64;
                    config.batch_size = 32;
                }
                "--help" | "-h" => return Err(CliError::Help(help_text())),
                other => return Err(CliError::Invalid(format!("unknown argument `{other}`"))),
            }
        }

        let default_mode = if cfg!(feature = "render") {
            Mode::Watch
        } else {
            Mode::Train
        };

        Ok(Self {
            mode: mode.unwrap_or(default_mode),
            config,
            runs_root,
            run_id,
            checkpoint,
            seed,
            screenshot,
            screenshot_frames,
            video_output,
            video_seconds,
            video_final_seconds,
        })
    }
}

#[derive(Debug)]
enum CliError {
    Help(&'static str),
    Invalid(String),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Help(text) => formatter.write_str(text),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl Error for CliError {}

fn main() -> Result<(), Box<dyn Error>> {
    let args = match Args::parse() {
        Ok(args) => args,
        Err(CliError::Help(text)) => {
            println!("{text}");
            return Ok(());
        }
        Err(error) => return Err(Box::new(error)),
    };

    match args.mode {
        Mode::Train => {
            let report = train(args)?;
            print_train_report(&report);
        }
        Mode::Eval => {
            let checkpoint = resolve_checkpoint(args.checkpoint.as_deref(), &args.runs_root)?;
            let policy = load_policy(&checkpoint)?;
            let summary = eval_policy(&policy, args.config.eval_episodes, args.seed);
            println!(
                "eval checkpoint={} episodes={} mean_reward={:.1} mean_length={:.1} min={:.1} max={:.1}",
                checkpoint.display(),
                summary.episodes,
                summary.mean_reward,
                summary.mean_length,
                summary.min_reward,
                summary.max_reward
            );
        }
        Mode::Video => {
            let run_dir = resolve_run_dir(args.checkpoint.as_deref(), &args.runs_root)?;
            let output = args
                .video_output
                .clone()
                .unwrap_or_else(|| run_dir.join("cartpole-training-timelapse.mp4"));
            render_training_video(
                &run_dir,
                &output,
                VideoConfig {
                    total_seconds: args.video_seconds,
                    final_seconds: args.video_final_seconds,
                    fps: DEFAULT_VIDEO_FPS,
                    width: DEFAULT_VIDEO_WIDTH,
                    height: DEFAULT_VIDEO_HEIGHT,
                    seed: args.seed,
                },
            )?;
            println!("video={}", output.display());
        }
        Mode::Watch => {
            let checkpoint = resolve_or_train_checkpoint(&args)?;
            run_visual(
                &checkpoint,
                args.screenshot.as_deref(),
                args.screenshot_frames,
            )?;
        }
        Mode::TrainWatch => {
            let report = train(args.clone())?;
            print_train_report(&report);
            run_visual(
                &report.paths.best_checkpoint,
                args.screenshot.as_deref(),
                args.screenshot_frames,
            )?;
        }
    }

    Ok(())
}

fn train(args: Args) -> Result<TrainReport, Box<dyn Error>> {
    let seeds = SeedConfig::from_root(args.seed);
    let run_id = match args.run_id {
        Some(run_id) => RunId::new(run_id)?,
        None => RunId::new(default_run_id()?)?,
    };
    let run = RunConfig::new("cartpole", AlgorithmKind::Dqn, run_id, args.runs_root)?;
    let paths = run.paths();
    fs::create_dir_all(&paths.checkpoints_dir)?;
    write_train_config(&paths, &args.config, seeds)?;

    let mut agent = DqnAgent::new(args.config.clone(), seeds);
    agent.behavior_clone_warmup()?;

    let initial = eval_policy(&agent.policy(), args.config.eval_episodes, seeds.eval);
    let mut best = initial.clone();
    save_policy(&agent.policy(), &paths.best_checkpoint)?;
    save_policy(&agent.policy(), &paths.latest_checkpoint)?;
    save_policy(&agent.policy(), &step_checkpoint_path(&paths, 0))?;
    append_eval_record(&paths, 0, &initial)?;

    let mut metrics = MetricsWriter::append(&paths.metrics_jsonl)?;
    metrics.write_record(
        &MetricRecord::new(0)
            .with_field("eval/mean_reward", MetricValue::Number(initial.mean_reward))
            .with_field("eval/mean_length", MetricValue::Number(initial.mean_length))
            .with_field(
                "warmup/batches",
                MetricValue::Number(args.config.warmup_batches as f64),
            ),
    )?;
    let mut env = CartPole::default();
    let mut observation = env.reset(Some(seeds.env_reset)).observation;
    let mut episode_reward = 0.0;
    let mut episode_steps = 0usize;
    let mut episodes = 0usize;

    for step in 1..=args.config.train_steps {
        let action = agent.act_explore(&observation);
        let result = env.step(action);
        let experience = Experience {
            observation,
            action_index: action.as_index(),
            reward: result.reward,
            next_observation: result.observation,
            status: result.status,
        };
        agent.observe(experience)?;

        episode_reward += result.reward;
        episode_steps += 1;

        if result.status.is_done() {
            episodes += 1;
            if episodes % 10 == 0 {
                let mut record = MetricRecord::new(step as u64)
                    .with_field("train/episode", MetricValue::Number(episodes as f64))
                    .with_field("train/reward", MetricValue::Number(episode_reward))
                    .with_field("train/length", MetricValue::Number(episode_steps as f64))
                    .with_field("train/epsilon", MetricValue::Number(agent.epsilon()));
                if let Some(loss) = agent.last_loss {
                    record = record.with_field("train/loss", MetricValue::Number(loss));
                }
                if let Some(loss) = agent.last_warmup_loss {
                    record = record.with_field("warmup/loss", MetricValue::Number(loss));
                }
                metrics.write_record(&record)?;
                println!(
                    "step {step:>6} episode {episodes:>4} reward {episode_reward:>6.1} length {episode_steps:>3} epsilon {:.3}",
                    agent.epsilon()
                );
            }

            observation = env
                .reset(Some(seeds.env_reset.wrapping_add(episodes as u64)))
                .observation;
            episode_reward = 0.0;
            episode_steps = 0;
        } else {
            observation = result.observation;
        }

        if step % args.config.eval_interval == 0 || step == args.config.train_steps {
            let policy = agent.policy();
            let summary = eval_policy(
                &policy,
                args.config.eval_episodes,
                seeds.eval.wrapping_add(step as u64),
            );
            append_eval_record(&paths, step, &summary)?;
            save_policy(&policy, &paths.latest_checkpoint)?;
            save_policy(&policy, &step_checkpoint_path(&paths, step))?;

            let is_best = summary.mean_reward > best.mean_reward;
            if is_best {
                best = summary.clone();
                save_policy(&policy, &paths.best_checkpoint)?;
            }

            metrics.write_record(
                &MetricRecord::new(step as u64)
                    .with_field("eval/mean_reward", MetricValue::Number(summary.mean_reward))
                    .with_field("eval/mean_length", MetricValue::Number(summary.mean_length))
                    .with_field("checkpoint/best", MetricValue::Bool(is_best)),
            )?;
            println!(
                "eval step {step:>6} mean_reward {:>6.1} mean_length {:>5.1} best {:>6.1}",
                summary.mean_reward, summary.mean_length, best.mean_reward
            );

            if best.mean_reward >= SOLVED_MEAN_REWARD {
                println!(
                    "solved threshold reached: best_mean_reward={:.1} >= {:.1}",
                    best.mean_reward, SOLVED_MEAN_REWARD
                );
                write_summary(&paths, step, &initial, &best, &args.config)?;
                return Ok(TrainReport {
                    paths,
                    initial,
                    best,
                    global_steps: step,
                });
            }
        }
    }

    write_summary(
        &paths,
        args.config.train_steps,
        &initial,
        &best,
        &args.config,
    )?;
    Ok(TrainReport {
        paths,
        initial,
        best,
        global_steps: args.config.train_steps,
    })
}

fn resolve_or_train_checkpoint(args: &Args) -> Result<PathBuf, Box<dyn Error>> {
    match resolve_checkpoint(args.checkpoint.as_deref(), &args.runs_root) {
        Ok(checkpoint) => Ok(checkpoint),
        Err(_) => {
            let mut train_args = args.clone();
            train_args.mode = Mode::Train;
            let report = train(train_args)?;
            Ok(report.paths.best_checkpoint)
        }
    }
}

fn resolve_checkpoint(
    explicit: Option<&Path>,
    runs_root: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
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

    find_latest_best_checkpoint(runs_root)
        .ok_or_else(|| format!("no best.mpk found under {}", runs_root.display()).into())
}

fn find_latest_best_checkpoint(runs_root: &Path) -> Option<PathBuf> {
    let root = runs_root.join("cartpole-dqn");
    let entries = fs::read_dir(root).ok()?;
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path().join("best.mpk");
            let modified = fs::metadata(&path).ok()?.modified().ok()?;
            Some((modified, path))
        })
        .max_by_key(|(modified, _)| *modified)
        .map(|(_, path)| path)
}

fn resolve_run_dir(explicit: Option<&Path>, runs_root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(path) = explicit {
        if path.is_dir() {
            if path.join("best.mpk").exists() {
                return Ok(path.to_path_buf());
            }
            return Err(format!("run directory missing best.mpk: {}", path.display()).into());
        }

        if path.exists() {
            let parent = path
                .parent()
                .ok_or_else(|| format!("checkpoint has no parent: {}", path.display()))?;
            if parent.file_name().and_then(|name| name.to_str()) == Some("checkpoints") {
                return parent.parent().map(Path::to_path_buf).ok_or_else(|| {
                    format!("checkpoint has no run directory: {}", path.display()).into()
                });
            }
            return Ok(parent.to_path_buf());
        }

        return Err(format!("checkpoint or run directory not found: {}", path.display()).into());
    }

    let best = find_latest_best_checkpoint(runs_root)
        .ok_or_else(|| format!("no best.mpk found under {}", runs_root.display()))?;
    best.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| format!("best checkpoint has no parent: {}", best.display()).into())
}

fn step_checkpoint_path(paths: &RunPaths, global_step: usize) -> PathBuf {
    paths
        .checkpoints_dir
        .join(format!("step-{global_step:06}.mpk"))
}

#[derive(Debug, Clone)]
struct VideoConfig {
    total_seconds: usize,
    final_seconds: usize,
    fps: usize,
    width: usize,
    height: usize,
    seed: u64,
}

#[derive(Debug, Clone)]
struct VideoCheckpoint {
    global_step: usize,
    mean_reward: f64,
    path: PathBuf,
}

fn render_training_video(
    run_dir: &Path,
    output: &Path,
    config: VideoConfig,
) -> Result<(), Box<dyn Error>> {
    if config.final_seconds >= config.total_seconds {
        return Err("--final-seconds must be smaller than --video-seconds".into());
    }

    let checkpoints = video_checkpoints(run_dir)?;
    if checkpoints.is_empty() {
        return Err(format!(
            "no step checkpoints found in {}; rerun `train` with the current example first",
            run_dir.display()
        )
        .into());
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let best_checkpoint = run_dir.join("best.mpk");
    if !best_checkpoint.exists() {
        return Err(format!("missing best checkpoint: {}", best_checkpoint.display()).into());
    }

    let timelapse_frames = (config.total_seconds - config.final_seconds) * config.fps;
    let final_frames = config.final_seconds * config.fps;
    let selected = sampled_checkpoints(&checkpoints, timelapse_frames);
    let final_mean = checkpoints
        .iter()
        .max_by(|left, right| left.mean_reward.total_cmp(&right.mean_reward))
        .map_or(0.0, |checkpoint| checkpoint.mean_reward);

    let fps = config.fps.to_string();
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "image2pipe",
            "-vcodec",
            "ppm",
            "-framerate",
            &fps,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(output)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("failed to start ffmpeg: {error}"))?;

    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or("failed to open ffmpeg stdin for frame pipe")?;
        let mut frame_index = 0usize;
        for (segment_index, checkpoint) in selected.iter().enumerate() {
            let segment_frames =
                distributed_frames(timelapse_frames, selected.len(), segment_index);
            let policy = load_policy(&checkpoint.path)?;
            let label = format!(
                "STEP {:>6}  MEAN {:>5.1}",
                checkpoint.global_step, checkpoint.mean_reward
            );
            write_policy_video_segment(
                &mut stdin,
                &policy,
                segment_frames,
                &config,
                config.seed.wrapping_add(checkpoint.global_step as u64),
                &label,
                &mut frame_index,
                timelapse_frames,
            )?;
        }

        let final_policy = load_policy(&best_checkpoint)?;
        let label = format!("FINAL BEST  MEAN {:>5.1}  REAL TIME", final_mean);
        write_policy_video_segment(
            &mut stdin,
            &final_policy,
            final_frames,
            &config,
            config.seed.wrapping_add(0xfeed_babe),
            &label,
            &mut frame_index,
            timelapse_frames,
        )?;
    }

    let output_result = child.wait_with_output()?;
    if !output_result.status.success() {
        return Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output_result.stderr)
        )
        .into());
    }

    Ok(())
}

fn video_checkpoints(run_dir: &Path) -> Result<Vec<VideoCheckpoint>, Box<dyn Error>> {
    let eval_records = read_eval_records(&run_dir.join("eval.jsonl"))?;
    let checkpoints_dir = run_dir.join("checkpoints");
    let mut checkpoints = Vec::new();

    for record in eval_records {
        let path = checkpoints_dir.join(format!("step-{:06}.mpk", record.global_step));
        if path.exists() {
            checkpoints.push(VideoCheckpoint {
                global_step: record.global_step,
                mean_reward: record.mean_reward,
                path,
            });
        }
    }

    checkpoints.sort_by_key(|checkpoint| checkpoint.global_step);
    Ok(checkpoints)
}

fn sampled_checkpoints(
    checkpoints: &[VideoCheckpoint],
    frame_budget: usize,
) -> Vec<VideoCheckpoint> {
    if checkpoints.len() <= frame_budget {
        return checkpoints.to_vec();
    }

    (0..frame_budget)
        .map(|index| {
            let source_index = index * (checkpoints.len() - 1) / (frame_budget - 1);
            checkpoints[source_index].clone()
        })
        .collect()
}

fn distributed_frames(total_frames: usize, segments: usize, index: usize) -> usize {
    let base = total_frames / segments;
    let extra = usize::from(index < total_frames % segments);
    base + extra
}

#[derive(Debug, Clone, Copy)]
struct EvalRecord {
    global_step: usize,
    mean_reward: f64,
}

fn read_eval_records(path: &Path) -> Result<Vec<EvalRecord>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut records = Vec::new();

    for line in content.lines() {
        let Some(global_step) = json_usize_field(line, "global_step") else {
            continue;
        };
        let Some(mean_reward) = json_f64_field(line, "mean_reward") else {
            continue;
        };
        records.push(EvalRecord {
            global_step,
            mean_reward,
        });
    }

    Ok(records)
}

fn json_usize_field(line: &str, key: &str) -> Option<usize> {
    json_number_field(line, key)?.parse().ok()
}

fn json_f64_field(line: &str, key: &str) -> Option<f64> {
    json_number_field(line, key)?.parse().ok()
}

fn json_number_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let start = line.find(&needle)? + needle.len();
    let rest = line[start..].trim_start();
    let end = rest
        .find(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E')))
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

fn write_policy_video_segment(
    writer: &mut impl Write,
    policy: &QNetwork<InferenceBackend>,
    frames: usize,
    config: &VideoConfig,
    seed: u64,
    label: &str,
    frame_index: &mut usize,
    timelapse_frames: usize,
) -> Result<(), Box<dyn Error>> {
    let mut env = CartPole::default();
    let mut episode = 0u64;
    let mut observation = env.reset(Some(seed)).observation;
    let device = inference_device();

    for _ in 0..frames {
        let progress = if timelapse_frames == 0 {
            1.0
        } else {
            (*frame_index).min(timelapse_frames) as f32 / timelapse_frames as f32
        };
        let mut frame = RgbFrame::new(config.width, config.height, [255, 255, 255]);
        draw_cartpole_frame(&mut frame, &observation, label, progress);
        frame.write_ppm(writer)?;

        let action = greedy_action(policy, &observation, &device);
        let result = env.step(action);
        if result.status.is_done() {
            episode += 1;
            observation = env.reset(Some(seed.wrapping_add(episode))).observation;
        } else {
            observation = result.observation;
        }
        *frame_index += 1;
    }

    Ok(())
}

#[derive(Debug)]
struct RgbFrame {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl RgbFrame {
    fn new(width: usize, height: usize, color: [u8; 3]) -> Self {
        let mut pixels = vec![0; width * height * 3];
        for chunk in pixels.chunks_exact_mut(3) {
            chunk.copy_from_slice(&color);
        }

        Self {
            width,
            height,
            pixels,
        }
    }

    fn write_ppm(&self, writer: &mut impl Write) -> Result<(), Box<dyn Error>> {
        write!(writer, "P6\n{} {}\n255\n", self.width, self.height)?;
        writer.write_all(&self.pixels)?;
        Ok(())
    }

    fn set_pixel(&mut self, x: i32, y: i32, color: [u8; 3]) {
        if x < 0 || y < 0 {
            return;
        }
        let x = x as usize;
        let y = y as usize;
        if x >= self.width || y >= self.height {
            return;
        }

        let index = (y * self.width + x) * 3;
        self.pixels[index..index + 3].copy_from_slice(&color);
    }

    fn fill_rect(&mut self, x: i32, y: i32, width: i32, height: i32, color: [u8; 3]) {
        for yy in y.max(0)..(y + height).min(self.height as i32) {
            for xx in x.max(0)..(x + width).min(self.width as i32) {
                self.set_pixel(xx, yy, color);
            }
        }
    }

    fn fill_circle(&mut self, cx: i32, cy: i32, radius: i32, color: [u8; 3]) {
        let radius_sq = radius * radius;
        for y in cy - radius..=cy + radius {
            for x in cx - radius..=cx + radius {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= radius_sq {
                    self.set_pixel(x, y, color);
                }
            }
        }
    }

    fn draw_thick_line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, radius: i32, color: [u8; 3]) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as i32;

        for step in 0..=steps {
            let t = step as f32 / steps as f32;
            let x = dx.mul_add(t, x0).round() as i32;
            let y = dy.mul_add(t, y0).round() as i32;
            self.fill_circle(x, y, radius, color);
        }
    }
}

fn draw_cartpole_frame(
    frame: &mut RgbFrame,
    observation: &[f32; OBS_SIZE],
    label: &str,
    progress: f32,
) {
    let black = [18, 18, 18];
    let gray = [110, 110, 110];
    let blue = [47, 111, 184];
    let width = frame.width as f32;
    let height = frame.height as f32;
    let [x, _, theta, _] = *observation;

    let track_y = (height * 0.79).round() as i32;
    let cart_w = (width * 0.13).round() as i32;
    let cart_h = (height * 0.075).round() as i32;
    let wheel_r = (height * 0.018).round() as i32;
    let pole_len = height * 0.42;
    let pole_radius = (width * 0.008).round().max(4.0) as i32;
    let world_scale = width * 0.42 / X_THRESHOLD;
    let cart_x = width * 0.5 + x * world_scale;
    let cart_y = track_y as f32 - cart_h as f32 * 0.5 - 2.0;
    let hinge_y = cart_y - cart_h as f32 * 0.5;
    let hinge_x = cart_x;
    let pole_tip_x = hinge_x + theta.sin() * pole_len;
    let pole_tip_y = hinge_y - theta.cos() * pole_len;

    frame.fill_rect(22, track_y, frame.width as i32 - 44, 5, black);
    for marker_x in [
        width * 0.5 - X_THRESHOLD * world_scale,
        width * 0.5 + X_THRESHOLD * world_scale,
    ] {
        frame.fill_rect(marker_x.round() as i32 - 2, track_y - 32, 4, 32, gray);
    }

    frame.draw_thick_line(hinge_x, hinge_y, pole_tip_x, pole_tip_y, pole_radius, black);
    frame.fill_rect(
        cart_x.round() as i32 - cart_w / 2,
        cart_y.round() as i32 - cart_h / 2,
        cart_w,
        cart_h,
        black,
    );
    frame.fill_circle(
        cart_x.round() as i32 - cart_w / 4,
        track_y - wheel_r,
        wheel_r,
        [0, 0, 0],
    );
    frame.fill_circle(
        cart_x.round() as i32 + cart_w / 4,
        track_y - wheel_r,
        wheel_r,
        [0, 0, 0],
    );
    frame.fill_circle(
        hinge_x.round() as i32,
        hinge_y.round() as i32,
        wheel_r + 1,
        [0, 0, 0],
    );

    draw_text(frame, 14, 16, 2, label, black);
    frame.fill_rect(14, 42, frame.width as i32 - 28, 5, [220, 220, 220]);
    frame.fill_rect(
        14,
        42,
        ((frame.width as f32 - 28.0) * progress.clamp(0.0, 1.0)).round() as i32,
        5,
        blue,
    );
}

fn draw_text(frame: &mut RgbFrame, x: i32, y: i32, scale: i32, text: &str, color: [u8; 3]) {
    let mut cursor = x;
    for ch in text.chars() {
        let glyph = glyph_rows(ch);
        for (row, pattern) in glyph.iter().enumerate() {
            for (col, pixel) in pattern.as_bytes().iter().enumerate() {
                if *pixel == b'X' {
                    frame.fill_rect(
                        cursor + col as i32 * scale,
                        y + row as i32 * scale,
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
        cursor += 6 * scale;
    }
}

fn glyph_rows(ch: char) -> [&'static str; 7] {
    match ch {
        'A' => [
            " XXX ", "X   X", "X   X", "XXXXX", "X   X", "X   X", "X   X",
        ],
        'B' => [
            "XXXX ", "X   X", "X   X", "XXXX ", "X   X", "X   X", "XXXX ",
        ],
        'D' => [
            "XXXX ", "X   X", "X   X", "X   X", "X   X", "X   X", "XXXX ",
        ],
        'E' => [
            "XXXXX", "X    ", "X    ", "XXXX ", "X    ", "X    ", "XXXXX",
        ],
        'F' => [
            "XXXXX", "X    ", "X    ", "XXXX ", "X    ", "X    ", "X    ",
        ],
        'I' => [
            "XXXXX", "  X  ", "  X  ", "  X  ", "  X  ", "  X  ", "XXXXX",
        ],
        'L' => [
            "X    ", "X    ", "X    ", "X    ", "X    ", "X    ", "XXXXX",
        ],
        'M' => [
            "X   X", "XX XX", "X X X", "X   X", "X   X", "X   X", "X   X",
        ],
        'N' => [
            "X   X", "XX  X", "XX  X", "X X X", "X  XX", "X  XX", "X   X",
        ],
        'P' => [
            "XXXX ", "X   X", "X   X", "XXXX ", "X    ", "X    ", "X    ",
        ],
        'R' => [
            "XXXX ", "X   X", "X   X", "XXXX ", "X X  ", "X  X ", "X   X",
        ],
        'S' => [
            " XXXX", "X    ", "X    ", " XXX ", "    X", "    X", "XXXX ",
        ],
        'T' => [
            "XXXXX", "  X  ", "  X  ", "  X  ", "  X  ", "  X  ", "  X  ",
        ],
        '0' => [
            " XXX ", "X   X", "X  XX", "X X X", "XX  X", "X   X", " XXX ",
        ],
        '1' => [
            "  X  ", " XX  ", "  X  ", "  X  ", "  X  ", "  X  ", " XXX ",
        ],
        '2' => [
            " XXX ", "X   X", "    X", "   X ", "  X  ", " X   ", "XXXXX",
        ],
        '3' => [
            " XXX ", "X   X", "    X", "  XX ", "    X", "X   X", " XXX ",
        ],
        '4' => [
            "   X ", "  XX ", " X X ", "X  X ", "XXXXX", "   X ", "   X ",
        ],
        '5' => [
            "XXXXX", "X    ", "X    ", "XXXX ", "    X", "X   X", " XXX ",
        ],
        '6' => [
            " XXX ", "X   X", "X    ", "XXXX ", "X   X", "X   X", " XXX ",
        ],
        '7' => [
            "XXXXX", "    X", "   X ", "  X  ", " X   ", " X   ", " X   ",
        ],
        '8' => [
            " XXX ", "X   X", "X   X", " XXX ", "X   X", "X   X", " XXX ",
        ],
        '9' => [
            " XXX ", "X   X", "X   X", " XXXX", "    X", "X   X", " XXX ",
        ],
        '.' => [
            "     ", "     ", "     ", "     ", "     ", " XX  ", " XX  ",
        ],
        '-' => [
            "     ", "     ", "     ", " XXX ", "     ", "     ", "     ",
        ],
        ':' => [
            "     ", " XX  ", " XX  ", "     ", " XX  ", " XX  ", "     ",
        ],
        '/' => [
            "    X", "   X ", "   X ", "  X  ", " X   ", " X   ", "X    ",
        ],
        ' ' => [
            "     ", "     ", "     ", "     ", "     ", "     ", "     ",
        ],
        _ => [
            "XXXXX", "    X", "   X ", "  X  ", "     ", "  X  ", "     ",
        ],
    }
}

fn eval_policy(policy: &QNetwork<InferenceBackend>, episodes: usize, seed: u64) -> EvalSummary {
    let mut total_reward = 0.0;
    let mut total_length = 0usize;
    let mut min_reward = f64::INFINITY;
    let mut max_reward = f64::NEG_INFINITY;

    for episode in 0..episodes {
        let mut env = CartPole::default();
        let mut observation = env
            .reset(Some(seed.wrapping_add(episode as u64)))
            .observation;
        let mut episode_reward = 0.0;
        let mut episode_length = 0usize;

        loop {
            let action = greedy_action(policy, &observation, &inference_device());
            let result = env.step(action);
            episode_reward += result.reward;
            episode_length += 1;

            if result.status.is_done() {
                break;
            }
            observation = result.observation;
        }

        total_reward += episode_reward;
        total_length += episode_length;
        min_reward = min_reward.min(episode_reward);
        max_reward = max_reward.max(episode_reward);
    }

    EvalSummary {
        episodes,
        mean_reward: total_reward / episodes as f64,
        mean_length: total_length as f64 / episodes as f64,
        min_reward,
        max_reward,
    }
}

fn greedy_action(
    policy: &QNetwork<InferenceBackend>,
    observation: &[f32; OBS_SIZE],
    device: &InferenceDevice,
) -> CartAction {
    let input = encode_one_infer(observation, device);
    let values = policy.forward(input);
    let index = values
        .argmax(1)
        .into_data()
        .to_vec::<i32>()
        .expect("argmax tensor converts to i32")[0] as usize;
    CartAction::from_index(index)
}

fn heuristic_action(observation: &[f32; OBS_SIZE]) -> CartAction {
    let [x, x_dot, theta, theta_dot] = *observation;
    let balance = 0.005_f32.mul_add(
        x_dot,
        0.01_f32.mul_add(x, 0.15_f32.mul_add(theta_dot, theta)),
    );
    if balance >= 0.0 {
        CartAction::Right
    } else {
        CartAction::Left
    }
}

fn encode_one_infer(
    observation: &[f32; OBS_SIZE],
    device: &InferenceDevice,
) -> Tensor<InferenceBackend, 2> {
    Tensor::<InferenceBackend, 1>::from_floats(observation.as_slice(), device)
        .reshape([1, OBS_SIZE])
}

fn encode_batch_infer(
    observations: &[[f32; OBS_SIZE]],
    device: &InferenceDevice,
) -> Tensor<InferenceBackend, 2> {
    let flat: Vec<f32> = observations.iter().flatten().copied().collect();
    Tensor::<InferenceBackend, 1>::from_floats(flat.as_slice(), device)
        .reshape([observations.len(), OBS_SIZE])
}

fn encode_batch_train(
    observations: &[[f32; OBS_SIZE]],
    device: &TrainingDevice,
) -> Tensor<TrainingBackend, 2> {
    let flat: Vec<f32> = observations.iter().flatten().copied().collect();
    Tensor::<TrainingBackend, 1>::from_floats(flat.as_slice(), device)
        .reshape([observations.len(), OBS_SIZE])
}

fn save_policy(policy: &QNetwork<InferenceBackend>, path: &Path) -> Result<(), Box<dyn Error>> {
    policy
        .clone()
        .save_file(path.to_path_buf(), &policy_recorder())?;
    Ok(())
}

fn load_policy(path: &Path) -> Result<QNetwork<InferenceBackend>, Box<dyn Error>> {
    let device = inference_device();
    let policy = QNetwork::<InferenceBackend>::new(&device).load_file(
        path.to_path_buf(),
        &policy_recorder(),
        &device,
    )?;
    Ok(policy)
}

fn jitter(seed: Option<u64>, lane: u64) -> f32 {
    let mut value = seed
        .unwrap_or(0)
        .wrapping_add(lane.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let normalized = (value as f64 / u64::MAX as f64) as f32;
    normalized.mul_add(0.1, -0.05)
}

fn write_train_config(
    paths: &RunPaths,
    config: &CartPoleDqnConfig,
    seeds: SeedConfig,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(&paths.config_json)?;
    writeln!(
        file,
        "{{\n  \"algorithm\": \"dqn\",\n  \"env\": \"cartpole\",\n  \"train_steps\": {},\n  \"eval_interval\": {},\n  \"eval_episodes\": {},\n  \"batch_size\": {},\n  \"replay_capacity\": {},\n  \"min_replay_size\": {},\n  \"target_update_interval\": {},\n  \"gamma\": {},\n  \"learning_rate\": {},\n  \"epsilon_start\": {},\n  \"epsilon_end\": {},\n  \"epsilon_decay_steps\": {},\n  \"warmup_batches\": {},\n  \"warmup_batch_size\": {}\n}}",
        config.train_steps,
        config.eval_interval,
        config.eval_episodes,
        config.batch_size,
        config.replay_capacity,
        config.min_replay_size,
        config.target_update_interval,
        config.gamma,
        config.learning_rate,
        config.epsilon_start,
        config.epsilon_end,
        config.epsilon_decay_steps,
        config.warmup_batches,
        config.warmup_batch_size
    )?;

    let mut seeds_file = File::create(&paths.seeds_json)?;
    writeln!(
        seeds_file,
        "{{\n  \"root\": {},\n  \"env_reset\": {},\n  \"action\": {},\n  \"replay\": {},\n  \"model\": {},\n  \"eval\": {}\n}}",
        seeds.root, seeds.env_reset, seeds.action, seeds.replay, seeds.model, seeds.eval
    )?;

    Ok(())
}

fn append_eval_record(
    paths: &RunPaths,
    global_step: usize,
    summary: &EvalSummary,
) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.eval_jsonl)?;
    writeln!(
        file,
        "{{\"global_step\":{},\"episodes\":{},\"mean_reward\":{},\"mean_length\":{},\"min_reward\":{},\"max_reward\":{}}}",
        global_step,
        summary.episodes,
        summary.mean_reward,
        summary.mean_length,
        summary.min_reward,
        summary.max_reward
    )?;
    Ok(())
}

fn write_summary(
    paths: &RunPaths,
    global_steps: usize,
    initial: &EvalSummary,
    best: &EvalSummary,
    config: &CartPoleDqnConfig,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(&paths.summary_json)?;
    writeln!(
        file,
        "{{\n  \"global_steps\": {},\n  \"initial_mean_reward\": {},\n  \"best_mean_reward\": {},\n  \"best_mean_length\": {},\n  \"learned_reward_delta\": {},\n  \"warmup_batches\": {},\n  \"solved_threshold\": {},\n  \"best_checkpoint\": \"{}\"\n}}",
        global_steps,
        initial.mean_reward,
        best.mean_reward,
        best.mean_length,
        best.mean_reward - initial.mean_reward,
        config.warmup_batches,
        SOLVED_MEAN_REWARD,
        json_escape(&paths.best_checkpoint.display().to_string())
    )?;
    Ok(())
}

fn print_train_report(report: &TrainReport) {
    println!(
        "run_dir={}\nbest_checkpoint={}\ninitial_mean_reward={:.1}\nbest_mean_reward={:.1}\nbest_mean_length={:.1}\nglobal_steps={}",
        report.paths.run_dir.display(),
        report.paths.best_checkpoint.display(),
        report.initial.mean_reward,
        report.best.mean_reward,
        report.best.mean_length,
        report.global_steps
    );
}

fn default_run_id() -> Result<String, Box<dyn Error>> {
    let since_epoch = SystemTime::now().duration_since(UNIX_EPOCH)?;
    Ok(format!("run-{}", since_epoch.as_secs()))
}

fn json_escape(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

fn parse_next<T>(raw: &mut impl Iterator<Item = String>, flag: &'static str) -> Result<T, CliError>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    let value = raw
        .next()
        .ok_or_else(|| CliError::Invalid(format!("{flag} requires a value")))?;
    value
        .parse()
        .map_err(|error| CliError::Invalid(format!("invalid value for {flag}: {error}")))
}

fn help_text() -> &'static str {
    "Usage: cargo run --example cartpole [--features render] -- [train|eval|video|watch|train-watch]\n\
     Flags: --steps N --eval-episodes N --eval-interval N --warmup-batches N --run-id ID\n\
     Flags: --runs-root PATH --checkpoint PATH --seed N --screenshot PATH --screenshot-frames N\n\
     Flags: --output PATH --video-seconds N --final-seconds N --smoke"
}

#[cfg(feature = "render")]
#[derive(Resource)]
struct VisualPolicy {
    policy: QNetwork<InferenceBackend>,
    device: InferenceDevice,
}

#[cfg(feature = "render")]
#[derive(Resource)]
struct ScreenshotRequest {
    path: PathBuf,
    wait_frames: u32,
    requested: bool,
}

#[cfg(feature = "render")]
#[derive(Component)]
struct CartBody;

#[cfg(feature = "render")]
#[derive(Component)]
struct PoleBody;

#[cfg(feature = "render")]
#[derive(Component)]
struct PivotBody;

#[cfg(feature = "render")]
#[derive(Component)]
struct Wheel {
    offset: f32,
}

#[cfg(feature = "render")]
fn run_visual(
    checkpoint: &Path,
    screenshot: Option<&Path>,
    screenshot_frames: u32,
) -> Result<(), Box<dyn Error>> {
    let policy = load_policy(checkpoint)?;
    let mut app = App::new();

    app.insert_resource(VisualPolicy {
        policy,
        device: inference_device(),
    })
    .insert_resource(ClearColor(Color::srgb(1.0, 1.0, 1.0)))
    .add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "bevy-gym CartPole DQN policy".into(),
                    resolution: WindowResolution::new(820, 520),
                    present_mode: PresentMode::AutoVsync,
                    ..default()
                }),
                ..default()
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    backends: Some(Backends::PRIMARY),
                    ..default()
                }),
                ..default()
            }),
    )
    .add_plugins(BevyGymPlugin::new(|_| CartPole::default()).with_tick_rate(50.0))
    .add_systems(Startup, setup_visuals)
    .add_systems(
        FixedUpdate,
        model_policy_system.in_set(GymSet::RequestActions),
    )
    .add_systems(Update, update_visuals);

    #[cfg(feature = "bevy-mcp")]
    app.add_plugins(BrpExtrasPlugin::with_port(brp_port()));

    #[cfg(all(feature = "render", not(feature = "bevy-mcp")))]
    app.add_plugins(RemotePlugin::default())
        .add_plugins(RemoteHttpPlugin::default().with_port(brp_port()));

    if let Some(path) = screenshot {
        app.insert_resource(ScreenshotRequest {
            path: path.to_path_buf(),
            wait_frames: screenshot_frames,
            requested: false,
        })
        .add_systems(Update, screenshot_once);
    }

    println!(
        "watching checkpoint={} brp_port={}",
        checkpoint.display(),
        brp_port()
    );
    app.run();
    Ok(())
}

#[cfg(not(feature = "render"))]
fn run_visual(
    _checkpoint: &Path,
    _screenshot: Option<&Path>,
    _screenshot_frames: u32,
) -> Result<(), Box<dyn Error>> {
    Err("visual mode requires `--features bevy_remote` or `--features render`".into())
}

#[cfg(feature = "render")]
fn setup_visuals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn((Camera2d, Name::new("CartPole Camera")));

    commands.spawn((
        Sprite::from_color(Color::srgb(0.12, 0.12, 0.12), Vec2::new(760.0, 6.0)),
        Transform::from_xyz(0.0, TRACK_Y, 0.0),
        Name::new("CartPole Track"),
    ));

    for x in [
        -X_THRESHOLD * CART_SCREEN_SCALE,
        X_THRESHOLD * CART_SCREEN_SCALE,
    ] {
        commands.spawn((
            Sprite::from_color(Color::srgb(0.45, 0.45, 0.45), Vec2::new(4.0, 36.0)),
            Transform::from_xyz(x, TRACK_Y + 16.0, 0.1),
            Name::new("CartPole Boundary Marker"),
        ));
    }

    commands.spawn((
        Sprite::from_color(Color::srgb(0.08, 0.08, 0.08), Vec2::new(CART_W, CART_H)),
        Transform::from_xyz(0.0, CART_Y, 2.0),
        CartBody,
        Name::new("CartPole Cart"),
    ));

    for offset in [-24.0, 24.0] {
        commands.spawn((
            Mesh2d(meshes.add(Circle::new(8.0))),
            MeshMaterial2d(materials.add(Color::srgb(0.02, 0.02, 0.02))),
            Transform::from_xyz(offset, CART_Y - 20.0, 2.2),
            Wheel { offset },
            Name::new("CartPole Wheel"),
        ));
    }

    commands.spawn((
        Sprite::from_color(Color::srgb(0.08, 0.08, 0.08), Vec2::new(POLE_W, POLE_LEN)),
        Transform::from_xyz(0.0, CART_Y + CART_H * 0.5 + POLE_LEN * 0.5, 3.0),
        PoleBody,
        Name::new("CartPole Pole"),
    ));

    commands.spawn((
        Mesh2d(meshes.add(Circle::new(7.0))),
        MeshMaterial2d(materials.add(Color::srgb(0.0, 0.0, 0.0))),
        Transform::from_xyz(0.0, CART_Y + CART_H * 0.5, 3.2),
        PivotBody,
        Name::new("CartPole Hinge"),
    ));
}

#[cfg(feature = "render")]
fn model_policy_system(
    policy: Res<VisualPolicy>,
    mut requests: MessageReader<ActionRequest<CartPole>>,
    mut responses: MessageWriter<ActionResponse<CartPole>>,
) {
    for request in requests.read() {
        responses.write(ActionResponse {
            entity: request.entity,
            action: greedy_action(&policy.policy, &request.observation, &policy.device),
        });
    }
}

#[cfg(feature = "render")]
fn update_visuals(
    env_query: Query<(&CurrentObservation<CartPole>, &EnvStats), With<EnvComponent<CartPole>>>,
    mut transforms: ParamSet<(
        Query<&mut Transform, With<CartBody>>,
        Query<&mut Transform, With<PivotBody>>,
        Query<&mut Transform, With<PoleBody>>,
        Query<(&Wheel, &mut Transform)>,
    )>,
) {
    let Ok((observation, _stats)) = env_query.single() else {
        return;
    };
    let [x, _, theta, _] = observation.observation;
    let cart_x = x * CART_SCREEN_SCALE;
    let pivot = Vec2::new(cart_x, CART_Y + CART_H * 0.5);

    if let Ok(mut transform) = transforms.p0().single_mut() {
        transform.translation.x = cart_x;
        transform.translation.y = CART_Y;
    }

    if let Ok(mut transform) = transforms.p1().single_mut() {
        transform.translation.x = pivot.x;
        transform.translation.y = pivot.y;
    }

    if let Ok(mut transform) = transforms.p2().single_mut() {
        let center = pivot + Vec2::new(theta.sin(), theta.cos()) * (POLE_LEN * 0.5);
        transform.translation.x = center.x;
        transform.translation.y = center.y;
        transform.rotation = Quat::from_rotation_z(-theta);
    }

    for (wheel, mut transform) in &mut transforms.p3() {
        transform.translation.x = cart_x + wheel.offset;
        transform.translation.y = CART_Y - 20.0;
    }
}

#[cfg(feature = "render")]
fn screenshot_once(
    mut commands: Commands,
    mut request: ResMut<ScreenshotRequest>,
    captures: Query<Entity, With<Capturing>>,
    mut exit: MessageWriter<AppExit>,
) {
    if request.wait_frames > 0 {
        request.wait_frames -= 1;
        return;
    }

    if !request.requested {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(request.path.clone()));
        request.requested = true;
        request.wait_frames = 20;
        return;
    }

    if captures.is_empty() {
        exit.write(AppExit::Success);
    }
}

#[cfg(feature = "render")]
fn brp_port() -> u16 {
    env::var("BRP_EXTRAS_PORT")
        .ok()
        .or_else(|| env::var("BEVY_GYM_BRP_PORT").ok())
        .and_then(|value| value.parse().ok())
        .unwrap_or(15_702)
}
