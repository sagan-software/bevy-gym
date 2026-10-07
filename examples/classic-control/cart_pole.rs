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
    clippy::disallowed_methods,
    clippy::disallowed_types,
    unused_crate_dependencies,
    reason = "the synchronous example keeps its complete workflow readable in one target"
)]

use tokio as _;

use std::env;
use std::error::Error;

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
use bevy_gym::environments::CartPoleState;
use bevy_gym::environments::{CartPole, CartPoleAction as CartAction};
use bevy_gym::recording::{CheckpointRole, CheckpointTimeline};
use bevy_gym::training::{
    AlgorithmKind, BevyTransitionCollector, DqnAgent, DqnConfig, DqnPolicy, MetricRecord,
    MetricValue, MetricsWriter, RunConfig, RunId, RunPaths, SeedConfig,
};
use bevy_gym::{Env, EpisodeStatus, TimeLimit};

const OBS_SIZE: usize = 4;
const NUM_ACTIONS: usize = 2;
const MAX_STEPS_PER_EPISODE: usize = 500;

const DEFAULT_TRAIN_STEPS: usize = 100_000;
const DEFAULT_EVAL_EPISODES: usize = 12;
const DEFAULT_EVAL_INTERVAL: usize = 5_000;
const DEFAULT_NUM_ENVS: usize = 8;
const DEFAULT_ROOT_SEED: u64 = 42;
const SOLVED_MEAN_REWARD: f64 = 475.0;
const DEFAULT_VIDEO_FPS: usize = 50;
const DEFAULT_VIDEO_WIDTH: usize = 1_280;
const DEFAULT_VIDEO_HEIGHT: usize = 720;
const OFFICIAL_VIEWPORT_WIDTH: i32 = 600;
const OFFICIAL_VIEWPORT_HEIGHT: i32 = 400;
const CART_SCREEN_SCALE: f32 = 125.0;
const TRACK_Y: f32 = -100.0;
const CART_Y: f32 = TRACK_Y;
const CART_W: f32 = 50.0;
const CART_H: f32 = 30.0;
const POLE_LEN: f32 = 125.0;
const POLE_W: f32 = 10.0;
const AXLE_OFFSET: f32 = CART_H / 4.0;
const POLE_CENTER_OFFSET: f32 = (POLE_LEN - POLE_W) / 2.0;

#[derive(Debug, Clone, Copy, PartialEq)]
struct CartPoleScene {
    cart_center: [f32; 2],
    pivot: [f32; 2],
    pole_center: [f32; 2],
    pole_rotation: f32,
}

impl CartPoleScene {
    fn from_observation(observation: &[f32; OBS_SIZE]) -> Self {
        let [x, _, theta, _] = *observation;
        let cart_x = x * CART_SCREEN_SCALE;
        let pivot = [cart_x, CART_Y + AXLE_OFFSET];
        let pole_center = [
            theta.sin().mul_add(POLE_CENTER_OFFSET, pivot[0]),
            theta.cos().mul_add(POLE_CENTER_OFFSET, pivot[1]),
        ];

        Self {
            cart_center: [cart_x, CART_Y],
            pivot,
            pole_center,
            pole_rotation: -theta,
        }
    }
}

type CartPoleV1 = TimeLimit<CartPole>;

fn cartpole_v1() -> CartPoleV1 {
    TimeLimit::new(CartPole::default(), MAX_STEPS_PER_EPISODE)
        .expect("CartPole-v1 has a positive time limit")
}

#[derive(Debug, Clone)]
struct CartPoleDqnConfig {
    train_steps: usize,
    eval_interval: usize,
    eval_episodes: usize,
    num_envs: usize,
    batch_size: usize,
    replay_capacity: usize,
    min_replay_size: usize,
    target_update_interval: usize,
    gamma: f32,
    learning_rate: f64,
    epsilon_start: f64,
    epsilon_end: f64,
    epsilon_decay_steps: usize,
}

impl Default for CartPoleDqnConfig {
    fn default() -> Self {
        Self {
            train_steps: DEFAULT_TRAIN_STEPS,
            eval_interval: DEFAULT_EVAL_INTERVAL,
            eval_episodes: DEFAULT_EVAL_EPISODES,
            num_envs: DEFAULT_NUM_ENVS,
            batch_size: 64,
            replay_capacity: 50_000,
            min_replay_size: 1_000,
            target_update_interval: 500,
            gamma: 0.99,
            learning_rate: 3e-4,
            epsilon_start: 0.15,
            epsilon_end: 0.01,
            epsilon_decay_steps: 10_000,
        }
    }
}

impl CartPoleDqnConfig {
    fn learner_config(&self) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![64, 64],
            gamma: self.gamma,
            learning_rate: self.learning_rate,
            replay_capacity: self.replay_capacity,
            min_replay_size: self.min_replay_size,
            batch_size: self.batch_size,
            target_update_interval: self.target_update_interval,
            epsilon_start: self.epsilon_start,
            epsilon_end: self.epsilon_end,
            epsilon_decay_steps: self.epsilon_decay_steps as u64,
            ..DqnConfig::default()
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalSuiteArg {
    Validation,
    Test,
    Demo,
}

impl EvalSuiteArg {
    const fn seed(self, seeds: SeedConfig) -> u64 {
        match self {
            Self::Validation => seeds.validation,
            Self::Test => seeds.test,
            Self::Demo => seeds.demo,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Validation => "validation",
            Self::Test => "test",
            Self::Demo => "demo",
        }
    }
}

#[derive(Debug, Clone)]
struct Args {
    mode: Mode,
    config: CartPoleDqnConfig,
    runs_root: PathBuf,
    run_id: Option<String>,
    checkpoint: Option<PathBuf>,
    seed: u64,
    eval_suite: EvalSuiteArg,
    screenshot: Option<PathBuf>,
    screenshot_frames: u32,
    video_output: Option<PathBuf>,
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
        let mut eval_suite = EvalSuiteArg::Test;
        let mut screenshot = None;
        let mut screenshot_frames = 90;
        let mut video_output = None;

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
                "--num-envs" => config.num_envs = parse_next(&mut raw, "--num-envs")?,
                "--run-id" => run_id = Some(parse_next::<String>(&mut raw, "--run-id")?),
                "--runs-root" => {
                    runs_root = PathBuf::from(parse_next::<String>(&mut raw, "--runs-root")?);
                }
                "--checkpoint" => {
                    checkpoint = Some(PathBuf::from(parse_next::<String>(
                        &mut raw,
                        "--checkpoint",
                    )?));
                }
                "--seed" => seed = parse_next(&mut raw, "--seed")?,
                "--suite" => {
                    let value = parse_next::<String>(&mut raw, "--suite")?;
                    eval_suite = match value.as_str() {
                        "validation" => EvalSuiteArg::Validation,
                        "test" => EvalSuiteArg::Test,
                        "demo" => EvalSuiteArg::Demo,
                        _ => {
                            return Err(CliError::Invalid(format!(
                                "--suite must be validation, test, or demo; got {value:?}"
                            )))
                        }
                    };
                }
                "--screenshot" => {
                    screenshot = Some(PathBuf::from(parse_next::<String>(
                        &mut raw,
                        "--screenshot",
                    )?));
                }
                "--screenshot-frames" => {
                    screenshot_frames = parse_next(&mut raw, "--screenshot-frames")?;
                }
                "--output" | "--video-output" => {
                    video_output = Some(PathBuf::from(parse_next::<String>(&mut raw, "--output")?));
                }
                "--smoke" => {
                    config.train_steps = 512;
                    config.eval_interval = 256;
                    config.eval_episodes = 2;
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
            eval_suite,
            screenshot,
            screenshot_frames,
            video_output,
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
            let eval_seed = args.eval_suite.seed(SeedConfig::from_root(args.seed));
            let summary = eval_policy(&policy, args.config.eval_episodes, eval_seed);
            println!(
                "eval checkpoint={} suite={} episodes={} mean_reward={:.1} mean_length={:.1} min={:.1} max={:.1}",
                checkpoint.display(),
                args.eval_suite.as_str(),
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
                    fps: DEFAULT_VIDEO_FPS,
                    width: DEFAULT_VIDEO_WIDTH,
                    height: DEFAULT_VIDEO_HEIGHT,
                    seed: SeedConfig::from_root(args.seed).demo,
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
    paths.create_new()?;
    write_train_config(&paths, &args.config, seeds)?;

    let mut agent = DqnAgent::new(OBS_SIZE, NUM_ACTIONS, args.config.learner_config(), seeds)?;

    let initial = eval_policy(&agent.policy(), args.config.eval_episodes, seeds.validation);
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
            .with_field("train/optimizer_updates", MetricValue::Number(0.0))
            .with_field("qualification/behavior_cloning", MetricValue::Bool(false))
            .with_field("qualification/reward_shaping", MetricValue::Bool(false)),
    )?;
    if args.config.num_envs < 2 {
        return Err("CartPole training requires at least two Bevy environment entities".into());
    }
    if args.config.eval_interval == 0 {
        return Err("--eval-interval must be greater than zero".into());
    }
    if !args.config.train_steps.is_multiple_of(args.config.num_envs) {
        return Err("--steps must be divisible by --num-envs for an exact training budget".into());
    }

    let reset_seeds = seeds;
    let mut collector = BevyTransitionCollector::new(
        |_| cartpole_v1(),
        args.config.num_envs,
        move |env_id, episode| Some(reset_seeds.environment_episode(env_id, episode)),
    )?;
    let mut episodes = 0usize;
    let mut last_loss = None;
    let mut last_policy_delta = None;
    let mut next_eval = args.config.eval_interval;

    while agent.global_steps() < args.config.train_steps as u64 {
        let requested_observations = collector
            .requests()
            .iter()
            .map(|request| request.observation)
            .collect::<Vec<_>>();
        let actions = requested_observations
            .iter()
            .map(|observation| {
                agent.select_action(observation).map(|selection| {
                    CartAction::try_from(selection.action_index).expect("two-output policy")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batch = collector.step(actions)?;

        for event in &batch.transitions {
            let transition = &event.transition;
            if let Some(update) = agent.observe(
                &transition.observation,
                usize::from(transition.action),
                transition.reward,
                &transition.next_observation,
                transition.status,
            )? {
                last_loss = Some(update.loss);
                last_policy_delta = Some(update.policy_output_delta_l1);
            }
        }

        let step = agent.global_steps() as usize;
        for episode in batch.episode_ends {
            episodes += 1;
            if episodes.is_multiple_of(10) {
                let mut record = MetricRecord::new(step as u64)
                    .with_field("train/episode", MetricValue::Number(episodes as f64))
                    .with_field("train/reward", MetricValue::Number(episode.total_reward))
                    .with_field(
                        "train/length",
                        MetricValue::Number(episode.episode_steps as f64),
                    )
                    .with_field("train/epsilon", MetricValue::Number(agent.epsilon()))
                    .with_field(
                        "train/optimizer_updates",
                        MetricValue::Number(agent.optimizer_steps() as f64),
                    );
                if let Some(loss) = last_loss {
                    record = record.with_field("train/loss", MetricValue::Number(loss));
                }
                if let Some(delta) = last_policy_delta {
                    record = record
                        .with_field("train/policy_output_delta_l1", MetricValue::Number(delta));
                }
                metrics.write_record(&record)?;
                println!(
                    "step {step:>6} episode {episodes:>4} reward {:>6.1} length {:>3} epsilon {:.3}",
                    episode.total_reward,
                    episode.episode_steps,
                    agent.epsilon()
                );
            }
        }

        if step >= next_eval || step == args.config.train_steps {
            let policy = agent.policy();
            let summary = eval_policy(&policy, args.config.eval_episodes, seeds.validation);
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
                write_summary(&paths, step, &initial, &best)?;
                return Ok(TrainReport {
                    paths,
                    initial,
                    best,
                    global_steps: step,
                });
            }

            next_eval = next_eval.saturating_add(args.config.eval_interval);
        }
    }

    write_summary(&paths, args.config.train_steps, &initial, &best)?;
    Ok(TrainReport {
        paths,
        initial,
        best,
        global_steps: args.config.train_steps,
    })
}

fn resolve_or_train_checkpoint(args: &Args) -> Result<PathBuf, Box<dyn Error>> {
    if let Ok(checkpoint) = resolve_checkpoint(args.checkpoint.as_deref(), &args.runs_root) {
        return Ok(checkpoint);
    }

    let mut train_args = args.clone();
    train_args.mode = Mode::Train;
    let report = train(train_args)?;
    Ok(report.paths.best_checkpoint)
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
    fps: usize,
    width: usize,
    height: usize,
    seed: u64,
}

fn render_training_video(
    run_dir: &Path,
    output: &Path,
    config: VideoConfig,
) -> Result<(), Box<dyn Error>> {
    let timeline = CheckpointTimeline::training_progress(run_dir)?;

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

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
        let timeline_frames = timeline.duration().as_secs() as usize * config.fps;
        for segment in timeline.segments() {
            let segment_frames = segment.duration().as_secs() as usize * config.fps;
            let policy = load_policy(segment.checkpoint())?;
            let label = match segment.role() {
                CheckpointRole::Best => "BEST CHECKPOINT",
                CheckpointRole::First => "FIRST CHECKPOINT",
                CheckpointRole::Progress33 => "33% CHECKPOINT",
                CheckpointRole::Progress66 => "66% CHECKPOINT",
            };
            write_policy_video_segment(
                &mut stdin,
                &policy,
                segment_frames,
                &config,
                config.seed,
                label,
                &mut frame_index,
                timeline_frames,
            )?;
        }
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
fn write_policy_video_segment(
    writer: &mut impl Write,
    policy: &DqnPolicy,
    frames: usize,
    config: &VideoConfig,
    seed: u64,
    label: &str,
    frame_index: &mut usize,
    timelapse_frames: usize,
) -> Result<(), Box<dyn Error>> {
    let mut env = cartpole_v1();
    let mut episode = 0u64;
    let mut observation = env.reset(Some(seed)).observation;

    for _ in 0..frames {
        let progress = if timelapse_frames == 0 {
            1.0
        } else {
            (*frame_index).min(timelapse_frames) as f32 / timelapse_frames as f32
        };
        let mut frame = RgbFrame::new(config.width, config.height, [255, 255, 255]);
        draw_cartpole_frame(&mut frame, &observation, label, progress);
        frame.write_ppm(writer)?;

        let action = greedy_action(policy, &observation);
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
        let (Ok(x), Ok(y)) = (usize::try_from(x), usize::try_from(y)) else {
            return;
        };
        if x >= self.width || y >= self.height {
            return;
        }

        let index = (y * self.width + x) * 3;
        if let Some(pixel) = self.pixels.get_mut(index..index + 3) {
            pixel.copy_from_slice(&color);
        }
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

    fn fill_convex_quad(&mut self, points: [[f32; 2]; 4], color: [u8; 3]) {
        let min_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min)
            .floor() as i32;
        let max_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil() as i32;
        let min_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min)
            .floor() as i32;
        let max_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil() as i32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let sample = [x as f32 + 0.5, y as f32 + 0.5];
                let mut has_positive = false;
                let mut has_negative = false;
                for (start, end) in points
                    .iter()
                    .copied()
                    .zip(points.iter().copied().cycle().skip(1))
                    .take(points.len())
                {
                    let cross = (end[0] - start[0]).mul_add(
                        sample[1] - start[1],
                        -((end[1] - start[1]) * (sample[0] - start[0])),
                    );
                    has_positive |= cross > 0.0;
                    has_negative |= cross < 0.0;
                }
                if !(has_positive && has_negative) {
                    self.set_pixel(x, y, color);
                }
            }
        }
    }
}

fn draw_cartpole_frame(
    frame: &mut RgbFrame,
    observation: &[f32; OBS_SIZE],
    label: &str,
    progress: f32,
) {
    let black = [0, 0, 0];
    let blue = [47, 111, 184];
    let tan = [202, 152, 101];
    let lavender = [129, 132, 203];
    let viewport_x = if frame.width as i32 > OFFICIAL_VIEWPORT_WIDTH {
        40
    } else {
        0
    };
    let viewport_y = ((frame.height as i32 - OFFICIAL_VIEWPORT_HEIGHT) / 2).max(0);
    let scene = CartPoleScene::from_observation(observation);
    let viewport_center_x = (OFFICIAL_VIEWPORT_WIDTH as f32).mul_add(0.5, viewport_x as f32);
    let viewport_center_y = (OFFICIAL_VIEWPORT_HEIGHT as f32).mul_add(0.5, viewport_y as f32);
    let to_screen = |point: [f32; 2]| [viewport_center_x + point[0], viewport_center_y - point[1]];

    frame.fill_rect(
        0,
        0,
        frame.width as i32,
        frame.height as i32,
        [242, 244, 248],
    );
    frame.fill_rect(
        viewport_x,
        viewport_y,
        OFFICIAL_VIEWPORT_WIDTH,
        OFFICIAL_VIEWPORT_HEIGHT,
        [255, 255, 255],
    );

    let cart = to_screen(scene.cart_center);
    frame.fill_rect(
        CART_W.mul_add(-0.5, cart[0]).round() as i32,
        CART_H.mul_add(-0.5, cart[1]).round() as i32,
        CART_W as i32,
        CART_H as i32,
        black,
    );

    let pole = to_screen(scene.pole_center);
    let sin = scene.pole_rotation.sin();
    let cos = scene.pole_rotation.cos();
    let half_width = POLE_W * 0.5;
    let half_length = POLE_LEN * 0.5;
    let rotate = |local_x: f32, local_y: f32| {
        [
            local_y.mul_add(-sin, local_x.mul_add(cos, pole[0])),
            local_x.mul_add(-sin, local_y.mul_add(-cos, pole[1])),
        ]
    };
    frame.fill_convex_quad(
        [
            rotate(-half_width, -half_length),
            rotate(-half_width, half_length),
            rotate(half_width, half_length),
            rotate(half_width, -half_length),
        ],
        tan,
    );

    let pivot = to_screen(scene.pivot);
    frame.fill_circle(
        pivot[0].round() as i32,
        pivot[1].round() as i32,
        (POLE_W * 0.5) as i32,
        lavender,
    );

    let track_y = to_screen([0.0, TRACK_Y])[1].round() as i32;
    frame.fill_rect(viewport_x, track_y, OFFICIAL_VIEWPORT_WIDTH, 1, black);

    let panel_x = viewport_x + OFFICIAL_VIEWPORT_WIDTH + 36;
    draw_text(frame, panel_x, viewport_y + 40, 2, label, black);
    frame.fill_rect(panel_x, viewport_y + 72, 520, 6, [220, 220, 220]);
    frame.fill_rect(
        panel_x,
        viewport_y + 72,
        (520.0 * progress.clamp(0.0, 1.0)).round() as i32,
        6,
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

const fn glyph_rows(ch: char) -> [&'static str; 7] {
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

fn eval_policy(policy: &DqnPolicy, episodes: usize, seed: u64) -> EvalSummary {
    let mut total_reward = 0.0;
    let mut total_length = 0usize;
    let mut min_reward = f64::INFINITY;
    let mut max_reward = f64::NEG_INFINITY;

    for episode in 0..episodes {
        let mut env = cartpole_v1();
        let mut observation = env
            .reset(Some(seed.wrapping_add(episode as u64)))
            .observation;
        let mut episode_reward = 0.0;
        let mut episode_length = 0usize;

        loop {
            let action = greedy_action(policy, &observation);
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

fn greedy_action(policy: &DqnPolicy, observation: &[f32; OBS_SIZE]) -> CartAction {
    let index = policy
        .greedy_action(observation)
        .expect("CartPole observation width matches the saved DQN policy");
    CartAction::try_from(index).expect("two-output policy")
}

#[cfg(test)]
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

fn save_policy(policy: &DqnPolicy, path: &Path) -> Result<(), Box<dyn Error>> {
    policy.save(path)?;
    Ok(())
}

fn load_policy(path: &Path) -> Result<DqnPolicy, Box<dyn Error>> {
    Ok(DqnPolicy::load(path, OBS_SIZE, NUM_ACTIONS, &[64, 64])?)
}

fn write_train_config(
    paths: &RunPaths,
    config: &CartPoleDqnConfig,
    seeds: SeedConfig,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(&paths.config_json)?;
    writeln!(
        file,
        "{{\n  \"algorithm\": \"dqn\",\n  \"env\": \"cartpole\",\n  \"train_steps\": {},\n  \"eval_interval\": {},\n  \"eval_episodes\": {},\n  \"num_envs\": {},\n  \"batch_size\": {},\n  \"replay_capacity\": {},\n  \"min_replay_size\": {},\n  \"target_update_interval\": {},\n  \"gamma\": {},\n  \"learning_rate\": {},\n  \"epsilon_start\": {},\n  \"epsilon_end\": {},\n  \"epsilon_decay_steps\": {},\n  \"behavior_cloning\": false,\n  \"reward_shaping\": false\n}}",
        config.train_steps,
        config.eval_interval,
        config.eval_episodes,
        config.num_envs,
        config.batch_size,
        config.replay_capacity,
        config.min_replay_size,
        config.target_update_interval,
        config.gamma,
        config.learning_rate,
        config.epsilon_start,
        config.epsilon_end,
        config.epsilon_decay_steps
    )?;

    let mut seeds_file = File::create(&paths.seeds_json)?;
    writeln!(
        seeds_file,
        "{{\n  \"root\": {},\n  \"env_construction\": {},\n  \"env_reset\": {},\n  \"action\": {},\n  \"replay\": {},\n  \"rollout\": {},\n  \"model\": {},\n  \"validation\": {},\n  \"test\": {},\n  \"demo\": {}\n}}",
        seeds.root,
        seeds.env_construction,
        seeds.env_reset,
        seeds.action,
        seeds.replay,
        seeds.rollout,
        seeds.model,
        seeds.validation,
        seeds.test,
        seeds.demo
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
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(&paths.summary_json)?;
    writeln!(
        file,
        "{{\n  \"global_steps\": {},\n  \"initial_mean_reward\": {},\n  \"best_mean_reward\": {},\n  \"best_mean_length\": {},\n  \"learned_reward_delta\": {},\n  \"behavior_cloning\": false,\n  \"reward_shaping\": false,\n  \"solved_threshold\": {},\n  \"best_checkpoint\": \"{}\"\n}}",
        global_steps,
        initial.mean_reward,
        best.mean_reward,
        best.mean_length,
        best.mean_reward - initial.mean_reward,
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

const fn help_text() -> &'static str {
    "Usage: cargo run --example cartpole [--features render] -- [train|eval|video|watch|train-watch]\n\
     Flags: --steps N --eval-episodes N --eval-interval N --num-envs N --run-id ID\n\
     Flags: --runs-root PATH --checkpoint PATH --seed N --screenshot PATH --screenshot-frames N\n\
     Flags: --output PATH --smoke"
}

fn run_visual(
    checkpoint: &Path,
    screenshot: Option<&Path>,
    screenshot_frames: u32,
) -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "render"))]
    let _ = (checkpoint, screenshot, screenshot_frames);
    #[cfg(feature = "render")]
    return render::run_visual(checkpoint, screenshot, screenshot_frames);
    #[cfg(not(feature = "render"))]
    return Err("visual mode requires the `render` feature".into());
}

/// Render-only `CartPole` scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        cartpole_v1, env, greedy_action, load_policy, CartPoleScene, CartPoleV1, DqnPolicy, Error,
        Path, PathBuf, AXLE_OFFSET, CART_H, CART_W, CART_Y, POLE_CENTER_OFFSET, POLE_LEN, POLE_W,
        TRACK_Y,
    };
    use bevy::prelude::*;
    #[cfg(not(feature = "bevy-mcp"))]
    use bevy::remote::{http::RemoteHttpPlugin, RemotePlugin};
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::render::{
        settings::{Backends, RenderCreation, WgpuSettings},
        RenderPlugin,
    };
    use bevy::window::{PresentMode, WindowResolution};
    #[cfg(feature = "bevy-mcp")]
    use bevy_brp_extras::BrpExtrasPlugin;
    use bevy_gym::{
        ActionRequest, ActionResponse, BevyGymPlugin, CurrentObservation, EnvComponent, EnvStats,
        GymSet,
    };

    /// `CartPole` renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb(1.0, 1.0, 1.0)));
        }
    }

    #[derive(Resource)]
    struct VisualPolicy {
        policy: DqnPolicy,
    }

    #[derive(Resource)]
    struct ScreenshotRequest {
        path: PathBuf,
        wait_frames: u32,
        requested: bool,
    }

    #[derive(Component)]
    struct CartBody;

    #[derive(Component)]
    struct PoleBody;

    #[derive(Component)]
    struct PivotBody;

    pub(super) fn run_visual(
        checkpoint: &Path,
        screenshot: Option<&Path>,
        screenshot_frames: u32,
    ) -> Result<(), Box<dyn Error>> {
        let policy = load_policy(checkpoint)?;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);

        app.insert_resource(VisualPolicy { policy })
            .add_plugins(
                DefaultPlugins
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "bevy-gym CartPole DQN policy".into(),
                            resolution: WindowResolution::new(600, 400),
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
            .add_plugins(BevyGymPlugin::new(|_| cartpole_v1()).with_tick_rate(50.0))
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

    fn setup_visuals(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        commands.spawn((Camera2d, Name::new("CartPole Camera")));

        commands.spawn((
            Sprite::from_color(Color::BLACK, Vec2::new(600.0, 2.0)),
            Transform::from_xyz(0.0, TRACK_Y, 0.0),
            Name::new("CartPole Track"),
        ));

        commands.spawn((
            Sprite::from_color(Color::BLACK, Vec2::new(CART_W, CART_H)),
            Transform::from_xyz(0.0, CART_Y, 2.0),
            CartBody,
            Name::new("CartPole Cart"),
        ));

        commands.spawn((
            Sprite::from_color(Color::srgb_u8(202, 152, 101), Vec2::new(POLE_W, POLE_LEN)),
            Transform::from_xyz(0.0, CART_Y + AXLE_OFFSET + POLE_CENTER_OFFSET, 3.0),
            PoleBody,
            Name::new("CartPole Pole"),
        ));

        commands.spawn((
            Mesh2d(meshes.add(Circle::new(POLE_W / 2.0))),
            MeshMaterial2d(materials.add(Color::srgb_u8(129, 132, 203))),
            Transform::from_xyz(0.0, CART_Y + AXLE_OFFSET, 3.2),
            PivotBody,
            Name::new("CartPole Hinge"),
        ));
    }

    fn model_policy_system(
        policy: Res<'_, VisualPolicy>,
        mut requests: MessageReader<'_, '_, ActionRequest<CartPoleV1>>,
        mut responses: MessageWriter<'_, ActionResponse<CartPoleV1>>,
    ) {
        for request in requests.read() {
            responses.write(ActionResponse {
                entity: request.entity,
                action: greedy_action(&policy.policy, &request.observation),
            });
        }
    }

    fn update_visuals(
        env_query: Query<
            '_,
            '_,
            (&CurrentObservation<CartPoleV1>, &EnvStats),
            With<EnvComponent<CartPoleV1>>,
        >,
        mut transforms: ParamSet<
            '_,
            '_,
            (
                Query<'_, '_, &mut Transform, With<CartBody>>,
                Query<'_, '_, &mut Transform, With<PivotBody>>,
                Query<'_, '_, &mut Transform, With<PoleBody>>,
            ),
        >,
    ) {
        let Ok((observation, _stats)) = env_query.single() else {
            return;
        };
        let scene = CartPoleScene::from_observation(&observation.observation);

        if let Ok(mut transform) = transforms.p0().single_mut() {
            transform.translation.x = scene.cart_center[0];
            transform.translation.y = scene.cart_center[1];
        }

        if let Ok(mut transform) = transforms.p1().single_mut() {
            transform.translation.x = scene.pivot[0];
            transform.translation.y = scene.pivot[1];
        }

        if let Ok(mut transform) = transforms.p2().single_mut() {
            transform.translation.x = scene.pole_center[0];
            transform.translation.y = scene.pole_center[1];
            transform.rotation = Quat::from_rotation_z(scene.pole_rotation);
        }
    }

    fn screenshot_once(
        mut commands: Commands<'_, '_>,
        mut request: ResMut<'_, ScreenshotRequest>,
        captures: Query<'_, '_, Entity, With<Capturing>>,
        mut exit: MessageWriter<'_, AppExit>,
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

    fn brp_port() -> u16 {
        env::var("BRP_EXTRAS_PORT")
            .ok()
            .or_else(|| env::var("BEVY_GYM_BRP_PORT").ok())
            .and_then(|value| value.parse().ok())
            .unwrap_or(15_702)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_initialization_uses_the_recorded_seed() {
        let config = CartPoleDqnConfig::default();
        let observation = [0.01, -0.02, 0.03, -0.04];

        let first = DqnAgent::new(
            OBS_SIZE,
            NUM_ACTIONS,
            config.learner_config(),
            SeedConfig::from_root(7),
        )
        .expect("valid first CartPole DQN")
        .policy();
        let second = DqnAgent::new(
            OBS_SIZE,
            NUM_ACTIONS,
            config.learner_config(),
            SeedConfig::from_root(7),
        )
        .expect("valid second CartPole DQN")
        .policy();
        let different = DqnAgent::new(
            OBS_SIZE,
            NUM_ACTIONS,
            config.learner_config(),
            SeedConfig::from_root(8),
        )
        .expect("valid different CartPole DQN")
        .policy();

        let values = |policy: &DqnPolicy| {
            policy
                .q_values(&observation)
                .expect("CartPole observation has four values")
        };

        assert_eq!(values(&first), values(&second));
        assert_ne!(values(&first), values(&different));
    }

    #[test]
    fn cartpole_v1_time_limit_truncates_without_masking_natural_failures() {
        let mut env = cartpole_v1();
        let mut observation = env.reset(Some(42)).observation;

        for step_index in 1..=MAX_STEPS_PER_EPISODE {
            let result = env.step(heuristic_action(&observation));
            if step_index < MAX_STEPS_PER_EPISODE {
                assert_eq!(result.status, EpisodeStatus::Continuing);
            } else {
                assert_eq!(result.status, EpisodeStatus::Truncated);
            }
            observation = result.observation;
        }

        let mut failed = cartpole_v1();
        *failed.inner_mut() = CartPole::from_state(
            CartPoleState::try_from([2.4, 5.0, 0.0, 0.0]).expect("finite state"),
        );
        assert_eq!(
            failed.step(CartAction::Right).status,
            EpisodeStatus::Terminated
        );
    }

    #[test]
    fn official_scene_mapping_and_video_viewport_keep_expected_anchors() {
        let observation = [0.0; OBS_SIZE];
        let scene = CartPoleScene::from_observation(&observation);
        assert_eq!(scene.cart_center, [0.0, -100.0]);
        assert_eq!(scene.pivot, [0.0, -92.5]);
        assert_eq!(scene.pole_center, [0.0, -35.0]);
        assert!(scene.pole_rotation.abs() < f32::EPSILON);

        let mut frame = RgbFrame::new(DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT, [0, 0, 0]);
        draw_cartpole_frame(&mut frame, &observation, "STEP 0", 0.0);
        let pixel = |x: usize, y: usize| {
            let index = (y * frame.width + x) * 3;
            frame.pixels[index..index + 3].to_vec()
        };

        assert_eq!(pixel(340, 460), vec![0, 0, 0]);
        assert_eq!(pixel(340, 453), vec![129, 132, 203]);
        assert_eq!(pixel(340, 400), vec![202, 152, 101]);
        assert_eq!(pixel(54, 176), vec![255, 255, 255]);
    }

    #[test]
    fn cartpole_collects_parallel_transitions_through_bevy_runner() {
        let seeds = SeedConfig::from_root(11);
        let mut collector = BevyTransitionCollector::new(
            |_| cartpole_v1(),
            4,
            move |env_id, episode| Some(seeds.environment_episode(env_id, episode)),
        )
        .expect("parallel CartPole collector starts");
        let actions = collector
            .requests()
            .iter()
            .map(|request| heuristic_action(&request.observation))
            .collect::<Vec<_>>();

        let batch = collector
            .step(actions)
            .expect("parallel runner step succeeds");

        assert_eq!(batch.environment_steps(), 4);
        assert_eq!(collector.requests().len(), 4);
        assert!(batch
            .transitions
            .iter()
            .all(|event| event.transition.reward == 1.0));
    }

    #[test]
    fn cartpole_dynamics_match_pinned_gymnasium_state_injection_cases() {
        let cases = [
            (
                [0.0, 0.0, 0.0, 0.0],
                CartAction::Right,
                [0.0, 0.195_121_94, 0.0, -0.292_682_92],
            ),
            (
                [0.1, -0.2, 0.05, 0.3],
                CartAction::Left,
                [0.096, -0.395_797_64, 0.056, 0.608_023_3],
            ),
        ];

        for (state, action, expected) in cases {
            let mut env =
                CartPole::from_state(CartPoleState::try_from(state).expect("finite state"));
            let result = env.step(action);
            for (actual, expected) in result.observation.into_iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-6);
            }
            assert_eq!(result.reward, 1.0);
            assert_eq!(result.status, EpisodeStatus::Continuing);
        }
    }

    #[test]
    fn reset_seed_restarts_rng_while_none_continues_the_stream() {
        let mut env = CartPole::default();
        let seeded = env.reset(Some(7)).observation;
        assert_eq!(seeded, env.reset(Some(7)).observation);

        let next = env.reset(None).observation;
        let following = env.reset(None).observation;
        assert_ne!(next, following);
        assert!(seeded
            .into_iter()
            .chain(next)
            .chain(following)
            .all(|value| (-0.05..=0.05).contains(&value)));
    }
}
