//! End-to-end ecosystem curriculum training, evaluation, and video orchestration.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Output;

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use bevy_gym as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
use clap::Parser;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde::{Deserialize, Serialize};
use shakmaty as _;
use tokio::process::Command;

#[path = "shared/qualification.rs"]
mod qualification;

use qualification::QualificationStage as Stage;

/// Refuse another child launch above 20 GiB so one operation cannot cross 25 GiB.
const CHILD_LAUNCH_ARTIFACT_LIMIT_BYTES: u64 = 20 * 1024 * 1024 * 1024;

/// Stop the suite after any child that leaves more than 25 GiB of artifacts.
const HARD_ARTIFACT_LIMIT_BYTES: u64 = 25 * 1024 * 1024 * 1024;

/// Train and prove every ecosystem stage in transfer order.
#[derive(Debug, Parser)]
struct Options {
    /// PPO updates per curriculum stage.
    #[arg(long, default_value_t = 32)]
    iterations: usize,

    /// Fresh episodes per update; multiples of ten preserve the exact retention mix.
    #[arg(long, default_value_t = 20)]
    rollout_episodes: usize,

    /// Held-out episodes for each evaluation seed.
    #[arg(long, default_value_t = 24)]
    eval_episodes: usize,

    /// Root of independent environment, model, and action streams.
    #[arg(long, default_value_t = 157)]
    seed: u64,

    /// Root directory for stage run artifacts.
    #[arg(long, default_value = "runs/ecosystem-curriculum")]
    runs_root: PathBuf,

    /// Root directory for stage and aggregate videos.
    #[arg(long, default_value = "runs/ecosystem-curriculum/videos")]
    videos_root: PathBuf,

    /// Train and evaluate without rendering videos.
    #[arg(long)]
    skip_videos: bool,

    /// Reuse complete stage checkpoints under the deterministic run IDs.
    #[arg(long)]
    reuse_existing: bool,
}

impl Stage {
    /// Return the stable artifact and checkpoint key.
    const fn key(self) -> &'static str {
        // Keep run directory keys identical to the stage executable keys.
        match self {
            Self::Forage => "forage",
            Self::Sprint => "sprint",
            Self::Gorge => "gorge",
            Self::Survival => "survival",
            Self::Shelter => "shelter",
            Self::Competition => "competition",
            Self::PredatorPrey => "predator-prey",
            Self::Obstacles => "obstacles",
        }
    }

    /// Return the Cargo example target.
    const fn example(self) -> &'static str {
        // Map every curriculum node to its independently runnable example.
        match self {
            Self::Forage => "ecosystem-forage",
            Self::Sprint => "ecosystem-sprint",
            Self::Gorge => "ecosystem-gorge",
            Self::Survival => "ecosystem-survival",
            Self::Shelter => "ecosystem-shelter",
            Self::Competition => "ecosystem-competition",
            Self::PredatorPrey => "ecosystem-predator-prey",
            Self::Obstacles => "ecosystem-obstacles",
        }
    }

    /// Whether held-out qualification uses the single-agent episode floor.
    const fn is_single_agent(self) -> bool {
        matches!(
            self,
            Self::Forage | Self::Sprint | Self::Gorge | Self::Survival | Self::Shelter
        )
    }
}

/// Curriculum stages in direct checkpoint-transfer order.
const STAGES: [Stage; 8] = [
    Stage::Forage,
    Stage::Sprint,
    Stage::Gorge,
    Stage::Survival,
    Stage::Shelter,
    Stage::Competition,
    Stage::PredatorPrey,
    Stage::Obstacles,
];

/// One typed predator cross-play matrix cell.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct CrossPlayCell {
    /// Bunny historical role.
    bunny_role: &'static str,
    /// Fox historical role.
    fox_role: &'static str,
    /// Mean bunny return across held-out streams.
    bunny_mean_return: f64,
    /// Mean fox return across held-out streams.
    fox_mean_return: f64,
    /// Mean bunny healthy lifetime across held-out streams.
    bunny_mean_lifetime: f64,
    /// Mean fox healthy lifetime across held-out streams.
    fox_mean_lifetime: f64,
    /// Independent held-out streams represented.
    held_out_seed_count: usize,
}

/// Complete typed predator cross-play evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct CrossPlayReport {
    /// Required minimum healthy-lifetime score relative to best-versus-best.
    required_minimum_score_fraction: f64,
    /// Observed worst current-policy healthy-lifetime score ratio.
    observed_minimum_score_fraction: f64,
    /// Complete ordered opponent matrix.
    cells: Vec<CrossPlayCell>,
    /// Whether every matrix gate passed.
    passed: bool,
}

/// Persisted stage qualification fields consumed by retention checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct QualificationReport {
    /// Curriculum stage key.
    stage: String,
    /// Independent held-out streams represented.
    held_out_seed_count: usize,
    /// Exact typed step-zero gate inputs.
    step_zero: qualification::StageEvidence,
    /// Exact typed selected-checkpoint gate inputs.
    candidate: qualification::StageEvidence,
    /// Step-zero bunny return.
    worst_mean_return: f64,
    /// Selected bunny return.
    best_mean_return: f64,
    /// Minimum absolute return improvement.
    minimum_return_gain: f64,
    /// Observed absolute return improvement.
    observed_return_gain: f64,
    /// Step-zero bunny lifetime.
    worst_mean_lifetime: f64,
    /// Selected bunny lifetime.
    best_mean_lifetime: f64,
    /// Optional required relative bunny lifetime gain.
    minimum_bunny_lifetime_gain_fraction: Option<f64>,
    /// Minimum retained predecessor-behavior fraction for later stages.
    retention_fraction: f64,
    /// Stable stdout metric key used for behavior retention.
    behavior_gate: String,
    /// Absolute behavior floor.
    behavior_threshold: f64,
    /// Selected behavior value.
    behavior_value: f64,
    /// Ordered stage gates applied to the held-out evidence.
    gates: Vec<qualification::GateResult>,
    /// Whether every qualification gate passed.
    passed: bool,
}

/// One earlier-environment retention result.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct RetentionRow {
    /// Current training stage.
    trained_through: String,
    /// Earlier evaluation environment.
    environment: String,
    /// Earlier selected return.
    baseline_best_return: f64,
    /// Observed current-policy return.
    retained_return: f64,
    /// Earlier behavior metric key.
    behavior_gate: String,
    /// Earlier selected behavior value.
    baseline_behavior: f64,
    /// Minimum retained behavior value.
    minimum_retained_behavior: f64,
    /// Observed current-policy behavior value.
    retained_behavior: f64,
    /// Independent held-out streams represented.
    held_out_seed_count: usize,
    /// Whether this retention row passed.
    passed: bool,
}

/// Complete earlier-environment retention matrix.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct RetentionReport {
    /// Current training stage.
    trained_through: String,
    /// Relative predecessor-behavior floor applied to every row.
    retention_fraction: f64,
    /// Ordered earlier-environment results.
    rows: Vec<RetentionRow>,
    /// Whether every retention row passed.
    passed: bool,
}

/// Equal-budget full-ecosystem control evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct ControlReport {
    /// Transferred curriculum run directory.
    transferred_run: PathBuf,
    /// From-scratch control run directory.
    control_run: PathBuf,
    /// Equal PPO update count.
    equal_training_iterations: usize,
    /// Equal fresh episodes per PPO update.
    equal_rollout_episodes: usize,
    /// Independent held-out streams represented.
    held_out_seed_count: usize,
    /// Transferred bunny lifetime.
    transferred_bunny_mean_lifetime: f64,
    /// Control bunny lifetime.
    control_bunny_mean_lifetime: f64,
    /// Transferred fox lifetime.
    transferred_fox_mean_lifetime: f64,
    /// Control fox lifetime.
    control_fox_mean_lifetime: f64,
    /// Whether both species beat the control.
    passed: bool,
}

/// Typed `ffprobe` stream fields required by the media contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VideoProbeStream {
    /// Encoded width in pixels.
    width: u32,
    /// Encoded height in pixels.
    height: u32,
    /// Rational encoded frame rate.
    r_frame_rate: String,
    /// Encoded pixel format.
    pix_fmt: String,
    /// Reported frame count when the container exposes it.
    nb_frames: Option<String>,
}

/// Typed `ffprobe` container fields required by the media contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VideoProbeFormat {
    /// Encoded duration in decimal seconds.
    duration: String,
}

/// Typed subset of one `ffprobe` JSON response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VideoProbe {
    /// Video streams selected by the command.
    streams: Vec<VideoProbeStream>,
    /// Container-level duration.
    format: VideoProbeFormat,
}

/// One stage row in the aggregate-video manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AggregateStageVideo {
    /// Curriculum stage key.
    stage: String,
    /// Complete stage progression video.
    progression_video: PathBuf,
    /// SHA-256 of the progression video.
    progression_video_sha256: String,
    /// Best-checkpoint introduction duration.
    best_intro_seconds: u32,
    /// Complete progression duration.
    progression_seconds: u32,
}

/// Complete aggregate-video provenance and media evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AggregateVideoManifest {
    /// Aggregate MP4 path.
    output: PathBuf,
    /// SHA-256 of the aggregate MP4.
    output_sha256: String,
    /// Encoded width in pixels.
    width: u32,
    /// Encoded height in pixels.
    height: u32,
    /// Encoded frames per second.
    frames_per_second: u32,
    /// Encoded pixel format.
    pixel_format: String,
    /// Exact aggregate duration.
    total_seconds: u32,
    /// Human-readable ordered layout.
    layout: String,
    /// Ordered stage components.
    stages: Vec<AggregateStageVideo>,
    /// Typed `ffprobe` evidence.
    ffprobe: VideoProbe,
}

/// One qualified lineage in the suite manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SuiteLineage {
    /// Independent training seed.
    training_seed: u64,
    /// Ordered stage run directories.
    stage_run_directories: Vec<PathBuf>,
}

/// Complete three-lineage curriculum manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SuiteManifest {
    /// Independent training lineages.
    training_seed_count: usize,
    /// Held-out streams per stage.
    held_out_seed_count_per_stage: usize,
    /// Ordered stage count.
    stage_count: usize,
    /// Qualified lineages.
    lineages: Vec<SuiteLineage>,
    /// Whether all required lineages qualified.
    passed: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

/// Run the ordered curriculum inside one process and filesystem runtime.
async fn run() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    if options.rollout_episodes == 0 || !options.rollout_episodes.is_multiple_of(10) {
        return Err("--rollout-episodes must be a positive multiple of ten".into());
    }
    tokio::fs::create_dir_all(&options.runs_root).await?;
    tokio::fs::create_dir_all(&options.videos_root).await?;
    require_artifact_budget(HARD_ARTIFACT_LIMIT_BYTES).await?;

    let mut lineages = Vec::with_capacity(3);
    for seed in training_seeds(options.seed) {
        lineages.push(run_lineage(&options, seed).await?);
    }

    if !options.skip_videos {
        let selected = lineages.first().ok_or("curriculum produced no lineages")?;
        let videos = render_stage_videos(&options, &selected.run_dirs).await?;
        assemble_aggregate(&options.videos_root, &videos).await?;
    }
    write_suite_manifest(&options, &lineages).await?;
    Ok(())
}

/// One complete eight-stage lineage that passed every stage and retention gate.
#[derive(Debug)]
struct QualifiedLineage {
    /// Independent training root used by every stage in this lineage.
    seed: u64,

    /// Ordered qualifying stage run directories.
    run_dirs: Vec<PathBuf>,
}

/// Train, evaluate, and retain one complete seed lineage.
async fn run_lineage(options: &Options, seed: u64) -> Result<QualifiedLineage, Box<dyn Error>> {
    let lineage_root = options.runs_root.join(format!("seed-{seed}"));
    tokio::fs::create_dir_all(&lineage_root).await?;

    let mut predecessor = None;
    let mut run_dirs = Vec::with_capacity(STAGES.len());
    for (index, stage) in STAGES.into_iter().enumerate() {
        let run_id = format!("{:02}-{}-seed{seed}", index + 1, stage.key());
        let run_dir = lineage_root
            .join(format!("ecosystem-{}-ppo", stage.key()))
            .join(&run_id);
        let best_checkpoint = run_dir.join("best.mpk");
        if options.reuse_existing && best_checkpoint.is_file() {
            require_stage_artifacts(&run_dir)?;
        } else {
            train_stage(
                stage,
                options,
                seed,
                &lineage_root,
                &run_id,
                predecessor.as_deref(),
            )
            .await?;
        }
        let qualification = run_dir.join("qualification/qualification.json");
        if !(options.reuse_existing && qualification.is_file()) {
            evaluate_stage(stage, options, seed, &run_dir).await?;
        }
        let cross_play = if stage == Stage::PredatorPrey {
            Some(evaluate_predator_cross_play(options, seed, &run_dir).await?)
        } else {
            None
        };
        let control = if stage == Stage::Obstacles {
            Some(evaluate_obstacles_control(options, seed, &lineage_root, &run_dir).await?)
        } else {
            None
        };
        run_dirs.push(run_dir);
        let retention = evaluate_retention(stage, options, seed, &run_dirs).await?;
        finalize_stage_qualification(
            stage,
            run_dirs.last().ok_or("current curriculum run is missing")?,
            cross_play.as_ref(),
            control.as_ref(),
            &retention,
        )
        .await?;
        predecessor = run_dirs.last().map(|directory| directory.join("best.mpk"));
    }
    Ok(QualifiedLineage { seed, run_dirs })
}

/// Require every checkpoint needed by qualification and progression video sampling.
fn require_stage_artifacts(run_dir: &Path) -> Result<(), Box<dyn Error>> {
    for path in [
        run_dir.join("best.mpk"),
        run_dir.join("best.config.json"),
        run_dir.join("checkpoints/step-000000.mpk"),
        run_dir.join("checkpoints/step-000000.config.json"),
    ] {
        if !path.is_file() {
            return Err(format!("reused stage lacks required artifact: {}", path.display()).into());
        }
    }
    Ok(())
}

/// Train one stage from its direct predecessor checkpoint.
async fn train_stage(
    stage: Stage,
    options: &Options,
    seed: u64,
    runs_root: &Path,
    run_id: &str,
    predecessor: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let iterations = options.iterations.to_string();
    let rollout_episodes = options.rollout_episodes.to_string();
    let eval_episodes = options.eval_episodes.to_string();
    let seed = seed.to_string();
    let mut command = cargo_example(stage.example(), false);
    command.args([
        "train",
        "--iterations",
        &iterations,
        "--rollout-episodes",
        &rollout_episodes,
        "--eval-episodes",
        &eval_episodes,
        "--eval-interval",
        "1",
        "--seed",
        &seed,
        "--run-id",
        run_id,
        "--runs-root",
    ]);
    command.arg(runs_root);
    if let Some(checkpoint) = predecessor {
        command.arg("--resume").arg(checkpoint);
    }
    let result = run_inherited(&mut command, "stage training").await;
    result
}

/// Evaluate one checkpoint on three disjoint deterministic seed streams.
async fn evaluate_stage(
    stage: Stage,
    options: &Options,
    training_seed: u64,
    run_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let evidence_dir = run_dir.join("qualification");
    tokio::fs::create_dir_all(&evidence_dir).await?;
    let mut evaluations = Vec::with_capacity(if stage == Stage::Forage { 9 } else { 6 });
    let episodes = qualification_episodes(stage, options.eval_episodes).to_string();
    for (checkpoint_kind, checkpoint) in [
        ("worst", run_dir.join("checkpoints/step-000000.mpk")),
        ("best", run_dir.join("best.mpk")),
    ] {
        for (index, seed) in held_out_seeds(training_seed).into_iter().enumerate() {
            let seed = seed.to_string();
            let mut command = cargo_example(stage.example(), false);
            command
                .arg("eval")
                .arg("--checkpoint")
                .arg(&checkpoint)
                .args(["--episodes", &episodes, "--seed", &seed]);
            let output = run_captured(&mut command, "held-out evaluation").await?;
            let stdout = String::from_utf8(output.stdout)?;
            let path = evidence_dir.join(format!("{checkpoint_kind}-seed-{}.txt", index + 1));
            let mut evidence = Vec::with_capacity(stdout.len() + output.stderr.len());
            evidence.extend_from_slice(stdout.as_bytes());
            evidence.extend_from_slice(&output.stderr);
            tokio::fs::write(path, evidence).await?;
            evaluations.push((checkpoint_kind, stdout));
        }
    }
    if stage == Stage::Forage {
        for (index, seed) in held_out_seeds(training_seed).into_iter().enumerate() {
            let seed = seed.to_string();
            let mut command = cargo_example(stage.example(), false);
            command
                .arg("eval")
                .arg("--checkpoint")
                .arg(run_dir.join("best.mpk"))
                .args([
                    "--episodes",
                    &episodes,
                    "--seed",
                    &seed,
                    "--food-perception",
                    "ablated",
                ]);
            let output = run_captured(&mut command, "forage food-perception ablation").await?;
            let stdout = String::from_utf8(output.stdout)?;
            tokio::fs::write(
                evidence_dir.join(format!("ablated-seed-{}.txt", index + 1)),
                &stdout,
            )
            .await?;
            evaluations.push(("ablated", stdout));
        }
    }
    qualify_stage(stage, &evaluations, &evidence_dir).await?;
    Ok(())
}

/// One immutable historical policy role used by predator cross-play.
#[derive(Debug, Clone)]
struct HistoricalPolicy {
    /// Stable role label stored in cross-play evidence.
    role: &'static str,

    /// Bunny policy checkpoint.
    bunny: PathBuf,

    /// Synchronized fox policy checkpoint.
    fox: PathBuf,
}

/// Evaluate every early, middle, and selected predator pairing on held-out seeds.
async fn evaluate_predator_cross_play(
    options: &Options,
    training_seed: u64,
    run_dir: &Path,
) -> Result<CrossPlayReport, Box<dyn Error>> {
    let policies = historical_policies(run_dir).await?;
    let evidence_dir = run_dir.join("qualification");
    let mut cells = Vec::with_capacity(policies.len() * policies.len());
    let episodes = qualification_episodes(Stage::PredatorPrey, options.eval_episodes).to_string();
    for bunny in &policies {
        for fox in &policies {
            let mut evaluations = Vec::with_capacity(3);
            for seed in held_out_seeds(training_seed) {
                let seed = seed.to_string();
                let mut command = cargo_example("ecosystem-predator-prey", false);
                command
                    .arg("eval")
                    .arg("--checkpoint")
                    .arg(&bunny.bunny)
                    .arg("--fox-checkpoint")
                    .arg(&fox.fox)
                    .arg("--environment-checkpoint")
                    .arg(run_dir.join("best.mpk"))
                    .args(["--episodes", &episodes, "--seed", &seed]);
                let output = run_captured(&mut command, "predator cross-play").await?;
                evaluations.push(("cross-play", String::from_utf8(output.stdout)?));
            }
            cells.push(CrossPlayCell {
                bunny_role: bunny.role,
                fox_role: fox.role,
                bunny_mean_return: mean_evaluation_metric(
                    &evaluations,
                    "cross-play",
                    "bunny_mean_return",
                )?,
                fox_mean_return: mean_evaluation_metric(
                    &evaluations,
                    "cross-play",
                    "fox_mean_return",
                )?,
                bunny_mean_lifetime: mean_evaluation_metric(
                    &evaluations,
                    "cross-play",
                    "bunny_mean_lifetime",
                )?,
                fox_mean_lifetime: mean_evaluation_metric(
                    &evaluations,
                    "cross-play",
                    "fox_mean_lifetime",
                )?,
                held_out_seed_count: 3,
            });
        }
    }

    let current = cells
        .iter()
        .find(|cell| cell.bunny_role == "best" && cell.fox_role == "best")
        .ok_or("cross-play matrix lacks best-versus-best cell")?;
    let current_bunny = current.bunny_mean_lifetime;
    let current_fox = current.fox_mean_lifetime;
    if current_bunny <= 0.0 || current_fox <= 0.0 {
        return Err("best-versus-best cross-play lifetimes must be positive".into());
    }
    let mut minimum_ratio = f64::INFINITY;
    for cell in &cells {
        if cell.bunny_role == "best" {
            minimum_ratio = minimum_ratio.min(cell.bunny_mean_lifetime / current_bunny);
        }
        if cell.fox_role == "best" {
            minimum_ratio = minimum_ratio.min(cell.fox_mean_lifetime / current_fox);
        }
    }
    let report = CrossPlayReport {
        required_minimum_score_fraction: 0.70,
        observed_minimum_score_fraction: minimum_ratio,
        cells,
        passed: minimum_ratio >= 0.70,
    };
    let mut output = serde_json::to_vec_pretty(&report)?;
    output.push(b'\n');
    tokio::fs::write(evidence_dir.join("cross-play.json"), output).await?;
    Ok(report)
}

/// Train and compare an equal-budget from-scratch full-ecosystem control.
async fn evaluate_obstacles_control(
    options: &Options,
    training_seed: u64,
    lineage_root: &Path,
    transferred_run: &Path,
) -> Result<ControlReport, Box<dyn Error>> {
    let control_run_id = format!("08-obstacles-control-seed{training_seed}");
    let control_run = lineage_root
        .join("ecosystem-obstacles-ppo")
        .join(&control_run_id);
    if options.reuse_existing && control_run.join("best.mpk").is_file() {
        require_stage_artifacts(&control_run)?;
    } else {
        train_stage(
            Stage::Obstacles,
            options,
            training_seed,
            lineage_root,
            &control_run_id,
            None,
        )
        .await?;
    }

    let evidence_dir = transferred_run.join("qualification");
    let episodes = qualification_episodes(Stage::Obstacles, options.eval_episodes).to_string();
    let mut transferred = Vec::with_capacity(3);
    let mut control = Vec::with_capacity(3);
    for (index, seed) in held_out_seeds(training_seed).into_iter().enumerate() {
        let seed = seed.to_string();
        for (kind, checkpoint, environment, evaluations) in [
            (
                "transferred",
                transferred_run.join("best.mpk"),
                transferred_run.join("best.mpk"),
                &mut transferred,
            ),
            (
                "control",
                control_run.join("best.mpk"),
                control_run.join("best.mpk"),
                &mut control,
            ),
        ] {
            let mut command = cargo_example("ecosystem-obstacles", false);
            command
                .arg("eval")
                .arg("--checkpoint")
                .arg(checkpoint)
                .arg("--environment-checkpoint")
                .arg(environment)
                .args(["--episodes", &episodes, "--seed", &seed]);
            let output = run_captured(&mut command, "obstacles control evaluation").await?;
            let stdout = String::from_utf8(output.stdout)?;
            tokio::fs::write(
                evidence_dir.join(format!("{kind}-control-seed-{}.txt", index + 1)),
                &stdout,
            )
            .await?;
            evaluations.push((kind, stdout));
        }
    }

    let transferred_bunny =
        mean_evaluation_metric(&transferred, "transferred", "bunny_mean_lifetime")?;
    let transferred_fox = mean_evaluation_metric(&transferred, "transferred", "fox_mean_lifetime")?;
    let control_bunny = mean_evaluation_metric(&control, "control", "bunny_mean_lifetime")?;
    let control_fox = mean_evaluation_metric(&control, "control", "fox_mean_lifetime")?;
    let passed = require_control_gain(
        transferred_bunny,
        transferred_fox,
        control_bunny,
        control_fox,
    )
    .is_ok();

    let report = ControlReport {
        transferred_run: transferred_run.to_path_buf(),
        control_run,
        equal_training_iterations: options.iterations,
        equal_rollout_episodes: options.rollout_episodes,
        held_out_seed_count: 3,
        transferred_bunny_mean_lifetime: transferred_bunny,
        control_bunny_mean_lifetime: control_bunny,
        transferred_fox_mean_lifetime: transferred_fox,
        control_fox_mean_lifetime: control_fox,
        passed,
    };
    let mut output = serde_json::to_vec_pretty(&report)?;
    output.push(b'\n');
    tokio::fs::write(evidence_dir.join("from-scratch-control.json"), output).await?;
    Ok(report)
}

/// Require strict healthy-lifetime improvement over both control policies.
fn require_control_gain(
    transferred_bunny: f64,
    transferred_fox: f64,
    control_bunny: f64,
    control_fox: f64,
) -> Result<(), Box<dyn Error>> {
    if transferred_bunny <= control_bunny {
        return Err(format!(
            "transferred bunny lifetime {transferred_bunny} did not exceed control {control_bunny}"
        )
        .into());
    }
    if transferred_fox <= control_fox {
        return Err(format!(
            "transferred fox lifetime {transferred_fox} did not exceed control {control_fox}"
        )
        .into());
    }
    Ok(())
}

/// Resolve worst, early, middle, and best synchronized policy pairs.
async fn historical_policies(run_dir: &Path) -> Result<Vec<HistoricalPolicy>, Box<dyn Error>> {
    let mut directory = tokio::fs::read_dir(run_dir.join("checkpoints")).await?;
    let mut steps = Vec::new();
    loop {
        let next_entry = directory.next_entry().await?;
        let Some(entry) = next_entry else {
            break;
        };
        let path = entry.path();
        let is_bunny_step = path.extension().and_then(|value| value.to_str()) == Some("mpk")
            && path
                .file_stem()
                .and_then(|value| value.to_str())
                .is_some_and(|stem| stem.starts_with("step-") && !stem.ends_with("-fox"));
        if is_bunny_step {
            steps.push(path);
        }
    }
    steps.sort();
    if steps.len() < 3 {
        return Err("predator cross-play requires at least three step checkpoints".into());
    }
    let worst = steps
        .first()
        .cloned()
        .ok_or("predator cross-play lacks a worst checkpoint")?;
    let early = steps
        .get(steps.len() / 3)
        .cloned()
        .ok_or("predator cross-play lacks an early checkpoint")?;
    let middle = steps
        .get(steps.len() * 2 / 3)
        .cloned()
        .ok_or("predator cross-play lacks a middle checkpoint")?;
    let roles = [
        ("worst", worst),
        ("early", early),
        ("middle", middle),
        ("best", run_dir.join("best.mpk")),
    ];
    roles
        .into_iter()
        .map(|(role, bunny)| {
            let fox = fox_checkpoint(&bunny)?;
            Ok(HistoricalPolicy { role, bunny, fox })
        })
        .collect()
}

/// Resolve and require the synchronized fox checkpoint beside one bunny policy.
fn fox_checkpoint(bunny: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let stem = bunny
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or("bunny checkpoint filename is not UTF-8")?;
    let fox = bunny.with_file_name(format!("{stem}-fox.mpk"));
    if !fox.is_file() {
        return Err(format!("missing synchronized fox checkpoint: {}", fox.display()).into());
    }
    Ok(fox)
}

/// Evaluate the current policy on every completed environment and fail on forgetting.
async fn evaluate_retention(
    current: Stage,
    options: &Options,
    training_seed: u64,
    run_dirs: &[PathBuf],
) -> Result<RetentionReport, Box<dyn Error>> {
    let retention_fraction = qualification_contract(current).retention_fraction;
    if run_dirs.len() <= 1 {
        return Ok(RetentionReport {
            trained_through: current.key().to_owned(),
            retention_fraction,
            rows: Vec::new(),
            passed: true,
        });
    }
    let current_run = run_dirs.last().ok_or("current curriculum run is missing")?;
    let checkpoint = current_run.join("best.mpk");
    let evidence_dir = current_run.join("retention");
    tokio::fs::create_dir_all(&evidence_dir).await?;
    let mut rows = Vec::with_capacity(run_dirs.len() - 1);

    for (prior, prior_run) in STAGES.into_iter().zip(run_dirs).take(run_dirs.len() - 1) {
        let qualification: QualificationReport = serde_json::from_slice(
            &tokio::fs::read(prior_run.join("qualification/qualification.json")).await?,
        )?;
        let baseline_return = qualification.best_mean_return;
        let behavior_key = qualification.behavior_gate;
        let baseline_behavior = qualification.behavior_value;
        let mut evaluations = Vec::with_capacity(3);
        let episodes = qualification_episodes(prior, options.eval_episodes).to_string();
        for (seed_index, seed) in held_out_seeds(training_seed).into_iter().enumerate() {
            let seed = seed.to_string();
            let mut command = cargo_example(prior.example(), false);
            command
                .arg("eval")
                .arg("--checkpoint")
                .arg(&checkpoint)
                .arg("--environment-checkpoint")
                .arg(prior_run.join("best.mpk"))
                .args(["--episodes", &episodes, "--seed", &seed]);
            let output = run_captured(&mut command, "retention evaluation").await?;
            let stdout = String::from_utf8(output.stdout)?;
            let path = evidence_dir.join(format!(
                "{}-on-{}-seed-{}.txt",
                current.key(),
                prior.key(),
                seed_index + 1
            ));
            let mut evidence = Vec::with_capacity(stdout.len() + output.stderr.len());
            evidence.extend_from_slice(stdout.as_bytes());
            evidence.extend_from_slice(&output.stderr);
            tokio::fs::write(path, evidence).await?;
            evaluations.push(("retained", stdout));
        }
        let retained_return =
            mean_evaluation_metric(&evaluations, "retained", "bunny_mean_return")?;
        let retained_behavior = mean_evaluation_metric(&evaluations, "retained", &behavior_key)?;
        let minimum_behavior = baseline_behavior * retention_fraction;
        let passed =
            retention_gate_passes(retained_behavior, baseline_behavior, retention_fraction);
        rows.push(RetentionRow {
            trained_through: current.key().to_owned(),
            environment: prior.key().to_owned(),
            baseline_best_return: baseline_return,
            retained_return,
            behavior_gate: behavior_key,
            baseline_behavior,
            minimum_retained_behavior: minimum_behavior,
            retained_behavior,
            held_out_seed_count: 3,
            passed,
        });
    }
    let report = RetentionReport {
        trained_through: current.key().to_owned(),
        retention_fraction,
        passed: rows.iter().all(|row| row.passed),
        rows,
    };
    let mut output = serde_json::to_vec_pretty(&report)?;
    output.push(b'\n');
    tokio::fs::write(evidence_dir.join("retention-matrix.json"), output).await?;
    Ok(report)
}

/// Combine core, retention, cross-play, and control evidence into one final decision.
async fn finalize_stage_qualification(
    stage: Stage,
    run_dir: &Path,
    cross_play: Option<&CrossPlayReport>,
    control: Option<&ControlReport>,
    retention: &RetentionReport,
) -> Result<(), Box<dyn Error>> {
    let path = run_dir.join("qualification/qualification.json");
    let mut report: QualificationReport = serde_json::from_slice(&tokio::fs::read(&path).await?)?;
    let retention_evidence = retention
        .rows
        .iter()
        .map(|row| {
            Ok(qualification::RetentionEvidence {
                stage: stage_from_key(&row.environment)?,
                passed: row.passed,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let from_scratch = control.map(|evidence| qualification::StageEvidence {
        bunny_lifetime: evidence.control_bunny_mean_lifetime,
        fox_lifetime: evidence.control_fox_mean_lifetime,
        ..qualification::StageEvidence::default()
    });
    let decision = qualification::qualify(qualification::QualificationInput {
        stage,
        candidate: report.candidate,
        step_zero: report.step_zero,
        from_scratch,
        minimum_cross_play_ratio: cross_play
            .map(|evidence| evidence.observed_minimum_score_fraction),
        retention: &retention_evidence,
    });
    report.gates = decision.gates;
    report.passed = decision.passed;
    let failed = report
        .gates
        .iter()
        .filter(|gate| !gate.passed)
        .map(|gate| gate.gate.clone())
        .collect::<Vec<_>>();
    let mut output = serde_json::to_vec_pretty(&report)?;
    output.push(b'\n');
    tokio::fs::write(path, output).await?;
    if !report.passed {
        return Err(format!("{} failed qualification gates: {failed:?}", stage.key()).into());
    }
    Ok(())
}

/// Parse one persisted stage key into the closed qualification vocabulary.
fn stage_from_key(key: &str) -> Result<Stage, Box<dyn Error>> {
    STAGES
        .into_iter()
        .find(|stage| stage.key() == key)
        .ok_or_else(|| format!("unknown curriculum stage {key:?}").into())
}

/// Apply one stage's retention fraction to its predecessor behavior value.
fn retention_gate_passes(
    retained_behavior: f64,
    baseline_behavior: f64,
    retention_fraction: f64,
) -> bool {
    retained_behavior >= baseline_behavior * retention_fraction
}

/// Return the three disjoint deterministic qualification streams.
const fn held_out_seeds(root: u64) -> [u64; 3] {
    [
        root ^ 0x4556_414c_5f41,
        root ^ 0x4556_414c_5f42,
        root ^ 0x4556_414c_5f43,
    ]
}

/// Expand one CLI seed into the three independent training lineages.
const fn training_seeds(root: u64) -> [u64; 3] {
    [root, root.wrapping_add(6), root.wrapping_add(112)]
}

/// Apply the declared held-out sample floor for one curriculum stage.
const fn qualification_episodes(stage: Stage, configured: usize) -> usize {
    let minimum = if stage.is_single_agent() { 100 } else { 50 };
    if configured < minimum {
        minimum
    } else {
        configured
    }
}

/// Quantitative gates that apply to one complete training lineage.
#[derive(Debug, Clone, Copy, PartialEq)]
struct QualificationContract {
    /// Minimum fraction of bunnies that must both eat and drink.
    minimum_food_and_water_fraction: Option<f64>,

    /// Minimum relative bunny lifetime gain from step zero.
    minimum_bunny_lifetime_gain: Option<f64>,

    /// Minimum relative fox lifetime gain from step zero.
    minimum_fox_lifetime_gain: Option<f64>,

    /// Minimum distinct bunny identities that must consume both resources.
    minimum_resource_consumers: Option<u32>,

    /// Maximum stable-identity lifetime advantage over the population mean.
    maximum_identity_advantage: Option<f64>,

    /// Minimum held-out agent contact observations.
    minimum_contact_events: Option<u64>,

    /// Minimum retained behavior relative to the prior certified policy.
    retention_fraction: f64,
}

/// Return the fixed plan contract for one stage key.
const fn qualification_contract(stage: Stage) -> QualificationContract {
    // Keep shared orchestration thresholds explicit for every curriculum node.
    match stage {
        Stage::Forage => QualificationContract {
            minimum_food_and_water_fraction: None,
            minimum_bunny_lifetime_gain: None,
            minimum_fox_lifetime_gain: None,
            minimum_resource_consumers: None,
            maximum_identity_advantage: None,
            minimum_contact_events: None,
            retention_fraction: 1.0,
        },
        Stage::Sprint | Stage::Gorge | Stage::Shelter => QualificationContract {
            minimum_food_and_water_fraction: None,
            minimum_bunny_lifetime_gain: None,
            minimum_fox_lifetime_gain: None,
            minimum_resource_consumers: None,
            maximum_identity_advantage: None,
            minimum_contact_events: None,
            retention_fraction: 0.90,
        },
        Stage::Survival => QualificationContract {
            minimum_food_and_water_fraction: Some(0.80),
            minimum_bunny_lifetime_gain: Some(0.30),
            minimum_fox_lifetime_gain: None,
            minimum_resource_consumers: None,
            maximum_identity_advantage: None,
            minimum_contact_events: None,
            retention_fraction: 0.90,
        },
        Stage::Competition => QualificationContract {
            minimum_food_and_water_fraction: None,
            minimum_bunny_lifetime_gain: Some(0.15),
            minimum_fox_lifetime_gain: None,
            minimum_resource_consumers: Some(4),
            maximum_identity_advantage: Some(0.20),
            minimum_contact_events: Some(1),
            retention_fraction: 0.85,
        },
        Stage::PredatorPrey => QualificationContract {
            minimum_food_and_water_fraction: None,
            minimum_bunny_lifetime_gain: Some(0.15),
            minimum_fox_lifetime_gain: Some(0.15),
            minimum_resource_consumers: None,
            maximum_identity_advantage: None,
            minimum_contact_events: None,
            retention_fraction: 0.85,
        },
        Stage::Obstacles => QualificationContract {
            minimum_food_and_water_fraction: None,
            minimum_bunny_lifetime_gain: None,
            minimum_fox_lifetime_gain: None,
            minimum_resource_consumers: None,
            maximum_identity_advantage: None,
            minimum_contact_events: None,
            retention_fraction: 0.85,
        },
    }
}

/// Enforce progression and stage-specific held-out behavior gates.
async fn qualify_stage(
    stage: Stage,
    evaluations: &[(&str, String)],
    evidence_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    // Gather candidate, baseline, control, and retention evidence before gating.
    let contract = qualification_contract(stage);
    let worst_return = mean_evaluation_metric(evaluations, "worst", "bunny_mean_return")?;
    let best_return = mean_evaluation_metric(evaluations, "best", "bunny_mean_return")?;
    let step_zero = stage_evidence(stage, evaluations, "worst")?;
    let candidate = stage_evidence(stage, evaluations, "best")?;
    let decision = qualification::qualify(qualification::QualificationInput {
        stage,
        candidate,
        step_zero,
        from_scratch: None,
        minimum_cross_play_ratio: None,
        retention: &[],
    });
    let gates = decision
        .gates
        .into_iter()
        .filter(|gate| !is_external_gate(&gate.gate))
        .collect::<Vec<_>>();
    let failed = gates
        .iter()
        .filter(|gate| !gate.passed)
        .map(|gate| gate.gate.clone())
        .collect::<Vec<_>>();
    let passed = failed.is_empty();
    let behavior_metric = match stage {
        Stage::Forage => ("bunny_food_fraction", 0.90),
        Stage::Sprint | Stage::Gorge => ("bunny_food_fraction", 0.80),
        Stage::Survival => ("bunny_food_and_water", 0.80),
        Stage::Shelter => ("bunny_shelter_before_critical", 0.80),
        Stage::Competition => ("bunny_resource_consumers", 4.0),
        Stage::PredatorPrey => ("predation_episode_fraction", 0.10),
        Stage::Obstacles => ("bunny_mean_lifetime", 10.0),
    };
    let behavior_value = mean_evaluation_metric(evaluations, "best", behavior_metric.0)?;
    let minimum_return_gain = 0.0;
    let report = QualificationReport {
        stage: stage.key().to_owned(),
        held_out_seed_count: 3,
        step_zero,
        candidate,
        worst_mean_return: worst_return,
        best_mean_return: best_return,
        minimum_return_gain,
        observed_return_gain: best_return - worst_return,
        worst_mean_lifetime: step_zero.bunny_lifetime,
        best_mean_lifetime: candidate.bunny_lifetime,
        minimum_bunny_lifetime_gain_fraction: contract.minimum_bunny_lifetime_gain,
        retention_fraction: contract.retention_fraction,
        behavior_gate: behavior_metric.0.to_owned(),
        behavior_threshold: behavior_metric.1,
        behavior_value,
        gates,
        passed,
    };
    let mut output = serde_json::to_vec_pretty(&report)?;
    output.push(b'\n');
    tokio::fs::write(evidence_dir.join("qualification.json"), output).await?;
    if !passed {
        return Err(format!("{} failed qualification gates: {failed:?}", stage.key()).into());
    }
    Ok(())
}

/// Build the exact typed gate input from one held-out checkpoint class.
fn stage_evidence(
    stage: Stage,
    evaluations: &[(&str, String)],
    checkpoint_kind: &str,
) -> Result<qualification::StageEvidence, Box<dyn Error>> {
    // Project persisted evaluation metrics into the shared qualification model.
    let ablated = if stage == Stage::Forage && checkpoint_kind == "best" {
        mean_evaluation_metric(evaluations, "ablated", "bunny_food_fraction")?
    } else {
        0.0
    };
    Ok(qualification::StageEvidence {
        bunny_lifetime: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_mean_lifetime",
        )?,
        fox_lifetime: mean_evaluation_metric(evaluations, checkpoint_kind, "fox_mean_lifetime")?,
        bunny_food_fraction: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_food_fraction",
        )?,
        bunny_mean_time_to_contact: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_mean_time_to_contact",
        )?,
        bunny_food_and_water_fraction: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_food_and_water",
        )?,
        bunny_shelter_fraction: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_shelter_before_critical",
        )?,
        bunny_complete_resource_consumers: minimum_evaluation_count(
            evaluations,
            checkpoint_kind,
            "bunny_resource_consumers",
        )?,
        bunny_identity_advantage: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_index_advantage",
        )?,
        fox_food_and_water_fraction: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "fox_prey_and_water",
        )?,
        fox_shelter_fraction: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "fox_shelter_before_critical",
        )?,
        bunny_left_food_turn: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_left_food_median_turn",
        )?,
        bunny_right_food_turn: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "bunny_right_food_median_turn",
        )?,
        bunny_ablated_food_fraction: ablated,
        predation_episode_fraction: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "predation_episode_fraction",
        )?,
        collision_contacts: sum_evaluation_count(
            evaluations,
            checkpoint_kind,
            "collision_contacts",
        )?,
        contact_displacements: sum_evaluation_count(
            evaluations,
            checkpoint_kind,
            "contact_displacements",
        )?,
        unresolved_solid_penetrations: sum_evaluation_count(
            evaluations,
            checkpoint_kind,
            "unresolved_solid_penetrations",
        )?,
        thorn_damage_per_agent_minute: mean_evaluation_metric(
            evaluations,
            checkpoint_kind,
            "thorn_damage_per_agent_minute",
        )?,
        exposure_deaths: u32::try_from(sum_evaluation_count(
            evaluations,
            checkpoint_kind,
            "exposure_deaths",
        )?)?,
    })
}

/// Return whether one gate is supplied by retention, cross-play, or control evidence.
fn is_external_gate(gate: &str) -> bool {
    gate.starts_with("retention_")
        || matches!(
            gate,
            "cross_play_ratio" | "bunny_from_scratch_control" | "fox_from_scratch_control"
        )
}

/// Return the minimum exact count across one checkpoint's held-out streams.
fn minimum_evaluation_count(
    evaluations: &[(&str, String)],
    checkpoint_kind: &str,
    key: &str,
) -> Result<u32, Box<dyn Error>> {
    let counts = evaluation_counts(evaluations, checkpoint_kind, key)?;
    let minimum = counts
        .into_iter()
        .min()
        .ok_or_else(|| format!("no {checkpoint_kind} evaluations were recorded"))?;
    Ok(u32::try_from(minimum)?)
}

/// Sum exact event counts across one checkpoint's held-out streams.
fn sum_evaluation_count(
    evaluations: &[(&str, String)],
    checkpoint_kind: &str,
    key: &str,
) -> Result<u64, Box<dyn Error>> {
    evaluation_counts(evaluations, checkpoint_kind, key)?
        .into_iter()
        .try_fold(0_u64, |total, value| {
            total
                .checked_add(value)
                .ok_or_else(|| "evaluation count overflowed u64".into())
        })
}

/// Parse every exact count for one checkpoint class.
fn evaluation_counts(
    evaluations: &[(&str, String)],
    checkpoint_kind: &str,
    key: &str,
) -> Result<Vec<u64>, Box<dyn Error>> {
    let counts = evaluations
        .iter()
        .filter(|(kind, _)| *kind == checkpoint_kind)
        .map(|(_, line)| Ok(evaluation_field(line, key)?.parse()?))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    if counts.is_empty() {
        return Err(format!("no {checkpoint_kind} evaluations were recorded").into());
    }
    Ok(counts)
}

/// Average one key from one checkpoint class across held-out seed reports.
fn mean_evaluation_metric(
    evaluations: &[(&str, String)],
    checkpoint_kind: &str,
    key: &str,
) -> Result<f64, Box<dyn Error>> {
    let values = evaluations
        .iter()
        .filter(|(kind, _)| *kind == checkpoint_kind)
        .map(|(_, line)| evaluation_metric(line, key))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("no {checkpoint_kind} evaluations were recorded").into());
    }
    let count = u32::try_from(values.len())?;
    Ok(values.iter().sum::<f64>() / f64::from(count))
}

/// Parse one stable `key=value` scalar from evaluation stdout.
fn evaluation_metric(line: &str, key: &str) -> Result<f64, Box<dyn Error>> {
    Ok(evaluation_field(line, key)?.parse()?)
}

/// Return one stable scalar field without changing its lexical number type.
fn evaluation_field<'a>(line: &'a str, key: &str) -> Result<&'a str, Box<dyn Error>> {
    let prefix = format!("{key}=");
    line.split_ascii_whitespace()
        .find_map(|field| field.strip_prefix(&prefix))
        .ok_or_else(|| format!("evaluation output lacks {key}: {line}").into())
        .map(|value| value.trim_end_matches(','))
}

/// Render one exact 30-second worst-to-best video per stage.
async fn render_stage_videos(
    options: &Options,
    run_dirs: &[PathBuf],
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut videos = Vec::with_capacity(run_dirs.len());
    for (stage, run_dir) in STAGES.into_iter().zip(run_dirs) {
        let output = options.videos_root.join(format!("{}.mp4", stage.key()));
        let mut command = cargo_example(stage.example(), true);
        command
            .arg("video")
            .arg("--checkpoint")
            .arg(run_dir)
            .arg("--output")
            .arg(&output)
            .args([
                "--fps",
                "30",
                "--intro-seconds",
                "5",
                "--checkpoint-seconds",
                "5",
                "--outro-seconds",
                "15",
            ]);
        run_inherited(&mut command, "stage video").await?;
        validate_video(&output, 30).await?;
        videos.push(output);
    }
    Ok(videos)
}

/// Prefix all best checkpoints, then append all complete progression videos.
async fn assemble_aggregate(video_root: &Path, videos: &[PathBuf]) -> Result<(), Box<dyn Error>> {
    let mut best_videos = Vec::with_capacity(videos.len());
    for (index, video) in videos.iter().enumerate() {
        let best = video_root.join(format!("best-{:02}.mp4", index + 1));
        let mut ffmpeg = Command::new("ffmpeg");
        ffmpeg
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                "25",
                "-t",
                "5",
                "-i",
            ])
            .arg(video)
            .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30"])
            .arg(&best);
        run_inherited(&mut ffmpeg, "best-checkpoint excerpt").await?;
        validate_video(&best, 5).await?;
        best_videos.push(best);
    }
    let concat_inputs = aggregate_inputs(&best_videos, videos)?;

    let concat_path = video_root.join("aggregate-concat.txt");
    let mut concat = String::new();
    for path in &concat_inputs {
        let canonical = tokio::fs::canonicalize(path).await?;
        writeln!(concat, "file '{}'", canonical.display())?;
    }
    tokio::fs::write(&concat_path, concat).await?;

    let aggregate = video_root.join("ecosystem-curriculum.mp4");
    let mut ffmpeg = Command::new("ffmpeg");
    ffmpeg
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
        ])
        .arg(&concat_path)
        .args(["-c", "copy"])
        .arg(&aggregate);
    run_inherited(&mut ffmpeg, "aggregate video").await?;
    let aggregate_probe = validate_video(&aggregate, 280).await?;
    let result = write_aggregate_manifest(video_root, videos, &aggregate, aggregate_probe).await;
    result
}

/// Order every best preview before every complete stage progression video.
fn aggregate_inputs(
    best_videos: &[PathBuf],
    progression_videos: &[PathBuf],
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if best_videos.len() != STAGES.len() || progression_videos.len() != STAGES.len() {
        return Err("aggregate video requires eight best previews and eight progressions".into());
    }
    Ok(best_videos
        .iter()
        .chain(progression_videos)
        .cloned()
        .collect())
}

/// Require exact curriculum video geometry, frame rate, format, and duration.
async fn validate_video(path: &Path, expected_seconds: u32) -> Result<VideoProbe, Box<dyn Error>> {
    let mut command = Command::new("ffprobe");
    command
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,pix_fmt,nb_frames:format=duration",
            "-of",
            "json",
        ])
        .arg(path);
    let output = run_captured(&mut command, "video validation").await?;
    let probe: VideoProbe = serde_json::from_slice(&output.stdout)?;
    let stream = probe
        .streams
        .first()
        .ok_or_else(|| format!("{} has no video stream", path.display()))?;
    if stream.width != 1280
        || stream.height != 720
        || stream.r_frame_rate != "30/1"
        || stream.pix_fmt != "yuv420p"
    {
        return Err(format!("{} has invalid video profile: {stream:?}", path.display()).into());
    }
    let duration = probe.format.duration.parse::<f64>()?;
    if (duration - f64::from(expected_seconds)).abs() > 0.05 {
        return Err(format!(
            "{} duration {duration:.6}s differs from required {expected_seconds}s",
            path.display()
        )
        .into());
    }
    Ok(probe)
}

/// Persist aggregate order, hashes, and the validated media probe.
async fn write_aggregate_manifest(
    video_root: &Path,
    stage_videos: &[PathBuf],
    aggregate: &Path,
    probe: VideoProbe,
) -> Result<(), Box<dyn Error>> {
    let mut stages = Vec::with_capacity(stage_videos.len());
    for (stage, video) in STAGES.into_iter().zip(stage_videos) {
        stages.push(AggregateStageVideo {
            stage: stage.key().to_owned(),
            progression_video: video.clone(),
            progression_video_sha256: sha256(video).await?,
            best_intro_seconds: 5,
            progression_seconds: 30,
        });
    }
    let manifest = AggregateVideoManifest {
        output: aggregate.to_path_buf(),
        output_sha256: sha256(aggregate).await?,
        width: 1280,
        height: 720,
        frames_per_second: 30,
        pixel_format: "yuv420p".to_owned(),
        total_seconds: 280,
        layout:
            "eight 5-second best-checkpoint excerpts, then eight 30-second worst-to-best stage videos"
                .to_owned(),
        stages,
        ffprobe: probe,
    };
    let mut output = serde_json::to_vec_pretty(&manifest)?;
    output.push(b'\n');
    tokio::fs::write(
        video_root.join("ecosystem-curriculum-video-manifest.json"),
        output,
    )
    .await?;
    Ok(())
}

/// Persist the three-lineage stage directories only after every gate passes.
async fn write_suite_manifest(
    options: &Options,
    lineages: &[QualifiedLineage],
) -> Result<(), Box<dyn Error>> {
    let entries = lineages
        .iter()
        .map(|lineage| SuiteLineage {
            training_seed: lineage.seed,
            stage_run_directories: lineage.run_dirs.clone(),
        })
        .collect::<Vec<_>>();
    let manifest = SuiteManifest {
        training_seed_count: lineages.len(),
        held_out_seed_count_per_stage: 3,
        stage_count: STAGES.len(),
        lineages: entries,
        passed: lineages.len() == 3,
    };
    let mut output = serde_json::to_vec_pretty(&manifest)?;
    output.push(b'\n');
    tokio::fs::write(options.runs_root.join("curriculum-suite.json"), output).await?;
    Ok(())
}

/// Compute one lowercase SHA-256 using the installed coreutils command.
async fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut command = Command::new("sha256sum");
    command.arg(path);
    let output = run_captured(&mut command, "SHA-256").await?;
    String::from_utf8(output.stdout)?
        .split_ascii_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or_else(|| format!("sha256sum returned no hash for {}", path.display()).into())
}

/// Construct one Cargo example command with the requested feature profile.
fn cargo_example(example: &str, render: bool) -> Command {
    let mut command = Command::new("cargo");
    // Reuse one optimized release cache for training, evaluation, and
    // rendering so a curriculum proof cannot duplicate the Bevy dependency tree.
    command.args(["run", "--release"]);
    command.env("CARGO_INCREMENTAL", "0");
    if render {
        command.args(["--features", "render"]);
    } else {
        command.arg("--no-default-features");
    }
    command.args(["--example", example, "--"]);
    command
}

/// Run one child with live output and require success.
async fn run_inherited(command: &mut Command, operation: &str) -> Result<(), Box<dyn Error>> {
    require_artifact_budget(CHILD_LAUNCH_ARTIFACT_LIMIT_BYTES).await?;
    let status = command.status().await?;
    require_artifact_budget(HARD_ARTIFACT_LIMIT_BYTES).await?;
    if !status.success() {
        return Err(format!("{operation} failed with {status}").into());
    }
    Ok(())
}

/// Run one child with captured evidence and require success.
async fn run_captured(command: &mut Command, operation: &str) -> Result<Output, Box<dyn Error>> {
    require_artifact_budget(CHILD_LAUNCH_ARTIFACT_LIMIT_BYTES).await?;
    let output = command.output().await?;
    require_artifact_budget(HARD_ARTIFACT_LIMIT_BYTES).await?;
    if !output.status.success() {
        return Err(format!(
            "{operation} failed with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(output)
}

/// Reject another operation when build and run artifacts exceed `limit_bytes`.
async fn require_artifact_budget(limit_bytes: u64) -> Result<(), Box<dyn Error>> {
    let target_bytes = directory_bytes(Path::new("target")).await?;
    let run_bytes = directory_bytes(Path::new("runs")).await?;
    let total_bytes = target_bytes
        .checked_add(run_bytes)
        .ok_or("artifact byte count overflowed u64")?;
    if total_bytes > limit_bytes {
        return Err(format!(
            "artifact budget exceeded: target plus runs use {total_bytes} bytes; limit is {limit_bytes} bytes"
        )
        .into());
    }
    Ok(())
}

/// Count regular files below one artifact root without following symlinks.
async fn directory_bytes(root: &Path) -> Result<u64, Box<dyn Error>> {
    if !tokio::fs::try_exists(root).await? {
        return Ok(0);
    }
    let mut pending = VecDeque::from([root.to_path_buf()]);
    let mut bytes = 0_u64;
    while let Some(directory) = pending.pop_front() {
        let mut entries = tokio::fs::read_dir(directory).await?;
        loop {
            let next_entry = entries.next_entry().await?;
            let Some(entry) = next_entry else {
                break;
            };
            let metadata = tokio::fs::symlink_metadata(entry.path()).await?;
            if metadata.is_dir() {
                pending.push_back(entry.path());
            } else if metadata.is_file() {
                bytes = bytes
                    .checked_add(metadata.len())
                    .ok_or("artifact byte count overflowed u64")?;
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stage gates must preserve the quantitative contract approved in the plan.
    #[test]
    fn survival_and_competition_gates_match_the_plan() {
        let survival = qualification_contract(Stage::Survival);
        assert_eq!(survival.minimum_food_and_water_fraction, Some(0.80));
        assert_eq!(survival.minimum_bunny_lifetime_gain, Some(0.30));
        assert_eq!(survival.retention_fraction, 0.90);

        let competition = qualification_contract(Stage::Competition);
        assert_eq!(competition.minimum_bunny_lifetime_gain, Some(0.15));
        assert_eq!(competition.minimum_resource_consumers, Some(4));
        assert_eq!(competition.maximum_identity_advantage, Some(0.20));
        assert_eq!(competition.minimum_contact_events, Some(1));
        assert_eq!(competition.retention_fraction, 0.85);
    }

    /// Retention must apply the plan fraction to predecessor behavior, not return.
    #[test]
    fn retention_uses_predecessor_behavior() {
        assert!(retention_gate_passes(0.91, 1.0, 0.90));
        assert!(!retention_gate_passes(0.89, 1.0, 0.90));
    }

    /// One root must expand to three independent training lineages.
    #[test]
    fn curriculum_uses_three_training_seeds() {
        assert_eq!(training_seeds(157), [157, 163, 269]);
    }

    /// Qualification must meet the plan's single- and multi-agent sample floors.
    #[test]
    fn qualification_episode_counts_meet_stage_floors() {
        assert_eq!(qualification_episodes(Stage::Forage, 24), 100);
        assert_eq!(qualification_episodes(Stage::Gorge, 120), 120);
        assert_eq!(qualification_episodes(Stage::Competition, 24), 50);
        assert_eq!(qualification_episodes(Stage::Obstacles, 64), 64);
    }

    /// Both transferred species must exceed their equal-budget control lifetimes.
    #[test]
    fn obstacles_control_requires_both_species_to_improve() {
        assert!(require_control_gain(12.0, 14.0, 10.0, 11.0).is_ok());
        assert!(require_control_gain(10.0, 14.0, 10.0, 11.0).is_err());
        assert!(require_control_gain(12.0, 9.0, 10.0, 11.0).is_err());
    }

    /// Aggregate order must contain eight previews followed by eight progressions.
    #[test]
    fn aggregate_video_order_matches_the_timeline() {
        let best = STAGES
            .into_iter()
            .map(|stage| PathBuf::from(format!("best-{}.mp4", stage.key())))
            .collect::<Vec<_>>();
        let progressions = STAGES
            .into_iter()
            .map(|stage| PathBuf::from(format!("{}.mp4", stage.key())))
            .collect::<Vec<_>>();

        let actual = aggregate_inputs(&best, &progressions).expect("eight-stage inputs validate");

        assert_eq!(&actual[..STAGES.len()], best);
        assert_eq!(&actual[STAGES.len()..], progressions);
        assert!(aggregate_inputs(&best[..STAGES.len() - 1], &progressions).is_err());
    }

    /// Every child must reuse the release cache without incremental artifacts.
    #[test]
    fn curriculum_children_use_one_bounded_build_profile() {
        let command = cargo_example("ecosystem-forage", false);
        let standard = command.as_std();
        let arguments = standard
            .get_args()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let incremental = standard
            .get_envs()
            .find(|(name, _)| *name == "CARGO_INCREMENTAL")
            .and_then(|(_, value)| value)
            .map(|value| value.to_string_lossy().into_owned());

        assert!(arguments.iter().any(|value| value == "--release"));
        assert_eq!(incremental.as_deref(), Some("0"));
    }
}
