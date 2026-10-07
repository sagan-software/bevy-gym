//! Validate and export one ecosystem policy set into the static web manifest.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoPolicy};
use clap::Parser;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::process::Command;

#[path = "../manifest.rs"]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the native exporter shares manifest validation without browser-only readers"
    )
)]
mod manifest;

use manifest::{
    Algorithm, Architecture, AssetPath, BurnVersion, CheckpointAsset, CheckpointManifest,
    CheckpointSidecar, CheckpointSize, ConfigPath, ManifestVersion, PolicyRole, Qualification,
    SelectionSummary, Sha256Digest, SourceCommit, Stage, StageManifest,
};

/// Current fixed ecosystem actor observation width.
const LOCAL_OBSERVATION_SIZE: usize = 267;

/// Current fixed ecosystem centralized critic width.
const GLOBAL_STATE_SIZE: usize = 363;

/// Current fixed centralized critic output count.
const MAX_AGENTS: usize = 12;

/// Ecosystem action lower bounds.
const ACTION_LOW: [f32; 4] = [-1.0; 4];

/// Ecosystem action upper bounds.
const ACTION_HIGH: [f32; 4] = [1.0; 4];

/// Native checkpoint validation and static export arguments.
#[derive(Debug, Parser)]
struct Args {
    /// Ecosystem route key.
    #[arg(long)]
    stage: String,
    /// Source run containing best checkpoints, sidecars, summary, and seeds.
    #[arg(long)]
    run_dir: PathBuf,
    /// Evidence label: qualified or best-compatible-available.
    #[arg(long)]
    qualification: String,
    /// Static manifest updated by this export.
    #[arg(long, default_value = "web/checkpoints/manifest.json")]
    manifest: PathBuf,
    /// Directory receiving content-addressed MPK and sidecar files.
    #[arg(long, default_value = "web/checkpoints")]
    output_dir: PathBuf,
}

/// Root training seed recorded by the native run.
#[derive(Debug, Deserialize)]
struct RunSeeds {
    /// Root seed from which environment, action, model, and evidence seeds derive.
    root: u64,
}

/// Summary subset required to justify a static checkpoint selection.
#[derive(Debug, Deserialize)]
struct RunSummary {
    /// Curriculum stage evaluated by this run.
    stage: Stage,
    /// Native environment steps completed by the run.
    global_steps: u64,
    /// Initial bunny validation return.
    initial_bunny_mean_return: f32,
    /// Selected bunny validation return.
    best_bunny_mean_return: f32,
    /// Initial bunny validation lifetime.
    initial_bunny_mean_lifetime: f32,
    /// Selected bunny validation lifetime.
    best_bunny_mean_lifetime: f32,
    /// Selected bunny lower confidence bound.
    best_bunny_ci95_lower: f32,
    /// Selected bunny upper confidence bound.
    best_bunny_ci95_upper: f32,
    /// Selected bunny food success fraction.
    best_bunny_food_fraction: f32,
    /// Selected bunny food-and-water success fraction.
    best_bunny_food_and_water_fraction: f32,
    /// Initial fox validation return.
    initial_fox_mean_return: f32,
    /// Selected fox validation return.
    best_fox_mean_return: f32,
    /// Initial fox validation lifetime.
    initial_fox_mean_lifetime: f32,
    /// Selected fox validation lifetime.
    best_fox_mean_lifetime: f32,
    /// Selected fox lower confidence bound.
    best_fox_ci95_lower: f32,
    /// Selected fox upper confidence bound.
    best_fox_ci95_upper: f32,
    /// Selected fox prey-and-water success fraction.
    best_fox_prey_and_water_fraction: f32,
    /// Source report's evidence boundary.
    qualification: Box<str>,
}

/// Partial manifest accepted while six entries are curated one at a time.
#[derive(Debug, serde::Serialize, Deserialize)]
struct MutableManifest {
    /// Closed manifest schema version.
    manifest_version: ManifestVersion,
    /// Source commit recorded by the exporter.
    source_commit: SourceCommit,
    /// Burn record compatibility version.
    burn_version: BurnVersion,
    /// Current stage entries.
    stages: Vec<StageManifest>,
}

fn main() -> Result<(), Box<dyn Error>> {
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(run())
}

/// Validate the selected policy set and update the static manifest.
async fn run() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let stage = Stage::from_str(&args.stage)?;
    let qualification = Qualification::from_str(&args.qualification)?;
    let source_run = repository_relative_path(&args.run_dir)?;
    let bunny_path = args.run_dir.join("best.mpk");
    let bunny_config_path = args.run_dir.join("best.config.json");
    let fox_path = args.run_dir.join("best-fox.mpk");
    let fox_config_path = args.run_dir.join("best-fox.config.json");
    let bunny_profile = validate_profile(&bunny_config_path, stage).await?;
    let bunny = validate_policy(&bunny_path).await?;
    let fox = if stage.requires_fox() {
        let fox_profile = validate_profile(&fox_config_path, stage).await?;
        if fox_profile != bunny_profile {
            return Err("bunny and fox checkpoint sidecars differ".into());
        }
        Some(validate_policy(&fox_path).await?)
    } else {
        if fs::try_exists(&fox_path).await? || fs::try_exists(&fox_config_path).await? {
            return Err("single-policy stage contains an unexpected fox checkpoint".into());
        }
        None
    };
    let summary = read_summary(&args.run_dir, stage, qualification).await?;
    let seeds = read_seeds(&args.run_dir).await?;

    fs::create_dir_all(&args.output_dir).await?;
    let bunny_asset = export_asset(
        stage,
        "bunny",
        &bunny_path,
        &bunny_config_path,
        &args.output_dir,
        &source_run,
    )
    .await?;
    let fox_asset = if stage.requires_fox() {
        Some(
            export_asset(
                stage,
                "fox",
                &fox_path,
                &fox_config_path,
                &args.output_dir,
                &source_run,
            )
            .await?,
        )
    } else {
        None
    };

    // Keep both loaded policies live through the export boundary.
    drop((bunny, fox));
    let mut manifest = read_mutable_manifest(&args.manifest).await?;
    manifest.stages.retain(|entry| entry.stage != stage);
    manifest.stages.push(StageManifest {
        stage,
        title: stage.title().into(),
        checkpoint_profile: bunny_profile.checkpoint_profile,
        experiment_tuning: bunny_profile.experiment_tuning,
        algorithm: Algorithm::RecurrentPpo,
        architecture: Architecture::CURRENT,
        required_policy_roles: if stage.requires_fox() {
            vec![PolicyRole::Bunny, PolicyRole::Fox]
        } else {
            vec![PolicyRole::Bunny]
        },
        training_seed: seeds.root,
        selection_summary: summary,
        qualification,
        bunny: bunny_asset,
        fox: fox_asset,
    });
    manifest.stages.sort_by_key(|entry| entry.stage.position());
    if manifest.stages.len() == Stage::ALL.len() {
        let complete = CheckpointManifest {
            manifest_version: manifest.manifest_version,
            source_commit: manifest.source_commit.clone(),
            burn_version: manifest.burn_version,
            stages: manifest.stages.clone(),
        };
        let json = serde_json::to_string(&complete)?;
        CheckpointManifest::from_json(&json)?;
    }
    let json = serde_json::to_string_pretty(&manifest)?;
    fs::write(&args.manifest, format!("{json}\n")).await?;
    Ok(())
}

/// Read and validate the immutable companion profile.
async fn validate_profile(
    config_path: &Path,
    stage: Stage,
) -> Result<CheckpointSidecar, Box<dyn Error>> {
    let text = fs::read_to_string(config_path).await?;
    let profile = CheckpointSidecar::from_json(&text)?;
    if profile.stage != stage {
        let profile_stage = profile.stage.as_key();
        let stage_key = stage.as_key();
        let config_display = config_path.display();
        return Err(format!(
            "checkpoint stage {profile_stage:?} differs from {stage_key}: {config_display}"
        )
        .into());
    }
    Ok(profile)
}

/// Read selection evidence and enforce the requested qualification boundary.
async fn read_summary(
    run_dir: &Path,
    stage: Stage,
    qualification: Qualification,
) -> Result<SelectionSummary, Box<dyn Error>> {
    let path = run_dir.join("summary.json");
    let summary: RunSummary = serde_json::from_slice(&fs::read(&path).await?)?;
    if summary.stage != stage {
        let summary_stage = summary.stage.as_key();
        let stage_key = stage.as_key();
        return Err(format!("summary stage {summary_stage} differs from {stage_key}").into());
    }
    if qualification == Qualification::Qualified && summary.qualification.as_ref() != "qualified" {
        return Err("source summary does not contain qualified evidence".into());
    }
    Ok(SelectionSummary {
        global_steps: summary.global_steps,
        initial_bunny_mean_return: summary.initial_bunny_mean_return,
        best_bunny_mean_return: summary.best_bunny_mean_return,
        initial_bunny_mean_lifetime: summary.initial_bunny_mean_lifetime,
        best_bunny_mean_lifetime: summary.best_bunny_mean_lifetime,
        best_bunny_ci95_lower: summary.best_bunny_ci95_lower,
        best_bunny_ci95_upper: summary.best_bunny_ci95_upper,
        best_bunny_food_fraction: summary.best_bunny_food_fraction,
        best_bunny_food_and_water_fraction: summary.best_bunny_food_and_water_fraction,
        initial_fox_mean_return: summary.initial_fox_mean_return,
        best_fox_mean_return: summary.best_fox_mean_return,
        initial_fox_mean_lifetime: summary.initial_fox_mean_lifetime,
        best_fox_mean_lifetime: summary.best_fox_mean_lifetime,
        best_fox_ci95_lower: summary.best_fox_ci95_lower,
        best_fox_ci95_upper: summary.best_fox_ci95_upper,
        best_fox_prey_and_water_fraction: summary.best_fox_prey_and_water_fraction,
        qualification_evidence: summary.qualification,
    })
}

/// Read the root seed required by the release provenance contract.
async fn read_seeds(run_dir: &Path) -> Result<RunSeeds, Box<dyn Error>> {
    let path = run_dir.join("seeds.json");
    Ok(serde_json::from_slice(&fs::read(path).await?)?)
}

/// Require a repository-relative run path so the manifest remains portable.
fn repository_relative_path(run_dir: &Path) -> Result<Box<str>, Box<dyn Error>> {
    let path = run_dir.to_string_lossy();
    if run_dir.is_absolute()
        || !path.starts_with("runs/")
        || run_dir
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(format!("source run must be a repository-relative runs/ path: {path}").into());
    }
    Ok(path.into_owned().into_boxed_str())
}

/// Resolve the full Git object ID recorded beside the static release.
async fn current_source_commit() -> Result<SourceCommit, Box<dyn Error>> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .await?;
    if !output.status.success() {
        return Err("git rev-parse HEAD failed".into());
    }
    let commit = String::from_utf8(output.stdout)?.trim().to_owned();
    Ok(SourceCommit::try_from(commit)?)
}

/// Load one record through the production byte path and require finite output.
async fn validate_policy(path: &Path) -> Result<RecurrentPpoPolicy, Box<dyn Error>> {
    let bytes = fs::read(path).await?;
    let config = ecosystem_algorithm();
    let policy = RecurrentPpoPolicy::load_bytes(
        bytes,
        LOCAL_OBSERVATION_SIZE,
        GLOBAL_STATE_SIZE,
        MAX_AGENTS,
        &ACTION_LOW,
        &ACTION_HIGH,
        &config,
    )?;
    let memory = policy.initial_memory();
    let output = policy.mean_action(&vec![0.0; LOCAL_OBSERVATION_SIZE], &memory)?;
    let global_state = vec![0.0; GLOBAL_STATE_SIZE];
    let mut values_are_finite = true;
    for index in 0..MAX_AGENTS {
        values_are_finite &= policy.value(&global_state, index)?.is_finite();
    }
    if output.action.iter().all(|value| value.is_finite())
        && output
            .next_memory
            .cell
            .iter()
            .all(|value| value.is_finite())
        && output
            .next_memory
            .hidden
            .iter()
            .all(|value| value.is_finite())
        && values_are_finite
    {
        Ok(policy)
    } else {
        let path_display = path.display();
        Err(format!("checkpoint probe emitted non-finite output: {path_display}").into())
    }
}

/// Copy one verified checkpoint under its content-addressed static name.
async fn export_asset(
    stage: Stage,
    role: &str,
    source: &Path,
    config_source: &Path,
    output_dir: &Path,
    source_run: &str,
) -> Result<CheckpointAsset, Box<dyn Error>> {
    let bytes = fs::read(source).await?;
    let digest = Sha256::digest(&bytes);
    let digest = format!("{digest:x}");
    let prefix = digest.chars().take(12).collect::<String>();
    let stage_key = stage.as_key();
    let filename = format!("{stage_key}-{role}-{prefix}.mpk");
    let output = output_dir.join(&filename);
    fs::write(output, &bytes).await?;
    let config_bytes = fs::read(config_source).await?;
    let config_digest = format!("{:x}", Sha256::digest(&config_bytes));
    let config_filename = format!("{stage_key}-{role}-{prefix}.config.json");
    let config_output = output_dir.join(&config_filename);
    fs::write(config_output, &config_bytes).await?;
    Ok(CheckpointAsset {
        path: AssetPath::try_from(format!("checkpoints/{filename}"))?,
        sha256: Sha256Digest::try_from(digest)?,
        bytes: CheckpointSize::try_from(u32::try_from(bytes.len())?)?,
        config_path: ConfigPath::try_from(format!("checkpoints/{config_filename}"))?,
        config_sha256: Sha256Digest::try_from(config_digest)?,
        config_bytes: CheckpointSize::try_from(u32::try_from(config_bytes.len())?)?,
        source_run: source_run.into(),
    })
}

/// Read an empty or partially populated manifest.
async fn read_mutable_manifest(path: &Path) -> Result<MutableManifest, Box<dyn Error>> {
    let bytes = fs::read(path).await?;
    let mut manifest: MutableManifest = serde_json::from_slice(&bytes)?;
    manifest.source_commit = current_source_commit().await?;
    manifest.burn_version = BurnVersion::V0_21_0;
    Ok(manifest)
}

/// Return the architecture and inference settings used by ecosystem training.
fn ecosystem_algorithm() -> RecurrentPpoConfig {
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
