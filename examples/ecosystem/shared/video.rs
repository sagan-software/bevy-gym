//! Deterministic checkpoint-sequence video capture and manifests.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::error::Error;
use std::fs::File;
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::rc::Rc;

use bevy::prelude::*;
use bevy::render::view::screenshot::{Capturing, Screenshot, ScreenshotCaptured};
use bevy::window::{PresentMode, WindowResolution};
use bevy_gym::training::RecurrentPpoPolicy;
use bevy_inspector_egui::bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_inspector_egui::DefaultInspectorConfigPlugin;
use serde::Serialize;
use tokio::io::AsyncWriteExt as _;
use tokio::process::{Child, ChildStdin, Command};
use tokio::runtime::{Builder, Runtime};

use super::domain::{CurriculumStage, SimulationConfig};
use super::rendering::{
    draw_perception_rays, ecosystem_hud_ui, load_policy, redraw_scene, setup_viewer, WatchSession,
};
use super::training::{
    checkpoint_experiment_tuning, ecosystem_algorithm, sibling_fox_checkpoint,
    validate_checkpoint_stage, validate_checkpoint_tuning_match, ExperimentTuningProfile,
};

/// Required output width in pixels.
const VIDEO_WIDTH: u32 = 1280;

/// Required output height in pixels.
const VIDEO_HEIGHT: u32 = 720;

/// Exact opening duration for the step-zero checkpoint.
const INTRO_SECONDS: u32 = 5;

/// Exact duration for each chronological training checkpoint.
const CHECKPOINT_SECONDS: u32 = 5;

/// Exact closing duration for the validation-selected checkpoint.
const OUTRO_SECONDS: u32 = 15;

/// Validated command-line video controls.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VideoOptions {
    /// Run directory containing synchronized checkpoints.
    run_dir: PathBuf,

    /// MP4 output path.
    output: PathBuf,

    /// Deterministic map and reset seed.
    seed: u64,

    /// Output frames per second.
    frames_per_second: u32,

    /// Per-episode environment horizon.
    episode_seconds: Option<u16>,

    /// Opening best-checkpoint seconds.
    intro_seconds: u32,

    /// Seconds per chronological checkpoint.
    checkpoint_seconds: u32,

    /// Closing best-checkpoint seconds.
    outro_seconds: u32,
}

/// Semantic role of one video segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum SegmentKind {
    /// Opening step-zero or earliest policy.
    Worst,

    /// Immutable chronological training policy.
    Checkpoint,

    /// Closing validation-selected policy.
    Best,
}

/// Paths and timing resolved before any GPU work begins.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SegmentDescriptor {
    /// Semantic segment role.
    kind: SegmentKind,

    /// Bunny actor and critic checkpoint.
    bunny_checkpoint: PathBuf,

    /// Synchronized fox checkpoint when required by the stage.
    fox_checkpoint: Option<PathBuf>,

    /// Stable HUD label.
    label: String,

    /// Exact captured frame count.
    frames: u32,
}

/// One loaded policy segment ready for an atomic switch.
#[derive(Debug, Clone)]
struct LoadedSegment {
    /// Resolved manifest metadata.
    descriptor: SegmentDescriptor,

    /// Bunny policy snapshot.
    bunny: RecurrentPpoPolicy,

    /// Fox policy snapshot when required by the stage.
    fox: Option<RecurrentPpoPolicy>,
}

/// Serializable evidence for one encoded segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct SegmentManifest {
    /// Semantic segment role.
    kind: SegmentKind,

    /// Bunny checkpoint path.
    bunny_checkpoint: PathBuf,

    /// SHA-256 of the bunny checkpoint.
    bunny_sha256: String,

    /// Fox checkpoint path when present.
    fox_checkpoint: Option<PathBuf>,

    /// SHA-256 of the fox checkpoint when present.
    fox_sha256: Option<String>,

    /// Inclusive first output frame.
    start_frame: u64,

    /// Exact number of frames in this segment.
    frames: u32,
}

/// Complete machine-readable provenance for one training-progress video.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct VideoManifest {
    /// Curriculum stage key.
    stage: String,

    /// Generated MP4 path.
    output: PathBuf,

    /// Deterministic evaluation seed.
    seed: u64,

    /// Output width in pixels.
    width: u32,

    /// Output height in pixels.
    height: u32,

    /// Output frames per second.
    frames_per_second: u32,

    /// H.264 pixel format required for broad playback compatibility.
    pixel_format: String,

    /// Exact effective environment tuning, including any CLI horizon override.
    experiment_tuning: ExperimentTuningProfile,

    /// Ordered checkpoint provenance and frame ranges.
    segments: Vec<SegmentManifest>,
}

/// `FFmpeg` child process receiving packed RGB frames.
struct VideoEncoder {
    /// Current-thread runtime for async process and pipe operations.
    runtime: Runtime,

    /// Running `FFmpeg` child.
    child: Child,

    /// Raw-video standard input, closed before waiting.
    input: Option<ChildStdin>,
}

impl std::fmt::Debug for VideoEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VideoEncoder")
            .field("child_id", &self.child.id())
            .field("has_input", &self.input.is_some())
            .finish_non_exhaustive()
    }
}

/// Frame sequencing and encoder state owned by the render world.
#[derive(Debug)]
struct VideoController {
    /// Later segments in exact playback order.
    remaining: VecDeque<LoadedSegment>,

    /// Frames still required from the current segment.
    frames_remaining: u32,

    /// Raw-frame encoder until successful finalization.
    encoder: Option<VideoEncoder>,

    /// First recoverable capture, policy, or encoder error.
    error: Option<String>,

    /// Whether all frames were encoded and `FFmpeg` exited successfully.
    is_complete: bool,

    /// Full render updates required before requesting the next screenshot.
    updates_before_capture: u8,

    /// Result handle retained after Bevy's runner consumes the app world.
    outcome: Rc<RefCell<CaptureOutcome>>,
}

/// Capture result shared with the command boundary outside Bevy's app runner.
#[derive(Debug, Default)]
struct CaptureOutcome {
    /// First recoverable capture, policy, or encoder error.
    error: Option<String>,

    /// Whether every requested frame reached a successful MP4 finalization.
    is_complete: bool,
}

/// Render an exact checkpoint sequence and write its provenance manifest.
pub(super) fn run_video(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let options = parse_video_options(arguments)?;
    let mut config = SimulationConfig::for_stage(stage)?;
    let descriptors = resolve_segments(stage, &options)?;
    let first_tuning = validate_video_profiles(stage, &descriptors)?;
    config.apply_experiment_tuning(first_tuning)?;
    if let Some(episode_seconds) = options.episode_seconds {
        config.episode_seconds = episode_seconds;
    }
    let manifest = build_manifest(stage, &options, &descriptors, config.experiment_tuning())?;
    let manifest_path = manifest_path(&options.output);
    if options.output.exists() || manifest_path.exists() {
        return Err("video refuses to overwrite an existing MP4 or manifest".into());
    }
    let mut loaded = load_segments(descriptors)?;
    let first = loaded
        .pop_front()
        .ok_or("video has no checkpoint segments")?;
    let playback_speed = options.frames_per_second as f32 * config.time_step;
    let session = WatchSession::new(
        stage,
        config,
        first.bunny.clone(),
        first.fox.clone(),
        options.seed,
        playback_speed,
        first.descriptor.label.clone(),
    )?;
    let encoder = VideoEncoder::spawn(&options.output, options.frames_per_second)?;
    let outcome = Rc::new(RefCell::new(CaptureOutcome::default()));
    let controller = VideoController {
        frames_remaining: first.descriptor.frames,
        remaining: loaded,
        encoder: Some(encoder),
        error: None,
        is_complete: false,
        updates_before_capture: 2,
        outcome: Rc::clone(&outcome),
    };

    // Screenshot capture advances only after the previous frame reaches CPU
    // memory, so every encoded frame maps to one deterministic policy step.
    let mut app = App::new();
    app.insert_non_send_resource(session)
        .insert_non_send_resource(controller)
        .insert_resource(ClearColor(Color::srgb_u8(15, 23, 18)))
        .add_plugins(
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: format!("bevy-gym ecosystem video: {}", stage.as_key()),
                    resolution: WindowResolution::new(VIDEO_WIDTH, VIDEO_HEIGHT)
                        .with_scale_factor_override(1.0),
                    present_mode: PresentMode::AutoNoVsync,
                    ..default()
                }),
                ..default()
            }),
        )
        .add_plugins(EguiPlugin::default())
        .add_plugins(DefaultInspectorConfigPlugin)
        .add_systems(Startup, setup_viewer)
        .add_systems(
            Update,
            (redraw_scene, draw_perception_rays, request_video_frame).chain(),
        )
        .add_systems(EguiPrimaryContextPass, ecosystem_hud_ui);
    app.run();

    let result = outcome.borrow();
    if let Some(error) = &result.error {
        return Err(error.clone().into());
    }
    if !result.is_complete {
        return Err("video window closed before every required frame was encoded".into());
    }
    drop(result);
    write_manifest(&manifest_path, &manifest)?;
    writeln!(
        io::stdout().lock(),
        "video={} manifest={} frames={}",
        options.output.display(),
        manifest_path.display(),
        manifest
            .segments
            .iter()
            .map(|segment| u64::from(segment.frames))
            .sum::<u64>(),
    )?;
    Ok(())
}

/// Require every video policy to share one immutable environment profile.
fn validate_video_profiles(
    stage: CurriculumStage,
    descriptors: &[SegmentDescriptor],
) -> Result<super::domain::ExperimentTuning, Box<dyn Error>> {
    let first_checkpoint = &descriptors
        .first()
        .ok_or("video has no checkpoint segments")?
        .bunny_checkpoint;
    let first_tuning = checkpoint_experiment_tuning(first_checkpoint)?
        .ok_or("first video checkpoint lacks experiment tuning")?;
    for descriptor in descriptors {
        validate_checkpoint_stage(&descriptor.bunny_checkpoint, stage)?;
        let tuning = checkpoint_experiment_tuning(&descriptor.bunny_checkpoint)?
            .ok_or("video checkpoint lacks experiment tuning")?;
        if tuning != first_tuning {
            return Err(format!(
                "video checkpoint tuning mismatch: {} differs from {}",
                descriptor.bunny_checkpoint.display(),
                first_checkpoint.display()
            )
            .into());
        }
        if let Some(fox_checkpoint) = &descriptor.fox_checkpoint {
            validate_checkpoint_stage(fox_checkpoint, stage)?;
            validate_checkpoint_tuning_match(&descriptor.bunny_checkpoint, fox_checkpoint)?;
        }
    }
    Ok(first_tuning)
}

/// Parse and validate video arguments without inheriting watch-only controls.
fn parse_video_options(
    mut arguments: impl Iterator<Item = String>,
) -> Result<VideoOptions, Box<dyn Error>> {
    let mut run_dir = None;
    let mut output = None;
    let mut seed = 201;
    let mut frames_per_second = 30;
    let mut episode_seconds = None;
    let mut intro_seconds = INTRO_SECONDS;
    let mut checkpoint_seconds = CHECKPOINT_SECONDS;
    let mut outro_seconds = OUTRO_SECONDS;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--checkpoint" => run_dir = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--seed" => seed = value.parse()?,
            "--fps" => frames_per_second = value.parse()?,
            "--episode-seconds" => episode_seconds = Some(value.parse()?),
            "--intro-seconds" => intro_seconds = value.parse()?,
            "--checkpoint-seconds" => checkpoint_seconds = value.parse()?,
            "--outro-seconds" => outro_seconds = value.parse()?,
            _ => return Err(format!("unknown video option {flag:?}").into()),
        }
    }
    let run_dir = run_dir.ok_or("video requires --checkpoint <run-dir>")?;
    if !run_dir.is_dir() {
        return Err("video --checkpoint must name a run directory".into());
    }
    let output = output.ok_or("video requires --output <video.mp4>")?;
    if output.extension().and_then(|value| value.to_str()) != Some("mp4") {
        return Err("video --output must end in .mp4".into());
    }
    if !(1..=60).contains(&frames_per_second)
        || episode_seconds.is_some_and(|seconds| !(5..=300).contains(&seconds))
        || intro_seconds == 0
        || checkpoint_seconds == 0
        || outro_seconds == 0
    {
        return Err("video segment timing must be positive, --fps must be in 1..=60, and --episode-seconds must be in 5..=300".into());
    }
    Ok(VideoOptions {
        run_dir,
        output,
        seed,
        frames_per_second,
        episode_seconds,
        intro_seconds,
        checkpoint_seconds,
        outro_seconds,
    })
}

/// Resolve best and chronological checkpoints into the required segment order.
fn resolve_segments(
    stage: CurriculumStage,
    options: &VideoOptions,
) -> Result<Vec<SegmentDescriptor>, Box<dyn Error>> {
    let best = options.run_dir.join("best.mpk");
    require_file(&best)?;
    let mut chronological = checkpoint_paths(&options.run_dir.join("checkpoints"))?;
    chronological.sort();
    if chronological.is_empty() {
        return Err("video run has no chronological step-*.mpk checkpoints".into());
    }

    // Sample the chronological run at worst, early, and middle points before
    // giving the validation-selected best checkpoint the final fifteen seconds.
    let worst = chronological
        .first()
        .cloned()
        .ok_or("video run has no worst checkpoint")?;
    let early = chronological
        .get(chronological.len() / 3)
        .cloned()
        .ok_or("video run has no early checkpoint")?;
    let middle = chronological
        .get(chronological.len() * 2 / 3)
        .cloned()
        .ok_or("video run has no middle checkpoint")?;
    let mut descriptors = Vec::with_capacity(4);
    descriptors.push(segment_descriptor(
        stage,
        SegmentKind::Worst,
        worst,
        "WORST / STEP ZERO".to_owned(),
        options.intro_seconds * options.frames_per_second,
    )?);
    for checkpoint in [early, middle] {
        let name = checkpoint
            .file_stem()
            .and_then(|value| value.to_str())
            .ok_or("checkpoint filename is not UTF-8")?
            .to_owned();
        descriptors.push(segment_descriptor(
            stage,
            SegmentKind::Checkpoint,
            checkpoint,
            name,
            options.checkpoint_seconds * options.frames_per_second,
        )?);
    }
    descriptors.push(segment_descriptor(
        stage,
        SegmentKind::Best,
        best,
        "BEST (final)".to_owned(),
        options.outro_seconds * options.frames_per_second,
    )?);
    Ok(descriptors)
}

/// Read only bunny step checkpoints from one run's immutable checkpoint folder.
fn checkpoint_paths(directory: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let runtime = Builder::new_current_thread().enable_io().build()?;
    let paths = runtime.block_on(async {
        let mut entries = tokio::fs::read_dir(directory).await?;
        let mut paths = Vec::new();
        loop {
            let next_entry = entries.next_entry().await?;
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
                paths.push(path);
            }
        }
        Ok::<_, io::Error>(paths)
    })?;
    Ok(paths)
}

/// Pair one bunny checkpoint with its synchronized fox file when required.
fn segment_descriptor(
    stage: CurriculumStage,
    kind: SegmentKind,
    bunny_checkpoint: PathBuf,
    label: String,
    frames: u32,
) -> Result<SegmentDescriptor, Box<dyn Error>> {
    let fox_checkpoint = matches!(
        stage,
        CurriculumStage::PredatorPrey | CurriculumStage::Obstacles
    )
    .then(|| sibling_fox_checkpoint(&bunny_checkpoint));
    if let Some(path) = &fox_checkpoint {
        require_file(path)?;
    }
    Ok(SegmentDescriptor {
        kind,
        bunny_checkpoint,
        fox_checkpoint,
        label,
        frames,
    })
}

/// Load every segment before opening a window so capture cannot partially fail.
fn load_segments(
    descriptors: Vec<SegmentDescriptor>,
) -> Result<VecDeque<LoadedSegment>, Box<dyn Error>> {
    let algorithm = ecosystem_algorithm();
    descriptors
        .into_iter()
        .map(|descriptor| {
            let bunny = load_policy(&descriptor.bunny_checkpoint, &algorithm)?;
            let fox = descriptor
                .fox_checkpoint
                .as_deref()
                .map(|path| load_policy(path, &algorithm))
                .transpose()?;
            Ok(LoadedSegment {
                descriptor,
                bunny,
                fox,
            })
        })
        .collect()
}

/// Build typed manifest data and checkpoint hashes before capture begins.
fn build_manifest(
    stage: CurriculumStage,
    options: &VideoOptions,
    descriptors: &[SegmentDescriptor],
    tuning: super::domain::ExperimentTuning,
) -> Result<VideoManifest, Box<dyn Error>> {
    let mut start_frame = 0_u64;
    let mut segments = Vec::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let frames = descriptor.frames;
        segments.push(SegmentManifest {
            kind: descriptor.kind,
            bunny_checkpoint: descriptor.bunny_checkpoint.clone(),
            bunny_sha256: sha256(&descriptor.bunny_checkpoint)?,
            fox_checkpoint: descriptor.fox_checkpoint.clone(),
            fox_sha256: descriptor
                .fox_checkpoint
                .as_deref()
                .map(sha256)
                .transpose()?,
            start_frame,
            frames,
        });
        start_frame = start_frame.saturating_add(u64::from(frames));
    }
    Ok(VideoManifest {
        stage: stage.as_key().to_owned(),
        output: options.output.clone(),
        seed: options.seed,
        width: VIDEO_WIDTH,
        height: VIDEO_HEIGHT,
        frames_per_second: options.frames_per_second,
        pixel_format: "yuv420p".to_owned(),
        experiment_tuning: ExperimentTuningProfile::from(tuning),
        segments,
    })
}

impl VideoEncoder {
    /// Start `FFmpeg` with an exact raw RGB input and H.264 MP4 output contract.
    fn spawn(output: &Path, frames_per_second: u32) -> Result<Self, Box<dyn Error>> {
        let runtime = Builder::new_current_thread().enable_io().build()?;
        if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
            runtime.block_on(tokio::fs::create_dir_all(parent))?;
        }
        let mut child = {
            let _runtime_guard = runtime.enter();
            Command::new("ffmpeg")
                .args([
                    "-n",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "1280x720",
                    "-framerate",
                    &frames_per_second.to_string(),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                ])
                .arg(output)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()?
        };
        let input = child.stdin.take().ok_or("FFmpeg did not expose stdin")?;
        Ok(Self {
            runtime,
            child,
            input: Some(input),
        })
    }

    /// Convert one Bevy screenshot to packed RGB and send it to `FFmpeg`.
    fn write_frame(&mut self, image: &Image) -> Result<(), Box<dyn Error>> {
        let rgb = image.clone().try_into_dynamic()?.to_rgb8();
        if rgb.width() != VIDEO_WIDTH || rgb.height() != VIDEO_HEIGHT {
            return Err(format!(
                "captured frame is {}x{}, expected {VIDEO_WIDTH}x{VIDEO_HEIGHT}",
                rgb.width(),
                rgb.height(),
            )
            .into());
        }
        let input = self
            .input
            .as_mut()
            .ok_or("FFmpeg input is already closed")?;
        self.runtime.block_on(input.write_all(rgb.as_raw()))?;
        Ok(())
    }

    /// Close the raw stream and require a successful encoder exit status.
    fn finish(mut self) -> Result<(), Box<dyn Error>> {
        drop(self.input.take());
        let status = self.runtime.block_on(self.child.wait())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("FFmpeg exited with {status}").into())
        }
    }

    /// Stop a failed encoder and reap its child process.
    fn abort(mut self) {
        drop(self.input.take());
        let _kill_result = self.runtime.block_on(self.child.kill());
        let _wait_result = self.runtime.block_on(self.child.wait());
    }
}

/// Request at most one asynchronous primary-window screenshot at a time.
fn request_video_frame(
    mut commands: Commands<'_, '_>,
    captures: Query<'_, '_, (), With<Capturing>>,
    mut controller: NonSendMut<'_, VideoController>,
) {
    if controller.updates_before_capture > 0 {
        controller.updates_before_capture -= 1;
        return;
    }
    if !controller.is_complete && controller.error.is_none() && captures.is_empty() {
        commands
            .spawn(Screenshot::primary_window())
            .observe(encode_video_frame);
    }
}

/// Encode one captured frame, advance policy state, and switch segments atomically.
fn encode_video_frame(
    captured: On<'_, '_, ScreenshotCaptured>,
    mut controller: NonSendMut<'_, VideoController>,
    mut session: NonSendMut<'_, WatchSession>,
    mut exit: MessageWriter<'_, AppExit>,
) {
    let result = controller.accept_frame(&captured.image, &mut session);
    if let Err(error) = result {
        let message = error.to_string();
        controller.error = Some(message.clone());
        controller.outcome.borrow_mut().error = Some(message);
        if let Some(encoder) = controller.encoder.take() {
            encoder.abort();
        }
        exit.write(AppExit::error());
    } else if controller.is_complete {
        exit.write(AppExit::Success);
    }
}

impl VideoController {
    /// Accept one frame and prepare the exact next world/checkpoint state.
    fn accept_frame(
        &mut self,
        image: &Image,
        session: &mut WatchSession,
    ) -> Result<(), Box<dyn Error>> {
        self.encoder
            .as_mut()
            .ok_or("video encoder is unavailable")?
            .write_frame(image)?;
        self.frames_remaining = self.frames_remaining.saturating_sub(1);
        if self.frames_remaining > 0 {
            let _did_reset = session.step_once()?;
            self.updates_before_capture = 1;
            return Ok(());
        }

        if let Some(next) = self.remaining.pop_front() {
            session.replace_policies(
                next.bunny.clone(),
                next.fox.clone(),
                next.descriptor.label.clone(),
            )?;
            self.frames_remaining = next.descriptor.frames;
            self.updates_before_capture = 2;
            return Ok(());
        }

        self.encoder
            .take()
            .ok_or("video encoder disappeared before finalization")?
            .finish()?;
        self.is_complete = true;
        self.outcome.borrow_mut().is_complete = true;
        Ok(())
    }
}

/// Require a regular checkpoint file before any policy loading begins.
fn require_file(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("checkpoint does not exist: {}", path.display()).into())
    }
}

/// Compute one lowercase SHA-256 using the installed coreutils implementation.
fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let runtime = Builder::new_current_thread().enable_io().build()?;
    let output = runtime.block_on(async { Command::new("sha256sum").arg(path).output().await })?;
    if !output.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let stdout = String::from_utf8(output.stdout)?;
    stdout
        .split_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or_else(|| format!("sha256sum returned no hash for {}", path.display()).into())
}

/// Derive a sibling JSON manifest path from an MP4 output path.
fn manifest_path(output: &Path) -> PathBuf {
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("ecosystem");
    output.with_file_name(format!("{stem}-video-manifest.json"))
}

/// Write a typed, human-readable JSON manifest only after successful encoding.
fn write_manifest(path: &Path, manifest: &VideoManifest) -> Result<(), Box<dyn Error>> {
    let mut output = io::BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut output, manifest)?;
    writeln!(output)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::super::training::write_checkpoint_profile;
    use super::*;

    /// Default video timing exactly matches the requested evidence sequence.
    #[test]
    fn video_defaults_are_exact() {
        let run_dir = PathBuf::from("runs/ecosystem-survival-ppo/continuation-probe-105");
        if !run_dir.is_dir() {
            return;
        }
        let options = parse_video_options(
            [
                "--checkpoint",
                run_dir.to_str().expect("fixture path is UTF-8"),
                "--output",
                "/tmp/ecosystem.mp4",
            ]
            .map(str::to_owned)
            .into_iter(),
        )
        .expect("default video arguments parse");
        assert_eq!(options.intro_seconds, 5);
        assert_eq!(options.checkpoint_seconds, 5);
        assert_eq!(options.outro_seconds, 15);
        assert_eq!(options.episode_seconds, None);
        for value in ["1", "301"] {
            assert!(parse_video_options(
                [
                    "--checkpoint",
                    run_dir.to_str().expect("fixture path is UTF-8"),
                    "--output",
                    "/tmp/ecosystem.mp4",
                    "--episode-seconds",
                    value,
                ]
                .map(str::to_owned)
                .into_iter(),
            )
            .is_err());
        }
        assert_eq!(options.frames_per_second, 30);
    }

    /// Manifest naming is stable and does not overwrite the MP4.
    #[test]
    fn manifest_path_is_a_sibling() {
        assert_eq!(
            manifest_path(Path::new("media/survival.mp4")),
            PathBuf::from("media/survival-video-manifest.json"),
        );
    }

    /// Segment resolution must produce four distinct roles and exactly 900 frames.
    #[test]
    fn progression_segments_match_the_thirty_second_contract() {
        let directory =
            std::env::temp_dir().join(format!("bevy-gym-video-segments-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).expect("stale test directory is removable");
        }
        fs::create_dir_all(directory.join("checkpoints"))
            .expect("checkpoint directory is creatable");
        fs::write(directory.join("best.mpk"), b"best").expect("best checkpoint writes");
        for step in [0, 100, 200, 300] {
            fs::write(
                directory
                    .join("checkpoints")
                    .join(format!("step-{step:06}.mpk")),
                step.to_string(),
            )
            .expect("step checkpoint writes");
        }
        let options = VideoOptions {
            run_dir: directory.clone(),
            output: directory.join("progression.mp4"),
            seed: 907,
            frames_per_second: 30,
            episode_seconds: None,
            intro_seconds: 5,
            checkpoint_seconds: 5,
            outro_seconds: 15,
        };

        let segments =
            resolve_segments(CurriculumStage::Forage, &options).expect("segments resolve");

        assert_eq!(segments.len(), 4);
        assert_eq!(segments[0].kind, SegmentKind::Worst);
        assert_eq!(segments[1].kind, SegmentKind::Checkpoint);
        assert_eq!(segments[2].kind, SegmentKind::Checkpoint);
        assert_eq!(segments[3].kind, SegmentKind::Best);
        assert_eq!(
            segments.iter().map(|segment| segment.frames).sum::<u32>(),
            900
        );
        assert!(segments
            .windows(2)
            .all(|pair| pair[0].bunny_checkpoint != pair[1].bunny_checkpoint));
        fs::remove_dir_all(directory).expect("test directory is removable");
    }

    /// Video segments must reject policy snapshots from different mechanics.
    #[test]
    fn video_profiles_reject_tuning_changes_between_segments() {
        let directory =
            std::env::temp_dir().join(format!("bevy-gym-video-profiles-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).expect("stale test directory is removable");
        }
        fs::create_dir_all(&directory).expect("test directory is creatable");
        let first = directory.join("step-000000.mpk");
        let second = directory.join("step-000100.mpk");
        let tuning = super::super::domain::ExperimentTuning::default();
        write_checkpoint_profile(&first, CurriculumStage::Survival, tuning)
            .expect("first profile writes");
        let mut changed = tuning;
        changed.episode_seconds += 1;
        write_checkpoint_profile(&second, CurriculumStage::Survival, changed)
            .expect("second profile writes");
        let descriptors = [
            SegmentDescriptor {
                kind: SegmentKind::Checkpoint,
                bunny_checkpoint: first,
                fox_checkpoint: None,
                label: "first".to_owned(),
                frames: 1,
            },
            SegmentDescriptor {
                kind: SegmentKind::Checkpoint,
                bunny_checkpoint: second,
                fox_checkpoint: None,
                label: "second".to_owned(),
                frames: 1,
            },
        ];

        assert!(validate_video_profiles(CurriculumStage::Survival, &descriptors).is_err());
        fs::remove_dir_all(directory).expect("test directory is removable");
    }

    /// Video provenance must retain the effective horizon after a CLI override.
    #[test]
    fn video_manifest_records_effective_tuning() {
        let directory =
            std::env::temp_dir().join(format!("bevy-gym-video-manifest-{}", std::process::id()));
        if directory.exists() {
            fs::remove_dir_all(&directory).expect("stale test directory is removable");
        }
        fs::create_dir_all(&directory).expect("test directory is creatable");
        let checkpoint = directory.join("best.mpk");
        fs::write(&checkpoint, b"checkpoint").expect("checkpoint fixture writes");
        let options = VideoOptions {
            run_dir: directory.clone(),
            output: directory.join("proof.mp4"),
            seed: 907,
            frames_per_second: 30,
            episode_seconds: Some(33),
            intro_seconds: 10,
            checkpoint_seconds: 5,
            outro_seconds: 20,
        };
        let descriptors = [SegmentDescriptor {
            kind: SegmentKind::Best,
            bunny_checkpoint: checkpoint,
            fox_checkpoint: None,
            label: "best".to_owned(),
            frames: 30,
        }];
        let mut tuning = super::super::domain::ExperimentTuning::default();
        tuning.episode_seconds = 33;

        let manifest = build_manifest(CurriculumStage::Survival, &options, &descriptors, tuning)
            .expect("manifest builds");

        assert_eq!(manifest.experiment_tuning.episode_seconds, 33);
        fs::remove_dir_all(directory).expect("test directory is removable");
    }
}
