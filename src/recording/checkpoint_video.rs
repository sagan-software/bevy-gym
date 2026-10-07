//! Checkpoint-progression video adapter for existing example GIF renderers.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

use crate::runtime_io;

use super::CheckpointTimeline;

/// Render selected checkpoints through the current example's `gif` command.
///
/// This adapter lets existing examples adopt the standard dynamic timeline
/// while their scene-specific frame driver remains unchanged.
///
/// # Errors
///
/// Returns an error when checkpoint selection, a child render, or ffmpeg fails.
pub fn record_training_video_from_gif_command(
    run_directory: &Path,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let timeline = CheckpointTimeline::training_progress(run_directory)?;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let workspace = std::env::temp_dir().join(format!(
        "bevy-gym-checkpoint-video-{}-{timestamp}",
        std::process::id()
    ));
    runtime_io::create_directory_all(&workspace)?;

    let result = render_and_encode(&timeline, output, &workspace);
    let cleanup = runtime_io::remove_directory_all(&workspace);
    result?;
    cleanup?;
    Ok(())
}

/// Render each distinct checkpoint once and encode the exact timeline.
fn render_and_encode(
    timeline: &CheckpointTimeline,
    output: &Path,
    workspace: &Path,
) -> Result<(), Box<dyn Error>> {
    let executable = std::env::current_exe()?;
    let mut clips = BTreeMap::<PathBuf, PathBuf>::new();
    // Render each selected checkpoint once even when the timeline reuses the
    // best checkpoint for both the opening and closing segments.
    for segment in timeline.segments() {
        if clips.contains_key(segment.checkpoint()) {
            continue;
        }
        let clip = workspace.join(format!("clip-{:02}.gif", clips.len()));
        let mut command = Command::new(&executable);
        command
            .arg("gif")
            .arg("--checkpoint")
            .arg(segment.checkpoint())
            .arg("--output")
            .arg(&clip);
        let status = runtime_io::command_status(&mut command)?;
        if !status.success() {
            return Err(format!(
                "checkpoint GIF renderer failed for {} with {status}",
                segment.checkpoint().display()
            )
            .into());
        }
        clips.insert(segment.checkpoint().to_path_buf(), clip);
    }

    let mut manifest = String::new();
    // Expand five-second source clips to the exact segment durations selected
    // by the dynamic checkpoint timeline.
    for segment in timeline.segments() {
        let clip = clips
            .get(segment.checkpoint())
            .ok_or("rendered checkpoint clip is missing")?;
        let repeats = segment.duration().as_secs() / 5;
        for _ in 0..repeats {
            writeln!(&mut manifest, "file '{}'", clip.display())?;
        }
    }
    let manifest_path = workspace.join("concat.txt");
    runtime_io::write(&manifest_path, manifest)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        runtime_io::create_directory_all(parent)?;
    }
    let mut command = Command::new("ffmpeg");
    // Normalize every source GIF into the documented 30-second MP4 profile.
    command
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
        ])
        .arg(&manifest_path)
        .args([
            "-vf",
            "fps=30",
            "-t",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(output);
    let status = runtime_io::command_status(&mut command)?;
    if !status.success() {
        return Err(format!("ffmpeg failed with {status}").into());
    }
    Ok(())
}
