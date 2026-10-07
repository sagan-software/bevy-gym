//! Ffmpeg encoding after deterministic Bevy screenshot capture.

use std::path::Path;
use tokio::process::Command;

use crate::runtime_io;

use super::{RecordingError, RecordingFormat, RecordingSettings};

/// Encode numbered PNG frames into the configured artifact.
pub(super) fn encode(
    settings: &RecordingSettings,
    frames_directory: &Path,
) -> Result<(), RecordingError> {
    if let Some(parent) = settings
        .output()
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        runtime_io::create_directory_all(parent).map_err(|source| {
            RecordingError::CreateDirectory {
                path: parent.to_path_buf(),
                source,
            }
        })?;
    }

    let input = frames_directory.join("frame-%06d.png");
    let rate = settings.frames_per_second().to_string();
    let mut command = Command::new("ffmpeg");
    command
        .args(["-y", "-loglevel", "error", "-framerate", &rate, "-i"])
        .arg(input);

    // GIFs need one palette generated from the whole recording. MP4 uses a
    // browser-compatible pixel format and places metadata before video data.
    match settings.format() {
        RecordingFormat::Gif => {
            let filter = format!(
                "fps={rate},split[frames][palette_source];\
                 [palette_source]palettegen=stats_mode=full[palette];\
                 [frames][palette]paletteuse=dither=sierra2_4a"
            );
            command.args(["-vf", &filter]);
        }
        RecordingFormat::Mp4 => {
            command.args([
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
                "-r",
                &rate,
            ]);
        }
    }
    command.arg(settings.output());
    let status =
        runtime_io::command_status(&mut command).map_err(|source| RecordingError::RunEncoder {
            output: settings.output().to_path_buf(),
            source,
        })?;
    if !status.success() {
        return Err(RecordingError::EncoderFailed {
            output: settings.output().to_path_buf(),
            status,
        });
    }
    Ok(())
}
