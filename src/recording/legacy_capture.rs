//! Compatibility capture adapter for example scenes that drive their own frames.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

use crate::runtime_io;
use bevy::prelude::{
    default, Assets, Color, Commands, Cuboid, DirectionalLight, EulerRot, Mesh, Mesh3d,
    MeshMaterial3d, Quat, ResMut, Resource, StandardMaterial, Transform,
};

/// Finite frame capture state for a five-second 20-FPS demonstration.
#[derive(Debug, Resource)]
pub struct GifCapture {
    /// Directory receiving numbered PNG frames.
    directory: PathBuf,
    /// Next zero-based frame number.
    next_frame: usize,
    /// Render frames remaining before the first screenshot.
    warmup_frames: usize,
}

impl GifCapture {
    /// Start a capture in an existing temporary directory.
    #[must_use]
    pub fn new(directory: &Path) -> Self {
        Self {
            directory: directory.to_path_buf(),
            next_frame: 0,
            warmup_frames: 30,
        }
    }

    /// Reserve initial frames for asset loading.
    pub const fn is_warming_up(&mut self) -> bool {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        if self.warmup_frames == 0 {
            false
        } else {
            self.warmup_frames -= 1;
            true
        }
    }

    /// Return whether at least one frame has been requested.
    #[must_use]
    pub const fn has_started(&self) -> bool {
        self.next_frame > 0
    }

    /// Return the zero-based index of the next requested frame.
    #[must_use]
    pub const fn frame_index(&self) -> usize {
        self.next_frame
    }

    /// Return whether all five seconds have been requested.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.next_frame >= 100
    }

    /// Return the next numbered PNG path and advance the capture count.
    pub fn next_path(&mut self) -> PathBuf {
        let path = self
            .directory
            .join(format!("frame-{:03}.png", self.next_frame));
        self.next_frame += 1;
        path
    }
}

/// Capture 100 rendered frames, encode a 20-FPS GIF, and remove temporary files.
///
/// # Errors
///
/// Returns an error when capture, directory I/O, or ffmpeg fails.
pub fn encode_gif(
    output: &Path,
    prefix: &str,
    width: u32,
    height: u32,
    capture: impl FnOnce(&Path) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let frames = std::env::temp_dir().join(format!(
        "bevy-gym-{prefix}-gif-{}-{timestamp}",
        std::process::id()
    ));
    runtime_io::create_directory_all(&frames)?;
    if let Err(error) = capture(&frames) {
        drop(runtime_io::remove_directory_all(&frames));
        return Err(error);
    }
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        runtime_io::create_directory_all(parent)?;
    }

    let input = frames.join("frame-%03d.png");
    let filter = format!(
        "fps=20,scale={width}:{height}:flags=neighbor,split[frames][palette_source];\
         [palette_source]palettegen=stats_mode=full[palette];\
         [frames][palette]paletteuse=dither=sierra2_4a"
    );
    let mut command = Command::new("ffmpeg");
    command
        .args(["-y", "-loglevel", "error", "-framerate", "20", "-i"])
        .arg(&input)
        .args(["-vf", &filter])
        .arg(output);
    let status = runtime_io::command_status(&mut command)?;
    runtime_io::remove_directory_all(&frames)?;
    if !status.success() {
        return Err(format!("ffmpeg failed with {status}").into());
    }
    Ok(())
}

/// Spawn the lighting and checker floor used by MuJoCo-style scenes.
pub fn spawn_mujoco_stage(
    commands: &mut Commands<'_, '_>,
    meshes: &mut ResMut<'_, Assets<Mesh>>,
    materials: &mut ResMut<'_, Assets<StandardMaterial>>,
) {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    commands.spawn((
        DirectionalLight {
            illuminance: 5_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.3, 0.0)),
    ));
    let black = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        perceptual_roughness: 0.5,
        ..default()
    });
    let pale = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.88, 0.76),
        perceptual_roughness: 0.5,
        ..default()
    });
    let tile = meshes.add(Cuboid::new(0.5, 0.02, 0.5));
    for x in -160..=160 {
        for z in -8..=10 {
            commands.spawn((
                Mesh3d(tile.clone()),
                MeshMaterial3d(if (x + z) % 2 == 0 {
                    black.clone()
                } else {
                    pale.clone()
                }),
                Transform::from_xyz(x as f32 * 0.5, -0.02, z as f32 * 0.5),
            ));
        }
    }
}
