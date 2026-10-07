//! Bevy plugin that drives checkpoint playback, screenshots, and encoding.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::app::{App, AppExit, Last, Plugin};
use bevy::ecs::entity::Entity;
use bevy::ecs::prelude::{Resource, With, World};
use bevy::log::{error, info};
use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};

use crate::runtime_io;

use super::encoder::encode;
use super::{RecordingError, RecordingFrame, RecordingSettings};

/// Error type returned by environment-specific recorder callbacks.
pub type RecordingCallbackError = Box<dyn Error + Send + Sync + 'static>;

/// Environment-specific checkpoint loader called at each segment boundary.
type CheckpointLoader =
    Arc<dyn Fn(&mut World, &Path) -> Result<(), RecordingCallbackError> + Send + Sync>;

/// Environment-specific policy step and scene synchronization callback.
type FrameDriver =
    Arc<dyn Fn(&mut World, RecordingFrame) -> Result<(), RecordingCallbackError> + Send + Sync>;

/// Plugin that records a rendered training preview or checkpoint progression.
pub struct BevyGymRecorderPlugin {
    /// Validated output and timing settings.
    settings: RecordingSettings,
    /// Optional checkpoint loader for multi-checkpoint recordings.
    checkpoint_loader: Option<CheckpointLoader>,
    /// Optional environment step and scene synchronization callback.
    frame_driver: Option<FrameDriver>,
}

impl std::fmt::Debug for BevyGymRecorderPlugin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        formatter
            .debug_struct("BevyGymRecorderPlugin")
            .field("settings", &self.settings)
            .field("has_checkpoint_loader", &self.checkpoint_loader.is_some())
            .field("has_frame_driver", &self.frame_driver.is_some())
            .finish()
    }
}

impl BevyGymRecorderPlugin {
    /// Construct a recorder from validated settings.
    #[must_use]
    pub const fn new(settings: RecordingSettings) -> Self {
        Self {
            settings,
            checkpoint_loader: None,
            frame_driver: None,
        }
    }

    /// Load each segment's checkpoint and reset its visible episode.
    #[must_use]
    pub fn with_checkpoint_loader(
        mut self,
        loader: impl Fn(&mut World, &Path) -> Result<(), RecordingCallbackError> + Send + Sync + 'static,
    ) -> Self {
        self.checkpoint_loader = Some(Arc::new(loader));
        self
    }

    /// Advance the policy and synchronize the scene before each screenshot.
    #[must_use]
    pub fn with_frame_driver(
        mut self,
        driver: impl Fn(&mut World, RecordingFrame) -> Result<(), RecordingCallbackError>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        self.frame_driver = Some(Arc::new(driver));
        self
    }
}

impl Plugin for BevyGymRecorderPlugin {
    fn build(&self, app: &mut App) {
        // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
        let (frames_directory, initialization_error) = create_frames_directory();
        app.insert_resource(RecordingSession {
            settings: self.settings.clone(),
            checkpoint_loader: self.checkpoint_loader.clone(),
            frame_driver: self.frame_driver.clone(),
            frames_directory,
            warmup_frames: self.settings.warmup_frames(),
            segment_index: 0,
            frame_in_segment: 0,
            is_encoded: false,
            initialization_error,
        })
        .add_systems(Last, capture_or_encode);
    }
}

/// Mutable state for one finite recording session.
#[derive(Resource)]
struct RecordingSession {
    /// Validated output and timing settings.
    settings: RecordingSettings,
    /// Environment-specific checkpoint loader.
    checkpoint_loader: Option<CheckpointLoader>,
    /// Environment-specific frame driver.
    frame_driver: Option<FrameDriver>,
    /// Temporary directory receiving numbered PNG frames.
    frames_directory: PathBuf,
    /// Remaining initialization frames.
    warmup_frames: usize,
    /// Current playback segment.
    segment_index: usize,
    /// Current position inside the segment.
    frame_in_segment: usize,
    /// Whether ffmpeg has already run.
    is_encoded: bool,
    /// Directory setup failure deferred into the Bevy lifecycle.
    initialization_error: Option<RecordingError>,
}

/// Create the collision-resistant frame directory before the app starts.
fn create_frames_directory() -> (PathBuf, Option<RecordingError>) {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let path = std::env::temp_dir().join(format!(
        "bevy-gym-recorder-{}-{timestamp}",
        std::process::id()
    ));
    let error = runtime_io::create_directory_all(&path).err().map(|source| {
        RecordingError::CreateDirectory {
            path: path.clone(),
            source,
        }
    });
    (path, error)
}

/// Wait for prior screenshots, drive one frame, then encode after completion.
fn capture_or_encode(world: &mut World) {
    let mut captures = world.query_filtered::<Entity, With<Capturing>>();
    if captures.iter(world).next().is_some() {
        return;
    }

    if let Some(error) = world
        .resource_mut::<RecordingSession>()
        .initialization_error
        .take()
    {
        fail_recording(world, &error);
        return;
    }

    {
        let mut session = world.resource_mut::<RecordingSession>();
        if session.warmup_frames > 0 {
            session.warmup_frames -= 1;
            return;
        }
    }

    let should_encode = {
        let session = world.resource::<RecordingSession>();
        session.segment_index >= session.settings.timeline().segments().len()
    };
    if should_encode {
        encode_and_exit(world);
        return;
    }

    let (checkpoint_loader, frame_driver, checkpoint, frame, path) = {
        let session = world.resource::<RecordingSession>();
        let Some(segment) = session
            .settings
            .timeline()
            .segments()
            .get(session.segment_index)
        else {
            return;
        };
        let frames_per_second = session.settings.frames_per_second();
        let segment_frames = segment.duration().as_secs() as usize * usize::from(frames_per_second);
        let frame = RecordingFrame::new(
            segment.role(),
            session.segment_index,
            session.frame_in_segment,
            segment_frames,
            frames_per_second,
        );
        let frame_number = completed_frame_count(session);
        (
            session.checkpoint_loader.clone(),
            session.frame_driver.clone(),
            segment.checkpoint().to_path_buf(),
            frame,
            session
                .frames_directory
                .join(format!("frame-{frame_number:06}.png")),
        )
    };

    // Loading occurs before the first frame driver call so every segment starts
    // from a reset state owned by its selected checkpoint.
    if frame.is_first_in_segment() {
        if let Some(loader) = checkpoint_loader {
            if let Err(error) = loader(world, &checkpoint) {
                fail_recording(world, error.as_ref());
                return;
            }
        }
    }
    if let Some(driver) = frame_driver {
        if let Err(error) = driver(world, frame) {
            fail_recording(world, error.as_ref());
            return;
        }
    }

    world
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
    advance_frame_cursor(world);
}

/// Return the number used by the next captured PNG.
fn completed_frame_count(session: &RecordingSession) -> usize {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let frames_per_second = usize::from(session.settings.frames_per_second());
    let previous = session
        .settings
        .timeline()
        .segments()
        .iter()
        .take(session.segment_index)
        .map(|segment| segment.duration().as_secs() as usize * frames_per_second)
        .sum::<usize>();
    previous + session.frame_in_segment
}

/// Advance to the next frame or checkpoint segment.
fn advance_frame_cursor(world: &mut World) {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let mut session = world.resource_mut::<RecordingSession>();
    let Some(segment) = session
        .settings
        .timeline()
        .segments()
        .get(session.segment_index)
    else {
        return;
    };
    let segment_seconds = segment.duration().as_secs() as usize;
    let segment_frames = segment_seconds * usize::from(session.settings.frames_per_second());
    session.frame_in_segment += 1;
    if session.frame_in_segment >= segment_frames {
        session.segment_index += 1;
        session.frame_in_segment = 0;
    }
}

/// Encode all captured frames and request a successful or failed app exit.
fn encode_and_exit(world: &mut World) {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let (settings, frames_directory) = {
        let mut session = world.resource_mut::<RecordingSession>();
        if session.is_encoded {
            return;
        }
        session.is_encoded = true;
        (session.settings.clone(), session.frames_directory.clone())
    };

    match encode(&settings, &frames_directory) {
        Ok(()) => {
            if let Err(error) = runtime_io::remove_directory_all(&frames_directory) {
                error!("recording encoded but temporary frames could not be removed: {error}");
            }
            info!("recording saved to {}", settings.output().display());
            world.write_message(AppExit::Success);
        }
        Err(error) => fail_recording(world, &error),
    }
}

/// Log one recording failure, remove temporary frames, and request error exit.
fn fail_recording(world: &mut World, failure: &(dyn Error + 'static)) {
    // Keep recording lifecycle changes ordered so each captured frame has one stable source state.
    let frames_directory = world
        .resource::<RecordingSession>()
        .frames_directory
        .clone();
    error!("recording failed: {failure}");
    if let Err(error) = runtime_io::remove_directory_all(&frames_directory) {
        error!("recording cleanup failed: {error}");
    }
    world.write_message(AppExit::error());
}
