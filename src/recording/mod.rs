//! Training checkpoint selection and rendered artifact recording.

#[cfg(feature = "render")]
mod bevy_gym_recorder_plugin;
mod checkpoint_role;
mod checkpoint_timeline;
#[cfg(feature = "render")]
mod checkpoint_video;
#[cfg(feature = "render")]
mod encoder;
mod error;
#[cfg(feature = "render")]
mod legacy_capture;
mod recording_format;
#[cfg(feature = "render")]
mod recording_frame;
mod recording_segment;
mod recording_settings;

#[cfg(feature = "render")]
pub use self::bevy_gym_recorder_plugin::{BevyGymRecorderPlugin, RecordingCallbackError};
pub use self::checkpoint_role::CheckpointRole;
pub use self::checkpoint_timeline::CheckpointTimeline;
#[cfg(feature = "render")]
pub use self::checkpoint_video::record_training_video_from_gif_command;
pub use self::error::RecordingError;
#[cfg(feature = "render")]
pub use self::legacy_capture::{encode_gif, spawn_mujoco_stage, GifCapture};
pub use self::recording_format::RecordingFormat;
#[cfg(feature = "render")]
pub use self::recording_frame::RecordingFrame;
pub use self::recording_segment::RecordingSegment;
pub use self::recording_settings::RecordingSettings;
