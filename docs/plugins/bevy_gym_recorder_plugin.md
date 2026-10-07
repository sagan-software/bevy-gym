# BevyGymRecorderPlugin

`BevyGymRecorderPlugin` records the primary Bevy window without adding capture
logic to an environment. It supports a five-second GIF of one checkpoint and a
30-second MP4 that compares four dynamically selected checkpoints.

The MP4 timeline is fixed:

1. `0..5s`: best checkpoint
2. `5..10s`: first saved checkpoint
3. `10..15s`: checkpoint at 33 percent of the saved progression
4. `15..20s`: checkpoint at 66 percent of the saved progression
5. `20..30s`: best checkpoint

The selector uses the available numbered checkpoint files. If a short run has
fewer than four distinct checkpoints, it reuses the nearest available file.

Create the settings and add the plugin to the rendered application:

```rust,no_run
use std::path::{Path, PathBuf};

use bevy::prelude::*;
use bevy_gym::recording::{BevyGymRecorderPlugin, RecordingSettings};

fn add_training_video(
    app: &mut App,
    run_directory: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let settings = RecordingSettings::training_video(
        PathBuf::from("docs/videos/my-environment.mp4"),
        run_directory,
    )?;
    app.add_plugins(
        BevyGymRecorderPlugin::new(settings)
            .with_checkpoint_loader(|world, checkpoint| {
                // Replace the active policy and reset the visible environment.
                load_checkpoint(world, checkpoint)
            })
            .with_frame_driver(|world, frame| {
                // Advance the policy and synchronize visible entities.
                drive_frame(world, frame)
            }),
    );
    Ok(())
}

# fn load_checkpoint(_: &mut World, _: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) }
# fn drive_frame(_: &mut World, _: bevy_gym::recording::RecordingFrame) -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) }
```

`RecordingSettings::gif` records five seconds at 20 FPS.
`RecordingSettings::training_video` records 30 seconds at 30 FPS. The
`with_frames_per_second` method accepts rates of 15 FPS or higher.

The plugin waits for each screenshot to finish before it advances the capture
cursor. It creates numbered PNG frames in a temporary directory, invokes
FFmpeg, removes the temporary frames, and exits the Bevy application.
