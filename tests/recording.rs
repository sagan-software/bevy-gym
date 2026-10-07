//! Public recording-plan contracts used by downstream example authors.

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde as _;
use serde_json as _;
use shakmaty as _;

use std::error::Error;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy_gym::recording::{CheckpointRole, CheckpointTimeline, RecordingSettings};
use tokio::runtime::{Builder, Runtime};

/// Build a current-thread runtime for test filesystem operations.
fn runtime() -> io::Result<Runtime> {
    Builder::new_current_thread().enable_all().build()
}

/// Create one directory tree for a test fixture.
fn create_directory(path: &Path) -> io::Result<()> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::create_dir_all(path))
}

/// Write one complete test fixture file.
fn write_file(path: &Path) -> io::Result<()> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::write(path, []))
}

/// Remove one complete test fixture directory.
fn remove_directory(path: &Path) -> io::Result<()> {
    let runtime = runtime()?;
    runtime.block_on(tokio::fs::remove_dir_all(path))
}

/// Create a collision-resistant temporary run directory for one test.
fn temporary_run() -> Result<PathBuf, Box<dyn Error>> {
    // Exercise the caller-visible boundary so the assertion covers the public contract.
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let run = std::env::temp_dir().join(format!(
        "bevy-gym-recording-test-{}-{nonce}",
        std::process::id()
    ));
    create_directory(&run.join("checkpoints"))?;
    Ok(run)
}

#[test]
fn progress_timeline_selects_dynamic_checkpoint_quantiles() {
    // Exercise the caller-visible boundary so the assertion covers the public contract.
    let run = temporary_run().expect("temporary run directory creates");
    for step in [0_u64, 10, 20, 30, 40, 50, 60] {
        write_file(&run.join("checkpoints").join(format!("step-{step:06}.mpk")))
            .expect("numbered checkpoint writes");
    }
    write_file(&run.join("best.mpk")).expect("best checkpoint writes");

    let timeline =
        CheckpointTimeline::training_progress(&run).expect("checkpoint timeline resolves");

    assert_eq!(timeline.duration().as_secs(), 30);
    assert_eq!(timeline.segments().len(), 5);
    let mut segments = timeline.segments().iter();
    let opening = segments.next().expect("opening best segment exists");
    let first = segments.next().expect("first checkpoint segment exists");
    let progress_33 = segments.next().expect("33 percent segment exists");
    let progress_66 = segments.next().expect("66 percent segment exists");
    let closing = segments.next().expect("closing best segment exists");
    assert_eq!(opening.role(), CheckpointRole::Best);
    assert_eq!(first.role(), CheckpointRole::First);
    assert_eq!(progress_33.role(), CheckpointRole::Progress33);
    assert_eq!(progress_66.role(), CheckpointRole::Progress66);
    assert_eq!(closing.role(), CheckpointRole::Best);
    assert!(first.checkpoint().ends_with("step-000000.mpk"));
    assert!(progress_33.checkpoint().ends_with("step-000020.mpk"));
    assert!(progress_66.checkpoint().ends_with("step-000040.mpk"));
    assert_eq!(closing.duration().as_secs(), 10);

    remove_directory(&run).expect("temporary run directory removes");
}

#[test]
fn progress_timeline_reuses_available_checkpoints_without_fixed_counts() {
    // Exercise the caller-visible boundary so the assertion covers the public contract.
    let run = temporary_run().expect("temporary run directory creates");
    write_file(&run.join("checkpoints").join("step-0.mpk")).expect("numbered checkpoint writes");
    write_file(&run.join("best.mpk")).expect("best checkpoint writes");

    let timeline =
        CheckpointTimeline::training_progress(&run).expect("checkpoint timeline resolves");

    assert_eq!(timeline.segments().len(), 5);
    for segment in timeline.segments().iter().skip(1).take(3) {
        assert!(segment.checkpoint().ends_with("step-0.mpk"));
    }

    remove_directory(&run).expect("temporary run directory removes");
}

#[test]
fn recorder_rejects_output_rates_below_fifteen_frames_per_second() {
    // Exercise the caller-visible boundary so the assertion covers the public contract.
    let run = temporary_run().expect("temporary run directory creates");
    let checkpoint = run.join("best.mpk");
    write_file(&checkpoint).expect("best checkpoint writes");

    let settings = RecordingSettings::gif(run.join("preview.gif"), checkpoint)
        .expect("valid GIF settings construct");

    settings.clone().with_frames_per_second(14).unwrap_err();
    assert_eq!(
        settings
            .with_frames_per_second(15)
            .expect("minimum frame rate is accepted")
            .frames_per_second(),
        15
    );

    remove_directory(&run).expect("temporary run directory removes");
}
