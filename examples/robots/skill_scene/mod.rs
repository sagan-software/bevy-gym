//! Read-only scene projection of a frozen RL lesson with explicit playback controls.

#[path = "../drone_model.rs"]
mod drone_model;
#[path = "../learning/encoding.rs"]
mod encoding;
#[path = "../learning/model.rs"]
mod model;
mod presentation;
mod session;
mod travel;
mod view;

use bevy::{asset::AssetMetaCheck, prelude::*};
use bevy_gym::robots::DroneHover;
use presentation::SceneSession;
use session::Session;
use sha2::{Digest, Sha256};

/// User playback mode; this never supplies an actuator command.
#[derive(Default)]
enum Playback {
    /// Wait for a step or resume request.
    #[default]
    Paused,
    /// Request one learned action per fixed timestep.
    Running,
}

/// Scene state keeps load failures visible without creating a fallback environment.
#[derive(Resource)]
struct Viewer {
    /// Fixed lesson selected by the executable, never by an agent controller.
    title: &'static str,
    /// Validated frozen session, or the diagnostic preventing playback.
    session: Result<SceneSession, String>,
    /// Frozen checkpoint identity and qualification state, fixed by the executable.
    checkpoint_label: String,
    /// Spectator's playback choice.
    playback: Playback,
}

/// Start one independent lesson with the recorded curriculum checkpoint.
pub(super) fn run(title: &'static str, factory: fn() -> DroneHover) {
    let bytes = include_bytes!("../../../docs/progress/drone-curriculum.mpk");
    let session = Session::load(bytes.to_vec(), factory, 42)
        .map(SceneSession::Control)
        .map_err(|error| error.to_string());
    start(
        title,
        session,
        checkpoint_label(bytes, "Frozen RL · curriculum"),
    );
}

/// Show the recorded failed RL travel candidate without claiming qualification.
pub(super) fn run_travel() {
    let bytes = include_bytes!("../../../docs/progress/drone-travel-trial.mpk");
    let session = travel::load(bytes.to_vec(), 42)
        .map(SceneSession::Travel)
        .map_err(|error| error.to_string());
    start(
        "Travel trial",
        session,
        checkpoint_label(bytes, "Unqualified RL · trial"),
    );
}

/// Derive the visible identity from the exact embedded policy record.
fn checkpoint_label(bytes: &[u8], status: &str) -> String {
    let digest = Sha256::digest(bytes);
    let hex = format!("{digest:x}");
    let prefix = hex.get(..12).expect("SHA-256 hex has 64 ASCII characters");
    format!("{status} {prefix}")
}

/// Share the established scene, assets and controls between independently runnable lessons.
fn start(title: &'static str, session: Result<SceneSession, String>, checkpoint_label: String) {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: format!("{title} | Bevy Gym"),
                        canvas: Some("#drone-canvas".to_owned()),
                        fit_canvas_to_parent: true,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: if cfg!(target_arch = "wasm32") {
                        "assets"
                    } else {
                        concat!(env!("CARGO_MANIFEST_DIR"), "/assets")
                    }
                    .to_owned(),
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                }),
        )
        .insert_resource(Time::<Fixed>::from_seconds(0.02))
        .insert_resource(Viewer {
            title,
            session,
            checkpoint_label,
            playback: Playback::Paused,
        })
        .add_systems(Startup, view::setup)
        .add_systems(FixedUpdate, advance)
        .add_systems(
            Update,
            (view::keyboard, view::project, view::target, view::refresh).chain(),
        )
        .run();
}

/// Fixed playback changes timing only; the policy owns every motor value.
fn advance(mut viewer: ResMut<'_, Viewer>) {
    if matches!(viewer.playback, Playback::Running) {
        if let Ok(session) = &mut viewer.session {
            session.step();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Display identity must follow loaded bytes, including the established SHA-256 test vector.
    #[test]
    fn checkpoint_label_uses_the_actual_bytes() {
        assert_eq!(checkpoint_label(b"abc", "Trial"), "Trial ba7816bf8f01");
        assert_eq!(
            checkpoint_label(
                include_bytes!("../../../docs/progress/drone-travel-trial.mpk"),
                "Unqualified RL · trial"
            ),
            "Unqualified RL · trial 0f5a36039389"
        );
    }
}
