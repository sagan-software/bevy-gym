//! Read-only scene projection of a frozen RL lesson with explicit playback controls.

#[path = "../drone_model.rs"]
mod drone_model;
#[path = "../learning/encoding.rs"]
mod encoding;
#[path = "../learning/model.rs"]
mod model;
mod session;
mod view;

use bevy::{asset::AssetMetaCheck, prelude::*};
use bevy_gym::robots::DroneHover;
use session::Session;

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
    session: Result<Session, String>,
    /// Spectator's playback choice.
    playback: Playback,
}

/// Start one independent lesson with the recorded curriculum checkpoint.
pub(super) fn run(title: &'static str, factory: fn() -> DroneHover) {
    let session = Session::load(
        include_bytes!("../../../docs/progress/drone-curriculum.mpk").to_vec(),
        factory,
        42,
    )
    .map_err(|error| error.to_string());
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
            playback: Playback::Paused,
        })
        .add_systems(Startup, view::setup)
        .add_systems(FixedUpdate, advance)
        .add_systems(
            Update,
            (view::keyboard, view::project, view::refresh).chain(),
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
