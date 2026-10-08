//! Inspect real hover physics with manual motor commands.
//!
//! Run `cargo run --features robots --example drone-flight`.
//! The short `hover.rs` guide introduces the environment loop. Here, `Session`
//! retains that loop's result while Bevy projects it into a scene and controls.

#[path = "flight/controls.rs"]
mod controls;
#[path = "flight/scene.rs"]
mod scene;
#[path = "flight/session.rs"]
mod session;

use bevy::{asset::AssetMetaCheck, prelude::*};

use session::Session;

/// Open one seeded environment, paused so the reader can choose a command.
fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Drone hover | Bevy Gym".to_owned(),
                        canvas: Some("#drone-canvas".to_owned()),
                        fit_canvas_to_parent: true,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: asset_root().to_owned(),
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                }),
        )
        // One environment action advances 20 ms, regardless of the display rate.
        .insert_resource(Time::<Fixed>::from_seconds(0.02))
        .init_resource::<Session>()
        .add_systems(Startup, (scene::setup, controls::setup))
        .add_systems(FixedUpdate, advance)
        .add_systems(
            Update,
            (controls::interact, scene::project, controls::refresh).chain(),
        )
        .run();
}

/// Use packaged browser assets or this checkout's native asset directory.
const fn asset_root() -> &'static str {
    if cfg!(target_arch = "wasm32") {
        "assets"
    } else {
        concat!(env!("CARGO_MANIFEST_DIR"), "/assets")
    }
}

/// The environment owns every pose update; rendering reads its last observation.
fn advance(mut session: ResMut<'_, Session>) {
    session.advance();
}
