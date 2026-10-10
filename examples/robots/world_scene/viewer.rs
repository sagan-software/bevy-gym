//! Inspect six frozen RL policies through one-way physical model projection.

mod camera;
mod player;
mod rig;
mod view;

use crate::standing::world_session::Session;
use bevy::{asset::AssetMetaCheck, prelude::*};
use player::Player;

/// Load both recorded policy identities and render the common twenty-second diagnostic.
pub(super) fn run() {
    let drone = include_bytes!("../../../docs/progress/drone-hover.mpk");
    let droid = include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk");
    let metadata =
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.json");
    let session =
        Session::load(drone.to_vec(), droid.to_vec(), metadata).map_err(|error| error.to_string());
    let identities = view::Identities::new(&session, droid);
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Shared world trial | Bevy Gym".to_owned(),
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
        .insert_resource(Player::new(session))
        .insert_resource(identities)
        .init_resource::<camera::Mode>()
        .add_systems(Startup, view::setup)
        .add_systems(FixedUpdate, advance)
        .add_systems(
            Update,
            (
                rig::asset_failures,
                view::keyboard,
                view::frame_camera,
                rig::project,
                view::project_labels,
                view::refresh,
            )
                .chain(),
        )
        .add_systems(PostUpdate, rig::capture.after(TransformSystems::Propagate))
        .run();
}

/// Readiness gates every common frame; the player owns the only session mutation.
fn advance(mut player: ResMut<'_, Player>, visuals: Query<'_, '_, &rig::Visual>) {
    player.advance(rig::ready(&visuals));
}
