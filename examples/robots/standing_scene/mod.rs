//! Frozen physical standing trial with presentation-only playback controls.

mod projection;
mod rig;
mod session;
mod view;

use bevy::{asset::AssetMetaCheck, prelude::*};
use sha2::{Digest, Sha256};

/// Spectator playback mode never supplies joint torques.
#[derive(Default)]
enum Playback {
    /// Wait for a step or resume request.
    #[default]
    Paused,
    /// Consume frozen policy actions at the selected presentation rate.
    Running,
}

/// Closed playback rates multiply action count without changing the 20 ms physical step.
#[derive(Clone, Copy, Default)]
enum Speed {
    /// One policy action per fixed presentation tick.
    #[default]
    One,
    /// Four policy actions per fixed presentation tick.
    Four,
    /// Sixteen policy actions per fixed presentation tick.
    Sixteen,
}
impl Speed {
    /// Read the number of policy actions per fixed presentation tick.
    const fn actions(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
            Self::Sixteen => 16,
        }
    }
    /// Cycle the available spectator rates.
    const fn next(self) -> Self {
        match self {
            Self::One => Self::Four,
            Self::Four => Self::Sixteen,
            Self::Sixteen => Self::One,
        }
    }
}

/// Validated physics/inference session and independent spectator state.
#[derive(Resource)]
struct Viewer {
    /// Lesson identity selected by the executable.
    title: &'static str,
    /// Failed loading prevents environment playback.
    session: Result<session::Session, String>,
    /// Byte identity and truthful qualification state.
    checkpoint_label: String,
    /// User's current presentation mode.
    playback: Playback,
    /// User's current presentation speed.
    speed: Speed,
}

/// Start the preserved failed PPO candidate; this is not a qualified standing lesson.
pub(super) fn run() {
    let bytes = include_bytes!("../../../docs/progress/droid-standing-trial.mpk");
    let metadata = include_bytes!("../../../docs/progress/droid-standing-trial.json");
    let session = session::load(bytes.to_vec(), metadata, 42).map_err(|error| error.to_string());
    let digest = Sha256::digest(bytes);
    let hex = format!("{digest:x}");
    let prefix = hex
        .get(..12)
        .expect("SHA-256 hex has sixty-four ASCII characters");
    let checkpoint_label = session.as_ref().map_or_else(
        |_| format!("Unqualified checkpoint · {prefix}"),
        |session| {
            let update = session.checkpoint_update();
            format!("Unqualified RL · PPO update {update} · {prefix}")
        },
    );
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Standing trial | Bevy Gym".to_owned(),
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
            title: "Standing trial",
            session,
            checkpoint_label,
            playback: Playback::Paused,
            speed: Speed::One,
        })
        .init_resource::<rig::Rig>()
        .add_systems(Startup, view::setup)
        .add_systems(FixedUpdate, advance)
        .add_systems(
            Update,
            (view::keyboard, rig::project, view::refresh).chain(),
        )
        .add_systems(PostUpdate, rig::capture.after(TransformSystems::Propagate))
        .run();
}

/// Advance only usable sessions; every substep infers a new action from updated physical state.
fn advance(mut viewer: ResMut<'_, Viewer>, rig: Res<'_, rig::Rig>) {
    if matches!(*rig, rig::Rig::Ready(_)) && matches!(viewer.playback, Playback::Running) {
        let count = viewer.speed.actions();
        if let Ok(session) = &mut viewer.session {
            for _ in 0..count {
                session.step();
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Acceleration changes only timing: four actions equal four ordinary physical steps.
    #[test]
    fn playback_requires_a_ready_model_and_preserves_policy_steps() {
        let load = || {
            session::load(
                include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec(),
                include_bytes!("../../../docs/progress/droid-standing-trial.json"),
                42,
            )
            .expect("RL candidate")
        };
        let mut direct = load();
        let mut app = App::new();
        app.init_resource::<rig::Rig>();
        app.insert_resource(Viewer {
            title: "Trial",
            session: Ok(load()),
            checkpoint_label: String::new(),
            playback: Playback::Paused,
            speed: Speed::Four,
        })
        .add_systems(Update, advance);
        app.update();
        assert_eq!(
            app.world()
                .resource::<Viewer>()
                .session
                .as_ref()
                .expect("session")
                .steps(),
            0
        );
        app.world_mut().resource_mut::<Viewer>().playback = Playback::Running;
        app.update();
        assert_eq!(
            app.world()
                .resource::<Viewer>()
                .session
                .as_ref()
                .expect("session")
                .steps(),
            0
        );
        *app.world_mut().resource_mut::<rig::Rig>() = rig::ready_fixture();
        app.world_mut().resource_mut::<Viewer>().playback = Playback::Paused;
        app.update();
        assert_eq!(
            app.world()
                .resource::<Viewer>()
                .session
                .as_ref()
                .expect("session")
                .steps(),
            0
        );
        app.world_mut().resource_mut::<Viewer>().playback = Playback::Running;
        app.update();
        for _ in 0..4 {
            direct.step();
        }
        let viewer = app.world().resource::<Viewer>();
        let session = viewer.session.as_ref().expect("session");
        assert_eq!(session.steps(), 4);
        assert_eq!(session.observation(), direct.observation());
        assert_eq!(session.last_action(), direct.last_action());
        assert_eq!(Speed::One.next().actions(), 4);
        assert_eq!(Speed::Four.next().actions(), 16);
        assert_eq!(Speed::Sixteen.next().actions(), 1);
        app.world_mut().resource_mut::<Viewer>().session = Err("invalid checkpoint".to_owned());
        app.update();
        assert!(app.world().resource::<Viewer>().session.is_err());
    }
}
