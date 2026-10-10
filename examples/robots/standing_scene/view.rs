//! Standing scene, read-only measurements and spectator playback controls.

use bevy::{prelude::*, scene::SceneInstanceReady};
use bevy_gym::{
    robots::{DroidActuator, DroidBody},
    EpisodeStatus,
};

use super::{rig::Rig, session::Session, Playback, Viewer};

/// Scene handle retained to report model loading failures.
#[derive(Resource)]
pub(super) struct Model(
    /// Asset handle used only to observe loading state.
    Handle<Scene>,
);

/// Playback and physics telemetry, including visible failure diagnostics.
#[derive(Component)]
pub(super) struct Readout;

/// The button label reflecting whether playback can resume or pause.
#[derive(Component)]
pub(super) struct PlayLabel;

/// Spectator commands cannot supply or modify policy actions.
#[derive(Clone, Copy)]
enum Control {
    /// Toggle fixed-rate playback.
    Play,
    /// Pause and request exactly one learned action.
    Step,
    /// Reset to the same seed and clear recurrent memory.
    Reset,
    /// Cycle presentation speed without changing physical timestep or policy inputs.
    Speed,
}

/// Construct the inspection scene and touch-sized controls.
pub(super) fn setup(
    mut commands: Commands<'_, '_>,
    assets: Res<'_, AssetServer>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
    viewer: Res<'_, Viewer>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.72, 0.77, 0.79)));
    let model = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/survival/mannequin.glb"));
    commands.insert_resource(Model(model.clone()));
    commands
        .spawn((
            SceneRoot(model),
            Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
        ))
        .observe(
            |_ready: On<'_, '_, SceneInstanceReady>, mut rig: ResMut<'_, Rig>| {
                *rig = Rig::AwaitingBind;
            },
        );
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(60.0, 60.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.55, 0.60, 0.56))),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(1.2))),
        MeshMaterial3d(materials.add(Color::srgb(0.84, 0.84, 0.77))),
        Transform::from_xyz(0.0, 0.002, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.7, 1.8, -3.2).looking_at(Vec3::new(0.0, 0.85, 0.0), Vec3::Y),
    ));
    controls(
        &mut commands,
        &assets,
        viewer.title,
        &viewer.checkpoint_label,
    );
}

/// Keep the scene visible between its telemetry and playback controls.
fn controls(commands: &mut Commands<'_, '_>, assets: &AssetServer, title: &str, checkpoint: &str) {
    let font = assets.load("fonts/MonaSans-VariableFont.ttf");
    commands
        .spawn(Node {
            width: percent(100),
            height: percent(100),
            padding: UiRect::all(px(16)),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        })
        .with_children(|root| {
            header(root, &font, title);
            footer(root, &font, checkpoint);
        });
}

/// Identify the lesson and retain a dedicated failure and telemetry readout.
fn header(root: &mut ChildSpawnerCommands<'_>, font: &Handle<Font>, title: &str) {
    root.spawn((
        Node {
            padding: UiRect::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            align_self: AlignSelf::FlexStart,
            max_width: percent(100),
            ..default()
        },
        BackgroundColor(Color::srgb(0.09, 0.10, 0.11)),
    ))
    .with_children(|panel| {
        panel.spawn((
            Text::new(title),
            TextFont {
                font: font.clone(),
                font_size: 24.0,
                ..default()
            },
        ));
        panel.spawn((
            Readout,
            Text::new("Loading checkpoint and model…"),
            TextFont {
                font: font.clone(),
                font_size: 15.0,
                ..default()
            },
        ));
    });
}

/// Group playback controls separately from the physics scene.
fn footer(root: &mut ChildSpawnerCommands<'_>, font: &Handle<Font>, checkpoint: &str) {
    root.spawn((
        Node {
            width: percent(100),
            padding: UiRect::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            align_self: AlignSelf::FlexStart,
            max_width: px(520),
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(Color::srgb(0.09, 0.10, 0.11)),
    ))
    .with_children(|panel| {
        panel
            .spawn(Node {
                flex_wrap: FlexWrap::Wrap,
                column_gap: px(8),
                row_gap: px(8),
                ..default()
            })
            .with_children(|row| {
                for (label, control) in [
                    ("Run [Space]", Control::Play),
                    ("Step [N]", Control::Step),
                    ("Reset [R]", Control::Reset),
                    ("Speed [V]", Control::Speed),
                ] {
                    button(row, label, control, font);
                }
            });
        panel.spawn((
            Text::new(format!("{checkpoint}\nMannequin by Quaternius · CC0")),
            TextFont {
                font: font.clone(),
                font_size: 13.0,
                ..default()
            },
        ));
    });
}

/// Use the same command path for pointer and keyboard input.
fn button(
    parent: &mut ChildSpawnerCommands<'_>,
    label: &str,
    control: Control,
    font: &Handle<Font>,
) {
    parent
        .spawn((
            Button,
            Node {
                min_height: px(44),
                padding: UiRect::axes(px(12), px(10)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.22, 0.24, 0.25)),
        ))
        .observe(
            move |mut press: On<'_, '_, Pointer<Press>>,
                  mut viewer: ResMut<'_, Viewer>,
                  rig: Res<'_, Rig>| {
                press.propagate(false);
                if press.button == PointerButton::Primary {
                    apply(control, &mut viewer, &rig);
                }
            },
        )
        .with_children(|parent| {
            let mut text = parent.spawn((
                Text::new(label),
                Pickable::IGNORE,
                TextFont {
                    font: font.clone(),
                    font_size: 15.0,
                    ..default()
                },
            ));
            if matches!(control, Control::Play) {
                text.insert(PlayLabel);
            }
        });
}

/// Pause before single-step or reset; neither operation supplies motor values.
fn apply(control: Control, viewer: &mut Viewer, rig: &Rig) {
    match control {
        Control::Speed => viewer.speed = viewer.speed.next(),
        Control::Play => {
            if !matches!(rig, Rig::Ready(_)) {
                return;
            }
            viewer.playback = match viewer.playback {
                Playback::Paused => Playback::Running,
                Playback::Running => Playback::Paused,
            }
        }
        Control::Step => {
            viewer.playback = Playback::Paused;
            if matches!(rig, Rig::Ready(_)) {
                if let Ok(session) = &mut viewer.session {
                    session.step();
                }
            }
        }
        Control::Reset => {
            viewer.playback = Playback::Paused;
            if let Ok(session) = &mut viewer.session {
                session.reset(42);
            }
        }
    }
}

/// Keyboard shortcuts are spectator input only.
pub(super) fn keyboard(
    keys: Res<'_, ButtonInput<KeyCode>>,
    mut viewer: ResMut<'_, Viewer>,
    rig: Res<'_, Rig>,
) {
    for (key, control) in [
        (KeyCode::Space, Control::Play),
        (KeyCode::KeyN, Control::Step),
        (KeyCode::KeyR, Control::Reset),
        (KeyCode::KeyV, Control::Speed),
    ] {
        if keys.just_pressed(key) {
            apply(control, &mut viewer, &rig);
        }
    }
}

/// Surface model and policy failures instead of drawing a seemingly active agent.
pub(super) fn refresh(
    viewer: Res<'_, Viewer>,
    rig: Res<'_, Rig>,
    assets: Res<'_, AssetServer>,
    model: Res<'_, Model>,
    mut text: Single<'_, '_, &mut Text, (With<Readout>, Without<PlayLabel>)>,
    mut play: Single<'_, '_, &mut Text, (With<PlayLabel>, Without<Readout>)>,
) {
    let play_label = match viewer.playback {
        Playback::Paused => "Run [Space]",
        Playback::Running => "Pause [Space]",
    };
    play_label.clone_into(&mut play.0);
    let model_state = assets.get_load_state(model.0.id());
    text.0 = match &viewer.session {
        Err(error) => format!("Checkpoint failed: {error}"),
        Ok(session) => {
            if let Some(error) = session.error() {
                format!("Inference stopped: {error}")
            } else if let Some(bevy::asset::LoadState::Failed(error)) = model_state {
                format!("Model failed: {error}")
            } else if let Rig::Failed(error) = &*rig {
                format!("Model mapping failed: {error}")
            } else if !matches!(*rig, Rig::Ready(_)) {
                "Loading model…".to_owned()
            } else {
                telemetry(session, &viewer.playback, viewer.speed)
            }
        }
    };
}

/// Format read-only measurements without equating episode completion with qualification.
fn telemetry(session: &Session, playback: &Playback, speed: super::Speed) -> String {
    let state = match session.status() {
        EpisodeStatus::Terminated => "Episode ended",
        EpisodeStatus::Truncated => "Time limit",
        EpisodeStatus::Continuing => match playback {
            Playback::Paused => "Paused",
            Playback::Running => "Running",
        },
    };
    let steps = session.steps();
    let horizon = session.horizon();
    let multiplier = speed.actions();
    let body = session.observation().body(DroidBody::Pelvis);
    let height = body.position().y;
    let velocity = body.linear_velocity().length();
    let torque = session.last_action().map_or(0.0, |action| {
        DroidActuator::ALL
            .into_iter()
            .map(|actuator| action.fraction(actuator).abs())
            .fold(0.0, f32::max)
    });
    format!("{state} · {steps}/{horizon} actions · {multiplier}x\nPelvis {height:.2} m · speed {velocity:.2} m/s\nMaximum torque fraction {torque:.2}")
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The complete overlay exposes checkpoint failures and model readiness before playback.
    #[test]
    fn scene_setup_and_readout_report_loading_and_failures() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Scene>()
            .init_asset::<Font>()
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_resource::<Rig>()
            .insert_resource(Viewer {
                title: "Standing trial",
                session: Err("invalid checkpoint".to_owned()),
                checkpoint_label: "Unqualified RL".to_owned(),
                playback: Playback::Paused,
                speed: super::super::Speed::One,
            })
            .add_systems(Startup, setup)
            .add_systems(Update, refresh);
        app.update();
        let readout = app
            .world_mut()
            .query_filtered::<Entity, With<Readout>>()
            .single(app.world())
            .expect("one readout");
        assert_eq!(
            app.world().get::<Text>(readout).expect("text").0,
            "Checkpoint failed: invalid checkpoint"
        );
        // A detached handle keeps the readout test independent of background asset I/O.
        app.insert_resource(Model(Handle::default()));
        app.world_mut().resource_mut::<Viewer>().session = Ok(super::super::session::load(
            include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec(),
            include_bytes!("../../../docs/progress/droid-standing-trial.json"),
            42,
        )
        .expect("RL candidate"));
        app.update();
        assert_eq!(
            app.world().get::<Text>(readout).expect("text").0,
            "Loading model…"
        );
        *app.world_mut().resource_mut::<Rig>() = Rig::Failed("missing pelvis".to_owned());
        app.update();
        assert_eq!(
            app.world().get::<Text>(readout).expect("text").0,
            "Model mapping failed: missing pelvis"
        );
        *app.world_mut().resource_mut::<Rig>() = super::super::rig::ready_fixture();
        app.world_mut().resource_mut::<Viewer>().playback = Playback::Running;
        app.update();
        assert!(app
            .world()
            .get::<Text>(readout)
            .expect("text")
            .0
            .contains("Running"));
    }

    /// A held shortcut cannot repeat an action; reset clears the learned episode memory.
    #[test]
    fn keyboard_shortcuts_run_once_per_press_and_reset_the_episode() {
        let session = super::super::session::load(
            include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec(),
            include_bytes!("../../../docs/progress/droid-standing-trial.json"),
            42,
        )
        .expect("RL candidate");
        let mut app = App::new();
        app.init_resource::<ButtonInput<KeyCode>>()
            .insert_resource(super::super::rig::ready_fixture())
            .insert_resource(Viewer {
                title: "Trial",
                session: Ok(session),
                checkpoint_label: String::new(),
                playback: Playback::Paused,
                speed: super::super::Speed::One,
            })
            .add_systems(Update, keyboard);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyN);
        app.update();
        assert_eq!(
            app.world()
                .resource::<Viewer>()
                .session
                .as_ref()
                .expect("session")
                .steps(),
            1
        );
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();
        assert_eq!(
            app.world()
                .resource::<Viewer>()
                .session
                .as_ref()
                .expect("session")
                .steps(),
            1
        );
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Space);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyV);
        app.update();
        assert!(matches!(
            app.world().resource::<Viewer>().playback,
            Playback::Running
        ));
        assert_eq!(app.world().resource::<Viewer>().speed.actions(), 4);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyR);
        app.update();
        let viewer = app.world().resource::<Viewer>();
        assert!(matches!(viewer.playback, Playback::Paused));
        let session = viewer.session.as_ref().expect("session");
        assert_eq!(session.steps(), 0);
        assert!(session.last_action().is_none());
    }

    /// Input changes timing/reset only, and an unavailable model cannot start playback.
    #[test]
    fn controls_preserve_policy_ownership_and_clear_episode_memory() {
        let load = || {
            super::super::session::load(
                include_bytes!("../../../docs/progress/droid-standing-trial.mpk").to_vec(),
                include_bytes!("../../../docs/progress/droid-standing-trial.json"),
                42,
            )
            .expect("RL candidate")
        };
        let mut viewer = Viewer {
            title: "Trial",
            session: Ok(load()),
            checkpoint_label: String::new(),
            playback: Playback::Paused,
            speed: super::super::Speed::One,
        };
        apply(Control::Play, &mut viewer, &Rig::Loading);
        assert!(matches!(viewer.playback, Playback::Paused));
        apply(Control::Step, &mut viewer, &Rig::Loading);
        assert_eq!(viewer.session.as_ref().expect("session").steps(), 0);
        let rig = super::super::rig::ready_fixture();
        apply(Control::Play, &mut viewer, &rig);
        assert!(matches!(viewer.playback, Playback::Running));
        apply(Control::Play, &mut viewer, &rig);
        assert!(matches!(viewer.playback, Playback::Paused));
        apply(Control::Speed, &mut viewer, &rig);
        assert_eq!(viewer.speed.actions(), 4);
        apply(Control::Step, &mut viewer, &rig);
        let mut direct = load();
        direct.step();
        assert_eq!(
            viewer.session.as_ref().expect("session").observation(),
            direct.observation()
        );
        assert!(telemetry(
            viewer.session.as_ref().expect("session"),
            &Playback::Running,
            viewer.speed
        )
        .contains("Running"));
        apply(Control::Reset, &mut viewer, &rig);
        let session = viewer.session.as_ref().expect("session");
        assert_eq!(session.observation(), load().observation());
        assert!(session.last_action().is_none());
        assert!(telemetry(session, &Playback::Paused, viewer.speed).contains("Paused"));
        let session = viewer.session.as_mut().expect("session");
        for _ in 0..session.horizon() {
            session.step();
            assert!(session.error().is_none());
            if session.status().is_done() {
                break;
            }
        }
        assert!(telemetry(session, &Playback::Running, viewer.speed).contains("Episode ended"));
        viewer.session = Err("invalid checkpoint".to_owned());
        apply(Control::Step, &mut viewer, &rig);
        apply(Control::Reset, &mut viewer, &rig);
        assert!(viewer.session.is_err());
    }
}
