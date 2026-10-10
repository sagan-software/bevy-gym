//! Licensed mesh, read-only telemetry, and spectator playback controls.

use bevy::prelude::*;
use bevy_gym::{robots::DroneMotor, EpisodeStatus};
use std::fmt::Write;

use super::{drone_model, Playback, Viewer};

/// Transform projected from the authoritative physics observation.
#[derive(Component)]
pub(super) struct Body;

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
    let model = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/drone.glb"));
    commands.insert_resource(Model(model.clone()));
    commands
        .spawn((Body, Transform::default(), Visibility::default()))
        .with_children(|parent| {
            parent.spawn((SceneRoot(model), drone_model::model_alignment()));
        });
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
        Transform::from_xyz(2.2, 3.4, -3.2).looking_at(Vec3::new(0.0, 2.0, 0.0), Vec3::Y),
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
            padding: UiRect::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            align_self: AlignSelf::FlexStart,
            max_width: percent(100),
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
                ] {
                    button(row, label, control, font);
                }
            });
        panel.spawn((
            Text::new(format!("{checkpoint}\nDrone by NateGazzard · CC BY 3.0")),
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
            move |mut press: On<'_, '_, Pointer<Press>>, mut viewer: ResMut<'_, Viewer>| {
                press.propagate(false);
                if press.button == PointerButton::Primary {
                    apply(control, &mut viewer);
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
fn apply(control: Control, viewer: &mut Viewer) {
    match control {
        Control::Play => {
            viewer.playback = match viewer.playback {
                Playback::Paused => Playback::Running,
                Playback::Running => Playback::Paused,
            }
        }
        Control::Step => {
            viewer.playback = Playback::Paused;
            if let Ok(session) = &mut viewer.session {
                session.step();
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
pub(super) fn keyboard(keys: Res<'_, ButtonInput<KeyCode>>, mut viewer: ResMut<'_, Viewer>) {
    for (key, control) in [
        (KeyCode::Space, Control::Play),
        (KeyCode::KeyN, Control::Step),
        (KeyCode::KeyR, Control::Reset),
    ] {
        if keys.just_pressed(key) {
            apply(control, &mut viewer);
        }
    }
}

/// Project physical pose and motor telemetry without writing into the environment.
pub(super) fn project(
    viewer: Res<'_, Viewer>,
    mut body: Single<'_, '_, &mut Transform, With<Body>>,
    mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<Body>)>,
    mut rotors: Query<'_, '_, (&Name, &mut Transform), (Without<Body>, Without<Camera3d>)>,
) {
    let Ok(session) = &viewer.session else {
        return;
    };
    let observation = session.observation();
    body.translation = observation.position();
    body.rotation = observation.orientation();
    let (focus, scale) = session.destination().map_or_else(
        || (observation.position(), 1.0),
        |goal| {
            let distance = observation.position().distance(goal.position());
            (
                (observation.position() + goal.position()) * 0.5,
                distance.mul_add(0.35, 1.0),
            )
        },
    );
    **camera = Transform::from_translation(focus + Vec3::new(2.2, 1.4, -3.2) * scale)
        .looking_at(focus, Vec3::Y);
    if let Some(action) = session.last_action() {
        // Illustration only: the actuator command is a force fraction, not rotor RPM.
        let phase = session.steps() as f32 * 0.02 * 200.0;
        let [fl, fr, br, bl] = action.fractions();
        for (name, mut transform) in &mut rotors {
            if let Some((pivot, motor, sign)) = drone_model::rotor(name.as_str()) {
                let fraction = match motor {
                    DroneMotor::FrontLeft => fl,
                    DroneMotor::FrontRight => fr,
                    DroneMotor::RearRight => br,
                    DroneMotor::RearLeft => bl,
                };
                *transform = drone_model::spin_about(pivot, -phase * fraction * sign);
            }
        }
    }
}

/// Draw task geometry without adding colliders or selecting policy actions.
pub(super) fn target(viewer: Res<'_, Viewer>, mut gizmos: Gizmos<'_, '_>) {
    let Ok(session) = &viewer.session else {
        return;
    };
    let Some(goal) = session.destination() else {
        return;
    };
    let position = goal.position();
    let heading = goal.heading();
    let color = Color::srgb(0.07, 0.23, 0.44);
    // The half-metre sphere shows the position gate; the arrow shows desired horizontal heading.
    gizmos.sphere(Isometry3d::from_translation(position), 0.5, color);
    gizmos.arrow(
        position,
        position + Vec3::new(heading.x, 0.0, heading.y) * 0.8,
        color,
    );
}

/// Surface model and policy failures instead of drawing a seemingly active agent.
pub(super) fn refresh(
    viewer: Res<'_, Viewer>,
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
            } else {
                telemetry(session, &viewer.playback)
            }
        }
    };
}

/// Format read-only measurements without equating episode completion with qualification.
fn telemetry(session: &super::presentation::SceneSession, playback: &Playback) -> String {
    let state = match session.status() {
        EpisodeStatus::Terminated => "Crashed",
        EpisodeStatus::Truncated => "Episode complete",
        EpisodeStatus::Continuing => match playback {
            Playback::Paused => "Paused",
            Playback::Running => "Running",
        },
    };
    let steps = session.steps();
    let horizon = session.horizon();
    let distance = session.observation().position().distance(session.target());
    let mut text = format!("{state} · {steps}/{horizon} actions\nTarget distance {distance:.2} m");
    if let Some(heading) = session.heading_error() {
        let degrees = heading.to_degrees();
        let speed = session.observation().linear_velocity().length();
        write!(
            &mut text,
            "\nHeading error {degrees:.1}° · speed {speed:.2} m/s"
        )
        .expect("formatting into a String is infallible");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_scene::session::Session;
    use bevy_gym::robots::DroneHover;

    /// Playback controls preserve the next policy action and pause at reset.
    #[test]
    fn controls_change_timing_without_selecting_motor_values() {
        let mut viewer = Viewer {
            title: "Hover",
            session: Session::load(
                include_bytes!("../../../docs/progress/drone-curriculum.mpk").to_vec(),
                DroneHover::default,
                42,
            )
            .map(super::super::presentation::SceneSession::Control)
            .map_err(|error| error.to_string()),
            checkpoint_label: "test checkpoint".to_owned(),
            playback: Playback::Paused,
        };
        let initial = viewer.session.as_ref().expect("valid session");
        assert_eq!(initial.horizon(), 500);
        assert!(initial.destination().is_none());
        assert!(initial.error().is_none());
        assert!(telemetry(initial, &viewer.playback).starts_with("Paused · 0/500 actions"));
        apply(Control::Play, &mut viewer);
        assert!(matches!(viewer.playback, Playback::Running));
        apply(Control::Play, &mut viewer);
        assert!(matches!(viewer.playback, Playback::Paused));
        apply(Control::Step, &mut viewer);
        let session = viewer.session.as_ref().expect("valid session");
        assert_eq!(session.steps(), 1);
        let first = session.last_action();
        apply(Control::Reset, &mut viewer);
        assert!(matches!(viewer.playback, Playback::Paused));
        assert_eq!(viewer.session.as_ref().expect("reset session").steps(), 0);
        apply(Control::Step, &mut viewer);
        assert_eq!(
            viewer
                .session
                .as_ref()
                .expect("replayed session")
                .last_action(),
            first
        );
        viewer.session = Err("invalid checkpoint".to_owned());
        for control in [Control::Play, Control::Step, Control::Reset] {
            apply(control, &mut viewer);
            assert_eq!(
                viewer.session.as_ref().err().map(String::as_str),
                Some("invalid checkpoint")
            );
        }
    }

    /// Travel playback and telemetry use the same immutable task and episode limits.
    #[test]
    fn travel_controls_preserve_actions_and_show_the_full_horizon() {
        use super::super::{presentation::SceneSession, travel};
        let mut viewer = Viewer {
            title: "Travel trial",
            session: travel::load(
                include_bytes!("../../../docs/progress/drone-travel-trial.mpk").to_vec(),
                42,
            )
            .map(SceneSession::Travel)
            .map_err(|error| error.to_string()),
            checkpoint_label: "Unqualified RL trial".to_owned(),
            playback: Playback::Paused,
        };
        let session = viewer.session.as_ref().expect("trial loads");
        assert_eq!(session.horizon(), 1_000);
        assert!(session.destination().is_some());
        assert!(session.error().is_none());
        let text = telemetry(session, &viewer.playback);
        assert!(text.starts_with("Paused · 0/1000 actions"));
        assert!(text.contains("Heading error"));
        assert!(text.contains("m/s"));
        apply(Control::Step, &mut viewer);
        let first = viewer.session.as_ref().expect("stepped").last_action();
        apply(Control::Reset, &mut viewer);
        assert_eq!(viewer.session.as_ref().expect("reset").steps(), 0);
        apply(Control::Step, &mut viewer);
        assert_eq!(
            viewer.session.as_ref().expect("replay").last_action(),
            first
        );
        apply(Control::Play, &mut viewer);
        assert!(
            telemetry(viewer.session.as_ref().expect("running"), &viewer.playback)
                .starts_with("Running")
        );
        let session = viewer.session.as_mut().expect("valid trial");
        while !session.status().is_done() {
            session.step();
        }
        assert_eq!(session.steps(), 1_000);
        assert!(telemetry(session, &viewer.playback).starts_with("Episode complete"));
        session.step();
        assert_eq!(session.steps(), 1_000);
    }

    /// A real failed RL episode must be identified as a crash, never as successful qualification.
    #[test]
    fn failed_trial_is_reported_as_crashed() {
        use super::super::{presentation::SceneSession, travel};
        let mut session = SceneSession::Travel(
            travel::load(
                include_bytes!("../../../docs/progress/drone-travel-trial.mpk").to_vec(),
                0,
            )
            .expect("recorded trial"),
        );
        while !session.status().is_done() {
            session.step();
        }
        assert!(session.status().is_terminal());
        assert!(session.steps() < session.horizon());
        assert!(telemetry(&session, &Playback::Paused).starts_with("Crashed"));
    }
}
