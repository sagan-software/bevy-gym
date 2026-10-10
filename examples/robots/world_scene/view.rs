//! Shared-world presentation, truthful measurements and spectator-only controls.

use super::{
    camera,
    player::{Command, Player},
    rig::{self, Visual},
};
use crate::standing::world_session::Session;
use bevy::{prelude::*, window::PrimaryWindow};
use bevy_gym::robots::{DroidActuator, DroidBody, RobotId, RobotSlot};
use sha2::{Digest, Sha256};
use std::fmt::Write;

/// Retained licensed scene handles expose loading failures without substituting meshes.
#[derive(Resource)]
pub(super) struct Models {
    /// Drone scene shared by three independently instantiated model roots.
    pub(super) drone: Handle<Scene>,
    /// Mannequin scene shared by three root-scoped skeleton mappings.
    pub(super) droid: Handle<Scene>,
}

/// Identities derived from validated session bytes before the UI is created.
#[derive(Resource)]
pub(super) struct Identities(
    /// Frozen PPO update and byte identities for the two teams.
    String,
);

impl Identities {
    /// Only a successfully validated session may display checkpoint provenance.
    pub(super) fn new(session: &Result<Session, String>, droid_bytes: &[u8]) -> Self {
        Self(session.as_ref().map_or_else(
            |_| "Checkpoints unavailable".to_owned(),
            |session| {
                let drone = session.drone_digest();
                let droid = Sha256::digest(droid_bytes);
                let drone = format!("{drone:x}");
                let droid = format!("{droid:x}");
                let update = session.droid_record().update();
                format!("Drones · PPO 280 · {drone:.12}\nDroids · PPO {update} · {droid:.12}")
            },
        ))
    }
}

/// Dedicated state and selected-robot telemetry.
#[derive(Component)]
pub(super) struct Readout;

/// A control label follows current playback or camera state.
#[derive(Component)]
pub(super) struct ButtonLabel(
    /// Closed spectator operation represented by this label.
    Control,
);

/// A read-only world label identifies one of six physical robots.
#[derive(Component)]
pub(super) struct RobotLabel(
    /// Physical identity whose position is borrowed for this text.
    RobotId,
);

/// Controls have no route into actor inputs or actuator requests.
#[derive(Clone, Copy)]
enum Control {
    /// Apply a timing or reset command through the tested player.
    Playback(Command),
    /// Cycle the overview and six individual follow targets.
    View,
    /// Toggle keyboard-driven spectator camera movement.
    Free,
}

/// Construct the existing visual system around all six original licensed models.
pub(super) fn setup(
    mut commands: Commands<'_, '_>,
    assets: Res<'_, AssetServer>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
    identities: Res<'_, Identities>,
    player: Res<'_, Player>,
    mode: Res<'_, camera::Mode>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.72, 0.77, 0.79)));
    let drone = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/drone.glb"));
    let droid = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/survival/mannequin.glb"));
    let font = assets.load("fonts/MonaSans-VariableFont.ttf");
    models(&mut commands, &drone, &droid);
    identity_labels(&mut commands, &font);
    pads(&mut commands, &mut meshes, &mut materials, &player);
    scenery(&mut commands, &mut meshes, &mut materials);
    controls(&mut commands, &font, &identities.0, &player, *mode);
    commands.insert_resource(Models { drone, droid });
}

/// Instantiate each licensed model at its original bind origin, hidden until projection succeeds.
fn models(commands: &mut Commands<'_, '_>, drone: &Handle<Scene>, droid: &Handle<Scene>) {
    for slot in RobotSlot::ALL {
        commands
            .spawn((
                SceneRoot(drone.clone()),
                Visual::drone(slot),
                crate::drone_model::model_alignment(),
                Visibility::Hidden,
            ))
            .observe(rig::scene_ready);
        commands
            .spawn((
                SceneRoot(droid.clone()),
                Visual::droid(slot),
                Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI)),
                Visibility::Hidden,
            ))
            .observe(rig::scene_ready);
    }
}

/// Create presentation text for all six identities; projection supplies logical screen coordinates.
fn identity_labels(commands: &mut Commands<'_, '_>, font: &Handle<Font>) {
    for slot in RobotSlot::ALL {
        for id in [RobotId::Drone(slot), RobotId::Droid(slot)] {
            commands.spawn((
                RobotLabel(id),
                Text::new(camera::Mode::Follow(id).label()),
                TextFont {
                    font: font.clone(),
                    font_size: 13.0,
                    ..default()
                },
                TextColor(Color::srgb(0.09, 0.10, 0.11)),
                Node {
                    position_type: PositionType::Absolute,
                    width: px(90),
                    ..default()
                },
                TextLayout::new_with_justify(Justify::Center),
                Pickable::IGNORE,
                GlobalZIndex(-1),
                Visibility::Hidden,
            ));
        }
    }
}

/// Ground pads mark immutable reset positions without adding collision geometry or choosing actions.
fn pads(
    commands: &mut Commands<'_, '_>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    player: &Player,
) {
    let Ok(session) = player.session() else {
        return;
    };
    for slot in RobotSlot::ALL {
        for position in [
            session.snapshot().drone(slot).position(),
            session
                .snapshot()
                .droid(slot)
                .body(DroidBody::Pelvis)
                .position(),
        ] {
            commands.spawn((
                Mesh3d(meshes.add(Circle::new(1.2))),
                MeshMaterial3d(materials.add(Color::srgb(0.84, 0.84, 0.77))),
                Transform::from_xyz(position.x, 0.002, position.z)
                    .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
            ));
        }
    }
}

/// Retain the incumbent floor, lighting and inspection camera.
fn scenery(
    commands: &mut Commands<'_, '_>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(60.0, 60.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.55, 0.60, 0.56))),
    ));
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, -3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
        ..default()
    });
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_3,
            ..default()
        }),
        Transform::from_xyz(12.0, 8.0, -14.0).looking_at(Vec3::Y, Vec3::Y),
    ));
}

/// Place readable state above a scene and touch-sized controls below it.
fn controls(
    commands: &mut Commands<'_, '_>,
    font: &Handle<Font>,
    identities: &str,
    player: &Player,
    mode: camera::Mode,
) {
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
            header(root, font);
            footer(root, font, identities, player, mode);
        });
}

/// Identify the trial and retain a dedicated state and failure readout.
fn header(root: &mut ChildSpawnerCommands<'_>, font: &Handle<Font>) {
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
            Text::new("Shared world trial"),
            TextFont {
                font: font.clone(),
                font_size: 24.0,
                ..default()
            },
        ));
        panel.spawn((
            Readout,
            Text::new("Loading six models…\nUnqualified for competition"),
            TextFont {
                font: font.clone(),
                font_size: 15.0,
                ..default()
            },
        ));
    });
}

/// Group timing controls, camera selection and the two frozen checkpoint identities.
fn footer(
    root: &mut ChildSpawnerCommands<'_>,
    font: &Handle<Font>,
    identities: &str,
    player: &Player,
    mode: camera::Mode,
) {
    root.spawn((
        Node {
            width: percent(100),
            max_width: px(600),
            padding: UiRect::all(px(12)),
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            align_self: AlignSelf::FlexStart,
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(Color::srgb(0.09, 0.10, 0.11)),
    ))
    .with_children(|panel| {
        control_row(
            panel,
            font,
            &[
                Control::Playback(Command::Play),
                Control::Playback(Command::Step),
                Control::Playback(Command::Reset),
                Control::Playback(Command::Speed),
            ],
            player,
            mode,
        );
        control_row(panel, font, &[Control::View, Control::Free], player, mode);
        panel.spawn((
            Node {
                flex_shrink: 0.0,
                ..default()
            },
            Text::new(identities),
            TextFont {
                font: font.clone(),
                font_size: 13.0,
                ..default()
            },
        ));
    });
}

/// Keep each wrapped control row at its content height so checkpoint text remains inside the panel.
fn control_row(
    parent: &mut ChildSpawnerCommands<'_>,
    font: &Handle<Font>,
    controls: &[Control],
    player: &Player,
    mode: camera::Mode,
) {
    parent
        .spawn(Node {
            width: percent(100),
            flex_shrink: 0.0,
            flex_wrap: FlexWrap::Wrap,
            column_gap: px(8),
            row_gap: px(8),
            ..default()
        })
        .with_children(|row| {
            for control in controls {
                button(row, font, *control, label(*control, player, mode));
            }
        });
}

/// Pointer and keyboard commands enter the same tested player boundary.
fn button(
    parent: &mut ChildSpawnerCommands<'_>,
    font: &Handle<Font>,
    control: Control,
    label: String,
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
                  mut player: ResMut<'_, Player>,
                  mut mode: ResMut<'_, camera::Mode>,
                  visuals: Query<'_, '_, &Visual>| {
                press.propagate(false);
                if press.button == PointerButton::Primary {
                    apply(control, &mut player, &mut mode, rig::ready(&visuals));
                }
            },
        )
        .with_children(|parent| {
            parent.spawn((
                ButtonLabel(control),
                Text::new(label),
                Pickable::IGNORE,
                TextFont {
                    font: font.clone(),
                    font_size: 15.0,
                    ..default()
                },
            ));
        });
}

/// Resolve control state without inventing policy values or a qualification result.
fn label(control: Control, player: &Player, mode: camera::Mode) -> String {
    match control {
        Control::Playback(Command::Play) => if player.running() {
            "Pause [Space]"
        } else {
            "Run [Space]"
        }
        .to_owned(),
        Control::Playback(Command::Step) => "Step [N]".to_owned(),
        Control::Playback(Command::Reset) => "Reset [R]".to_owned(),
        Control::Playback(Command::Speed) => {
            let rate = player.rate();
            format!("Speed: {rate}× [V]")
        }
        Control::View => {
            let target = mode.label();
            format!("View: {target} [C]")
        }
        Control::Free => "Free camera [F]".to_owned(),
    }
}

/// Timing commands and camera commands have separate destinations.
fn apply(control: Control, player: &mut Player, mode: &mut camera::Mode, ready: bool) {
    match control {
        Control::Playback(command) => player.apply(command, ready),
        Control::View => *mode = mode.next(),
        Control::Free => {
            *mode = if *mode == camera::Mode::Free {
                camera::Mode::All
            } else {
                camera::Mode::Free
            }
        }
    }
}

/// Shortcuts select presentation commands and any of six follow targets.
pub(super) fn keyboard(
    keys: Res<'_, ButtonInput<KeyCode>>,
    mut player: ResMut<'_, Player>,
    mut mode: ResMut<'_, camera::Mode>,
    visuals: Query<'_, '_, &Visual>,
) {
    let ready = rig::ready(&visuals);
    for (key, control) in [
        (KeyCode::Space, Control::Playback(Command::Play)),
        (KeyCode::KeyN, Control::Playback(Command::Step)),
        (KeyCode::KeyR, Control::Playback(Command::Reset)),
        (KeyCode::KeyV, Control::Playback(Command::Speed)),
        (KeyCode::KeyC, Control::View),
        (KeyCode::KeyF, Control::Free),
    ] {
        if keys.just_pressed(key) {
            apply(control, &mut player, &mut mode, ready);
        }
    }
    if keys.just_pressed(KeyCode::Digit0) {
        *mode = camera::Mode::All;
    }
    for (key, id) in [
        (KeyCode::Digit1, RobotId::Drone(RobotSlot::First)),
        (KeyCode::Digit2, RobotId::Drone(RobotSlot::Second)),
        (KeyCode::Digit3, RobotId::Drone(RobotSlot::Third)),
        (KeyCode::Digit4, RobotId::Droid(RobotSlot::First)),
        (KeyCode::Digit5, RobotId::Droid(RobotSlot::Second)),
        (KeyCode::Digit6, RobotId::Droid(RobotSlot::Third)),
    ] {
        if keys.just_pressed(key) {
            *mode = camera::Mode::Follow(id);
        }
    }
}

/// Fit physical bounds or move only the free camera; the session is borrowed read-only.
pub(super) fn frame_camera(
    player: Res<'_, Player>,
    mode: Res<'_, camera::Mode>,
    keys: Res<'_, ButtonInput<KeyCode>>,
    time: Res<'_, Time>,
    window: Single<'_, '_, &Window, With<PrimaryWindow>>,
    mut camera: Single<'_, '_, (&Projection, &mut Transform), With<Camera3d>>,
) {
    let (projection, transform) = &mut *camera;
    if *mode == camera::Mode::Free {
        // Three metres/second and 0.9 radians/second affect presentation only.
        let delta = time.delta_secs();
        let mut movement = Vec3::ZERO;
        for (key, direction) in [
            (KeyCode::KeyW, Vec3::NEG_Z),
            (KeyCode::KeyS, Vec3::Z),
            (KeyCode::KeyA, Vec3::NEG_X),
            (KeyCode::KeyD, Vec3::X),
        ] {
            if keys.pressed(key) {
                movement += transform.rotation * direction;
            }
        }
        if keys.pressed(KeyCode::KeyQ) {
            movement += Vec3::Y;
        }
        if keys.pressed(KeyCode::KeyE) {
            movement -= Vec3::Y;
        }
        transform.translation += movement.normalize_or_zero() * delta * 3.0;
        for (key, angle) in [(KeyCode::ArrowLeft, 0.9), (KeyCode::ArrowRight, -0.9)] {
            if keys.pressed(key) {
                transform.rotate_y(angle * delta);
            }
        }
        for (key, angle) in [(KeyCode::ArrowUp, 0.9), (KeyCode::ArrowDown, -0.9)] {
            if keys.pressed(key) {
                transform.rotate_local_x(angle * delta);
            }
        }
    } else if let (Ok(session), Projection::Perspective(perspective)) =
        (player.session(), *projection)
    {
        let aspect = window.width() / window.height();
        if let Some((position, target)) =
            camera::frame(session.snapshot(), *mode, aspect, perspective.fov)
        {
            **transform = Transform::from_translation(position).looking_at(target, Vec3::Y);
        }
    }
}

/// Project identity labels into UI pixels from their own physical body positions.
pub(super) fn project_labels(
    player: Res<'_, Player>,
    camera: Single<'_, '_, (&Camera, &Transform), With<Camera3d>>,
    mut labels: Query<'_, '_, (&RobotLabel, &mut Node, &mut Visibility)>,
) {
    let (camera, transform) = *camera;
    let global = GlobalTransform::from(*transform);
    for (label, mut node, mut visible) in &mut labels {
        let Ok(session) = player.session() else {
            *visible = Visibility::Hidden;
            continue;
        };
        let position = match label.0 {
            RobotId::Drone(slot) => session.snapshot().drone(slot).position() + Vec3::Y * 0.45,
            RobotId::Droid(slot) => {
                session
                    .snapshot()
                    .droid(slot)
                    .body(DroidBody::Pelvis)
                    .position()
                    + Vec3::Y * 1.1
            }
        };
        // Bevy 0.18.1 returns logical viewport pixels, matching UI coordinates.
        // <https://docs.rs/bevy/0.18.1/bevy/camera/struct.Camera.html#method.world_to_viewport>.
        match (
            camera.world_to_viewport(&global, position),
            camera.logical_viewport_size(),
        ) {
            (Ok(point), Some(size))
                if (0.0..=size.x).contains(&point.x) && (0.0..=size.y).contains(&point.y) =>
            {
                node.left = px(point.x - 45.0);
                node.top = px(point.y);
                *visible = Visibility::Visible;
            }
            _ => *visible = Visibility::Hidden,
        }
    }
}

/// Show load and mapping failures before physical telemetry, keeping qualification explicit.
pub(super) fn refresh(
    player: Res<'_, Player>,
    mode: Res<'_, camera::Mode>,
    visuals: Query<'_, '_, &Visual>,
    mut texts: Query<'_, '_, (&mut Text, Option<&Readout>, Option<&ButtonLabel>)>,
) {
    for (mut text, readout, button) in &mut texts {
        if readout.is_some() {
            let state = match player.session() {
                Err(error) => format!("Checkpoint failed: {error}. Reload to retry."),
                Ok(session) => session.error().map_or_else(
                    || {
                        visuals.iter().find_map(Visual::error).map_or_else(
                            || {
                                if visuals.iter().count() != 6 {
                                    "Model roster failed: six roots are required. Reload to retry."
                                        .to_owned()
                                } else if !rig::ready(&visuals) {
                                    "Loading six models…".to_owned()
                                } else {
                                    telemetry(session, &player, *mode)
                                }
                            },
                            |error| format!("Model failed: {error}. Reload to retry."),
                        )
                    },
                    |error| format!("Inference stopped: {error}. Reload to retry."),
                ),
            };
            text.0 = format!("{state}\nUnqualified for competition");
        } else if let Some(button) = button {
            text.0 = label(button.0, &player, *mode);
        }
    }
}

/// Read actual applied requests for the selected robot; no action exists before the first frame.
fn telemetry(session: &Session, player: &Player, mode: camera::Mode) -> String {
    let state = if session.finished() {
        "Clip complete"
    } else if player.running() {
        "Running"
    } else {
        "Paused"
    };
    let seconds = session.snapshot().elapsed().as_secs_f32();
    let rate = player.rate();
    let mut text = format!("{state} · {seconds:.2}/20.00 s · {rate}×");
    match mode {
        camera::Mode::All => {}
        camera::Mode::Free => text.push_str("\nWASD move · Q/E rise/lower · arrows turn"),
        camera::Mode::Follow(RobotId::Drone(slot)) => {
            drone_telemetry(&mut text, session, slot, mode.label());
        }
        camera::Mode::Follow(RobotId::Droid(slot)) => {
            droid_telemetry(&mut text, session, slot, mode.label());
        }
    }
    text
}

/// Drone measurements use metres, metres/second and actual dimensionless rotor fractions.
fn drone_telemetry(text: &mut String, session: &Session, slot: RobotSlot, name: &str) {
    let body = session.snapshot().drone(slot);
    let height = body.position().y;
    let speed = body.linear_velocity().length();
    write!(text, "\n{name} · {height:.2} m · {speed:.2} m/s")
        .expect("writing to a String cannot fail");
    if let Some(action) = session.last_drone_action(slot) {
        let [fl, fr, rr, rl] = action.fractions();
        write!(text, "\nRotor fractions {fl:.2} {fr:.2} {rr:.2} {rl:.2}")
            .expect("writing to a String cannot fail");
    } else {
        text.push_str("\nAwaiting first policy frame");
    }
}

/// Droid measurements use the physical pelvis and the largest absolute requested torque fraction.
fn droid_telemetry(text: &mut String, session: &Session, slot: RobotSlot, name: &str) {
    let body = session.snapshot().droid(slot).body(DroidBody::Pelvis);
    let height = body.position().y;
    let speed = body.linear_velocity().length();
    write!(text, "\n{name} · pelvis {height:.2} m · {speed:.2} m/s")
        .expect("writing to a String cannot fail");
    if let Some(action) = session.last_droid_action(slot) {
        let fraction = DroidActuator::ALL
            .into_iter()
            .map(|actuator| action.fraction(actuator).abs())
            .fold(0.0, f32::max);
        write!(text, "\nMaximum torque fraction {fraction:.2}")
            .expect("writing to a String cannot fail");
    } else {
        text.push_str("\nAwaiting first policy frame");
    }
}

#[cfg(test)]
#[path = "view_tests.rs"]
mod tests;
