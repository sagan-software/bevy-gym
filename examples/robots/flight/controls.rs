//! Buttons and keyboard shortcuts for the same deterministic session controls.

use bevy::prelude::*;
use bevy_gym::EpisodeStatus;

use super::scene::DroneModel;
use super::session::{MotorPreset, Playback, Session, StartProfile};

/// An available user action, shared by buttons and keyboard shortcuts.
#[derive(Component, Clone, Copy)]
pub(super) enum Control {
    /// Toggle continuous stepping while the episode is active.
    Playback,
    /// Apply exactly one action while paused.
    Step,
    /// Restore the seeded initial state and clear policy memory.
    Reset,
    /// Reset the episode with the bundled recovery policy.
    Learned,
    /// Select one of the four bounded motor commands.
    Preset(MotorPreset),
    /// Replace the episode with the selected initial conditions.
    Start(StartProfile),
}

/// Text that reports state and measured motion.
#[derive(Component)]
pub(super) struct StatusText;

/// Text on the button whose label changes with playback.
#[derive(Component)]
pub(super) struct PlaybackText;

/// Keep the scene visible between a compact heading and wrapping controls.
pub(super) fn setup(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
    let font = assets.load("fonts/MonaSans-VariableFont.ttf");
    commands
        .spawn(Node {
            width: percent(100),
            height: percent(100),
            padding: UiRect::all(px(20)),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        })
        .with_children(|root| {
            header(root, &font);
            footer(root, &font);
        });
}

/// Identify the lesson and show one measured state readout.
fn header(root: &mut ChildSpawnerCommands<'_>, font: &Handle<Font>) {
    root.spawn((panel(), BackgroundColor(Color::srgb(0.09, 0.10, 0.11))))
        .with_children(|header| {
            header.spawn((
                Text::new("Drone hover"),
                TextFont {
                    font: font.clone(),
                    font_size: 25.0,
                    ..default()
                },
            ));
            header.spawn((
                StatusText,
                Text::new("Loading drone model…"),
                TextFont {
                    font: font.clone(),
                    font_size: 16.0,
                    ..default()
                },
            ));
        });
}

/// Separate playback actions from motor choices and retain the model attribution.
fn footer(root: &mut ChildSpawnerCommands<'_>, font: &Handle<Font>) {
    root.spawn((panel(), BackgroundColor(Color::srgb(0.09, 0.10, 0.11))))
        .with_children(|footer| {
            footer.spawn(row()).with_children(|row| {
                button(
                    row,
                    "Calm start [C]",
                    Control::Start(StartProfile::Calm),
                    font,
                );
                button(
                    row,
                    "Disturbed start [D]",
                    Control::Start(StartProfile::Disturbed),
                    font,
                );
            });
            footer.spawn(row()).with_children(|row| {
                button(row, "Run [Space]", Control::Playback, font);
                button(row, "Step [N]", Control::Step, font);
                button(row, "Reset [R]", Control::Reset, font);
            });
            footer.spawn(row()).with_children(|row| {
                button(row, "Learned policy [P]", Control::Learned, font);
                for (label, preset) in [
                    ("Power off [1]", MotorPreset::PowerOff),
                    ("Hover [2]", MotorPreset::Hover),
                    ("Climb [3]", MotorPreset::Climb),
                    ("Tilt [4]", MotorPreset::Tilt),
                ] {
                    button(row, label, Control::Preset(preset), font);
                }
            });
            #[cfg(not(target_arch = "wasm32"))]
            footer.spawn((
                Text::new("Drone by NateGazzard | CC BY 3.0"),
                TextFont {
                    font: font.clone(),
                    font_size: 13.0,
                    ..default()
                },
            ));
        });
}

/// Group related text and controls without fixed dimensions that clip on mobile.
fn panel() -> Node {
    Node {
        padding: UiRect::all(px(14)),
        flex_direction: FlexDirection::Column,
        row_gap: px(8),
        align_self: AlignSelf::FlexStart,
        max_width: percent(100),
        ..default()
    }
}

/// Wrap touch-sized buttons when the viewport cannot fit a full row.
fn row() -> Node {
    Node {
        flex_wrap: FlexWrap::Wrap,
        column_gap: px(8),
        row_gap: px(8),
        ..default()
    }
}

/// Spawn a labeled control; state-dependent colors are supplied by `refresh`.
fn button(
    parent: &mut ChildSpawnerCommands<'_>,
    label: &str,
    control: Control,
    font: &Handle<Font>,
) {
    parent
        .spawn((
            Button,
            control,
            Node {
                min_height: px(44),
                padding: UiRect::axes(px(12), px(10)),
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.22, 0.24, 0.25)),
        ))
        // Complete click events retain rapid presses between rendered frames.
        .observe(
            move |mut click: On<'_, '_, Pointer<Click>>, mut session: ResMut<'_, Session>| {
                click.propagate(false);
                if click.button == PointerButton::Primary {
                    apply(control, &mut session);
                }
            },
        )
        .with_children(|parent| {
            let mut text = parent.spawn((
                Text::new(label),
                // The whole button is one hit target, including gaps between glyphs.
                Pickable::IGNORE,
                TextFont {
                    font: font.clone(),
                    font_size: 15.0,
                    ..default()
                },
            ));
            if matches!(control, Control::Playback) {
                text.insert(PlaybackText);
            }
        });
}

/// Apply documented keyboard shortcuts through the same boundary as clicks.
pub(super) fn interact(keys: Res<'_, ButtonInput<KeyCode>>, mut session: ResMut<'_, Session>) {
    for (key, control) in [
        (KeyCode::Space, Control::Playback),
        (KeyCode::KeyN, Control::Step),
        (KeyCode::KeyR, Control::Reset),
        (KeyCode::KeyP, Control::Learned),
        (KeyCode::KeyC, Control::Start(StartProfile::Calm)),
        (KeyCode::KeyD, Control::Start(StartProfile::Disturbed)),
        (KeyCode::Digit1, Control::Preset(MotorPreset::PowerOff)),
        (KeyCode::Digit2, Control::Preset(MotorPreset::Hover)),
        (KeyCode::Digit3, Control::Preset(MotorPreset::Climb)),
        (KeyCode::Digit4, Control::Preset(MotorPreset::Tilt)),
    ] {
        if keys.just_pressed(key) {
            apply(control, &mut session);
        }
    }
}

/// Reject unavailable actions before they can reach the session.
fn apply(control: Control, session: &mut Session) {
    if !enabled(control, session) {
        return;
    }
    match control {
        Control::Playback => session.toggle_playback(),
        Control::Step => session.single_step(),
        Control::Reset => session.reset(),
        Control::Learned => session.select_learned(),
        Control::Preset(preset) => session.select(preset),
        Control::Start(profile) => session.select_start(profile),
    }
}

/// Completion disables stepping; reset and motor selection remain available.
fn enabled(control: Control, session: &Session) -> bool {
    match control {
        Control::Playback => session.can_step(),
        Control::Step => session.can_step() && session.playback() == Playback::Paused,
        Control::Reset | Control::Learned | Control::Preset(_) | Control::Start(_) => true,
    }
}

/// Highlight the current motor command and initial-condition choice.
fn selected(control: Control, session: &Session) -> bool {
    match control {
        Control::Preset(preset) => Some(preset) == session.preset(),
        Control::Learned => session.is_learned(),
        Control::Start(profile) => profile == session.start_profile(),
        Control::Playback | Control::Step | Control::Reset => false,
    }
}

/// Show load failures, measured state, and disabled controls without changing physics.
pub(super) fn refresh(
    session: Res<'_, Session>,
    model: Res<'_, DroneModel>,
    assets: Res<'_, AssetServer>,
    mut status: Single<'_, '_, &mut Text, (With<StatusText>, Without<PlaybackText>)>,
    mut playback: Single<'_, '_, &mut Text, (With<PlaybackText>, Without<StatusText>)>,
    mut buttons: Query<'_, '_, (&Control, &Interaction, &mut BackgroundColor)>,
) {
    let failed = assets
        .get_load_state(model.0.id())
        .is_some_and(|state| state.is_failed())
        || assets
            .get_recursive_dependency_load_state(model.0.id())
            .is_some_and(|state| state.is_failed());
    status.0 = if failed {
        "Drone model could not load. Reload the example to retry.".to_owned()
    } else if !assets.is_loaded_with_dependencies(model.0.id()) {
        "Loading drone model…".to_owned()
    } else {
        status_label(&session)
    };
    let playback_label = if session.status().is_done() || session.playback() == Playback::Paused {
        "Run [Space]"
    } else {
        "Pause [Space]"
    };
    playback_label.clone_into(&mut playback.0);
    for (control, interaction, mut background) in &mut buttons {
        background.0 = if !enabled(*control, &session) {
            Color::srgb(0.13, 0.14, 0.15)
        } else if selected(*control, &session) || *interaction == Interaction::Hovered {
            Color::srgb(0.25, 0.40, 0.42)
        } else {
            Color::srgb(0.22, 0.24, 0.25)
        };
    }
}

/// Derive visible state from the authoritative result and current playback choice.
fn status_label(session: &Session) -> String {
    if let Some(error) = session.error() {
        return format!("Policy failed: {error}\nReset or choose a controller to continue.");
    }
    let state = match session.status() {
        EpisodeStatus::Terminated => "Episode ended",
        EpisodeStatus::Truncated => "Time limit",
        EpisodeStatus::Continuing => match session.playback() {
            Playback::Paused => "Paused",
            Playback::Running => "Running",
        },
    };
    let command = match session.preset() {
        Some(MotorPreset::PowerOff) => "Power off",
        Some(MotorPreset::Hover) => "Hover",
        Some(MotorPreset::Climb) => "Climb",
        Some(MotorPreset::Tilt) => "Tilt",
        None => "Learned policy",
    };
    let steps = session.steps();
    let height = session.observation().position().y;
    let velocity = session.observation().linear_velocity().y;
    let start = match session.start_profile() {
        StartProfile::Calm => "Calm start",
        StartProfile::Disturbed => "Disturbed start",
    };
    format!("{state} | {command} | Step {steps}/500\n{start} | Height {height:.2} m | Vertical speed {velocity:.2} m/s")
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{
        camera::NormalizedRenderTarget,
        picking::{
            backend::HitData,
            pointer::{Location, PointerId},
        },
    };

    #[test]
    fn viewer_offers_a_labeled_learned_policy_button() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Font>()
            .add_systems(Startup, setup);
        app.update();
        let world = app.world_mut();
        let button = world
            .query::<(Entity, &Control)>()
            .iter(world)
            .find_map(|(entity, control)| matches!(control, Control::Learned).then_some(entity))
            .expect("the viewer exposes learned inference");
        let labels = world.get::<Children>(button).unwrap();
        assert!(labels.iter().any(|child| {
            world
                .get::<Text>(child)
                .is_some_and(|text| text.0 == "Learned policy [P]")
        }));
    }

    #[test]
    fn learned_selection_reports_state_and_faults_disable_only_playback() {
        let mut session = Session::default();
        assert!(!selected(Control::Learned, &session));
        apply(Control::Learned, &mut session);
        assert!(selected(Control::Learned, &session));
        assert!(!selected(Control::Preset(MotorPreset::Hover), &session));
        assert!(status_label(&session).starts_with("Paused | Learned policy | Step 0/500"));
        session.select_policy(Vec::new());
        assert!(status_label(&session).starts_with("Policy failed:"));
        assert!(status_label(&session).ends_with("Reset or choose a controller to continue."));
        assert!(!enabled(Control::Playback, &session));
        assert!(!enabled(Control::Step, &session));
        assert!(!selected(Control::Learned, &session));
        for control in [
            Control::Reset,
            Control::Learned,
            Control::Preset(MotorPreset::Hover),
        ] {
            assert!(enabled(control, &session));
        }
        apply(Control::Playback, &mut session);
        apply(Control::Step, &mut session);
        assert_eq!(session.steps(), 0);
        apply(Control::Reset, &mut session);
        assert!(enabled(Control::Playback, &session));
    }

    #[test]
    fn learned_keyboard_shortcut_resets_a_running_episode() {
        let mut app = App::new();
        app.init_resource::<Session>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, interact);
        let mut session = app.world_mut().resource_mut::<Session>();
        session.toggle_playback();
        session.advance();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyP);
        app.update();
        let session = app.world().resource::<Session>();
        assert!(session.is_learned());
        assert_eq!(session.steps(), 0);
        assert_eq!(session.playback(), Playback::Paused);
    }

    #[test]
    fn selecting_a_start_updates_the_label_and_restarts_after_completion() {
        let mut session = Session::default();
        assert!(status_label(&session).contains("Calm start"));
        apply(Control::Start(StartProfile::Disturbed), &mut session);
        assert!(status_label(&session).contains("Disturbed start"));
        apply(Control::Playback, &mut session);
        for _ in 0..500 {
            session.advance();
        }
        assert!(session.status().is_done());
        for profile in [StartProfile::Disturbed, StartProfile::Calm] {
            assert!(enabled(Control::Start(profile), &session));
            apply(Control::Start(profile), &mut session);
            assert_eq!(session.start_profile(), profile);
            assert_eq!(session.steps(), 0);
            assert_eq!(session.status(), EpisodeStatus::Continuing);
        }
    }

    #[test]
    fn selection_tracks_the_start_and_motor_command_only() {
        let mut session = Session::default();
        for control in [Control::Playback, Control::Step, Control::Reset] {
            assert!(!selected(control, &session));
        }
        assert!(selected(Control::Preset(MotorPreset::Hover), &session));
        assert!(!selected(Control::Preset(MotorPreset::Climb), &session));
        assert!(selected(Control::Start(StartProfile::Calm), &session));
        assert!(!selected(Control::Start(StartProfile::Disturbed), &session));
        apply(Control::Start(StartProfile::Disturbed), &mut session);
        assert!(selected(Control::Start(StartProfile::Disturbed), &session));
        assert!(!selected(Control::Start(StartProfile::Calm), &session));
    }

    #[test]
    fn keyboard_start_choices_reset_the_session() {
        let mut app = App::new();
        app.init_resource::<Session>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, interact);
        for (key, profile) in [
            (KeyCode::KeyD, StartProfile::Disturbed),
            (KeyCode::KeyC, StartProfile::Calm),
        ] {
            app.world_mut().resource_mut::<Session>().single_step();
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.clear();
            keys.press(key);
            app.update();
            let session = app.world().resource::<Session>();
            assert_eq!(session.start_profile(), profile);
            assert_eq!(session.steps(), 0);
            assert_eq!(session.playback(), Playback::Paused);
        }
    }

    #[test]
    fn consecutive_clicks_on_button_labels_are_applied_before_the_next_frame() {
        let mut world = World::new();
        world.init_resource::<Session>();
        // Pointer traversal queries this component even for an offscreen target.
        world.register_component::<Window>();
        let root = world.spawn_empty().id();
        world.commands().entity(root).with_children(|parent| {
            button(
                parent,
                "Climb",
                Control::Preset(MotorPreset::Climb),
                &default(),
            );
            button(parent, "Step", Control::Step, &default());
        });
        world.flush();
        let labels = world
            .query_filtered::<Entity, With<Text>>()
            .iter(&world)
            .collect::<Vec<_>>();
        for label in &labels {
            let pickable = world.get::<Pickable>(*label).unwrap();
            assert!(!pickable.is_hoverable && !pickable.should_block_lower);
        }
        let click = |entity, button| {
            Pointer::new(
                PointerId::Mouse,
                Location {
                    target: NormalizedRenderTarget::None {
                        width: 1,
                        height: 1,
                    },
                    position: Vec2::ZERO,
                },
                Click {
                    button,
                    hit: HitData::new(Entity::PLACEHOLDER, 0.0, None, None),
                    duration: std::time::Duration::from_millis(1),
                },
                entity,
            )
        };
        assert_eq!(world.get::<Text>(labels[0]).unwrap().0, "Climb");
        let parent = world.get::<ChildOf>(labels[0]).unwrap().parent();
        assert!(matches!(
            world.get::<Control>(parent).unwrap(),
            Control::Preset(MotorPreset::Climb)
        ));
        world.trigger(click(parent, PointerButton::Primary));
        assert_eq!(
            world.resource::<Session>().preset(),
            Some(MotorPreset::Climb)
        );
        world.resource_mut::<Session>().reset();
        // Both complete clicks arrive between application updates, on child text.
        world.trigger(click(labels[0], PointerButton::Primary));
        world.trigger(click(labels[1], PointerButton::Primary));
        assert_eq!(
            world.resource::<Session>().preset(),
            Some(MotorPreset::Climb)
        );
        assert_eq!(world.resource::<Session>().steps(), 1);
        for button in [PointerButton::Secondary, PointerButton::Middle] {
            world.trigger(click(labels[1], button));
        }
        assert_eq!(world.resource::<Session>().steps(), 1);
    }

    #[test]
    fn controls_report_and_enforce_playback_boundaries() {
        let mut session = Session::default();
        assert!(status_label(&session).starts_with("Paused | Hover | Step 0/500"));
        apply(Control::Playback, &mut session);
        assert!(status_label(&session).starts_with("Running"));
        assert!(!enabled(Control::Step, &session));
        apply(Control::Step, &mut session);
        assert_eq!(session.steps(), 0);
        apply(Control::Playback, &mut session);
        apply(Control::Step, &mut session);
        assert_eq!(session.steps(), 1);
        apply(Control::Reset, &mut session);
        assert_eq!(session.steps(), 0);
        for (preset, label) in [
            (MotorPreset::PowerOff, "Power off"),
            (MotorPreset::Hover, "Hover"),
            (MotorPreset::Climb, "Climb"),
            (MotorPreset::Tilt, "Tilt"),
        ] {
            assert!(enabled(Control::Preset(preset), &session));
            apply(Control::Preset(preset), &mut session);
            assert!(status_label(&session).contains(label));
        }
    }

    #[test]
    fn both_completion_types_disable_steps_but_allow_reset() {
        for preset in [MotorPreset::PowerOff, MotorPreset::Hover] {
            let mut session = Session::default();
            apply(Control::Preset(preset), &mut session);
            apply(Control::Playback, &mut session);
            for _ in 0..500 {
                session.advance();
            }
            assert!(session.status().is_done());
            assert!(!enabled(Control::Playback, &session));
            assert!(!enabled(Control::Step, &session));
            assert!(enabled(Control::Reset, &session));
            let label = if preset == MotorPreset::Hover {
                "Time limit"
            } else {
                "Episode ended"
            };
            assert!(status_label(&session).starts_with(label));
            apply(Control::Reset, &mut session);
            assert!(enabled(Control::Step, &session));
        }
    }

    #[test]
    fn keyboard_inputs_reach_the_session() {
        let mut app = App::new();
        app.init_resource::<Session>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, interact);
        app.update();
        for (key, preset) in [
            (KeyCode::Digit1, MotorPreset::PowerOff),
            (KeyCode::Digit2, MotorPreset::Hover),
            (KeyCode::Digit3, MotorPreset::Climb),
            (KeyCode::Digit4, MotorPreset::Tilt),
        ] {
            let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keys.clear();
            keys.press(key);
            app.update();
            assert_eq!(app.world().resource::<Session>().preset(), Some(preset));
        }
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.clear();
        keys.press(KeyCode::KeyN);
        app.update();
        assert_eq!(app.world().resource::<Session>().steps(), 1);
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.clear();
        keys.press(KeyCode::KeyR);
        app.update();
        assert_eq!(app.world().resource::<Session>().steps(), 0);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Space);
        app.update();
        assert_eq!(
            app.world().resource::<Session>().playback(),
            Playback::Running
        );
    }
}
