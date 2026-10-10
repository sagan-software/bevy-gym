//! Viewer controls and measurements are separate from every agent's policy input.

use super::*;
use bevy::{
    camera::{CameraProjection, RenderTargetInfo},
    picking::pointer::PointerId,
};
use std::time::Duration;

/// Instantiate the real overlay without opening a rendering device or choosing actor requests.
fn overlay(session: Result<Session, String>) -> App {
    let mut app = App::new();
    let identities = Identities::new(
        &session,
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk"),
    );
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Scene>()
        .init_asset::<Font>()
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .insert_resource(identities)
        .insert_resource(Player::new(session))
        .init_resource::<camera::Mode>()
        .add_systems(Startup, setup)
        .add_systems(Update, (rig::project, refresh).chain());
    app.update();
    app
}

/// Loading and checkpoint failure both retain six named licensed roots and usable spectator controls.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn startup_overlay_keeps_identity_controls_and_failures_visible() {
    for session in [Ok(load()), Err("invalid checkpoint".to_owned())] {
        let mut app = overlay(session);
        assert_eq!(
            app.world_mut().query::<&Visual>().iter(app.world()).count(),
            6
        );
        assert_eq!(
            app.world_mut()
                .query::<&RobotLabel>()
                .iter(app.world())
                .count(),
            6
        );
        assert!(app
            .world_mut()
            .query_filtered::<&Visibility, With<Visual>>()
            .iter(app.world())
            .all(|visible| *visible == Visibility::Hidden));
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<Button>>()
                .iter(app.world())
                .count(),
            6
        );
        let ready = app.world().resource::<Player>().session().is_ok();
        let readout = app
            .world_mut()
            .query_filtered::<&Text, With<Readout>>()
            .single(app.world())
            .expect("state");
        assert_eq!(
            readout.0,
            if ready {
                "Loading six models…\nUnqualified for competition"
            } else {
                "Checkpoint failed: invalid checkpoint. Reload to retry.\nUnqualified for competition"
            }
        );
        // Six reset markers are added only when their physical source positions are available.
        assert_eq!(
            app.world_mut().query::<&Mesh3d>().iter(app.world()).count(),
            if ready { 7 } else { 1 }
        );
        assert!(
            app.world_mut()
                .query::<&Node>()
                .iter(app.world())
                .filter(|node| node.min_height == px(44))
                .count()
                == 6
        );
    }
}

/// Real pointer observers ignore secondary presses and route primary presses through spectator commands.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn pointer_controls_share_the_keyboard_boundary_and_cannot_bypass_loading() {
    let mut app = overlay(Ok(load()));
    let buttons: Vec<_> = app
        .world_mut()
        .query::<(&Text, &ButtonLabel, &ChildOf)>()
        .iter(app.world())
        .map(|(text, _, parent)| (text.0.clone(), parent.parent()))
        .collect();
    for (name, button) in buttons {
        let rate = app.world().resource::<Player>().rate();
        press(&mut app, button, PointerButton::Secondary);
        assert_eq!(app.world().resource::<Player>().rate(), rate);
        assert_eq!(*app.world().resource::<camera::Mode>(), camera::Mode::All);
        press(&mut app, button, PointerButton::Primary);
        if name.starts_with("View:") {
            assert_eq!(
                *app.world().resource::<camera::Mode>(),
                camera::Mode::Follow(RobotId::Drone(RobotSlot::First))
            );
        } else if name.starts_with("Free camera") {
            assert_eq!(*app.world().resource::<camera::Mode>(), camera::Mode::Free);
        } else if name.starts_with("Speed:") {
            assert_eq!(app.world().resource::<Player>().rate(), 4);
        }
        app.insert_resource(camera::Mode::All);
        let player = app.world().resource::<Player>();
        assert!(!player.running(), "unloaded models cannot start playback");
        assert_eq!(
            player.session().expect("session").snapshot().elapsed(),
            Duration::ZERO
        );
        assert!(player
            .session()
            .expect("session")
            .last_drone_action(RobotSlot::First)
            .is_none());
    }
}

/// Trigger Bevy's actual pointer event on the spawned button, without a synthetic actuator input.
fn press(app: &mut App, entity: Entity, button: PointerButton) {
    app.world_mut().trigger(Pointer::new(
        PointerId::Mouse,
        bevy::picking::pointer::Location {
            target: bevy::camera::NormalizedRenderTarget::None {
                width: 1280,
                height: 800,
            },
            position: Vec2::ZERO,
        },
        Press {
            button,
            hit: bevy::picking::backend::HitData::new(Entity::PLACEHOLDER, 0.0, None, None),
        },
        entity,
    ));
}

/// Labels use logical pixels at high DPI, hide outside the viewport, and never advance an actor.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn labels_project_own_physical_positions_and_hide_unavailable_views() {
    let mut app = App::new();
    app.insert_resource(Player::new(Ok(load())))
        .add_systems(Update, project_labels);
    let camera_entity = app.world_mut().spawn(Camera3d::default()).id();
    for slot in RobotSlot::ALL {
        app.world_mut().spawn((
            RobotLabel(RobotId::Drone(slot)),
            Node::default(),
            Visibility::Hidden,
        ));
        app.world_mut().spawn((
            RobotLabel(RobotId::Droid(slot)),
            Node::default(),
            Visibility::Hidden,
        ));
    }
    for size in [UVec2::new(1280, 744), UVec2::new(390, 788)] {
        let mut projection = PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_3,
            ..default()
        };
        projection.update(size.x as f32, size.y as f32);
        let (position, target) = camera::frame(
            app.world()
                .resource::<Player>()
                .session()
                .expect("session")
                .snapshot(),
            camera::Mode::All,
            projection.aspect_ratio,
            projection.fov,
        )
        .expect("overview");
        let transform = Transform::from_translation(position).looking_at(target, Vec3::Y);
        let mut camera = Camera::default();
        camera.computed.clip_from_view = projection.get_clip_from_view();
        camera.computed.target_info = Some(RenderTargetInfo {
            physical_size: size * 2,
            scale_factor: 2.0,
        });
        app.world_mut()
            .entity_mut(camera_entity)
            .insert((camera, transform));
        app.update();
        let mut labels = app.world_mut().query::<(&RobotLabel, &Node, &Visibility)>();
        for (_, node, visible) in labels.iter(app.world()) {
            assert_eq!(*visible, Visibility::Visible);
            let Val::Px(left) = node.left else {
                panic!("logical x pixels")
            };
            let Val::Px(top) = node.top else {
                panic!("logical y pixels")
            };
            assert!((0.0..=size.x as f32).contains(&(left + 45.0)));
            assert!((0.0..=size.y as f32).contains(&top));
        }
    }
    app.world_mut().entity_mut(camera_entity).insert(
        Transform::from_xyz(1000.0, 2.0, 10.0).looking_at(Vec3::new(1000.0, 2.0, 0.0), Vec3::Y),
    );
    app.update();
    assert!(app
        .world_mut()
        .query_filtered::<&Visibility, With<RobotLabel>>()
        .iter(app.world())
        .all(|visible| *visible == Visibility::Hidden));
    app.world_mut().entity_mut(camera_entity).insert(
        Transform::from_xyz(0.0, 1000.0, 50.0).looking_at(Vec3::new(0.0, 1000.0, 0.0), Vec3::Y),
    );
    app.update();
    assert!(app
        .world_mut()
        .query_filtered::<&Visibility, With<RobotLabel>>()
        .iter(app.world())
        .all(|visible| *visible == Visibility::Hidden));
    app.world_mut()
        .get_mut::<Camera>(camera_entity)
        .expect("camera")
        .computed
        .target_info = None;
    app.update();
    assert!(app
        .world_mut()
        .query_filtered::<&Visibility, With<RobotLabel>>()
        .iter(app.world())
        .all(|visible| *visible == Visibility::Hidden));
    assert_eq!(
        app.world()
            .resource::<Player>()
            .session()
            .expect("session")
            .snapshot()
            .elapsed(),
        Duration::ZERO
    );
    app.insert_resource(Player::new(Err("checkpoint failed".to_owned())));
    app.update();
    assert!(app
        .world_mut()
        .query_filtered::<&Visibility, With<RobotLabel>>()
        .iter(app.world())
        .all(|visible| *visible == Visibility::Hidden));
}

/// Load the recorded checkpoint pair used by the shipping shared-world view.
fn load() -> Session {
    Session::load(
        include_bytes!("../../../docs/progress/drone-hover.mpk").to_vec(),
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk")
            .to_vec(),
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.json"),
    )
    .expect("frozen PPO policies")
}

/// Provenance and initial measurements state exact bytes and never invent an unapplied action.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn provenance_and_telemetry_preserve_checkpoint_identity_and_clip_state() {
    let session = Ok(load());
    let identities = Identities::new(
        &session,
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk"),
    );
    assert_eq!(
        identities.0,
        "Drones · PPO 280 · 5aa47c6941b1\nDroids · PPO 22940 · 9c9e4c5bc8f7"
    );
    assert_eq!(
        Identities::new(&Err("invalid checkpoint".to_owned()), &[]).0,
        "Checkpoints unavailable"
    );
    let mut player = Player::new(session);
    assert_eq!(
        telemetry(
            player.session().expect("session"),
            &player,
            camera::Mode::All
        ),
        "Paused · 0.00/20.00 s · 1×"
    );
    assert_eq!(
        telemetry(
            player.session().expect("session"),
            &player,
            camera::Mode::Follow(RobotId::Drone(RobotSlot::First))
        ),
        "Paused · 0.00/20.00 s · 1×\nDrone 1 · 2.00 m · 0.00 m/s\nAwaiting first policy frame"
    );
    assert_eq!(telemetry(player.session().expect("session"), &player, camera::Mode::Follow(RobotId::Droid(RobotSlot::First))), "Paused · 0.00/20.00 s · 1×\nDroid 1 · pelvis 1.02 m · 0.00 m/s\nAwaiting first policy frame");
    assert!(telemetry(
        player.session().expect("session"),
        &player,
        camera::Mode::Free
    )
    .contains("WASD move · Q/E rise/lower · arrows turn"));
    player.apply(Command::Play, true);
    assert_eq!(
        label(Control::Playback(Command::Play), &player, camera::Mode::All),
        "Pause [Space]"
    );
    player.advance(true);
    let session = player.session().expect("policy frame");
    assert!(telemetry(
        session,
        &player,
        camera::Mode::Follow(RobotId::Drone(RobotSlot::First))
    )
    .contains("Rotor fractions"));
    assert!(telemetry(
        session,
        &player,
        camera::Mode::Follow(RobotId::Droid(RobotSlot::First))
    )
    .contains("Maximum torque fraction"));
    player.apply(Command::Speed, true);
    player.apply(Command::Speed, true);
    while !player.session().expect("session").finished() {
        player.advance(true);
    }
    assert_eq!(
        telemetry(
            player.session().expect("complete clip"),
            &player,
            camera::Mode::All
        ),
        "Clip complete · 20.00/20.00 s · 16×"
    );
    assert!(!player.running());
    player.apply(Command::Play, true);
    assert!(!player.running(), "completion cannot resume");
    player.apply(Command::Step, true);
    assert_eq!(
        player.session().expect("session").snapshot().elapsed(),
        Duration::from_secs(20)
    );
    assert_eq!(
        label(Control::Playback(Command::Play), &player, camera::Mode::All),
        "Run [Space]"
    );
    assert_eq!(
        label(Control::Playback(Command::Step), &player, camera::Mode::All),
        "Step [N]"
    );
    assert_eq!(
        label(
            Control::Playback(Command::Reset),
            &player,
            camera::Mode::All
        ),
        "Reset [R]"
    );
    assert_eq!(
        label(
            Control::Playback(Command::Speed),
            &player,
            camera::Mode::All
        ),
        "Speed: 16× [V]"
    );
    assert_eq!(
        label(
            Control::View,
            &player,
            camera::Mode::Follow(RobotId::Droid(RobotSlot::Third))
        ),
        "View: Droid 3 [C]"
    );
    assert_eq!(
        label(Control::Free, &player, camera::Mode::All),
        "Free camera [F]"
    );
}

/// View shortcuts select all six identities, and loading roots cannot enable policy playback.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn keyboard_controls_are_camera_or_timing_commands_without_actuator_values() {
    let mut app = App::new();
    app.insert_resource(Player::new(Ok(load())))
        .init_resource::<camera::Mode>()
        .init_resource::<ButtonInput<KeyCode>>()
        .add_systems(Update, keyboard);
    for (key, expected) in [
        (
            KeyCode::Digit1,
            camera::Mode::Follow(RobotId::Drone(RobotSlot::First)),
        ),
        (
            KeyCode::Digit2,
            camera::Mode::Follow(RobotId::Drone(RobotSlot::Second)),
        ),
        (
            KeyCode::Digit3,
            camera::Mode::Follow(RobotId::Drone(RobotSlot::Third)),
        ),
        (
            KeyCode::Digit4,
            camera::Mode::Follow(RobotId::Droid(RobotSlot::First)),
        ),
        (
            KeyCode::Digit5,
            camera::Mode::Follow(RobotId::Droid(RobotSlot::Second)),
        ),
        (
            KeyCode::Digit6,
            camera::Mode::Follow(RobotId::Droid(RobotSlot::Third)),
        ),
        (KeyCode::Digit0, camera::Mode::All),
        (
            KeyCode::KeyC,
            camera::Mode::Follow(RobotId::Drone(RobotSlot::First)),
        ),
        (KeyCode::KeyF, camera::Mode::Free),
        (KeyCode::KeyF, camera::Mode::All),
    ] {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
        app.update();
        assert_eq!(*app.world().resource::<camera::Mode>(), expected);
        // InputPlugin clears the edge after each rendered frame; holding a key cannot repeat it.
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .clear();
        app.update();
        assert_eq!(*app.world().resource::<camera::Mode>(), expected);
    }
    for key in [KeyCode::Space, KeyCode::KeyN, KeyCode::KeyR, KeyCode::KeyV] {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
        app.update();
    }
    let player = app.world().resource::<Player>();
    assert!(!player.running());
    assert_eq!(
        player.session().expect("session").snapshot().elapsed(),
        Duration::ZERO
    );
    assert_eq!(player.rate(), 4);
    assert!(player
        .session()
        .expect("session")
        .last_drone_action(RobotSlot::First)
        .is_none());
}

/// Every free-camera key moves presentation while observations and policy time remain unchanged.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn framing_and_free_camera_never_mutate_physics_and_failed_loading_retains_the_camera() {
    let mut app = App::new();
    app.insert_resource(Player::new(Ok(load())))
        .init_resource::<camera::Mode>()
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<Time>()
        .add_systems(Update, frame_camera);
    app.world_mut().spawn((Window::default(), PrimaryWindow));
    let camera = app
        .world_mut()
        .spawn((
            Camera3d::default(),
            Projection::Perspective(PerspectiveProjection {
                fov: std::f32::consts::FRAC_PI_3,
                ..default()
            }),
            Transform::IDENTITY,
        ))
        .id();
    app.update();
    let overview = *app.world().get::<Transform>(camera).expect("camera");
    *app.world_mut().resource_mut::<camera::Mode>() =
        camera::Mode::Follow(RobotId::Drone(RobotSlot::First));
    app.update();
    assert_ne!(app.world().get::<Transform>(camera), Some(&overview));
    *app.world_mut().resource_mut::<camera::Mode>() = camera::Mode::Free;
    app.world_mut()
        .resource_mut::<Time>()
        .advance_by(Duration::from_millis(500));
    for key in [
        KeyCode::KeyW,
        KeyCode::KeyS,
        KeyCode::KeyA,
        KeyCode::KeyD,
        KeyCode::KeyQ,
        KeyCode::KeyE,
        KeyCode::ArrowLeft,
        KeyCode::ArrowRight,
        KeyCode::ArrowUp,
        KeyCode::ArrowDown,
    ] {
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(key);
        let before = *app.world().get::<Transform>(camera).expect("camera");
        app.update();
        assert_ne!(app.world().get::<Transform>(camera), Some(&before));
    }
    let actual = app.world().resource::<Player>().session().expect("session");
    let unchanged = load();
    assert_eq!(actual.snapshot().elapsed(), Duration::ZERO);
    for slot in RobotSlot::ALL {
        assert_eq!(
            actual.snapshot().drone(slot),
            unchanged.snapshot().drone(slot)
        );
        assert_eq!(
            actual.snapshot().droid(slot),
            unchanged.snapshot().droid(slot)
        );
    }
    let last_camera = *app.world().get::<Transform>(camera).expect("camera");
    *app.world_mut().resource_mut::<camera::Mode>() = camera::Mode::All;
    app.world_mut()
        .query_filtered::<&mut Window, With<PrimaryWindow>>()
        .single_mut(app.world_mut())
        .expect("window")
        .resolution
        .set(0.0, 0.0);
    app.update();
    assert_eq!(app.world().get::<Transform>(camera), Some(&last_camera));
    app.insert_resource(Player::new(Err("invalid checkpoint".to_owned())));
    app.update();
    assert_eq!(app.world().get::<Transform>(camera), Some(&last_camera));
}

/// Loading and missing roster states remain visible, with the competition qualification unchanged.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn refresh_preserves_visible_load_failure_and_missing_roster_diagnostics() {
    let mut app = App::new();
    app.insert_resource(Player::new(Err("invalid checkpoint".to_owned())))
        .init_resource::<camera::Mode>()
        .add_systems(Update, refresh);
    let readout = app.world_mut().spawn((Readout, Text::default())).id();
    let button = app
        .world_mut()
        .spawn((
            ButtonLabel(Control::Playback(Command::Play)),
            Text::default(),
        ))
        .id();
    app.update();
    assert_eq!(
        app.world().get::<Text>(readout).expect("failure").0,
        "Checkpoint failed: invalid checkpoint. Reload to retry.\nUnqualified for competition"
    );
    assert_eq!(
        app.world().get::<Text>(button).expect("play label").0,
        "Run [Space]"
    );
    app.insert_resource(Player::new(Ok(load())));
    app.update();
    assert!(app
        .world()
        .get::<Text>(readout)
        .expect("roster failure")
        .0
        .starts_with("Model roster failed: six roots are required."));
    for slot in RobotSlot::ALL {
        app.world_mut().spawn(Visual::drone(slot));
        app.world_mut().spawn(Visual::droid(slot));
    }
    app.update();
    assert_eq!(
        app.world().get::<Text>(readout).expect("loading").0,
        "Loading six models…\nUnqualified for competition"
    );
    let failed = app
        .world_mut()
        .query_filtered::<Entity, With<Visual>>()
        .iter(app.world())
        .next()
        .expect("model root");
    app.world_mut().entity_mut(failed).insert(Visual::Drone {
        slot: RobotSlot::First,
        state: rig::State::Failed("missing licensed rotor".to_owned()),
    });
    app.update();
    assert_eq!(
        app.world().get::<Text>(readout).expect("model failure").0,
        "Model failed: missing licensed rotor. Reload to retry.\nUnqualified for competition"
    );
}
