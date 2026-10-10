//! Synthetic display models test root isolation without commanding an example agent.

use super::*;
use bevy::ecs::system::RunSystemOnce;

/// Exact licensed body-name vocabulary; fixtures have no locomotion animation.
const BONE_NAMES: [&str; 13] = [
    "pelvis",
    "spine_01",
    "neck_01",
    "upperarm_l",
    "lowerarm_l",
    "upperarm_r",
    "lowerarm_r",
    "thigh_l",
    "calf_l",
    "foot_l",
    "thigh_r",
    "calf_r",
    "foot_r",
];
/// Exact rotor vocabulary preserved from the licensed drone mesh.
const ROTOR_NAMES: [&str; 4] = ["Rotor_FL", "Rotor_FR", "Rotor_BR", "Rotor_BL"];

/// Retained PPO policies provide every applied request, including these display tests.
fn load() -> crate::standing::world_session::Session {
    crate::standing::world_session::Session::load(
        include_bytes!("../../../docs/progress/drone-hover.mpk").to_vec(),
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.mpk")
            .to_vec(),
        include_bytes!("../../../docs/progress/standing-seed17-update22940/checkpoint.json"),
    )
    .expect("retained PPO policies")
}

/// Create one isolated bind skeleton; state is awaiting the same capture used after scene readiness.
fn fixture(app: &mut App, mut visual: Visual) -> Entity {
    visual.awaiting_bind();
    let drone = matches!(visual, Visual::Drone { .. });
    let root = app
        .world_mut()
        .spawn((
            visual,
            Transform::IDENTITY,
            GlobalTransform::IDENTITY,
            Visibility::Hidden,
        ))
        .id();
    if drone {
        for name in ROTOR_NAMES {
            app.world_mut().spawn((
                Name::new(name),
                Transform::IDENTITY,
                GlobalTransform::IDENTITY,
                ChildOf(root),
            ));
        }
    } else {
        for name in BONE_NAMES {
            let body = projection::body_named(name).expect("licensed mapping");
            let transform = Transform::from_translation(body.bind_position());
            app.world_mut().spawn((
                Name::new(name),
                transform,
                GlobalTransform::from(transform),
                ChildOf(root),
            ));
        }
    }
    root
}

/// Six equal name sets stay isolated; body centres and cosmetic rotors follow their own physical slot.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn six_complete_models_capture_once_and_project_only_their_own_robot() {
    let mut app = App::new();
    app.insert_resource(Player::new(Ok(load())))
        .init_resource::<super::super::camera::Mode>()
        .add_systems(
            Update,
            (
                capture,
                super::super::advance,
                project,
                super::super::view::refresh,
            )
                .chain(),
        );
    let readout = app
        .world_mut()
        .spawn((super::super::view::Readout, Text::default()))
        .id();
    let roots = RobotSlot::ALL.map(|slot| {
        [
            fixture(&mut app, Visual::drone(slot)),
            fixture(&mut app, Visual::droid(slot)),
        ]
    });
    app.update();
    assert_eq!(
        app.world().get::<Text>(readout).expect("ready state").0,
        "Paused · 0.00/20.00 s · 1×\nUnqualified for competition"
    );
    assert!(app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("roster"));
    for [drone, droid] in roots {
        assert_eq!(
            app.world().get::<Visibility>(drone),
            Some(&Visibility::Visible)
        );
        assert_eq!(
            app.world().get::<Visibility>(droid),
            Some(&Visibility::Visible)
        );
        let visual = app.world().get::<Visual>(droid).expect("model");
        let Visual::Droid {
            state: State::Ready(bones),
            ..
        } = visual
        else {
            panic!("complete own skeleton")
        };
        for bone in bones.iter() {
            assert_eq!(
                app.world()
                    .get::<ChildOf>(bone.entity)
                    .expect("own parent")
                    .parent(),
                droid
            );
        }
    }
    app.world_mut()
        .resource_mut::<Player>()
        .apply(super::super::player::Command::Step, true);
    app.update();
    let session = app
        .world()
        .resource::<Player>()
        .session()
        .expect("physical frame");
    for (slot, [drone, droid]) in RobotSlot::ALL.into_iter().zip(roots) {
        let transform = app.world().get::<Transform>(drone).expect("own drone root");
        // Source body-centre alignment must equal this slot's observed centre in metres.
        assert!(
            (transform.transform_point(drone_model::BODY_CENTRE)
                - session.snapshot().drone(slot).position())
            .length()
                < 0.000_01
        );
        let Visual::Drone {
            state: State::Ready(rotors),
            ..
        } = app.world().get::<Visual>(drone).expect("drone")
        else {
            panic!("rotor mapping")
        };
        for rotor in rotors.iter() {
            let spinning = app
                .world()
                .get::<Transform>(rotor.entity)
                .expect("rotor transform");
            assert!(
                spinning.transform_point(rotor.pivot).distance(rotor.pivot) < 0.000_01,
                "rotor stays at its pivot"
            );
            assert!(
                spinning.rotation.dot(Quat::IDENTITY).abs() < 0.999_999,
                "actual PPO request spins the display mesh"
            );
        }
        let Visual::Droid {
            state: State::Ready(bones),
            ..
        } = app.world().get::<Visual>(droid).expect("droid")
        else {
            panic!("skeleton mapping")
        };
        for bone in bones.iter() {
            let local = app
                .world()
                .get::<Transform>(bone.entity)
                .expect("bone transform");
            assert!(
                local
                    .translation
                    .distance(session.snapshot().droid(slot).body(bone.body).position())
                    < 0.000_01
            );
        }
    }
    // Captured bind frames remain immutable even when later propagated display frames change.
    let [_, droid] = roots.into_iter().next().expect("first pair");
    let bone = match app.world().get::<Visual>(droid).expect("model") {
        Visual::Droid {
            state: State::Ready(bones),
            ..
        } => bones.iter().next().expect("bone").entity,
        _ => panic!("ready"),
    };
    let original = *app.world().get::<GlobalTransform>(bone).expect("bind");
    app.world_mut()
        .entity_mut(bone)
        .insert(GlobalTransform::from_xyz(90.0, 90.0, 90.0));
    app.world_mut()
        .get_mut::<Visual>(droid)
        .expect("model")
        .awaiting_bind();
    app.update();
    let retained = match app.world().get::<Visual>(droid).expect("model") {
        Visual::Droid {
            state: State::Ready(bones),
            ..
        } => bones.iter().next().expect("bone").bind,
        _ => panic!("ready"),
    };
    assert_eq!(retained, original);
    // Losing a mapped transform stops readiness and hides the failed model.
    app.world_mut().entity_mut(bone).remove::<Transform>();
    app.update();
    assert!(app
        .world()
        .get::<Visual>(droid)
        .expect("model")
        .error()
        .is_some());
    assert_eq!(
        app.world().get::<Visibility>(droid),
        Some(&Visibility::Hidden)
    );
    assert!(!app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("failed roster"));
}

/// Missing and repeated names cannot borrow a different root's complete model or enable playback.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn missing_duplicate_and_foreign_nodes_fail_their_own_models() {
    for drone in [false, true] {
        for duplicate in [false, true] {
            let mut app = App::new();
            app.add_systems(Update, capture);
            let good = if drone {
                Visual::drone(RobotSlot::First)
            } else {
                Visual::droid(RobotSlot::First)
            };
            let good = fixture(&mut app, good);
            let mut bad = if drone {
                Visual::drone(RobotSlot::Second)
            } else {
                Visual::droid(RobotSlot::Second)
            };
            bad.awaiting_bind();
            let bad = app
                .world_mut()
                .spawn((bad, Transform::IDENTITY, GlobalTransform::IDENTITY))
                .id();
            if duplicate {
                let name = if drone { "Rotor_FL" } else { "pelvis" };
                for _ in 0..2 {
                    app.world_mut().spawn((
                        Name::new(name),
                        Transform::IDENTITY,
                        GlobalTransform::IDENTITY,
                        ChildOf(bad),
                    ));
                }
            }
            app.update();
            assert!(app
                .world()
                .get::<Visual>(good)
                .expect("other complete root")
                .is_ready());
            let error = app
                .world()
                .get::<Visual>(bad)
                .expect("failed root")
                .error()
                .expect("mapping failure");
            assert!(if duplicate {
                error.contains("duplicate")
            } else {
                error.contains("found 0")
            });
            let error = error.to_owned();
            let mut visual = app.world_mut().get_mut::<Visual>(bad).expect("failed root");
            visual.awaiting_bind();
            visual.fail("later error".to_owned());
            assert_eq!(visual.error(), Some(error.as_str()));
        }
    }
}

/// Extra or repeated model identities and loading roots cannot grant six-model readiness.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn readiness_requires_exactly_six_distinct_complete_models() {
    let mut app = App::new();
    app.add_systems(Update, capture);
    assert!(!app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("empty"));
    let pairs = RobotSlot::ALL.map(|slot| {
        [
            fixture(&mut app, Visual::drone(slot)),
            fixture(&mut app, Visual::droid(slot)),
        ]
    });
    assert!(!app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("uncaptured"));
    app.update();
    assert!(app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("complete"));
    let extra = fixture(&mut app, Visual::drone(RobotSlot::First));
    app.update();
    assert!(!app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("extra"));
    app.world_mut().entity_mut(extra).despawn();
    let [drone, _] = pairs.into_iter().last().expect("third pair");
    let mut visual = app
        .world_mut()
        .get_mut::<Visual>(drone)
        .expect("third drone");
    let Visual::Drone { slot, .. } = &mut *visual else {
        panic!("drone")
    };
    *slot = RobotSlot::First;
    assert!(!app
        .world_mut()
        .run_system_once(|visuals: Query<'_, '_, &Visual>| ready(&visuals))
        .expect("duplicate identity"));
}

/// A ready event changes only its owning root, and repeated events cannot replace captured data.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn scene_notifications_are_root_scoped_and_cannot_revive_failure() {
    let mut app = App::new();
    let first = app
        .world_mut()
        .spawn(Visual::drone(RobotSlot::First))
        .observe(scene_ready)
        .id();
    let second = app
        .world_mut()
        .spawn(Visual::droid(RobotSlot::Second))
        .observe(scene_ready)
        .id();
    let instance_id = SceneSpawner::default().spawn(Handle::<Scene>::default());
    app.world_mut().trigger(SceneInstanceReady {
        entity: first,
        instance_id,
    });
    assert!(matches!(
        app.world().get::<Visual>(first),
        Some(Visual::Drone {
            state: State::AwaitingBind,
            ..
        })
    ));
    assert!(matches!(
        app.world().get::<Visual>(second),
        Some(Visual::Droid {
            state: State::Loading,
            ..
        })
    ));
    app.world_mut()
        .get_mut::<Visual>(first)
        .expect("root")
        .fail("missing rotor".to_owned());
    app.world_mut().trigger(SceneInstanceReady {
        entity: first,
        instance_id,
    });
    assert_eq!(
        app.world().get::<Visual>(first).expect("root").error(),
        Some("missing rotor")
    );
    app.world_mut().trigger(SceneInstanceReady {
        entity: second,
        instance_id,
    });
    assert!(matches!(
        app.world().get::<Visual>(second),
        Some(Visual::Droid {
            state: State::AwaitingBind,
            ..
        })
    ));
    let unrelated = app.world_mut().spawn_empty().observe(scene_ready).id();
    app.world_mut().trigger(SceneInstanceReady {
        entity: unrelated,
        instance_id,
    });
    assert!(app.world().get::<Visual>(unrelated).is_none());
}

/// Missing ancestor transforms reject both teams' bind capture before display readiness.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ancestor_without_global_transform_rejects_model_capture() {
    for visual in [
        Visual::drone(RobotSlot::First),
        Visual::droid(RobotSlot::First),
    ] {
        let mut app = App::new();
        app.add_systems(Update, capture);
        let root = fixture(&mut app, visual);
        let children = app
            .world()
            .get::<Children>(root)
            .expect("model nodes")
            .to_vec();
        let ancestor = app
            .world_mut()
            .spawn((Transform::IDENTITY, ChildOf(root)))
            .id();
        app.world_mut()
            .entity_mut(ancestor)
            .remove::<GlobalTransform>();
        for entity in children {
            app.world_mut().entity_mut(entity).insert(ChildOf(ancestor));
        }
        app.update();
        let visual = app.world().get::<Visual>(root).expect("model root");
        assert!(!visual.is_ready());
        assert!(visual.error().is_some());
    }
}

/// A missing droid root transform fails parent-bind capture without projecting the model.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn droid_root_without_global_transform_rejects_model_capture() {
    let mut app = App::new();
    app.add_systems(Update, capture);
    let root = fixture(&mut app, Visual::droid(RobotSlot::First));
    app.world_mut().entity_mut(root).remove::<GlobalTransform>();
    app.update();
    let visual = app.world().get::<Visual>(root).expect("model root");
    assert!(!visual.is_ready());
    assert!(visual.error().is_some());
}

/// Untracked asset handles leave every model loading and cannot grant readiness.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn absent_asset_load_state_keeps_all_models_loading() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Scene>()
        .insert_resource(Models {
            drone: Handle::default(),
            droid: Handle::default(),
        })
        .add_systems(Update, asset_failures);
    for slot in RobotSlot::ALL {
        app.world_mut().spawn(Visual::drone(slot));
        app.world_mut().spawn(Visual::droid(slot));
    }
    app.update();
    let mut visuals = app.world_mut().query::<&Visual>();
    assert_eq!(visuals.iter(app.world()).count(), 6);
    for visual in visuals.iter(app.world()) {
        assert!(!visual.is_ready());
        assert!(visual.error().is_none());
    }
}

/// An unmapped spine parent retains its bind offset while its mapped ancestor uses the current frame.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn unmapped_parent_offsets_follow_the_current_owning_physical_ancestor() {
    let mut app = App::new();
    app.insert_resource(Player::new(Ok(load())))
        .add_systems(Update, (capture, project).chain());
    let root = fixture(&mut app, Visual::droid(RobotSlot::Second));
    let nodes: Vec<_> = app
        .world_mut()
        .query::<(Entity, &Name)>()
        .iter(app.world())
        .map(|(entity, name)| (entity, name.as_str().to_owned()))
        .collect();
    let pelvis = nodes
        .iter()
        .find(|(_, name)| name == "pelvis")
        .expect("pelvis")
        .0;
    let torso = nodes
        .iter()
        .find(|(_, name)| name == "spine_01")
        .expect("torso")
        .0;
    let bind = GlobalTransform::from_xyz(0.02, 1.15, -0.03);
    let pelvis_bind = *app
        .world()
        .get::<GlobalTransform>(pelvis)
        .expect("bind pelvis");
    let intermediate = app
        .world_mut()
        .spawn((
            Name::new("spine_offset"),
            bind.reparented_to(&pelvis_bind),
            bind,
            ChildOf(pelvis),
        ))
        .id();
    let torso_bind = *app
        .world()
        .get::<GlobalTransform>(torso)
        .expect("bind torso");
    app.world_mut()
        .entity_mut(torso)
        .insert((ChildOf(intermediate), torso_bind.reparented_to(&bind)));
    // A detached matching name is outside this model and cannot become its mapped ancestor.
    app.world_mut().spawn((
        Name::new("pelvis"),
        Transform::IDENTITY,
        GlobalTransform::IDENTITY,
    ));
    app.update();
    let Visual::Droid {
        state: State::Ready(bones),
        ..
    } = app.world().get::<Visual>(root).expect("root")
    else {
        panic!("own complete skeleton")
    };
    assert_eq!(
        bones
            .iter()
            .find(|bone| bone.body == DroidBody::Torso)
            .expect("torso")
            .ancestor,
        Some(DroidBody::Pelvis)
    );
    for _ in 0..40 {
        app.world_mut()
            .resource_mut::<Player>()
            .apply(super::super::player::Command::Step, true);
        app.update();
        let pelvis_global = GlobalTransform::from(
            *app.world()
                .get::<Transform>(pelvis)
                .expect("current pelvis"),
        );
        let parent_global = pelvis_global
            * *app
                .world()
                .get::<Transform>(intermediate)
                .expect("unmapped offset");
        let torso_global =
            parent_global * *app.world().get::<Transform>(torso).expect("current torso");
        let body = app
            .world()
            .resource::<Player>()
            .session()
            .expect("session")
            .snapshot()
            .droid(RobotSlot::Second)
            .body(DroidBody::Torso);
        assert!(torso_global.translation().distance(body.position()) < 0.000_01);
        assert!(
            torso_global
                .rotation()
                .angle_between(body.orientation())
                .abs()
                < 0.001
        );
    }
}

/// Losing either the drone root or one rotor transform hides the model and stops readiness.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lost_drone_transforms_fail_without_replacing_the_model() {
    for lose_root in [false, true] {
        let mut app = App::new();
        app.insert_resource(Player::new(Ok(load())))
            .add_systems(Update, (capture, project).chain());
        let root = fixture(&mut app, Visual::drone(RobotSlot::Third));
        app.update();
        let Visual::Drone {
            state: State::Ready(rotors),
            ..
        } = app.world().get::<Visual>(root).expect("root")
        else {
            panic!("four rotors")
        };
        let lost = if lose_root { root } else { rotors[0].entity };
        app.world_mut().entity_mut(lost).remove::<Transform>();
        app.update();
        assert!(app
            .world()
            .get::<Visual>(root)
            .expect("root")
            .error()
            .is_some());
        assert_eq!(
            app.world().get::<Visibility>(root),
            Some(&Visibility::Hidden)
        );
        assert!(!app.world().get::<Visual>(root).expect("root").is_ready());
    }
}

/// Asset-source failures reach all six model readouts and cannot make a model ready.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn asset_loading_failures_stop_every_affected_model() {
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Scene>()
        .add_systems(Update, asset_failures);
    let assets = app.world().resource::<AssetServer>();
    let drone = assets.load("unregistered-world-test://drone.glb#Scene0");
    let droid = assets.load("unregistered-world-test://droid.glb#Scene0");
    app.insert_resource(Models { drone, droid });
    for slot in RobotSlot::ALL {
        app.world_mut().spawn(Visual::drone(slot));
        app.world_mut().spawn(Visual::droid(slot));
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        app.update();
        if app
            .world_mut()
            .query::<&Visual>()
            .iter(app.world())
            .all(|visual| visual.error().is_some())
        {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "both requested source failures must finish"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    for visual in app.world_mut().query::<&Visual>().iter(app.world()) {
        assert!(visual
            .error()
            .expect("source failure")
            .contains("unregistered-world-test"));
        assert!(!visual.is_ready());
    }
}
