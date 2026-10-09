//! Model alignment, lights, and read-only projection of the physics observation.

use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;
use bevy_gym::robots::{DroneMotor, DroneMotorState};

pub(super) use super::drone_model::rotor;
#[cfg(test)]
use super::drone_model::BODY_CENTRE;
use super::drone_model::{model_alignment, spin_about};
use super::session::Session;

/// The transform whose position and orientation come directly from physics.
#[derive(Component)]
pub(super) struct DroneBody;

/// Read-only ring identifying the failed front-left actuator.
#[derive(Component)]
pub(super) struct FailedMotorMarker;

/// Read-only destination marker and ground ruler for waypoint flight.
#[derive(Component)]
pub(super) struct FlightTarget;

/// Retained model handle, including its observable loading or failure state.
#[derive(Resource)]
pub(super) struct DroneModel(pub(super) Handle<Scene>);

/// Fixed camera offset in metres, facing the front of the drone.
const CAMERA_OFFSET: Vec3 = Vec3::new(2.2, 1.4, -3.2);

/// Spawn the asset and a fixed inspection camera without changing physics state.
pub(super) fn setup(
    mut commands: Commands<'_, '_>,
    assets: Res<'_, AssetServer>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<StandardMaterial>>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.72, 0.77, 0.79)));
    let model = assets.load(GltfAssetLabel::Scene(0).from_asset("robots/drone.glb"));
    commands.insert_resource(DroneModel(model.clone()));
    commands
        .spawn((DroneBody, Transform::default(), Visibility::default()))
        .with_children(|parent| {
            parent.spawn((SceneRoot(model), model_alignment()));
            failure_marker(parent, &mut meshes, &mut materials);
        });
    floor(&mut commands, &mut meshes, &mut materials);
    tracking_marker(&mut commands, &mut meshes, &mut materials);
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
        Transform::from_xyz(3.3, 3.1, 4.7).looking_at(Vec3::new(0.0, 1.6, 0.0), Vec3::Y),
    ));
}

/// Mark the destination and metre spacing without adding collision geometry.
fn tracking_marker(
    commands: &mut Commands<'_, '_>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let paint = materials.add(StandardMaterial {
        base_color: Color::srgb(0.05, 0.34, 0.37),
        unlit: true,
        ..default()
    });
    commands
        .spawn((FlightTarget, Transform::default(), Visibility::Hidden))
        .with_children(|target| {
            // The ring marks the two-metre flight goal; the stem locates it above ground.
            target.spawn((
                Mesh3d(meshes.add(Torus::new(0.35, 0.38))),
                MeshMaterial3d(paint.clone()),
                Transform::default(),
            ));
            target.spawn((
                Mesh3d(meshes.add(Cuboid::new(0.025, 2.0, 0.025))),
                MeshMaterial3d(paint.clone()),
                Transform::from_xyz(0.0, -1.0, 0.0),
            ));
            // Nine ground ticks measure the fixed eight-metre route from its origin.
            for metre in 0..=8 {
                target.spawn((
                    Mesh3d(meshes.add(Cuboid::new(0.025, 0.005, 1.2))),
                    MeshMaterial3d(paint.clone()),
                    Transform::from_xyz(metre as f32 - 8.0, -1.994, 0.0),
                ));
            }
        });
}

/// Derive destination position and visibility from the selected controller.
pub(super) fn project_target(
    session: Res<'_, Session>,
    mut target: Single<'_, '_, (&mut Transform, &mut Visibility), With<FlightTarget>>,
) {
    let (transform, visibility) = &mut *target;
    if let Some(position) = session.target() {
        transform.translation = position;
        **visibility = Visibility::Inherited;
    } else {
        **visibility = Visibility::Hidden;
    }
}

/// Attach a hidden visual marker above the front-left rotor in body coordinates.
fn failure_marker(
    parent: &mut ChildSpawnerCommands<'_>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let (pivot, _, _) = rotor("Rotor_FL").expect("known front-left rotor");
    parent.spawn((
        FailedMotorMarker,
        Mesh3d(meshes.add(Torus::new(0.10, 0.12))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.95, 0.12, 0.03),
            unlit: true,
            ..default()
        })),
        Transform::from_translation(model_alignment().transform_point(pivot) + Vec3::Y * 0.025),
        Visibility::Hidden,
    ));
}

/// A ground plane and landing marker make altitude and ground contact visible.
fn floor(
    commands: &mut Commands<'_, '_>,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(60.0, 60.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.55, 0.60, 0.56))),
    ));
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(1.2))),
        MeshMaterial3d(materials.add(Color::srgb(0.84, 0.84, 0.77))),
        Transform::from_xyz(0.0, 0.002, 0.0).with_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
    ));
    let paint = materials.add(Color::srgb(0.24, 0.31, 0.29));
    for (position, size) in [
        (Vec3::new(-0.28, 0.005, 0.0), Vec3::new(0.10, 0.004, 0.70)),
        (Vec3::new(0.28, 0.005, 0.0), Vec3::new(0.10, 0.004, 0.70)),
        (Vec3::new(0.0, 0.005, 0.0), Vec3::new(0.64, 0.004, 0.10)),
    ] {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(paint.clone()),
            Transform::from_translation(position),
        ));
    }
}

/// Update only display transforms; these values never flow back into the solver.
pub(super) fn project(
    session: Res<'_, Session>,
    mut drone: Single<'_, '_, &mut Transform, With<DroneBody>>,
    mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<DroneBody>)>,
    mut rotors: Query<'_, '_, (&Name, &mut Transform), (Without<DroneBody>, Without<Camera3d>)>,
) {
    let observation = session.observation();
    drone.translation = observation.position();
    drone.rotation = observation.orientation();
    // Follow translation without inheriting body tilt, so climb and falls stay visible.
    let position = observation.position();
    **camera = camera_pose(position, session.target());
    let phase = session.steps() as f32 * 0.02 * 200.0;
    let fractions = session.last_action().fractions();
    for (name, mut transform) in &mut rotors {
        if let Some((pivot, motor, sign)) = rotor(name.as_str()) {
            // Preserve the displayed angle when an actuator fails, including while paused.
            if observation.motor_state(motor) == DroneMotorState::Failed {
                continue;
            }
            let [front_left, front_right, rear_right, rear_left] = fractions;
            let fraction = match motor {
                DroneMotor::FrontLeft => front_left,
                DroneMotor::FrontRight => front_right,
                DroneMotor::RearRight => rear_right,
                DroneMotor::RearLeft => rear_left,
            };
            // This illustrates a spinning rotor; the physics command is force, not RPM.
            *transform = spin_about(pivot, -phase * fraction * sign);
        }
    }
}

/// Frame the route during waypoint flight, then converge on the usual inspection view.
fn camera_pose(position: Vec3, target: Option<Vec3>) -> Transform {
    // Divide route length by two metres to obtain a dimensionless zoom multiplier.
    let (focus, scale, look_down) = target.map_or((position, 1.0, 0.0), |goal| {
        let scale = 1.0 + position.distance(goal) / 2.0;
        // Lower the look point by 0.55 metres per scale unit to clear the lower controls.
        (position.lerp(goal, 0.5), scale, 0.55 * scale)
    });
    Transform::from_translation(focus + CAMERA_OFFSET * scale)
        .looking_at(focus - Vec3::Y * look_down, Vec3::Y)
}

/// Derive marker visibility from actuator health without changing physics.
pub(super) fn project_damage(
    session: Res<'_, Session>,
    mut marker: Single<'_, '_, &mut Visibility, With<FailedMotorMarker>>,
) {
    **marker = match session.observation().motor_state(DroneMotor::FrontLeft) {
        DroneMotorState::Working => Visibility::Hidden,
        DroneMotorState::Failed => Visibility::Inherited,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_attaches_the_hidden_marker_to_the_front_left_rotor() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Scene>()
            .add_systems(Startup, setup);
        app.update();
        let world = app.world_mut();
        let (parent, transform, visibility) = world
            .query_filtered::<(&ChildOf, &Transform, &Visibility), With<FailedMotorMarker>>()
            .single(world)
            .expect("one marker");
        assert!(world.get::<DroneBody>(parent.parent()).is_some());
        assert_eq!(*visibility, Visibility::Hidden);
        let (pivot, _, _) = rotor("Rotor_FL").expect("front-left rotor");
        assert_eq!(
            transform.translation,
            model_alignment().transform_point(pivot) + Vec3::Y * 0.025
        );
    }

    #[test]
    fn failed_rotor_stops_and_its_marker_follows_motor_health() {
        use bevy::ecs::system::RunSystemOnce;
        use bevy_gym::robots::DroneMotor;

        let mut world = World::new();
        let mut session = Session::default();
        session.single_step();
        world.insert_resource(session);
        world.spawn((DroneBody, Transform::default()));
        world.spawn((Camera3d::default(), Transform::default()));
        let front = world
            .spawn((Name::new("Rotor_FL"), Transform::default()))
            .id();
        let other = world
            .spawn((Name::new("Rotor_FR"), Transform::default()))
            .id();
        let marker = world.spawn((FailedMotorMarker, Visibility::Hidden)).id();
        world.run_system_once(project).expect("first frame");
        let stopped = *world.get::<Transform>(front).expect("rotor");
        let spinning = *world.get::<Transform>(other).expect("rotor");
        world
            .resource_mut::<Session>()
            .fail_motor(DroneMotor::FrontLeft);
        world.resource_mut::<Session>().single_step();
        world.run_system_once(project).expect("failed frame");
        world
            .run_system_once(project_damage)
            .expect("damage marker");
        assert_eq!(*world.get::<Transform>(front).expect("rotor"), stopped);
        assert_ne!(*world.get::<Transform>(other).expect("rotor"), spinning);
        assert_eq!(
            world.get::<Visibility>(marker),
            Some(&Visibility::Inherited)
        );
        world.resource_mut::<Session>().reset();
        world.run_system_once(project).expect("reset frame");
        world.run_system_once(project_damage).expect("reset marker");
        assert_eq!(world.get::<Visibility>(marker), Some(&Visibility::Hidden));
        assert_eq!(
            *world.get::<Transform>(front).expect("rotor"),
            Transform::default()
        );
    }

    #[test]
    fn projection_copies_physics_without_moving_unrelated_entities() {
        use bevy::ecs::system::RunSystemOnce;

        let mut session = Session::default();
        session.select(super::super::session::MotorPreset::Tilt);
        session.single_step();
        let observation = session.observation();
        let mut world = World::new();
        world.insert_resource(session);
        let drone = world.spawn((DroneBody, Transform::default())).id();
        let camera = world
            .spawn((Camera3d::default(), Transform::default()))
            .id();
        let unrelated = world
            .spawn((Name::new("Body"), Transform::from_xyz(7.0, 0.0, 0.0)))
            .id();
        for name in ["Rotor_FL", "Rotor_FR", "Rotor_BR", "Rotor_BL"] {
            world.spawn((Name::new(name), Transform::default()));
        }
        world.run_system_once(project).expect("projection runs");
        let transform = world
            .entity(drone)
            .get::<Transform>()
            .expect("drone transform");
        assert_eq!(transform.translation, observation.position());
        assert_eq!(transform.rotation, observation.orientation());
        assert_eq!(
            world
                .entity(camera)
                .get::<Transform>()
                .expect("camera transform")
                .translation,
            observation.position() + CAMERA_OFFSET
        );
        assert_eq!(
            world
                .entity(unrelated)
                .get::<Transform>()
                .expect("unrelated transform")
                .translation,
            Vec3::X * 7.0
        );
        assert_eq!(world.resource::<Session>().observation(), observation);
    }

    #[test]
    fn model_centre_and_front_align_with_the_physics_body() {
        let transform = model_alignment();
        assert!(transform.transform_point(BODY_CENTRE).length() < 1e-6);
        assert!((transform.rotation * Vec3::Z).distance(Vec3::NEG_Z) < 1e-6);
    }

    #[test]
    fn every_rotor_spins_about_its_own_pivot_and_maps_to_a_motor() {
        for (name, expected) in ["Rotor_FL", "Rotor_FR", "Rotor_BR", "Rotor_BL"]
            .into_iter()
            .zip(DroneMotor::ALL)
        {
            let (pivot, motor, sign) = rotor(name).expect("known rotor");
            assert_eq!(expected, motor);
            let transform = spin_about(pivot, sign * FRAC_PI_2);
            assert!(transform.transform_point(pivot).distance(pivot) < 1e-6);
            assert!(
                transform
                    .transform_point(pivot + Vec3::X)
                    .distance(pivot + Vec3::Z * -sign)
                    < 1e-6
            );
        }
        assert!(rotor("Body").is_none());
    }
}

#[cfg(test)]
mod tracking_tests {
    use super::*;
    use crate::session::MotorPreset;

    #[test]
    fn tracking_camera_keeps_both_route_ends_above_controls_at_mobile_width() {
        let target = Vec3::new(8.0, 2.0, 0.0);
        for position in [Vec3::new(0.0, 2.0, 0.0), target] {
            let camera = camera_pose(position, Some(target));
            let inverse = camera.compute_affine().inverse();
            let half_fov = PerspectiveProjection::default().fov / 2.0;
            // The 390-pixel browser leaves a 375 by 760 CSS-pixel canvas after scrolling space.
            let aspect = 375.0 / 760.0;
            for point in [position, target] {
                let local = inverse.transform_point3(point);
                let half_height = -local.z * half_fov.tan();
                let horizontal = local.x / (half_height * aspect);
                let vertical = local.y / half_height;
                assert!(horizontal.abs() < 0.95);
                assert!((0.05..0.55).contains(&vertical));
            }
        }
        let arrived = camera_pose(target, Some(target));
        assert!(arrived
            .translation
            .abs_diff_eq(target + CAMERA_OFFSET, 0.000_001));
    }

    #[test]
    fn waypoint_marker_follows_controller_selection_without_changing_physics() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Scene>()
            .init_resource::<Session>()
            .add_systems(Startup, setup)
            .add_systems(Update, project_target);
        app.update();
        let world = app.world_mut();
        let marker = world
            .query_filtered::<Entity, With<FlightTarget>>()
            .single(world)
            .unwrap();
        assert_eq!(
            *world.get::<Visibility>(marker).unwrap(),
            Visibility::Hidden
        );
        world.resource_mut::<Session>().select_tracking();
        let initial = world.resource::<Session>().observation();
        app.update();
        let world = app.world_mut();
        assert_eq!(
            world.get::<Transform>(marker).unwrap().translation,
            Vec3::new(8.0, 2.0, 0.0)
        );
        assert_eq!(
            *world.get::<Visibility>(marker).unwrap(),
            Visibility::Inherited
        );
        assert_eq!(world.resource::<Session>().observation(), initial);
        assert_eq!(world.get::<Children>(marker).unwrap().len(), 11);
        world.resource_mut::<Session>().select(MotorPreset::Hover);
        app.update();
        assert_eq!(
            *app.world().get::<Visibility>(marker).unwrap(),
            Visibility::Hidden
        );
    }
}
