//! Model alignment, lights, and read-only projection of the physics observation.

use std::f32::consts::{FRAC_PI_2, PI};

use bevy::prelude::*;
use bevy_gym::robots::{DroneMotor, DroneMotorState};

use super::session::Session;

/// The transform whose position and orientation come directly from physics.
#[derive(Component)]
pub(super) struct DroneBody;

/// Read-only ring identifying the failed front-left actuator.
#[derive(Component)]
pub(super) struct FailedMotorMarker;

/// Retained model handle, including its observable loading or failure state.
#[derive(Resource)]
pub(super) struct DroneModel(pub(super) Handle<Scene>);

/// Source-model body centre in metres, before its +Z front is rotated to -Z.
const BODY_CENTRE: Vec3 = Vec3::new(0.000_070_5, -0.055_414, -0.134_026_5);

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

/// Translate the model's body centre to zero, then turn its front toward -Z.
fn model_alignment() -> Transform {
    let rotation = Quat::from_rotation_y(PI);
    Transform::from_translation(rotation * -BODY_CENTRE).with_rotation(rotation)
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
    **camera = Transform::from_translation(position + CAMERA_OFFSET).looking_at(position, Vec3::Y);
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

/// Source-space pivots and motor identities for the four named mesh nodes.
pub(super) fn rotor(name: &str) -> Option<(Vec3, DroneMotor, f32)> {
    let (x, z, motor, sign) = match name {
        "Rotor_FL" => (0.250_664_5, 0.126_471, DroneMotor::FrontLeft, 1.0),
        "Rotor_FR" => (-0.250_420_5, 0.126_471, DroneMotor::FrontRight, -1.0),
        "Rotor_BR" => (-0.250_420_5, -0.394_633_5, DroneMotor::RearRight, 1.0),
        "Rotor_BL" => (0.250_664_5, -0.394_633_5, DroneMotor::RearLeft, -1.0),
        _ => return None,
    };
    Some((Vec3::new(x, 0.032_098_5, z), motor, sign))
}

/// Rotate vertices around their mesh centre without orbiting the whole rotor.
fn spin_about(pivot: Vec3, angle: f32) -> Transform {
    let rotation = Quat::from_rotation_y(angle);
    Transform::from_translation(pivot - rotation * pivot).with_rotation(rotation)
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
