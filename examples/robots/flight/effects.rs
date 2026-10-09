//! Bounded destruction feedback, isolated from the training environment.

use super::{debris, particles};

use super::session::{EpisodeRevision, Session};
use bevy::prelude::*;
use bevy_gym::{
    robots::{DroneMotor, DroneMotorState},
    EpisodeStatus,
};
use debris::{Debris, Fragment};
use particles::{Materials, Particle};
use std::time::Duration;

/// Render-only events, ordered with reset before any new destruction.
#[derive(Debug, PartialEq, Eq)]
enum Event {
    /// Remove all effects left by the previous episode.
    Reset,
    /// One actuator stopped working.
    Rotor(DroneMotor),
    /// The flight task terminated after contact or leaving its flight region.
    Crash,
}

/// Read-only transition tracking; never supplies inputs to the trained policy.
#[derive(Resource, Default)]
struct Destruction {
    /// Last observed reset revision.
    revision: EpisodeRevision,
    /// Failures already shown, in action order.
    failed: [bool; 4],
    /// Whether this episode has already emitted its crash event.
    crashed: bool,
}

/// Motor order shared with the observation and action recipes.
const MOTORS: [DroneMotor; 4] = [
    DroneMotor::FrontLeft,
    DroneMotor::FrontRight,
    DroneMotor::RearRight,
    DroneMotor::RearLeft,
];

impl Destruction {
    /// Emit only newly observed transitions; allocation occurs only for events.
    fn observe(&mut self, session: &Session) -> Vec<Event> {
        let mut events = Vec::new();
        if self.revision != session.revision() {
            *self = Self {
                revision: session.revision(),
                ..Self::default()
            };
            events.push(Event::Reset);
        }
        let observation = session.observation();
        for (previous, motor) in self.failed.iter_mut().zip(MOTORS) {
            let failed = observation.motor_state(motor) == DroneMotorState::Failed;
            if failed && !*previous {
                events.push(Event::Rotor(motor));
            }
            *previous = failed;
        }
        // Task termination destroys the visual body; a time-limit truncation does not.
        if !self.crashed && session.status() == EpisodeStatus::Terminated {
            self.crashed = true;
            events.push(Event::Crash);
        }
        events
    }
}

/// A bounded smoke source, following either a damaged actuator or a fixed impact.
#[derive(Component)]
struct Plume {
    /// The source's position recipe.
    source: Source,
    /// Time remaining before emission stops.
    remaining: Duration,
    /// Time accumulated toward the next puff.
    elapsed: Duration,
}

/// Mutually exclusive smoke attachment choices.
#[expect(
    variant_size_differences,
    reason = "Both inline variants fit in sixteen bytes; boxing would allocate for each plume."
)]
enum Source {
    /// Follow the current world position of one rotor.
    Rotor(DroneMotor),
    /// Remain at the terminal position after the body disappears.
    Impact(Vec3),
}

/// Register rendering-only state and systems with the existing viewer.
pub(super) fn install(app: &mut App) {
    app.init_resource::<Destruction>()
        .init_resource::<Debris>()
        .add_systems(Startup, particles::setup)
        .add_systems(
            Update,
            (observe, hide_rotors, smoke, wreckage, particles::animate)
                .chain()
                .after(super::scene::project),
        );
}

/// Project authoritative changes once, then remove all transients on reset.
fn observe(
    mut commands: Commands<'_, '_>,
    session: Res<'_, Session>,
    mut state: ResMut<'_, Destruction>,
    mut debris: ResMut<'_, Debris>,
    materials: Res<'_, Materials>,
    transients: Query<'_, '_, Entity, Or<(With<Particle>, With<Fragment>, With<Plume>)>>,
    mut body: Query<'_, '_, &mut Visibility, With<super::scene::DroneBody>>,
) {
    for event in state.observe(&session) {
        match event {
            Event::Reset => {
                for entity in &transients {
                    commands.entity(entity).despawn();
                }
                *debris = Debris::default();
            }
            Event::Rotor(motor) => {
                let position = rotor_position(&session, motor);
                materials.burst(&mut commands, position, 0.18);
                commands.spawn(Plume {
                    source: Source::Rotor(motor),
                    remaining: Duration::from_secs(4),
                    elapsed: Duration::ZERO,
                });
            }
            Event::Crash => {
                let position = session.observation().position();
                materials.burst(&mut commands, position, 0.6);
                commands.spawn(Plume {
                    source: Source::Impact(position),
                    remaining: Duration::from_secs(2),
                    elapsed: Duration::ZERO,
                });
                let observation = session.observation();
                for (fragment, transform) in debris.burst(
                    Isometry3d::new(observation.position(), observation.orientation()),
                    observation.linear_velocity(),
                    crash_world(),
                ) {
                    commands.spawn((
                        fragment,
                        Mesh3d(materials.fragment.clone()),
                        MeshMaterial3d(materials.steel.clone()),
                        transform,
                    ));
                }
            }
        }
    }
    for mut visibility in &mut body {
        *visibility = if state.crashed {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
}

/// Preserve the hover viewer's existing isolated floor for visual wreckage.
fn crash_world() -> rapier3d::prelude::PhysicsWorld {
    use rapier3d::prelude::{ColliderBuilder, PhysicsWorld, RigidBodyBuilder, Vector};
    let mut world = PhysicsWorld::default();
    world.insert(
        RigidBodyBuilder::fixed().translation(Vector::new(0.0, -0.1, 0.0)),
        ColliderBuilder::cuboid(30.0, 0.1, 30.0),
    );
    world
}

/// Remove the failed rotor mesh while retaining its smoke source and physical failure.
fn hide_rotors(session: Res<'_, Session>, mut rotors: Query<'_, '_, (&Name, &mut Visibility)>) {
    for (name, mut visibility) in &mut rotors {
        if let Some((_, motor, _)) = super::scene::rotor(name.as_str()) {
            *visibility = match session.observation().motor_state(motor) {
                DroneMotorState::Working => Visibility::Inherited,
                DroneMotorState::Failed => Visibility::Hidden,
            };
        }
    }
}

/// Compute the same actuator centres used by the physics model, in metres.
fn rotor_position(session: &Session, motor: DroneMotor) -> Vec3 {
    let offset = match motor {
        DroneMotor::FrontLeft => Vec3::new(-0.2505, 0.0875, -0.2606),
        DroneMotor::FrontRight => Vec3::new(0.2505, 0.0875, -0.2606),
        DroneMotor::RearRight => Vec3::new(0.2505, 0.0875, 0.2606),
        DroneMotor::RearLeft => Vec3::new(-0.2505, 0.0875, 0.2606),
    };
    let observation = session.observation();
    observation.position() + observation.orientation() * offset
}

/// Emit at most one puff per frame and twelve per second for each finite source.
fn smoke(
    mut commands: Commands<'_, '_>,
    time: Res<'_, Time>,
    session: Res<'_, Session>,
    materials: Res<'_, Materials>,
    mut plumes: Query<'_, '_, (Entity, &mut Plume)>,
) {
    let delta = time.delta().min(Duration::from_millis(100));
    for (entity, mut plume) in &mut plumes {
        plume.remaining = plume.remaining.saturating_sub(delta);
        if plume.remaining.is_zero() {
            commands.entity(entity).despawn();
            continue;
        }
        plume.elapsed += delta;
        if plume.elapsed >= Duration::from_millis(84) {
            plume.elapsed = Duration::ZERO;
            let position = match plume.source {
                Source::Rotor(motor) => rotor_position(&session, motor),
                Source::Impact(position) => position,
            };
            materials.puff(&mut commands, position);
        }
    }
}

/// Project debris collision poses until the finite world expires.
fn wreckage(
    mut commands: Commands<'_, '_>,
    time: Res<'_, Time>,
    mut debris: ResMut<'_, Debris>,
    mut fragments: Query<'_, '_, (Entity, &Fragment, &mut Transform)>,
) {
    let expired = debris.tick(time.delta());
    for (entity, fragment, mut transform) in &mut fragments {
        if expired {
            commands.entity(entity).despawn();
        } else if let Some(pose) = debris.pose(fragment) {
            *transform = pose;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::MotorPreset;

    /// Count a visual component without requiring a renderer or GPU.
    fn count<T: Component>(app: &mut App) -> usize {
        let world = app.world_mut();
        world.query::<&T>().iter(world).count()
    }

    #[test]
    fn rotor_failure_emits_once_and_reset_rearms_it_even_without_a_step() {
        let mut effects = Destruction::default();
        let mut session = Session::default();
        assert!(effects.observe(&session).is_empty());
        session.fail_motor(DroneMotor::FrontLeft);
        assert_eq!(
            effects.observe(&session),
            vec![Event::Rotor(DroneMotor::FrontLeft)]
        );
        assert!(effects.observe(&session).is_empty());
        session.reset();
        assert_eq!(effects.observe(&session), vec![Event::Reset]);
        session.fail_motor(DroneMotor::FrontLeft);
        assert_eq!(
            effects.observe(&session),
            vec![Event::Rotor(DroneMotor::FrontLeft)]
        );
    }

    #[test]
    fn a_ground_crash_emits_once_but_a_time_limit_does_not() {
        let mut effects = Destruction::default();
        let mut session = Session::default();
        session.select(MotorPreset::PowerOff);
        session.toggle_playback();
        for _ in 0..100 {
            session.advance();
        }
        assert_eq!(effects.observe(&session), vec![Event::Crash]);
        assert!(effects.observe(&session).is_empty());
        session.reset();
        assert_eq!(effects.observe(&session), vec![Event::Reset]);
        session.toggle_playback();
        for _ in 0..500 {
            session.advance();
        }
        assert!(effects.observe(&session).is_empty());
    }
    #[test]
    fn rendering_bounds_lifetimes_clears_on_reset_and_restores_the_body() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<Image>()
            .init_asset::<StandardMaterial>()
            .init_resource::<Session>()
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                Duration::from_millis(100),
            ));
        install(&mut app);
        let body = app
            .world_mut()
            .spawn((super::super::scene::DroneBody, Visibility::Visible))
            .id();
        let rotor = app
            .world_mut()
            .spawn((Name::new("Rotor_FL"), Visibility::Inherited))
            .id();
        let other = app
            .world_mut()
            .spawn((Name::new("Body"), Visibility::Hidden))
            .id();
        app.update();
        let material_count = app.world().resource::<Assets<StandardMaterial>>().len();
        app.world_mut()
            .resource_mut::<Session>()
            .fail_motor(DroneMotor::FrontLeft);
        app.update();
        assert_eq!(count::<Plume>(&mut app), 1);
        assert_eq!(count::<Particle>(&mut app), 14);
        assert_eq!(
            app.world().get::<Visibility>(rotor),
            Some(&Visibility::Hidden)
        );
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_millis(10),
        ));
        app.update();
        assert_eq!(count::<Particle>(&mut app), 14);
        assert_eq!(
            app.world().get::<Visibility>(other),
            Some(&Visibility::Hidden)
        );
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_millis(100),
        ));
        app.world_mut()
            .spawn((Camera3d::default(), Transform::from_xyz(3.0, 3.0, 3.0)));
        for _ in 0..5 {
            app.update();
        }
        assert_eq!(count::<Plume>(&mut app), 1);
        app.world_mut().resource_mut::<Session>().reset();
        app.update();
        assert_eq!(count::<Particle>(&mut app), 0);
        assert_eq!(count::<Plume>(&mut app), 0);
        assert_eq!(
            app.world().get::<Visibility>(rotor),
            Some(&Visibility::Inherited)
        );
        {
            let mut session = app.world_mut().resource_mut::<Session>();
            session.select(MotorPreset::PowerOff);
            session.toggle_playback();
            for _ in 0..100 {
                session.advance();
            }
        }
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(body),
            Some(&Visibility::Hidden)
        );
        assert_eq!(count::<Fragment>(&mut app), 8);
        // Reset while bodies are still moving, then exercise a second complete burst.
        app.world_mut().resource_mut::<Session>().reset();
        app.update();
        assert_eq!(count::<Fragment>(&mut app), 0);
        assert!(app
            .world()
            .resource::<Debris>()
            .pose(&Fragment(rapier3d::prelude::RigidBodyHandle::invalid()))
            .is_none());
        {
            let mut session = app.world_mut().resource_mut::<Session>();
            session.select(MotorPreset::PowerOff);
            session.toggle_playback();
            for _ in 0..100 {
                session.advance();
            }
        }
        app.update();
        for _ in 0..65 {
            app.update();
        }
        assert_eq!(count::<Fragment>(&mut app), 0);
        assert_eq!(count::<Particle>(&mut app), 0);
        assert_eq!(count::<Plume>(&mut app), 0);
        app.world_mut().resource_mut::<Session>().reset();
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(body),
            Some(&Visibility::Visible)
        );
        assert_eq!(
            app.world().resource::<Assets<StandardMaterial>>().len(),
            material_count
        );
    }
    #[test]
    fn each_motor_emits_at_its_own_body_relative_point() {
        let mut session = Session::default();
        let mut state = Destruction::default();
        let mut points = Vec::new();
        for motor in MOTORS {
            session.fail_motor(motor);
            assert_eq!(state.observe(&session), vec![Event::Rotor(motor)]);
            let point = rotor_position(&session, motor);
            assert!(!points.contains(&point));
            points.push(point);
        }
        assert!(state.observe(&session).is_empty());
        session.reset();
        assert_eq!(state.observe(&session), vec![Event::Reset]);
    }
}
