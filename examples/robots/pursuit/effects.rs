//! Consume one-shot combat events through the shared destruction renderer.

use super::{
    debris::{Debris, Fragment},
    firing::Effect,
    particles::{self, Materials, Particle},
    Game,
};
use bevy::prelude::*;
use bevy_gym::robots::DroneMotor;
use rapier3d::prelude::{ColliderBuilder, PhysicsWorld, RigidBodyBuilder, Vector};
use std::time::Duration;

/// Rotor smoke or a timed destruction plume; reset removes every emitter.
#[derive(Component)]
struct Plume {
    /// Motor attachment or fixed body impact.
    source: Source,
    /// Accumulated time toward the next puff.
    elapsed: Duration,
}

impl Plume {
    /// Every new emitter starts with an empty puff clock.
    const fn new(source: Source) -> Self {
        Self {
            source,
            elapsed: Duration::ZERO,
        }
    }
}

/// A rotor follows the target; a body impact retains the destruction position.
enum Source {
    /// Follow one damaged rotor until destruction or reset.
    DamagedRotor(DroneMotor),
    /// Follow one named rotor, including after its mesh disappears.
    Rotor {
        /// Destroyed actuator whose moving attachment emits smoke.
        motor: DroneMotor,
        /// Time before the destruction plume expires.
        remaining: Duration,
    },
    /// Remain at the last body position.
    Impact {
        /// Fixed destruction position in world metres.
        position: Vec3,
        /// Time before the body plume expires.
        remaining: Duration,
    },
}

impl Source {
    /// Damaged rotors follow health; destruction plumes have finite lifetimes.
    fn advance(&mut self, game: &Game, delta: Duration) -> Option<Vec3> {
        let (position, remaining) = match self {
            Self::DamagedRotor(motor) => {
                let target = game.combat.target();
                return (target.health().is_alive()
                    && target.health().rotor(*motor)
                        == super::combat::health::RotorHealth::Damaged)
                    .then(|| target.rotor_centre(*motor));
            }
            Self::Rotor { motor, remaining } => {
                (game.combat.target().rotor_centre(*motor), remaining)
            }
            Self::Impact {
                position,
                remaining,
            } => (*position, remaining),
        };
        *remaining = remaining.saturating_sub(delta);
        (!remaining.is_zero()).then_some(position)
    }
}

/// Install shared assets and bounded transients after target transforms update.
pub(super) fn install(app: &mut App) {
    app.init_resource::<Debris>()
        .add_systems(Startup, particles::setup)
        .add_systems(
            Update,
            (observe, smoke, wreckage, particles::animate)
                .chain()
                .after(super::scene::project),
        );
}

/// Consume reset before subsequent damage, without deriving events from transient HUD text.
fn observe(
    mut commands: Commands<'_, '_>,
    mut game: ResMut<'_, Game>,
    mut debris: ResMut<'_, Debris>,
    materials: Res<'_, Materials>,
    transients: Query<'_, '_, Entity, Or<(With<Particle>, With<Fragment>, With<Plume>)>>,
) {
    while let Some(event) = game.combat.pop_effect() {
        match event {
            Effect::Reset => {
                for entity in &transients {
                    commands.entity(entity).despawn();
                }
                *debris = Debris::default();
            }
            Effect::RotorDamaged(motor) => {
                commands.spawn(Plume::new(Source::DamagedRotor(motor)));
            }
            Effect::Rotor(motor) => {
                let position = game.combat.target().rotor_centre(motor);
                materials.burst(&mut commands, position, 0.18);
                commands.spawn(Plume::new(Source::Rotor {
                    motor,
                    remaining: Duration::from_secs(4),
                }));
            }
            Effect::Destroyed => {
                let target = game.combat.target();
                let position = target.position();
                materials.burst(&mut commands, position, 0.6);
                commands.spawn(Plume::new(Source::Impact {
                    position,
                    remaining: Duration::from_secs(2),
                }));
                // Fragments inherit the last authoritative world-space velocity in metres per second.
                let pose = Isometry3d::new(position, target.rotation());
                for (fragment, transform) in debris.burst(
                    pose,
                    game.flight.observation().linear_velocity(),
                    collision_world(&game),
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
}

/// Copy immutable arena geometry into an independent presentation solver only on body death.
fn collision_world(game: &Game) -> PhysicsWorld {
    let mut world = PhysicsWorld::default();
    for block in game.arena.blocks() {
        world.insert(
            RigidBodyBuilder::fixed()
                .translation(Vector::from_array(block.centre.to_array()))
                .rotation(Vector::from_array(
                    block.rotation.to_scaled_axis().to_array(),
                )),
            ColliderBuilder::cuboid(block.half.x, block.half.y, block.half.z),
        );
    }
    world
}

/// Emit at most one puff per frame and twelve per second for each source.
fn smoke(
    mut commands: Commands<'_, '_>,
    time: Res<'_, Time>,
    game: Res<'_, Game>,
    materials: Res<'_, Materials>,
    mut plumes: Query<'_, '_, (Entity, &mut Plume)>,
) {
    let delta = time.delta().min(Duration::from_millis(100));
    for (entity, mut plume) in &mut plumes {
        let Some(position) = plume.source.advance(&game, delta) else {
            commands.entity(entity).despawn();
            continue;
        };
        plume.elapsed += delta;
        if plume.elapsed >= Duration::from_millis(84) {
            plume.elapsed = Duration::ZERO;
            materials.puff(&mut commands, position);
        }
    }
}

/// Project each fragment until its private collision world expires.
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
    use crate::{firing::Action, shot::target::Part};

    /// Run the presentation systems without a window or GPU.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<Image>()
            .init_asset::<StandardMaterial>()
            .init_resource::<Game>()
            .insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
                Duration::from_millis(100),
            ));
        install(&mut app);
        app.update();
        app
    }

    /// Shoot through the same character-origin boundary used by player controls.
    fn hit(game: &mut Game, part: Part, hits: u8) {
        if !game.combat.is_armed() {
            game.act(Action::PickUp);
        }
        let target = game.combat.target();
        let point = match part {
            Part::Body => target.position(),
            Part::Rotor(motor) => target.rotor_centre(motor),
        };
        let aim = Dir3::new(point - crate::firing::Combat::origin(game.arena.position()))
            .expect("Authored target direction");
        for _ in 0..hits {
            game.advance_combat(Duration::from_millis(250));
            game.act(Action::Fire(aim));
            game.advance_combat(Duration::from_millis(100));
        }
    }

    /// Count live presentation entities without depending on GPU output.
    fn count<T: Component>(app: &mut App) -> usize {
        let world = app.world_mut();
        world.query::<&T>().iter(world).count()
    }

    #[test]
    fn rotor_burst_emits_once_and_reset_precedes_same_frame_damage() {
        let mut app = app();
        let material_count = app.world().resource::<Assets<StandardMaterial>>().len();
        let rotor = Part::Rotor(DroneMotor::RearRight);
        hit(&mut app.world_mut().resource_mut::<Game>(), rotor, 1);
        app.update();
        assert_eq!(count::<Plume>(&mut app), 1);
        assert_eq!(count::<Particle>(&mut app), 1);
        hit(&mut app.world_mut().resource_mut::<Game>(), rotor, 1);
        app.update();
        assert_eq!(count::<Particle>(&mut app), 15);
        assert_eq!(count::<Plume>(&mut app), 1);
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_millis(10),
        ));
        app.update();
        assert_eq!(count::<Particle>(&mut app), 15);
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_millis(100),
        ));
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.reset();
            hit(&mut game, rotor, 2);
        }
        app.update();
        assert_eq!(count::<Particle>(&mut app), 14);
        assert_eq!(count::<Plume>(&mut app), 1);
        for _ in 0..65 {
            app.update();
        }
        assert_eq!(count::<Particle>(&mut app), 0);
        assert_eq!(count::<Plume>(&mut app), 0);
        assert_eq!(
            app.world().resource::<Assets<StandardMaterial>>().len(),
            material_count
        );
    }

    #[test]
    fn first_hit_smokes_only_its_rotor_until_reset() {
        let mut app = app();
        let motor = DroneMotor::RearRight;
        hit(
            &mut app.world_mut().resource_mut::<Game>(),
            Part::Rotor(motor),
            1,
        );
        app.update();
        let position = app
            .world()
            .resource::<Game>()
            .combat
            .target()
            .rotor_centre(motor);
        let world = app.world_mut();
        let smoke = world
            .query_filtered::<&Transform, With<Particle>>()
            .single(world)
            .expect("One puff");
        assert!(smoke.translation.distance(position) < 0.1);
        for _ in 0..60 {
            app.update();
        }
        assert_eq!(count::<Plume>(&mut app), 1);
        assert!(count::<Particle>(&mut app) > 0);
        hit(&mut app.world_mut().resource_mut::<Game>(), Part::Body, 6);
        app.update();
        assert_eq!(count::<Plume>(&mut app), 1);
        app.world_mut().resource_mut::<Game>().reset();
        app.update();
        assert_eq!(count::<Plume>(&mut app), 0);
        assert_eq!(count::<Particle>(&mut app), 0);
    }

    #[test]
    fn body_burst_moves_expires_and_reset_clears_in_flight_fragments() {
        let mut app = app();
        for reset_early in [true, false] {
            hit(&mut app.world_mut().resource_mut::<Game>(), Part::Body, 6);
            app.update();
            assert_eq!(count::<Fragment>(&mut app), 8);
            assert_eq!(count::<Plume>(&mut app), 1);
            let world = app.world_mut();
            let (entity, before) = world
                .query_filtered::<(Entity, &Transform), With<Fragment>>()
                .iter(world)
                .next()
                .map(|(entity, pose)| (entity, pose.translation))
                .expect("Fragment");
            app.update();
            assert_ne!(
                app.world()
                    .get::<Transform>(entity)
                    .expect("Live fragment")
                    .translation,
                before
            );
            assert_eq!(count::<Fragment>(&mut app), 8);
            if reset_early {
                app.world_mut().resource_mut::<Game>().reset();
                app.update();
            } else {
                for _ in 0..65 {
                    app.update();
                }
            }
            assert_eq!(count::<Fragment>(&mut app), 0);
            assert_eq!(count::<Particle>(&mut app), 0);
            assert_eq!(count::<Plume>(&mut app), 0);
        }
    }

    #[test]
    fn debris_collides_with_the_authored_house_wall() {
        let game = Game::default();
        let world = collision_world(&game);
        assert_eq!(world.colliders.len(), game.arena.blocks().len());
        let mut debris = Debris::default();
        let fragments = debris.burst(
            Isometry3d::from_translation(Vec3::new(-1.5, 1.4, -5.0)),
            Vec3::NEG_X * 3.0,
            world,
        );
        for _ in 0..20 {
            debris.tick(Duration::from_micros(16_667));
        }
        // This part of the east wall has no doorway: the outside face is X=-1.85 m.
        assert!(fragments.iter().all(|(fragment, _)| debris
            .pose(fragment)
            .expect("Live body")
            .translation
            .x
            > -1.85));
        assert!(fragments.iter().any(|(fragment, initial)| debris
            .pose(fragment)
            .expect("Live body")
            .translation
            .x
            > initial.translation.x));
        assert_eq!(game.arena.position(), Vec3::new(0.0, 0.92, 9.0));
    }
}
