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

/// A finite smoke emitter; reset removes it together with particles and debris.
#[derive(Component)]
struct Plume {
    /// Motor attachment or fixed body impact.
    source: Source,
    /// Time before emission stops.
    remaining: Duration,
    /// Accumulated time toward the next puff.
    elapsed: Duration,
}

/// A rotor follows the target; a body impact retains the destruction position.
#[expect(
    variant_size_differences,
    reason = "Both variants fit inline without allocating a plume payload."
)]
enum Source {
    /// Follow one named rotor, including after its mesh disappears.
    Rotor(DroneMotor),
    /// Remain at the last body position.
    Impact(Vec3),
}

/// Install shared assets and bounded transients after target transforms update.
pub(super) fn install(app: &mut App) {
    app.init_resource::<Debris>()
        .add_systems(Startup, particles::setup)
        .add_systems(
            Update,
            (observe, smoke, wreckage, particles::animate)
                .chain()
                .after(super::weapons::project),
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
            Effect::Rotor(motor) => {
                let position = game.combat.target().rotor_centre(motor);
                materials.burst(&mut commands, position, 0.18);
                commands.spawn(Plume {
                    source: Source::Rotor(motor),
                    remaining: Duration::from_secs(4),
                    elapsed: Duration::ZERO,
                });
            }
            Effect::Destroyed => {
                let target = game.combat.target();
                let position = target.position();
                materials.burst(&mut commands, position, 0.6);
                commands.spawn(Plume {
                    source: Source::Impact(position),
                    remaining: Duration::from_secs(2),
                    elapsed: Duration::ZERO,
                });
                // The current target is stationary; future flight must supply its actual velocity.
                let pose = Isometry3d::new(position, target.rotation());
                for (fragment, transform) in debris.burst(pose, Vec3::ZERO, collision_world(&game))
                {
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

/// Emit at most one puff per frame and twelve per second for each finite source.
fn smoke(
    mut commands: Commands<'_, '_>,
    time: Res<'_, Time>,
    game: Res<'_, Game>,
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
                Source::Rotor(motor) => game.combat.target().rotor_centre(motor),
                Source::Impact(position) => position,
            };
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
            game.combat.advance(Duration::from_millis(250));
            game.act(Action::Fire(aim));
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
        assert_eq!(count::<Particle>(&mut app), 0);
        hit(&mut app.world_mut().resource_mut::<Game>(), rotor, 1);
        app.update();
        assert_eq!(count::<Particle>(&mut app), 14);
        assert_eq!(count::<Plume>(&mut app), 1);
        app.insert_resource(bevy::time::TimeUpdateStrategy::ManualDuration(
            Duration::from_millis(10),
        ));
        app.update();
        assert_eq!(count::<Particle>(&mut app), 14);
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
