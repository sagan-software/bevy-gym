//! Explore a collision-aware arena as an articulated robot.
//!
//! Run `cargo run --features robots --example drone-pursuit`.
//! `pursuit-walk` shows the same movement loop without rendering.

#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/audio.rs"]
mod audio;
#[path = "pursuit/controls.rs"]
mod controls;
#[path = "destruction_debris.rs"]
mod debris;
#[path = "destruction_particles.rs"]
mod particles;
#[path = "pursuit/return_fire.rs"]
mod return_fire;
#[path = "pursuit/robot.rs"]
mod robot;
#[path = "pursuit/scene.rs"]
mod scene;

#[path = "pursuit/combat.rs"]
mod combat;
#[path = "pursuit/enemy.rs"]
mod enemy;
#[path = "pursuit/firing.rs"]
mod firing;
#[path = "pursuit/shot.rs"]
mod shot;

#[path = "drone_model.rs"]
mod drone_model;

#[path = "pursuit/aiming.rs"]
mod aiming;
#[cfg(test)]
#[path = "pursuit/return_fire_tests.rs"]
mod return_fire_tests;

#[path = "pursuit/effects.rs"]
mod effects;
#[path = "pursuit/hearing.rs"]
mod hearing;
#[path = "pursuit/hud.rs"]
mod hud;
#[path = "pursuit/perception.rs"]
mod perception;
#[path = "pursuit/sight.rs"]
mod sight;
#[path = "pursuit/weapons.rs"]
mod weapons;

use arena::{Arena, Movement};
use bevy::{asset::AssetMetaCheck, prelude::*};

#[cfg(test)]
#[path = "pursuit/flight_tests.rs"]
mod flight_tests;

#[cfg(test)]
#[path = "pursuit/perception_tests.rs"]
mod perception_tests;

#[cfg(test)]
#[path = "pursuit/hearing_tests.rs"]
mod hearing_tests;

#[path = "pursuit/flight.rs"]
mod flight;
#[path = "flight_control/mod.rs"]
mod flight_control;
#[path = "pursuit/navigation/mod.rs"]
mod navigation;

#[path = "pursuit/camera.rs"]
mod camera;
use camera::EYE_OFFSET as DRONE_EYE;

/// One character's simulation and measured motion for visual animation.
#[derive(Resource)]
struct Game {
    /// Shared movement and collision model.
    arena: Arena,
    /// Pickup, target damage, and firing feedback.
    combat: firing::Combat,
    /// Private flight physics, programmed search, and learned motor control.
    flight: flight::Flight,
    /// Current sighting or bounded last-seen memory, without hidden target state.
    sight: sight::Sight,
    /// Bounded sound events cannot affect aiming, damage, or observations.
    sound: audio::Events,
    /// Drone weapon timing uses only the current filtered sighting.
    gun: enemy::gun::Gun,
    /// At most three moving rounds, independent of the drone's later survival.
    projectiles: enemy::projectile::Projectiles,
    /// Authoritative humanoid life; death blocks player and future policy actions.
    robot_health: enemy::robot_health::RobotHealth,
    /// Last audible event; hidden source coordinates are never retained.
    hearing: hearing::Hearing,
    /// Distance walked in metres since reset, used as animation phase.
    distance: f32,
    /// Grounded travel since the last footstep, in metres below one 0.8 m stride.
    footstep_distance: f32,
    /// Last horizontal displacement, in metres per action.
    motion: Vec3,
    /// Last nonzero movement heading in radians; retained while idle.
    heading: f32,
    /// Latest validated pointing direction, separate from walking direction.
    aim: Option<Dir3>,
}

impl Default for Game {
    fn default() -> Self {
        let arena = Arena::default();
        let flight = flight::Flight::new(&arena);
        let mut combat = firing::Combat::default();
        combat
            .project_flight(flight.observation())
            .expect("Initial flight pose is valid");
        Self {
            arena,
            combat,
            flight,
            sight: sight::Sight::default(),
            hearing: hearing::Hearing::default(),
            sound: audio::Events::default(),
            gun: enemy::gun::Gun::default(),
            projectiles: enemy::projectile::Projectiles::default(),
            robot_health: enemy::robot_health::RobotHealth::default(),
            distance: 0.0,
            footstep_distance: 0.0,
            motion: Vec3::ZERO,
            heading: 0.0,
            aim: None,
        }
    }
}

impl Game {
    /// Reset simulation and animation together.
    fn reset(&mut self) {
        self.arena.reset();
        self.combat = firing::Combat::default();
        self.flight.reset();
        self.sight.forget();
        self.hearing.forget();
        self.sound.reset();
        self.gun = enemy::gun::Gun::default();
        self.projectiles = enemy::projectile::Projectiles::default();
        self.robot_health = enemy::robot_health::RobotHealth::default();
        self.combat
            .project_flight(self.flight.observation())
            .expect("Reset flight pose is valid");
        self.distance = 0.0;
        self.footstep_distance = 0.0;
        self.motion = Vec3::ZERO;
        self.heading = 0.0;
        self.aim = None;
    }

    /// Derive the same body-mounted camera pose used by sensing and the lens mesh.
    fn eye(&self) -> Isometry3d {
        let target = self.combat.target();
        Isometry3d::new(
            target.position() + target.rotation() * DRONE_EYE,
            target.rotation(),
        )
    }

    /// Sample after motion; unavailable flight cannot retain a target observation.
    fn sample_sight(&mut self) {
        if !self.combat.target().health().is_alive()
            || self.flight.error().is_some()
            || !self.robot_health.is_alive()
        {
            self.sight.forget();
            self.hearing.forget();
            return;
        }
        let eye = self.eye();
        self.sight.sample(
            &self.arena,
            eye.translation.into(),
            eye.rotation * Dir3::NEG_Z,
            self.arena.position(),
            std::time::Duration::from_millis(20),
        );
    }

    /// Original under-body weapon position, derived from the authoritative flight pose.
    fn muzzle(&self) -> Vec3 {
        let target = self.combat.target();
        target.position() + target.rotation() * Vec3::new(0.0, -0.14, -0.22)
    }

    /// Submit an actual event only while the listener can sense.
    fn hear(&mut self, source: Vec3, noise: hearing::Noise) {
        if self.combat.target().health().is_alive()
            && self.flight.error().is_none()
            && self.robot_health.is_alive()
        {
            self.hearing
                .hear(&self.arena, self.combat.target().position(), source, noise);
        }
    }

    /// Armed characters face their aim while retaining independent movement.
    fn facing(&self) -> f32 {
        self.aim
            .filter(|_| self.combat.is_armed())
            .map_or(self.heading, |aim| (-aim.x).atan2(-aim.z))
    }

    /// Elevate the held pistol and shoulder along the current aim.
    fn pitch(&self) -> f32 {
        self.aim.map_or(0.0, |aim| aim.y.clamp(-1.0, 1.0).asin())
    }

    /// Interactions use the current arena position rather than a caller-supplied origin.
    fn act(&mut self, action: firing::Action) {
        if !self.robot_health.is_alive() {
            return;
        }
        let previous_rounds = self.combat.rounds();
        self.combat.act(&self.arena, action);
        if !self.combat.target().health().is_alive() {
            self.gun.disable();
            self.sight.forget();
            self.hearing.forget();
        } else if self.combat.rounds() < previous_rounds {
            // Accepted shots spend ammunition; retained feedback cannot replay a noise.
            self.hear(
                firing::Combat::origin(self.arena.position()),
                hearing::Noise::Gunshot,
            );
        }
    }

    /// Animate measured displacement, so holding a blocked direction does not walk in place.
    fn step(&mut self, movement: Movement) {
        self.combat.advance(std::time::Duration::from_millis(20));
        self.hearing.advance(std::time::Duration::from_millis(20));
        if let Some(step) = self
            .flight
            .advance(self.combat.target().health(), self.sight.contact())
        {
            if let Err(error) = self.combat.project_flight(step.observation) {
                self.flight.fail(format!("Invalid flight pose: {error:?}"));
            } else if step.is_done() {
                self.combat.crash();
            }
        }
        let previous = self.arena.position();
        let movement = if self.robot_health.is_alive() {
            movement
        } else {
            Movement::Idle
        };
        self.arena.step(movement);
        self.motion = (self.arena.position() - previous) * Vec3::new(1.0, 0.0, 1.0);
        let distance = self.motion.length();
        self.distance += distance;
        if distance > 0.001 {
            self.heading = (-self.motion.x).atan2(-self.motion.z);
        }
        self.projectiles
            .advance(&self.arena, &mut self.robot_health);
        self.sample_sight();
        if self.robot_health.is_alive()
            && self.combat.target().health().is_alive()
            && self.flight.error().is_none()
        {
            if let Some(projectile) =
                self.gun
                    .advance(&self.arena, self.muzzle(), self.sight.contact())
            {
                // A volley expires before another can finish its full warning and cooldown.
                self.sound.shot(projectile.position());
                self.projectiles
                    .launch(projectile)
                    .expect("Weapon cadence bounds live projectiles to three");
            }
        } else {
            self.gun.disable();
        }
        // Only measured grounded travel crosses a walking stride. A blocked input is silent.
        if self.arena.is_grounded() {
            self.footstep_distance += distance;
            // A 20 ms action moves at most 0.08 m, so it can cross only one stride.
            if self.footstep_distance >= 0.8 {
                self.footstep_distance -= 0.8;
                // Emit above the sole so the floor endpoint is not an occluder.
                self.hear(
                    self.arena.position() - Vec3::Y * 0.7,
                    hearing::Noise::Footstep,
                );
            }
        }
    }
}

/// Launch the player-controlled arena with programmed search and learned flight.
fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Pursuit arena | Bevy Gym".to_owned(),
                        canvas: Some("#pursuit-canvas".to_owned()),
                        fit_canvas_to_parent: true,
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: if cfg!(target_arch = "wasm32") {
                        "assets"
                    } else {
                        concat!(env!("CARGO_MANIFEST_DIR"), "/assets")
                    }
                    .to_owned(),
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                }),
        )
        .init_resource::<Game>()
        .init_resource::<controls::Input>()
        .insert_resource(Time::<Fixed>::from_seconds(0.02))
        .add_systems(
            Startup,
            (scene::setup, controls::setup, weapons::setup, hud::setup),
        )
        .add_systems(PreUpdate, controls::read.after(bevy::input::InputSystems))
        .add_systems(FixedUpdate, advance)
        .add_systems(
            Update,
            (
                scene::project,
                aiming::read,
                weapons::project,
                weapons::rotors,
                weapons::traces,
                robot::animate,
                hud::project,
            )
                .chain(),
        )
        .add_plugins((
            effects::install,
            perception::install,
            return_fire::install,
            audio::install,
        ))
        .run();
}

/// Player buttons and future policy output enter the same fixed action boundary.
fn advance(mut game: ResMut<'_, Game>, mut input: ResMut<'_, controls::Input>) {
    game.step(input.movement);
    game.aim = input.aim;
    if let Some(action) = input.action.take() {
        game.act(action);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise rendering projection and input without opening a GPU window.
    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<Font>()
            .init_asset::<Scene>()
            .init_asset::<Image>()
            .init_resource::<Game>()
            .init_resource::<controls::Input>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_systems(
                Startup,
                (scene::setup, controls::setup, weapons::setup, hud::setup),
            )
            .add_systems(
                Update,
                (
                    controls::read,
                    aiming::read,
                    advance,
                    scene::project,
                    weapons::project,
                    weapons::rotors,
                    weapons::traces,
                    robot::animate,
                    hud::project,
                )
                    .chain(),
            );
        let mut gizmos = GizmoConfigStore::default();
        gizmos.insert(GizmoConfig::default(), DefaultGizmoConfigGroup);
        app.insert_resource(gizmos)
            .init_resource::<bevy::gizmos::gizmos::GizmoStorage<DefaultGizmoConfigGroup, ()>>();
        effects::install(&mut app);
        perception::install(&mut app);
        return_fire::install(&mut app);
        app.update();
        app
    }

    #[test]
    fn flight_failure_is_visible_and_reset_restores_pursuit() {
        let mut app = app();
        app.world_mut()
            .resource_mut::<Game>()
            .flight
            .fail("Invalid checkpoint".to_owned());
        app.update();
        let world = app.world_mut();
        let text = world
            .query_filtered::<&Text, With<hud::Status>>()
            .single(world)
            .expect("Status");
        assert!(text.0.contains("Flight stopped: Invalid checkpoint"));
        app.world_mut().resource_mut::<Game>().reset();
        app.update();
        let world = app.world_mut();
        let text = world
            .query_filtered::<&Text, With<hud::Status>>()
            .single(world)
            .expect("Status");
        assert_eq!(text.0, "Programmed search · learned flight");
    }

    #[test]
    fn player_motion_drives_pose_animation_and_reset() {
        let mut app = app();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyD);
        app.update();
        let game = app.world().resource::<Game>();
        assert!(game.distance > 0.07);
        assert!(game.heading < -1.5);
        let position = game.arena.position();
        let world = app.world_mut();
        let mut roots = world.query_filtered::<&Transform, With<scene::Character>>();
        assert_eq!(
            roots.single(world).expect("one character").translation,
            position
        );
        let mut joints = world.query_filtered::<&Transform, With<robot::Joint>>();
        assert!(joints
            .iter(world)
            .all(|joint| joint.rotation != Quat::IDENTITY));
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.update();
        let world = app.world_mut();
        let mut joints = world.query_filtered::<&Transform, With<robot::Joint>>();
        assert!(joints
            .iter(world)
            .all(|joint| joint.rotation == Quat::IDENTITY));
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyR);
        app.update();
        assert_eq!(app.world().resource::<Game>().distance, 0.0);
        assert_eq!(app.world().resource::<Game>().heading, 0.0);
        app.update();
        assert_eq!(
            app.world().resource::<controls::Input>().movement,
            Movement::Idle
        );
    }

    #[test]
    fn reset_tap_between_frames_is_not_lost() {
        let mut app = app();
        app.world_mut().resource_mut::<Game>().step(Movement::Right);
        let mut keys = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keys.press(KeyCode::KeyR);
        keys.release(KeyCode::KeyR);
        app.update();
        assert!(app.world().resource::<Game>().distance.abs() < f32::EPSILON);
    }

    #[test]
    fn held_buttons_share_keyboard_actions_and_camera_avoids_house_wall() {
        let mut app = app();
        let world = app.world_mut();
        let buttons: Vec<Entity> = world
            .query_filtered::<Entity, With<controls::Control>>()
            .iter(world)
            .collect();
        for entity in buttons {
            *app.world_mut()
                .get_mut::<Interaction>(entity)
                .expect("button") = Interaction::Pressed;
            app.update();
            *app.world_mut()
                .get_mut::<Interaction>(entity)
                .expect("button") = Interaction::None;
            app.update();
        }
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.reset();
            for _ in 0..150 {
                game.step(Movement::Forward);
            }
            for _ in 0..62 {
                game.step(Movement::Left);
            }
        }
        app.update();
        let game = app.world().resource::<Game>();
        let target = game.arena.position() + Vec3::Y * 0.4;
        let world = app.world_mut();
        let mut cameras = world.query_filtered::<&Transform, With<Camera3d>>();
        let eye = cameras.single(world).expect("one camera").translation;
        assert!(eye.distance(target) < 4.0);
        assert!(world
            .resource::<Game>()
            .arena
            .obstruction(target, eye)
            .is_none());
    }
    #[test]
    fn open_arena_keeps_the_original_camera_offset() {
        let mut app = app();
        app.update();
        let world = app.world_mut();
        let target = world.resource::<Game>().arena.position() + Vec3::Y * 0.4;
        let eye = world
            .query_filtered::<&Transform, With<Camera3d>>()
            .single(world)
            .expect("One camera")
            .translation;
        assert!((eye - target).abs_diff_eq(Vec3::new(0.0, 3.3, 6.0), 1e-5));
    }

    #[test]
    fn pipe_camera_keeps_the_upper_subject_view_clear_of_the_ceiling() {
        let mut app = app();
        for _ in 0..62 {
            app.world_mut()
                .resource_mut::<Game>()
                .arena
                .step(Movement::Right);
        }
        for _ in 0..100 {
            app.world_mut()
                .resource_mut::<Game>()
                .arena
                .step(Movement::Forward);
            app.update();
            let world = app.world_mut();
            let position = world.resource::<Game>().arena.position();
            if position.z >= 4.0 {
                continue;
            }
            let mut cameras = world.query_filtered::<(&Transform, &Projection), With<Camera3d>>();
            let (eye, projection) = cameras.single(world).expect("One camera");
            let Projection::Perspective(projection) = projection else {
                panic!("Perspective camera");
            };
            let upper =
                eye.rotation * Vec3::new(0.0, (projection.fov * 0.5).tan() * 0.5, -1.0).normalize();
            let from = eye.translation;
            let end = from + upper * from.distance(position);
            assert!(
                world
                    .resource::<Game>()
                    .arena
                    .obstruction(from, end)
                    .is_none(),
                "Pipe camera upper subject view is obstructed at {position:?}: {from:?} -> {end:?}"
            );
        }
    }

    #[test]
    fn pickup_shooting_feedback_and_reset_share_the_game_state() {
        let mut game = Game::default();
        game.act(firing::Action::Fire(Dir3::NEG_Z));
        assert!(!game.combat.is_armed());
        game.act(firing::Action::PickUp);
        assert!(game.combat.is_armed());
        assert_eq!(game.combat.rounds(), 12);
        let target = game.combat.target().position();
        let origin = firing::Combat::origin(game.arena.position());
        let direction = Dir3::new(target - origin).expect("Target ahead");
        game.act(firing::Action::Fire(direction));
        assert_eq!(game.combat.rounds(), 11);
        assert_eq!(game.combat.target().health().body_hits_remaining(), 5);
        assert!(game.combat.trace().is_some());
        game.act(firing::Action::Fire(direction));
        assert_eq!(game.combat.rounds(), 11);
        for _ in 0..20 {
            game.step(Movement::Idle);
        }
        assert!(game.combat.trace().is_none());
        game.reset();
        assert!(!game.combat.is_armed());
        assert_eq!(game.combat.target().health().body_hits_remaining(), 6);
        assert!(game.combat.trace().is_none());
    }

    #[test]
    fn queued_pickup_survives_until_the_next_simulation_action() {
        let mut app = app();
        app.world_mut().resource_mut::<controls::Input>().action = Some(firing::Action::PickUp);
        app.update();
        assert!(app.world().resource::<Game>().combat.is_armed());
        assert!(app.world().resource::<controls::Input>().action.is_none());
    }
    #[test]
    fn keyboard_pickup_and_ui_hover_do_not_accidentally_fire() {
        let mut app = app();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyE);
        app.update();
        assert!(app.world().resource::<Game>().combat.is_armed());
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut().resource_mut::<controls::Input>().aim = Some(Dir3::Y);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        let world = app.world_mut();
        let entity = world
            .query_filtered::<Entity, With<controls::Control>>()
            .iter(world)
            .next()
            .expect("Movement control");
        *world.get_mut::<Interaction>(entity).expect("Interaction") = Interaction::Hovered;
        app.update();
        assert_eq!(app.world().resource::<Game>().combat.rounds(), 12);
        *app.world_mut()
            .get_mut::<Interaction>(entity)
            .expect("Interaction") = Interaction::None;
        app.update();
        assert_eq!(app.world().resource::<Game>().combat.rounds(), 11);
    }

    #[test]
    fn pickup_and_fire_buttons_use_the_same_actions() {
        let mut app = app();
        let world = app.world_mut();
        let buttons: Vec<_> = world
            .query::<(Entity, &hud::ActionButton)>()
            .iter(world)
            .map(|(entity, button)| (entity, matches!(button, hud::ActionButton::PickUp)))
            .collect();
        for (entity, pickup) in buttons {
            if pickup {
                *app.world_mut()
                    .get_mut::<Interaction>(entity)
                    .expect("Button") = Interaction::Pressed;
                app.update();
                *app.world_mut()
                    .get_mut::<Interaction>(entity)
                    .expect("Button") = Interaction::None;
            }
        }
        assert!(app.world().resource::<Game>().combat.is_armed());
        app.world_mut().resource_mut::<controls::Input>().aim = Some(Dir3::Y);
        let world = app.world_mut();
        let fire = world
            .query::<(Entity, &hud::ActionButton)>()
            .iter(world)
            .find_map(|(entity, button)| {
                matches!(button, hud::ActionButton::Fire).then_some(entity)
            })
            .expect("Fire button");
        *world.get_mut::<Interaction>(fire).expect("Button") = Interaction::Pressed;
        app.update();
        assert_eq!(app.world().resource::<Game>().combat.rounds(), 11);
        app.update();
        assert_eq!(app.world().resource::<Game>().combat.rounds(), 11);
    }

    #[test]
    fn damage_and_reset_update_the_named_rotor_and_weapon_visibility() {
        let mut app = app();
        let rotor = app
            .world_mut()
            .spawn((
                Name::new("Rotor_BR"),
                Transform::default(),
                Visibility::Inherited,
            ))
            .id();
        let unrelated = app
            .world_mut()
            .spawn((
                Name::new("Body"),
                Transform::default(),
                Visibility::Inherited,
            ))
            .id();
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.act(firing::Action::PickUp);
            let centre = game
                .combat
                .target()
                .rotor_centre(bevy_gym::robots::DroneMotor::RearRight);
            let aim = Dir3::new(centre - firing::Combat::origin(game.arena.position()))
                .expect("Rotor direction");
            game.act(firing::Action::Fire(aim));
            for _ in 0..13 {
                game.step(Movement::Idle);
            }
            game.act(firing::Action::Fire(aim));
        }
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(rotor),
            Some(&Visibility::Hidden)
        );
        assert_eq!(
            app.world().get::<Visibility>(unrelated),
            Some(&Visibility::Inherited)
        );
        let world = app.world_mut();
        for (kind, visible) in world.query::<(&weapons::Visual, &Visibility)>().iter(world) {
            assert_eq!(
                *visible,
                if *kind == weapons::Visual::Pickup {
                    Visibility::Hidden
                } else {
                    Visibility::Inherited
                }
            );
        }
        app.world_mut().resource_mut::<Game>().reset();
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(rotor),
            Some(&Visibility::Inherited)
        );
        let world = app.world_mut();
        for (kind, visible) in world.query::<(&weapons::Visual, &Visibility)>().iter(world) {
            assert_eq!(
                *visible,
                if *kind == weapons::Visual::Held {
                    Visibility::Hidden
                } else {
                    Visibility::Inherited
                }
            );
        }
    }
    #[test]
    fn aiming_keeps_the_pistol_grip_at_the_right_hand() {
        let mut app = app();
        app.world_mut()
            .resource_mut::<Game>()
            .act(firing::Action::PickUp);
        for aim in [Dir3::NEG_Z, Dir3::Y] {
            app.world_mut().resource_mut::<controls::Input>().aim = Some(aim);
            app.update();
            let world = app.world_mut();
            let character = *world
                .query_filtered::<&Transform, With<scene::Character>>()
                .single(world)
                .expect("Character");
            let shoulder = *world
                .query::<(&robot::Joint, &Transform)>()
                .iter(world)
                .find_map(|(joint, pose)| {
                    matches!(joint, robot::Joint::RightShoulder).then_some(pose)
                })
                .expect("Right shoulder");
            let gun = *world
                .query::<(&weapons::Visual, &Transform)>()
                .iter(world)
                .find_map(|(kind, pose)| (*kind == weapons::Visual::Held).then_some(pose))
                .expect("Held pistol");
            let hand =
                character.transform_point(shoulder.transform_point(Vec3::new(0.0, -0.5, 0.0)));
            let grip = gun.transform_point(Vec3::new(0.0, -0.09, 0.0));
            assert!(hand.distance(grip) < 1.0e-5);
        }
    }
    #[test]
    fn body_destruction_hides_the_target_and_reset_restores_it() {
        let mut app = app();
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.act(firing::Action::PickUp);
            let aim = Dir3::new(
                game.combat.target().position() - firing::Combat::origin(game.arena.position()),
            )
            .expect("Body direction");
            for _ in 0..6 {
                game.act(firing::Action::Fire(aim));
                for _ in 0..13 {
                    game.step(Movement::Idle);
                }
            }
            assert_eq!(game.combat.target().health().body_hits_remaining(), 0);
        }
        app.update();
        let world = app.world_mut();
        let drone = world
            .query::<(Entity, &weapons::Visual)>()
            .iter(world)
            .find_map(|(entity, kind)| (*kind == weapons::Visual::Drone).then_some(entity))
            .expect("Drone root");
        assert_eq!(world.get::<Visibility>(drone), Some(&Visibility::Hidden));
        world.resource_mut::<Game>().reset();
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(drone),
            Some(&Visibility::Inherited)
        );
    }
    #[test]
    fn a_damaged_rotor_remains_visible_before_its_second_hit() {
        let mut app = app();
        {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.act(firing::Action::PickUp);
            let centre = game
                .combat
                .target()
                .rotor_centre(bevy_gym::robots::DroneMotor::RearRight);
            let aim = Dir3::new(centre - firing::Combat::origin(game.arena.position()))
                .expect("Rotor direction");
            game.act(firing::Action::Fire(aim));
        }
        let rotor = app
            .world_mut()
            .spawn((
                Name::new("Rotor_BR"),
                Transform::default(),
                Visibility::Inherited,
            ))
            .id();
        app.update();
        assert_eq!(
            app.world()
                .resource::<Game>()
                .combat
                .target()
                .health()
                .rotor(bevy_gym::robots::DroneMotor::RearRight),
            combat::health::RotorHealth::Damaged
        );
        assert_eq!(
            app.world().get::<Visibility>(rotor),
            Some(&Visibility::Inherited)
        );
    }
}
