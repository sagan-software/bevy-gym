//! Explore a collision-aware arena as an articulated robot.
//!
//! Run `cargo run --features robots --example drone-pursuit`.
//! `pursuit-walk` shows the same movement loop without rendering.

#[path = "pursuit/arena.rs"]
mod arena;
#[path = "pursuit/controls.rs"]
mod controls;
#[path = "pursuit/robot.rs"]
mod robot;
#[path = "pursuit/scene.rs"]
mod scene;

use arena::{Arena, Movement};
use bevy::{asset::AssetMetaCheck, prelude::*};

/// One character's simulation and measured motion for visual animation.
#[derive(Resource, Default)]
struct Game {
    /// Shared movement and collision model.
    arena: Arena,
    /// Distance walked in metres since reset, used as animation phase.
    distance: f32,
    /// Last horizontal displacement, in metres per action.
    motion: Vec3,
    /// Last nonzero movement heading in radians; retained while idle.
    heading: f32,
}

impl Game {
    /// Reset simulation and animation together.
    const fn reset(&mut self) {
        self.arena.reset();
        self.distance = 0.0;
        self.motion = Vec3::ZERO;
        self.heading = 0.0;
    }

    /// Animate measured displacement, so holding a blocked direction does not walk in place.
    fn step(&mut self, movement: Movement) {
        let previous = self.arena.position();
        self.arena.step(movement);
        self.motion = (self.arena.position() - previous) * Vec3::new(1.0, 0.0, 1.0);
        let distance = self.motion.length();
        self.distance += distance;
        if distance > 0.001 {
            self.heading = (-self.motion.x).atan2(-self.motion.z);
        }
    }
}

/// Launch the player-controlled arena; drone pursuit is a later checkpoint.
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
        .add_systems(Startup, (scene::setup, controls::setup))
        .add_systems(PreUpdate, controls::read.after(bevy::input::InputSystems))
        .add_systems(FixedUpdate, advance)
        .add_systems(Update, (scene::project, robot::animate))
        .run();
}

/// Player buttons and future policy output enter the same fixed action boundary.
fn advance(mut game: ResMut<'_, Game>, input: Res<'_, controls::Input>) {
    game.step(input.movement);
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
            .init_resource::<Game>()
            .init_resource::<controls::Input>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Startup, (scene::setup, controls::setup))
            .add_systems(
                Update,
                (controls::read, advance, scene::project, robot::animate).chain(),
            );
        app.update();
        app
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
}
