//! Shoulder camera, pointer capture, and aim-down-sights presentation.

use super::Game;
#[cfg(any(not(target_arch = "wasm32"), test))]
use bevy::window::CursorGrabMode;
use bevy::{input::mouse::AccumulatedMouseMotion, prelude::*, window::CursorOptions};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin, PanOrbitCameraSystemSet};

/// Hip fire and aimed fire choose bounded lens and boom settings.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
enum AimMode {
    /// Wider situational view.
    #[default]
    Hip,
    /// Closer shoulder view while holding right mouse.
    Sight,
}

/// Player-owned orbit angles; simulation reads only the horizontal movement basis.
#[derive(Resource)]
pub(super) struct View {
    /// Radians about world Y.
    pub(super) yaw: f32,
    /// Radians above the horizon, clamped away from the poles.
    pitch: f32,
    /// Current aim mode.
    mode: AimMode,
}

impl Default for View {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: -0.10,
            mode: AimMode::Hip,
        }
    }
}

impl View {
    /// A discharged pistol lifts aim by 0.7 degrees without changing the orbit limits.
    pub(super) fn recoil(&mut self) {
        self.pitch = (self.pitch + 0.7_f32.to_radians()).min(1.1);
    }

    /// Convert pointer pixels to radians without frame-rate scaling.
    fn turn(&mut self, delta: Vec2) {
        let sensitivity = if self.mode == AimMode::Sight {
            0.0015
        } else {
            0.0025
        };
        self.yaw = delta.x.mul_add(-sensitivity, self.yaw);
        self.pitch = delta.y.mul_add(-sensitivity, self.pitch).clamp(-1.1, 1.1);
    }

    /// World rotation shared by shoulder placement and movement.
    fn rotation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// Camera boom length in metres.
    const fn radius(&self) -> f32 {
        match self.mode {
            AimMode::Hip => 3.2,
            AimMode::Sight => 1.5,
        }
    }

    /// Vertical field of view in radians.
    const fn fov(&self) -> f32 {
        match self.mode {
            AimMode::Hip => 60.0_f32.to_radians(),
            AimMode::Sight => 40.0_f32.to_radians(),
        }
    }
}

/// Use `PanOrbit` 0.34 for Bevy 0.18; shooter input and collision wrap its orbit solver.
pub(super) fn install(app: &mut App) {
    app.insert_resource(StartingPistol)
        .init_resource::<View>()
        .add_plugins(PanOrbitCameraPlugin)
        .add_systems(Startup, (reticle, equip))
        .add_systems(
            PreUpdate,
            input
                .after(bevy::input::InputSystems)
                .before(super::controls::read),
        )
        .add_systems(PostUpdate, follow.before(PanOrbitCameraSystemSet))
        .add_systems(
            PostUpdate,
            (collision, super::aiming::read)
                .chain()
                .after(PanOrbitCameraSystemSet)
                .before(TransformSystems::Propagate),
        );
}

/// The shooter starts with its pistol; the separate pickup tutorial retains its own flow.
#[derive(Resource)]
pub(super) struct StartingPistol;

/// Equip through the same checked action used by the pickup tutorial.
fn equip(mut game: ResMut<'_, Game>) {
    game.act(super::firing::Action::PickUp);
}

/// A small centre reticle marks the camera ray used to select the weapon target.
fn reticle(mut commands: Commands<'_, '_>) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: percent(50.0),
            top: percent(50.0),
            width: px(4),
            height: px(4),
            margin: UiRect::all(px(-2)),
            ..default()
        },
        BackgroundColor(Color::WHITE),
        bevy::ui::FocusPolicy::Pass,
    ));
}

/// Input is disabled inside `PanOrbit` because shooter pointer capture owns orbiting.
pub(super) fn camera() -> PanOrbitCamera {
    PanOrbitCamera {
        yaw: Some(0.0),
        pitch: Some(0.10),
        radius: Some(3.2),
        enabled: false,
        orbit_smoothness: 0.0,
        pan_smoothness: 0.0,
        zoom_smoothness: 0.5,
        ..default()
    }
}

/// Click the scene to capture the pointer; Escape releases it for interface controls.
fn input(
    mut view: ResMut<'_, View>,
    motion: Res<'_, AccumulatedMouseMotion>,
    keys: Res<'_, ButtonInput<KeyCode>>,
    mouse: Res<'_, ButtonInput<MouseButton>>,
    mut windows: Query<'_, '_, (&Window, &mut CursorOptions)>,
    buttons: Query<'_, '_, &Interaction, With<Button>>,
) {
    let Ok((window, mut cursor)) = windows.single_mut() else {
        return;
    };
    if keys.just_pressed(KeyCode::Escape) || !window.focused {
        release(&mut cursor);
    } else if mouse.just_pressed(MouseButton::Left)
        && !buttons.iter().any(|state| *state != Interaction::None)
    {
        capture(&mut cursor);
    }
    view.mode = if mouse.pressed(MouseButton::Right) {
        AimMode::Sight
    } else {
        AimMode::Hip
    };
    if window.focused && (captured(&cursor) || mouse.pressed(MouseButton::Right)) {
        view.turn(motion.delta);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        *view = View::default();
    }
}

/// Desktop capture uses Bevy's window backend.
#[cfg(not(target_arch = "wasm32"))]
const fn capture(cursor: &mut CursorOptions) {
    cursor.grab_mode = CursorGrabMode::Locked;
    cursor.visible = false;
}

/// Desktop release returns the visible pointer to interface controls.
#[cfg(not(target_arch = "wasm32"))]
const fn release(cursor: &mut CursorOptions) {
    cursor.grab_mode = CursorGrabMode::None;
    cursor.visible = true;
}

/// Desktop rotation follows the window's capture state.
#[cfg(not(target_arch = "wasm32"))]
fn captured(cursor: &CursorOptions) -> bool {
    cursor.grab_mode == CursorGrabMode::Locked
}

/// Browser capture catches rejected promises in the page's camera adapter.
#[cfg(target_arch = "wasm32")]
fn capture(_: &mut CursorOptions) {
    browser_capture();
}

/// Browser release updates the browser's actual pointer lock.
#[cfg(target_arch = "wasm32")]
fn release(_: &mut CursorOptions) {
    browser_release();
}

/// Browser rotation follows actual capture, including Escape and denied requests.
#[cfg(target_arch = "wasm32")]
fn captured(_: &CursorOptions) -> bool {
    browser_captured()
}

/// Page adapter installed before the WASM module starts.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
extern "C" {
    /// Request capture with rejection handling.
    #[wasm_bindgen(js_namespace = globalThis, js_name = droneCameraCapture)]
    fn browser_capture();
    /// Release any current browser capture.
    #[wasm_bindgen(js_namespace = globalThis, js_name = droneCameraRelease)]
    fn browser_release();
    /// Read the browser's current captured element.
    #[wasm_bindgen(js_namespace = globalThis, js_name = droneCameraCaptured)]
    fn browser_captured() -> bool;
}

/// Follow the current capsule without smoothing its position behind the player.
pub(super) fn follow(
    game: Res<'_, Game>,
    view: Res<'_, View>,
    time: Res<'_, Time>,
    mut cameras: Query<'_, '_, (&mut PanOrbitCamera, &mut Projection)>,
) {
    for (mut camera, mut projection) in &mut cameras {
        camera.target_focus =
            game.arena.position() + Vec3::Y * 0.45 + view.rotation() * Vec3::X * 0.65;
        camera.target_yaw = view.yaw;
        camera.target_pitch = -view.pitch;
        camera.target_radius = view.radius();
        camera.force_update = true;
        if let Projection::Perspective(lens) = &mut *projection {
            lens.fov =
                (view.fov() - lens.fov).mul_add((time.delta_secs() * 16.0).min(1.0), lens.fov);
        }
    }
}

/// Retract the camera before the closest wall, including its near-plane clearance.
fn clear_boom(game: &Game, pivot: Vec3, desired: Vec3) -> Vec3 {
    game.arena
        .obstruction(pivot, desired)
        .map_or(desired, |distance| {
            pivot + (desired - pivot).normalize_or_zero() * (distance - 0.25).max(0.0)
        })
}

/// Collision changes only the rendered eye, preserving the requested orbit radius.
pub(super) fn collision(
    game: Res<'_, Game>,
    mut cameras: Query<'_, '_, &mut Transform, With<PanOrbitCamera>>,
) {
    let pivot = game.arena.position() + Vec3::Y * 0.45;
    for mut camera in &mut cameras {
        camera.translation = clear_boom(&game, pivot, camera.translation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recoil_lifts_aim_and_preserves_the_pitch_limit() {
        let mut view = View::default();
        let before = view.pitch;
        view.recoil();
        assert!((view.pitch - before - 0.7_f32.to_radians()).abs() < 0.00001);
        view.pitch = 1.1;
        view.recoil();
        assert_eq!(view.pitch, 1.1);
        assert_eq!(view.yaw, 0.0);
    }

    #[test]
    fn installed_camera_creates_reticle_and_preserves_orthographic_lens() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, bevy::input::InputPlugin))
            .init_resource::<Game>()
            .init_resource::<super::super::controls::Input>()
            .add_message::<bevy::input::gestures::PinchGesture>();
        install(&mut app);
        let eye = app
            .world_mut()
            .spawn((
                Camera3d::default(),
                camera(),
                Projection::Orthographic(OrthographicProjection::default_3d()),
            ))
            .id();
        app.update();
        assert!(app.world().resource::<Game>().combat.is_armed());
        assert!(matches!(
            app.world().get::<Projection>(eye),
            Some(Projection::Orthographic(_))
        ));
        let world = app.world_mut();
        assert_eq!(world.query::<&Node>().iter(world).count(), 1);
    }

    #[test]
    fn clicking_interface_does_not_capture_the_pointer() {
        let mut app = App::new();
        app.init_resource::<View>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_systems(Update, input);
        let window = app
            .world_mut()
            .spawn((
                Window {
                    focused: true,
                    ..default()
                },
                CursorOptions::default(),
            ))
            .id();
        app.world_mut().spawn((Button, Interaction::Pressed));
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        assert_eq!(
            app.world()
                .get::<CursorOptions>(window)
                .expect("Cursor")
                .grab_mode,
            CursorGrabMode::None
        );
    }

    #[test]
    fn shooter_starts_armed_and_reset_restores_its_loadout() {
        let mut app = App::new();
        app.init_resource::<Game>()
            .init_resource::<super::super::controls::Input>()
            .init_resource::<ButtonInput<KeyCode>>()
            .insert_resource(StartingPistol)
            .add_systems(Startup, equip)
            .add_systems(Update, super::super::controls::read);
        app.update();
        assert!(app.world().resource::<Game>().combat.is_armed());
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyR);
        app.update();
        assert!(app.world().resource::<Game>().combat.is_armed());
        assert_eq!(app.world().resource::<Game>().combat.rounds(), 12);
    }

    #[test]
    fn pointer_capture_ads_and_escape_have_distinct_states() {
        let mut app = App::new();
        app.insert_resource(StartingPistol)
            .init_resource::<View>()
            .init_resource::<AccumulatedMouseMotion>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_systems(Update, input);
        let window = app
            .world_mut()
            .spawn((
                Window {
                    focused: true,
                    ..default()
                },
                CursorOptions::default(),
            ))
            .id();
        app.world_mut()
            .resource_mut::<AccumulatedMouseMotion>()
            .delta = Vec2::splat(100.0);
        app.update();
        assert_eq!(app.world().resource::<View>().yaw, 0.0);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        assert_eq!(
            app.world()
                .get::<CursorOptions>(window)
                .expect("Cursor")
                .grab_mode,
            CursorGrabMode::Locked
        );
        assert!(app.world().resource::<View>().yaw < 0.0);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Right);
        app.update();
        assert!(app.world().resource::<View>().mode == AimMode::Sight);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Escape);
        app.update();
        assert_eq!(
            app.world()
                .get::<CursorOptions>(window)
                .expect("Cursor")
                .grab_mode,
            CursorGrabMode::None
        );
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .reset_all();
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Right);
        let previous = app.world().resource::<View>().yaw;
        app.update();
        assert!(app.world().resource::<View>().yaw < previous);
        assert_eq!(
            app.world()
                .get::<CursorOptions>(window)
                .expect("Cursor")
                .grab_mode,
            CursorGrabMode::None
        );
        app.world_mut()
            .get_mut::<Window>(window)
            .expect("Window")
            .focused = false;
        let previous = app.world().resource::<View>().yaw;
        app.update();
        assert_eq!(app.world().resource::<View>().yaw, previous);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyR);
        app.update();
        assert_eq!(app.world().resource::<View>().yaw, 0.0);
    }

    #[test]
    fn camera_orbits_clamps_pitch_and_ads_changes_lens_and_distance() {
        let mut view = View::default();
        let initial = view.rotation();
        view.turn(Vec2::new(200.0, 100_000.0));
        assert_ne!(view.rotation(), initial);
        assert!((view.pitch + 1.1).abs() < 1e-5);
        view.turn(Vec2::new(0.0, -200_000.0));
        assert!((view.pitch - 1.1).abs() < 1e-5);
        let hip = (view.radius(), view.fov());
        view.mode = AimMode::Sight;
        assert!(view.radius() < hip.0);
        assert!(view.fov() < hip.1);
    }

    #[test]
    fn boom_cannot_cross_cover_or_a_ceiling() {
        let game = Game::default();
        for position in [Vec3::new(0.0, 0.92, 9.0), Vec3::new(9.0, 0.92, 0.0)] {
            for yaw in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] {
                let view = View { yaw, ..default() };
                let pivot = position + Vec3::Y * 0.45;
                let desired = pivot + view.rotation() * Vec3::new(0.65, 0.0, view.radius());
                let eye = clear_boom(&game, pivot, desired);
                assert!(game.arena.obstruction(pivot, eye).is_none());
            }
        }
    }
}
