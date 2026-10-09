//! Capture pointer aim and retain interactions until the fixed simulation consumes them.

use super::{
    controls::Input,
    firing::{Action, Combat},
    hud::ActionButton,
    shot::aim::Aim,
    Game,
};
use bevy::prelude::*;

/// UI clicks never fire into the world; pickup takes precedence over a simultaneous shot.
pub(super) fn read(
    keys: Res<'_, ButtonInput<KeyCode>>,
    mouse: Res<'_, ButtonInput<MouseButton>>,
    windows: Query<'_, '_, &Window>,
    cameras: Query<'_, '_, (&Camera, &Transform), With<Camera3d>>,
    buttons: Query<'_, '_, (Option<&ActionButton>, Ref<'_, Interaction>)>,
    mut input: ResMut<'_, Input>,
    game: Res<'_, Game>,
    view: Option<Res<'_, super::view::View>>,
) {
    if keys.pressed(KeyCode::KeyR) || keys.just_pressed(KeyCode::KeyR) {
        return;
    }
    let mut over_ui = false;
    let mut pickup = keys.just_pressed(KeyCode::KeyE);
    let mut fire = keys.pressed(KeyCode::Space) || keys.just_pressed(KeyCode::Space);
    for (button, interaction) in &buttons {
        over_ui |= *interaction != Interaction::None;
        if *interaction == Interaction::Pressed {
            match button {
                Some(ActionButton::PickUp) => pickup |= interaction.is_changed(),
                Some(ActionButton::Fire) => fire = true,
                None => {}
            }
        }
    }
    if !over_ui {
        if let (Ok(window), Ok((camera, transform))) = (windows.single(), cameras.single()) {
            let cursor = if view.is_some() {
                Some(window.size() * 0.5)
            } else {
                window.cursor_position()
            };
            if let Some(cursor) = cursor {
                input.aim = direction(&game, camera, transform, cursor);
            }
        }
        fire |= mouse.pressed(MouseButton::Left) || mouse.just_pressed(MouseButton::Left);
    }
    if pickup {
        input.action = Some(Action::PickUp);
    } else if fire && input.action.is_none() {
        if let Some(aim) = input.aim {
            input.action = Some(Action::Fire(aim));
        }
    }
}

/// Aim through the cursor's visible point, then fire from the actor rather than the camera.
fn direction(game: &Game, camera: &Camera, transform: &Transform, cursor: Vec2) -> Option<Dir3> {
    let ray = camera
        .viewport_to_world(&GlobalTransform::from(*transform), cursor)
        .ok()?;
    let aim = Aim::try_from((ray.origin, *ray.direction)).ok()?;
    let point = game.combat.target().aim_point(&game.arena, aim);
    Dir3::new(point - Combat::origin(game.arena.position())).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::camera::{ComputedCameraValues, RenderTargetInfo};

    /// A square perspective camera with the same projection convention as the viewer.
    fn camera() -> Camera {
        Camera {
            computed: ComputedCameraValues {
                clip_from_view: Mat4::perspective_infinite_reverse_rh(
                    std::f32::consts::FRAC_PI_3,
                    1.0,
                    0.1,
                ),
                target_info: Some(RenderTargetInfo {
                    physical_size: UVec2::splat(800),
                    scale_factor: 1.0,
                }),
                ..default()
            },
            ..default()
        }
    }

    #[test]
    fn cursor_aim_uses_visible_geometry_and_rejects_unavailable_camera_data() {
        let mut game = Game::default();
        let pose =
            Transform::from_xyz(0.0, 3.0, 5.0).looking_at(game.combat.target().position(), Vec3::Y);
        assert!(direction(&game, &Camera::default(), &pose, Vec2::splat(400.0)).is_none());
        assert!(direction(&game, &camera(), &pose, Vec2::NAN).is_none());
        let distant = Transform::from_xyz(2000.0, 3.0, 5.0);
        assert!(direction(&game, &camera(), &distant, Vec2::splat(400.0)).is_none());
        let aim = direction(&game, &camera(), &pose, Vec2::splat(400.0)).expect("Visible target");
        game.act(Action::PickUp);
        game.act(Action::Fire(aim));
        assert_eq!(game.combat.target().health().body_hits_remaining(), 5);
        let trace = game.combat.trace().expect("Accepted shot");
        assert_eq!(trace.from, Combat::origin(game.arena.position()));
    }

    #[test]
    fn pointer_input_retains_a_shot_and_prioritizes_pickup() {
        let mut app = App::new();
        app.init_resource::<Game>()
            .init_resource::<Input>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_systems(Update, read);
        let mut window = Window::default();
        window.set_cursor_position(Some(Vec2::splat(400.0)));
        let window = app.world_mut().spawn(window).id();
        let target = app.world().resource::<Game>().combat.target().position();
        app.world_mut().spawn((
            Camera3d::default(),
            camera(),
            Transform::from_xyz(0.0, 3.0, 5.0).looking_at(target, Vec3::Y),
        ));
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::KeyE);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .press(MouseButton::Left);
        app.update();
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.update();
        let action = app
            .world_mut()
            .resource_mut::<Input>()
            .action
            .take()
            .expect("Pickup queued");
        assert!(matches!(action, Action::PickUp));
        app.world_mut().resource_mut::<Game>().act(action);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .reset_all();
        app.update();
        let action = app
            .world_mut()
            .resource_mut::<Input>()
            .action
            .take()
            .expect("Shot queued");
        app.world_mut().resource_mut::<Game>().act(action);
        assert_eq!(
            app.world()
                .resource::<Game>()
                .combat
                .target()
                .health()
                .body_hits_remaining(),
            5
        );
        app.world_mut()
            .get_mut::<Window>(window)
            .expect("Window")
            .set_cursor_position(None);
        app.world_mut()
            .resource_mut::<ButtonInput<MouseButton>>()
            .reset_all();
        app.update();
        assert!(app.world().resource::<Input>().aim.is_some());
    }
    #[test]
    fn held_and_tapped_fire_require_a_captured_aim() {
        let mut app = App::new();
        app.init_resource::<Game>()
            .init_resource::<Input>()
            .init_resource::<ButtonInput<KeyCode>>()
            .init_resource::<ButtonInput<MouseButton>>()
            .add_systems(Update, read);
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .press(KeyCode::Space);
        app.update();
        assert!(app.world().resource::<Input>().action.is_none());
        app.world_mut()
            .resource_mut::<ButtonInput<KeyCode>>()
            .release(KeyCode::Space);
        app.world_mut().resource_mut::<Input>().aim = Some(Dir3::Y);
        app.update();
        assert!(matches!(
            app.world().resource::<Input>().action,
            Some(Action::Fire(Dir3::Y))
        ));
    }
}
