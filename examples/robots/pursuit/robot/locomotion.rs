//! Directional pelvis orientation and authored-stride playback speed.

use super::Game;
use bevy::prelude::*;

/// A forward or reverse stride plus a pelvis turn bounded to ninety degrees.
pub(super) struct Stride {
    /// Radians of pelvis yaw relative to the aiming torso.
    pub(super) yaw: f32,
    /// Signed playback multiplier of the authored forward jog.
    pub(super) speed: f32,
}

impl Stride {
    /// Convert displacement in metres per 20 ms step to authored clip playback.
    pub(super) fn new(motion: Vec3, facing: f32) -> Self {
        if motion.length_squared() < 1e-6 {
            return Self {
                yaw: 0.0,
                speed: 0.0,
            };
        }
        let difference = (-motion.x).atan2(-motion.z) - facing;
        let mut yaw = difference.sin().atan2(difference.cos());
        let direction = if yaw.abs() > std::f32::consts::FRAC_PI_2 + 1e-5 {
            yaw = yaw.signum().mul_add(-std::f32::consts::PI, yaw);
            -1.0
        } else {
            1.0
        };
        // Source root motion travels 5 metres over 28 frames at 30 frames per second.
        let authored_speed = 5.0 / (28.0 / 30.0);
        Self {
            yaw,
            speed: direction * motion.length() / 0.02 / authored_speed,
        }
    }
}

/// Pelvis turns toward travel; the first spine bone counter-turns the weapon torso.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bone {
    /// Root of the leg and torso hierarchy.
    Pelvis,
    /// Root of the upper-body aim mask.
    Spine,
}

/// Presentation-only turn history prevents abrupt pelvis reversals.
#[derive(Default)]
pub(crate) struct Turn {
    /// Current pelvis yaw in radians, relative to the aiming torso.
    yaw: f32,
}

impl Turn {
    /// Exponential settling uses a 100 ms time constant and is frame-rate independent.
    fn advance(&mut self, target: f32, elapsed: std::time::Duration) -> f32 {
        let blend = 1.0 - (-elapsed.as_secs_f32() / 0.1).exp();
        self.yaw = (target - self.yaw).mul_add(blend, self.yaw);
        self.yaw
    }
}

/// Apply world-up rotation in each bone parent's basis after animation evaluation.
pub(crate) fn pose(
    game: Res<'_, Game>,
    time: Res<'_, Time>,
    mut turn: Local<'_, Turn>,
    bones: Query<'_, '_, (Entity, &Bone, &ChildOf)>,
    mut transforms: ParamSet<'_, '_, (TransformHelper<'_, '_>, Query<'_, '_, &mut Transform>)>,
) {
    if !game.robot_health.is_alive() {
        return;
    }
    let stride = Stride::new(game.motion, game.facing());
    let yaw = turn.advance(stride.yaw, time.delta());
    for (kind, angle) in [(Bone::Pelvis, yaw), (Bone::Spine, -yaw)] {
        for (entity, bone, parent) in &bones {
            if *bone != kind {
                continue;
            }
            let Ok(parent_pose) = transforms.p0().compute_global_transform(parent.parent()) else {
                continue;
            };
            let axis = parent_pose.rotation().inverse() * Vec3::Y;
            if let Ok(mut local) = transforms.p1().get_mut(entity) {
                local.rotation = Quat::from_axis_angle(axis, angle) * local.rotation;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_parent_and_disabled_robot_leave_bones_unchanged() {
        let mut app = App::new();
        app.init_resource::<Game>()
            .init_resource::<Time>()
            .add_systems(Update, pose);
        let valid_parent = app.world_mut().spawn(Transform::default()).id();
        let incomplete = app
            .world_mut()
            .spawn((Bone::Spine, ChildOf(valid_parent)))
            .id();
        let parent = app.world_mut().spawn_empty().id();
        let bone = app
            .world_mut()
            .spawn((Bone::Pelvis, Transform::default(), ChildOf(parent)))
            .id();
        app.update();
        assert!(app.world().get::<Transform>(incomplete).is_none());
        assert_eq!(
            *app.world().get::<Transform>(bone).expect("Bone"),
            Transform::default()
        );
        for _ in 0..1000 {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.step(super::super::super::Movement::Idle);
            if !game.robot_health.is_alive() {
                break;
            }
        }
        assert!(!app.world().resource::<Game>().robot_health.is_alive());
        app.update();
        assert_eq!(
            *app.world().get::<Transform>(bone).expect("Bone"),
            Transform::default()
        );
    }

    #[test]
    fn pelvis_turns_in_world_space_while_spine_keeps_its_authored_orientation() {
        let mut app = App::new();
        let mut game = Game::default();
        game.act(super::super::super::firing::Action::PickUp);
        game.aim = Some(Dir3::NEG_Z);
        game.motion = Vec3::NEG_X * 0.08;
        let mut time = Time::<()>::default();
        time.advance_by(std::time::Duration::from_millis(100));
        app.insert_resource(game)
            .insert_resource(time)
            .add_systems(Update, pose);
        let basis = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        let root = app.world_mut().spawn(Transform::from_rotation(basis)).id();
        let pelvis = app
            .world_mut()
            .spawn((Bone::Pelvis, Transform::default(), ChildOf(root)))
            .id();
        let spine = app
            .world_mut()
            .spawn((Bone::Spine, Transform::default(), ChildOf(pelvis)))
            .id();
        app.update();
        let pelvis_pose = app
            .world()
            .get::<Transform>(pelvis)
            .expect("Pelvis")
            .rotation;
        let spine_pose = app.world().get::<Transform>(spine).expect("Spine").rotation;
        assert!((basis * pelvis_pose).abs_diff_eq(
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2 * (1.0 - (-1.0_f32).exp())) * basis,
            1e-5
        ));
        assert!((basis * pelvis_pose * spine_pose).abs_diff_eq(basis, 1e-5));
    }

    #[test]
    fn direction_reversal_is_smooth_and_frame_rate_independent() {
        let mut whole = Turn::default();
        let mut split = Turn::default();
        let target = std::f32::consts::FRAC_PI_2;
        let angle = whole.advance(target, std::time::Duration::from_millis(100));
        assert!(angle > 0.0 && angle < target);
        for _ in 0..10 {
            split.advance(target, std::time::Duration::from_millis(10));
        }
        assert!((whole.yaw - split.yaw).abs() < 0.00001);
        let reversed = whole.advance(-target, std::time::Duration::from_millis(16));
        assert!(reversed > 0.0 && reversed < angle);
        assert!((whole.advance(-target, std::time::Duration::ZERO) - reversed).abs() < 0.00001);
    }

    #[test]
    fn stride_matches_forward_backward_and_strafing_motion() {
        for (motion, yaw, sign) in [
            (Vec3::NEG_Z, 0.0, 1.0),
            (Vec3::Z, 0.0, -1.0),
            (Vec3::NEG_X, std::f32::consts::FRAC_PI_2, 1.0),
            (Vec3::X, -std::f32::consts::FRAC_PI_2, 1.0),
        ] {
            let stride = Stride::new(motion * 0.08, 0.0);
            assert!((stride.yaw - yaw).abs() < 1e-5);
            assert!((stride.speed - sign * 4.0 / (5.0 / (28.0 / 30.0))).abs() < 1e-5);
        }
        let idle = Stride::new(Vec3::ZERO, 2.0);
        assert_eq!(idle.yaw, 0.0);
        assert_eq!(idle.speed, 0.0);
    }
}
