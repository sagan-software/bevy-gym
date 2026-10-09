//! Two-bone arm posing keeps the authored hands on the shared weapon frame.

use super::Game;
use bevy::prelude::*;

/// Only the mannequin's two validated hand chains receive grip constraints.
#[derive(Component, Clone, Copy)]
pub(crate) enum Arm {
    /// Supporting hand and outward left elbow.
    Left,
    /// Trigger hand and outward right elbow.
    Right,
}

impl Arm {
    /// Select the source skeleton's hand joints without depending on entity order.
    pub(super) fn named(name: &str) -> Option<Self> {
        match name {
            "hand_l" => Some(Self::Left),
            "hand_r" => Some(Self::Right),
            _ => None,
        }
    }

    /// Wrist offsets and orientations from the bundled neutral pistol pose, in gun space.
    fn wrist(self) -> Transform {
        let (position, rotation) = match self {
            Self::Left => (
                Vec3::new(0.107_642_3, -0.080_043_17, 0.060_546_86),
                Quat::from_xyzw(-0.349_819_57, 0.173_445_18, 0.569_753_35, 0.723_135_2),
            ),
            Self::Right => (
                Vec3::new(0.072_226_66, -0.017_104_64, -0.008_047_65),
                Quat::from_xyzw(0.455_837_7, -0.561_770_56, -0.446_557_6, -0.526_509_7),
            ),
        };
        Transform::from_translation(position).with_rotation(rotation.normalize())
    }

    /// The elbow bends outward and down rather than through the chest.
    fn pole(self, game: &Game) -> Vec3 {
        let side = match self {
            Self::Left => -0.55,
            Self::Right => 0.55,
        };
        game.arena.position() + Quat::from_rotation_y(game.facing()) * Vec3::new(side, 0.18, -0.12)
    }
}

/// Run after animation and pelvis posing, before the weapon and global transforms.
pub(crate) fn pose(
    game: Res<'_, Game>,
    arms: Query<'_, '_, (Entity, &Arm)>,
    parents: Query<'_, '_, &ChildOf>,
    mut transforms: ParamSet<'_, '_, (TransformHelper<'_, '_>, Query<'_, '_, &mut Transform>)>,
) {
    if !game.robot_health.is_alive() || !game.combat.is_armed() {
        return;
    }
    let gun = game.weapon_pose();
    for (hand, arm) in &arms {
        // Incomplete loading or a removed bone leaves the authored pose available.
        let _solved = solve(hand, *arm, gun, arm.pole(&game), &parents, &mut transforms);
    }
}

/// Solve one fixed three-joint chain with no allocation or retained derived transforms.
fn solve(
    hand: Entity,
    arm: Arm,
    gun: Transform,
    pole: Vec3,
    parents: &Query<'_, '_, &ChildOf>,
    transforms: &mut ParamSet<'_, '_, (TransformHelper<'_, '_>, Query<'_, '_, &mut Transform>)>,
) -> Option<()> {
    let lower = parents.get(hand).ok()?.parent();
    let upper = parents.get(lower).ok()?.parent();
    let root = parents.get(upper).ok()?.parent();
    let [Ok(wrist), Ok(elbow), Ok(shoulder), Ok(parent)] =
        [hand, lower, upper, root].map(|entity| transforms.p0().compute_global_transform(entity))
    else {
        return None;
    };
    let desired = gun.mul_transform(arm.wrist());
    let bend = bend(
        shoulder.translation(),
        elbow.translation(),
        wrist.translation(),
        desired.translation,
        pole,
    )?;
    let turn = Quat::from_rotation_arc(
        (elbow.translation() - shoulder.translation()).normalize(),
        (bend.elbow - shoulder.translation()).normalize(),
    );
    transforms.p1().get_mut(upper).ok()?.rotation =
        parent.rotation().inverse() * turn * shoulder.rotation();
    let elbow = transforms.p0().compute_global_transform(lower).ok()?;
    let wrist = transforms.p0().compute_global_transform(hand).ok()?;
    let upper_pose = transforms.p0().compute_global_transform(upper).ok()?;
    let turn = Quat::from_rotation_arc(
        (wrist.translation() - elbow.translation()).normalize(),
        (bend.wrist - elbow.translation()).normalize(),
    );
    let lower_rotation = turn * elbow.rotation();
    transforms.p1().get_mut(lower).ok()?.rotation =
        upper_pose.rotation().inverse() * lower_rotation;
    transforms.p1().get_mut(hand).ok()?.rotation = lower_rotation.inverse() * desired.rotation;
    Some(())
}

/// Reachable elbow and wrist positions preserve both source bone lengths.
struct Bend {
    /// Elbow in the pole's half-plane.
    elbow: Vec3,
    /// Wrist at the target, or at the nearest nonsingular reachable distance.
    wrist: Vec3,
}

/// The cosine rule solves a bounded triangle; degenerate source bones retain animation.
fn bend(shoulder: Vec3, elbow: Vec3, wrist: Vec3, target: Vec3, pole: Vec3) -> Option<Bend> {
    let upper = shoulder.distance(elbow);
    let lower = elbow.distance(wrist);
    if !upper.is_finite() || !lower.is_finite() || upper < 0.0001 || lower < 0.0001 {
        return None;
    }
    let axis = Dir3::new(target - shoulder).ok()?;
    let reach = shoulder
        .distance(target)
        .clamp((upper - lower).abs() + 0.00001, upper + lower - 0.00001);
    let sideways = (pole - shoulder).reject_from_normalized(*axis);
    let sideways = sideways
        .try_normalize()
        .unwrap_or_else(|| axis.any_orthonormal_vector());
    // All lengths are metres; the squared-length ratio is dimensionless.
    let cosine = ((lower.mul_add(-lower, upper.mul_add(upper, reach * reach)))
        / (2.0 * upper * reach))
        .clamp(-1.0, 1.0);
    Some(Bend {
        elbow: shoulder
            + *axis * (upper * cosine)
            + sideways * (upper * (1.0 - cosine * cosine).sqrt()),
        wrist: shoulder + *axis * reach,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_wrists_follow_the_weapon_while_preserving_bone_lengths() {
        for aim in [
            Dir3::NEG_Z,
            Dir3::new(Vec3::new(0.15, 0.2, -1.0)).expect("Aim"),
        ] {
            let mut app = App::new();
            app.add_plugins((MinimalPlugins, TransformPlugin))
                .init_resource::<Game>()
                .add_systems(Update, pose);
            let game = &mut *app.world_mut().resource_mut::<Game>();
            game.act(crate::firing::Action::PickUp);
            game.aim = Some(aim);
            let origin = game.arena.position();
            for (arm, shoulder) in [
                (Arm::Left, Vec3::new(-0.16, 0.515, 0.043)),
                (Arm::Right, Vec3::new(0.214, 0.509, 0.144)),
            ] {
                let root = app
                    .world_mut()
                    .spawn(Transform::from_translation(origin))
                    .id();
                let upper = app
                    .world_mut()
                    .spawn((Transform::from_translation(shoulder), ChildOf(root)))
                    .id();
                let lower = app
                    .world_mut()
                    .spawn((Transform::from_xyz(0.0, 0.0, -0.28), ChildOf(upper)))
                    .id();
                app.world_mut()
                    .spawn((arm, Transform::from_xyz(0.0, 0.0, -0.28), ChildOf(lower)));
            }
            app.update();
            let gun = app.world().resource::<Game>().weapon_pose();
            let world = app.world_mut();
            for (arm, pose, local) in world
                .query::<(&Arm, &GlobalTransform, &Transform)>()
                .iter(world)
            {
                let desired = gun.mul_transform(arm.wrist());
                assert!(pose.translation().abs_diff_eq(desired.translation, 0.0001));
                // Opposite quaternion signs describe the same wrist orientation.
                assert!((pose.rotation().dot(desired.rotation).abs() - 1.0).abs() < 0.00001);
                assert!((local.translation.length() - 0.28).abs() < 0.00001);
            }
        }
    }

    #[test]
    fn unarmed_and_incomplete_chains_retain_the_authored_pose() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Game>()
            .add_systems(Update, pose);
        let original = Transform::from_xyz(0.0, 1.0, 0.0);
        let hand = app.world_mut().spawn((Arm::Right, original)).id();
        app.update();
        assert_eq!(*app.world().get::<Transform>(hand).expect("Hand"), original);
        app.world_mut()
            .resource_mut::<Game>()
            .act(crate::firing::Action::PickUp);
        app.update();
        assert_eq!(*app.world().get::<Transform>(hand).expect("Hand"), original);
        // A complete parent chain can still be awaiting its loaded transforms.
        let root = app.world_mut().spawn_empty().id();
        let upper = app
            .world_mut()
            .spawn((Transform::default(), ChildOf(root)))
            .id();
        let lower = app
            .world_mut()
            .spawn((Transform::default(), ChildOf(upper)))
            .id();
        app.world_mut().entity_mut(hand).insert(ChildOf(lower));
        app.update();
        assert_eq!(*app.world().get::<Transform>(hand).expect("Hand"), original);
        // Loaded but coincident arm joints must not introduce invalid rotations.
        app.world_mut()
            .entity_mut(root)
            .insert(Transform::default());
        app.update();
        assert_eq!(*app.world().get::<Transform>(hand).expect("Hand"), original);
        for _ in 0..1000 {
            let mut game = app.world_mut().resource_mut::<Game>();
            game.step(crate::Movement::Idle);
            if !game.robot_health.is_alive() {
                break;
            }
        }
        assert!(!app.world().resource::<Game>().robot_health.is_alive());
        app.update();
        assert_eq!(*app.world().get::<Transform>(hand).expect("Hand"), original);
        assert!(Arm::named("hand_l").is_some());
        assert!(Arm::named("hand_r").is_some());
        assert!(Arm::named("spine_01").is_none());
    }

    #[test]
    fn two_bone_solution_preserves_lengths_and_clamps_unreachable_targets() {
        for target in [Vec3::new(0.4, 0.2, 0.0), Vec3::X * 4.0, Vec3::X * 0.000001] {
            let result = bend(
                Vec3::ZERO,
                Vec3::X * 0.3,
                Vec3::X * 0.55,
                target,
                Vec3::NEG_Y,
            )
            .expect("Nondegenerate chain");
            assert!((result.elbow.length() - 0.3).abs() < 0.00001);
            assert!((result.wrist.distance(result.elbow) - 0.25).abs() < 0.00001);
            assert!(result.wrist.length() < 0.55);
            assert!(result.wrist.length() > 0.05);
        }
        assert!(bend(Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z).is_none());
        assert!(bend(Vec3::ZERO, Vec3::X, Vec3::X, Vec3::Y, Vec3::Z).is_none());
        assert!(bend(Vec3::ZERO, Vec3::NAN, Vec3::X, Vec3::Y, Vec3::Z).is_none());
        assert!(bend(Vec3::ZERO, Vec3::X, Vec3::NAN, Vec3::Y, Vec3::Z).is_none());
        assert!(bend(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ZERO, Vec3::Z).is_none());
        assert!(bend(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::NAN, Vec3::Z).is_none());
        assert!(bend(Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::X, Vec3::X).is_some());
    }
}
