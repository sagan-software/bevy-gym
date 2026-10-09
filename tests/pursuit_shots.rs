//! Aimed pistol shots use the same geometry as walking and camera obstruction.
#![cfg(feature = "robots")]

#[expect(
    dead_code,
    reason = "Shot tests only query the shared arena; movement is tested separately."
)]
#[path = "../examples/robots/pursuit/arena.rs"]
mod arena;
#[expect(
    dead_code,
    reason = "Damage and pickup boundary cases are tested in pursuit_combat."
)]
#[path = "../examples/robots/pursuit/combat.rs"]
mod combat;
#[path = "../examples/robots/pursuit/shot.rs"]
mod shot;

use arena::Arena;
use bevy::math::{InvalidDirectionError, Quat, Vec3};
use bevy_gym::robots::DroneMotor;
use combat::{
    health::{Damage, RotorHealth},
    pistol::{FireError, Pistol},
};
use shot::{
    aim::{Aim, InvalidAim},
    target::{InvalidTarget, Part, Shot, Target},
};
use std::time::Duration;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn flight_pose_updates_preserve_damage_and_reject_invalid_state_atomically() {
    let arena = Arena::default();
    let mut pistol = Pistol::default();
    pistol.pick_up(Pistol::LOCATION).expect("Pickup");
    let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).expect("Target");
    let aim = Aim::try_from((Vec3::new(0.0, 4.0, 0.0), Vec3::NEG_Y)).expect("Aim");
    shoot(&mut target, &arena, &mut pistol, aim).expect("Body hit");
    let health = target.health().clone();
    target
        .move_to(Vec3::new(1.0, 3.0, 0.0), Quat::from_rotation_y(0.4))
        .expect("Flight pose");
    assert_eq!(target.health(), &health);
    let position = target.position();
    let rotation = target.rotation();
    for (centre, orientation, error) in [
        (
            Vec3::splat(f32::NAN),
            Quat::IDENTITY,
            InvalidTarget::Position,
        ),
        (
            Vec3::ZERO,
            Quat::from_array([0.0; 4]),
            InvalidTarget::Rotation,
        ),
    ] {
        assert_eq!(target.move_to(centre, orientation), Err(error));
        assert_eq!(target.position(), position);
        assert_eq!(target.rotation(), rotation);
        assert_eq!(target.health(), &health);
    }
    assert_eq!(target.crash(), Damage::Destroyed);
    assert_eq!(target.crash(), Damage::Ignored);
    assert!(!target.health().is_alive());
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// Construct a loaded pistol at the authored pickup without changing its rules.
fn pistol() -> Pistol {
    let mut pistol = Pistol::default();
    pistol.pick_up(Pistol::LOCATION).expect("At the pickup");
    pistol
}

/// Aim validation rejects malformed input and stores a normalized direction.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn aim_rejects_invalid_coordinates_and_normalizes_direction() {
    for origin in [Vec3::NAN, Vec3::INFINITY, Vec3::splat(1_000.01)] {
        assert_eq!(Aim::try_from((origin, Vec3::Z)), Err(InvalidAim::Origin));
    }
    for (direction, error) in [
        (Vec3::ZERO, InvalidDirectionError::Zero),
        (Vec3::NAN, InvalidDirectionError::NaN),
        (Vec3::INFINITY, InvalidDirectionError::Infinite),
    ] {
        assert_eq!(
            Aim::try_from((Vec3::ZERO, direction)),
            Err(InvalidAim::Direction(error))
        );
    }
    let aim = Aim::try_from((Vec3::Y, Vec3::Z * 4.0)).unwrap();
    assert_eq!(aim.origin(), Vec3::Y);
    assert_eq!(*aim.direction(), Vec3::Z);
}

/// Target poses reject invalid coordinates and rotations before storing state.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn target_pose_is_validated_and_normalized() {
    assert_eq!(
        Target::try_from((Vec3::NAN, Quat::IDENTITY)).unwrap_err(),
        InvalidTarget::Position
    );
    for rotation in [
        Quat::from_xyzw(0.0, 0.0, 0.0, 0.0),
        Quat::NAN,
        Quat::from_xyzw(f32::INFINITY, 0.0, 0.0, 0.0),
    ] {
        assert_eq!(
            Target::try_from((Vec3::ZERO, rotation)).unwrap_err(),
            InvalidTarget::Rotation
        );
    }
    let target = Target::try_from((Vec3::Y, Quat::from_xyzw(0.0, 0.0, 0.0, 2.0))).unwrap();
    assert_eq!(target.position(), Vec3::Y);
    assert_eq!(target.rotation(), Quat::IDENTITY);
    assert_eq!(target.health().body_hits_remaining(), 6);
}

/// Each transformed rotor has a separate two-hit weak point and disappears afterward.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn each_rotor_can_be_aimed_at_independently() {
    let arena = Arena::default();
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_y(0.8),
        Quat::from_rotation_x(0.6) * Quat::from_rotation_z(0.4),
    ] {
        for motor in DroneMotor::ALL {
            let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), rotation)).unwrap();
            let centre = target.rotor_centre(motor);
            let up = rotation * Vec3::Y;
            let aim = Aim::try_from((centre + up, -up)).unwrap();
            let mut pistol = pistol();
            assert!(
                matches!(shoot(&mut target, &arena, &mut pistol, aim), Ok(Shot::Hit { part: Part::Rotor(hit), damage: Damage::Hit, .. }) if hit == motor)
            );
            pistol.advance(Duration::from_secs(1));
            assert!(
                matches!(shoot(&mut target, &arena, &mut pistol, aim), Ok(Shot::Hit { damage: Damage::RotorDestroyed(hit), .. }) if hit == motor)
            );
            pistol.advance(Duration::from_secs(1));
            assert!(matches!(
                shoot(&mut target, &arena, &mut pistol, aim),
                Ok(Shot::Wall { .. })
            ));
            assert_eq!(target.health().body_hits_remaining(), 6);
        }
    }
}

/// The body receives the nearest hit and disappears after its sixth accepted shot.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn body_dies_once_and_rejected_shots_preserve_health() {
    let arena = Arena::default();
    let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).unwrap();
    let aim = Aim::try_from((Vec3::new(0.0, 3.0, 2.0), Vec3::NEG_Z)).unwrap();
    let mut unarmed = Pistol::default();
    assert_eq!(
        shoot(&mut target, &arena, &mut unarmed, aim),
        Err(FireError::Unarmed)
    );
    let mut pistol = pistol();
    for hit in 1..=6 {
        let shot = shoot(&mut target, &arena, &mut pistol, aim).unwrap();
        let expected = if hit == 6 {
            Damage::Destroyed
        } else {
            Damage::Hit
        };
        assert!(
            matches!(shot, Shot::Hit { part: Part::Body, damage, point } if damage == expected && (point.z - 0.18).abs() < 1.0e-5)
        );
        assert_eq!(
            shoot(&mut target, &arena, &mut pistol, aim),
            Err(FireError::CoolingDown)
        );
        pistol.advance(Duration::from_secs(1));
    }
    assert!(matches!(
        shoot(&mut target, &arena, &mut pistol, aim),
        Ok(Shot::Miss { .. })
    ));
    assert_eq!(target.health().body_hits_remaining(), 0);
}

/// Windows and the pipe transmit shots; the house wall blocks damage and consumes a round.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn authored_geometry_blocks_only_closed_surfaces() {
    let arena = Arena::default();
    for (origin, position, clear) in [
        (Vec3::new(-5.0, 1.8, 2.0), Vec3::new(-5.0, 1.8, -3.0), true),
        (Vec3::new(-7.0, 1.8, 2.0), Vec3::new(-7.0, 1.8, -3.0), false),
        (Vec3::new(5.0, 1.7, 6.0), Vec3::new(5.0, 1.7, -1.0), true),
    ] {
        let mut target = Target::try_from((position, Quat::IDENTITY)).unwrap();
        let mut pistol = pistol();
        let aim = Aim::try_from((origin, position - origin)).unwrap();
        let shot = shoot(&mut target, &arena, &mut pistol, aim).unwrap();
        assert_eq!(
            matches!(
                shot,
                Shot::Hit {
                    part: Part::Body,
                    ..
                }
            ),
            clear
        );
        assert_eq!(
            target.health().body_hits_remaining(),
            if clear { 5 } else { 6 }
        );
        assert_eq!(pistol.rounds(), 11);
    }
}

/// Range is inclusive; misses consume a round and end exactly thirty metres away.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn range_and_inside_shape_queries_have_defined_results() {
    let arena = Arena::default();
    for (height, hit) in [(31.104, true), (31.105, false)] {
        let mut target = Target::try_from((Vec3::new(0.0, height, 0.0), Quat::IDENTITY)).unwrap();
        let mut pistol = pistol();
        let aim = Aim::try_from((Vec3::Y, Vec3::Y)).unwrap();
        let shot = shoot(&mut target, &arena, &mut pistol, aim).unwrap();
        if hit {
            assert!(matches!(shot, Shot::Hit { point, .. } if (point.y - 31.0).abs() < 1.0e-5));
        } else {
            assert_eq!(
                shot,
                Shot::Miss {
                    point: Vec3::new(0.0, 31.0, 0.0)
                }
            );
        }
    }
    let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).unwrap();
    let mut pistol = pistol();
    let origin = target.position();
    let aim = Aim::try_from((origin, Vec3::Y)).unwrap();
    assert!(
        matches!(shoot(&mut target, &arena,&mut pistol,aim),Ok(Shot::Hit { point, .. }) if point == origin)
    );
}

/// A shot starting in both a wall and body must not bypass static occlusion.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn static_geometry_wins_zero_distance_ties() {
    let arena = Arena::default();
    let origin = Vec3::new(-7.0, 1.8, 0.0);
    let mut target = Target::try_from((origin, Quat::IDENTITY)).unwrap();
    let mut pistol = pistol();
    let aim = Aim::try_from((origin, Vec3::Z)).unwrap();
    assert_eq!(
        shoot(&mut target, &arena, &mut pistol, aim),
        Ok(Shot::Wall { point: origin })
    );
    assert_eq!(target.health().body_hits_remaining(), 6);
}

/// A nearer rear rotor shields the front rotor until its hitbox is destroyed.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn nearest_rotor_shields_the_farther_rotor() {
    let arena = Arena::default();
    let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).unwrap();
    let rear = target.rotor_centre(DroneMotor::RearRight);
    let aim = Aim::try_from((rear + Vec3::Z * 2.0, Vec3::NEG_Z)).unwrap();
    let mut pistol = pistol();
    for expected in [
        DroneMotor::RearRight,
        DroneMotor::RearRight,
        DroneMotor::FrontRight,
    ] {
        assert!(
            matches!(shoot(&mut target, &arena,&mut pistol,aim),Ok(Shot::Hit { part:Part::Rotor(actual), .. }) if actual == expected)
        );
        pistol.advance(Duration::from_secs(1));
    }
}

/// Position bounds include both endpoints and reject either axis overflow.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn position_bounds_apply_to_aim_and_target_construction() {
    for position in [Vec3::splat(-1_000.0), Vec3::splat(1_000.0)] {
        assert_eq!(
            Aim::try_from((position, Vec3::Z)).unwrap().origin(),
            position
        );
        assert_eq!(
            Target::try_from((position, Quat::IDENTITY))
                .unwrap()
                .position(),
            position
        );
    }
    for position in [
        Vec3::splat(-1_000.01),
        Vec3::splat(1_000.01),
        Vec3::INFINITY,
    ] {
        assert_eq!(Aim::try_from((position, Vec3::Z)), Err(InvalidAim::Origin));
        assert_eq!(
            Target::try_from((position, Quat::IDENTITY)).unwrap_err(),
            InvalidTarget::Position
        );
    }
}

/// Finite nonzero directions and rotations normalize without squared-length overflow or underflow.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn extreme_finite_magnitudes_preserve_unit_direction_and_rotation() {
    for magnitude in [f32::from_bits(1), 1.0e-22, f32::MAX] {
        let aim = Aim::try_from((Vec3::Y, Vec3::Z * magnitude)).unwrap();
        assert!((*aim.direction() - Vec3::Z).length() < 1.0e-6);
        let rotation = Quat::from_xyzw(0.0, 0.0, 0.0, magnitude);
        let target = Target::try_from((Vec3::Y, rotation)).unwrap();
        assert!(target.rotation().is_normalized());
        assert_eq!(target.rotation(), Quat::IDENTITY);
    }
}

/// A ray aimed at a rear rotor damages the closer body instead of passing through it.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn body_shields_a_rotor_behind_it() {
    let arena = Arena::default();
    let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).unwrap();
    let origin = Vec3::new(-0.4, 3.0875, -2.0);
    let aim = Aim::try_from((origin, target.rotor_centre(DroneMotor::RearRight) - origin)).unwrap();
    let mut pistol = pistol();
    assert!(matches!(
        shoot(&mut target, &arena, &mut pistol, aim),
        Ok(Shot::Hit {
            part: Part::Body,
            ..
        })
    ));
    assert_eq!(target.health().body_hits_remaining(), 5);
    assert_eq!(
        target.health().rotor(DroneMotor::RearRight),
        RotorHealth::Intact
    );
}

/// The pointer queries visible geometry without consuming ammunition or changing health.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn visible_aim_points_respect_target_wall_and_range() {
    let arena = Arena::default();
    let target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY)).unwrap();
    let aim = Aim::try_from((Vec3::new(0.0, 3.0, 2.0), Vec3::NEG_Z)).unwrap();
    assert!((target.aim_point(&arena, aim).z - 0.18).abs() < 1.0e-5);
    let aim = Aim::try_from((Vec3::new(0.0, 4.0, 2.0), Vec3::Y)).unwrap();
    assert_eq!(target.aim_point(&arena, aim), Vec3::new(0.0, 34.0, 2.0));
    let hidden = Target::try_from((Vec3::new(-7.0, 1.8, -3.0), Quat::IDENTITY)).unwrap();
    let aim = Aim::try_from((Vec3::new(-7.0, 1.8, 2.0), Vec3::NEG_Z)).unwrap();
    assert!(hidden.aim_point(&arena, aim).z > 0.0);
    assert_eq!(target.health().body_hits_remaining(), 6);
    assert_eq!(hidden.health().body_hits_remaining(), 6);
}

/// Keep the original geometry cases independent of projectile scheduling.
fn shoot(
    target: &mut Target,
    arena: &Arena,
    pistol: &mut Pistol,
    aim: Aim,
) -> Result<Shot, FireError> {
    pistol.fire()?;
    Ok(target
        .sweep(arena, aim, Aim::RANGE)
        .unwrap_or_else(|| Shot::Miss {
            point: aim.point(Aim::RANGE),
        }))
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn a_swept_segment_hits_only_within_its_travelled_distance() {
    let arena = Arena::default();
    let mut target = Target::try_from((Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY))
        .expect("Target above the floor");
    let aim = Aim::try_from((Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y)).expect("Downward aim");
    assert!(target.sweep(&arena, aim, 1.8).is_none());
    assert_eq!(target.health().body_hits_remaining(), 6);
    assert!(matches!(
        target.sweep(&arena, aim, 2.0),
        Some(Shot::Hit {
            part: Part::Body,
            ..
        })
    ));
    assert_eq!(target.health().body_hits_remaining(), 5);
}
