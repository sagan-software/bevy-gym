//! Collision queries through the shared world's public read-only boundary.
#![cfg(feature = "robots")]
use bevy::math::Vec3;
use bevy_gym::robots::{
    DroidBody, InvalidRobotRay, RobotActions, RobotId, RobotRay, RobotRayTarget, RobotSlot,
    RobotWorld,
};
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
/// Queries must see the actual floor before any action or physics step.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ray_queries_read_current_colliders_without_advancing_physics() {
    let world = RobotWorld::default();
    let before = world.snapshot().unwrap();
    let ray = RobotRay::try_from((Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y, 2.0)).unwrap();
    let hit = world.cast_ray(&ray, None).unwrap().unwrap();
    assert_eq!(hit.target(), RobotRayTarget::Environment);
    assert!((hit.metres() - 2.0).abs() < 1e-6);
    assert_eq!(world.snapshot().unwrap().frame(), before.frame());
}

/// Every physical robot segment is classified and excluded by its owning identity.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rays_identify_all_six_agents_and_exclude_all_droid_segments() {
    let world = RobotWorld::default();
    let snapshot = world.snapshot().unwrap();
    for slot in RobotSlot::ALL {
        let ray = RobotRay::try_from((snapshot.drone(slot).position(), Vec3::Y, 0.001)).unwrap();
        let hit = world.cast_ray(&ray, None).unwrap().unwrap();
        assert_eq!(hit.target(), RobotRayTarget::Drone(slot));
        assert_eq!(hit.metres().to_bits(), 0.0_f32.to_bits());
        assert!(world
            .cast_ray(&ray, Some(RobotId::Drone(slot)))
            .unwrap()
            .is_none());
        for body in DroidBody::ALL {
            let ray =
                RobotRay::try_from((snapshot.droid(slot).body(body).position(), Vec3::Y, 0.001))
                    .unwrap();
            let hit = world.cast_ray(&ray, None).unwrap().unwrap();
            assert_eq!(hit.target(), RobotRayTarget::Droid(slot, body));
            assert_eq!(hit.metres().to_bits(), 0.0_f32.to_bits());
            assert!(world
                .cast_ray(&ray, Some(RobotId::Droid(slot)))
                .unwrap()
                .is_none());
        }
    }
    let ray =
        RobotRay::try_from((snapshot.drone(RobotSlot::First).position(), Vec3::Z, 4.0)).unwrap();
    let hit = world
        .cast_ray(&ray, Some(RobotId::Drone(RobotSlot::First)))
        .unwrap()
        .unwrap();
    assert_eq!(hit.target(), RobotRayTarget::Drone(RobotSlot::Second));
    assert!(hit.metres() > 0.0 && hit.metres() < 2.0);
}
/// Range, solid interiors and normalization preserve physical distance semantics.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn rays_preserve_inclusive_ranges_and_normalize_extreme_directions() {
    let world = RobotWorld::default();
    let origin = Vec3::new(0.0, 2.0, 0.0);
    for magnitude in [1.0, 10.0, f32::MAX, f32::from_bits(1)] {
        let ray = RobotRay::try_from((origin, Vec3::NEG_Y * magnitude, 2.0)).unwrap();
        let hit = world.cast_ray(&ray, None).unwrap().unwrap();
        assert!((hit.metres() - 2.0).abs() < 1e-6);
        let ray = RobotRay::try_from((origin, Vec3::NEG_Y, hit.metres())).unwrap();
        assert!(world.cast_ray(&ray, None).unwrap().is_some());
    }
    let short = RobotRay::try_from((origin, Vec3::NEG_Y, 1.99)).unwrap();
    assert!(world.cast_ray(&short, None).unwrap().is_none());
    let away = RobotRay::try_from((origin, Vec3::Y, 10.0)).unwrap();
    assert!(world.cast_ray(&away, None).unwrap().is_none());
    let inside = RobotRay::try_from((Vec3::new(0.0, -0.1, 0.0), Vec3::Y, 0.01)).unwrap();
    assert_eq!(
        world
            .cast_ray(&inside, None)
            .unwrap()
            .unwrap()
            .metres()
            .to_bits(),
        0.0_f32.to_bits()
    );
}
/// Invalid geometry is rejected before it can reach collision arithmetic.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_rays_have_closed_errors_and_explicit_validation_order() {
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for axis in 0..3 {
            let mut vector = Vec3::ONE;
            vector[axis] = invalid;
            assert_eq!(
                RobotRay::try_from((vector, Vec3::ZERO, -1.0)).unwrap_err(),
                InvalidRobotRay::Origin
            );
            assert_eq!(
                RobotRay::try_from((Vec3::ZERO, vector, -1.0)).unwrap_err(),
                InvalidRobotRay::Direction
            );
        }
        assert_eq!(
            RobotRay::try_from((Vec3::ZERO, Vec3::Y, invalid)).unwrap_err(),
            InvalidRobotRay::Distance
        );
    }
    assert_eq!(
        RobotRay::try_from((Vec3::ZERO, Vec3::ZERO, -1.0)).unwrap_err(),
        InvalidRobotRay::Direction
    );
    for distance in [0.0, -0.0, -1.0] {
        assert_eq!(
            RobotRay::try_from((Vec3::ZERO, Vec3::Y, distance)).unwrap_err(),
            InvalidRobotRay::Distance
        );
    }
    assert_eq!(
        RobotRay::try_from((Vec3::splat(f32::MAX), Vec3::X, f32::MAX)).unwrap_err(),
        InvalidRobotRay::Endpoint
    );
    let _maximum_range = RobotRay::try_from((Vec3::ZERO, Vec3::X, f32::MAX)).unwrap();
}

/// Error categories retain their exact messages and have no nested source.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn ray_errors_keep_their_closed_diagnostic_contract() {
    for (error, message) in [
        (InvalidRobotRay::Origin, "robot ray origin must be finite"),
        (
            InvalidRobotRay::Direction,
            "robot ray direction must be finite and nonzero",
        ),
        (
            InvalidRobotRay::Distance,
            "robot ray distance must be finite and positive",
        ),
        (
            InvalidRobotRay::Endpoint,
            "robot ray endpoint must remain finite",
        ),
    ] {
        assert_eq!(error.to_string(), message);
        assert!(std::error::Error::source(&error).is_none());
    }
}
/// A stopped world rejects queries and reset restores read capability.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn stopped_world_cannot_be_queried_until_reset() {
    use bevy_gym::robots::{DroidAction, DroneAction};
    let mut world = RobotWorld::default();
    let other = RobotWorld::default().snapshot().unwrap();
    let actions = RobotActions::new(
        &other,
        [DroneAction::try_from([0.0; 4]).unwrap(); 3],
        [DroidAction::try_from([0.0; 26]).unwrap(); 3],
    );
    world.advance(actions).unwrap_err();
    let ray = RobotRay::try_from((Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y, 2.0)).unwrap();
    assert_eq!(
        world.cast_ray(&ray, None).unwrap_err(),
        world.snapshot().unwrap_err()
    );
    world.reset().unwrap();
    assert!(world.cast_ray(&ray, None).unwrap().is_some());
}
