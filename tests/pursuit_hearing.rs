//! Noise events expose coarse bearings with a finite lifetime, never exact positions.
#![cfg(feature = "robots")]

#[expect(
    dead_code,
    reason = "Movement and camera queries are covered by pursuit_arena."
)]
#[path = "../examples/robots/pursuit/arena.rs"]
mod arena;
#[path = "../examples/robots/pursuit/hearing.rs"]
mod hearing;

use arena::Arena;
use bevy::math::Vec3;
use hearing::{Bearing, Hearing, Noise};
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn hearing_retains_only_an_event_bearing_and_expires_without_new_noise() {
    let arena = Arena::default();
    let mut hearing = Hearing::default();
    let listener = Vec3::new(0.0, 10.0, 0.0);
    assert!(hearing.latest().is_none());
    hearing.hear(
        &arena,
        listener,
        Vec3::new(0.0, 10.0, -5.0),
        Noise::Footstep,
    );
    let heard = hearing.latest().expect("Audible footstep");
    assert_eq!(heard.bearing(), Bearing::North);
    assert_eq!(heard.noise(), Noise::Footstep);
    assert_eq!(heard.age(), Duration::ZERO);
    hearing.advance(Duration::from_millis(1999));
    assert_eq!(
        hearing.latest().expect("Still remembered").age(),
        Duration::from_millis(1999)
    );
    hearing.advance(Duration::from_millis(1));
    assert!(hearing.latest().is_none());
    hearing.advance(Duration::MAX);
    assert!(hearing.latest().is_none());
    hearing.hear(&arena, listener, Vec3::new(4.0, 10.0, 0.0), Noise::Gunshot);
    hearing.forget();
    assert!(hearing.latest().is_none());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn clear_and_obstructed_ranges_have_inclusive_boundaries() {
    let arena = Arena::default();
    for (noise, range) in [(Noise::Footstep, 8.0), (Noise::Gunshot, 24.0)] {
        let listener = Vec3::new(0.0, 10.0, 0.0);
        for (distance, audible) in [(range, true), (range + 0.001, false)] {
            let mut hearing = Hearing::default();
            hearing.hear(&arena, listener, Vec3::new(distance, 10.0, 0.0), noise);
            assert_eq!(hearing.latest().is_some(), audible);
        }
    }
    for (noise, listener_z, source_z) in [(Noise::Footstep, 2.0, -2.0), (Noise::Gunshot, 6.0, -6.0)]
    {
        let listener = Vec3::new(-7.0, 1.7, listener_z);
        for (offset, audible) in [(0.0, true), (-0.001, false)] {
            let mut hearing = Hearing::default();
            let source = Vec3::new(-7.0, 1.7, source_z + offset);
            assert!(arena.obstruction(listener, source).is_some());
            hearing.hear(&arena, listener, source, noise);
            assert_eq!(hearing.latest().is_some(), audible);
        }
    }
    for (listener, source, audible) in [
        (Vec3::new(-5.0, 1.7, 3.0), Vec3::new(-5.0, 1.7, -2.0), true),
        (Vec3::new(5.0, 1.7, 6.0), Vec3::new(5.0, 1.7, 0.0), true),
        (Vec3::new(0.0, 1.7, 0.0), Vec3::new(5.0, 1.7, 0.0), false),
    ] {
        let mut hearing = Hearing::default();
        hearing.hear(&arena, listener, source, Noise::Footstep);
        assert_eq!(hearing.latest().is_some(), audible);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn every_coarse_bearing_and_label_matches_world_axes() {
    let arena = Arena::default();
    let listener = Vec3::new(0.0, 10.0, 0.0);
    for (x, z, bearing, label) in [
        (0.0, -1.0, Bearing::North, "north"),
        (1.0, -1.0, Bearing::NorthEast, "northeast"),
        (1.0, 0.0, Bearing::East, "east"),
        (1.0, 1.0, Bearing::SouthEast, "southeast"),
        (0.0, 1.0, Bearing::South, "south"),
        (-1.0, 1.0, Bearing::SouthWest, "southwest"),
        (-1.0, 0.0, Bearing::West, "west"),
        (-1.0, -1.0, Bearing::NorthWest, "northwest"),
        (0.0, 0.0, Bearing::Unresolved, "unresolved"),
    ] {
        let mut hearing = Hearing::default();
        hearing.hear(
            &arena,
            listener,
            listener + Vec3::new(x, 0.0, z),
            Noise::Footstep,
        );
        let cue = hearing.latest().expect("Nearby source");
        assert_eq!(cue.bearing(), bearing);
        assert_eq!(bearing.label(), label);
    }
    assert_eq!(Noise::Footstep.label(), "steps");
    assert_eq!(Noise::Gunshot.label(), "shot");
    let mut hearing = Hearing::default();
    hearing.hear(&arena, listener, listener + Vec3::Y, Noise::Footstep);
    assert_eq!(
        hearing.latest().expect("Vertical source").bearing(),
        Bearing::Unresolved
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn cardinal_sectors_own_ties_and_diagonals_start_beyond_them() {
    let arena = Arena::default();
    let listener = Vec3::new(0.0, 10.0, 0.0);
    let boundary = std::f32::consts::SQRT_2 - 1.0;
    for (x_sign, z_sign, horizontal, vertical, diagonal) in [
        (1.0, -1.0, Bearing::East, Bearing::North, Bearing::NorthEast),
        (1.0, 1.0, Bearing::East, Bearing::South, Bearing::SouthEast),
        (-1.0, 1.0, Bearing::West, Bearing::South, Bearing::SouthWest),
        (
            -1.0,
            -1.0,
            Bearing::West,
            Bearing::North,
            Bearing::NorthWest,
        ),
    ] {
        for (x, z, expected) in [
            (x_sign * boundary, z_sign, vertical),
            (x_sign, z_sign * boundary, horizontal),
            (x_sign * (boundary + 0.001), z_sign, diagonal),
            (x_sign, z_sign * (boundary + 0.001), diagonal),
        ] {
            let mut hearing = Hearing::default();
            hearing.hear(
                &arena,
                listener,
                listener + Vec3::new(x, 0.0, z),
                Noise::Footstep,
            );
            assert_eq!(
                hearing.latest().expect("Sector boundary").bearing(),
                expected
            );
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn silent_or_invalid_events_cannot_refresh_or_replace_an_existing_cue() {
    let arena = Arena::default();
    let listener = Vec3::new(0.0, 10.0, 0.0);
    let source = Vec3::new(0.0, 10.0, -3.0);
    let mut hearing = Hearing::default();
    hearing.hear(&arena, listener, source, Noise::Gunshot);
    hearing.advance(Duration::from_millis(500));
    let previous = hearing.latest();
    for invalid in [
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NEG_INFINITY),
        Vec3::splat(1000.01),
        Vec3::splat(-1000.01),
    ] {
        hearing.hear(&arena, invalid, source, Noise::Footstep);
        assert_eq!(hearing.latest(), previous);
        hearing.hear(&arena, listener, invalid, Noise::Footstep);
        assert_eq!(hearing.latest(), previous);
    }
    hearing.hear(&arena, listener, Vec3::new(9.0, 10.0, 0.0), Noise::Footstep);
    assert_eq!(hearing.latest(), previous);
    hearing.advance(Duration::MAX);
    assert!(hearing.latest().is_none());
    hearing.hear(&arena, listener, Vec3::new(3.0, 10.0, 0.0), Noise::Footstep);
    let cue = hearing.latest().expect("New event");
    assert_eq!(cue.bearing(), Bearing::East);
    assert_eq!(cue.noise(), Noise::Footstep);
    assert_eq!(cue.age(), Duration::ZERO);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn equal_sectors_do_not_disclose_different_source_distances() {
    let arena = Arena::default();
    let listener = Vec3::new(0.0, 10.0, 0.0);
    let mut first = Hearing::default();
    let mut second = Hearing::default();
    first.hear(
        &arena,
        listener,
        Vec3::new(0.0, 10.0, -3.0),
        Noise::Footstep,
    );
    second.hear(
        &arena,
        listener,
        Vec3::new(0.2, 10.0, -6.0),
        Noise::Footstep,
    );
    assert_eq!(first.latest(), second.latest());
    for (x, target_x) in [
        (1000.0, 999.0),
        (-1000.0, -999.0),
        (999.0, 1000.0),
        (-999.0, -1000.0),
    ] {
        let mut hearing = Hearing::default();
        hearing.hear(
            &arena,
            Vec3::new(x, 10.0, 0.0),
            Vec3::new(target_x, 10.0, 0.0),
            Noise::Footstep,
        );
        assert!(hearing.latest().is_some());
    }
}
