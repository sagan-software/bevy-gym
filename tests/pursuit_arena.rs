//! Gameplay collision contracts through the example's shared simulation helper.

#![cfg(feature = "robots")]

#[path = "../examples/robots/pursuit/arena.rs"]
mod arena;

use arena::{Arena, Movement};
use bevy::math::Vec3;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn movement_is_fixed_step_and_reset_restores_the_character() {
    let mut arena = Arena::default();
    let start = arena.position();
    arena.step(Movement::Right);
    assert!((arena.position().x - start.x - 0.08).abs() < 1.0e-5);
    arena.reset();
    assert_eq!(arena.position(), start);
    let mut fresh = Arena::default();
    for _ in 0..20 {
        arena.step(Movement::ForwardRight);
        fresh.step(Movement::ForwardRight);
        assert_eq!(arena.position(), fresh.position());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn perimeter_stops_the_character_and_idle_settles_on_the_floor() {
    let mut arena = Arena::default();
    for _ in 0..600 {
        arena.step(Movement::Backward);
    }
    assert!((12.3..12.7).contains(&arena.position().z));
    for _ in 0..100 {
        arena.step(Movement::Idle);
    }
    assert!((0.89..0.94).contains(&arena.position().y));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn windows_and_door_are_open_but_wall_and_roof_block_rays() {
    let arena = Arena::default();
    assert!(!blocked(
        &arena,
        Vec3::new(-5.0, 1.8, 2.0),
        Vec3::new(-5.0, 1.8, -3.0)
    ));
    assert!(blocked(
        &arena,
        Vec3::new(-7.0, 1.8, 2.0),
        Vec3::new(-7.0, 1.8, -3.0)
    ));
    assert!(!blocked(
        &arena,
        Vec3::new(0.0, 1.0, -3.0),
        Vec3::new(-5.0, 1.0, -3.0)
    ));
    assert!(blocked(
        &arena,
        Vec3::new(-5.0, 5.0, -3.0),
        Vec3::new(-5.0, 1.0, -3.0)
    ));
    assert!(!blocked(
        &arena,
        Vec3::new(5.0, 1.7, 6.0),
        Vec3::new(5.0, 1.7, -4.0)
    ));
    assert!(blocked(
        &arena,
        Vec3::new(2.0, 1.7, 1.0),
        Vec3::new(5.0, 1.7, 1.0)
    ));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn all_button_combinations_cancel_or_keep_unit_speed() {
    let expected = [
        Movement::Idle,
        Movement::Forward,
        Movement::Backward,
        Movement::Idle,
        Movement::Left,
        Movement::ForwardLeft,
        Movement::BackwardLeft,
        Movement::Left,
        Movement::Right,
        Movement::ForwardRight,
        Movement::BackwardRight,
        Movement::Right,
        Movement::Idle,
        Movement::Forward,
        Movement::Backward,
        Movement::Idle,
    ];
    for (mask, expected) in (0_u8..16).zip(expected) {
        let keys = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0, mask & 8 != 0];
        assert_eq!(Movement::from(keys), expected);
        let mut arena = Arena::default();
        let start = arena.position();
        arena.step(Movement::from(keys));
        let delta = arena.position() - start;
        let length = Vec3::new(delta.x, 0.0, delta.z).length();
        if keys[0] == keys[1] && keys[2] == keys[3] {
            assert!(length < 1.0e-6);
        } else {
            assert!((length - 0.08).abs() < 1.0e-5);
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn layout_has_positive_finite_boxes_and_every_landmark_material() {
    use arena::layout::Surface;
    let arena = Arena::default();
    assert_eq!(arena.blocks().len(), 31);
    for surface in [Surface::Floor, Surface::Wall, Surface::Cover, Surface::Pipe] {
        assert!(arena.blocks().iter().any(|block| block.surface == surface));
    }
    for block in arena.blocks() {
        assert!(block.centre.is_finite());
        assert!(block.half.min_element() > 0.0);
        assert!(block.rotation.is_normalized());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_sight_coordinates_fail_closed_and_empty_segments_are_clear() {
    let arena = Arena::default();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(blocked(&arena, Vec3::splat(bad), Vec3::ZERO));
        assert!(blocked(&arena, Vec3::ZERO, Vec3::splat(bad)));
    }
    assert!(!blocked(&arena, Vec3::Y, Vec3::Y));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn character_enters_door_and_pipe_but_cannot_walk_through_window_sill() {
    let mut arena = Arena::default();
    for _ in 0..150 {
        arena.step(Movement::Forward);
    }
    for _ in 0..62 {
        arena.step(Movement::Left);
    }
    assert!(arena.position().x < -4.8);
    let entered = arena.position();
    assert!(
        (-3.7..-2.3).contains(&entered.z),
        "door position {entered:?}"
    );
    for _ in 0..100 {
        arena.step(Movement::Backward);
    }
    assert!(arena.position().z < -0.44);
    arena.reset();
    for _ in 0..62 {
        arena.step(Movement::Right);
    }
    for _ in 0..200 {
        arena.step(Movement::Forward);
    }
    assert!((arena.position().x - 4.96).abs() < 0.1);
    let exited = arena.position();
    assert!(exited.z < -4.4, "pipe exit {exited:?}");
}

/// Express visibility checks in terms of the same nearest hit used by the camera.
fn blocked(arena: &Arena, from: Vec3, to: Vec3) -> bool {
    arena.obstruction(from, to).is_some()
}
