//! Sight exposes only measured points and finite remembered sightings.
#![cfg(feature = "robots")]

#[expect(dead_code, reason = "Movement is covered by pursuit_arena.")]
#[path = "../examples/robots/pursuit/arena.rs"]
mod arena;
#[path = "../examples/robots/pursuit/sight.rs"]
mod sight;

use arena::Arena;
use bevy::math::{Dir3, Vec3};
use sight::{Contact, Sight};
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn window_sighting_does_not_follow_hidden_movement_and_expires() {
    let arena = Arena::default();
    let eye = Vec3::new(-5.0, 1.7, 3.0);
    let character = Vec3::new(-5.0, 1.1, -2.0);
    let mut sight = Sight::default();
    let seen = sight.sample(&arena, eye, Dir3::NEG_Z, character, Duration::ZERO);
    assert_eq!(seen, Contact::Visible(character + Vec3::Y * 0.6));
    let mut other = sight.clone();
    let hidden = Vec3::new(-7.0, 1.1, -2.0);
    let moved = Vec3::new(-7.5, 1.1, -2.0);
    let remembered = sight.sample(&arena, eye, Dir3::NEG_Z, hidden, Duration::from_secs(1));
    assert_eq!(
        remembered,
        other.sample(&arena, eye, Dir3::NEG_Z, moved, Duration::from_secs(1))
    );
    assert_eq!(
        remembered,
        Contact::Remembered {
            point: character + Vec3::Y * 0.6,
            age: Duration::from_secs(1)
        }
    );
    assert!(matches!(
        sight.sample(
            &arena,
            eye,
            Dir3::NEG_Z,
            hidden,
            Duration::from_millis(1999)
        ),
        Contact::Remembered { .. }
    ));
    assert_eq!(
        sight.sample(&arena, eye, Dir3::NEG_Z, hidden, Duration::from_millis(1)),
        Contact::Unknown
    );
    assert_eq!(
        sight.sample(&arena, eye, Dir3::NEG_Z, character, Duration::MAX),
        seen
    );
    sight.forget();
    assert_eq!(
        sight.sample(&arena, eye, Dir3::NEG_Z, hidden, Duration::ZERO),
        Contact::Unknown
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn cone_and_range_include_their_boundaries() {
    let arena = Arena::default();
    let eye = Vec3::new(0.0, 10.0, 0.0);
    for (character, visible) in [
        (Vec3::new(0.0, 9.4, 20.0), true),
        (Vec3::new(0.0, 9.4, 20.001), false),
        (Vec3::new(5.0, 9.4, 5.0), true),
        (Vec3::new(5.001, 9.4, 5.0), false),
        (Vec3::new(0.0, 9.4, -5.0), false),
        (Vec3::new(0.0, 9.4, 0.0), false),
    ] {
        let contact = Sight::default().sample(&arena, eye, Dir3::Z, character, Duration::ZERO);
        assert_eq!(
            matches!(contact, Contact::Visible(_)),
            visible,
            "Character {character:?}"
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn openings_reveal_points_but_solid_surfaces_and_embedded_eyes_do_not() {
    let arena = Arena::default();
    for (eye, character, visible) in [
        (Vec3::new(-5.0, 1.7, 3.0), Vec3::new(-5.0, 1.1, -2.0), true),
        (Vec3::new(-7.0, 1.7, 3.0), Vec3::new(-7.0, 1.1, -2.0), false),
        (Vec3::new(0.0, 1.7, -3.0), Vec3::new(-4.0, 1.1, -3.0), true),
        (Vec3::new(5.0, 1.7, 6.0), Vec3::new(5.0, 1.1, 0.0), true),
        (Vec3::new(2.0, 1.7, 0.0), Vec3::new(5.0, 1.1, 0.0), false),
        (Vec3::new(-7.0, 1.7, 0.0), Vec3::new(-7.0, 1.1, -2.0), false),
        (Vec3::new(-7.0, 1.7, 3.0), Vec3::new(-7.0, 1.1, 0.15), false),
        (
            Vec3::new(-7.0, 1.7, 3.0),
            Vec3::new(-7.0, 1.1, 0.1505),
            false,
        ),
        (Vec3::new(-7.0, 1.7, 3.0), Vec3::new(-7.0, 1.1, 0.152), true),
    ] {
        let forward = Dir3::new(character + Vec3::Y * 0.6 - eye).expect("Nonzero sight line");
        let contact = Sight::default().sample(&arena, eye, forward, character, Duration::ZERO);
        assert_eq!(
            matches!(contact, Contact::Visible(_)),
            visible,
            "Eye {eye:?}, character {character:?}"
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn an_exposed_head_can_be_seen_above_a_blocked_chest() {
    let arena = Arena::default();
    let eye = Vec3::new(-5.0, 1.1, 3.0);
    let character = Vec3::new(-5.0, 0.5, -2.0);
    assert!(arena.obstruction(eye, character + Vec3::Y * 0.2).is_some());
    assert_eq!(
        Sight::default().sample(&arena, eye, Dir3::NEG_Z, character, Duration::ZERO),
        Contact::Visible(character + Vec3::Y * 0.6)
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn invalid_centres_fail_closed_without_replacing_memory() {
    let arena = Arena::default();
    let eye = Vec3::new(0.0, 10.0, 0.0);
    let character = Vec3::new(0.0, 9.4, 5.0);
    for invalid in [
        Vec3::splat(f32::NAN),
        Vec3::splat(f32::INFINITY),
        Vec3::splat(f32::NEG_INFINITY),
        Vec3::splat(1000.01),
        Vec3::splat(-1000.01),
    ] {
        for (origin, target) in [(invalid, character), (eye, invalid)] {
            let mut sight = Sight::default();
            let seen = sight.sample(&arena, eye, Dir3::Z, character, Duration::ZERO);
            let Contact::Visible(point) = seen else {
                panic!("Expected sighting")
            };
            assert_eq!(
                sight.sample(&arena, origin, Dir3::Z, target, Duration::ZERO),
                Contact::Remembered {
                    point,
                    age: Duration::ZERO
                }
            );
            assert_eq!(
                sight.sample(&arena, origin, Dir3::Z, target, Duration::MAX),
                Contact::Unknown
            );
        }
    }
    for (eye_x, target_x) in [
        (1000.0, 999.0),
        (-1000.0, -999.0),
        (999.0, 1000.0),
        (-999.0, -1000.0),
    ] {
        let eye = Vec3::new(eye_x, 10.0, 0.0);
        let target = Vec3::new(target_x, 9.4, 0.0);
        let forward = Dir3::new(target + Vec3::Y * 0.6 - eye).expect("Direction");
        assert!(matches!(
            Sight::default().sample(&arena, eye, forward, target, Duration::ZERO),
            Contact::Visible(_)
        ));
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lower_exposed_points_are_used_when_the_lintel_hides_the_head() {
    let arena = Arena::default();
    for (eye_height, character_height, exposed_height) in [(2.4, 2.1, 0.2), (2.7, 2.4, -0.3)] {
        let eye = Vec3::new(-5.0, eye_height, 3.0);
        let character = Vec3::new(-5.0, character_height, -2.0);
        assert!(arena.obstruction(eye, character + Vec3::Y * 0.6).is_some());
        assert_eq!(
            Sight::default().sample(&arena, eye, Dir3::NEG_Z, character, Duration::ZERO),
            Contact::Visible(character + Vec3::Y * exposed_height)
        );
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn memory_age_saturates_and_reacquisition_replaces_the_old_point() {
    let arena = Arena::default();
    let eye = Vec3::new(0.0, 10.0, 0.0);
    let mut sight = Sight::default();
    let first = Vec3::new(0.0, 9.4, 5.0);
    let second = Vec3::new(1.0, 9.4, 5.0);
    sight.sample(&arena, eye, Dir3::Z, first, Duration::ZERO);
    let hidden = Vec3::new(0.0, 9.4, -5.0);
    sight.sample(&arena, eye, Dir3::Z, hidden, Duration::from_secs(1));
    assert_eq!(
        sight.sample(&arena, eye, Dir3::Z, hidden, Duration::MAX),
        Contact::Unknown
    );
    assert_eq!(
        sight.sample(&arena, eye, Dir3::Z, second, Duration::ZERO),
        Contact::Visible(second + Vec3::Y * 0.6)
    );
    assert_eq!(
        sight.sample(&arena, eye, Dir3::Z, hidden, Duration::from_millis(20)),
        Contact::Remembered {
            point: second + Vec3::Y * 0.6,
            age: Duration::from_millis(20)
        }
    );
}
