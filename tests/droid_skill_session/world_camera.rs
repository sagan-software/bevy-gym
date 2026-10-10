//! Shared-world camera selection and framing cannot advance physics.

use crate::{
    standing::world_session::Session,
    world_camera::{frame, Mode},
};
use bevy::math::{Mat4, Vec3};
use bevy_gym::robots::{DroidBody, RobotId, RobotSlot};

/// Retained RL policies provide the same snapshot seam used by browser playback.
fn load() -> Session {
    Session::load(
        include_bytes!("../../docs/progress/drone-hover.mpk").to_vec(),
        include_bytes!("../../docs/progress/standing-seed17-update22940/checkpoint.mpk").to_vec(),
        include_bytes!("../../docs/progress/standing-seed17-update22940/checkpoint.json"),
    )
    .expect("retained RL policies")
}

/// Every body centre stays inside desktop and narrow frusta at reset and clip completion.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn overview_and_six_follow_targets_fit_both_viewports_without_mutating_physics() {
    let mut session = load();
    for finish in [false, true] {
        if finish {
            while !session.finished() {
                session.step();
            }
        }
        let before = session.snapshot();
        let time = before.elapsed();
        for aspect in [1280.0 / 744.0, 390.0 / 788.0] {
            assert_framed(before, aspect);
        }
        assert_eq!(session.snapshot().elapsed(), time);
    }
}

/// Compare homogeneous clip coordinates for all seven automatic camera targets.
fn assert_framed(snapshot: &bevy_gym::robots::RobotSnapshot, aspect: f32) {
    let fov = std::f32::consts::FRAC_PI_3;
    let mut mode = Mode::default();
    for _ in 0..7 {
        let (position, target) = frame(snapshot, mode, aspect, fov).expect("finite frame");
        let clip_from_world = Mat4::perspective_rh(fov, aspect, 0.1, 1000.0)
            * Mat4::look_at_rh(position, target, Vec3::Y);
        assert_selected_centres(clip_from_world, snapshot, mode);
        mode = mode.next();
    }
    assert_eq!(mode, Mode::All);
}

/// The selected robot's own body centres supply independent visibility assertions.
fn assert_selected_centres(clip: Mat4, snapshot: &bevy_gym::robots::RobotSnapshot, mode: Mode) {
    for slot in RobotSlot::ALL {
        if matches!(mode, Mode::All) || mode == Mode::Follow(RobotId::Drone(slot)) {
            assert_visible(clip, snapshot.drone(slot).position());
        }
        if matches!(mode, Mode::All) || mode == Mode::Follow(RobotId::Droid(slot)) {
            for body in DroidBody::ALL {
                assert_visible(clip, snapshot.droid(slot).body(body).position());
            }
        }
    }
}

/// Invalid viewport profiles and free mode retain presentation; each closed label is stable.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn viewport_rejection_and_target_labels_preserve_the_closed_camera_profile() {
    let session = load();
    let snapshot = session.snapshot();
    assert!(frame(snapshot, Mode::Free, 1.0, 1.0).is_none());
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(frame(snapshot, Mode::All, invalid, 1.0).is_none());
        assert!(frame(snapshot, Mode::All, 1.0, invalid).is_none());
    }
    assert!(frame(snapshot, Mode::All, 1.0, std::f32::consts::PI).is_none());
    assert_eq!(Mode::Free.next(), Mode::All);
    assert_eq!(Mode::All.label(), "All");
    assert_eq!(
        Mode::Follow(RobotId::Drone(RobotSlot::First)).label(),
        "Drone 1"
    );
    assert_eq!(
        Mode::Follow(RobotId::Drone(RobotSlot::Second)).label(),
        "Drone 2"
    );
    assert_eq!(
        Mode::Follow(RobotId::Drone(RobotSlot::Third)).label(),
        "Drone 3"
    );
    assert_eq!(
        Mode::Follow(RobotId::Droid(RobotSlot::First)).label(),
        "Droid 1"
    );
    assert_eq!(
        Mode::Follow(RobotId::Droid(RobotSlot::Second)).label(),
        "Droid 2"
    );
    assert_eq!(
        Mode::Follow(RobotId::Droid(RobotSlot::Third)).label(),
        "Droid 3"
    );
    assert_eq!(Mode::Free.label(), "Free");
}

/// Assert homogeneous device coordinates against the documented 10 percent viewing margin.
fn assert_visible(clip_from_world: Mat4, centre: Vec3) {
    let clip = clip_from_world * centre.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    assert!(
        clip.w > 0.0 && (0.0..=1.0).contains(&ndc.z) && ndc.x.abs() < 0.9 && ndc.y.abs() < 0.9,
        "visible centre {ndc:?}"
    );
}
