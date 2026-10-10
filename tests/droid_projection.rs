//! Bone projection follows the current physical snapshot and never depends on last-frame poses.
#![cfg(feature = "robots")]
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[path = "../examples/robots/standing_scene/projection.rs"]
mod projection;
use bevy::prelude::*;
use bevy_gym::{
    robots::{DroidBody, DroidStanding},
    Env,
};

/// A rotated physical bind snapshot rotates every rendered anchor around its segment centre.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn bone_pose_matches_a_worked_bind_frame_rotation() {
    let observation = DroidStanding::default().reset(Some(42)).observation;
    for body in DroidBody::ALL {
        let state = observation.body(body);
        let offset = Vec3::new(0.1, 0.2, -0.05);
        let bind =
            GlobalTransform::from(Transform::from_translation(body.bind_position() + offset));
        let pose = projection::world_pose(body, state, bind);
        assert!(
            pose.translation()
                .distance(state.position() + state.orientation() * offset)
                < 0.000_002
        );
        assert!(pose.rotation().dot(state.orientation()).abs() > 0.999_99);
        // Recover the same target after conversion through a moving parent.
        let parent = GlobalTransform::from(
            Transform::from_xyz(0.3, 0.5, -0.7).with_rotation(Quat::from_rotation_z(0.4)),
        );
        let local = pose.reparented_to(&parent);
        assert!((parent * local).translation().distance(pose.translation()) < 0.000_002);
    }
}

/// All physical segments map once to the model, while unrelated bones remain uncontrolled.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn named_mapping_is_complete_and_closed() {
    let names = [
        "pelvis",
        "spine_01",
        "neck_01",
        "upperarm_l",
        "lowerarm_l",
        "upperarm_r",
        "lowerarm_r",
        "thigh_l",
        "calf_l",
        "foot_l",
        "thigh_r",
        "calf_r",
        "foot_r",
    ];
    for (name, expected) in names.into_iter().zip(DroidBody::ALL) {
        assert_eq!(projection::body_named(name), Some(expected));
    }
    for other in ["root", "spine_02", "hand_r", "unknown", ""] {
        assert_eq!(projection::body_named(other), None);
    }
}
