//! Rigid projection between mannequin bind frames and authoritative segment poses.
//!
//! The source mannequin has unit scale and no shear. Bevy 0.18.1 preserves the global
//! target through `GlobalTransform::reparented_to` before ordinary transform propagation.
//! <https://docs.rs/bevy/0.18.1/bevy/prelude/struct.GlobalTransform.html#method.reparented_to>.

use bevy::prelude::*;
use bevy_gym::robots::{DroidBody, DroidBodyState};

/// Closed mapping from the licensed mannequin's thirteen anchors to physical segments.
pub(crate) fn body_named(name: &str) -> Option<DroidBody> {
    use DroidBody as B;
    match name {
        "pelvis" => Some(B::Pelvis),
        "spine_01" => Some(B::Torso),
        "neck_01" => Some(B::Head),
        "upperarm_l" => Some(B::LeftUpperArm),
        "lowerarm_l" => Some(B::LeftForearm),
        "upperarm_r" => Some(B::RightUpperArm),
        "lowerarm_r" => Some(B::RightForearm),
        "thigh_l" => Some(B::LeftThigh),
        "calf_l" => Some(B::LeftCalf),
        "foot_l" => Some(B::LeftFoot),
        "thigh_r" => Some(B::RightThigh),
        "calf_r" => Some(B::RightCalf),
        "foot_r" => Some(B::RightFoot),
        _ => None,
    }
}

/// Move an immutable bind frame around its owning physical segment centre.
///
/// All positions use metres and all rotations map segment-local axes to world axes.
/// Multiplication order is physical pose, inverse bind-centre translation, then model bind.
/// The result is display data; it cannot mutate physics or policy state.
pub(crate) fn world_pose(
    body: DroidBody,
    state: DroidBodyState,
    bind: GlobalTransform,
) -> GlobalTransform {
    let physical = GlobalTransform::from(
        Transform::from_translation(state.position()).with_rotation(state.orientation()),
    );
    physical * GlobalTransform::from(Transform::from_translation(-body.bind_position())) * bind
}
