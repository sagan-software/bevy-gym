//! Licensed drone mesh alignment and named rotor pivots shared by both viewers.

use bevy::prelude::*;
use bevy_gym::robots::DroneMotor;
use std::f32::consts::PI;

/// Source-model body centre in metres, before its +Z front is rotated to -Z.
pub(super) const BODY_CENTRE: Vec3 = Vec3::new(0.000_070_5, -0.055_414, -0.134_026_5);

/// Translate the model's body centre to zero, then turn its front toward -Z.
pub(super) fn model_alignment() -> Transform {
    let rotation = Quat::from_rotation_y(PI);
    Transform::from_translation(rotation * -BODY_CENTRE).with_rotation(rotation)
}

/// Source-space pivots and motor identities for the four named mesh nodes.
pub(super) fn rotor(name: &str) -> Option<(Vec3, DroneMotor, f32)> {
    let (x, z, motor, sign) = match name {
        "Rotor_FL" => (0.250_664_5, 0.126_471, DroneMotor::FrontLeft, 1.0),
        "Rotor_FR" => (-0.250_420_5, 0.126_471, DroneMotor::FrontRight, -1.0),
        "Rotor_BR" => (-0.250_420_5, -0.394_633_5, DroneMotor::RearRight, 1.0),
        "Rotor_BL" => (0.250_664_5, -0.394_633_5, DroneMotor::RearLeft, -1.0),
        _ => return None,
    };
    Some((Vec3::new(x, 0.032_098_5, z), motor, sign))
}

/// Rotate vertices around their mesh centre without orbiting the whole rotor.
pub(super) fn spin_about(pivot: Vec3, angle: f32) -> Transform {
    let rotation = Quat::from_rotation_y(angle);
    Transform::from_translation(pivot - rotation * pivot).with_rotation(rotation)
}
