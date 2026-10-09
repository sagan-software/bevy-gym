//! Append actuator health to the existing body-frame motion inputs.

use bevy_gym::robots::{DroneMotor, DroneMotorState, DroneObservation};

/// Encode twelve motion features followed by four working indicators in motor order.
pub(crate) fn with_motor_health(observation: DroneObservation, motion: [f32; 12]) -> [f32; 16] {
    let mut features = [0.0; 16];
    features[..12].copy_from_slice(&motion);
    for (feature, motor) in features[12..].iter_mut().zip(DroneMotor::ALL) {
        *feature = match observation.motor_state(motor) {
            DroneMotorState::Working => 1.0,
            DroneMotorState::Failed => 0.0,
        };
    }
    features
}
