//! Body-frame flight and desired-heading features.
use super::super::encoding::encode_target;
use bevy::math::Vec3;
use bevy_gym::robots::DroneTravelObservation;

/// Preserve twelve flight features and append unit desired heading in body XYZ coordinates.
pub(crate) fn encode(observation: DroneTravelObservation) -> [f32; 15] {
    let body = observation.body();
    let destination = observation.destination();
    let original = encode_target(body, destination.position());
    let heading = destination.heading();
    let local_heading = body.orientation().inverse() * Vec3::new(heading.x, 0.0, heading.y);
    let mut result = [0.0; 15];
    let (original_features, heading_features) = result.split_at_mut(12);
    original_features.copy_from_slice(&original);
    heading_features.copy_from_slice(&local_heading.to_array());
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::Env;
    /// The appended heading retains the complete body rotation while flight units remain exact.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn heading_features_preserve_flight_prefix_and_unit_length() {
        let mut environment = super::super::TravelTask::new();
        let observation = environment.reset(Some(42)).observation;
        let features = encode(observation);
        assert_eq!(
            &features[..12],
            &encode_target(observation.body(), observation.destination().position())
        );
        let heading = observation.destination().heading();
        let expected =
            observation.body().orientation().inverse() * Vec3::new(heading.x, 0.0, heading.y);
        assert_eq!(&features[12..], &expected.to_array());
        assert!((expected.length() - 1.0).abs() < 1e-6);
    }
}
