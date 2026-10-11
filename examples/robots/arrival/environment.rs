//! Reset-only destination sampling on the original drone travel dynamics.
use bevy::math::{Vec2, Vec3};
use bevy_gym::robots::{DroneAction, DroneDestination, DroneTravel, DroneTravelObservation};
use bevy_gym::training::SplitMix64;
use bevy_gym::{Env, Reset, Step};

/// Own the authoritative travel body without selecting any actuator command.
pub(crate) struct TravelTask {
    /// Original physical environment and immutable episode destination.
    environment: DroneTravel,
}
impl TravelTask {
    /// Initialize a valid task before the caller's explicit seeded reset.
    pub(crate) fn new() -> Self {
        Self {
            environment: DroneTravel::new(Self::destination(0)),
        }
    }

    /// Draw azimuth, radius in metres, then independent heading in radians, as in training.
    fn destination(seed: u64) -> DroneDestination {
        let mut random = SplitMix64::new(seed ^ 0x7472_6176_656c_0001);
        let angle = random.f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let radius_m = random.f32_between(1.0, 3.0);
        let heading_angle = random.f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        // Radius and height are metres; heading components are dimensionless.
        DroneDestination::try_from((
            Vec3::new(radius_m * angle.cos(), 2.0, radius_m * angle.sin()),
            Vec2::new(heading_angle.cos(), heading_angle.sin()),
        ))
        .expect("finite interior sampled task")
    }
}
impl Env for TravelTask {
    type Observation = DroneTravelObservation;
    type Action = DroneAction;
    type Info = ();
    /// Reset the original physical task, selecting a destination only for explicit roots.
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        // An explicit root selects a destination; None retains it, exactly as in collection.
        if let Some(seed) = seed {
            self.environment = DroneTravel::new(Self::destination(seed));
        }
        self.environment.reset(seed)
    }
    /// Apply the caller's motor action through the original physical rules.
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        self.environment.step(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Sampling changes only the destination; reset and motor dynamics remain original.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn sampled_tasks_preserve_original_physics_and_continuing_resets() {
        let mut task = TravelTask::new();
        for seed in [0, 1, 42, u64::MAX] {
            let destination = TravelTask::destination(seed);
            assert!((1.0..3.0).contains(&(destination.position().length_squared() - 4.0).sqrt()));
            let mut direct = DroneTravel::new(destination);
            assert_eq!(
                task.reset(Some(seed)).observation,
                direct.reset(Some(seed)).observation
            );
            assert_eq!(task.reset(None).observation, direct.reset(None).observation);
            for motors in [[0.0; 4], [0.5; 4], [0.2, 0.4, 0.6, 0.8]] {
                let action = DroneAction::try_from(motors).expect("bounded fixture");
                let actual = task.step(action);
                let expected = direct.step(action);
                assert_eq!(actual.observation, expected.observation);
                assert_eq!(actual.reward.to_bits(), expected.reward.to_bits());
                assert_eq!(actual.status, expected.status);
            }
        }
    }
}
