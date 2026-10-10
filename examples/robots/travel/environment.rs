//! Sample one task destination per episode without selecting motor actions.

use bevy::math::{Vec2, Vec3};
use bevy_gym::robots::{DroneAction, DroneDestination, DroneTravel, DroneTravelObservation};
use bevy_gym::training::{SeedConfig, SplitMix64};
use bevy_gym::{Env, Reset, Step};

use super::stage::Stage;

/// Reusable sampled task around the same fixed-destination physical environment.
pub(crate) struct TravelTask {
    /// Closed sampling profile retained across resets.
    stage: Stage,
    /// Root seed streams for independently sampled body and destination resets.
    seeds: SeedConfig,
    /// Reset index within the current root stream.
    episode: u64,
    /// Authoritative physical environment; no controller or writable world is exposed.
    environment: DroneTravel,
}

impl TravelTask {
    /// Construct the same episode as an explicit reset with seed zero.
    pub(crate) fn new(stage: Stage) -> Self {
        let seeds = SeedConfig::from_root(0);
        Self {
            stage,
            seeds,
            episode: 0,
            environment: make_episode(stage, seeds, 0),
        }
    }

    /// Supply a noncapturing factory to the shared rollout collector.
    pub(crate) fn factory(stage: Stage) -> fn() -> Self {
        match stage {
            Stage::Near => || Self::new(Stage::Near),
            Stage::Far => || Self::new(Stage::Far),
            Stage::Fast => || Self::new(Stage::Fast),
        }
    }
}

impl Env for TravelTask {
    type Observation = DroneTravelObservation;
    type Action = DroneAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        // Explicit roots restart both streams; continuing resets advance one episode.
        if let Some(seed) = seed {
            self.seeds = SeedConfig::from_root(seed);
            self.episode = 0;
        } else {
            self.episode = self.episode.wrapping_add(1);
        }
        self.environment = make_episode(self.stage, self.seeds, self.episode);
        Reset {
            observation: self.environment.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        self.environment.step(action)
    }
}

/// Draw radius, azimuth, height, and heading in that order from the destination stream.
fn make_episode(stage: Stage, seeds: SeedConfig, episode: u64) -> DroneTravel {
    let mut random = SplitMix64::new(seeds.environment_episode(1, episode));
    let ((minimum_radius, maximum_radius), (minimum_height, maximum_height)) = stage.bounds();
    let radius = random.f32_between(minimum_radius, maximum_radius);
    let azimuth = random.f32_between(-std::f32::consts::PI, std::f32::consts::PI);
    let height = random.f32_between(minimum_height, maximum_height);
    let heading = random.f32_between(-std::f32::consts::PI, std::f32::consts::PI);
    // Radius and height are metres; trigonometric factors and normalized heading are unitless.
    let position = Vec3::new(radius * azimuth.cos(), height, radius * azimuth.sin());
    let destination =
        DroneDestination::try_from((position, Vec2::new(heading.cos(), heading.sin())))
            .expect("closed stages generate finite interior positions and nonzero headings");
    let mut environment = DroneTravel::new(destination);
    environment.reset(Some(seeds.environment_episode(0, episode)));
    environment
}
