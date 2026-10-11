//! Bounded RL travel research; every motor request comes from the actual PPO policy.
#![expect(dead_code, unused_imports, reason = "Shared lesson modules include entry points unused by this research binary.")]
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/learning/mod.rs"]
mod learning;
use bevy::math::{Vec2, Vec3};
use bevy_gym::robots::{DroneAction, DroneDestination, DroneTravel, DroneTravelObservation};
use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoPolicy, SeedConfig, SplitMix64};
use bevy_gym::{Env, EpisodeStatus, Reset, Step, TimeLimit};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Write, num::NonZeroU16, path::Path};

/// Reset-only task sampling retains the original travel dynamics and reward.
struct TravelTask { environment: DroneTravel }
impl TravelTask {
    /// Sample only an episode destination, never an actuator or route.
    fn destination(seed: u64) -> DroneDestination {
        let mut random = SplitMix64::new(seed ^ 0x7472_6176_656c_0001);
        let angle = random.f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let radius_m = random.f32_between(1.0, 3.0);
        let heading_angle = random.f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        DroneDestination::try_from((
            Vec3::new(radius_m * angle.cos(), 2.0, radius_m * angle.sin()),
            Vec2::new(heading_angle.cos(), heading_angle.sin()),
        )).expect("finite interior sampled task")
    }
    /// Create a private body with a valid initial task before the collector's seeded reset.
    fn new() -> Self { Self { environment: DroneTravel::new(Self::destination(0)) } }
}
impl Env for TravelTask {
    type Observation = DroneTravelObservation;
    type Action = DroneAction;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        if let Some(seed) = seed { self.environment = DroneTravel::new(Self::destination(seed)); }
        self.environment.reset(seed)
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> { self.environment.step(action) }
}

/// Retain the twelve flight features and append the desired unit heading in body XYZ coordinates.
fn encode(observation: &DroneTravelObservation) -> [f32; 15] {
    let body = observation.body();
    let destination = observation.destination();
    let original = learning::encoding::encode_target(body, destination.position());
    let heading = destination.heading();
    let local_heading = body.orientation().inverse() * Vec3::new(heading.x, 0.0, heading.y);
    let mut result = [0.0; 15];
    let (original_features, heading_features) = result.split_at_mut(12);
    original_features.copy_from_slice(&original);
    heading_features.copy_from_slice(&local_heading.to_array());
    result
}

/// Original-physics frozen-policy outcome, independent of the rollout collector.
#[derive(Serialize)]
struct Outcome {
    seed: u64,
    steps: usize,
    reward: f64,
    stable_tail_actions: usize,
    final_distance_m: f32,
    final_speed_m_s: f32,
    final_heading_alignment: f32,
    passed: bool,
}
/// Check every final two-second action against the predeclared travel, heading and motion limits.
fn evaluate(policy: &RecurrentPpoPolicy, seed: u64) -> Result<Outcome, Box<dyn Error>> {
    let mut environment = TimeLimit::new(TravelTask::new(), 500)?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut reward = 0.0;
    let mut steps = 0;
    let mut stable_tail_actions = 0;
    for action_number in 1..=500 {
        let decision = policy.mean_action(&encode(&observation), &memory)?;
        let action = learning::decode_action(&decision.action)?;
        memory = decision.next_memory;
        let result = environment.step(action);
        observation = result.observation;
        reward += result.reward;
        steps = action_number;
        let body = observation.body();
        let goal = observation.destination();
        let heading = goal.heading();
        let alignment = (body.orientation() * Vec3::NEG_Z).dot(Vec3::new(heading.x, 0.0, heading.y));
        if action_number > 400
            && body.position().distance(goal.position()) <= 0.3
            && body.linear_velocity().length() <= 0.3
            && (body.orientation() * Vec3::Y).y >= 0.95
            && body.angular_velocity().length() <= 0.5
            && alignment >= 15.0_f32.to_radians().cos()
        { stable_tail_actions += 1; }
        if result.status.is_done() { break; }
    }
    let body = observation.body();
    let goal = observation.destination();
    let heading = goal.heading();
    Ok(Outcome {
        seed, steps, reward, stable_tail_actions,
        final_distance_m: body.position().distance(goal.position()),
        final_speed_m_s: body.linear_velocity().length(),
        final_heading_alignment: (body.orientation() * Vec3::NEG_Z).dot(Vec3::new(heading.x, 0.0, heading.y)),
        passed: steps == 500 && stable_tail_actions == 100,
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let checkpoint = Path::new("/home/sagan/.cache/bevy-gym-quality-validation/drone-travel-continuation-research-20261011/run/travel-440.mpk");
    let bytes = fs::read(checkpoint)?;
    let digest = Sha256::digest(&bytes);
    let sha256 = format!("{digest:x}");
    if sha256 != "3e6964d0f1475081b2572387f2193e070fd396800023639074ca073e69b54c88" { return Err("selected travel checkpoint hash mismatch".into()); }
    let policy = RecurrentPpoPolicy::load_bytes(bytes,15,15,1,&[0.0;4],&[1.0;4],&learning::learning_config())?;
    let before = policy.to_bytes()?;
    let seeds: [u64;32] = std::array::from_fn(|index|u64::MAX-index as u64-1);
    let episodes = seeds.iter().map(|seed|evaluate(&policy,*seed)).collect::<Result<Vec<_>,_>>()?;
    assert_eq!(policy.to_bytes()?,before);
    let passed = episodes.iter().all(|episode|episode.passed);
    let report = serde_json::json!({"schema":"drone-travel-held-out-research-v1","checkpoint_sha256":sha256,"seed_partition":"existing robot convention: MAX-1 through MAX-32; all high-bit roots excluded from training; selection MAX excluded","model_selection":"highest original return among the two passing selection checkpoints: update440; held-out results do not choose weights","episodes":episodes,"passed":passed,"parameter_bytes_unchanged":true,"qualification":"native held-out only; browser and inference scene still required"});
    fs::write("/home/sagan/.cache/bevy-gym-quality-validation/drone-travel-continuation-research-20261011/held-out.json",serde_json::to_vec_pretty(&report)?)?;
    println!("held-out passes: {} / 32",episodes.iter().filter(|episode|episode.passed).count());
    if passed {Ok(())} else {Err("travel held-out gates failed".into())}
}
