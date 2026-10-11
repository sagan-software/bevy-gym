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
    let output = Path::new("/home/sagan/.cache/bevy-gym-quality-validation/drone-travel-research-20261011/run");
    let config = learning::learning_config();
    let mut agent = RecurrentPpoAgent::new(15, 15, 1, &[0.0; 4], &[1.0; 4], config.clone(), SeedConfig::from_root(157))?;
    agent.reduce_action_mean_head();
    fs::create_dir(output)?;
    fs::write(output.join("initialization.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"drone-travel-research-initialization-v1", "seed":157, "budget_updates":1001,
        "initialization":"fresh seeded PPO network; new 15-input architecture; no weights transferred", "mean_head_scale":0.01,
        "physics":"original DroneTravel/DroneHover disturbed dynamics, reward, motor bounds and 20 ms interval",
        "observation":"original 12 body-frame destination/flight features plus three body-frame heading components",
        "tasks":"reset-only radius [1,3) m at Y=2 m, independent uniform world-XZ destination and heading angles [-pi,pi)",
        "evaluation":"five fixed selection roots, 500 actions, all final 100 actions distance<=0.3 m, speed<=0.3 m/s, upright>=0.95, angular speed<=0.5 rad/s, heading error<=15 degrees",
        "selection":"highest original return among all-five passes; earliest update breaks ties; held-out roots excluded",
        "qualification":"research only; budget completion never promotes"
    }))?)?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut batch = learning::RecoveryBatch::with_task(157, &agent.policy(), TravelTask::new, encode, NonZeroU16::new(500).expect("positive limit"));
    for update in 1_u32..=1001 {
        let sequences = batch.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        assert_eq!(metrics.valid_samples, 512);
        assert!(metrics.optimizer_updates > 0);
        writeln!(journal, "{}", serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_updates":metrics.optimizer_updates,"optimizer_steps":metrics.optimizer_steps,"actor_loss":metrics.actor_loss,"critic_loss":metrics.critic_loss,"approximate_kl":metrics.approximate_kl,"entropy":metrics.entropy}))?;
        journal.flush()?;
        if update % 20 != 0 && update != 1001 { continue; }
        let bytes = agent.policy().to_bytes()?;
        let digest = Sha256::digest(&bytes);
        let policy = RecurrentPpoPolicy::load_bytes(bytes.clone(), 15, 15, 1, &[0.0; 4], &[1.0; 4], &config)?;
        let episodes = learning::SELECTION_SEEDS.iter().map(|seed| evaluate(&policy, *seed)).collect::<Result<Vec<_>, _>>()?;
        let passed = episodes.iter().all(|episode| episode.passed);
        let report = serde_json::json!({"schema":"drone-travel-research-checkpoint-v1","seed":157,"update":update,"transitions":512_u64*u64::from(update),"optimizer_steps":metrics.optimizer_steps,"algorithm":"PPO","sha256":format!("{digest:x}"),"actor_features":15,"critic_features":15,"motors":4,"episodes":episodes,"passed":passed,"qualification":"unqualified research; requires disjoint held-out tests and browser validation"});
        fs::write(output.join(format!("travel-{update}.mpk")), bytes)?;
        fs::write(output.join(format!("travel-{update}.research.json")), serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
    }
    Err("Research budget completed without checkpoint promotion".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Reset-only tasks preserve exact original observations and torques.
    #[test]
    fn task_reset_and_encoding_preserve_original_physics() {
        let mut task = TravelTask::new();
        let reset = task.reset(Some(42));
        let mut reference = DroneTravel::new(TravelTask::destination(42));
        assert_eq!(reset, reference.reset(Some(42)));
        let encoded = encode(&reset.observation);
        assert_eq!(&encoded[..12], &learning::encoding::encode_target(reset.observation.body(), reset.observation.destination().position()));
        assert!(encoded.iter().all(|value| value.is_finite()));
        for fraction in [0.0, 0.5, 1.0] {
            let action = DroneAction::try_from([fraction;4]).expect("isolated parity fixture");
            assert_eq!(task.step(action), reference.step(action));
        }
        assert_eq!(task.reset(Some(42)), reset);
    }
    /// Desired headings stay unit-length after the body-frame transformation.
    #[test]
    fn heading_features_retain_unit_direction() {
        let mut task = TravelTask::new();
        for seed in [0,1,2,42,u64::MAX] {
            let observation = task.reset(Some(seed)).observation;
            let encoded = encode(&observation);
            let (_, heading) = encoded.split_at(12);
            let squared_length = heading.iter().map(|component| component * component).sum::<f32>();
            assert!((squared_length - 1.0).abs() < 1e-5);
        }
    }
}
