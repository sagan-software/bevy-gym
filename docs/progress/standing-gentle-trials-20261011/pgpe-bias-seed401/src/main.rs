//! Bounded research trial; no published example or qualification claim.
#![expect(
    dead_code,
    unused_imports,
    reason = "The research harness reuses shared modules without their CLI and viewer entry points."
)]
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/learning/mod.rs"]
#[expect(
    clippy::type_complexity,
    reason = "The imported collector retains its owning repository's allowed function-pointer contract."
)]
mod learning;
mod safety;
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/standing/mod.rs"]
#[expect(
    clippy::type_complexity,
    reason = "The imported standing collector retains its owning repository allowed function-pointer contract."
)]
mod standing;

mod gradient;
use std::{error::Error, fs, io::Write, path::Path};

use bevy_gym::{
    training::{RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoPolicy, SeedConfig, SplitMix64},
    Env, TimeLimit,
};
use sha2::{Digest, Sha256};

/// Paired exploration learns neural parameters; no programmed actuator request is supplied.
fn main() -> Result<(), Box<dyn Error>> {
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(run())
}

/// Await artifact I/O between completed physical trajectory batches.
async fn run() -> Result<(), Box<dyn Error>> {
    let output = Path::new(
        "/home/sagan/.cache/bevy-gym-quality-validation/standing-pgpe-bias-probe-20261011/run",
    );
    let parent = load_parent().await?;
    let logs: Vec<f32> = serde_json::from_slice(include_bytes!("../initial-log-std.json"))?;
    if logs.len() != 26
        || logs
            .iter()
            .any(|value| !value.is_finite() || !(-9.0..=1.0).contains(value))
    {
        return Err("invalid PGPE calibration profile".into());
    }
    let sigma: [f64; 26] = std::array::from_fn(|index| 0.5 * f64::from(logs[index]).exp());
    let mut mean = [0.0_f64; 26];
    let seeds = SeedConfig::from_root(401);
    let mut random = SplitMix64::new(seeds.action);
    tokio::fs::create_dir(output).await?;
    tokio::fs::write(output.join("parent.mpk"), &parent).await?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut transitions = 0_u64;
    for generation in 1_u64..=64 {
        // Both signs share four training roots, disjoint from selection and held-out roots.
        let roots = std::array::from_fn::<_, 4, _>(|lane| {
            (seeds.environment_episode(lane, generation) & (u64::MAX >> 1)) | (1 << 10)
        });
        let mut pairs = Vec::with_capacity(32);
        for _ in 0..32 {
            let noise = std::array::from_fn(|index| sigma[index] * random.standard_normal_f64());
            let positive = std::array::from_fn(|index| mean[index] + noise[index]);
            let negative = std::array::from_fn(|index| mean[index] - noise[index]);
            let positive = score(&candidate(&parent, &positive)?, roots)?;
            let negative = score(&candidate(&parent, &negative)?, roots)?;
            transitions += positive.actions + negative.actions;
            pairs.push(gradient::Pair {
                noise,
                positive: positive.reward,
                negative: negative.reward,
            });
        }
        mean = gradient::update_mean(&mean, &pairs, 0.2)?;
        let policy = candidate(&parent, &mean)?;
        let bytes = policy.to_bytes()?;
        let digest = Sha256::digest(&bytes);
        let episodes =
            standing::evaluation::evaluate(&policy, &standing::evaluation::SELECTION_SEEDS)?;
        let control = standing::control::evaluate(&policy, &standing::evaluation::SELECTION_SEEDS)?;
        let passed =
            standing::evaluation::passes(&episodes, &standing::evaluation::SELECTION_SEEDS)
                && standing::control::passes(&control, &standing::evaluation::SELECTION_SEEDS);
        let report = serde_json::json!({"algorithm":"PPO parent plus fixed-variance symmetric PGPE over26 mean-head biases","seed":401,"generation":generation,"transitions":transitions,"mean_offset":mean,"parameter_sigma":sigma,"training_roots":roots,"episodes":episodes,"control_episodes":control,"passed":passed,"sha256":format!("{digest:x}"),"qualification":false,"promotion":"none; complete early-motion, held-out and native/WASM qualification still required"});
        tokio::fs::write(output.join(format!("standing-{generation}.mpk")), bytes).await?;
        tokio::fs::write(
            output.join(format!("standing-{generation}.evaluation.json")),
            serde_json::to_vec_pretty(&report)?,
        )
        .await?;
        writeln!(journal, "{report}")?;
        journal.flush()?;
        println!("{report}");
    }
    Err("PGPE research budget finished without policy promotion".into())
}

/// Read and verify the frozen parent within the caller's existing asynchronous runtime.
async fn load_parent() -> Result<Vec<u8>, Box<dyn Error>> {
    let parent = tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-head-scaled-motion-probe-20261011/run/standing-23480.mpk").await?;
    let digest = Sha256::digest(&parent);
    if format!("{digest:x}") != "73b1a1096d2a0dd9cac9d6f5ec6b3e7632ed71ef914a03e2d60ee3aa284006e5" {
        return Err("PGPE parent hash mismatch".into());
    }
    Ok(parent)
}

/// Load retained PPO weights and apply only finite learned bias offsets before inference.
fn candidate(parent: &[u8], offset: &[f64; 26]) -> Result<RecurrentPpoPolicy, Box<dyn Error>> {
    let offsets = offset.map(|value| value as f32);
    if offsets.iter().any(|value| !value.is_finite()) {
        return Err("PGPE bias exceeds finite actor precision".into());
    }
    let config = RecurrentPpoConfig {
        log_std_min: -9.0,
        ..Default::default()
    };
    let mut agent = RecurrentPpoAgent::load_bytes(
        parent.to_vec(),
        204,
        204,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        config,
        SeedConfig::from_root(401),
    )?;
    for (dimension, offset) in offsets.into_iter().enumerate() {
        // Zero offsets retain every existing parameter bit, including signed zero.
        if offset != 0.0 {
            agent.shift_actor_mean_bias(dimension, offset)?;
        }
    }
    Ok(agent.policy())
}

/// One actual discounted return and its physical action count.
#[derive(Clone, Copy)]
struct Score {
    /// Dense reward units with gamma0.995 per20ms decision boundary.
    reward: f64,
    /// Applied policy actions, capped at1000 in each episode.
    actions: u64,
}

/// Join all four independent physical episodes before consuming any error.
fn score(policy: &RecurrentPpoPolicy, roots: [u64; 4]) -> Result<Score, Box<dyn Error>> {
    let results = std::thread::scope(|scope| {
        let handles = roots.map(|seed| scope.spawn(move || episode(policy, seed)));
        handles.map(|handle| handle.join())
    });
    let mut score = Score {
        reward: 0.0,
        actions: 0,
    };
    for result in results {
        let result = result.map_err(|_| "PGPE physical rollout worker panicked")?;
        let result = result.map_err(|error| {
            let error: Box<dyn Error> = error;
            error
        })?;
        score.reward += result.reward / 4.0;
        score.actions += result.actions;
    }
    Ok(score)
}

/// Every living action is the sampled neural policy's deterministic inference on its own state.
fn episode(policy: &RecurrentPpoPolicy, seed: u64) -> Result<Score, Box<dyn Error + Send + Sync>> {
    let mut environment = TimeLimit::new(safety::ControlTask::default(), 1000)?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut discount = 1.0;
    let mut score = Score {
        reward: 0.0,
        actions: 0,
    };
    loop {
        let decision = policy.mean_action(&standing::encoding::encode(&observation), &memory)?;
        let step = environment.step(standing::encoding::decode(&decision.action)?);
        score.reward += discount * step.reward;
        score.actions += 1;
        if !score.reward.is_finite() || score.reward > 201.0 {
            return Err("PGPE episode violates the finite return bound".into());
        }
        if step.status.is_done() {
            return Ok(score);
        }
        discount *= f64::from(learning::GAMMA);
        observation = step.observation;
        memory = decision.next_memory;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Launcher initialization must not construct a nested runtime inside its async task.
    #[test]
    fn parent_loading_uses_the_existing_runtime() -> Result<(), Box<dyn Error>> {
        let parent = tokio::runtime::Builder::new_current_thread()
            .build()?
            .block_on(load_parent())?;
        assert!(!parent.is_empty());
        Ok(())
    }
    /// Mean zero retains the entire trained parent rather than installing a controller.
    #[test]
    fn zero_offset_retains_parent_and_invalid_offset_fails_before_inference(
    ) -> Result<(), Box<dyn Error>> {
        let parent=tokio::runtime::Builder::new_current_thread().build()?.block_on(tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-head-scaled-motion-probe-20261011/run/standing-23480.mpk"))?;
        let policy = candidate(&parent, &[0.0; 26])?;
        assert_eq!(policy.to_bytes()?, parent);
        assert!(candidate(&parent, &[f64::MAX; 26]).is_err());
        assert!(candidate(b"corrupt checkpoint", &[0.0; 26]).is_err());
        let score = score(&policy, [1024, 1025, 1026, 1027])?;
        assert!(score.reward.is_finite() && score.reward <= 201.0);
        assert!((4..=4000).contains(&score.actions));
        Ok(())
    }
}
