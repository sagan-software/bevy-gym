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

use std::{
    error::Error,
    fs,
    io::Write,
    num::{NonZeroU32, NonZeroU64},
    path::Path,
};

use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoPolicy, SeedConfig};
use sha2::{Digest, Sha256};

/// Execute the bounded research task without publishing a candidate.
fn main() -> Result<(), Box<dyn Error>> {
    tokio::runtime::Builder::new_current_thread()
        .build()?
        .block_on(run())
}

/// Keep physical rollout and PPO boundaries intact between awaited artifact writes.
async fn run() -> Result<(), Box<dyn Error>> {
    let output = Path::new(
        "/home/sagan/.cache/bevy-gym-quality-validation/standing-calibrated-additive-motion-probe-20261011/run",
    );
    let config = RecurrentPpoConfig {
        gamma: learning::GAMMA,
        gae_lambda: learning::GAE_LAMBDA,
        actor_learning_rate: 0.000003,
        critic_learning_rate: 0.001,
        initial_log_std: -5.0,
        log_std_min: -9.0,
        entropy_coefficient: 0.0,
        epochs: 1,
        minibatch_sequences: 512,
        ..RecurrentPpoConfig::default()
    };
    let parent = tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-calibrated-exploration-probe-20261011/run/standing-960.mpk").await?;
    let digest = Sha256::digest(&parent);
    if format!("{digest:x}") != "2be83022bd828e62e66593cf7cb812dcd7e7e65fed0e59aa9865ee6af226df3b" {
        return Err("standing parent hash mismatch".into());
    }
    let mut agent = RecurrentPpoAgent::load_bytes(
        parent.clone(),
        204,
        204,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        config.clone(),
        SeedConfig::from_root(283),
    )?;
    let scales: Vec<f32> = serde_json::from_slice(include_bytes!("../initial-log-std.json"))?;
    agent.reset_exploration_with_log_std(&scales)?;
    tokio::fs::create_dir(output).await?;
    tokio::fs::write(output.join("parent.mpk"), parent).await?;
    tokio::fs::write(
        output.join("initialization.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"standing-calibrated-additive-motion-research-v1", "seed":283, "budget_updates":10001,
            "initialization":"retain calibrated seed211 update960 actor and critic; reset calibrated exploration; fresh Adam states", "mean_head_scale":1.0, "exploration":"calibrated per-channel reset", "log_std_min":-9.0, "log_std_max":1.0,
            "actor_learning_rate":0.000003, "critic_learning_rate":0.001, "entropy_coefficient":0.0, "epochs":1, "minibatch_sequences":512,
            "reward":"posture-height minus 0.1 times squared maximum segment speed divided by squared 8 rad/s; original physical termination and selection gates unchanged", "qualification":"unqualified research; no promotion"
        }))?,
    ).await?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut batch = safe_batch(283, &agent.policy());
    for update in 1_u32..=10001 {
        let sequences = batch.collect_parallel(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        assert_eq!(metrics.valid_samples, 512);
        assert_eq!(metrics.optimizer_updates, 1);
        let optimization = serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_updates":metrics.optimizer_updates,"optimizer_steps":metrics.optimizer_steps,"actor_loss":metrics.actor_loss,"critic_loss":metrics.critic_loss,"approximate_kl":metrics.approximate_kl,"entropy":metrics.entropy});
        writeln!(journal, "{optimization}")?;
        journal.flush()?;
        if update % 20 != 0 && update != 10001 {
            continue;
        }
        let bytes = agent.policy().to_bytes()?;
        let digest = Sha256::digest(&bytes);
        let record = serde_json::json!({
            "schema":"droid-standing-calibrated-additive-motion-research-v1", "algorithm":"PPO", "seed":283,
            "update":update, "transitions":512_u64*u64::from(update), "optimizer_steps":metrics.optimizer_steps,
            "sha256":format!("{digest:x}"), "actor_features":204, "critic_features":204,
            "torque_channels":26, "log_std_min":-9.0, "log_std_max":1.0,
            "qualification":"unqualified research; not accepted by the published standing loader"
        });
        let metadata = serde_json::to_vec_pretty(&record)?;
        let policy = RecurrentPpoPolicy::load_bytes(
            bytes.clone(),
            204,
            204,
            1,
            &[-1.0; 26],
            &[1.0; 26],
            &config,
        )?;
        let seeds = &standing::evaluation::SELECTION_SEEDS;
        let standing = standing::evaluation::evaluate(&policy, seeds)?;
        let control = standing::control::evaluate(&policy, seeds)?;
        let passed = standing::evaluation::passes(&standing, seeds)
            && standing::control::passes(&control, seeds);
        let report = serde_json::json!({"provenance":record,"mode":"selection","episodes":standing,"control_episodes":control,"passed":passed,"qualification":"not qualified; requires held-out evaluation"});
        tokio::fs::write(output.join(format!("standing-{update}.mpk")), bytes).await?;
        tokio::fs::write(
            output.join(format!("standing-{update}.research.json")),
            metadata,
        )
        .await?;
        tokio::fs::write(
            output.join(format!("standing-{update}.evaluation.json")),
            serde_json::to_vec_pretty(&report)?,
        )
        .await?;
        println!("{report}");
    }
    Err("Research budget finished without policy promotion".into())
}

fn safe_batch(
    seed: u64,
    policy: &RecurrentPpoPolicy,
) -> standing::parallel_rollout::ParallelRecoveryBatch<safety::ControlTask, 204> {
    standing::parallel_rollout::ParallelRecoveryBatch::with_parallel_actions(
        seed,
        policy,
        safety::ControlTask::default,
        standing::encoding::encode,
        |values| Ok(standing::encoding::decode(values)?),
        std::num::NonZeroU16::new(1000).expect("fixed standing horizon"),
    )
}

#[cfg(test)]
mod batching_tests {
    use super::*;
    /// Optimizer grouping cannot change the first physical rollout.
    #[test]
    fn rollout_is_identical_but_one_update_covers_all_sequences() {
        let old = RecurrentPpoConfig {
            gamma: learning::GAMMA,
            gae_lambda: learning::GAE_LAMBDA,
            actor_learning_rate: 0.0003,
            critic_learning_rate: 0.001,
            initial_log_std: -5.0,
            log_std_min: -9.0,
            entropy_coefficient: 0.0,
            epochs: 4,
            ..Default::default()
        };
        let new = RecurrentPpoConfig {
            actor_learning_rate: 0.000003,
            epochs: 1,
            minibatch_sequences: 512,
            ..old.clone()
        };
        let create = |config| {
            let mut a = RecurrentPpoAgent::new(
                204,
                204,
                1,
                &[-1.0; 26],
                &[1.0; 26],
                config,
                SeedConfig::from_root(211),
            )
            .expect("valid research actor");
            a.reduce_action_mean_head();
            a.reset_exploration();
            a
        };
        let a = create(old);
        let mut b = create(new);
        let bytes = b.policy().to_bytes().expect("initial record");
        let first = safe_batch(211, &a.policy())
            .collect_parallel(&a.policy())
            .expect("first physical rollout");
        let second = safe_batch(211, &b.policy())
            .collect_parallel(&b.policy())
            .expect("same physical rollout");
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        let update = b.update(&second).expect("one full-batch PPO step");
        assert_eq!(update.valid_samples, 512);
        assert_eq!(update.optimizer_updates, 1);
        assert_eq!(update.optimizer_steps, 1);
        assert_ne!(bytes, b.policy().to_bytes().expect("updated record"));
    }
    /// Reload the actual parent without resetting its trained exploration or mean.
    #[test]
    fn actual_parent_reloads_and_continues_one_full_batch() {
        let bytes=tokio::runtime::Builder::new_current_thread().build().expect("test runtime").block_on(tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-calibrated-exploration-probe-20261011/run/standing-960.mpk")).expect("frozen research parent");
        let config = RecurrentPpoConfig {
            gamma: learning::GAMMA,
            gae_lambda: learning::GAE_LAMBDA,
            actor_learning_rate: 0.000003,
            critic_learning_rate: 0.001,
            initial_log_std: -5.0,
            log_std_min: -9.0,
            entropy_coefficient: 0.0,
            epochs: 1,
            minibatch_sequences: 512,
            ..Default::default()
        };
        let reference = RecurrentPpoPolicy::load_bytes(
            bytes.clone(),
            204,
            204,
            1,
            &[-1.0; 26],
            &[1.0; 26],
            &config,
        )
        .expect("frozen reference");
        let mut agent = RecurrentPpoAgent::load_bytes(
            bytes,
            204,
            204,
            1,
            &[-1.0; 26],
            &[1.0; 26],
            config,
            SeedConfig::from_root(283),
        )
        .expect("continuation");
        let mut environment = bevy_gym::robots::DroidStanding::default();
        let observation = bevy_gym::Env::reset(&mut environment, Some(42)).observation;
        let features = standing::encoding::encode(&observation);
        let memory = reference.initial_memory();
        let before = reference
            .mean_action(&features, &memory)
            .expect("reference decision");
        let restored = agent
            .policy()
            .mean_action(&features, &memory)
            .expect("retained decision");
        assert_eq!(before.action, restored.action);
        assert_eq!(before.next_memory, restored.next_memory);
        assert_eq!(
            before.log_probability.to_bits(),
            restored.log_probability.to_bits()
        );
        let rollout = safe_batch(283, &agent.policy())
            .collect_parallel(&agent.policy())
            .expect("actual parent rollout");
        let metrics = agent.update(&rollout).expect("actual PPO update");
        assert_eq!(metrics.valid_samples, 512);
        assert_eq!(metrics.optimizer_updates, 1);
        assert_eq!(metrics.optimizer_steps, 1);
        let after = agent
            .policy()
            .mean_action(&features, &memory)
            .expect("trained decision");
        assert_ne!(before.action, after.action);
    }
}
