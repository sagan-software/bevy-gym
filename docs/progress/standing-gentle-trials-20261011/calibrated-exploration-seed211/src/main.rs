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
        "/home/sagan/.cache/bevy-gym-quality-validation/standing-calibrated-exploration-probe-20261011/run",
    );
    let config = RecurrentPpoConfig {
        gamma: learning::GAMMA,
        gae_lambda: learning::GAE_LAMBDA,
        actor_learning_rate: 0.00003,
        critic_learning_rate: 0.001,
        initial_log_std: -5.0,
        log_std_min: -9.0,
        entropy_coefficient: 0.0,
        epochs: 1,
        minibatch_sequences: 512,
        ..RecurrentPpoConfig::default()
    };
    let mut agent = RecurrentPpoAgent::new(
        204,
        204,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        config.clone(),
        SeedConfig::from_root(211),
    )?;
    agent.reduce_action_mean_head();
    let scale_bytes = tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-calibrated-exploration-probe-20261011/initial-log-std.json").await?;
    let digest = Sha256::digest(&scale_bytes);
    if format!("{digest:x}") != "4fb1a75112bb0b67137bd37e258b188544b02e8bc9a4162f05e9dd2fb50ae78a" {
        return Err("calibrated exploration input hash mismatch".into());
    }
    let scales: Vec<f32> = serde_json::from_slice(&scale_bytes)?;
    agent.reset_exploration_with_log_std(&scales)?;
    tokio::fs::create_dir(output).await?;
    tokio::fs::write(
        output.join("initialization.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"standing-calibrated-exploration-research-v1", "seed":211, "budget_updates":1001,
            "initialization":"fresh random network before mean-head scale", "mean_head_scale":0.01, "configured_initial_log_std":-5.0, "reset_log_std_per_action":scales, "log_std_min":-9.0, "log_std_max":1.0,
            "actor_learning_rate":0.00003, "critic_learning_rate":0.001, "entropy_coefficient":0.0, "epochs":1, "minibatch_sequences":512,
            "reward":"posture-height minus 0.1 mean squared torque fraction; any segment above 8 rad/s terminates training with reward -1; original selection gates unchanged", "qualification":"unqualified research; no promotion"
        }))?,
    ).await?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut batch = safe_batch(211, &agent.policy());
    for update in 1_u32..=1001 {
        let sequences = batch.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        assert_eq!(metrics.valid_samples, 512);
        assert_eq!(metrics.optimizer_updates, 1);
        let optimization = serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_updates":metrics.optimizer_updates,"optimizer_steps":metrics.optimizer_steps,"actor_loss":metrics.actor_loss,"critic_loss":metrics.critic_loss,"approximate_kl":metrics.approximate_kl,"entropy":metrics.entropy});
        writeln!(journal, "{optimization}")?;
        journal.flush()?;
        if update % 20 != 0 && update != 1001 {
            continue;
        }
        let bytes = agent.policy().to_bytes()?;
        let digest = Sha256::digest(&bytes);
        let record = serde_json::json!({
            "schema":"droid-standing-calibrated-exploration-research-v1", "algorithm":"PPO", "seed":211,
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
) -> learning::RecoveryBatch<safety::ControlTask, 204> {
    learning::RecoveryBatch::with_actions(
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
    /// Calibrated exploration retains all samples in one optimizer step.
    #[test]
    fn calibrated_rollout_is_repeatable_and_one_update_covers_all_sequences() {
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
            actor_learning_rate: 0.00003,
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
            let scales: Vec<f32> =
                serde_json::from_slice(include_bytes!("../initial-log-std.json"))
                    .expect("frozen exploration scales");
            a.reset_exploration_with_log_std(&scales)
                .expect("valid calibrated exploration");
            a
        };
        let a = create(old);
        let mut b = create(new);
        let bytes = b.policy().to_bytes().expect("initial record");
        let first = safe_batch(211, &a.policy())
            .collect(&a.policy())
            .expect("first physical rollout");
        let second = safe_batch(211, &b.policy())
            .collect(&b.policy())
            .expect("same physical rollout");
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
        let update = b.update(&second).expect("one full-batch PPO step");
        assert_eq!(update.valid_samples, 512);
        assert_eq!(update.optimizer_updates, 1);
        assert_eq!(update.optimizer_steps, 1);
        assert_ne!(bytes, b.policy().to_bytes().expect("updated record"));
    }
}
