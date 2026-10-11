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
        "/home/sagan/.cache/bevy-gym-quality-validation/standing-safe-episode-probe-20261011/run",
    );
    let config = RecurrentPpoConfig {
        gamma: learning::GAMMA,
        gae_lambda: learning::GAE_LAMBDA,
        actor_learning_rate: 0.0003,
        critic_learning_rate: 0.001,
        initial_log_std: -5.0,
        log_std_min: -5.0,
        entropy_coefficient: 0.0,
        epochs: 4,
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
    agent.reset_exploration();
    tokio::fs::create_dir(output).await?;
    tokio::fs::write(
        output.join("initialization.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":"standing-safe-episode-research-v1", "seed":211, "budget_updates":501,
            "initialization":"fresh random network before mean-head scale", "mean_head_scale":0.01, "reset_initial_log_std":-5.0, "log_std_min":-5.0, "log_std_max":1.0,
            "actor_learning_rate":0.0003, "critic_learning_rate":0.001, "entropy_coefficient":0.0, "epochs":4,
            "reward":"posture-height minus 0.1 mean squared torque fraction; any segment above 8 rad/s terminates training with reward -1; original selection gates unchanged", "qualification":"unqualified research; no promotion"
        }))?,
    ).await?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut batch = safe_batch(211, &agent.policy());
    for update in 1_u32..=501 {
        let sequences = batch.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        assert_eq!(metrics.valid_samples, 512);
        assert!(metrics.optimizer_updates > 0);
        writeln!(
            journal,
            "{}",
            serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_updates":metrics.optimizer_updates,"optimizer_steps":metrics.optimizer_steps,"actor_loss":metrics.actor_loss,"critic_loss":metrics.critic_loss,"approximate_kl":metrics.approximate_kl,"entropy":metrics.entropy})
        )?;
        journal.flush()?;
        if update % 20 != 0 && update != 501 {
            continue;
        }
        let bytes = agent.policy().to_bytes()?;
        let digest = Sha256::digest(&bytes);
        let record = serde_json::json!({
            "schema":"droid-standing-safe-episode-research-v1", "algorithm":"PPO", "seed":211,
            "update":update, "transitions":512_u64*u64::from(update), "optimizer_steps":metrics.optimizer_steps,
            "sha256":format!("{digest:x}"), "actor_features":204, "critic_features":204,
            "torque_channels":26, "log_std_min":-5.0, "log_std_max":1.0,
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
