//! Train recovery from disturbed starts, then compare held-out episodes.

mod learning;

use std::{error::Error, num::NonZeroU32, path::PathBuf};

use clap::Parser;
use learning::{baseline, evaluate, load_policy, new_agent, RecoveryBatch, SELECTION_SEEDS};

/// Reproducible training budget and output directory for this lesson.
#[derive(Parser)]
struct Options {
    /// Number of 512-transition PPO updates.
    #[arg(long, default_value = "260")]
    updates: NonZeroU32,
    /// Root seed for model initialization, training resets, and action sampling.
    #[arg(long, default_value = "7")]
    seed: u64,
    /// Directory for checkpoints and JSON evaluation records.
    #[arg(long, default_value = "runs/drone-recovery")]
    output: PathBuf,
}

/// Collect observations, improve the policy, and inspect separate evaluation episodes.
#[expect(
    clippy::disallowed_methods,
    reason = "This synchronous guide writes local checkpoints without an async runtime."
)]
fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    std::fs::create_dir_all(&options.output)?;
    let mut agent = new_agent(options.seed)?;
    let mut batch = RecoveryBatch::new(options.seed, &agent.policy());
    let initial = serde_json::json!({
        "seed": options.seed, "updates": options.updates.get(), "steps": 0,
        "baseline": baseline(&SELECTION_SEEDS)?,
        "episodes": evaluate(&agent.policy(), &SELECTION_SEEDS)?,
    });
    std::fs::write(options.output.join("0.json"), initial.to_string())?;
    println!("{initial}");

    for update in 1..=options.updates.get() {
        let sequences = batch.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        if update == 1 || update % 20 == 0 || update == options.updates.get() {
            let policy = agent.policy();
            let episodes = evaluate(&policy, &SELECTION_SEEDS)?;
            let steps = u64::from(update) * metrics.valid_samples;
            let record = serde_json::json!({"steps": steps, "episodes": episodes,
                "actor_loss": metrics.actor_loss, "critic_loss": metrics.critic_loss,
                "approximate_kl": metrics.approximate_kl, "entropy": metrics.entropy,
                "optimizer_steps": metrics.optimizer_steps});
            println!("{record}");
            std::fs::write(
                options.output.join(format!("{steps}.json")),
                record.to_string(),
            )?;
            std::fs::write(
                options.output.join(format!("{steps}.mpk")),
                policy.to_bytes()?,
            )?;
        }
    }
    let restored = load_policy(agent.policy().to_bytes()?)?;
    let final_episodes = evaluate(&restored, &SELECTION_SEEDS)?;
    let encoded = serde_json::to_string(&final_episodes)?;
    println!("{encoded}");
    Ok(())
}
