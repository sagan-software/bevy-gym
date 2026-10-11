//! Bounded research trial; no published example or qualification claim.
#![expect(dead_code, unused_imports, reason = "The research harness reuses shared modules without their CLI and viewer entry points.")]
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/learning/mod.rs"]
mod learning;
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/standing/mod.rs"]
mod standing;

use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoConfig, SeedConfig};
use std::{error::Error, fs, io::Write, num::{NonZeroU32, NonZeroU64}, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let output = Path::new("/home/sagan/.cache/bevy-gym-quality-validation/standing-warm-small-head-probe-20261011/run");
    let parent = Path::new("/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/docs/progress/standing-seed43-update100/checkpoint.mpk");
    let bytes = fs::read(parent)?;
    let metadata = fs::read(parent.with_extension("json"))?;
    let identity = standing::checkpoint::Record::for_bytes(&bytes, &metadata)?;
    let config = RecurrentPpoConfig {
        gamma: learning::GAMMA, gae_lambda: learning::GAE_LAMBDA,
        actor_learning_rate: 0.00003, critic_learning_rate: 0.001,
        initial_log_std: -5.0, entropy_coefficient: 0.0, epochs: 1,
        ..RecurrentPpoConfig::default()
    };
    let mut agent = RecurrentPpoAgent::load_bytes(bytes.clone(), 204, 204, 1, &[-1.0;26], &[1.0;26], config, SeedConfig::from_root(139))?;
    agent.reduce_action_mean_head();
    agent.reset_exploration();
    fs::create_dir(output)?;
    fs::write(output.join("parent.mpk"), &bytes)?;
    fs::write(output.join("parent.json"), &metadata)?;
    fs::write(output.join("initialization.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"standing-warm-small-head-research-v1", "seed":139, "budget_updates":1001,
        "parent":identity, "mean_head_scale":0.01, "reset_initial_log_std":-5.0,
        "actor_learning_rate":0.00003, "critic_learning_rate":0.001, "entropy_coefficient":0.0, "epochs":1,
        "reward":"droid-standing-posture-quiet-reward-v1", "qualification":"unqualified research; no promotion"
    }))?)?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut batch = standing::batch_with_recipe(139, &agent.policy(), standing::reward::Recipe::PostureQuietV1);
    for update in 1..=1001 {
        let sequences = batch.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        assert_eq!(metrics.valid_samples,512);
        assert!(metrics.optimizer_updates>0);
        writeln!(journal,"{}",serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_updates":metrics.optimizer_updates,"optimizer_steps":metrics.optimizer_steps,"actor_loss":metrics.actor_loss,"critic_loss":metrics.critic_loss,"approximate_kl":metrics.approximate_kl,"entropy":metrics.entropy}))?;
        journal.flush()?;
        if update % 20 != 0 && update != 1001 {continue;}
        let bytes = agent.policy().to_bytes()?;
        let record = standing::checkpoint::Record::new(139,NonZeroU32::new(update).expect("positive budget iteration"),NonZeroU64::new(metrics.optimizer_steps).ok_or("missing optimizer update")?,&bytes);
        let metadata = serde_json::to_vec_pretty(&record)?;
        let (policy, _) = standing::checkpoint::from_bytes(bytes.clone(),&metadata)?;
        let seeds = &standing::evaluation::SELECTION_SEEDS;
        let standing = standing::evaluation::evaluate(&policy,seeds)?;
        let control = standing::control::evaluate(&policy,seeds)?;
        let passed = standing::evaluation::passes(&standing,seeds) && standing::control::passes(&control,seeds);
        let report = serde_json::json!({"provenance":record,"mode":"selection","episodes":standing,"control_episodes":control,"passed":passed,"qualification":"not qualified; requires held-out evaluation"});
        fs::write(output.join(format!("standing-{update}.mpk")),bytes)?;
        fs::write(output.join(format!("standing-{update}.json")),metadata)?;
        fs::write(output.join(format!("standing-{update}.evaluation.json")),serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
    }
    Err("Research budget finished without policy promotion".into())
}
