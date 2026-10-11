//! Bounded research trial; no published example or qualification claim.
#![expect(dead_code, unused_imports, reason = "The research harness reuses shared modules without their CLI and viewer entry points.")]
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/learning/mod.rs"]
mod learning;
#[path = "/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/examples/robots/standing/mod.rs"]
mod standing;

use bevy_gym::training::{RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoPolicy, SeedConfig};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Write, num::{NonZeroU32, NonZeroU64}, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let output = Path::new("/home/sagan/.cache/bevy-gym-quality-validation/standing-critic-warmup-small-step-probe-20261011/run");
    let parent = Path::new("/home/sagan/Code/github.com/sagan-software/bevy-gym-quality/docs/progress/standing-seed43-update100/checkpoint.mpk");
    let bytes = fs::read(parent)?;
    let metadata = fs::read(parent.with_extension("json"))?;
    let identity = standing::checkpoint::Record::for_bytes(&bytes, &metadata)?;
    let config = RecurrentPpoConfig {
        gamma: learning::GAMMA, gae_lambda: learning::GAE_LAMBDA,
        actor_learning_rate: 0.000001, critic_learning_rate: 0.001,
        initial_log_std: -7.0, log_std_min: -9.0, entropy_coefficient: 0.0, epochs: 1,
        ..RecurrentPpoConfig::default()
    };
    let digest = Sha256::digest(&bytes);
    if format!("{digest:x}") != "657f1cbe1fad9cf0782ceac76a51490cdca93b6d51c54366faa26be7ac264b6d" { return Err("standing parent hash mismatch".into()); }
    let mut agent = RecurrentPpoAgent::load_actor_with_fresh_critic(parent, 204, 204, 1, &[-1.0;26], &[1.0;26], config.clone(), SeedConfig::from_root(193))?;
    agent.reset_exploration();
    fs::create_dir(output)?;
    fs::write(output.join("parent.mpk"), &bytes)?;
    fs::write(output.join("parent.json"), &metadata)?;
    fs::write(output.join("initialization.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"standing-critic-warmup-small-step-research-v1", "seed":193, "budget_updates":5001,
        "parent":identity, "critic":"fresh", "critic_only_warmup_updates":100, "critic_warmup_root":197, "optimizer":"fresh", "mean_head_scale":1.0, "reset_initial_log_std":-7.0, "log_std_min":-9.0, "log_std_max":1.0,
        "actor_learning_rate":0.000001, "critic_learning_rate":0.001, "entropy_coefficient":0.0, "epochs":1,
        "reward":"droid-standing-posture-quiet-reward-v1", "qualification":"unqualified research; no promotion"
    }))?)?;
    let before_warmup = agent.policy();
    let mut warmup = standing::batch_with_recipe(197, &before_warmup, standing::reward::Recipe::PostureQuietV1);
    let mut warmup_journal = fs::File::create(output.join("critic-warmup.jsonl"))?;
    for update in 1..=100 {
        let sequences = warmup.collect(&agent.policy())?;
        let metrics = agent.update_critic(&sequences)?;
        assert_eq!(metrics.valid_samples,512);
        writeln!(warmup_journal,"{}",serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_steps":metrics.optimizer_steps,"critic_loss":metrics.critic_loss}))?;
        warmup_journal.flush()?;
    }
    // Compare complete decisions over real physical trajectories before permitting actor updates.
    let after_warmup = agent.policy();
    use bevy_gym::Env;
    for seed in standing::evaluation::SELECTION_SEEDS {
        let mut environment = bevy_gym::robots::DroidStanding::default();
        let mut observation = environment.reset(Some(seed)).observation;
        let mut memory = before_warmup.initial_memory();
        for _ in 0..1000 {
            let features = standing::encoding::encode(&observation);
            let before = before_warmup.mean_action(&features,&memory)?;
            let after = after_warmup.mean_action(&features,&memory)?;
            assert_eq!(before,after);
            memory = before.next_memory;
            let result = environment.step(standing::encoding::decode(&before.action)?);
            observation = result.observation;
            if result.status.is_done(){break;}
        }
    }
    fs::write(output.join("warmup-actor-identity.json"),serde_json::to_vec_pretty(&serde_json::json!({"complete_actor_decisions_unchanged":true,"selection_roots":standing::evaluation::SELECTION_SEEDS,"maximum_actions_per_root":1000,"critic_only_updates":100,"warmup_transitions":51200}))?)?;
    let mut journal = fs::File::create(output.join("optimization.jsonl"))?;
    let mut batch = standing::batch_with_recipe(193, &agent.policy(), standing::reward::Recipe::PostureQuietV1);
    for update in 1_u32..=5001 {
        let sequences = batch.collect(&agent.policy())?;
        let metrics = agent.update(&sequences)?;
        assert_eq!(metrics.valid_samples,512);
        assert!(metrics.optimizer_updates>0);
        writeln!(journal,"{}",serde_json::json!({"update":update,"valid_samples":metrics.valid_samples,"optimizer_updates":metrics.optimizer_updates,"optimizer_steps":metrics.optimizer_steps,"actor_loss":metrics.actor_loss,"critic_loss":metrics.critic_loss,"approximate_kl":metrics.approximate_kl,"entropy":metrics.entropy}))?;
        journal.flush()?;
        if update % 20 != 0 && update != 5001 {continue;}
        let bytes = agent.policy().to_bytes()?;
        let digest = Sha256::digest(&bytes);
        let record = serde_json::json!({
            "schema":"droid-standing-critic-warmup-small-step-research-v1", "algorithm":"PPO", "seed":193,
            "update":update, "transitions":512_u64*u64::from(update), "optimizer_steps":metrics.optimizer_steps,
            "sha256":format!("{digest:x}"), "actor_features":204, "critic_features":204,
            "torque_channels":26, "log_std_min":-9.0, "log_std_max":1.0,
            "qualification":"unqualified research; not accepted by the published standing loader"
        });
        let metadata = serde_json::to_vec_pretty(&record)?;
        let policy = RecurrentPpoPolicy::load_bytes(bytes.clone(),204,204,1,&[-1.0;26],&[1.0;26],&config)?;
        let seeds = &standing::evaluation::SELECTION_SEEDS;
        let standing = standing::evaluation::evaluate(&policy,seeds)?;
        let control = standing::control::evaluate(&policy,seeds)?;
        let passed = standing::evaluation::passes(&standing,seeds) && standing::control::passes(&control,seeds);
        let report = serde_json::json!({"provenance":record,"mode":"selection","episodes":standing,"control_episodes":control,"passed":passed,"qualification":"not qualified; requires held-out evaluation"});
        fs::write(output.join(format!("standing-{update}.mpk")),bytes)?;
        fs::write(output.join(format!("standing-{update}.research.json")),metadata)?;
        fs::write(output.join(format!("standing-{update}.evaluation.json")),serde_json::to_vec_pretty(&report)?)?;
        println!("{report}");
    }
    Err("Research budget finished without policy promotion".into())
}
