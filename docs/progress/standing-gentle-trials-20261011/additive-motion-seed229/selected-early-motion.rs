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

/// Inspect learned inference and initialization fixtures without qualification.
fn main() -> Result<(), Box<dyn Error>> {
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
    let bytes = tokio::runtime::Builder::new_current_thread().build()?.block_on(tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-additive-motion-probe-20261011/run/standing-60.mpk"))?;
    let trained =
        RecurrentPpoPolicy::load_bytes(bytes, 204, 204, 1, &[-1.0; 26], &[1.0; 26], &config)?;
    for (label, policy) in [("trained-update60", trained)] {
        for seed in standing::evaluation::SELECTION_SEEDS {
            let mut env = bevy_gym::robots::DroidStanding::default();
            let mut observation = bevy_gym::Env::reset(&mut env, Some(seed)).observation;
            let mut memory = policy.initial_memory();
            for step in 1..=100 {
                let decision =
                    policy.mean_action(&standing::encoding::encode(&observation), &memory)?;
                let action = standing::encoding::decode(&decision.action)?;
                let state = bevy_gym::Env::step(&mut env, action);
                let speeds = bevy_gym::robots::DroidBody::ALL
                    .map(|body| state.observation.body(body).angular_velocity().length());
                let record = serde_json::json!({"label":label,"seed":seed,"step":step,"action":decision.action,"angular_speed_rad_s":speeds,"done":state.status.is_done(),"qualification":false});
                println!("{record}");
                observation = state.observation;
                memory = decision.next_memory;
                if state.status.is_done() {
                    break;
                }
            }
        }
    }
    Ok(())
}
