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

/// Compare original serial collection and scoped threads on actual frozen RL inference.
fn main() -> Result<(), Box<dyn Error>> {
    let bytes=tokio::runtime::Builder::new_current_thread().build()?.block_on(tokio::fs::read("/home/sagan/.cache/bevy-gym-quality-validation/standing-calibrated-exploration-probe-20261011/run/standing-960.mpk"))?;
    let config = RecurrentPpoConfig {
        log_std_min: -9.0,
        ..Default::default()
    };
    let policy =
        RecurrentPpoPolicy::load_bytes(bytes, 204, 204, 1, &[-1.0; 26], &[1.0; 26], &config)?;
    let mut original = learning::RecoveryBatch::with_actions(
        227,
        &policy,
        safety::ControlTask::default,
        standing::encoding::encode,
        |values| Ok(standing::encoding::decode(values)?),
        std::num::NonZeroU16::new(1000).expect("original horizon"),
    );
    let mut parallel = standing::parallel_rollout::ParallelRecoveryBatch::with_parallel_actions(
        227,
        &policy,
        safety::ControlTask::default,
        standing::encoding::encode,
        |values| Ok(standing::encoding::decode(values)?),
        std::num::NonZeroU16::new(1000).expect("original horizon"),
    );
    for round in 0..6 {
        let start = std::time::Instant::now();
        let reference = original.collect(&policy)?;
        let serial_s = start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let result = parallel.collect_parallel(&policy)?;
        let parallel_s = start.elapsed().as_secs_f64();
        assert_eq!(format!("{reference:?}"), format!("{result:?}"));
        let report = serde_json::json!({"round":round,"serial_seconds":serial_s,"parallel_seconds":parallel_s,"complete_sequence_debug_equality":true,"samples":result.iter().map(|s|s.observations.len()).sum::<usize>(),"qualification":false});
        println!("{report}");
    }
    Ok(())
}
