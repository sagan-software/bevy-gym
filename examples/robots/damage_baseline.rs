//! Compare intact-policy inference with constant thrust after one motor fails.
//!
//! Run `cargo run --no-default-features --features robots --example drone-damage-baseline`.
//! Each JSON line contains paired calm-start trials with ten seconds after failure.
//! The bundled policy learned intact flight; these measurements do not train it.

#[path = "damage/assessment.rs"]
mod assessment;
#[path = "learning/encoding.rs"]
mod encoding;
#[path = "learning/model.rs"]
mod model;

use assessment::{assess, FailureTime};
use bevy_gym::robots::{DroneAction, DroneMotor};
use encoding::{decode_action, encode};

/// Evaluate every motor and schedule on 32 seeds excluded from hover training.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let policy = model::load_policy(include_bytes!("../../assets/robots/recovery.mpk").to_vec())?;
    let half_thrust = DroneAction::try_from([0.5; 4])?;

    for seed in (1..=32).map(|offset| u64::MAX - offset) {
        for motor in DroneMotor::ALL {
            for failure in [FailureTime::TwoSeconds, FailureTime::FiveSeconds] {
                let constant = assess(seed, motor, failure, |_| Ok(half_thrust))?;
                // Clear recurrent memory between trials while retaining the same frozen weights.
                let mut memory = policy.initial_memory();
                let intact_policy = assess(seed, motor, failure, |observation| {
                    let output = policy.mean_action(&encode(observation), &memory)?;
                    let action = decode_action(&output.action)?;
                    memory = output.next_memory;
                    Ok(action)
                })?;
                let motor_name = format!("{motor:?}");
                let row = serde_json::json!({
                    "seed": seed,
                    "motor": motor_name,
                    "failure": failure,
                    "half_thrust": constant,
                    "intact_policy": intact_policy,
                });
                println!("{row}");
            }
        }
    }
    Ok(())
}
