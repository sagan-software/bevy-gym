//! Portable policy records used by native and browser inference.

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
#[cfg(feature = "render")]
use bevy_inspector_egui as _;
use burn as _;
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde as _;
use serde_json as _;
use shakmaty as _;
use tokio as _;

use bevy_gym::training::{DqnAgent, DqnConfig, DqnPolicy, SeedConfig};

#[test]
fn policy_bytes_preserve_q_values_without_filesystem_access() {
    let config = DqnConfig {
        hidden_sizes: vec![8],
        ..DqnConfig::default()
    };
    let policy = DqnAgent::new(4, 2, config, SeedConfig::from_root(42))
        .expect("valid learner")
        .policy();
    let observation = [0.01, -0.02, 0.03, -0.04];
    let expected = policy.q_values(&observation).expect("four observations");
    let bytes = policy.to_bytes().expect("serializable policy");
    let restored = DqnPolicy::load_bytes(bytes, 4, 2, &[8]).expect("matching architecture");
    assert_eq!(
        restored.q_values(&observation).expect("four observations"),
        expected
    );
}

#[test]
fn policy_bytes_reject_corruption_and_architecture_mismatch() {
    DqnPolicy::load_bytes(vec![0, 1, 2], 4, 2, &[8]).expect_err("corrupt record");
    let policy = DqnAgent::new(
        4,
        2,
        DqnConfig {
            hidden_sizes: vec![8],
            ..DqnConfig::default()
        },
        SeedConfig::from_root(42),
    )
    .expect("valid learner")
    .policy();
    let bytes = policy.to_bytes().expect("serializable policy");
    for (observations, actions, hidden) in [
        (3, 2, vec![8]),
        (4, 3, vec![8]),
        (4, 2, vec![9]),
        (4, 2, vec![8, 8]),
        (0, 2, vec![8]),
        (4, 0, vec![8]),
        (4, 2, vec![0]),
    ] {
        DqnPolicy::load_bytes(bytes.clone(), observations, actions, &hidden)
            .expect_err("incompatible architecture");
    }
}
