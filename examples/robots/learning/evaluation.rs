//! Held-out evaluation with deterministic actions and unscaled task rewards.

use std::error::Error;

use bevy_gym::robots::{DroneAction, DroneHover};
use bevy_gym::training::RecurrentPpoPolicy;

pub(crate) use super::episode::evaluate_episode;
pub(crate) use super::episode::{evaluate_with, EpisodeScore};

/// Model-selection seeds excluded from every training lane.
pub(crate) const SELECTION_SEEDS: [u64; 5] = [0, 1, 2, 42, u64::MAX];

/// Evaluate each requested seed without sampling or changing policy parameters.
pub(crate) fn evaluate(
    policy: &RecurrentPpoPolicy,
    seeds: &[u64],
) -> Result<Vec<EpisodeScore>, Box<dyn Error>> {
    evaluate_with(policy, seeds, DroneHover::disturbed)
}

/// Measure constant half-thrust using the same environment and episode accounting.
pub(crate) fn baseline(seeds: &[u64]) -> Result<Vec<EpisodeScore>, Box<dyn Error>> {
    let action = DroneAction::try_from([0.5; 4])?;
    let mut episodes = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        episodes.push(evaluate_episode(seed, DroneHover::disturbed(), |_| {
            Ok(action)
        })?);
    }
    Ok(episodes)
}
