//! Held-out evaluation with deterministic actions and unscaled task rewards.

use std::error::Error;

use bevy::math::Vec3;
use bevy_gym::robots::{DroneAction, DroneHover, DroneObservation};
use bevy_gym::training::RecurrentPpoPolicy;
use bevy_gym::{Env, TimeLimit};
use serde::Serialize;

use super::{decode_action, encode};

/// Model-selection seeds excluded from every training lane.
pub(crate) const SELECTION_SEEDS: [u64; 5] = [0, 1, 2, 42, u64::MAX];

/// One complete evaluation episode, independent of training rollout state.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct EpisodeScore {
    /// Reset seed used for this episode.
    pub(crate) seed: u64,
    /// Applied actions, capped at 500.
    pub(crate) steps: usize,
    /// Sum of unscaled environment rewards.
    pub(crate) reward: f64,
    /// Distance from the body centre to the hover target, in metres.
    pub(crate) final_distance: f32,
    /// Whether the drone reached the external time limit without a crash.
    pub(crate) survived: bool,
}

/// Evaluate each requested seed without sampling or changing policy parameters.
pub(crate) fn evaluate(
    policy: &RecurrentPpoPolicy,
    seeds: &[u64],
) -> Result<Vec<EpisodeScore>, Box<dyn Error>> {
    evaluate_with(policy, seeds, DroneHover::disturbed)
}

/// Score the selected lesson using independent environments and episode memory.
pub(crate) fn evaluate_with(
    policy: &RecurrentPpoPolicy,
    seeds: &[u64],
    make_environment: fn() -> DroneHover,
) -> Result<Vec<EpisodeScore>, Box<dyn Error>> {
    let mut episodes = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let mut memory = policy.initial_memory();
        episodes.push(evaluate_episode(seed, make_environment(), |observation| {
            let action = policy.mean_action(&encode(observation), &memory)?;
            memory = action.next_memory;
            Ok(decode_action(&action.action)?)
        })?);
    }
    Ok(episodes)
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

/// Stop at the first completion result; reset observations never enter the score.
pub(crate) fn evaluate_episode(
    seed: u64,
    environment: DroneHover,
    mut action: impl FnMut(DroneObservation) -> Result<DroneAction, Box<dyn Error>>,
) -> Result<EpisodeScore, Box<dyn Error>> {
    let mut environment = TimeLimit::new(environment, 500)?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut reward = 0.0;
    let mut steps = 0;
    loop {
        let result = environment.step(action(observation)?);
        steps += 1;
        observation = result.observation;
        reward += result.reward;
        if result.status.is_done() {
            return Ok(EpisodeScore {
                seed,
                steps,
                reward,
                final_distance: observation.position().distance(Vec3::new(0.0, 2.0, 0.0)),
                survived: result.status.is_truncated(),
            });
        }
    }
}
