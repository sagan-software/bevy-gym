//! Frozen position, heading, arrival-time and settling gates for travel stages.

use bevy_gym::robots::DroneTravelObservation;
use bevy_gym::training::RecurrentPpoPolicy;
use bevy_gym::{Env, TimeLimit};
use serde::{Deserialize, Serialize};
use std::{error::Error, num::NonZeroU16};

use super::{encoding, environment::TravelTask, stage::Stage};
use crate::learning::decode_action;

/// One frozen evaluation episode; these records never become optimizer input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Score {
    /// Independent reset root.
    pub(crate) seed: u64,
    /// Applied 20 ms motor commands, at most 1,000.
    pub(crate) steps: u16,
    /// Sum of unscaled environment rewards.
    pub(crate) reward: f64,
    /// Whether the episode reached its time limit without termination.
    pub(crate) survived: bool,
    /// First action inside the joint position and heading band, or no arrival.
    #[serde(deserialize_with = "required_arrival")]
    pub(crate) first_arrival: Option<NonZeroU16>,
    /// Consecutive final actions satisfying position, heading and speed gates.
    pub(crate) settled_actions: u16,
    /// Final body-centre distance to the destination, in metres.
    pub(crate) final_distance: f32,
    /// Final absolute heading error, in radians.
    pub(crate) final_heading_error: f32,
    /// Final world linear speed, in metres per second.
    pub(crate) final_speed: f32,
}

/// Require the nullable arrival member while retaining Serde's standard nonzero conversion.
fn required_arrival<'de, D>(deserializer: D) -> Result<Option<NonZeroU16>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<NonZeroU16>::deserialize(deserializer)
}

/// Require the exact ordered evaluation set and every episode's physical gates.
pub(crate) fn passes(stage: Stage, scores: &[Score], seeds: &[u64]) -> bool {
    !seeds.is_empty()
        && scores.len() == seeds.len()
        && scores.iter().zip(seeds).all(|(score, seed)| {
            score.seed == *seed
                && score.survived
                && score.steps == 1_000
                && score
                    .first_arrival
                    .is_some_and(|action| action.get() <= stage.deadline())
                && (100..=1_000).contains(&score.settled_actions)
                && score.final_distance <= 0.5
                && score.final_heading_error <= std::f32::consts::PI / 12.0
                && score.final_speed <= 0.5
        })
        && scores.iter().map(|score| score.reward).sum::<f64>() / scores.len() as f64
            >= stage.minimum_return()
}

/// Evaluate frozen policies with independent memory and the shared sampled environment.
pub(crate) fn evaluate(
    stage: Stage,
    policy: &RecurrentPpoPolicy,
    seeds: &[u64],
) -> Result<Vec<Score>, Box<dyn Error>> {
    seeds
        .iter()
        .map(|&seed| episode(stage, policy, seed))
        .collect()
}

/// Run one episode without training, fallback actions, or updates to policy parameters.
fn episode(stage: Stage, policy: &RecurrentPpoPolicy, seed: u64) -> Result<Score, Box<dyn Error>> {
    let mut environment = TimeLimit::new(TravelTask::new(stage), 1_000)?;
    let mut observation = environment.reset(Some(seed)).observation;
    let mut memory = policy.initial_memory();
    let mut reward = 0.0;
    let mut first_arrival = None;
    let mut settled_actions = 0;
    let mut steps = 0;
    loop {
        steps += 1;
        let action = policy.mean_action(&encoding::encode(observation), &memory)?;
        let result = environment.step(decode_action(&action.action)?);
        observation = result.observation;
        memory = action.next_memory;
        reward += result.reward;
        let (distance, heading, linear_speed) = measurements(observation);
        let arrived = distance <= 0.5 && heading <= std::f32::consts::PI / 12.0;
        // Arrival time measures progress; final consecutive samples prove stable arrival.
        if arrived && first_arrival.is_none() {
            first_arrival = NonZeroU16::new(steps);
        }
        settled_actions = if arrived && linear_speed <= 0.5 {
            settled_actions + 1
        } else {
            0
        };
        if result.status.is_done() {
            return Ok(Score {
                seed,
                steps,
                reward,
                survived: result.status.is_truncated(),
                first_arrival,
                settled_actions,
                final_distance: distance,
                final_heading_error: heading,
                final_speed: linear_speed,
            });
        }
    }
}

/// Derive final task measurements from the physical observation without changing it.
fn measurements(observation: DroneTravelObservation) -> (f32, f32, f32) {
    let body = observation.body();
    let distance = body
        .position()
        .distance(observation.destination().position());
    let heading = encoding::heading_error(body, observation.destination()).abs();
    (distance, heading, body.linear_velocity().length())
}
