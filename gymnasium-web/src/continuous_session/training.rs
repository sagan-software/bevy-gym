//! Eight independent lanes contribute one fixed 512-transition PPO rollout.
use super::{
    config,
    episode::EpisodeState,
    rollout::{self, RolloutStep},
};
use crate::{continuous_task::ContinuousTask, Episode};
use bevy_gym::training::{
    RecurrentMemory, RecurrentPpoAgent, RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy,
    RecurrentPpoSequence, RecurrentPpoUpdate, RecurrentSampler, SeedConfig,
};

/// Number of independently seeded contributing environments.
pub(super) const LANES: usize = 8;
/// On-policy transitions per lane and optimizer rollout.
const ROLLOUT_LENGTH: usize = 64;

/// Learner state exists only in training mode.
#[derive(Debug)]
pub(super) struct Training {
    /// Actor, critic, and both Adam optimizers.
    agent: RecurrentPpoAgent,
    /// Frozen parameters used by every lane in the current rollout.
    pub policy: RecurrentPpoPolicy,
    /// Fixed optimizer and GAE profile.
    pub config: RecurrentPpoConfig,
    /// Stable lane order also fixes minibatch input ordering.
    lanes: [Lane; LANES],
    /// Independent model, action, and reset streams.
    seeds: SeedConfig,
    /// Most recent successful PPO update.
    pub update: Option<RecurrentPpoUpdate>,
}

/// Per-lane memory and trajectory segments never cross an episode boundary.
#[derive(Debug)]
struct Lane {
    /// Original environment and raw reward counters.
    episode: EpisodeState,
    /// Policy memory for this trajectory segment.
    memory: RecurrentMemory,
    /// Independent action sampling stream.
    sampler: RecurrentSampler,
    /// Current segment, bounded by 64 transitions.
    steps: Vec<RolloutStep>,
    /// Finished segments in this rollout, bounded by 64 total transitions.
    finished: Vec<RecurrentPpoSequence>,
}

impl Training {
    /// Initialize one model and eight deterministic environment streams.
    pub(super) fn new(task: ContinuousTask, seed: u64) -> Result<Self, RecurrentPpoError> {
        let seeds = SeedConfig::from_root(seed);
        let config = config();
        let dimension = task.observation_dim();
        let (low, high) = task.action_bounds();
        let agent =
            RecurrentPpoAgent::new(dimension, dimension, 1, low, high, config.clone(), seeds)?;
        let policy = agent.policy();
        let lanes = std::array::from_fn(|index| Lane {
            episode: EpisodeState::new(task, seeds.environment_episode(index, 0)),
            memory: policy.initial_memory(),
            sampler: RecurrentSampler::new(seeds.action ^ (index as u64).wrapping_mul(0x9e37_79b9)),
            steps: Vec::with_capacity(ROLLOUT_LENGTH),
            finished: Vec::new(),
        });
        Ok(Self {
            agent,
            policy,
            config,
            lanes,
            seeds,
            update: None,
        })
    }

    /// Collect one transition; update both networks only after a complete rollout.
    pub(super) fn step(&mut self, transition: u64) -> Result<Option<Episode>, RecurrentPpoError> {
        let index = ((transition - 1) % LANES as u64) as usize;
        let lane = self
            .lanes
            .get_mut(index)
            .expect("modulo lane count bounds the index");
        let raw = lane.episode.observation();
        let observation = raw.encoded();
        let sampled =
            self.policy
                .sample_action(observation.as_ref(), &lane.memory, &mut lane.sampler)?;
        let value = self.policy.value(observation.as_ref(), 0)?;
        let result = lane.episode.step(&sampled.action)?;
        // Bootstrap from the actual next observation, including time-limit boundaries.
        let next_value = self
            .policy
            .value(result.observation.encoded().as_ref(), 0)?;
        lane.steps.push(RolloutStep {
            observation,
            pre_tanh_action: sampled.pre_tanh_action,
            log_probability: sampled.log_probability,
            reward: raw.training_reward(result.observation, result.reward, result.status) as f32,
            value,
            next_value,
            status: result.status,
        });
        lane.memory = sampled.next_memory;
        let completed = if result.is_done() {
            let completed = Episode {
                transition,
                reward: lane.episode.reward,
            };
            lane.finish(&self.config);
            let next_seed = self
                .seeds
                .environment_episode(index, lane.episode.completed + 1);
            lane.episode.reset(Some(next_seed));
            lane.memory = self.policy.initial_memory();
            Some(completed)
        } else {
            None
        };
        if transition.is_multiple_of((LANES * ROLLOUT_LENGTH) as u64) {
            self.optimize()?;
        }
        Ok(completed)
    }

    /// Consume all lanes in stable order, then begin a new frozen-policy rollout.
    fn optimize(&mut self) -> Result<(), RecurrentPpoError> {
        let mut sequences = Vec::with_capacity(LANES);
        for lane in &mut self.lanes {
            lane.finish(&self.config);
            sequences.append(&mut lane.finished);
        }
        self.update = Some(self.agent.update(&sequences)?);
        self.policy = self.agent.policy();
        // The native recipe starts each rollout with zero recurrent memory.
        for lane in &mut self.lanes {
            lane.memory = self.policy.initial_memory();
        }
        Ok(())
    }

    /// Show lane zero, which contributes to every optimizer rollout.
    pub(super) const fn displayed(&self) -> &EpisodeState {
        &self.lanes[0].episode
    }

    /// Derive completed episodes from the independent lane counters.
    pub(super) fn completed(&self) -> u64 {
        self.lanes.iter().map(|lane| lane.episode.completed).sum()
    }
}

impl Lane {
    /// Close a nonempty segment without creating empty PPO sequences.
    fn finish(&mut self, config: &RecurrentPpoConfig) {
        if !self.steps.is_empty() {
            let steps = std::mem::replace(&mut self.steps, Vec::with_capacity(ROLLOUT_LENGTH));
            self.finished.push(rollout::finish(steps, config));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_an_empty_segment_does_not_create_a_ppo_sequence() {
        let mut training = Training::new(ContinuousTask::MountainCar, 42).expect("fixed profile");
        let lane = training.lanes.first_mut().expect("eight lanes");
        lane.finish(&training.config);
        assert!(lane.finished.is_empty());
        assert!(lane.steps.is_empty());
    }
}
