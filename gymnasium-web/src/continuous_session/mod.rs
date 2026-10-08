//! Continuous `MountainCar` training and frozen inference in a bounded worker session.
mod episode;
mod rollout;
mod training;

use crate::{AdvanceSteps, Episode, SessionError, Snapshot};
use bevy_gym::training::{RecurrentMemory, RecurrentPpoConfig, RecurrentPpoPolicy};
use episode::EpisodeState;
use training::Training;

/// Continuous-action state with mutually exclusive training and inference capabilities.
#[derive(Debug)]
pub(crate) struct ContinuousSession {
    /// Only training can own an optimizer or an on-policy rollout.
    mode: Mode,
    /// Number of completed environment transitions across contributing lanes.
    transitions: u64,
}

/// An inference session owns no optimizer.
#[derive(Debug)]
enum Mode {
    /// Eight contributing lanes and PPO optimizers.
    Training(Box<Training>),
    /// Deterministic mean-action replay on one environment.
    Inference(Box<Inference>),
}

/// A frozen policy and one persistent episode have no training capability.
#[derive(Debug)]
struct Inference {
    /// Frozen parameters validated before mode activation.
    policy: RecurrentPpoPolicy,
    /// Original environment and raw return accounting.
    episode: EpisodeState,
    /// Memory persists until this episode ends.
    memory: RecurrentMemory,
}

impl ContinuousSession {
    /// Start fresh PPO training with the native example's declared recipe.
    pub(crate) fn train(seed: u64) -> Result<Self, SessionError> {
        Ok(Self {
            mode: Mode::Training(Box::new(Training::new(seed)?)),
            transitions: 0,
        })
    }

    /// Validate the record before starting a frozen deterministic episode.
    pub(crate) fn inference(bytes: Vec<u8>, seed: u64) -> Result<Self, SessionError> {
        let policy = RecurrentPpoPolicy::load_bytes(bytes, 2, 2, 1, &[-1.0], &[1.0], &config())?;
        let memory = policy.initial_memory();
        Ok(Self {
            mode: Mode::Inference(Box::new(Inference {
                policy,
                episode: EpisodeState::new(seed),
                memory,
            })),
            transitions: 0,
        })
    }

    /// Consume at most 256 transitions and return only this batch's episode evidence.
    pub(crate) fn advance(&mut self, steps: AdvanceSteps) -> Result<Snapshot, SessionError> {
        let mut completed = Vec::new();
        for _ in 0..steps.get() {
            let transition = self.transitions + 1;
            let finished = match &mut self.mode {
                Mode::Training(training) => training.step(transition)?,
                Mode::Inference(inference) => {
                    let Inference {
                        policy,
                        episode,
                        memory,
                    } = inference.as_mut();
                    let action = policy.mean_action(&episode.encoded(), memory)?;
                    let result = episode.step(&action.action)?;
                    *memory = action.next_memory;
                    if result.is_done() {
                        let finished = Episode {
                            transition,
                            reward: episode.reward,
                        };
                        episode.reset(None);
                        *memory = policy.initial_memory();
                        Some(finished)
                    } else {
                        None
                    }
                }
            };
            self.transitions = transition;
            completed.extend(finished);
        }
        Ok(self.snapshot(completed))
    }

    /// Derive metrics from the active mode without retaining contradictory counters.
    fn snapshot(&self, completed: Vec<Episode>) -> Snapshot {
        let (episode, count, update, rates, parallel_environments) = match &self.mode {
            Mode::Training(training) => (
                training.displayed(),
                training.completed(),
                training.update,
                Some((
                    training.config.actor_learning_rate,
                    training.config.critic_learning_rate,
                )),
                training::LANES,
            ),
            Mode::Inference(inference) => (
                &inference.episode,
                inference.episode.completed,
                None,
                None,
                1,
            ),
        };
        Snapshot {
            transitions: self.transitions,
            optimizer_steps: update.map_or(0, |value| value.optimizer_steps),
            learning_rate: rates.map(|(actor, _critic)| actor),
            critic_learning_rate: rates.map(|(_actor, critic)| critic),
            epsilon: None,
            loss: update.map(|value| value.actor_loss),
            critic_loss: update.map(|value| value.critic_loss),
            parallel_environments,
            state: episode.state(),
            episode_return: episode.reward,
            episode_count: count,
            completed,
        }
    }

    /// Export actor and critic parameters without optimizer state.
    pub(crate) fn export_policy(&self) -> Result<Vec<u8>, SessionError> {
        match &self.mode {
            Mode::Training(training) => training.policy.to_bytes().map_err(Into::into),
            Mode::Inference(inference) => inference.policy.to_bytes().map_err(Into::into),
        }
    }
}

/// Native `MountainCarContinuous` recipe; reward shaping uses the same gamma 0.99.
fn config() -> RecurrentPpoConfig {
    RecurrentPpoConfig {
        actor_hidden_size: 32,
        critic_hidden_sizes: vec![64, 32],
        gamma: 0.99,
        gae_lambda: 0.95,
        actor_learning_rate: 0.003,
        critic_learning_rate: 0.001,
        entropy_coefficient: 0.0,
        epochs: 4,
        minibatch_sequences: 4,
        initial_log_std: -0.5,
        ..RecurrentPpoConfig::default()
    }
}
