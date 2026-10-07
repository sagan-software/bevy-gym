//! Training and inference session lifecycle.
use crate::{AdvanceSteps, Episode, SessionError, Snapshot};
use bevy_gym::environments::{CartPole, CartPoleAction};
use bevy_gym::training::{DqnAgent, DqnConfig, DqnError, DqnPolicy, SeedConfig};
use bevy_gym::wrappers::time_limit::TimeLimit;
use bevy_gym::Env;

/// An isolated simulation with either a learner or a frozen policy.
#[derive(Debug)]
pub struct Session {
    /// Mutually exclusive optimizer and inference state.
    mode: Mode,
    /// Shared Gymnasium dynamics with the `CartPole`-v1 500-step cap.
    environment: TimeLimit<CartPole>,
    /// Last observation, before the next selected action.
    observation: [f32; 4],
    /// Count of consumed environment transitions.
    transitions: u64,
    /// Count of completed episodes.
    episodes: u64,
    /// Sum of original rewards in the active episode.
    episode_return: f64,
    /// Last successful optimizer loss.
    loss: Option<f64>,
}

/// A frozen policy cannot retain or call an optimizer.
#[derive(Debug)]
enum Mode {
    /// Owns network, replay, independent random streams, and optimizer.
    Training(Box<DqnAgent>),
    /// Owns only inference parameters.
    Inference(DqnPolicy),
}

impl Session {
    /// Start `CartPole` DQN training from a reproducible fresh model.
    ///
    /// # Errors
    /// Returns the shared learner's configuration or tensor error.
    pub fn train(seed: u64) -> Result<Self, SessionError> {
        let agent = DqnAgent::new(4, 2, DqnConfig::default(), SeedConfig::from_root(seed))?;
        Ok(Self::new(Mode::Training(Box::new(agent)), seed))
    }

    /// Start `CartPole` inference using a frozen, validated DQN record.
    ///
    /// # Errors
    /// Rejects corrupt records and records with a different architecture.
    pub fn inference(bytes: Vec<u8>, seed: u64) -> Result<Self, SessionError> {
        let policy = DqnPolicy::load_bytes(bytes, 4, 2, &[64, 64])?;
        Ok(Self::new(Mode::Inference(policy), seed))
    }

    /// Initialize shared episode state after the mode has been validated.
    fn new(mode: Mode, seed: u64) -> Self {
        let mut environment =
            TimeLimit::new(CartPole::default(), 500).expect("positive constant limit");
        let observation = environment.reset(Some(seed)).observation;
        Self {
            mode,
            environment,
            observation,
            transitions: 0,
            episodes: 0,
            episode_return: 0.0,
            loss: None,
        }
    }

    /// Advance fixed 20 ms transitions with at most 256 optimizer calls per batch.
    ///
    /// Work is linear in the bounded step count; retained replay has the learner's
    /// fixed capacity. Episode evidence is bounded by this batch, not run duration.
    ///
    /// # Errors
    /// Returns an observation, action, or optimizer error from the shared learner.
    pub fn advance(&mut self, steps: AdvanceSteps) -> Result<Snapshot, SessionError> {
        let mut completed = Vec::new();
        for _ in 0..steps.get() {
            let action_index = match &mut self.mode {
                Mode::Training(agent) => agent.select_action(&self.observation)?.action_index,
                Mode::Inference(policy) => policy.greedy_action(&self.observation)?,
            };
            let action = CartPoleAction::try_from(action_index).map_err(|_invalid_action| {
                DqnError::ActionOutOfBounds {
                    action_index,
                    action_dim: 2,
                }
            })?;
            let result = self.environment.step(action);
            if let Mode::Training(agent) = &mut self.mode {
                if let Some(update) = agent.observe(
                    &self.observation,
                    action_index,
                    result.reward,
                    &result.observation,
                    result.status,
                )? {
                    self.loss = Some(update.loss);
                }
            }
            self.transitions += 1;
            self.episode_return += result.reward;
            // Reset only after storing the actual terminal or truncated observation in replay.
            if result.is_done() {
                completed.push(Episode {
                    transition: self.transitions,
                    reward: self.episode_return,
                });
                self.episodes += 1;
                self.episode_return = 0.0;
                self.observation = self.environment.reset(None).observation;
            } else {
                self.observation = result.observation;
            }
        }
        let (optimizer_steps, learning_rate, epsilon) = match &self.mode {
            Mode::Training(agent) => (
                agent.optimizer_steps(),
                Some(agent.config().learning_rate),
                Some(agent.epsilon()),
            ),
            Mode::Inference(_) => (0, None, None),
        };
        Ok(Snapshot {
            transitions: self.transitions,
            optimizer_steps,
            learning_rate,
            epsilon,
            loss: self.loss,
            state: *self.environment.inner().state(),
            episode_return: self.episode_return,
            episode_count: self.episodes,
            completed,
        })
    }

    /// Export policy parameters for inference in a fresh session.
    ///
    /// # Errors
    /// Returns the shared recorder's encoding error.
    pub fn export_policy(&self) -> Result<Vec<u8>, SessionError> {
        match &self.mode {
            Mode::Training(agent) => agent.policy().to_bytes().map_err(Into::into),
            Mode::Inference(policy) => policy.to_bytes().map_err(Into::into),
        }
    }
}
