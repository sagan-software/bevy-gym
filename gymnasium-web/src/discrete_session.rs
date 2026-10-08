//! Training and inference session lifecycle.
use crate::{
    discrete_task::DiscreteTask, environment::Environment, observation::Observation, AdvanceSteps,
    Episode, SessionError, Snapshot,
};
use bevy_gym::training::{DqnAgent, DqnPolicy, SeedConfig};

/// An isolated simulation with either a learner or a frozen policy.
#[derive(Debug)]
pub(crate) struct DiscreteSession {
    /// Mutually exclusive optimizer and inference state.
    mode: Mode,
    /// Shared Gymnasium dynamics with the selected task's episode cap.
    environment: Environment,
    /// Last observation, before the next selected action.
    observation: Observation,
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

impl DiscreteSession {
    /// Start the selected environment with its documented learner settings.
    ///
    /// # Errors
    /// Returns the shared learner's configuration or tensor error.
    pub(crate) fn train(task: DiscreteTask, seed: u64) -> Result<Self, SessionError> {
        let (observations, actions) = task.dimensions();
        let agent = DqnAgent::new(
            observations,
            actions,
            task.config(),
            SeedConfig::from_root(seed),
        )?;
        Ok(Self::new(Mode::Training(Box::new(agent)), task, seed))
    }

    /// Load an inference record for the selected task's architecture.
    ///
    /// # Errors
    /// Rejects corrupt records and incompatible network dimensions.
    pub(crate) fn inference(
        task: DiscreteTask,
        bytes: Vec<u8>,
        seed: u64,
    ) -> Result<Self, SessionError> {
        let (observations, actions) = task.dimensions();
        let policy =
            DqnPolicy::load_bytes(bytes, observations, actions, &task.config().hidden_sizes)?;
        Ok(Self::new(Mode::Inference(policy), task, seed))
    }

    /// Initialize shared episode state after the mode has been validated.
    fn new(mode: Mode, task: DiscreteTask, seed: u64) -> Self {
        let mut environment = Environment::new(task);
        let observation = environment.reset(Some(seed));
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

    /// Advance fixed environment transitions with at most 256 optimizer calls per batch.
    ///
    /// Work is linear in the bounded step count; retained replay has the learner's
    /// fixed capacity. Episode evidence is bounded by this batch, not run duration.
    ///
    /// # Errors
    /// Returns an observation, action, or optimizer error from the shared learner.
    pub(crate) fn advance(&mut self, steps: AdvanceSteps) -> Result<Snapshot, SessionError> {
        let mut completed = Vec::new();
        for _ in 0..steps.get() {
            let encoded = self.observation.encoded();
            let action_index = match &mut self.mode {
                Mode::Training(agent) => agent.select_action(encoded.as_ref())?.action_index,
                Mode::Inference(policy) => policy.greedy_action(encoded.as_ref())?,
            };
            let result = self.environment.step(action_index)?;
            if let Mode::Training(agent) = &mut self.mode {
                if let Some(update) = agent.observe(
                    encoded.as_ref(),
                    action_index,
                    self.observation.training_reward(
                        result.observation,
                        result.reward,
                        result.status,
                    ),
                    result.observation.encoded().as_ref(),
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
                self.observation = self.environment.reset(None);
            } else {
                self.observation = result.observation;
            }
        }
        Ok(self.snapshot(completed))
    }

    /// Project current counters and the bounded episode batch into a worker response.
    fn snapshot(&self, completed: Vec<Episode>) -> Snapshot {
        let (optimizer_steps, learning_rate, epsilon) = match &self.mode {
            Mode::Training(agent) => (
                agent.optimizer_steps(),
                Some(agent.config().learning_rate),
                Some(agent.epsilon()),
            ),
            Mode::Inference(_) => (0, None, None),
        };
        Snapshot {
            transitions: self.transitions,
            optimizer_steps,
            learning_rate,
            critic_learning_rate: None,
            critic_loss: None,
            parallel_environments: 1,
            epsilon,
            loss: self.loss,
            state: self.environment.state(),
            episode_return: self.episode_return,
            episode_count: self.episodes,
            completed,
        }
    }

    /// Export policy parameters for inference in a fresh session.
    ///
    /// # Errors
    /// Returns the shared recorder's encoding error.
    pub(crate) fn export_policy(&self) -> Result<Vec<u8>, SessionError> {
        match &self.mode {
            Mode::Training(agent) => agent.policy().to_bytes().map_err(Into::into),
            Mode::Inference(policy) => policy.to_bytes().map_err(Into::into),
        }
    }
}
