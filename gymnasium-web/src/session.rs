//! Browser session dispatch keeps each action space with its learner.
use crate::{
    continuous_session::ContinuousSession, continuous_task::ContinuousTask,
    discrete_session::DiscreteSession, discrete_task::DiscreteTask, AdvanceSteps, SessionError,
    Snapshot, Task,
};

/// An isolated simulation with either a learner or a frozen policy.
#[derive(Debug)]
pub struct Session {
    /// Task-specific state cannot combine a continuous environment with DQN.
    inner: SessionKind,
}

/// Closed action-space and learner combinations.
#[derive(Debug)]
enum SessionKind {
    /// DQN over an environment's original discrete actions.
    Discrete(DiscreteSession),
    /// PPO over the original continuous force interval.
    Continuous(Box<ContinuousSession>),
}

impl Session {
    /// Start `CartPole` DQN training from a reproducible fresh model.
    ///
    /// # Errors
    /// Returns the shared learner's configuration or tensor error.
    pub fn train(seed: u64) -> Result<Self, SessionError> {
        Self::train_task(Task::CartPole, seed)
    }

    /// Start `CartPole` inference using a frozen, validated DQN record.
    ///
    /// # Errors
    /// Rejects corrupt records and records with a different architecture.
    pub fn inference(bytes: Vec<u8>, seed: u64) -> Result<Self, SessionError> {
        Self::inference_task(Task::CartPole, bytes, seed)
    }

    /// Start the selected environment with its documented learner settings.
    ///
    /// # Errors
    /// Returns the shared learner's configuration or tensor error.
    pub fn train_task(task: Task, seed: u64) -> Result<Self, SessionError> {
        let inner = match task {
            Task::CartPole => {
                SessionKind::Discrete(DiscreteSession::train(DiscreteTask::CartPole, seed)?)
            }
            Task::MountainCar => {
                SessionKind::Discrete(DiscreteSession::train(DiscreteTask::MountainCar, seed)?)
            }
            Task::MountainCarContinuous => SessionKind::Continuous(Box::new(
                ContinuousSession::train(ContinuousTask::MountainCar, seed)?,
            )),
            Task::Pendulum => SessionKind::Continuous(Box::new(ContinuousSession::train(
                ContinuousTask::Pendulum,
                seed,
            )?)),
        };
        Ok(Self { inner })
    }

    /// Load an inference record for the selected task's architecture.
    ///
    /// # Errors
    /// Rejects corrupt records and incompatible network dimensions.
    pub fn inference_task(task: Task, bytes: Vec<u8>, seed: u64) -> Result<Self, SessionError> {
        let inner = match task {
            Task::CartPole => SessionKind::Discrete(DiscreteSession::inference(
                DiscreteTask::CartPole,
                bytes,
                seed,
            )?),
            Task::MountainCar => SessionKind::Discrete(DiscreteSession::inference(
                DiscreteTask::MountainCar,
                bytes,
                seed,
            )?),
            Task::MountainCarContinuous => SessionKind::Continuous(Box::new(
                ContinuousSession::inference(ContinuousTask::MountainCar, bytes, seed)?,
            )),
            Task::Pendulum => SessionKind::Continuous(Box::new(ContinuousSession::inference(
                ContinuousTask::Pendulum,
                bytes,
                seed,
            )?)),
        };
        Ok(Self { inner })
    }

    /// Advance a bounded batch of fixed environment transitions.
    ///
    /// Time is linear in the step count plus any scheduled optimizer update.
    /// Replay and on-policy rollout storage have fixed capacities.
    ///
    /// # Errors
    /// Returns an observation, action, or optimizer error from the shared learner.
    pub fn advance(&mut self, steps: AdvanceSteps) -> Result<Snapshot, SessionError> {
        match &mut self.inner {
            SessionKind::Discrete(session) => session.advance(steps),
            SessionKind::Continuous(session) => session.advance(steps),
        }
    }

    /// Export policy parameters for inference in a fresh session.
    ///
    /// # Errors
    /// Returns the shared recorder's encoding error.
    pub fn export_policy(&self) -> Result<Vec<u8>, SessionError> {
        match &self.inner {
            SessionKind::Discrete(session) => session.export_policy(),
            SessionKind::Continuous(session) => session.export_policy(),
        }
    }
}
