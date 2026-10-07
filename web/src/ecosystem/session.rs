//! One selected stage, policy set, and deterministic simulation session.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use bevy_gym::ecosystem::{
    EcosystemAction, EcosystemAgentId, EcosystemRuntime, EcosystemRuntimeError, EcosystemSpecies,
    EcosystemStage, EcosystemVisualSnapshot, ACTION_HIGH, ACTION_LOW, GLOBAL_STATE_SIZE,
    MAX_AGENTS,
};
use bevy_gym::training::{
    RecurrentMemory, RecurrentPpoConfig, RecurrentPpoError, RecurrentPpoPolicy,
};

use crate::checkpoint_observation::{project, CHECKPOINT_OBSERVATION_SIZE};
use crate::manifest::Stage;

/// Deterministic seed used when a route first opens.
const INITIAL_ENVIRONMENT_SEED: u64 = 907;

/// Pause at a terminal snapshot before starting the next seed.
const EPISODE_PAUSE_SECONDS: f32 = 0.8;

/// Bound catch-up work after a suspended or throttled browser frame.
const MAX_STEPS_PER_FRAME: usize = 4;

/// Browser-owned policy playback independent from render entities.
pub(crate) struct InferenceSession {
    /// Route and simulation stage.
    stage: Stage,
    /// Episode settings retained across deterministic resets.
    environment_stage: EcosystemStage,
    /// Bunny actor used by every herbivore identity.
    bunny: RecurrentPpoPolicy,
    /// Fox actor for predator stages.
    fox: Option<RecurrentPpoPolicy>,
    /// Authoritative Avian simulation.
    ecosystem: EcosystemRuntime,
    /// Recurrent state keyed by stable agent identity.
    memories: BTreeMap<EcosystemAgentId, RecurrentMemory>,
    /// Current renderer-only snapshot.
    snapshot: EcosystemVisualSnapshot,
    /// Number of completed deterministic resets.
    episode: u64,
    /// Wall-clock time awaiting the next fixed step.
    accumulator: f32,
    /// Remaining terminal-frame pause.
    episode_pause: f32,
}

impl InferenceSession {
    /// Load one compatible policy set and create its first deterministic world.
    pub(crate) fn from_checkpoint_bytes(
        stage: Stage,
        bunny_bytes: Vec<u8>,
        fox_bytes: Option<Vec<u8>>,
    ) -> Result<Self, SessionError> {
        if stage.requires_fox() != fox_bytes.is_some() {
            return Err(SessionError::fox_cardinality(stage));
        }
        let environment_stage = ecosystem_stage(stage);
        let algorithm = ecosystem_algorithm();
        let bunny = load_policy(bunny_bytes, &algorithm)?;
        let fox = fox_bytes
            .map(|bytes| load_policy(bytes, &algorithm))
            .transpose()?;
        let ecosystem = EcosystemRuntime::new(environment_stage, INITIAL_ENVIRONMENT_SEED)?;
        let snapshot = ecosystem.visual_snapshot().clone();
        Ok(Self {
            stage,
            environment_stage,
            bunny,
            fox,
            ecosystem,
            memories: BTreeMap::new(),
            snapshot,
            episode: 0,
            accumulator: 0.0,
            episode_pause: 0.0,
        })
    }

    /// Return the latest exact simulation projection.
    pub(super) const fn snapshot(&self) -> &EcosystemVisualSnapshot {
        &self.snapshot
    }

    /// Return the completed simulation step for the live status surface.
    pub(crate) const fn step(&self) -> u32 {
        self.snapshot.step
    }

    /// Return the number of agents that can still act.
    pub(crate) const fn living_agents(&self) -> usize {
        self.snapshot.living_agents
    }

    /// Return the deterministic episode number.
    pub(crate) const fn episode(&self) -> u64 {
        self.episode
    }

    /// End the current trajectory and start the next deterministic episode.
    pub(crate) fn restart(&mut self) -> Result<(), SessionError> {
        self.reset_episode()
    }

    /// Advance deterministic inference using bounded fixed-step catch-up.
    pub(crate) fn advance(&mut self, delta_seconds: f32) -> Result<(), SessionError> {
        if self.episode_pause > 0.0 {
            self.episode_pause = (self.episode_pause - delta_seconds).max(0.0);
            if self.episode_pause == 0.0 {
                self.reset_episode()?;
            }
            return Ok(());
        }

        self.accumulator += delta_seconds.min(0.1);
        let mut steps = 0;
        while self.accumulator >= self.ecosystem.time_step() && steps < MAX_STEPS_PER_FRAME {
            self.accumulator -= self.ecosystem.time_step();
            if self.step_once()? {
                self.episode_pause = EPISODE_PAUSE_SECONDS;
                break;
            }
            steps += 1;
        }
        Ok(())
    }

    /// Replace uploaded policies atomically and restart the same initial seed.
    pub(crate) fn replace_checkpoint_bytes(
        &mut self,
        bunny_bytes: Vec<u8>,
        fox_bytes: Option<Vec<u8>>,
    ) -> Result<(), SessionError> {
        if self.stage.requires_fox() != fox_bytes.is_some() {
            return Err(SessionError::fox_cardinality(self.stage));
        }
        let algorithm = ecosystem_algorithm();
        let bunny = load_policy(bunny_bytes, &algorithm)?;
        let fox = fox_bytes
            .map(|bytes| load_policy(bytes, &algorithm))
            .transpose()?;

        // Commit both actors only after every supplied record has loaded.
        self.bunny = bunny;
        self.fox = fox;
        self.episode = 0;
        self.reset_episode()
    }

    /// Run one mean-action joint step and retain memory only for continuing agents.
    fn step_once(&mut self) -> Result<bool, SessionError> {
        let mut actions = Vec::with_capacity(self.ecosystem.observations().count());
        let mut next_memories = BTreeMap::new();
        for observation in self.ecosystem.observations() {
            let policy = match observation.species {
                EcosystemSpecies::Bunny => &self.bunny,
                EcosystemSpecies::Fox => self.fox.as_ref().ok_or_else(SessionError::missing_fox)?,
            };
            let memory = self
                .memories
                .entry(observation.id)
                .or_insert_with(|| policy.initial_memory());
            let policy_observation = project(observation.observation);
            let output = policy.mean_action(&policy_observation, memory)?;
            let [forward, turn, gaze, attack] = output.action.as_slice() else {
                return Err(SessionError::action_width(output.action.len()));
            };
            let action = EcosystemAction::try_from([*forward, *turn, *gaze, *attack])?;
            actions.push((observation.id, action));
            next_memories.insert(observation.id, output.next_memory);
        }

        // The simulation applies every action before physics, preserving joint semantics.
        let result = self.ecosystem.step(&actions)?;
        self.snapshot = self.ecosystem.visual_snapshot().clone();
        self.memories.clear();
        for id in result.living_agents {
            if let Some(memory) = next_memories.remove(&id) {
                self.memories.insert(id, memory);
            }
        }
        Ok(result.is_done)
    }

    /// Start the next deterministic seed without carrying recurrent memory.
    fn reset_episode(&mut self) -> Result<(), SessionError> {
        self.episode = self.episode.saturating_add(1);
        let seed = INITIAL_ENVIRONMENT_SEED.wrapping_add(self.episode);
        self.ecosystem = EcosystemRuntime::new(self.environment_stage, seed)?;
        self.snapshot = self.ecosystem.visual_snapshot().clone();
        self.memories.clear();
        self.accumulator = 0.0;
        self.episode_pause = 0.0;
        Ok(())
    }
}

/// Load the stable ecosystem architecture from one named MessagePack record.
fn load_policy(
    bytes: Vec<u8>,
    algorithm: &RecurrentPpoConfig,
) -> Result<RecurrentPpoPolicy, RecurrentPpoError> {
    RecurrentPpoPolicy::load_bytes(
        bytes,
        CHECKPOINT_OBSERVATION_SIZE,
        GLOBAL_STATE_SIZE,
        MAX_AGENTS,
        &ACTION_LOW,
        &ACTION_HIGH,
        algorithm,
    )
}

/// Return the architecture and inference settings used by ecosystem training.
fn ecosystem_algorithm() -> RecurrentPpoConfig {
    RecurrentPpoConfig {
        gamma: 0.999,
        gae_lambda: 1.0,
        actor_learning_rate: 3e-4,
        critic_learning_rate: 3e-4,
        entropy_coefficient: 0.005,
        epochs: 3,
        minibatch_sequences: 8,
        initial_log_std: -0.5,
        ..RecurrentPpoConfig::default()
    }
}

/// Opaque session construction or inference failure.
#[derive(Debug)]
pub(crate) struct SessionError(SessionErrorKind);

/// Internal category and source for a browser session failure.
#[derive(Debug)]
enum SessionErrorKind {
    /// Portable ecosystem construction, action, or progression failed.
    Ecosystem(EcosystemRuntimeError),
    /// Burn rejected a checkpoint or inference input.
    Policy(RecurrentPpoError),
    /// Fox checkpoint presence does not match the selected stage.
    FoxCardinality(Stage),
    /// A predator stage reached inference without a fox policy.
    MissingFox,
    /// A policy emitted a vector outside the four-axis ecosystem contract.
    ActionWidth(usize),
}

impl SessionError {
    /// Construct a fox-cardinality failure.
    const fn fox_cardinality(stage: Stage) -> Self {
        Self(SessionErrorKind::FoxCardinality(stage))
    }

    /// Construct a missing-fox failure.
    const fn missing_fox() -> Self {
        Self(SessionErrorKind::MissingFox)
    }

    /// Construct an invalid actor-output-width failure.
    const fn action_width(width: usize) -> Self {
        Self(SessionErrorKind::ActionWidth(width))
    }
}

impl fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            SessionErrorKind::Ecosystem(error) => {
                write!(formatter, "ecosystem runtime failed: {error}")
            }
            SessionErrorKind::Policy(error) => {
                write!(formatter, "checkpoint inference failed: {error}")
            }
            SessionErrorKind::FoxCardinality(stage) => write!(
                formatter,
                "fox checkpoint cardinality is invalid for {}",
                stage.as_key()
            ),
            SessionErrorKind::MissingFox => formatter.write_str("fox policy is missing"),
            SessionErrorKind::ActionWidth(width) => {
                write!(
                    formatter,
                    "ecosystem policy emitted {width} action axes; expected 4"
                )
            }
        }
    }
}

impl Error for SessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.0 {
            SessionErrorKind::Ecosystem(error) => Some(error),
            SessionErrorKind::Policy(error) => Some(error),
            SessionErrorKind::FoxCardinality(_)
            | SessionErrorKind::MissingFox
            | SessionErrorKind::ActionWidth(_) => None,
        }
    }
}

impl From<EcosystemRuntimeError> for SessionError {
    fn from(error: EcosystemRuntimeError) -> Self {
        Self(SessionErrorKind::Ecosystem(error))
    }
}

impl From<RecurrentPpoError> for SessionError {
    fn from(error: RecurrentPpoError) -> Self {
        Self(SessionErrorKind::Policy(error))
    }
}

/// Map the browser route vocabulary to the portable library stage.
const fn ecosystem_stage(stage: Stage) -> EcosystemStage {
    match stage {
        Stage::Forage => EcosystemStage::Forage,
        Stage::Survival => EcosystemStage::Survival,
        Stage::Shelter => EcosystemStage::Shelter,
        Stage::Competition => EcosystemStage::Competition,
        Stage::PredatorPrey => EcosystemStage::PredatorPrey,
        Stage::Obstacles => EcosystemStage::Obstacles,
    }
}
