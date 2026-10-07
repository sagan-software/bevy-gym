//! Portable ecosystem simulation facade for native and browser inference.
//!
//! The implementation remains the source used by the ecosystem examples. This
//! module exposes only the observations, actions, progression, and visual
//! projection needed by an inference consumer.

use std::error::Error;
use std::fmt;

use crate::EpisodeStatus;

#[path = "../../examples/ecosystem/shared/domain.rs"]
#[expect(
    dead_code,
    reason = "the narrow inference facade intentionally hides training-only domain operations"
)]
#[cfg_attr(
    test,
    expect(
        clippy::assertions_on_result_states,
        reason = "the extracted source retains its existing exact contract tests"
    )
)]
mod domain;
#[path = "../../examples/ecosystem/shared/reward.rs"]
#[cfg_attr(
    test,
    expect(
        clippy::float_cmp,
        reason = "the extracted source retains its existing exact reward tests"
    )
)]
mod reward;
#[path = "../../examples/ecosystem/shared/rng.rs"]
mod rng;
#[path = "../../examples/ecosystem/shared/simulation.rs"]
#[expect(
    dead_code,
    reason = "the narrow inference facade intentionally hides native evaluation operations"
)]
#[cfg_attr(
    test,
    expect(
        clippy::assertions_on_constants,
        clippy::float_cmp,
        clippy::suboptimal_flops,
        reason = "the extracted source retains its existing deterministic simulation tests"
    )
)]
mod simulation;

use domain::{
    ActionError, AgentId, CurriculumStage, EcosystemState, LocomotionAction, SimulationConfig,
    Species, VisualObjectKind, VisualWorldSnapshot,
};
use simulation::{Ecosystem, SimulationError};

/// Local actor observation width shared by every ecosystem stage.
pub const LOCAL_OBSERVATION_SIZE: usize = domain::LOCAL_OBSERVATION_SIZE;

/// Centralized critic state width retained by current checkpoints.
pub const GLOBAL_STATE_SIZE: usize = domain::GLOBAL_STATE_SIZE;

/// Maximum agent slots represented by current checkpoints.
pub const MAX_AGENTS: usize = domain::MAX_AGENTS;

/// Lower bounds for forward, turn, gaze, and attack axes.
pub const ACTION_LOW: [f32; 4] = domain::ACTION_LOW;

/// Upper bounds for forward, turn, gaze, and attack axes.
pub const ACTION_HIGH: [f32; 4] = domain::ACTION_HIGH;

/// Closed ecosystem curriculum used by inference clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EcosystemStage {
    /// One bunny foraging for food.
    Forage,
    /// One bunny sprinting to ephemeral food.
    Sprint,
    /// One bunny crossing a safe bridge over a lethal gorge.
    Gorge,
    /// One bunny balancing food and water.
    Survival,
    /// One bunny balancing needs and weather shelter.
    Shelter,
    /// Several bunnies competing for resources.
    Competition,
    /// Bunnies and foxes using separate policies.
    PredatorPrey,
    /// Predator-prey with solid obstacles and thorns.
    Obstacles,
}

impl EcosystemStage {
    /// Every stage in curriculum order.
    pub const ALL: [Self; 8] = [
        Self::Forage,
        Self::Sprint,
        Self::Gorge,
        Self::Survival,
        Self::Shelter,
        Self::Competition,
        Self::PredatorPrey,
        Self::Obstacles,
    ];
}

impl From<EcosystemStage> for CurriculumStage {
    fn from(stage: EcosystemStage) -> Self {
        // Preserve the public facade's declared curriculum order internally.
        match stage {
            EcosystemStage::Forage => Self::Forage,
            EcosystemStage::Sprint => Self::Sprint,
            EcosystemStage::Gorge => Self::Gorge,
            EcosystemStage::Survival => Self::Survival,
            EcosystemStage::Shelter => Self::Shelter,
            EcosystemStage::Competition => Self::Competition,
            EcosystemStage::PredatorPrey => Self::PredatorPrey,
            EcosystemStage::Obstacles => Self::Obstacles,
        }
    }
}

/// Biological policy role for one agent observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EcosystemSpecies {
    /// Herbivore controlled by the bunny policy.
    Bunny,
    /// Predator controlled by the fox policy.
    Fox,
}

impl From<Species> for EcosystemSpecies {
    fn from(species: Species) -> Self {
        match species {
            Species::Bunny => Self::Bunny,
            Species::Fox => Self::Fox,
        }
    }
}

/// Stable identity for one possible agent slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EcosystemAgentId(u16);

impl From<AgentId> for EcosystemAgentId {
    fn from(id: AgentId) -> Self {
        Self(id.0)
    }
}

impl From<EcosystemAgentId> for AgentId {
    fn from(id: EcosystemAgentId) -> Self {
        Self(id.0)
    }
}

/// Validated continuous action in forward, turn, gaze, and attack order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcosystemAction(LocomotionAction);

impl TryFrom<[f32; 4]> for EcosystemAction {
    type Error = EcosystemRuntimeError;

    fn try_from([forward, turn, gaze, attack]: [f32; 4]) -> Result<Self, Self::Error> {
        LocomotionAction::new(forward, turn, gaze, attack)
            .map(Self)
            .map_err(EcosystemRuntimeError::action)
    }
}

/// Borrowed actor input at the current world instant.
#[derive(Debug, Clone, Copy)]
pub struct EcosystemObservation<'a> {
    /// Stable agent identity.
    pub id: EcosystemAgentId,
    /// Policy role that consumes this observation.
    pub species: EcosystemSpecies,
    /// Fixed-width local actor observation.
    pub observation: &'a [f32; LOCAL_OBSERVATION_SIZE],
}

/// Result of one atomic joint action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EcosystemStep {
    /// Whether no further action is valid in this episode.
    pub is_done: bool,
    /// Stable identities whose trajectories continue.
    pub living_agents: Vec<EcosystemAgentId>,
}

/// Renderer-only biological role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualSpecies {
    /// Herbivore body.
    Bunny,
    /// Predator body.
    Fox,
}

impl From<Species> for VisualSpecies {
    fn from(species: Species) -> Self {
        match species {
            Species::Bunny => Self::Bunny,
            Species::Fox => Self::Fox,
        }
    }
}

/// Renderer-only world object category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualObjectType {
    /// Consumable bunny food.
    Food,
    /// Refillable drinking well.
    Well,
    /// Solid circular tree.
    Tree,
    /// Solid rock.
    Rock,
    /// Traversable damaging thorn bush.
    Thorn,
    /// Passive protection zone.
    Shelter,
}

impl From<VisualObjectKind> for VisualObjectType {
    fn from(kind: VisualObjectKind) -> Self {
        match kind {
            VisualObjectKind::Food => Self::Food,
            VisualObjectKind::Well => Self::Well,
            VisualObjectKind::Tree => Self::Tree,
            VisualObjectKind::Rock => Self::Rock,
            VisualObjectKind::Thorn => Self::Thorn,
            VisualObjectKind::Shelter => Self::Shelter,
        }
    }
}

/// Renderer-only agent projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualAgent {
    /// Policy role used for body styling.
    pub species: VisualSpecies,
    /// World position in simulation units.
    pub position: [f32; 2],
    /// Counterclockwise heading in radians.
    pub heading: f32,
    /// Whether the body can still act.
    pub is_alive: bool,
}

/// Renderer-only perception segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualRay {
    /// Ray origin in simulation units.
    pub start: [f32; 2],
    /// Hit point or maximum-range endpoint.
    pub end: [f32; 2],
}

/// Renderer-only resource or obstacle projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualObject {
    /// Semantic object category.
    pub kind: VisualObjectType,
    /// World position in simulation units.
    pub position: [f32; 2],
    /// Approximate visual and collision radius.
    pub radius: f32,
}

/// Complete read-only world projection for inference viewers.
#[derive(Debug, Clone, PartialEq)]
pub struct EcosystemVisualSnapshot {
    /// Completed joint steps.
    pub step: u32,
    /// Number of agents that can still act.
    pub living_agents: usize,
    /// Half extent of the square playable area.
    pub map_half_extent: f32,
    /// Agent bodies in stable identity order.
    pub agents: Vec<VisualAgent>,
    /// Resources and obstacles.
    pub objects: Vec<VisualObject>,
    /// Perception sectors consumed by actor observations.
    pub rays: Vec<VisualRay>,
}

impl From<VisualWorldSnapshot> for EcosystemVisualSnapshot {
    fn from(snapshot: VisualWorldSnapshot) -> Self {
        Self {
            step: snapshot.summary.step,
            living_agents: snapshot.summary.living_agents,
            map_half_extent: snapshot.map_half_extent,
            agents: snapshot
                .agents
                .into_iter()
                .map(|agent| VisualAgent {
                    species: agent.species.into(),
                    position: agent.position,
                    heading: agent.heading,
                    is_alive: agent.is_alive,
                })
                .collect(),
            objects: snapshot
                .objects
                .into_iter()
                .map(|object| VisualObject {
                    kind: object.kind.into(),
                    position: object.position,
                    radius: object.radius,
                })
                .collect(),
            rays: snapshot
                .rays
                .into_iter()
                .map(|ray| VisualRay {
                    start: ray.start,
                    end: ray.end,
                })
                .collect(),
        }
    }
}

/// One deterministic ecosystem episode with a narrow inference interface.
#[derive(Debug)]
pub struct EcosystemRuntime {
    /// Validated stage settings retained for the fixed-step contract.
    config: SimulationConfig,
    /// Authoritative physics and environment state.
    ecosystem: Ecosystem,
    /// Actor observations at the current world instant.
    state: EcosystemState,
    /// Read-only renderer projection at the current world instant.
    snapshot: EcosystemVisualSnapshot,
}

impl EcosystemRuntime {
    /// Construct a deterministic episode for one stage and seed.
    ///
    /// # Errors
    ///
    /// Returns an error when stage configuration or procedural placement fails.
    pub fn new(stage: EcosystemStage, seed: u64) -> Result<Self, EcosystemRuntimeError> {
        let config = SimulationConfig::for_stage(stage.into())?;
        let mut ecosystem = Ecosystem::new(config.clone(), seed)?;
        let ecosystem_state = ecosystem.state();
        let snapshot = ecosystem.visual_snapshot().into();
        Ok(Self {
            config,
            ecosystem,
            state: ecosystem_state,
            snapshot,
        })
    }

    /// Return the exact simulated seconds per joint action.
    pub const fn time_step(&self) -> f32 {
        self.config.time_step
    }

    /// Iterate current actor observations in stable identity order.
    pub fn observations(&self) -> impl Iterator<Item = EcosystemObservation<'_>> {
        self.state
            .agents
            .iter()
            .map(|(id, species, observation)| EcosystemObservation {
                id: (*id).into(),
                species: (*species).into(),
                observation,
            })
    }

    /// Return the current renderer-only projection.
    pub const fn visual_snapshot(&self) -> &EcosystemVisualSnapshot {
        &self.snapshot
    }

    /// Apply one action for every living agent and update both projections.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, stale, or otherwise invalid actions.
    pub fn step(
        &mut self,
        actions: &[(EcosystemAgentId, EcosystemAction)],
    ) -> Result<EcosystemStep, EcosystemRuntimeError> {
        let actions = actions
            .iter()
            .map(|(id, action)| ((*id).into(), action.0))
            .collect::<Vec<_>>();
        let result = self.ecosystem.step(&actions)?;
        let living_agents = result
            .agents
            .iter()
            .filter(|agent| agent.status == EpisodeStatus::Continuing)
            .map(|agent| agent.id.into())
            .collect();
        if !result.is_done {
            self.state.agents = result
                .agents
                .into_iter()
                .filter(|agent| agent.status == EpisodeStatus::Continuing)
                .map(|agent| (agent.id, agent.species, agent.observation))
                .collect();
            self.state.global_state = result.global_state;
        }
        self.snapshot = self.ecosystem.visual_snapshot().into();
        Ok(EcosystemStep {
            is_done: result.is_done,
            living_agents,
        })
    }
}

/// Portable simulation construction, action, or progression failure.
#[derive(Debug)]
pub struct EcosystemRuntimeError(EcosystemRuntimeErrorKind);

/// Internal error category preserving the concrete implementation source.
#[derive(Debug)]
enum EcosystemRuntimeErrorKind {
    /// Stage configuration failed validation.
    Config(domain::ConfigError),
    /// Procedural construction or progression failed.
    Simulation(SimulationError),
    /// One continuous action axis was invalid.
    Action(ActionError),
}

impl EcosystemRuntimeError {
    /// Wrap one invalid action without exposing the private domain type.
    const fn action(error: ActionError) -> Self {
        Self(EcosystemRuntimeErrorKind::Action(error))
    }
}

impl fmt::Display for EcosystemRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            EcosystemRuntimeErrorKind::Config(error) => error.fmt(formatter),
            EcosystemRuntimeErrorKind::Simulation(error) => error.fmt(formatter),
            EcosystemRuntimeErrorKind::Action(error) => error.fmt(formatter),
        }
    }
}

impl Error for EcosystemRuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.0 {
            EcosystemRuntimeErrorKind::Config(error) => Some(error),
            EcosystemRuntimeErrorKind::Simulation(error) => Some(error),
            EcosystemRuntimeErrorKind::Action(error) => Some(error),
        }
    }
}

impl From<domain::ConfigError> for EcosystemRuntimeError {
    fn from(error: domain::ConfigError) -> Self {
        Self(EcosystemRuntimeErrorKind::Config(error))
    }
}

impl From<SimulationError> for EcosystemRuntimeError {
    fn from(error: SimulationError) -> Self {
        Self(EcosystemRuntimeErrorKind::Simulation(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_stage_constructs_and_advances_through_the_portable_facade() {
        for stage in EcosystemStage::ALL {
            let mut runtime = EcosystemRuntime::new(stage, 907).expect("stage should construct");
            let actions = runtime
                .observations()
                .map(|observation| {
                    (
                        observation.id,
                        EcosystemAction::try_from([0.0; 4]).expect("zero action is valid"),
                    )
                })
                .collect::<Vec<_>>();
            let step = runtime.step(&actions).expect("stage should advance");
            assert!(!step.is_done);
            assert_eq!(runtime.visual_snapshot().step, 1);
        }
    }
}
