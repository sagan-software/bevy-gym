//! Stable domain vocabulary and tensor-shape contracts.

use std::error::Error;
use std::fmt;

use bevy_gym::EpisodeStatus;

/// Number of uniform all-around semantic sectors emitted by every agent.
pub(super) const RAY_COUNT: usize = 36;

/// Semantic channels encoded after every ray distance and presence value.
pub(super) const RAY_KIND_COUNT: usize = 7;

/// Scalar features reserved for proprioception.
pub(super) const PROPRIOCEPTION_SIZE: usize = 11;

/// One-hot curriculum lesson width.
pub(super) const LESSON_COUNT: usize = 4;

/// Stable local actor input width across every curriculum lesson.
pub(super) const LOCAL_OBSERVATION_SIZE: usize =
    PROPRIOCEPTION_SIZE + RAY_COUNT * (2 + RAY_KIND_COUNT) + LESSON_COUNT;

/// Maximum possible agents represented in centralized training state.
pub(super) const MAX_AGENTS: usize = 12;

/// Maximum live food slots represented in centralized training state.
pub(super) const MAX_FOOD: usize = 24;

/// Maximum solid and thorn slots represented in centralized training state.
pub(super) const MAX_OBSTACLES: usize = 24;

/// Scalar features in one padded centralized agent slot.
pub(super) const GLOBAL_AGENT_FEATURES: usize = 11;

/// Scalar features in one padded centralized food slot.
pub(super) const GLOBAL_FOOD_FEATURES: usize = 3;

/// Scalar features in one padded centralized obstacle slot.
pub(super) const GLOBAL_OBSTACLE_FEATURES: usize = 6;

/// Stable critic input width across every curriculum lesson.
pub(super) const GLOBAL_STATE_SIZE: usize = MAX_AGENTS * GLOBAL_AGENT_FEATURES
    + 4
    + MAX_FOOD * GLOBAL_FOOD_FEATURES
    + MAX_OBSTACLES * GLOBAL_OBSTACLE_FEATURES
    + 1
    + LESSON_COUNT;

/// Fixed 360-degree sector centers in radians, alternating around forward.
pub(super) const RAY_ANGLES: [f32; RAY_COUNT] = [
    0.0,
    10.0_f32.to_radians(),
    (-10.0_f32).to_radians(),
    20.0_f32.to_radians(),
    (-20.0_f32).to_radians(),
    30.0_f32.to_radians(),
    (-30.0_f32).to_radians(),
    40.0_f32.to_radians(),
    (-40.0_f32).to_radians(),
    50.0_f32.to_radians(),
    (-50.0_f32).to_radians(),
    60.0_f32.to_radians(),
    (-60.0_f32).to_radians(),
    70.0_f32.to_radians(),
    (-70.0_f32).to_radians(),
    80.0_f32.to_radians(),
    (-80.0_f32).to_radians(),
    90.0_f32.to_radians(),
    (-90.0_f32).to_radians(),
    100.0_f32.to_radians(),
    (-100.0_f32).to_radians(),
    110.0_f32.to_radians(),
    (-110.0_f32).to_radians(),
    120.0_f32.to_radians(),
    (-120.0_f32).to_radians(),
    130.0_f32.to_radians(),
    (-130.0_f32).to_radians(),
    140.0_f32.to_radians(),
    (-140.0_f32).to_radians(),
    150.0_f32.to_radians(),
    (-150.0_f32).to_radians(),
    160.0_f32.to_radians(),
    (-160.0_f32).to_radians(),
    170.0_f32.to_radians(),
    (-170.0_f32).to_radians(),
    180.0_f32.to_radians(),
];

/// Ordered ecosystem curriculum lessons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CurriculumStage {
    /// One bunny with food and one refillable well.
    Survival,

    /// Several bunnies competing for the same resources.
    Competition,

    /// Bunnies and foxes with species-specific food.
    PredatorPrey,

    /// Predator-prey plus solid obstacles and damaging thorns.
    Obstacles,
}

impl CurriculumStage {
    /// Every lesson in progression order.
    pub(crate) const ALL: [Self; LESSON_COUNT] = [
        Self::Survival,
        Self::Competition,
        Self::PredatorPrey,
        Self::Obstacles,
    ];

    /// Return the stable stage key used by commands and run directories.
    pub(crate) const fn as_key(self) -> &'static str {
        match self {
            Self::Survival => "survival",
            Self::Competition => "competition",
            Self::PredatorPrey => "predator-prey",
            Self::Obstacles => "obstacles",
        }
    }

    /// Return a human-readable stage title.
    pub(crate) const fn title(self) -> &'static str {
        match self {
            Self::Survival => "Single-agent ecosystem survival",
            Self::Competition => "Multi-agent resource competition",
            Self::PredatorPrey => "Predator-prey ecosystem",
            Self::Obstacles => "Obstacle and thorn ecosystem",
        }
    }

    /// Return the zero-based one-hot channel index.
    pub(super) const fn index(self) -> usize {
        match self {
            Self::Survival => 0,
            Self::Competition => 1,
            Self::PredatorPrey => 2,
            Self::Obstacles => 3,
        }
    }
}

/// Biological role controlling compatible food and policy sharing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Species {
    /// Herbivore that consumes spawned food.
    Bunny,

    /// Predator that consumes bunnies.
    Fox,
}

impl Species {
    /// Return the stable centralized and local one-hot index.
    pub(super) const fn index(self) -> usize {
        match self {
            Self::Bunny => 0,
            Self::Fox => 1,
        }
    }
}

/// Stable identity for one possible agent slot within an ecosystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct AgentId(pub(super) u16);

/// Natural cause that ended an agent trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum DeathCause {
    /// Hunger damage exhausted hit points.
    Starvation,

    /// Thirst damage exhausted hit points.
    Dehydration,

    /// Combined hunger and thirst damage exhausted hit points.
    Deprivation,

    /// A fox consumed a bunny.
    Predation,

    /// Thorn damage exhausted hit points.
    Thorns,

    /// Non-finite or escaped physics state invalidated the trajectory.
    InvalidPhysics,
}

/// Semantic meaning of the closest collider hit by one eye ray.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum PerceptKind {
    /// Spawned bunny food.
    Food,

    /// Refillable water-well interaction area.
    Well,

    /// Living bunny body.
    Bunny,

    /// Living fox body.
    Fox,

    /// Tree or rock.
    SolidObstacle,

    /// Traversable damaging thorn bush.
    Thorn,

    /// Static map boundary.
    Boundary,
}

impl PerceptKind {
    /// Return the stable ray one-hot channel index.
    pub(super) const fn index(self) -> usize {
        match self {
            Self::Food => 0,
            Self::Well => 1,
            Self::Bunny => 2,
            Self::Fox => 3,
            Self::SolidObstacle => 4,
            Self::Thorn => 5,
            Self::Boundary => 6,
        }
    }
}

/// Bounded continuous locomotion command applied for one physics step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct LocomotionAction {
    /// Forward or reverse acceleration in `[-1, 1]`.
    pub(super) forward: f32,

    /// Counterclockwise turn rate in `[-1, 1]`.
    pub(super) turn: f32,
}

impl LocomotionAction {
    /// Construct a finite bounded action.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if either axis is non-finite or outside
    /// `[-1, 1]`.
    pub(super) fn new(forward: f32, turn: f32) -> Result<Self, ActionError> {
        validate_action_axis("forward", forward)?;
        validate_action_axis("turn", turn)?;
        Ok(Self { forward, turn })
    }
}

/// Invalid continuous action axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ActionError {
    /// Axis that violated the action contract.
    field: &'static str,

    /// Rejected scalar value.
    value: f32,
}

impl fmt::Display for ActionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "action {} must be finite and in -1..=1; got {}",
            self.field, self.value
        )
    }
}

impl Error for ActionError {}

/// Validate one continuous action axis at the actor boundary.
fn validate_action_axis(field: &'static str, value: f32) -> Result<(), ActionError> {
    if value.is_finite() && (-1.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ActionError { field, value })
    }
}

/// Fixed-shape local actor observation.
pub(super) type LocalObservation = [f32; LOCAL_OBSERVATION_SIZE];

/// Fixed-shape centralized training-only critic state.
pub(super) type GlobalState = [f32; GLOBAL_STATE_SIZE];

/// Valid number of evenly spaced perception sectors sampled by each agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PerceptionRayCount(u8);

impl PerceptionRayCount {
    /// Smallest supported all-around perception profile.
    pub(super) const MIN: u8 = 1;

    /// Fixed actor-tensor capacity reserved for perception sectors.
    pub(super) const MAX: u8 = RAY_COUNT as u8;

    /// Return the validated active-sector count.
    pub(super) const fn get(self) -> u8 {
        self.0
    }
}

impl Default for PerceptionRayCount {
    fn default() -> Self {
        Self(Self::MAX)
    }
}

impl TryFrom<u8> for PerceptionRayCount {
    type Error = ConfigError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        // Reject counts above the reserved tensor slots so every accepted value
        // can be applied without rebuilding a policy.
        if (Self::MIN..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ConfigError::new(
                "perception_ray_count",
                "must be in 1..=36 sectors",
            ))
        }
    }
}

/// Runtime-adjustable perception, reward, physiology, and horizon settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ExperimentTuning {
    /// Evenly spaced semantic sectors sampled within the fixed tensor capacity.
    pub(super) perception_ray_count: PerceptionRayCount,

    /// Reward earned for each simulated second alive.
    pub(super) survival_reward_per_second: f32,

    /// Reward earned for one compatible food or prey event.
    pub(super) food_reward: f32,

    /// Reward earned per conserved unit of well water consumed.
    pub(super) water_reward_per_unit: f32,

    /// Normalized health and hit-point fraction assigned at reset.
    pub(super) initial_health_fraction: f32,

    /// Multiplier applied to hunger and thirst drain rates.
    pub(super) need_drain_multiplier: f32,

    /// Multiplier applied to deprivation and thorn damage rates.
    pub(super) damage_multiplier: f32,

    /// Maximum joint steps before time-limit truncation.
    pub(super) episode_step_limit: u32,
}

impl Default for ExperimentTuning {
    fn default() -> Self {
        // Preserve the established survival task unless the user moves a demo
        // control or supplies a shorter command-line horizon.
        Self {
            perception_ray_count: PerceptionRayCount::default(),
            survival_reward_per_second: 1.0,
            food_reward: 0.0,
            water_reward_per_unit: 0.0,
            initial_health_fraction: 1.0,
            need_drain_multiplier: 1.0,
            damage_multiplier: 1.0,
            episode_step_limit: 1_200,
        }
    }
}

impl ExperimentTuning {
    /// Validate the bounded playground surface before applying it to a world.
    fn validate(self) -> Result<(), ConfigError> {
        // Bounds prevent an accidental slider value from creating non-finite
        // rewards, immortal agents, or an impractically long demo episode.
        if !self.survival_reward_per_second.is_finite()
            || !(0.0..=5.0).contains(&self.survival_reward_per_second)
            || !self.food_reward.is_finite()
            || !(0.0..=20.0).contains(&self.food_reward)
            || !self.water_reward_per_unit.is_finite()
            || !(0.0..=10.0).contains(&self.water_reward_per_unit)
        {
            return Err(ConfigError::new(
                "reward_tuning",
                "reward weights must be finite and within their demo bounds",
            ));
        }
        if !self.initial_health_fraction.is_finite()
            || !(0.1..=1.0).contains(&self.initial_health_fraction)
            || !self.need_drain_multiplier.is_finite()
            || !(0.25..=4.0).contains(&self.need_drain_multiplier)
            || !self.damage_multiplier.is_finite()
            || !(0.25..=4.0).contains(&self.damage_multiplier)
        {
            return Err(ConfigError::new(
                "physiology_tuning",
                "health and rate multipliers must be finite and within their demo bounds",
            ));
        }
        if !(100..=3_000).contains(&self.episode_step_limit) {
            return Err(ConfigError::new(
                "episode_step_limit",
                "must be in 100..=3000 steps",
            ));
        }
        Ok(())
    }
}

/// Configuration shared by simulation, rendering, evaluation, and artifacts.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SimulationConfig {
    /// Curriculum lesson represented by this world.
    pub(super) stage: CurriculumStage,

    /// Number of possible bunny slots.
    pub(super) bunny_count: usize,

    /// Number of possible fox slots.
    pub(super) fox_count: usize,

    /// Half width and half height of the square playable area.
    pub(super) map_half_extent: f32,

    /// Exact simulated seconds per joint action.
    pub(super) time_step: f32,

    /// External time-limit step count.
    pub(super) max_steps: u32,

    /// Food items placed during reset.
    pub(super) initial_food: usize,

    /// Maximum simultaneous food items.
    pub(super) max_food: usize,

    /// Fixed steps between food-spawn attempts.
    pub(super) food_spawn_interval: u32,

    /// Maximum well-water units.
    pub(super) well_capacity: f32,

    /// Well-water units restored per simulated second.
    pub(super) well_refill_rate: f32,

    /// Well-water units consumed per drinking agent-second.
    pub(super) well_drink_rate: f32,

    /// Number of solid trees and rocks.
    pub(super) solid_obstacles: usize,

    /// Number of traversable thorn bushes.
    pub(super) thorn_obstacles: usize,

    /// Reward earned for each simulated second alive.
    pub(super) survival_reward_per_second: f32,

    /// Reward earned for one compatible food or prey event.
    pub(super) food_reward: f32,

    /// Reward earned per conserved unit of well water consumed.
    pub(super) water_reward_per_unit: f32,

    /// Normalized health and hit-point fraction assigned at reset.
    pub(super) initial_health_fraction: f32,

    /// Multiplier applied to hunger and thirst drain rates.
    pub(super) need_drain_multiplier: f32,

    /// Multiplier applied to deprivation and thorn damage rates.
    pub(super) damage_multiplier: f32,

    /// Active perception sectors within the fixed actor tensor capacity.
    pub(super) perception_ray_count: PerceptionRayCount,
}

impl SimulationConfig {
    /// Construct and validate the default settings for one curriculum lesson.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if the stage defaults violate a fixed tensor or
    /// simulation constraint.
    pub(super) fn for_stage(stage: CurriculumStage) -> Result<Self, ConfigError> {
        // Derive every live tuning field from one default profile to prevent drift.
        let tuning = ExperimentTuning::default();
        let (
            bunny_count,
            fox_count,
            map_half_extent,
            initial_food,
            solid_obstacles,
            thorn_obstacles,
        ) = match stage {
            CurriculumStage::Survival => (1, 0, 10.0, 2, 0, 0),
            CurriculumStage::Competition => (4, 0, 14.0, 12, 0, 0),
            CurriculumStage::PredatorPrey => (6, 2, 20.0, 12, 0, 0),
            CurriculumStage::Obstacles => (6, 2, 25.0, 12, 12, 8),
        };
        let food_spawn_interval = match stage {
            CurriculumStage::Survival => 20,
            CurriculumStage::Competition
            | CurriculumStage::PredatorPrey
            | CurriculumStage::Obstacles => 30,
        };
        let max_food = match stage {
            CurriculumStage::Survival => 2,
            CurriculumStage::Competition
            | CurriculumStage::PredatorPrey
            | CurriculumStage::Obstacles => 20,
        };
        let config = Self {
            stage,
            bunny_count,
            fox_count,
            map_half_extent,
            time_step: 0.1,
            max_steps: tuning.episode_step_limit,
            initial_food,
            max_food,
            food_spawn_interval,
            well_capacity: 8.0,
            well_refill_rate: 0.12,
            well_drink_rate: 0.6,
            solid_obstacles,
            thorn_obstacles,
            survival_reward_per_second: tuning.survival_reward_per_second,
            food_reward: tuning.food_reward,
            water_reward_per_unit: tuning.water_reward_per_unit,
            initial_health_fraction: tuning.initial_health_fraction,
            need_drain_multiplier: tuning.need_drain_multiplier,
            damage_multiplier: tuning.damage_multiplier,
            perception_ray_count: tuning.perception_ray_count,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate runtime counts against physics and fixed tensor capacities.
    fn validate(&self) -> Result<(), ConfigError> {
        let agent_count = self.bunny_count.saturating_add(self.fox_count);
        if agent_count == 0 || agent_count > MAX_AGENTS {
            return Err(ConfigError::new("agent_count", "must be in 1..=12"));
        }
        if self.max_food == 0 || self.max_food > MAX_FOOD || self.initial_food > self.max_food {
            return Err(ConfigError::new(
                "food_count",
                "initial <= maximum and maximum must be in 1..=24",
            ));
        }
        if self.solid_obstacles.saturating_add(self.thorn_obstacles) > MAX_OBSTACLES {
            return Err(ConfigError::new(
                "obstacle_count",
                "must fit the 24-slot critic state",
            ));
        }
        if !self.map_half_extent.is_finite() || self.map_half_extent <= 5.0 {
            return Err(ConfigError::new(
                "map_half_extent",
                "must be finite and greater than five",
            ));
        }
        if !self.time_step.is_finite() || self.time_step <= 0.0 {
            return Err(ConfigError::new("time_step", "must be finite and positive"));
        }
        if self.max_steps == 0 || self.food_spawn_interval == 0 {
            return Err(ConfigError::new("step_counts", "must be greater than zero"));
        }
        if !self.well_capacity.is_finite()
            || self.well_capacity <= 0.0
            || !self.well_refill_rate.is_finite()
            || self.well_refill_rate < 0.0
            || !self.well_drink_rate.is_finite()
            || self.well_drink_rate <= 0.0
        {
            return Err(ConfigError::new(
                "well",
                "capacity and drink rate must be positive and refill non-negative",
            ));
        }
        Ok(())
    }

    /// Apply one validated playground profile without changing tensor shapes.
    pub(super) fn apply_experiment_tuning(
        &mut self,
        tuning: ExperimentTuning,
    ) -> Result<(), ConfigError> {
        // Validate before mutation so a rejected profile leaves the prior
        // simulation contract intact.
        tuning.validate()?;
        self.survival_reward_per_second = tuning.survival_reward_per_second;
        self.food_reward = tuning.food_reward;
        self.water_reward_per_unit = tuning.water_reward_per_unit;
        self.initial_health_fraction = tuning.initial_health_fraction;
        self.need_drain_multiplier = tuning.need_drain_multiplier;
        self.damage_multiplier = tuning.damage_multiplier;
        self.perception_ray_count = tuning.perception_ray_count;
        self.max_steps = tuning.episode_step_limit;
        self.validate()
    }

    /// Return the complete currently applied playground profile.
    pub(super) const fn experiment_tuning(&self) -> ExperimentTuning {
        // Derive this view from authoritative simulation fields so the HUD and
        // metrics cannot report a profile that the world does not use.
        ExperimentTuning {
            perception_ray_count: self.perception_ray_count,
            survival_reward_per_second: self.survival_reward_per_second,
            food_reward: self.food_reward,
            water_reward_per_unit: self.water_reward_per_unit,
            initial_health_fraction: self.initial_health_fraction,
            need_drain_multiplier: self.need_drain_multiplier,
            damage_multiplier: self.damage_multiplier,
            episode_step_limit: self.max_steps,
        }
    }

    /// Return the fixed possible-agent population.
    pub(super) const fn agent_count(&self) -> usize {
        self.bunny_count + self.fox_count
    }
}

/// Invalid simulation configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ConfigError {
    /// Invalid configuration field or group.
    field: &'static str,

    /// Stable human-readable constraint.
    reason: &'static str,
}

impl ConfigError {
    /// Construct one field-scoped configuration error.
    const fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid {}: {}", self.field, self.reason)
    }
}

impl Error for ConfigError {}

/// One post-step result for a previously living agent.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct AgentStep {
    /// Agent whose trajectory advanced.
    pub(super) id: AgentId,

    /// Species policy that owns the trajectory.
    pub(super) species: Species,

    /// Local post-action actor observation.
    pub(super) observation: LocalObservation,

    /// Configured survival and resource reward for this transition.
    pub(super) reward: f64,

    /// Natural death, time-limit truncation, or continuing status.
    pub(super) status: EpisodeStatus,

    /// Natural death cause when status is terminal.
    pub(super) death_cause: Option<DeathCause>,
}

/// Atomic joint-step result for one ecosystem instance.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct JointStep {
    /// Results for every agent alive before the action.
    pub(super) agents: Vec<AgentStep>,

    /// Training-only global state after the action.
    pub(super) global_state: GlobalState,

    /// Whether no further action is valid in this episode.
    pub(super) is_done: bool,
}

/// Current decentralized observations plus training-only centralized state.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EcosystemState {
    /// Local observation and species for every living agent.
    pub(super) agents: Vec<(AgentId, Species, LocalObservation)>,

    /// Centralized critic state for the same world instant.
    pub(super) global_state: GlobalState,
}

/// Small world snapshot used by smoke output and later rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct EcosystemSnapshot {
    /// Completed joint steps.
    pub(super) step: u32,

    /// Number of agents still earning survival reward.
    pub(super) living_agents: usize,

    /// Number of live food sensors.
    pub(super) food_count: usize,

    /// Current well-water units.
    pub(super) well_water: f32,
}

/// Renderer-only kind for a static or consumable world object.
#[cfg(feature = "render")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VisualObjectKind {
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
}

/// Renderer-only state for one agent body.
#[cfg(feature = "render")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VisualAgent {
    /// Stable agent identity.
    pub(super) id: AgentId,

    /// Species controlling its policy and color.
    pub(super) species: Species,

    /// World position in simulation units.
    pub(super) position: [f32; 2],

    /// Counterclockwise heading in radians.
    pub(super) heading: f32,

    /// Whether the agent can still act.
    pub(super) is_alive: bool,

    /// Current normalized hunger reserve.
    pub(super) hunger: f32,

    /// Current normalized thirst reserve.
    pub(super) thirst: f32,

    /// Current hit points.
    pub(super) hit_points: f32,
}

/// One exact actor perception sector projected into world coordinates.
#[cfg(feature = "render")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VisualRay {
    /// Stable identity of the observing agent.
    pub(super) agent: AgentId,

    /// Ray origin in simulation units.
    pub(super) start: [f32; 2],

    /// Sampled hit point or maximum-range miss endpoint.
    pub(super) end: [f32; 2],

    /// Closest semantic hit, or none for a miss.
    pub(super) kind: Option<PerceptKind>,
}

/// Renderer-only state for one non-agent entity.
#[cfg(feature = "render")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VisualObject {
    /// Semantic object kind.
    pub(super) kind: VisualObjectKind,

    /// World position in simulation units.
    pub(super) position: [f32; 2],

    /// Approximate visual and collision radius.
    pub(super) radius: f32,
}

/// Complete read-only scene projection for watch and video modes.
#[cfg(feature = "render")]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct VisualWorldSnapshot {
    /// Compact counters shared with smoke output.
    pub(super) summary: EcosystemSnapshot,

    /// Half extent of the square playable area.
    pub(super) map_half_extent: f32,

    /// Current agent bodies in stable identity order.
    pub(super) agents: Vec<VisualAgent>,

    /// Current resources and obstacles.
    pub(super) objects: Vec<VisualObject>,

    /// Exact post-physics semantic rays consumed by actor observations.
    pub(super) rays: Vec<VisualRay>,
}

/// Final per-agent behavior evidence for evaluation and HUD summaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct AgentEpisodeMetrics {
    /// Stable agent identity.
    pub(super) id: AgentId,

    /// Biological policy role.
    pub(super) species: Species,

    /// Simulated lifetime through death or horizon.
    pub(super) lifetime_seconds: f32,

    /// Compatible food or prey consumed.
    pub(super) food_eaten: u32,

    /// Finite well-water units consumed.
    pub(super) water_consumed: f32,

    /// Bunnies consumed by a fox.
    pub(super) kills: u32,

    /// Hit points lost to thorn contact.
    pub(super) thorn_damage: f32,

    /// Agent-agent overlap contacts observed across physics steps.
    pub(super) collision_contacts: u32,

    /// Natural death cause, or none at a horizon.
    pub(super) death_cause: Option<DeathCause>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every lesson must preserve the direct-checkpoint tensor contract.
    #[test]
    fn all_stage_defaults_fit_fixed_tensor_capacities() {
        for stage in [
            CurriculumStage::Survival,
            CurriculumStage::Competition,
            CurriculumStage::PredatorPrey,
            CurriculumStage::Obstacles,
        ] {
            let config = SimulationConfig::for_stage(stage).expect("stage defaults are valid");
            assert!(config.agent_count() <= MAX_AGENTS);
            assert!(config.max_food <= MAX_FOOD);
            assert!(config.solid_obstacles + config.thorn_obstacles <= MAX_OBSTACLES);
        }
        assert_eq!(LOCAL_OBSERVATION_SIZE, 339);
        assert_eq!(GLOBAL_STATE_SIZE, 357);
    }

    /// Demo tuning must update every experiment lever as one validated profile.
    #[test]
    fn experiment_tuning_applies_reward_physiology_and_timeout() {
        // Change every lever together to detect an omitted assignment.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        let tuning = ExperimentTuning {
            perception_ray_count: PerceptionRayCount::try_from(8_u8).expect("eight rays fit"),
            survival_reward_per_second: 0.5,
            food_reward: 4.0,
            water_reward_per_unit: 2.0,
            initial_health_fraction: 0.4,
            need_drain_multiplier: 1.5,
            damage_multiplier: 2.0,
            episode_step_limit: 300,
        };

        config
            .apply_experiment_tuning(tuning)
            .expect("bounded tuning is valid");

        assert_eq!(config.experiment_tuning(), tuning);
        assert_eq!(config.max_steps, 300);
    }

    /// Unsafe or non-finite demo settings must not enter a simulation.
    #[test]
    fn experiment_tuning_rejects_invalid_values() {
        // Exercise a non-finite value that no bounded slider can produce.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        let invalid = ExperimentTuning {
            damage_multiplier: f32::NAN,
            ..ExperimentTuning::default()
        };

        assert!(config.apply_experiment_tuning(invalid).is_err());
    }

    /// Perception counts must stay within the fixed actor tensor capacity.
    #[test]
    fn perception_ray_count_validates_fixed_capacity() {
        // Check both rejected edges and one small demo-friendly profile.
        assert!(PerceptionRayCount::try_from(0_u8).is_err());
        assert!(PerceptionRayCount::try_from(37_u8).is_err());
        assert_eq!(
            PerceptionRayCount::try_from(4_u8)
                .expect("four rays fit the fixed capacity")
                .get(),
            4
        );
    }

    /// Invalid action scalars must not reach physics.
    #[test]
    fn action_boundary_rejects_non_finite_and_unbounded_values() {
        assert!(LocomotionAction::new(f32::NAN, 0.0).is_err());
        assert!(LocomotionAction::new(0.0, 1.01).is_err());
        assert_eq!(
            LocomotionAction::new(-1.0, 1.0).expect("closed endpoints are valid"),
            LocomotionAction {
                forward: -1.0,
                turn: 1.0,
            }
        );
    }
}
