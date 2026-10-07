//! Stable domain vocabulary and tensor-shape contracts.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::EpisodeStatus;

/// Fixed center-origin semantic-ray capacity emitted by every agent.
pub(super) const RAY_COUNT: usize = 24;

/// Semantic channels encoded after every ray distance value.
pub(super) const RAY_KIND_COUNT: usize = 10;

/// Distance plus the semantic one-hot channels stored for each perception ray.
pub(super) const RAY_FEATURE_SIZE: usize = 1 + RAY_KIND_COUNT;

/// First semantic one-hot channel within one perception ray.
pub(super) const RAY_KIND_START: usize = 1;

/// Scalar features before the fixed semantic-ray channels.
pub(super) const PROPRIOCEPTION_SIZE: usize = 11;

/// Signed body-forward velocity channel.
pub(super) const LOCAL_LONGITUDINAL_VELOCITY_INDEX: usize = 4;

/// Signed body-left velocity channel.
pub(super) const LOCAL_LATERAL_VELOCITY_INDEX: usize = 5;

/// Normalized conjugate gaze yaw relative to the body.
pub(super) const LOCAL_GAZE_YAW_INDEX: usize = 7;

/// Normalized protection reserve supplied by shelter use.
pub(super) const LOCAL_EXPOSURE_INDEX: usize = 8;

/// Shelter state: `-1` in active weather, `0` before weather, and `1` inside shelter.
pub(super) const LOCAL_IN_SHELTER_INDEX: usize = 9;

/// Normalized fixed-step delay before another interaction can start.
pub(super) const LOCAL_INTERACTION_COOLDOWN_INDEX: usize = 10;

/// One-hot curriculum lesson width.
pub(super) const LESSON_COUNT: usize = 8;

/// Stable local actor input width across every curriculum lesson.
pub(super) const LOCAL_OBSERVATION_SIZE: usize = PROPRIOCEPTION_SIZE + RAY_COUNT * RAY_FEATURE_SIZE;

/// Stable lower bounds for forward, turn, gaze, and attack controls.
pub(super) const ACTION_LOW: [f32; 4] = [-1.0; 4];

/// Stable upper bounds for forward, turn, gaze, and attack controls.
pub(super) const ACTION_HIGH: [f32; 4] = [1.0; 4];

/// Maximum possible agents represented in centralized training state.
pub(super) const MAX_AGENTS: usize = 12;

/// Environment lanes shown and trained as the survival 3x3 batch.
pub(super) const SURVIVAL_BATCH_ENVIRONMENTS: usize = 9;

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
    + 4
    + MAX_FOOD * GLOBAL_FOOD_FEATURES
    + MAX_OBSTACLES * GLOBAL_OBSTACLE_FEATURES
    + 1
    + LESSON_COUNT;

/// Center-origin ray angles, dense near gaze and sparse at the periphery.
pub(super) const RAY_ANGLES: [f32; RAY_COUNT] = [
    2.0_f32.to_radians(),
    (-2.0_f32).to_radians(),
    5.0_f32.to_radians(),
    (-5.0_f32).to_radians(),
    9.0_f32.to_radians(),
    (-9.0_f32).to_radians(),
    14.0_f32.to_radians(),
    (-14.0_f32).to_radians(),
    20.0_f32.to_radians(),
    (-20.0_f32).to_radians(),
    28.0_f32.to_radians(),
    (-28.0_f32).to_radians(),
    37.0_f32.to_radians(),
    (-37.0_f32).to_radians(),
    47.0_f32.to_radians(),
    (-47.0_f32).to_radians(),
    55.0_f32.to_radians(),
    (-55.0_f32).to_radians(),
    62.0_f32.to_radians(),
    (-62.0_f32).to_radians(),
    68.0_f32.to_radians(),
    (-68.0_f32).to_radians(),
    73.0_f32.to_radians(),
    (-73.0_f32).to_radians(),
];

/// Ordered ecosystem curriculum lessons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CurriculumStage {
    /// One bunny learning to perceive, approach, and eat food.
    Forage,

    /// One bunny learning to reach visible food before it disappears.
    Sprint,

    /// One bunny retaining sprinting while routing across a safe bridge.
    Gorge,

    /// One bunny with food and one refillable well.
    Survival,

    /// One bunny retaining homeostasis while seeking weather shelter.
    Shelter,

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
        Self::Forage,
        Self::Sprint,
        Self::Gorge,
        Self::Survival,
        Self::Shelter,
        Self::Competition,
        Self::PredatorPrey,
        Self::Obstacles,
    ];

    /// Return the stable stage key used by commands and run directories.
    pub(crate) const fn as_key(self) -> &'static str {
        // Keep artifact keys aligned with the executable stage names.
        match self {
            Self::Forage => "forage",
            Self::Sprint => "sprint",
            Self::Gorge => "gorge",
            Self::Survival => "survival",
            Self::Shelter => "shelter",
            Self::Competition => "competition",
            Self::PredatorPrey => "predator-prey",
            Self::Obstacles => "obstacles",
        }
    }

    /// Return a human-readable stage title.
    pub(crate) const fn title(self) -> &'static str {
        // Keep display text centralized for training, watch, and video modes.
        match self {
            Self::Forage => "Single-agent food foraging",
            Self::Sprint => "Ephemeral-food sprinting",
            Self::Gorge => "Gorge bridge crossing",
            Self::Survival => "Single-agent ecosystem survival",
            Self::Shelter => "Single-agent shelter survival",
            Self::Competition => "Multi-agent resource competition",
            Self::PredatorPrey => "Predator-prey ecosystem",
            Self::Obstacles => "Obstacle and thorn ecosystem",
        }
    }

    /// Return the zero-based one-hot channel index.
    pub(super) const fn index(self) -> usize {
        // Preserve transfer order in the centralized critic stage channels.
        match self {
            Self::Forage => 0,
            Self::Sprint => 1,
            Self::Gorge => 2,
            Self::Survival => 3,
            Self::Shelter => 4,
            Self::Competition => 5,
            Self::PredatorPrey => 6,
            Self::Obstacles => 7,
        }
    }

    /// Return whether food uses the movement lesson's fixed expiry window.
    pub(super) const fn uses_ephemeral_food(self) -> bool {
        matches!(self, Self::Sprint | Self::Gorge)
    }

    /// Return whether this lesson requires food and water homeostasis.
    pub(super) const fn uses_hydration(self) -> bool {
        !matches!(self, Self::Forage | Self::Sprint | Self::Gorge)
    }
}

/// Closed forage reset distribution used by curriculum promotion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ForageDifficulty {
    /// Short targets near the body centerline.
    Foundation,
    /// Held-out left, right, near, and far target distribution.
    #[default]
    Expanded,
}

impl ForageDifficulty {
    /// Return the stable artifact and metric key.
    pub(super) const fn as_key(self) -> &'static str {
        match self {
            Self::Foundation => "foundation",
            Self::Expanded => "expanded",
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
    /// Zero satiation exhausted hit points.
    Starvation,

    /// Zero hydration exhausted hit points.
    Dehydration,

    /// Combined zero satiation and hydration exhausted hit points.
    Deprivation,

    /// A fox consumed a bunny.
    Predation,

    /// Thorn damage exhausted hit points.
    Thorns,

    /// A high-speed solid collision exhausted hit points.
    Collision,

    /// The agent left the bridge and fell into the gorge.
    Gorge,

    /// Concurrent deprivation and thorn damage exhausted hit points.
    CombinedDamage,

    /// Unprotected exposure exhausted hit points.
    Exposure,

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

    /// Passive protection zone.
    Shelter,

    /// Lethal floor below the gorge cliffs.
    Gorge,

    /// Solid bridge rail that identifies the safe crossing.
    Bridge,
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
            Self::Shelter => 7,
            Self::Gorge => 8,
            Self::Bridge => 9,
        }
    }
}

/// Bounded continuous locomotion command applied for one physics step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct LocomotionAction {
    /// Forward throttle control in `[-1, 1]`; `-1` stops and `1` reaches full throttle.
    pub(super) forward: f32,

    /// Counterclockwise turn rate in `[-1, 1]`.
    pub(super) turn: f32,

    /// Conjugate left/right eye yaw in `[-1, 1]`.
    pub(super) gaze: f32,

    /// Intent to use the overlapping mouth hitbox in `[-1, 1]`.
    pub(super) attack: f32,
}

impl LocomotionAction {
    /// Construct a finite bounded action.
    ///
    /// # Errors
    ///
    /// Returns [`ActionError`] if any axis is non-finite or outside
    /// `[-1, 1]`.
    pub(super) fn new(
        forward: f32,
        turn: f32,
        gaze: f32,
        attack: f32,
    ) -> Result<Self, ActionError> {
        validate_action_axis("forward", forward)?;
        validate_action_axis("turn", turn)?;
        validate_action_axis("gaze", gaze)?;
        validate_action_axis("attack", attack)?;
        Ok(Self {
            forward,
            turn,
            gaze,
            attack,
        })
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

/// Valid number of center-origin perception rays sampled by each agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PerceptionRayCount(u8);

impl PerceptionRayCount {
    /// Smallest symmetric profile around the gaze direction.
    pub(super) const MIN: u8 = 2;

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
        if (Self::MIN..=Self::MAX).contains(&value) && value.is_multiple_of(2) {
            Ok(Self(value))
        } else {
            Err(ConfigError::new(
                "perception_ray_count",
                "must be an even count in 2..=24 center-origin rays",
            ))
        }
    }
}

/// Runtime-adjustable perception, physiology, and horizon settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ExperimentTuning {
    /// Active forage reset distribution.
    pub(super) forage_difficulty: ForageDifficulty,

    /// Paired center-origin sectors sampled within the fixed tensor capacity.
    pub(super) perception_ray_count: PerceptionRayCount,

    /// Maximum discrete HP available to an agent.
    pub(super) maximum_hit_points: u8,

    /// Discrete HP assigned at reset.
    pub(super) initial_hit_points: u8,

    /// Maximum discrete satiation points.
    pub(super) maximum_satiation: u8,

    /// Discrete satiation points assigned at reset.
    pub(super) initial_satiation: u8,

    /// Maximum discrete hydration points.
    pub(super) maximum_hydration: u8,

    /// Discrete hydration points assigned at reset.
    pub(super) initial_hydration: u8,

    /// Simulated seconds between one-point need losses.
    pub(super) need_loss_interval_seconds: u16,

    /// Simulated seconds at zero satiation between HP losses.
    pub(super) starvation_damage_interval_seconds: u16,

    /// Simulated seconds at zero hydration between HP losses.
    pub(super) dehydration_damage_interval_seconds: u16,

    /// Extra need-clock progress while translating, as a percentage.
    pub(super) movement_need_cost_percent: u8,

    /// Multiplier applied to the base movement speed and acceleration.
    pub(super) movement_speed_multiplier: f32,

    /// Maximum conjugate eye yaw relative to the body, in degrees.
    pub(super) gaze_yaw_limit_degrees: f32,

    /// Maximum simulated episode duration.
    pub(super) episode_seconds: u16,
}

impl Default for ExperimentTuning {
    fn default() -> Self {
        // Start agents under visible resource pressure while retaining enough
        // exploration time for the demo policy to discover food and water.
        Self {
            forage_difficulty: ForageDifficulty::Expanded,
            perception_ray_count: PerceptionRayCount::default(),
            maximum_hit_points: 5,
            initial_hit_points: 5,
            maximum_satiation: 5,
            initial_satiation: 2,
            maximum_hydration: 5,
            initial_hydration: 2,
            need_loss_interval_seconds: 4,
            starvation_damage_interval_seconds: 2,
            dehydration_damage_interval_seconds: 2,
            movement_need_cost_percent: 25,
            movement_speed_multiplier: 1.0,
            gaze_yaw_limit_degrees: 30.0,
            episode_seconds: 20,
        }
    }
}

impl ExperimentTuning {
    /// Validate the bounded playground surface before applying it to a world.
    pub(super) fn validate(self) -> Result<(), ConfigError> {
        // Bounds prevent an accidental slider value from creating immortal
        // agents or an impractically long demo episode.
        if self.maximum_hit_points == 0
            || self.maximum_hit_points > 20
            || self.initial_hit_points == 0
            || self.initial_hit_points > self.maximum_hit_points
            || self.maximum_satiation == 0
            || self.maximum_satiation > 20
            || self.initial_satiation > self.maximum_satiation
            || self.maximum_hydration == 0
            || self.maximum_hydration > 20
            || self.initial_hydration > self.maximum_hydration
            || self.need_loss_interval_seconds == 0
            || self.need_loss_interval_seconds > 60
            || self.starvation_damage_interval_seconds == 0
            || self.starvation_damage_interval_seconds > 60
            || self.dehydration_damage_interval_seconds == 0
            || self.dehydration_damage_interval_seconds > 60
            || self.movement_need_cost_percent > 100
            || !self.movement_speed_multiplier.is_finite()
            || !(0.5..=2.0).contains(&self.movement_speed_multiplier)
            || !self.gaze_yaw_limit_degrees.is_finite()
            || !(10.0..=35.0).contains(&self.gaze_yaw_limit_degrees)
        {
            return Err(ConfigError::new(
                "physiology_tuning",
                "point, interval, movement, and gaze values must be within their demo bounds",
            ));
        }
        if !(5..=300).contains(&self.episode_seconds) {
            return Err(ConfigError::new(
                "episode_seconds",
                "must be in 5..=300 seconds",
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

    /// Active forage reset distribution.
    pub(super) forage_difficulty: ForageDifficulty,

    /// Number of possible bunny slots.
    pub(super) bunny_count: usize,

    /// Number of possible fox slots.
    pub(super) fox_count: usize,

    /// Half width and half height of the square playable area.
    pub(super) map_half_extent: f32,

    /// Exact simulated seconds per joint action.
    pub(super) time_step: f32,

    /// External time limit in simulated seconds.
    pub(super) episode_seconds: u16,

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

    /// Maximum discrete HP available to an agent.
    pub(super) maximum_hit_points: u8,

    /// Discrete HP assigned at reset.
    pub(super) initial_hit_points: u8,

    /// Maximum discrete satiation points.
    pub(super) maximum_satiation: u8,

    /// Discrete satiation points assigned at reset.
    pub(super) initial_satiation: u8,

    /// Maximum discrete hydration points.
    pub(super) maximum_hydration: u8,

    /// Discrete hydration points assigned at reset.
    pub(super) initial_hydration: u8,

    /// Simulated seconds between one-point need losses.
    pub(super) need_loss_interval_seconds: u16,

    /// Simulated seconds at zero satiation between HP losses.
    pub(super) starvation_damage_interval_seconds: u16,

    /// Simulated seconds at zero hydration between HP losses.
    pub(super) dehydration_damage_interval_seconds: u16,

    /// Extra need-clock progress while translating, as a percentage.
    pub(super) movement_need_cost_percent: u8,

    /// Multiplier applied to movement speed and acceleration.
    pub(super) movement_speed_multiplier: f32,

    /// Maximum conjugate gaze yaw, in degrees.
    pub(super) gaze_yaw_limit_degrees: f32,

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
            CurriculumStage::Forage | CurriculumStage::Sprint => (1, 0, 16.0, 1, 0, 0),
            CurriculumStage::Gorge => (1, 0, 12.0, 1, 0, 0),
            CurriculumStage::Survival => (1, 0, 10.0, 2, 5, 0),
            CurriculumStage::Shelter => (1, 0, 12.0, 2, 0, 0),
            CurriculumStage::Competition => (4, 0, 14.0, 2, 0, 0),
            CurriculumStage::PredatorPrey => (6, 2, 20.0, 6, 0, 0),
            CurriculumStage::Obstacles => (6, 2, 25.0, 6, 12, 8),
        };
        let food_spawn_interval = match stage {
            CurriculumStage::Forage
            | CurriculumStage::Sprint
            | CurriculumStage::Gorge
            | CurriculumStage::Survival => 20,
            CurriculumStage::Shelter
            | CurriculumStage::Competition
            | CurriculumStage::PredatorPrey
            | CurriculumStage::Obstacles => 30,
        };
        let max_food = match stage {
            CurriculumStage::Forage | CurriculumStage::Sprint | CurriculumStage::Gorge => 1,
            CurriculumStage::Survival => 6,
            CurriculumStage::Shelter => 2,
            CurriculumStage::Competition => 3,
            CurriculumStage::PredatorPrey | CurriculumStage::Obstacles => 8,
        };
        let (initial_satiation, initial_hydration) = if stage == CurriculumStage::Survival {
            (3, 3)
        } else {
            (tuning.initial_satiation, tuning.initial_hydration)
        };
        let need_loss_interval_seconds = if stage == CurriculumStage::Survival {
            60
        } else {
            tuning.need_loss_interval_seconds
        };
        let (starvation_damage_interval_seconds, dehydration_damage_interval_seconds) =
            if stage == CurriculumStage::Survival {
                (10, 10)
            } else {
                (
                    tuning.starvation_damage_interval_seconds,
                    tuning.dehydration_damage_interval_seconds,
                )
            };
        let config = Self {
            stage,
            forage_difficulty: tuning.forage_difficulty,
            bunny_count,
            fox_count,
            map_half_extent,
            time_step: 0.1,
            episode_seconds: tuning.episode_seconds,
            initial_food,
            max_food,
            food_spawn_interval,
            well_capacity: 8.0,
            well_refill_rate: if stage == CurriculumStage::Survival {
                0.24
            } else {
                0.12
            },
            well_drink_rate: 0.6,
            solid_obstacles,
            thorn_obstacles,
            maximum_hit_points: tuning.maximum_hit_points,
            initial_hit_points: tuning.initial_hit_points,
            maximum_satiation: tuning.maximum_satiation,
            initial_satiation,
            maximum_hydration: tuning.maximum_hydration,
            initial_hydration,
            need_loss_interval_seconds,
            starvation_damage_interval_seconds,
            dehydration_damage_interval_seconds,
            movement_need_cost_percent: tuning.movement_need_cost_percent,
            movement_speed_multiplier: tuning.movement_speed_multiplier,
            gaze_yaw_limit_degrees: tuning.gaze_yaw_limit_degrees,
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
        if self.episode_seconds == 0 || self.food_spawn_interval == 0 {
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
        self.maximum_hit_points = tuning.maximum_hit_points;
        self.forage_difficulty = tuning.forage_difficulty;
        self.initial_hit_points = tuning.initial_hit_points;
        self.maximum_satiation = tuning.maximum_satiation;
        self.initial_satiation = tuning.initial_satiation;
        self.maximum_hydration = tuning.maximum_hydration;
        self.initial_hydration = tuning.initial_hydration;
        self.need_loss_interval_seconds = tuning.need_loss_interval_seconds;
        self.starvation_damage_interval_seconds = tuning.starvation_damage_interval_seconds;
        self.dehydration_damage_interval_seconds = tuning.dehydration_damage_interval_seconds;
        self.movement_need_cost_percent = tuning.movement_need_cost_percent;
        self.movement_speed_multiplier = tuning.movement_speed_multiplier;
        self.gaze_yaw_limit_degrees = tuning.gaze_yaw_limit_degrees;
        self.perception_ray_count = tuning.perception_ray_count;
        self.episode_seconds = tuning.episode_seconds;
        self.validate()
    }

    /// Return the complete currently applied playground profile.
    pub(super) const fn experiment_tuning(&self) -> ExperimentTuning {
        // Derive this view from authoritative simulation fields so the HUD and
        // metrics cannot report a profile that the world does not use.
        ExperimentTuning {
            forage_difficulty: self.forage_difficulty,
            perception_ray_count: self.perception_ray_count,
            maximum_hit_points: self.maximum_hit_points,
            initial_hit_points: self.initial_hit_points,
            maximum_satiation: self.maximum_satiation,
            initial_satiation: self.initial_satiation,
            maximum_hydration: self.maximum_hydration,
            initial_hydration: self.initial_hydration,
            need_loss_interval_seconds: self.need_loss_interval_seconds,
            starvation_damage_interval_seconds: self.starvation_damage_interval_seconds,
            dehydration_damage_interval_seconds: self.dehydration_damage_interval_seconds,
            movement_need_cost_percent: self.movement_need_cost_percent,
            movement_speed_multiplier: self.movement_speed_multiplier,
            gaze_yaw_limit_degrees: self.gaze_yaw_limit_degrees,
            episode_seconds: self.episode_seconds,
        }
    }

    /// Return the joint-step horizon derived from seconds and fixed-step time.
    pub(super) fn max_steps(&self) -> u32 {
        self.steps_for_seconds(self.episode_seconds)
    }

    /// Convert simulated seconds to the nearest positive fixed-step count.
    pub(super) fn steps_for_seconds(&self, seconds: u16) -> u32 {
        let duration_nanos = Duration::from_secs(u64::from(seconds)).as_nanos();
        let step_nanos = Duration::from_secs_f32(self.time_step).as_nanos().max(1);
        let rounded_steps = duration_nanos.saturating_add(step_nanos / 2) / step_nanos;
        u32::try_from(rounded_steps).unwrap_or(u32::MAX).max(1)
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
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
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

    /// Passive protection zone.
    Shelter,
}

/// Renderer-only state for one agent body.
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
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

    /// Conjugate eye yaw relative to the head in radians.
    pub(super) gaze_yaw: f32,

    /// Whether the agent can still act.
    pub(super) is_alive: bool,

    /// Whether the forward mouth hitbox is active for this step.
    pub(super) attack_active: bool,

    /// Current satiation points.
    pub(super) satiation: u8,

    /// Current hydration points.
    pub(super) hydration: u8,

    /// Current hit points.
    pub(super) hit_points: u8,

    /// Current environmental protection reserve.
    pub(super) exposure: u8,

    /// Whether the agent currently occupies shelter.
    pub(super) in_shelter: bool,
}

/// One exact actor perception sector projected into world coordinates.
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
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
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VisualObject {
    /// Semantic object kind.
    pub(super) kind: VisualObjectKind,

    /// World position in simulation units.
    pub(super) position: [f32; 2],

    /// Approximate visual and collision radius.
    pub(super) radius: f32,
}

/// Renderer-only rectangular gorge and safe bridge dimensions.
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VisualGorge {
    /// Half-width of the gorge along the world X axis.
    pub(super) half_width: f32,

    /// Half-width of the safe bridge corridor along the world Y axis.
    pub(super) bridge_half_width: f32,
}

/// Complete read-only scene projection for watch and video modes.
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct VisualWorldSnapshot {
    /// Compact counters shared with smoke output.
    pub(super) summary: EcosystemSnapshot,

    /// Half extent of the square playable area.
    pub(super) map_half_extent: f32,

    /// Configured HP segment count.
    pub(super) maximum_hit_points: u8,

    /// Configured satiation segment count.
    pub(super) maximum_satiation: u8,

    /// Configured hydration segment count.
    pub(super) maximum_hydration: u8,

    /// Configured environmental protection segment count.
    pub(super) maximum_exposure: u8,

    /// Whether damaging weather has started in this episode.
    pub(super) weather_active: bool,

    /// Seeded simulated second at which damaging weather starts.
    pub(super) weather_onset_seconds: f32,

    /// Gorge and bridge geometry when the movement lesson enables it.
    pub(super) gorge: Option<VisualGorge>,

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

    /// Total undiscounted reward earned during the episode.
    pub(super) episode_return: f32,

    /// Hit points present at death or the episode horizon.
    pub(super) terminal_hit_points: u8,

    /// Satiation points present at death or the episode horizon.
    pub(super) terminal_satiation: u8,

    /// Hydration points present at death or the episode horizon.
    pub(super) terminal_hydration: u8,

    /// Compatible food or prey consumed.
    pub(super) food_eaten: u32,

    /// Finite well-water units consumed.
    pub(super) water_consumed: f32,

    /// Bunnies consumed by a fox.
    pub(super) kills: u32,

    /// Hit points lost to thorn contact.
    pub(super) thorn_damage: f32,

    /// Hit points lost to high-speed solid collisions.
    pub(super) collision_damage: f32,

    /// Hit points lost to excess food or water.
    pub(super) overconsumption_damage: f32,

    /// Agent-agent overlap contacts observed across physics steps.
    pub(super) collision_contacts: u32,

    /// Agent-contact steps in which world position changed.
    pub(super) contact_displacements: u32,

    /// Solid contacts that exceeded the post-solver penetration tolerance.
    pub(super) unresolved_solid_penetrations: u32,

    /// World-space distance traveled during the episode.
    pub(super) path_length: f32,

    /// Simulated seconds spent inside shelter.
    pub(super) shelter_seconds: f32,

    /// Whether shelter was entered before exposure reached zero.
    pub(super) sheltered_before_critical_exposure: bool,

    /// Simulated time of the first food or well contact.
    pub(super) first_resource_contact_seconds: Option<f32>,

    /// Visible-resource transitions eligible for approach scoring.
    pub(super) visible_resource_transitions: u32,

    /// Eligible transitions that reduced ray-derived resource distance.
    pub(super) resource_approach_transitions: u32,

    /// Natural death cause, or none at a horizon.
    pub(super) death_cause: Option<DeathCause>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suite must expose every curriculum stage in transfer order.
    #[test]
    fn curriculum_places_speed_and_bridge_lessons_before_survival() {
        // Assert the exact transfer chain consumed by curriculum orchestration.
        let keys: Vec<_> = CurriculumStage::ALL
            .into_iter()
            .map(CurriculumStage::as_key)
            .collect();

        assert_eq!(
            keys,
            [
                "forage",
                "sprint",
                "gorge",
                "survival",
                "shelter",
                "competition",
                "predator-prey",
                "obstacles",
            ]
        );
    }

    /// Every lesson must preserve the direct-checkpoint tensor contract.
    #[test]
    fn all_stage_defaults_fit_fixed_tensor_capacities() {
        for stage in CurriculumStage::ALL {
            let config = SimulationConfig::for_stage(stage).expect("stage defaults are valid");
            assert!(config.agent_count() <= MAX_AGENTS);
            assert!(config.max_food <= MAX_FOOD);
            assert!(config.solid_obstacles + config.thorn_obstacles <= MAX_OBSTACLES);
        }
        assert_eq!(LOCAL_OBSERVATION_SIZE, 275);
        assert_eq!(GLOBAL_STATE_SIZE, 365);
    }

    /// Demo tuning must update every experiment lever as one validated profile.
    #[test]
    fn experiment_tuning_applies_perception_physiology_and_timeout() {
        // Change every lever together to detect an omitted assignment.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        let tuning = ExperimentTuning {
            forage_difficulty: ForageDifficulty::Foundation,
            perception_ray_count: PerceptionRayCount::try_from(8_u8).expect("eight rays fit"),
            maximum_hit_points: 8,
            initial_hit_points: 4,
            maximum_satiation: 7,
            initial_satiation: 3,
            maximum_hydration: 6,
            initial_hydration: 2,
            need_loss_interval_seconds: 4,
            starvation_damage_interval_seconds: 2,
            dehydration_damage_interval_seconds: 1,
            movement_need_cost_percent: 50,
            movement_speed_multiplier: 1.25,
            gaze_yaw_limit_degrees: 25.0,
            episode_seconds: 30,
        };

        config
            .apply_experiment_tuning(tuning)
            .expect("bounded tuning is valid");

        assert_eq!(config.experiment_tuning(), tuning);
        assert_eq!(config.max_steps(), 300);
    }

    /// Unsafe or non-finite demo settings must not enter a simulation.
    #[test]
    fn experiment_tuning_rejects_invalid_values() {
        // Exercise a non-finite value that no bounded slider can produce.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        let invalid = ExperimentTuning {
            maximum_hit_points: 0,
            ..ExperimentTuning::default()
        };

        assert!(config.apply_experiment_tuning(invalid).is_err());
    }

    /// Perception counts must stay within the fixed actor tensor capacity.
    #[test]
    fn perception_ray_count_validates_fixed_capacity() {
        // Check both rejected edges and one small demo-friendly profile.
        assert!(PerceptionRayCount::try_from(0_u8).is_err());
        assert!(PerceptionRayCount::try_from(1_u8).is_err());
        assert!(PerceptionRayCount::try_from(3_u8).is_err());
        assert!(PerceptionRayCount::try_from(25_u8).is_err());
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
        assert!(LocomotionAction::new(f32::NAN, 0.0, 0.0, 0.0).is_err());
        assert!(LocomotionAction::new(0.0, 1.01, 0.0, 0.0).is_err());
        assert!(LocomotionAction::new(0.0, 0.0, -1.01, 0.0).is_err());
        assert!(LocomotionAction::new(0.0, 0.0, 0.0, 1.01).is_err());
        assert_eq!(
            LocomotionAction::new(-1.0, 1.0, 0.5, 1.0).expect("bounded action is valid"),
            LocomotionAction {
                forward: -1.0,
                turn: 1.0,
                gaze: 0.5,
                attack: 1.0,
            }
        );
    }
}
