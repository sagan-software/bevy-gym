//! Deterministic `Avian2D` ecosystem mechanics shared by every lesson.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::time::Duration;

use avian2d::prelude::*;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;

#[cfg(test)]
use super::domain::ExperimentTuning;
use super::domain::{
    AgentEpisodeMetrics, AgentId, AgentStep, CurriculumStage, DeathCause, EcosystemSnapshot,
    EcosystemState, ForageDifficulty, GlobalState, JointStep, LocalObservation, LocomotionAction,
    PerceptKind, PerceptionRayCount, SimulationConfig, Species, GLOBAL_AGENT_FEATURES,
    GLOBAL_FOOD_FEATURES, GLOBAL_OBSTACLE_FEATURES, LOCAL_EXPOSURE_INDEX, LOCAL_GAZE_YAW_INDEX,
    LOCAL_INTERACTION_COOLDOWN_INDEX, LOCAL_IN_SHELTER_INDEX, LOCAL_LATERAL_VELOCITY_INDEX,
    LOCAL_LONGITUDINAL_VELOCITY_INDEX, LOCAL_OBSERVATION_SIZE, MAX_AGENTS, MAX_FOOD, MAX_OBSTACLES,
    PROPRIOCEPTION_SIZE, RAY_ANGLES, RAY_COUNT, RAY_FEATURE_SIZE, RAY_KIND_START,
};
#[cfg(any(feature = "render", feature = "ecosystem-inference"))]
use super::domain::{
    VisualAgent, VisualGorge, VisualObject, VisualObjectKind, VisualRay, VisualWorldSnapshot,
};
use super::reward::{HomeostaticReward, NormalizedDrive};
use super::rng::SplitMix64;
use super::EpisodeStatus;

/// Conservative agent half-extent used for spawn clearance and route tests.
const AGENT_RADIUS: f32 = 0.75;

/// Rectangle dimensions shared by agent physics and rendering.
pub(super) const AGENT_SIZE: Vec2 = Vec2::new(1.5, 1.1);

/// Radius of a food interaction sensor.
const FOOD_RADIUS: f32 = 0.55;

/// Seconds available to reach food in both movement lessons.
const EPHEMERAL_FOOD_LIFETIME_SECONDS: u16 = 5;

/// Extra reward available when ephemeral food is reached immediately.
const EPHEMERAL_FOOD_SPEED_REWARD: f32 = 2.0;

/// Half-width of the lethal center gorge.
const GORGE_HALF_WIDTH: f32 = 2.0;

/// Half-width of the safe bridge corridor along the world Y axis.
const BRIDGE_HALF_WIDTH: f32 = 1.75;

/// Horizontal center of either alternating gorge food spawn.
const GORGE_BANK_FOOD_X: f32 = 8.0;

/// Solid radius of the well base.
const WELL_RADIUS: f32 = 1.25;

/// Radius in which an agent automatically drinks.
pub(super) const WELL_SENSOR_RADIUS: f32 = 2.25;

/// Half-second intentional contact required to complete one drink.
const DRINKING_INTERVAL_STEPS: u32 = 5;

/// Fixed steps after one completed interaction before another can start.
const INTERACTION_COOLDOWN_STEPS: u16 = 5;

/// Radius of the non-blocking forward interaction hitbox.
pub(super) const INTERACTION_HITBOX_RADIUS: f32 = 0.4;

/// Body-local forward offset of the interaction hitbox center.
pub(super) const INTERACTION_HITBOX_FORWARD_OFFSET: f32 = 0.9;

/// Maximum distance represented by every semantic sector.
const SIGHT_RANGE: f32 = 18.0;

/// Survival terminal-objective weight relative to one healthy simulated minute.
const SURVIVAL_HORIZON_REWARD_SCALE: f32 = 5.0;

/// Half-width of one fixed-capacity semantic sector.
const RAY_HALF_WIDTH: f32 = 5.0_f32.to_radians();

/// Forward offset of each eye from the agent center.
const EYE_FORWARD_OFFSET: f32 = 0.48;

/// Maximum linear speed used for physics and observation normalization.
const MAX_LINEAR_SPEED: f32 = 8.0;

/// Maximum angular speed in radians per second.
///
/// Sixteen radians per second permits a 180-degree turn in 0.20 seconds. This
/// gives the policy twice the angular-speed headroom of the approximately
/// 0.40-second extreme hare turn reported by
/// `https://doi.org/10.1007/s42991-026-00566-7`.
const MAX_ANGULAR_SPEED: f32 = 16.0;

/// Fraction of sideways velocity removed before each physics step.
const GROUND_LATERAL_GRIP: f32 = 0.8;

/// Velocity retained for one fixed step while full braking is requested.
const FULL_BRAKE_VELOCITY_SCALE: f32 = 0.25;

/// Translation multiplier while an agent body overlaps visible shallow water.
const SHALLOW_WATER_MOVEMENT_SCALE: f32 = 0.5;

/// Avian normal impact speed that produces each additional collision-damage point.
const COLLISION_DAMAGE_SPEED_PER_POINT: f32 = 3.0;

/// Fixed steps before sustained solid contact can deal collision damage again.
const COLLISION_DAMAGE_COOLDOWN_STEPS: u16 = 5;

/// Local forward acceleration at a unit action.
const FORWARD_ACCELERATION: f32 = 28.0;

/// Simulated thorn-contact seconds between one-point HP losses.
const THORN_DAMAGE_INTERVAL_SECONDS: u16 = 1;

/// Discrete shelter-protection reserve.
const MAXIMUM_EXPOSURE: u8 = 3;

/// Unprotected seconds between one-point exposure losses.
const EXPOSURE_LOSS_INTERVAL_SECONDS: u16 = 2;

/// Seconds at zero exposure between one-point HP losses.
const EXPOSURE_DAMAGE_INTERVAL_SECONDS: u16 = 2;

/// Earliest seeded weather onset in shelter-enabled stages.
const MINIMUM_WEATHER_ONSET_SECONDS: u16 = 2;

/// Latest seeded weather onset in shelter-enabled stages.
const MAXIMUM_WEATHER_ONSET_SECONDS: u16 = 6;

/// Shelter radius used by physics, rendering, and spawn clearance.
const SHELTER_RADIUS: f32 = 2.5;

/// Placement attempts before deterministic generation fails.
const MAX_SPAWN_ATTEMPTS: usize = 256;

/// Physics collision and spatial-query groups.
#[derive(PhysicsLayer, Default)]
enum PhysicsGroup {
    /// Dynamic bunny and fox bodies.
    #[default]
    Agent,

    /// Static map geometry.
    Solid,

    /// Non-blocking interaction and damage volumes.
    Sensor,
}

/// Schedule used to sample rays after interactions and physiology resolve.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
struct PerceptionSchedule;

/// Active perception capacity copied from the validated episode profile.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
struct ActivePerception(PerceptionRayCount);

/// Current lifecycle of one possible agent slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LifeState {
    /// Agent receives actions, observations, and reward.
    Alive,

    /// Agent trajectory ended naturally with one cause.
    Dead(DeathCause),
}

/// Discrete needs, HP, and their fixed-step clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Physiology {
    /// Current satiation points.
    satiation: u8,

    /// Current hydration points.
    hydration: u8,

    /// Current HP.
    hit_points: u8,

    /// Current protection from environmental exposure.
    exposure: u8,

    /// Whether the body currently overlaps a shelter zone.
    in_shelter: bool,

    /// Steps elapsed toward the next satiation loss.
    satiation_need_steps: u32,

    /// Steps elapsed toward the next hydration loss.
    hydration_need_steps: u32,

    /// Fractional extra need-clock progress caused by translation.
    movement_need_progress: u16,

    /// Steps elapsed at zero satiation.
    starvation_steps: u32,

    /// Steps elapsed at zero hydration.
    dehydration_steps: u32,

    /// Steps elapsed during continuous thorn contact.
    thorn_steps: u32,

    /// Steps elapsed during continuous well contact.
    drinking_steps: u32,

    /// Steps elapsed outside a shelter zone.
    exposure_steps: u32,

    /// Steps elapsed at zero exposure.
    exposure_damage_steps: u32,

    /// Steps elapsed during continuous shelter contact.
    shelter_steps: u32,
}

impl Physiology {
    /// Construct reset physiology from validated integer settings.
    const fn new(hit_points: u8, satiation: u8, hydration: u8) -> Self {
        Self {
            satiation,
            hydration,
            hit_points,
            exposure: MAXIMUM_EXPOSURE,
            in_shelter: false,
            satiation_need_steps: 0,
            hydration_need_steps: 0,
            movement_need_progress: 0,
            starvation_steps: 0,
            dehydration_steps: 0,
            thorn_steps: 0,
            drinking_steps: 0,
            exposure_steps: 0,
            exposure_damage_steps: 0,
            shelter_steps: 0,
        }
    }
}

/// Closest semantic hit for one eye ray.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RaySample {
    /// Hit distance divided by [`SIGHT_RANGE`].
    normalized_distance: f32,

    /// Semantic collider class, or `None` for a missed ray.
    kind: Option<PerceptKind>,
}

impl RaySample {
    /// Stable representation of a missed ray.
    const MISS: Self = Self {
        normalized_distance: 0.0,
        kind: None,
    };
}

/// Per-agent counters independent from the survival reward.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct AgentMetrics {
    /// Total undiscounted reward emitted by completed transitions.
    episode_return: f32,

    /// Compatible food or prey consumed.
    food_eaten: u32,

    /// Well-water units consumed.
    water_consumed: f32,

    /// Bunnies consumed by a fox.
    kills: u32,

    /// Hit points lost to thorn contact.
    thorn_damage: f32,

    /// Hit points lost to high-speed solid collisions.
    collision_damage: f32,

    /// Legacy metric retained at zero for artifact compatibility.
    overconsumption_damage: f32,

    /// Agent-agent overlap contacts observed across physics steps.
    collision_contacts: u32,

    /// Contact steps in which the agent changed world position.
    contact_displacements: u32,

    /// Post-solver solid contacts that exceeded the allowed penetration tolerance.
    unresolved_solid_penetrations: u32,

    /// World-space distance traveled during the episode.
    path_length: f32,

    /// Fixed simulation steps spent inside shelter.
    shelter_steps: u16,

    /// Whether shelter was entered before exposure reached zero.
    sheltered_before_critical_exposure: bool,

    /// Simulated time of the first food or well contact.
    first_resource_contact_seconds: Option<f32>,

    /// Visible-resource transitions eligible for approach scoring.
    visible_resource_transitions: u32,

    /// Eligible transitions that reduced ray-derived resource distance.
    resource_approach_transitions: u32,
}

/// Pre-action navigation state used only for behavior diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
struct NavigationState {
    /// Agent position before physics advances.
    position: Vec2,

    /// Ray-derived food proximity before the action.
    food_proximity: Option<f32>,

    /// Ray-derived well proximity before the action.
    well_proximity: Option<f32>,
}

/// Simulation state attached to one dynamic Avian body.
#[derive(Component, Debug, Clone, PartialEq)]
struct AgentBody {
    /// Stable possible-agent identity.
    id: AgentId,

    /// Species policy role.
    species: Species,

    /// Natural lifecycle state.
    life: LifeState,

    /// Current needs and damage buffers.
    physiology: Physiology,

    /// Fixed simulation steps elapsed since episode spawn.
    age_steps: u16,

    /// Conjugate eye yaw relative to the body heading.
    gaze_yaw: f32,

    /// Whether the current action enables the agent's interaction hitbox.
    interaction_active: bool,

    /// Whether the body currently overlaps visible shallow water.
    in_shallow_water: bool,

    /// Fixed steps remaining before the hitbox can reactivate.
    interaction_cooldown_steps: u16,

    /// Fixed steps before another solid impact can remove hit points.
    collision_damage_cooldown_steps: u16,

    /// Post-physics local ray samples.
    rays: [RaySample; RAY_COUNT],

    /// Behavior diagnostics excluded from reward.
    metrics: AgentMetrics,
}

impl AgentBody {
    /// Construct one clean episode agent.
    fn new(
        id: AgentId,
        species: Species,
        initial_hit_points: u8,
        initial_satiation: u8,
        initial_hydration: u8,
    ) -> Self {
        Self {
            id,
            species,
            life: LifeState::Alive,
            physiology: Physiology::new(initial_hit_points, initial_satiation, initial_hydration),
            age_steps: 0,
            gaze_yaw: 0.0,
            interaction_active: false,
            in_shallow_water: false,
            interaction_cooldown_steps: 0,
            collision_damage_cooldown_steps: 0,
            rays: [RaySample::MISS; RAY_COUNT],
            metrics: AgentMetrics::default(),
        }
    }

    /// Return whether the agent can still act and earn survival reward.
    const fn is_alive(&self) -> bool {
        matches!(self.life, LifeState::Alive)
    }

    /// Return exact simulated age at the configured fixed time step.
    fn age_seconds(&self, time_step: f32) -> f32 {
        f32::from(self.age_steps) * time_step
    }
}

/// Semantic class attached to every queryable collider.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticCollider(PerceptKind);

/// Collider role that initiates an intentional local interaction.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct HitBox;

/// Stable owner and species of one child interaction hitbox.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct InteractionHitbox {
    /// Agent that owns the hitbox.
    owner: AgentId,

    /// Species needed to filter resource and predation contacts.
    species: Species,
}

/// Collider role that receives eating, drinking, attack, or hazard interaction.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct HurtBox;

/// Approximate clearance radius used only during bounded spawn rejection.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
struct SpawnBlocker(f32);

/// Stable padded critic slot for a food entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct FoodSlot(u16);

/// Absolute fixed step at which one movement-lesson food item disappears.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct EphemeralFood {
    /// First step on which the item is no longer available.
    expires_at_step: u32,
}

/// Marker for a lethal gorge-floor sensor.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct GorgeHazard;

/// Closed side of the gorge used to alternate resource targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BankSide {
    /// Negative world X coordinates.
    Left,
    /// Positive world X coordinates.
    Right,
}

impl BankSide {
    /// Return the bank's fixed food-spawn X coordinate.
    const fn food_x(self) -> f32 {
        match self {
            Self::Left => -GORGE_BANK_FOOD_X,
            Self::Right => GORGE_BANK_FOOD_X,
        }
    }

    /// Return the opposite bank for the next target.
    const fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

/// Stable padded critic slot for an obstacle or thorn.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct ObstacleSlot(u16);

/// Obstacle semantic needed by the centralized critic.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
enum ObstacleKind {
    /// Circular solid tree.
    Tree,

    /// Circular solid rock.
    Rock,

    /// Circular traversable damaging sensor.
    Thorn,
}

impl ObstacleKind {
    /// Return the fixed three-channel critic index.
    const fn index(self) -> usize {
        match self {
            Self::Tree => 0,
            Self::Rock => 1,
            Self::Thorn => 2,
        }
    }
}

/// Marker for the single well drinking sensor.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct WellSensor;

/// Visible non-blocking shallow water that slows overlapping agent bodies.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct ShallowWater;

/// Marker for the passive shelter sensor.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct ShelterSensor;

/// Fixed shelter location shared with the critic and renderer.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
struct ShelterState {
    /// Whether this curriculum stage includes exposure and shelter.
    active: bool,

    /// Shelter center in world coordinates.
    position: Vec2,

    /// Passive protection radius.
    radius: f32,
}

/// Mutable finite well state.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
struct WellState {
    /// Static position chosen during reset.
    position: Vec2,

    /// Current water units.
    water: f32,

    /// Maximum water units.
    capacity: f32,
}

/// One interacting Avian world and its deterministic environment stream.
pub(super) struct Ecosystem {
    /// Bevy schedules, entities, resources, and Avian state.
    app: App,

    /// Validated immutable lesson settings.
    config: SimulationConfig,

    /// Environment-only random stream.
    rng: SplitMix64,

    /// Completed joint action count.
    step: u32,

    /// Seeded step at which environmental exposure starts.
    weather_onset_step: u32,

    /// Next padded food slot assigned within this episode.
    next_food_slot: u16,

    /// Bank used by the next gorge food spawn.
    next_gorge_food_bank: BankSide,

    /// Bunny identity offset applied to the deterministic spawn slots.
    bunny_spawn_rotation: usize,

    /// Fox identity offset applied to the deterministic spawn slots.
    fox_spawn_rotation: usize,
}

impl fmt::Debug for Ecosystem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Ecosystem")
            .field("config", &self.config)
            .field("step", &self.step)
            .field("weather_onset_step", &self.weather_onset_step)
            .field("next_food_slot", &self.next_food_slot)
            .field("next_gorge_food_bank", &self.next_gorge_food_bank)
            .field("bunny_spawn_rotation", &self.bunny_spawn_rotation)
            .field("fox_spawn_rotation", &self.fox_spawn_rotation)
            .finish_non_exhaustive()
    }
}

impl Ecosystem {
    /// Construct a deterministic episode and sample its initial perception.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Spawn`] when bounded procedural placement
    /// cannot satisfy non-overlap constraints.
    pub(super) fn new(config: SimulationConfig, seed: u64) -> Result<Self, SimulationError> {
        Self::new_rotated(config, seed, 0)
    }

    /// Construct an episode while rotating agent identities through spawn slots.
    pub(super) fn new_rotated(
        config: SimulationConfig,
        seed: u64,
        rotation: usize,
    ) -> Result<Self, SimulationError> {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            PhysicsPlugins::default().with_length_unit(1.0),
        ));
        app.insert_resource(Gravity::ZERO);
        app.insert_resource(ActivePerception(config.perception_ray_count));
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(
            config.time_step,
        )));
        app.init_schedule(PerceptionSchedule);
        app.add_systems(PerceptionSchedule, update_perceptions);
        app.finish();

        let bunny_spawn_rotation = rotation % config.bunny_count;
        let fox_spawn_rotation = if config.fox_count == 0 {
            0
        } else {
            rotation % config.fox_count
        };
        let mut rng = SplitMix64::new(seed);
        let next_gorge_food_bank = if rng.next_u64().is_multiple_of(2) {
            BankSide::Left
        } else {
            BankSide::Right
        };
        let weather_onset_step = if stage_uses_shelter(config.stage) {
            let onset_span =
                u64::from(MAXIMUM_WEATHER_ONSET_SECONDS - MINIMUM_WEATHER_ONSET_SECONDS + 1);
            let onset_seconds = MINIMUM_WEATHER_ONSET_SECONDS
                + u16::try_from(rng.next_u64() % onset_span).unwrap_or_default();
            config.steps_for_seconds(onset_seconds)
        } else {
            u32::MAX
        };
        let mut ecosystem = Self {
            app,
            config,
            rng,
            step: 0,
            weather_onset_step,
            next_food_slot: 0,
            next_gorge_food_bank,
            bunny_spawn_rotation,
            fox_spawn_rotation,
        };
        ecosystem.spawn_episode()?;

        // One zero-velocity physics update registers colliders before the first
        // post-reset ray sample.
        ecosystem.app.update();
        ecosystem.app.world_mut().run_schedule(PerceptionSchedule);
        Ok(ecosystem)
    }

    /// Return whether environmental exposure is active on the current step.
    const fn weather_active(&self) -> bool {
        self.step >= self.weather_onset_step
    }

    /// Return stable IDs of every living agent in ascending order.
    pub(super) fn living_agents(&mut self) -> Vec<AgentId> {
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        let mut agents: Vec<_> = query
            .iter(world)
            .filter(|agent| agent.is_alive())
            .map(|agent| agent.id)
            .collect();
        agents.sort_unstable();
        agents
    }

    /// Return current local actor observations and centralized critic state.
    pub(super) fn state(&mut self) -> EcosystemState {
        let mut agents = self
            .living_agents()
            .into_iter()
            .filter_map(|id| {
                self.agent_output(id)
                    .map(|(species, observation, _)| (id, species, observation))
            })
            .collect::<Vec<_>>();
        agents.sort_by_key(|(id, _, _)| *id);
        EcosystemState {
            agents,
            global_state: self.global_state(),
        }
    }

    /// Advance every living agent from the same pre-step state.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::ActionSet`] when actions are missing,
    /// duplicated, or supplied for dead agents.
    pub(super) fn step(
        &mut self,
        actions: &[(AgentId, LocomotionAction)],
    ) -> Result<JointStep, SimulationError> {
        let living_before = self.living_agents();
        let actions = validate_joint_actions(&living_before, actions)?;
        if living_before.is_empty() || self.step >= self.config.max_steps() {
            return Err(SimulationError::EpisodeFinished);
        }
        let drives_before = self.agent_drives_by_id();
        let navigation_before = self.navigation_states_by_id();

        self.step = self.step.saturating_add(1);
        self.refill_well();

        // All controls are written before Avian advances, preserving joint
        // action semantics independent of entity query order.
        self.apply_actions(&actions);
        let incoming_velocities = self.agent_linear_velocities();
        self.app.update();
        self.record_unresolved_solid_penetrations();
        self.advance_agent_ages();

        let contacts = self.collect_contacts(&incoming_velocities);
        self.resolve_gorge_falls();
        let food_speed_rewards = self.resolve_food(&contacts.food_winners);
        self.expire_ephemeral_food();
        self.replenish_food()?;
        self.resolve_shallow_water(&contacts.shallow_water_agents);
        self.resolve_drinking(&contacts.well_agents);
        self.resolve_predation(&contacts.predation);
        self.resolve_physiology(
            &contacts.thorn_agents,
            &contacts.shelter_agents,
            &contacts.solid_impact_speeds,
        );
        self.disable_newly_dead();

        self.app.world_mut().run_schedule(PerceptionSchedule);
        self.record_navigation_metrics(&navigation_before, &contacts);

        let horizon = self.step >= self.config.max_steps();
        let stage = self.config.stage;
        let drives_after = self.agent_drives_by_id();
        let mut results = Vec::with_capacity(living_before.len());
        let mut emitted_rewards = BTreeMap::new();
        for id in living_before {
            if let Some((species, observation, life)) = self.agent_output(id) {
                let (status, death_cause) = match life {
                    LifeState::Alive if horizon => (EpisodeStatus::Truncated, None),
                    LifeState::Alive => (EpisodeStatus::Continuing, None),
                    LifeState::Dead(cause) => (EpisodeStatus::Terminated, Some(cause)),
                };
                let mut reward = drives_before.get(&id).zip(drives_after.get(&id)).map_or(
                    0.0,
                    |(before, after)| {
                        HomeostaticReward::default().transition_reward(
                            matches!(life, LifeState::Alive),
                            Duration::from_secs_f32(self.config.time_step),
                            *before,
                            *after,
                        )
                    },
                );
                if horizon && matches!(life, LifeState::Alive) {
                    if let Some(after) = drives_after.get(&id) {
                        let horizon_reward = HomeostaticReward::horizon_reward(*after);
                        reward += if stage == CurriculumStage::Survival {
                            horizon_reward * SURVIVAL_HORIZON_REWARD_SCALE
                        } else {
                            horizon_reward
                        };
                    }
                }
                reward += food_speed_rewards.get(&id).copied().unwrap_or_default();
                emitted_rewards.insert(id, reward);
                results.push(AgentStep {
                    id,
                    species,
                    observation,
                    reward: f64::from(reward),
                    status,
                    death_cause,
                });
            }
        }
        results.sort_by_key(|result| result.id);
        self.record_step_rewards(&emitted_rewards);

        let is_done = horizon || self.living_agents().is_empty();
        Ok(JointStep {
            agents: results,
            global_state: self.global_state(),
            is_done,
        })
    }

    /// Count the completed fixed step for every agent alive at its start.
    fn advance_agent_ages(&mut self) {
        let world = self.app.world_mut();
        let mut query = world.query::<&mut AgentBody>();
        for mut agent in query.iter_mut(world) {
            if agent.is_alive() {
                agent.age_steps = agent.age_steps.saturating_add(1);
            }
        }
    }

    /// Return the compact current world counts used by smoke and HUD paths.
    pub(super) fn snapshot(&mut self) -> EcosystemSnapshot {
        let living_agents = self.living_agents().len();
        let world = self.app.world_mut();
        let mut food_query = world.query::<&FoodSlot>();
        let food_count = food_query.iter(world).count();
        let well_water = world.resource::<WellState>().water;
        EcosystemSnapshot {
            step: self.step,
            living_agents,
            food_count,
            well_water,
        }
    }

    /// Project the live physics world into renderer-only value types.
    #[cfg(any(feature = "render", feature = "ecosystem-inference"))]
    pub(super) fn visual_snapshot(&mut self) -> VisualWorldSnapshot {
        let weather_active = self.weather_active();
        let weather_onset_seconds = self.weather_onset_step as f32 * self.config.time_step;
        let summary = self.snapshot();
        let perception_ray_count = self.config.perception_ray_count;
        let world = self.app.world_mut();

        // Copy only observable scene state so the window cannot mutate the
        // authoritative Avian world or leak global positions into the actor.
        let mut agent_query = world.query::<(&Transform, &AgentBody)>();
        let mut agents = Vec::new();
        let mut rays = Vec::new();
        for (transform, agent) in agent_query.iter(world) {
            let position = transform.translation.truncate();
            let heading = transform.rotation.to_euler(EulerRot::XYZ).2;
            agents.push(VisualAgent {
                id: agent.id,
                species: agent.species,
                position: position.to_array(),
                heading,
                gaze_yaw: agent.gaze_yaw,
                is_alive: agent.is_alive(),
                attack_active: agent.interaction_active,
                satiation: agent.physiology.satiation,
                hydration: agent.physiology.hydration,
                hit_points: agent.physiology.hit_points,
                exposure: agent.physiology.exposure,
                in_shelter: agent.physiology.in_shelter,
            });
            if agent.is_alive() {
                // Reconstruct endpoints from the exact stored sector samples
                // used by the actor's current local observation.
                let body_forward = Vec2::from_angle(heading);
                rays.extend(
                    active_ray_indices(perception_ray_count).filter_map(|ray_index| {
                        let sample = agent.rays.get(ray_index)?;
                        let angle = RAY_ANGLES.get(ray_index)?;
                        let start = perception_origin(position, body_forward);
                        let direction =
                            Vec2::from_angle(agent.gaze_yaw + *angle).rotate(body_forward);
                        let distance = sample
                            .kind
                            .map_or(SIGHT_RANGE, |_| sample.normalized_distance * SIGHT_RANGE);
                        Some(VisualRay {
                            agent: agent.id,
                            start: start.to_array(),
                            end: (direction * distance + start).to_array(),
                            kind: sample.kind,
                        })
                    }),
                );
            }
        }
        agents.sort_by_key(|agent| agent.id);

        let mut objects = Vec::new();
        let mut food_query = world.query_filtered::<&Transform, With<FoodSlot>>();
        objects.extend(food_query.iter(world).map(|transform| VisualObject {
            kind: VisualObjectKind::Food,
            position: [transform.translation.x, transform.translation.y],
            radius: FOOD_RADIUS,
        }));
        let well = *world.resource::<WellState>();
        objects.push(VisualObject {
            kind: VisualObjectKind::Well,
            position: [well.position.x, well.position.y],
            radius: WELL_RADIUS,
        });
        let shelter = *world.resource::<ShelterState>();
        if shelter.active {
            objects.push(VisualObject {
                kind: VisualObjectKind::Shelter,
                position: shelter.position.to_array(),
                radius: shelter.radius,
            });
        }
        let mut obstacle_query = world.query::<(&Transform, &ObstacleKind, &SpawnBlocker)>();
        objects.extend(
            obstacle_query
                .iter(world)
                .map(|(transform, kind, blocker)| {
                    let kind = match kind {
                        ObstacleKind::Tree => VisualObjectKind::Tree,
                        ObstacleKind::Rock => VisualObjectKind::Rock,
                        ObstacleKind::Thorn => VisualObjectKind::Thorn,
                    };
                    VisualObject {
                        kind,
                        position: [transform.translation.x, transform.translation.y],
                        radius: blocker.0,
                    }
                }),
        );
        VisualWorldSnapshot {
            summary,
            map_half_extent: self.config.map_half_extent,
            maximum_hit_points: self.config.maximum_hit_points,
            maximum_satiation: self.config.maximum_satiation,
            maximum_hydration: self.config.maximum_hydration,
            maximum_exposure: MAXIMUM_EXPOSURE,
            weather_active,
            weather_onset_seconds,
            gorge: (self.config.stage == CurriculumStage::Gorge).then_some(VisualGorge {
                half_width: GORGE_HALF_WIDTH,
                bridge_half_width: BRIDGE_HALF_WIDTH,
            }),
            agents,
            objects,
            rays,
        }
    }

    /// Return final behavior evidence for every possible agent slot.
    pub(super) fn episode_metrics(&mut self) -> Vec<AgentEpisodeMetrics> {
        let config = &self.config;
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        let mut metrics = query
            .iter(world)
            .map(|agent| AgentEpisodeMetrics {
                id: agent.id,
                species: agent.species,
                lifetime_seconds: agent.age_seconds(config.time_step),
                episode_return: agent.metrics.episode_return,
                terminal_hit_points: agent.physiology.hit_points,
                terminal_satiation: agent.physiology.satiation,
                terminal_hydration: agent.physiology.hydration,
                food_eaten: agent.metrics.food_eaten,
                water_consumed: agent.metrics.water_consumed,
                kills: agent.metrics.kills,
                thorn_damage: agent.metrics.thorn_damage,
                collision_damage: agent.metrics.collision_damage,
                overconsumption_damage: agent.metrics.overconsumption_damage,
                collision_contacts: agent.metrics.collision_contacts,
                contact_displacements: agent.metrics.contact_displacements,
                unresolved_solid_penetrations: agent.metrics.unresolved_solid_penetrations,
                path_length: agent.metrics.path_length,
                shelter_seconds: f32::from(agent.metrics.shelter_steps) * config.time_step,
                sheltered_before_critical_exposure: agent
                    .metrics
                    .sheltered_before_critical_exposure,
                first_resource_contact_seconds: agent.metrics.first_resource_contact_seconds,
                visible_resource_transitions: agent.metrics.visible_resource_transitions,
                resource_approach_transitions: agent.metrics.resource_approach_transitions,
                death_cause: match agent.life {
                    LifeState::Alive => None,
                    LifeState::Dead(cause) => Some(cause),
                },
            })
            .collect::<Vec<_>>();
        metrics.sort_by_key(|agent| agent.id);
        metrics
    }

    /// Spawn all static geometry, resources, hazards, and possible agents.
    fn spawn_episode(&mut self) -> Result<(), SimulationError> {
        self.spawn_boundaries();
        self.app.world_mut().insert_resource(ShelterState {
            active: false,
            position: Vec2::ZERO,
            radius: SHELTER_RADIUS,
        });

        match self.config.stage {
            CurriculumStage::Forage => {
                self.spawn_forage_lesson();
                return Ok(());
            }
            CurriculumStage::Sprint => {
                self.spawn_sprint_lesson();
                return Ok(());
            }
            CurriculumStage::Gorge => {
                self.spawn_gorge_lesson();
                return Ok(());
            }
            CurriculumStage::Survival => {
                self.spawn_survival_lesson();
                self.spawn_survival_obstacles()?;
                return Ok(());
            }
            CurriculumStage::Shelter
            | CurriculumStage::Competition
            | CurriculumStage::PredatorPrey
            | CurriculumStage::Obstacles => {}
        }

        let well_position = self.open_position(WELL_SENSOR_RADIUS + 0.5)?;
        self.spawn_well(well_position);
        if stage_uses_shelter(self.config.stage) {
            let shelter_position = self.open_position(SHELTER_RADIUS + 0.5)?;
            self.spawn_shelter(shelter_position);
        }

        self.spawn_solid_obstacles()?;
        for slot in 0..self.config.thorn_obstacles {
            let radius = 1.25;
            let position = self.open_position(radius + 0.5)?;
            self.spawn_thorn(self.config.solid_obstacles + slot, position, radius);
        }

        for slot in 0..self.config.bunny_count {
            let position = self.open_position(AGENT_RADIUS + 0.5)?;
            let identity = (slot + self.bunny_spawn_rotation) % self.config.bunny_count;
            self.spawn_agent(
                AgentId(u16::try_from(identity).unwrap_or(u16::MAX)),
                Species::Bunny,
                position,
            );
        }
        for slot in 0..self.config.fox_count {
            let position = self.open_position(AGENT_RADIUS + 0.5)?;
            let identity =
                self.config.bunny_count + (slot + self.fox_spawn_rotation) % self.config.fox_count;
            self.spawn_agent(
                AgentId(u16::try_from(identity).unwrap_or(u16::MAX)),
                Species::Fox,
                position,
            );
        }

        for _ in 0..self.config.initial_food {
            self.spawn_one_food()?;
        }
        Ok(())
    }

    /// Spawn one varied food target without adding an active water requirement.
    fn spawn_forage_lesson(&mut self) {
        let heading = self
            .rng
            .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let bunny_position = Vec2::new(
            self.rng.f32_between(-0.5, 0.5),
            self.rng.f32_between(-0.5, 0.5),
        );
        let (maximum_bearing_degrees, minimum_distance, maximum_distance) =
            match self.config.forage_difficulty {
                ForageDifficulty::Foundation => (12.0, 2.5, 4.0),
                ForageDifficulty::Expanded => (70.0, 8.0, 14.0),
            };
        let bearing = self
            .rng
            .f32_between(-maximum_bearing_degrees, maximum_bearing_degrees)
            .to_radians();
        let distance = self.rng.f32_between(minimum_distance, maximum_distance);
        let food_position = bunny_position + Vec2::from_angle(heading + bearing) * distance;
        let well_position = Vec2::splat(self.config.map_half_extent - WELL_SENSOR_RADIUS - 0.5);
        self.spawn_well(well_position);
        self.spawn_agent_facing(AgentId(0), Species::Bunny, bunny_position, heading);
        self.spawn_food_at(food_position);
    }

    /// Spawn a distant visible target whose remaining lifetime rewards speed.
    fn spawn_sprint_lesson(&mut self) {
        // Randomize a reachable target across the forward visual field.
        let heading = self
            .rng
            .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let bunny_position = Vec2::new(
            self.rng.f32_between(-0.5, 0.5),
            self.rng.f32_between(-0.5, 0.5),
        );
        let bearing = self.rng.f32_between(-65.0, 65.0).to_radians();
        let distance = self.rng.f32_between(9.0, 13.0);
        let food_position = bunny_position + Vec2::from_angle(heading + bearing) * distance;
        let well_position = Vec2::splat(self.config.map_half_extent - WELL_SENSOR_RADIUS - 0.5);
        self.spawn_well(well_position);
        self.spawn_agent_facing(AgentId(0), Species::Bunny, bunny_position, heading);
        self.spawn_food_at(food_position);
    }

    /// Spawn two banks, a lethal gorge, one visible bridge, and an opposite-bank target.
    fn spawn_gorge_lesson(&mut self) {
        // Start the bunny opposite the next food so collection proves a crossing.
        let food_bank = self.next_gorge_food_bank;
        let bunny_bank = food_bank.opposite();
        let bunny_y = self.rng.f32_between(-1.0, 1.0);
        let food_y = self.rng.f32_between(-1.0, 1.0);
        let bunny_position = Vec2::new(bunny_bank.food_x(), bunny_y);
        let heading = if food_bank == BankSide::Right {
            0.0
        } else {
            std::f32::consts::PI
        };
        let well_position = Vec2::new(
            bunny_bank.food_x(),
            self.config.map_half_extent - WELL_SENSOR_RADIUS - 0.5,
        );
        self.spawn_well(well_position);
        self.spawn_gorge_geometry();
        self.spawn_agent_facing(AgentId(0), Species::Bunny, bunny_position, heading);
        self.spawn_food_at(Vec2::new(food_bank.food_x(), food_y));
        self.next_gorge_food_bank = food_bank.opposite();
    }

    /// Spawn lethal floor above and below a rail-guided safe bridge corridor.
    fn spawn_gorge_geometry(&mut self) {
        // Cover the unsafe strip with sensors while rails mark the safe corridor.
        let hazard_height = self.config.map_half_extent - BRIDGE_HALF_WIDTH;
        let hazard_center_y = hazard_height.mul_add(0.5, BRIDGE_HALF_WIDTH);
        let world = self.app.world_mut();
        for center_y in [hazard_center_y, -hazard_center_y] {
            let position = Vec2::new(0.0, center_y);
            world.spawn((
                RigidBody::Static,
                Position(position),
                Transform::from_translation(position.extend(0.0)),
                Collider::rectangle(GORGE_HALF_WIDTH * 2.0, hazard_height),
                Sensor,
                sensor_layers(),
                SemanticCollider(PerceptKind::Gorge),
                GorgeHazard,
            ));
        }
        for rail_y in [BRIDGE_HALF_WIDTH, -BRIDGE_HALF_WIDTH] {
            let position = Vec2::new(0.0, rail_y);
            world.spawn((
                RigidBody::Static,
                Position(position),
                Transform::from_translation(position.extend(0.0)),
                Collider::rectangle(GORGE_HALF_WIDTH * 2.0, 0.2),
                solid_layers(),
                SemanticCollider(PerceptKind::Bridge),
            ));
        }
    }

    /// Spawn independently varied food and water targets for the single-agent lesson.
    fn spawn_survival_lesson(&mut self) {
        // Vary body position, heading, resource sides, and resource order.
        // Seeded obstacles may then occlude either resource and require search.
        let heading = self
            .rng
            .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let bunny_position = Vec2::new(
            self.rng.f32_between(-0.5, 0.5),
            self.rng.f32_between(-0.5, 0.5),
        );
        let food_side = if self.rng.next_u64().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let well_side = if self.rng.next_u64().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let food_bearing = food_side * self.rng.f32_between(8.0, 16.0).to_radians();
        let well_bearing = well_side * self.rng.f32_between(45.0, 55.0).to_radians();
        let nearer_distance = self.rng.f32_between(3.0, 4.5);
        let farther_distance = self.rng.f32_between(5.5, 7.0);
        let food_is_nearer = self.rng.next_u64().is_multiple_of(2);
        let (food_distance, well_distance) = if food_is_nearer {
            (nearer_distance, farther_distance)
        } else {
            (farther_distance, nearer_distance)
        };
        let food_direction = Vec2::from_angle(heading + food_bearing);
        let well_direction = Vec2::from_angle(heading + well_bearing);
        let first_food_position = bunny_position + food_direction * food_distance;
        let well_position = bunny_position + well_direction * well_distance;
        self.spawn_well(well_position);
        self.spawn_agent_facing(AgentId(0), Species::Bunny, bunny_position, heading);
        self.spawn_food_at(first_food_position);
        if self.config.initial_food > 1 {
            self.spawn_food_at(bunny_position + food_direction * 8.5);
        }
    }

    /// Spawn four static solid colliders around the playable square.
    fn spawn_boundaries(&mut self) {
        let extent = self.config.map_half_extent;
        let thickness = 1.0;
        let full = extent.mul_add(2.0, thickness * 2.0);
        let world = self.app.world_mut();
        for (position, size) in [
            (
                Vec2::new(0.0, extent + thickness * 0.5),
                Vec2::new(full, thickness),
            ),
            (
                Vec2::new(0.0, -extent - thickness * 0.5),
                Vec2::new(full, thickness),
            ),
            (
                Vec2::new(extent + thickness * 0.5, 0.0),
                Vec2::new(thickness, full),
            ),
            (
                Vec2::new(-extent - thickness * 0.5, 0.0),
                Vec2::new(thickness, full),
            ),
        ] {
            world.spawn((
                RigidBody::Static,
                Position(position),
                Transform::from_translation(position.extend(0.0)),
                Collider::rectangle(size.x, size.y),
                solid_layers(),
                SemanticCollider(PerceptKind::Boundary),
            ));
        }
    }

    /// Spawn visible shallow water and its larger intentional drinking radius.
    fn spawn_well(&mut self, position: Vec2) {
        let world = self.app.world_mut();
        world.insert_resource(WellState {
            position,
            water: self.config.well_capacity,
            capacity: self.config.well_capacity,
        });
        world.spawn((
            RigidBody::Static,
            Position(position),
            Transform::from_translation(position.extend(0.0)),
            Collider::circle(WELL_RADIUS),
            Sensor,
            sensor_layers(),
            CollidingEntities::default(),
            SemanticCollider(PerceptKind::Well),
            ShallowWater,
            SpawnBlocker(WELL_SENSOR_RADIUS),
        ));
        world.spawn((
            RigidBody::Static,
            Position(position),
            Transform::from_translation(position.extend(0.0)),
            Collider::circle(WELL_SENSOR_RADIUS),
            Sensor,
            sensor_layers(),
            CollidingEntities::default(),
            WellSensor,
            HurtBox,
        ));
    }

    /// Spawn one passive shelter zone for exposure-enabled stages.
    fn spawn_shelter(&mut self, position: Vec2) {
        self.app.world_mut().insert_resource(ShelterState {
            active: true,
            position,
            radius: SHELTER_RADIUS,
        });
        self.app.world_mut().spawn((
            RigidBody::Static,
            Position(position),
            Transform::from_translation(position.extend(0.0)),
            Collider::circle(SHELTER_RADIUS),
            Sensor,
            sensor_layers(),
            CollidingEntities::default(),
            SemanticCollider(PerceptKind::Shelter),
            ShelterSensor,
            SpawnBlocker(SHELTER_RADIUS),
        ));
    }

    /// Spawn one dynamic mutually pushable agent body.
    fn spawn_agent(&mut self, id: AgentId, species: Species, position: Vec2) {
        let facing = self
            .rng
            .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        self.spawn_agent_facing(id, species, position, facing);
    }

    /// Spawn one agent with an explicit tutorial or procedural heading.
    fn spawn_agent_facing(&mut self, id: AgentId, species: Species, position: Vec2, facing: f32) {
        let mut entity = self.app.world_mut().spawn((
            AgentBody::new(
                id,
                species,
                self.config.initial_hit_points,
                self.config.initial_satiation,
                self.config.initial_hydration,
            ),
            RigidBody::Dynamic,
            Position(position),
            Rotation::radians(facing),
            Transform {
                translation: position.extend(0.0),
                rotation: Quat::from_rotation_z(facing),
                ..default()
            },
            Collider::ellipse(AGENT_SIZE.x * 0.5, AGENT_SIZE.y * 0.5),
            agent_layers(),
            LinearVelocity::ZERO,
            AngularVelocity(0.0),
            ConstantLocalLinearAcceleration::default(),
            LinearDamping(1.5),
            AngularDamping(4.0),
        ));
        entity.insert((
            MaxLinearSpeed(MAX_LINEAR_SPEED * self.config.movement_speed_multiplier),
            MaxAngularSpeed(MAX_ANGULAR_SPEED),
            SleepingDisabled,
            CollidingEntities::default(),
            SemanticCollider(match species {
                Species::Bunny => PerceptKind::Bunny,
                Species::Fox => PerceptKind::Fox,
            }),
            SpawnBlocker(AGENT_RADIUS),
            HurtBox,
        ));
        let body = entity.id();
        self.app.world_mut().spawn((
            ChildOf(body),
            Transform::from_xyz(INTERACTION_HITBOX_FORWARD_OFFSET, 0.0, 0.0),
            Collider::circle(INTERACTION_HITBOX_RADIUS),
            Sensor,
            agent_layers(),
            CollidingEntities::default(),
            HitBox,
            InteractionHitbox { owner: id, species },
        ));
    }

    /// Spawn one tree or rock with a stable centralized-state slot.
    fn spawn_solid_obstacle(&mut self, slot: usize, position: Vec2, radius: f32) {
        let kind = if slot.is_multiple_of(2) {
            ObstacleKind::Tree
        } else {
            ObstacleKind::Rock
        };
        let collider = Collider::circle(radius);
        self.app.world_mut().spawn((
            RigidBody::Static,
            Position(position),
            Transform::from_translation(position.extend(0.0)),
            collider,
            solid_layers(),
            SemanticCollider(PerceptKind::SolidObstacle),
            ObstacleSlot(u16::try_from(slot).unwrap_or(u16::MAX)),
            kind,
            SpawnBlocker(radius),
        ));
    }

    /// Place the configured solid obstacle count in open seeded positions.
    fn spawn_solid_obstacles(&mut self) -> Result<(), SimulationError> {
        for slot in 0..self.config.solid_obstacles {
            let radius = if slot.is_multiple_of(2) { 1.4 } else { 1.1 };
            let position = self.open_position(radius + 0.5)?;
            self.spawn_solid_obstacle(slot, position, radius);
        }
        Ok(())
    }

    /// Place survival obstacles around clear initial resource corridors.
    fn spawn_survival_obstacles(&mut self) -> Result<(), SimulationError> {
        let well = self.app.world().resource::<WellState>().position;
        let agent = self
            .agent_entities()
            .values()
            .next()
            .and_then(|entity| self.app.world().get::<Position>(*entity))
            .map_or(Vec2::ZERO, |position| position.0);
        let food = {
            let world = self.app.world_mut();
            let mut query = world.query::<(&FoodSlot, &Position)>();
            query
                .iter(world)
                .min_by_key(|(slot, _)| slot.0)
                .map(|(_, position)| position.0)
        };
        // Keep the urgent opening recoverable. Replacement food uses the full
        // random map and can require exploration behind these obstacles.
        for slot in 0..self.config.solid_obstacles {
            let radius = if slot.is_multiple_of(2) { 1.4 } else { 1.1 };
            let position = (0..MAX_SPAWN_ATTEMPTS)
                .find_map(|_| {
                    let candidate = self.open_position(radius + 0.5).ok()?;
                    let clears_water =
                        point_segment_distance(candidate, agent, well) > radius + AGENT_RADIUS;
                    let clears_food = food.is_none_or(|food| {
                        point_segment_distance(candidate, agent, food) > radius + AGENT_RADIUS
                    });
                    (clears_water && clears_food).then_some(candidate)
                })
                .ok_or(SimulationError::Spawn {
                    radius,
                    attempts: MAX_SPAWN_ATTEMPTS,
                })?;
            self.spawn_solid_obstacle(slot, position, radius);
        }
        Ok(())
    }

    /// Spawn one traversable thorn damage sensor.
    fn spawn_thorn(&mut self, slot: usize, position: Vec2, radius: f32) {
        self.app.world_mut().spawn((
            RigidBody::Static,
            Position(position),
            Transform::from_translation(position.extend(0.0)),
            Collider::circle(radius),
            Sensor,
            sensor_layers(),
            CollidingEntities::default(),
            SemanticCollider(PerceptKind::Thorn),
            ObstacleSlot(u16::try_from(slot).unwrap_or(u16::MAX)),
            ObstacleKind::Thorn,
            SpawnBlocker(radius),
            HurtBox,
        ));
    }

    /// Spawn one sensor food item when the padded capacity permits it.
    fn spawn_one_food(&mut self) -> Result<(), SimulationError> {
        // Movement lessons use controlled targets instead of open-world placement.
        let position = match self.config.stage {
            CurriculumStage::Sprint => self.sprint_food_position()?,
            CurriculumStage::Gorge => {
                let bank = self.next_gorge_food_bank;
                self.next_gorge_food_bank = bank.opposite();
                Vec2::new(bank.food_x(), self.rng.f32_between(-1.0, 1.0))
            }
            CurriculumStage::Forage
            | CurriculumStage::Survival
            | CurriculumStage::Shelter
            | CurriculumStage::Competition
            | CurriculumStage::PredatorPrey
            | CurriculumStage::Obstacles => self.open_position(FOOD_RADIUS + 0.25)?,
        };
        self.spawn_food_at(position);
        Ok(())
    }

    /// Choose a reachable sprint target near the bunny's current position.
    fn sprint_food_position(&mut self) -> Result<Vec2, SimulationError> {
        // Keep replacement targets distant and inside the map boundary.
        let bunny_position = {
            let world = self.app.world_mut();
            let mut query = world.query::<(&AgentBody, &Position)>();
            query
                .iter(world)
                .find(|(agent, _)| agent.species == Species::Bunny && agent.is_alive())
                .map_or(Vec2::ZERO, |(_, position)| position.0)
        };
        let limit = self.config.map_half_extent - FOOD_RADIUS - 0.5;
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let angle = self
                .rng
                .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
            let distance = self.rng.f32_between(8.0, 12.0);
            let candidate = bunny_position + Vec2::from_angle(angle) * distance;
            if candidate.x.abs() <= limit && candidate.y.abs() <= limit {
                return Ok(candidate);
            }
        }
        Err(SimulationError::Spawn {
            radius: FOOD_RADIUS,
            attempts: MAX_SPAWN_ATTEMPTS,
        })
    }

    /// Spawn food at a validated lesson or procedural position.
    fn spawn_food_at(&mut self, position: Vec2) {
        let used_slots = {
            let world = self.app.world_mut();
            let mut query = world.query::<&FoodSlot>();
            query
                .iter(world)
                .map(|slot| slot.0)
                .collect::<BTreeSet<_>>()
        };
        // Reuse only a vacant critic slot so two live food entities can never
        // overwrite each other in the padded global state.
        let capacity = u16::try_from(self.config.max_food).unwrap_or(u16::MAX);
        let slot = (0..capacity)
            .map(|offset| self.next_food_slot.wrapping_add(offset) % capacity)
            .find(|candidate| !used_slots.contains(candidate));
        let Some(slot) = slot else {
            debug_assert!(false, "a validated food capacity must have a vacant slot");
            return;
        };
        self.next_food_slot = slot.wrapping_add(1) % capacity;
        let ephemeral = self
            .config
            .stage
            .uses_ephemeral_food()
            .then_some(EphemeralFood {
                expires_at_step: self.step.saturating_add(
                    self.config
                        .steps_for_seconds(EPHEMERAL_FOOD_LIFETIME_SECONDS),
                ),
            });
        let entity = self
            .app
            .world_mut()
            .spawn((
                RigidBody::Static,
                Position(position),
                Transform::from_translation(position.extend(0.0)),
                Collider::circle(FOOD_RADIUS),
                Sensor,
                sensor_layers(),
                CollidingEntities::default(),
                SemanticCollider(PerceptKind::Food),
                FoodSlot(slot),
                SpawnBlocker(FOOD_RADIUS),
                HurtBox,
            ))
            .id();
        if let Some(ephemeral) = ephemeral {
            self.app.world_mut().entity_mut(entity).insert(ephemeral);
        }
    }

    /// Preserve one available item and periodically replenish the second slot.
    fn replenish_food(&mut self) -> Result<(), SimulationError> {
        // Replace missing single-agent targets immediately to sustain the lesson.
        let live_food = {
            let world = self.app.world_mut();
            let mut query = world.query::<&FoodSlot>();
            query.iter(world).count()
        };
        let is_due = self.step.is_multiple_of(self.config.food_spawn_interval);
        let single_agent_food_floor = matches!(
            self.config.stage,
            CurriculumStage::Forage
                | CurriculumStage::Sprint
                | CurriculumStage::Gorge
                | CurriculumStage::Survival
        ) && live_food == 0;
        if single_agent_food_floor || (live_food < self.config.max_food && is_due) {
            self.spawn_one_food()?;
        }
        Ok(())
    }

    /// Find a bounded non-overlapping procedural spawn position.
    fn open_position(&mut self, radius: f32) -> Result<Vec2, SimulationError> {
        let occupied = {
            let world = self.app.world_mut();
            let mut query = world.query::<(&Position, &SpawnBlocker)>();
            query
                .iter(world)
                .map(|(position, blocker)| (position.0, blocker.0))
                .collect::<Vec<_>>()
        };
        let limit = self.config.map_half_extent - radius - 0.5;

        // Bounded rejection keeps invalid procedural settings diagnosable.
        for _ in 0..MAX_SPAWN_ATTEMPTS {
            let candidate = Vec2::new(
                self.rng.f32_between(-limit, limit),
                self.rng.f32_between(-limit, limit),
            );
            let is_clear = occupied.iter().all(|(position, occupied_radius)| {
                candidate.distance_squared(*position) >= (radius + occupied_radius + 0.35).powi(2)
            });
            if is_clear {
                return Ok(candidate);
            }
        }
        Err(SimulationError::Spawn {
            radius,
            attempts: MAX_SPAWN_ATTEMPTS,
        })
    }

    /// Refill the finite well by fixed simulated time.
    fn refill_well(&mut self) {
        let refill = self.config.well_refill_rate * self.config.time_step;
        let mut well = self.app.world_mut().resource_mut::<WellState>();
        well.water = (well.water + refill).min(well.capacity);
    }

    /// Write all control components before the physics schedule runs.
    fn apply_actions(&mut self, actions: &BTreeMap<AgentId, LocomotionAction>) {
        let world = self.app.world_mut();
        let mut query = world.query::<(
            &mut AgentBody,
            &mut ConstantLocalLinearAcceleration,
            &mut AngularVelocity,
            &mut MaxLinearSpeed,
            &Rotation,
            &mut LinearVelocity,
        )>();
        for (
            mut agent,
            mut acceleration,
            mut angular_velocity,
            mut max_speed,
            rotation,
            mut linear_velocity,
        ) in query.iter_mut(world)
        {
            if !agent.is_alive() {
                continue;
            }
            if let Some(action) = actions.get(&agent.id) {
                let movement_scale = self.config.movement_speed_multiplier
                    * shallow_water_movement_scale(agent.in_shallow_water);
                let forward_throttle = action.forward.mul_add(0.5, 0.5).clamp(0.0, 1.0);
                let body_forward = Vec2::new(rotation.cos, rotation.sin);
                apply_ground_traction(&mut linear_velocity.0, body_forward, forward_throttle);
                acceleration.0 = Vec2::X * forward_throttle * FORWARD_ACCELERATION * movement_scale;
                max_speed.0 = MAX_LINEAR_SPEED * movement_scale;
                angular_velocity.0 = action.turn * MAX_ANGULAR_SPEED;
                agent.gaze_yaw = action.gaze * self.config.gaze_yaw_limit_degrees.to_radians();
                if agent.interaction_cooldown_steps > 0 {
                    agent.interaction_cooldown_steps -= 1;
                    agent.interaction_active = false;
                } else {
                    agent.interaction_active = action.attack > 0.0;
                }
                agent.collision_damage_cooldown_steps =
                    agent.collision_damage_cooldown_steps.saturating_sub(1);
            }
        }
    }

    /// Capture incoming Avian body velocities before contact resolution changes them.
    fn agent_linear_velocities(&mut self) -> BTreeMap<AgentId, Vec2> {
        let world = self.app.world_mut();
        let mut query = world.query::<(&AgentBody, &LinearVelocity)>();
        // Capture only agent bodies because all damaging solids are static.
        query
            .iter(world)
            .map(|(agent, velocity)| (agent.id, velocity.0))
            .collect()
    }

    /// Collect current Avian overlaps before mutating any interaction state.
    fn collect_contacts(
        &mut self,
        incoming_velocities: &BTreeMap<AgentId, Vec2>,
    ) -> ContactResolution {
        let bunny_count = self.config.bunny_count;
        let fox_count = self.config.fox_count;
        let bunny_priority_rotation = self.step as usize % bunny_count;
        let fox_priority_rotation = if fox_count == 0 {
            0
        } else {
            self.step as usize % fox_count
        };
        let world = self.app.world_mut();
        let (body_by_entity, active_hitters) = active_contact_participants(world);

        let mut food_winners = Vec::new();
        {
            let mut query = world.query::<(Entity, &FoodSlot, &CollidingEntities)>();
            for (food, _, collisions) in query.iter(world) {
                let winner = collisions
                    .0
                    .iter()
                    .filter_map(|entity| active_hitters.get(entity))
                    .filter(|(_, species)| *species == Species::Bunny)
                    .map(|(id, _)| *id)
                    .min_by_key(|id| {
                        rotated_priority(*id, 0, bunny_count, bunny_priority_rotation)
                    });
                if let Some(agent) = winner {
                    food_winners.push((food, agent));
                }
            }
        }
        food_winners.sort_by_key(|(_, agent)| *agent);

        let well_agents = {
            let mut query = world.query_filtered::<&CollidingEntities, With<WellSensor>>();
            query
                .iter(world)
                .flat_map(|collisions| collisions.0.iter())
                .filter_map(|entity| active_hitters.get(entity).map(|(id, _)| *id))
                .collect::<BTreeSet<_>>()
        };

        let shallow_water_agents = {
            let mut query = world.query_filtered::<&CollidingEntities, With<ShallowWater>>();
            query
                .iter(world)
                .flat_map(|collisions| collisions.0.iter())
                .filter_map(|entity| body_by_entity.get(entity).map(|(id, _)| *id))
                .collect::<BTreeSet<_>>()
        };

        let thorn_agents = {
            let mut query =
                world.query_filtered::<(&ObstacleKind, &CollidingEntities), With<ObstacleSlot>>();
            query
                .iter(world)
                .filter(|(kind, _)| **kind == ObstacleKind::Thorn)
                .flat_map(|(_, collisions)| collisions.0.iter())
                .filter_map(|entity| body_by_entity.get(entity).map(|(id, _)| *id))
                .collect::<BTreeSet<_>>()
        };

        let shelter_agents = {
            let mut query = world.query_filtered::<&CollidingEntities, With<ShelterSensor>>();
            query
                .iter(world)
                .flat_map(|collisions| collisions.0.iter())
                .filter_map(|entity| body_by_entity.get(entity).map(|(id, _)| *id))
                .collect::<BTreeSet<_>>()
        };

        // Avian may resolve a fast impact to exact tangency in one step. Keep
        // manifold pairs whose separation is within the solver tolerance so
        // the blocking contact remains observable after position correction.
        let agent_contacts = {
            let graph = world.resource::<ContactGraph>();
            graph
                .iter_active()
                .chain(graph.iter_sleeping())
                .filter(|pair| {
                    pair.manifolds.iter().any(|manifold| {
                        manifold
                            .points
                            .iter()
                            .any(|point| point.penetration >= -0.01 || point.normal_impulse > 0.0)
                    })
                })
                .filter_map(|pair| {
                    let first = body_by_entity.get(&pair.collider1)?;
                    let second = body_by_entity.get(&pair.collider2)?;
                    Some((pair.collider1, *first, pair.collider2, *second))
                })
                .collect::<Vec<_>>()
        };
        let agent_agents = agent_contacts
            .iter()
            .flat_map(|(_, first, _, second)| [first.0, second.0])
            .collect::<BTreeSet<_>>();
        record_agent_contacts(world, agent_contacts);
        let predation_by_bunny = collect_predation_contacts(
            world,
            &body_by_entity,
            &active_hitters,
            bunny_count,
            fox_count,
            fox_priority_rotation,
        );
        let solid_impact_speeds =
            collect_solid_impact_speeds(world, &body_by_entity, incoming_velocities);

        ContactResolution {
            food_winners,
            well_agents,
            shallow_water_agents,
            thorn_agents,
            shelter_agents,
            solid_impact_speeds,
            predation: predation_by_bunny
                .into_iter()
                .map(|(bunny, fox)| (fox, bunny))
                .collect(),
            agent_agents,
        }
    }

    /// Consume each contested food entity and return its time-sensitive reward.
    fn resolve_food(&mut self, winners: &[(Entity, AgentId)]) -> BTreeMap<AgentId, f32> {
        // Convert each ephemeral target's remaining lifetime into a speed bonus.
        let entity_by_id = self.agent_entities();
        let maximum_satiation = self.config.maximum_satiation;
        let lifetime_steps = self
            .config
            .steps_for_seconds(EPHEMERAL_FOOD_LIFETIME_SECONDS);
        let mut speed_rewards = BTreeMap::new();
        for (food, winner) in winners {
            let ephemeral = self.app.world().get::<EphemeralFood>(*food).copied();
            if ephemeral.is_some_and(|food| self.step >= food.expires_at_step) {
                let _despawned = self.app.world_mut().despawn(*food);
                continue;
            }
            let speed_reward = ephemeral
                .map(|food| {
                    let remaining = food.expires_at_step.saturating_sub(self.step);
                    remaining as f32 / lifetime_steps as f32 * EPHEMERAL_FOOD_SPEED_REWARD
                })
                .unwrap_or_default();
            if let Some(agent_entity) = entity_by_id.get(winner).copied() {
                if let Some(mut agent) = self.app.world_mut().get_mut::<AgentBody>(agent_entity) {
                    if agent.is_alive() && agent.species == Species::Bunny {
                        agent.physiology.satiation =
                            if self.config.stage == CurriculumStage::Survival {
                                maximum_satiation
                            } else {
                                agent
                                    .physiology
                                    .satiation
                                    .saturating_add(1)
                                    .min(maximum_satiation)
                            };
                        agent.physiology.satiation_need_steps = 0;
                        agent.metrics.food_eaten = agent.metrics.food_eaten.saturating_add(1);
                        start_interaction_cooldown(&mut agent);
                        speed_rewards.insert(*winner, speed_reward);
                    }
                }
            }
            let _despawned = self.app.world_mut().despawn(*food);
        }
        speed_rewards
    }

    /// Remove uneaten movement targets on their exact fixed-step deadline.
    fn expire_ephemeral_food(&mut self) {
        // Collect before despawning so query iteration never mutates its world view.
        let expired = {
            let world = self.app.world_mut();
            let mut query = world.query::<(Entity, &EphemeralFood)>();
            query
                .iter(world)
                .filter_map(|(entity, food)| (self.step >= food.expires_at_step).then_some(entity))
                .collect::<Vec<_>>()
        };
        for entity in expired {
            let _despawned = self.app.world_mut().despawn(entity);
        }
    }

    /// End gorge trajectories whose body center leaves the bridge corridor.
    fn resolve_gorge_falls(&mut self) {
        // Classify every unsafe center crossing without affecting other stages.
        if self.config.stage != CurriculumStage::Gorge {
            return;
        }
        let world = self.app.world_mut();
        let mut query = world.query::<(&mut AgentBody, &Position)>();
        for (mut agent, position) in query.iter_mut(world) {
            let in_gorge =
                position.x.abs() < GORGE_HALF_WIDTH && position.y.abs() > BRIDGE_HALF_WIDTH;
            if agent.is_alive() && in_gorge {
                agent.life = LifeState::Dead(DeathCause::Gorge);
                agent.physiology.hit_points = 0;
            }
        }
    }

    /// Clamp entry velocity and retain shallow-water state for the next action.
    fn resolve_shallow_water(&mut self, agents: &BTreeSet<AgentId>) {
        let maximum_speed =
            MAX_LINEAR_SPEED * self.config.movement_speed_multiplier * SHALLOW_WATER_MOVEMENT_SCALE;
        let world = self.app.world_mut();
        let mut query = world.query::<(&mut AgentBody, Option<&mut LinearVelocity>)>();
        // Clamp on entry, then let the next action retain the lower speed cap.
        for (mut agent, velocity) in query.iter_mut(world) {
            agent.in_shallow_water = agents.contains(&agent.id);
            if !agent.in_shallow_water {
                continue;
            }
            if let Some(mut velocity) = velocity {
                velocity.0 = velocity.0.clamp_length_max(maximum_speed);
            }
        }
    }

    /// Divide scarce well water fairly without withdrawing more than agents absorb.
    fn resolve_drinking(&mut self, drinkers: &BTreeSet<AgentId>) {
        let entity_by_id = self.agent_entities();
        let requested = self.config.well_drink_rate;
        let maximum_hydration = self.config.maximum_hydration;
        let drinking_interval = DRINKING_INTERVAL_STEPS;
        let completed_drinks = entity_by_id
            .iter()
            .filter_map(|(id, entity)| {
                let mut agent = self.app.world_mut().get_mut::<AgentBody>(*entity)?;
                let active = agent.is_alive()
                    && drinkers.contains(id)
                    && agent.physiology.hydration < maximum_hydration;
                damage_clock_due(
                    active,
                    &mut agent.physiology.drinking_steps,
                    drinking_interval,
                )
                .then_some(*id)
            })
            .collect::<BTreeSet<_>>();
        if completed_drinks.is_empty() {
            return;
        }
        let available = self.app.world().resource::<WellState>().water;
        let mut candidates = completed_drinks.into_iter().collect::<Vec<_>>();
        let completed_drink_index = self.step / drinking_interval;
        let rotation = usize::try_from(completed_drink_index).unwrap_or(usize::MAX);
        let recipients = complete_drink_recipients(&mut candidates, available, requested, rotation);
        let withdrawn = requested * recipients.len() as f32;
        self.app.world_mut().resource_mut::<WellState>().water = (available - withdrawn).max(0.0);

        // A discrete hydration point requires one complete conserved drink.
        for id in recipients {
            if let Some(entity) = entity_by_id.get(&id).copied() {
                if let Some(mut agent) = self.app.world_mut().get_mut::<AgentBody>(entity) {
                    agent.physiology.hydration = if self.config.stage == CurriculumStage::Survival {
                        maximum_hydration
                    } else {
                        agent
                            .physiology
                            .hydration
                            .saturating_add(1)
                            .min(maximum_hydration)
                    };
                    agent.physiology.hydration_need_steps = 0;
                    agent.metrics.water_consumed += requested;
                    start_interaction_cooldown(&mut agent);
                }
            }
        }
    }

    /// Resolve one deterministic fox winner per contacted bunny.
    fn resolve_predation(&mut self, contacts: &[(AgentId, AgentId)]) {
        let entity_by_id = self.agent_entities();
        for (fox_id, bunny_id) in contacts {
            let Some(bunny_entity) = entity_by_id.get(bunny_id).copied() else {
                continue;
            };
            let Some(fox_entity) = entity_by_id.get(fox_id).copied() else {
                continue;
            };
            let bunny_was_alive = self
                .app
                .world()
                .get::<AgentBody>(bunny_entity)
                .is_some_and(AgentBody::is_alive);
            if !bunny_was_alive {
                continue;
            }
            if let Some(mut bunny) = self.app.world_mut().get_mut::<AgentBody>(bunny_entity) {
                bunny.life = LifeState::Dead(DeathCause::Predation);
                bunny.physiology.hit_points = 0;
            }
            if let Some(mut fox) = self.app.world_mut().get_mut::<AgentBody>(fox_entity) {
                if fox.is_alive() {
                    fox.physiology.satiation = fox
                        .physiology
                        .satiation
                        .saturating_add(1)
                        .min(self.config.maximum_satiation);
                    fox.metrics.food_eaten = fox.metrics.food_eaten.saturating_add(1);
                    fox.metrics.kills = fox.metrics.kills.saturating_add(1);
                    start_interaction_cooldown(&mut fox);
                }
            }
        }
    }

    /// Advance needs and apply starvation, dehydration, and thorn damage.
    fn resolve_physiology(
        &mut self,
        thorn_agents: &BTreeSet<AgentId>,
        shelter_agents: &BTreeSet<AgentId>,
        solid_impact_speeds: &BTreeMap<AgentId, f32>,
    ) {
        let need_interval = self
            .config
            .steps_for_seconds(self.config.need_loss_interval_seconds);
        let starvation_interval = self
            .config
            .steps_for_seconds(self.config.starvation_damage_interval_seconds);
        let dehydration_interval = self
            .config
            .steps_for_seconds(self.config.dehydration_damage_interval_seconds);
        let thorn_interval = self.config.steps_for_seconds(THORN_DAMAGE_INTERVAL_SECONDS);
        let exposure_loss_interval = self
            .config
            .steps_for_seconds(EXPOSURE_LOSS_INTERVAL_SECONDS);
        let exposure_damage_interval = self
            .config
            .steps_for_seconds(EXPOSURE_DAMAGE_INTERVAL_SECONDS);
        let shelter_interval = self.config.steps_for_seconds(1);
        let exposure_active = self.weather_active();
        let hydration_active = self.config.stage.uses_hydration();
        let extent = self.config.map_half_extent + 2.0;
        let world = self.app.world_mut();
        let mut query = world.query::<(&mut AgentBody, &Position, Option<&LinearVelocity>)>();
        for (mut agent, position, velocity) in query.iter_mut(world) {
            if !agent.is_alive() {
                continue;
            }
            let was_starving = agent.physiology.satiation == 0;
            let was_dehydrated = hydration_active && agent.physiology.hydration == 0;
            let was_exposed = exposure_active && agent.physiology.exposure == 0;
            // Integer percentage units retain a small movement cost without
            // charging rotation or eye motion as translation.
            let is_translating = velocity.is_some_and(|linear| linear.length_squared() > 1.0e-4);
            if is_translating {
                agent.physiology.movement_need_progress = agent
                    .physiology
                    .movement_need_progress
                    .saturating_add(u16::from(self.config.movement_need_cost_percent));
            }
            let extra_need_steps = agent.physiology.movement_need_progress / 100;
            agent.physiology.movement_need_progress %= 100;
            let elapsed_need_steps = 1 + u32::from(extra_need_steps);
            agent.physiology.satiation_need_steps = agent
                .physiology
                .satiation_need_steps
                .saturating_add(elapsed_need_steps);
            agent.physiology.hydration_need_steps = agent
                .physiology
                .hydration_need_steps
                .saturating_add(elapsed_need_steps);
            while agent.physiology.satiation_need_steps >= need_interval {
                agent.physiology.satiation_need_steps -= need_interval;
                agent.physiology.satiation = agent.physiology.satiation.saturating_sub(1);
            }
            while hydration_active && agent.physiology.hydration_need_steps >= need_interval {
                agent.physiology.hydration_need_steps -= need_interval;
                agent.physiology.hydration = agent.physiology.hydration.saturating_sub(1);
            }

            let starvation_due = damage_clock_due(
                was_starving,
                &mut agent.physiology.starvation_steps,
                starvation_interval,
            );
            let dehydration_due = damage_clock_due(
                was_dehydrated,
                &mut agent.physiology.dehydration_steps,
                dehydration_interval,
            );
            let thorn_due = damage_clock_due(
                thorn_agents.contains(&agent.id),
                &mut agent.physiology.thorn_steps,
                thorn_interval,
            );
            let collision_damage = if agent.collision_damage_cooldown_steps == 0 {
                solid_impact_speeds
                    .get(&agent.id)
                    .copied()
                    .map_or(0, collision_damage_from_impact_speed)
            } else {
                0
            };
            let in_shelter = shelter_agents.contains(&agent.id);
            if in_shelter && agent.physiology.exposure > 0 {
                agent.metrics.sheltered_before_critical_exposure = true;
            }
            let exposure_due = update_exposure(
                &mut agent.physiology,
                exposure_active,
                in_shelter,
                was_exposed,
                shelter_interval,
                exposure_loss_interval,
                exposure_damage_interval,
            );
            if agent.physiology.in_shelter {
                agent.metrics.shelter_steps = agent.metrics.shelter_steps.saturating_add(1);
            }
            let deprivation_damage = u8::from(starvation_due) + u8::from(dehydration_due);
            let total_damage = deprivation_damage
                .saturating_add(u8::from(thorn_due))
                .saturating_add(u8::from(exposure_due))
                .saturating_add(collision_damage);
            agent.physiology.hit_points = agent.physiology.hit_points.saturating_sub(total_damage);
            if thorn_due {
                agent.metrics.thorn_damage += 1.0;
            }
            agent.metrics.collision_damage += f32::from(collision_damage);
            if collision_damage > 0 {
                agent.collision_damage_cooldown_steps = COLLISION_DAMAGE_COOLDOWN_STEPS;
            }

            let damage_due = u8::from(starvation_due)
                | (u8::from(dehydration_due) << 1)
                | (u8::from(thorn_due) << 2)
                | (u8::from(exposure_due) << 3)
                | (u8::from(collision_damage > 0) << 4);
            let lethal_cause = lethal_damage_cause(DamageDue(damage_due));

            if !position.0.is_finite() || position.x.abs() > extent || position.y.abs() > extent {
                agent.life = LifeState::Dead(DeathCause::InvalidPhysics);
                agent.physiology.hit_points = 0;
            } else if agent.physiology.hit_points == 0 {
                let cause = lethal_cause.unwrap_or(DeathCause::Deprivation);
                agent.life = LifeState::Dead(cause);
            }
        }
    }

    /// Remove physics and perception capabilities from newly dead bodies.
    fn disable_newly_dead(&mut self) {
        let (dead_entities, dead_ids) = {
            let world = self.app.world_mut();
            let mut query = world.query::<(Entity, &AgentBody, Has<Collider>)>();
            let dead = query
                .iter(world)
                .filter(|(_, agent, _)| !agent.is_alive())
                .map(|(entity, agent, has_collider)| (entity, agent.id, has_collider))
                .collect::<Vec<_>>();
            let entities = dead
                .iter()
                .filter_map(|(entity, _, has_collider)| has_collider.then_some(*entity))
                .collect::<Vec<_>>();
            let ids = dead.iter().map(|(_, id, _)| *id).collect::<BTreeSet<_>>();
            (entities, ids)
        };
        for entity in dead_entities {
            let mut entity_mut = self.app.world_mut().entity_mut(entity);
            entity_mut.remove::<Collider>();
            entity_mut.remove::<SemanticCollider>();
            entity_mut.remove::<RigidBody>();
            entity_mut.remove::<ConstantLocalLinearAcceleration>();
            entity_mut.remove::<AngularVelocity>();
            if let Some(mut agent) = entity_mut.get_mut::<AgentBody>() {
                agent.rays = [RaySample::MISS; RAY_COUNT];
                agent.interaction_active = false;
            }
        }
        let dead_hitboxes = {
            let world = self.app.world_mut();
            let mut query = world.query::<(Entity, &InteractionHitbox, Has<Collider>)>();
            query
                .iter(world)
                .filter(|(_, hitbox, has_collider)| {
                    *has_collider && dead_ids.contains(&hitbox.owner)
                })
                .map(|(entity, _, _)| entity)
                .collect::<Vec<_>>()
        };
        for entity in dead_hitboxes {
            let mut entity_mut = self.app.world_mut().entity_mut(entity);
            entity_mut.remove::<Collider>();
            entity_mut.remove::<CollidingEntities>();
        }
    }

    /// Build stable agent-ID to entity mapping for deterministic mutations.
    fn agent_entities(&mut self) -> BTreeMap<AgentId, Entity> {
        let world = self.app.world_mut();
        let mut query = world.query::<(Entity, &AgentBody)>();
        query
            .iter(world)
            .map(|(entity, agent)| (agent.id, entity))
            .collect()
    }

    /// Snapshot normalized physiological drives before or after one transition.
    fn agent_drives_by_id(&mut self) -> BTreeMap<AgentId, NormalizedDrive> {
        let stage = self.config.stage;
        let maximum_satiation = self.config.maximum_satiation;
        let maximum_hydration = self.config.maximum_hydration;
        let maximum_hit_points = self.config.maximum_hit_points;
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        query
            .iter(world)
            .map(|agent| {
                (
                    agent.id,
                    physiological_drive(
                        stage,
                        agent.physiology,
                        maximum_satiation,
                        maximum_hydration,
                        maximum_hit_points,
                    ),
                )
            })
            .collect()
    }

    /// Add emitted rewards to stable per-agent episode totals.
    fn record_step_rewards(&mut self, rewards: &BTreeMap<AgentId, f32>) {
        let world = self.app.world_mut();
        let mut query = world.query::<&mut AgentBody>();
        for mut agent in query.iter_mut(world) {
            if let Some(reward) = rewards.get(&agent.id) {
                agent.metrics.episode_return += reward;
            }
        }
    }

    /// Snapshot positions and visible resource proximity before one action.
    fn navigation_states_by_id(&mut self) -> BTreeMap<AgentId, NavigationState> {
        let world = self.app.world_mut();
        let mut query = world.query::<(&AgentBody, &Position)>();
        query
            .iter(world)
            .filter(|(agent, _)| agent.is_alive())
            .map(|(agent, position)| {
                (
                    agent.id,
                    NavigationState {
                        position: position.0,
                        food_proximity: nearest_visible_proximity(agent, PerceptKind::Food),
                        well_proximity: nearest_visible_proximity(agent, PerceptKind::Well),
                    },
                )
            })
            .collect()
    }

    /// Record movement, visible-target progress, and first resource contact.
    fn record_navigation_metrics(
        &mut self,
        before: &BTreeMap<AgentId, NavigationState>,
        contacts: &ContactResolution,
    ) {
        let food_contacts = contacts
            .food_winners
            .iter()
            .map(|(_, id)| *id)
            .collect::<BTreeSet<_>>();
        let time_step = self.config.time_step;
        let world = self.app.world_mut();
        let mut query = world.query::<(&mut AgentBody, &Position)>();
        for (mut agent, position) in query.iter_mut(world) {
            let Some(previous) = before.get(&agent.id) else {
                continue;
            };
            let displacement = position.0.distance(previous.position);
            agent.metrics.path_length += displacement;
            if contacts.agent_agents.contains(&agent.id) && displacement > 1.0e-4 {
                agent.metrics.contact_displacements =
                    agent.metrics.contact_displacements.saturating_add(1);
            }
            if agent.metrics.first_resource_contact_seconds.is_none()
                && (food_contacts.contains(&agent.id) || contacts.well_agents.contains(&agent.id))
            {
                agent.metrics.first_resource_contact_seconds = Some(agent.age_seconds(time_step));
            }
            let current = [
                nearest_visible_proximity(&agent, PerceptKind::Food),
                nearest_visible_proximity(&agent, PerceptKind::Well),
            ];
            for (previous, current) in [previous.food_proximity, previous.well_proximity]
                .into_iter()
                .zip(current)
            {
                if let (Some(previous), Some(current)) = (previous, current) {
                    agent.metrics.visible_resource_transitions =
                        agent.metrics.visible_resource_transitions.saturating_add(1);
                    if current > previous + 1e-4 {
                        agent.metrics.resource_approach_transitions = agent
                            .metrics
                            .resource_approach_transitions
                            .saturating_add(1);
                    }
                }
            }
        }
    }

    /// Count solid-obstacle contacts that remain beyond the solver tolerance.
    fn record_unresolved_solid_penetrations(&mut self) {
        let world = self.app.world_mut();
        let agents = {
            let mut query = world.query_filtered::<Entity, With<AgentBody>>();
            query.iter(world).collect::<BTreeSet<_>>()
        };
        let solids = {
            let mut query = world.query::<(Entity, &ObstacleKind)>();
            query
                .iter(world)
                .filter(|(_, kind)| **kind != ObstacleKind::Thorn)
                .map(|(entity, _)| entity)
                .collect::<BTreeSet<_>>()
        };
        let unresolved = {
            let graph = world.resource::<ContactGraph>();
            graph
                .iter_active()
                .chain(graph.iter_sleeping())
                .filter_map(|pair| {
                    let agent = if agents.contains(&pair.collider1)
                        && solids.contains(&pair.collider2)
                    {
                        pair.collider1
                    } else if agents.contains(&pair.collider2) && solids.contains(&pair.collider1) {
                        pair.collider2
                    } else {
                        return None;
                    };
                    let maximum_penetration = pair
                        .manifolds
                        .iter()
                        .flat_map(|manifold| manifold.points.iter())
                        .map(|point| point.penetration.max(0.0))
                        .fold(0.0, f32::max);
                    (maximum_penetration > 0.02).then_some(agent)
                })
                .collect::<BTreeSet<_>>()
        };
        for entity in unresolved {
            if let Some(mut agent) = world.get_mut::<AgentBody>(entity) {
                agent.metrics.unresolved_solid_penetrations = agent
                    .metrics
                    .unresolved_solid_penetrations
                    .saturating_add(1);
            }
        }
    }

    /// Return one agent's local output after interaction resolution.
    fn agent_output(&mut self, id: AgentId) -> Option<(Species, LocalObservation, LifeState)> {
        let max_age = f32::from(self.config.episode_seconds);
        let max_linear_speed = MAX_LINEAR_SPEED * self.config.movement_speed_multiplier;
        let gaze_yaw_limit = self.config.gaze_yaw_limit_degrees.to_radians();
        let maximum_satiation = self.config.maximum_satiation;
        let maximum_hydration = self.config.maximum_hydration;
        let maximum_hit_points = self.config.maximum_hit_points;
        let time_step = self.config.time_step;
        let weather_active = self.weather_active();
        let world = self.app.world_mut();
        let mut query = world.query::<(
            &AgentBody,
            &Position,
            Option<&Rotation>,
            Option<&LinearVelocity>,
            Option<&AngularVelocity>,
        )>();
        query
            .iter(world)
            .find(|(agent, _, _, _, _)| agent.id == id)
            .map(|(agent, position, rotation, velocity, angular_velocity)| {
                let observation = encode_local_observation(
                    agent,
                    position.0,
                    rotation.copied().unwrap_or_default(),
                    velocity.copied().unwrap_or_default(),
                    angular_velocity.copied().unwrap_or_default(),
                    agent.age_seconds(time_step),
                    max_age,
                    max_linear_speed,
                    gaze_yaw_limit,
                    maximum_satiation,
                    maximum_hydration,
                    maximum_hit_points,
                    weather_active,
                );
                (agent.species, observation, agent.life)
            })
    }

    /// Encode the fixed padded centralized critic state.
    fn global_state(&mut self) -> GlobalState {
        let mut state = [0.0; super::domain::GLOBAL_STATE_SIZE];
        let extent = self.config.map_half_extent;
        let max_linear_speed = MAX_LINEAR_SPEED * self.config.movement_speed_multiplier;
        let maximum_satiation = self.config.maximum_satiation;
        let maximum_hydration = self.config.maximum_hydration;
        let maximum_hit_points = self.config.maximum_hit_points;
        let weather_active = self.weather_active();
        let world = self.app.world_mut();

        write_global_agents(
            world,
            &mut state,
            extent,
            max_linear_speed,
            maximum_satiation,
            maximum_hydration,
            maximum_hit_points,
        );

        let well_start = MAX_AGENTS * GLOBAL_AGENT_FEATURES;
        let well = world.resource::<WellState>();
        write_feature(&mut state, well_start, 1.0);
        write_feature(
            &mut state,
            well_start + 1,
            (well.position.x / extent).clamp(-1.0, 1.0),
        );
        write_feature(
            &mut state,
            well_start + 2,
            (well.position.y / extent).clamp(-1.0, 1.0),
        );
        write_feature(&mut state, well_start + 3, well.water / well.capacity);

        let shelter_start = well_start + 4;
        let shelter = world.resource::<ShelterState>();
        write_feature(&mut state, shelter_start, f32::from(shelter.active));
        write_feature(
            &mut state,
            shelter_start + 1,
            (shelter.position.x / extent).clamp(-1.0, 1.0),
        );
        write_feature(
            &mut state,
            shelter_start + 2,
            (shelter.position.y / extent).clamp(-1.0, 1.0),
        );
        write_feature(&mut state, shelter_start + 3, shelter.radius / extent);

        let food_start = shelter_start + 4;
        let mut food_query = world.query::<(&FoodSlot, &Position)>();
        for (slot, position) in food_query.iter(world) {
            let slot = usize::from(slot.0);
            if slot >= MAX_FOOD {
                continue;
            }
            let start = food_start + slot * GLOBAL_FOOD_FEATURES;
            write_feature(&mut state, start, 1.0);
            write_feature(
                &mut state,
                start + 1,
                (position.x / extent).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 2,
                (position.y / extent).clamp(-1.0, 1.0),
            );
        }

        let obstacle_start = food_start + MAX_FOOD * GLOBAL_FOOD_FEATURES;
        let mut obstacle_query = world.query::<(&ObstacleSlot, &ObstacleKind, &Position)>();
        for (slot, kind, position) in obstacle_query.iter(world) {
            let slot = usize::from(slot.0);
            if slot >= MAX_OBSTACLES {
                continue;
            }
            let start = obstacle_start + slot * GLOBAL_OBSTACLE_FEATURES;
            write_feature(&mut state, start, 1.0);
            write_feature(
                &mut state,
                start + 1,
                (position.x / extent).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 2,
                (position.y / extent).clamp(-1.0, 1.0),
            );
            write_feature(&mut state, start + 3 + kind.index(), 1.0);
        }

        let time_start = obstacle_start + MAX_OBSTACLES * GLOBAL_OBSTACLE_FEATURES;
        let remaining_time = 1.0 - self.step as f32 / self.config.max_steps() as f32;
        write_feature(
            &mut state,
            time_start,
            if weather_active {
                -remaining_time
            } else {
                remaining_time
            },
        );
        write_feature(&mut state, time_start + 1 + self.config.stage.index(), 1.0);
        state
    }
}

/// Remove sideways slip and make the zero-throttle endpoint an active brake.
fn apply_ground_traction(velocity: &mut Vec2, body_forward: Vec2, forward_throttle: f32) {
    // Preserve forward momentum while removing velocity that causes sideways skating.
    let forward_speed = velocity.dot(body_forward).max(0.0);
    let lateral_velocity = *velocity - body_forward * forward_speed;
    *velocity = body_forward * forward_speed + lateral_velocity * (1.0 - GROUND_LATERAL_GRIP);
    if forward_throttle <= f32::EPSILON {
        *velocity *= FULL_BRAKE_VELOCITY_SCALE;
    }
}

/// Return live body identities and active child-hitbox identities.
fn active_contact_participants(
    world: &mut World,
) -> (
    BTreeMap<Entity, (AgentId, Species)>,
    BTreeMap<Entity, (AgentId, Species)>,
) {
    let agents = {
        let mut query = world.query::<(Entity, &AgentBody)>();
        query
            .iter(world)
            .filter(|(_, agent)| agent.is_alive())
            .map(|(entity, agent)| (entity, agent.id, agent.species, agent.interaction_active))
            .collect::<Vec<_>>()
    };
    let body_by_entity = agents
        .iter()
        .map(|(entity, id, species, _)| (*entity, (*id, *species)))
        .collect::<BTreeMap<_, _>>();
    let active_agents = agents
        .into_iter()
        .filter_map(|(_, id, species, active)| active.then_some((id, species)))
        .collect::<BTreeMap<_, _>>();
    let mut hitbox_query = world.query::<(Entity, &InteractionHitbox)>();
    let active_hitboxes = hitbox_query
        .iter(world)
        .filter(|(_, hitbox)| active_agents.get(&hitbox.owner) == Some(&hitbox.species))
        .map(|(entity, hitbox)| (entity, (hitbox.owner, hitbox.species)))
        .collect();
    (body_by_entity, active_hitboxes)
}

/// Disable the hitbox until the fixed cooldown reaches zero.
const fn start_interaction_cooldown(agent: &mut AgentBody) {
    agent.interaction_active = false;
    agent.interaction_cooldown_steps = INTERACTION_COOLDOWN_STEPS;
}

/// Advance passive shelter recovery, outdoor exposure, and exposure damage.
fn update_exposure(
    physiology: &mut Physiology,
    exposure_active: bool,
    inside_shelter: bool,
    was_exposed: bool,
    shelter_interval: u32,
    exposure_loss_interval: u32,
    exposure_damage_interval: u32,
) -> bool {
    physiology.in_shelter = exposure_active && inside_shelter;
    if physiology.in_shelter {
        physiology.exposure_steps = 0;
        physiology.exposure_damage_steps = 0;
        physiology.shelter_steps = physiology.shelter_steps.saturating_add(1);
        while physiology.shelter_steps >= shelter_interval {
            physiology.shelter_steps -= shelter_interval;
            physiology.exposure = physiology.exposure.saturating_add(1).min(MAXIMUM_EXPOSURE);
        }
    } else if exposure_active {
        physiology.shelter_steps = 0;
        physiology.exposure_steps = physiology.exposure_steps.saturating_add(1);
        while physiology.exposure_steps >= exposure_loss_interval {
            physiology.exposure_steps -= exposure_loss_interval;
            physiology.exposure = physiology.exposure.saturating_sub(1);
        }
    } else {
        physiology.exposure_steps = 0;
        physiology.exposure_damage_steps = 0;
        physiology.shelter_steps = 0;
    }
    damage_clock_due(
        was_exposed,
        &mut physiology.exposure_damage_steps,
        exposure_damage_interval,
    )
}

/// Select one terminal cause from the damage clocks due on this step.
const fn lethal_damage_cause(due: DamageDue) -> Option<DeathCause> {
    let hazard_with_other_damage = (due.thorns()
        && (due.starvation() || due.dehydration() || due.exposure() || due.collision()))
        || (due.exposure() && (due.starvation() || due.dehydration() || due.collision()))
        || (due.collision() && (due.starvation() || due.dehydration()));
    if hazard_with_other_damage {
        Some(DeathCause::CombinedDamage)
    } else if due.starvation() && due.dehydration() {
        Some(DeathCause::Deprivation)
    } else if due.starvation() {
        Some(DeathCause::Starvation)
    } else if due.dehydration() {
        Some(DeathCause::Dehydration)
    } else if due.thorns() {
        Some(DeathCause::Thorns)
    } else if due.exposure() {
        Some(DeathCause::Exposure)
    } else if due.collision() {
        Some(DeathCause::Collision)
    } else {
        None
    }
}

/// Packed damage clocks due for one agent on one simulation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DamageDue(u8);

impl DamageDue {
    /// Return whether the starvation clock reached its interval.
    const fn starvation(self) -> bool {
        self.0 & 1 != 0
    }

    /// Return whether the dehydration clock reached its interval.
    const fn dehydration(self) -> bool {
        self.0 & 2 != 0
    }

    /// Return whether the thorn contact clock reached its interval.
    const fn thorns(self) -> bool {
        self.0 & 4 != 0
    }

    /// Return whether the outdoor exposure clock reached its interval.
    const fn exposure(self) -> bool {
        self.0 & 8 != 0
    }

    /// Return whether a high-speed collision damaged the agent.
    const fn collision(self) -> bool {
        self.0 & 16 != 0
    }
}

/// Read pre-solver solid impact speeds from Avian's active contact graph.
fn collect_solid_impact_speeds(
    world: &mut World,
    body_by_entity: &BTreeMap<Entity, (AgentId, Species)>,
    incoming_velocities: &BTreeMap<AgentId, Vec2>,
) -> BTreeMap<AgentId, f32> {
    let solids = {
        let mut query = world.query::<(Entity, &SemanticCollider, &Position)>();
        query
            .iter(world)
            .filter_map(|(entity, semantic, position)| {
                matches!(
                    semantic.0,
                    PerceptKind::SolidObstacle | PerceptKind::Boundary
                )
                .then_some((entity, (semantic.0, position.0)))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let agent_positions = {
        let mut query = world.query::<(Entity, &Position)>();
        query
            .iter(world)
            .filter_map(|(entity, position)| {
                body_by_entity
                    .contains_key(&entity)
                    .then_some((entity, position.0))
            })
            .collect::<BTreeMap<_, _>>()
    };
    let graph = world.resource::<ContactGraph>();
    let mut impact_speeds = BTreeMap::<AgentId, f32>::new();
    for pair in graph
        .iter_active()
        .filter(|pair| pair.generates_constraints())
    {
        let (id, agent_entity, solid_entity, agent_is_first) = match (
            body_by_entity.get(&pair.collider1),
            body_by_entity.get(&pair.collider2),
        ) {
            (Some((id, _)), _) if solids.contains_key(&pair.collider2) => {
                (*id, pair.collider1, pair.collider2, true)
            }
            (_, Some((id, _))) if solids.contains_key(&pair.collider1) => {
                (*id, pair.collider2, pair.collider1, false)
            }
            _ => continue,
        };
        let Some(velocity) = incoming_velocities.get(&id).copied() else {
            continue;
        };
        // Avian's manifold normal points from collider one to collider two.
        // Project the incoming Avian velocity onto that contact normal.
        let manifold_impact_speed = pair
            .manifolds
            .iter()
            .filter(|manifold| !manifold.points.is_empty())
            .map(|manifold| {
                let directed_speed = velocity.dot(manifold.normal);
                if agent_is_first {
                    directed_speed.max(0.0)
                } else {
                    (-directed_speed).max(0.0)
                }
            })
            .reduce(f32::max);
        // Avian can clear a solved contact's points before this read. The
        // active pair still proves contact, so recover its geometric normal.
        let impact_speed = manifold_impact_speed.unwrap_or_else(|| {
            let Some(agent_position) = agent_positions.get(&agent_entity).copied() else {
                return 0.0;
            };
            let Some((kind, solid_position)) = solids.get(&solid_entity).copied() else {
                return 0.0;
            };
            velocity
                .dot(solid_normal_from_agent(
                    kind,
                    agent_position,
                    solid_position,
                ))
                .max(0.0)
        });
        impact_speeds
            .entry(id)
            .and_modify(|maximum| *maximum = maximum.max(impact_speed))
            .or_insert(impact_speed);
    }
    impact_speeds
}

/// Derive the agent-to-solid normal for a solved contact without manifold points.
fn solid_normal_from_agent(kind: PerceptKind, agent: Vec2, solid: Vec2) -> Vec2 {
    // Boundary centers identify their inward axis; round obstacles use center direction.
    if kind == PerceptKind::Boundary {
        if solid.x.abs() > solid.y.abs() {
            Vec2::new(solid.x.signum(), 0.0)
        } else {
            Vec2::new(0.0, solid.y.signum())
        }
    } else {
        (solid - agent).normalize_or_zero()
    }
}

/// Count body contacts without assigning interaction semantics to them.
fn record_agent_contacts(
    world: &mut World,
    agent_contacts: Vec<(Entity, (AgentId, Species), Entity, (AgentId, Species))>,
) {
    for (first_entity, _, second_entity, _) in agent_contacts {
        for entity in [first_entity, second_entity] {
            if let Some(mut agent) = world.get_mut::<AgentBody>(entity) {
                agent.metrics.collision_contacts =
                    agent.metrics.collision_contacts.saturating_add(1);
            }
        }
    }
}

/// Select one stable attacking fox for every bunny touched by a forward hitbox.
fn collect_predation_contacts(
    world: &mut World,
    body_by_entity: &BTreeMap<Entity, (AgentId, Species)>,
    active_hitters: &BTreeMap<Entity, (AgentId, Species)>,
    bunny_count: usize,
    fox_count: usize,
    fox_priority_rotation: usize,
) -> BTreeMap<AgentId, AgentId> {
    let mut predation_by_bunny = BTreeMap::new();
    let mut hitbox_query = world.query::<(Entity, &CollidingEntities)>();
    for (hitbox_entity, collisions) in hitbox_query.iter(world) {
        let Some((fox_id, Species::Fox)) = active_hitters.get(&hitbox_entity).copied() else {
            continue;
        };
        for bunny_id in collisions.0.iter().filter_map(|entity| {
            body_by_entity
                .get(entity)
                .and_then(|(id, species)| (*species == Species::Bunny).then_some(*id))
        }) {
            predation_by_bunny
                .entry(bunny_id)
                .and_modify(|winner| {
                    let winner_rank =
                        rotated_priority(*winner, bunny_count, fox_count, fox_priority_rotation);
                    let challenger_rank =
                        rotated_priority(fox_id, bunny_count, fox_count, fox_priority_rotation);
                    if challenger_rank < winner_rank {
                        *winner = fox_id;
                    }
                })
                .or_insert(fox_id);
        }
    }
    predation_by_bunny
}

/// Encode all live agent slots into the centralized critic state.
fn write_global_agents(
    world: &mut World,
    state: &mut GlobalState,
    extent: f32,
    max_linear_speed: f32,
    maximum_satiation: u8,
    maximum_hydration: u8,
    maximum_hit_points: u8,
) {
    let mut agent_query = world.query::<(&AgentBody, &Position, Option<&LinearVelocity>)>();
    for (agent, position, velocity) in agent_query.iter(world) {
        let slot = usize::from(agent.id.0);
        if slot >= MAX_AGENTS {
            continue;
        }
        let start = slot * GLOBAL_AGENT_FEATURES;
        let velocity = velocity.copied().unwrap_or_default().0;
        write_feature(state, start + agent.species.index(), 1.0);
        write_feature(state, start + 2, f32::from(agent.is_alive()));
        write_feature(state, start + 3, (position.x / extent).clamp(-1.0, 1.0));
        write_feature(state, start + 4, (position.y / extent).clamp(-1.0, 1.0));
        write_feature(
            state,
            start + 5,
            (velocity.x / max_linear_speed).clamp(-1.0, 1.0),
        );
        write_feature(
            state,
            start + 6,
            (velocity.y / max_linear_speed).clamp(-1.0, 1.0),
        );
        write_feature(
            state,
            start + 7,
            f32::from(agent.physiology.satiation) / f32::from(maximum_satiation),
        );
        write_feature(
            state,
            start + 8,
            f32::from(agent.physiology.hydration) / f32::from(maximum_hydration),
        );
        let health_fraction =
            f32::from(agent.physiology.hit_points) / f32::from(maximum_hit_points);
        write_feature(state, start + 9, health_fraction);
        write_feature(
            state,
            start + 10,
            f32::from(agent.physiology.exposure) / f32::from(MAXIMUM_EXPOSURE),
        );
    }
}

/// Current overlap sets resolved in stable order after one physics step.
#[derive(Debug, Default)]
struct ContactResolution {
    /// One winner for every contacted food entity.
    food_winners: Vec<(Entity, AgentId)>,

    /// Agents currently inside the well sensor.
    well_agents: BTreeSet<AgentId>,

    /// Agents whose bodies currently overlap visible shallow water.
    shallow_water_agents: BTreeSet<AgentId>,

    /// Agents currently inside at least one thorn sensor.
    thorn_agents: BTreeSet<AgentId>,

    /// Agents currently inside the passive shelter sensor.
    shelter_agents: BTreeSet<AgentId>,

    /// Largest solid-contact approach speed for each impacted agent.
    solid_impact_speeds: BTreeMap<AgentId, f32>,

    /// Stable `(fox, bunny)` predation winners.
    predation: Vec<(AgentId, AgentId)>,

    /// Agents with one same-step body contact against another agent.
    agent_agents: BTreeSet<AgentId>,
}

/// Simulation setup, action-set, or lifecycle failure.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SimulationError {
    /// Bounded procedural placement could not find an open point.
    Spawn {
        /// Requested clearance radius.
        radius: f32,

        /// Number of rejected candidates.
        attempts: usize,
    },

    /// Joint action IDs did not exactly match the current live set.
    ActionSet {
        /// Required living IDs.
        expected: Vec<AgentId>,

        /// Supplied unique IDs.
        actual: Vec<AgentId>,
    },

    /// Caller attempted to step an already completed ecosystem.
    EpisodeFinished,
}

impl fmt::Display for SimulationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { radius, attempts } => write!(
                formatter,
                "failed to place radius {radius} after {attempts} deterministic attempts"
            ),
            Self::ActionSet { expected, actual } => write!(
                formatter,
                "joint action IDs must match living agents; expected {expected:?}, got {actual:?}"
            ),
            Self::EpisodeFinished => formatter.write_str("ecosystem episode is already finished"),
        }
    }
}

impl Error for SimulationError {}

/// Ensure one and only one action exists for every living agent.
fn validate_joint_actions(
    living: &[AgentId],
    actions: &[(AgentId, LocomotionAction)],
) -> Result<BTreeMap<AgentId, LocomotionAction>, SimulationError> {
    let action_map: BTreeMap<_, _> = actions.iter().copied().collect();
    let actual: Vec<_> = action_map.keys().copied().collect();
    if actual != living || action_map.len() != actions.len() {
        return Err(SimulationError::ActionSet {
            expected: living.to_vec(),
            actual,
        });
    }
    Ok(action_map)
}

/// Rank one contiguous species identity under a rotating fair priority order.
fn rotated_priority(id: AgentId, base: usize, count: usize, rotation: usize) -> usize {
    debug_assert!(count > 0, "a priority population must be nonempty");
    let local_identity = usize::from(id.0).saturating_sub(base);
    (local_identity + count - rotation % count) % count
}

/// Return whether one curriculum stage enables exposure and shelter mechanics.
const fn stage_uses_shelter(stage: CurriculumStage) -> bool {
    matches!(
        stage,
        CurriculumStage::Shelter
            | CurriculumStage::Competition
            | CurriculumStage::PredatorPrey
            | CurriculumStage::Obstacles
    )
}

/// Combine only the physiological drives active in one curriculum stage.
fn physiological_drive(
    stage: CurriculumStage,
    physiology: Physiology,
    maximum_satiation: u8,
    maximum_hydration: u8,
    maximum_hit_points: u8,
) -> NormalizedDrive {
    let food_deficit = 1.0 - f32::from(physiology.satiation) / f32::from(maximum_satiation.max(1));
    let water_deficit = 1.0 - f32::from(physiology.hydration) / f32::from(maximum_hydration.max(1));
    let health_deficit =
        1.0 - f32::from(physiology.hit_points) / f32::from(maximum_hit_points.max(1));
    let drive = match stage {
        CurriculumStage::Forage | CurriculumStage::Sprint | CurriculumStage::Gorge => {
            food_deficit.mul_add(0.7, health_deficit * 0.3)
        }
        CurriculumStage::Survival => {
            food_deficit.mul_add(0.20, water_deficit.mul_add(0.60, health_deficit * 0.20))
        }
        CurriculumStage::Shelter
        | CurriculumStage::Competition
        | CurriculumStage::PredatorPrey
        | CurriculumStage::Obstacles => {
            let exposure_deficit =
                1.0 - f32::from(physiology.exposure) / f32::from(MAXIMUM_EXPOSURE);
            food_deficit.mul_add(
                0.25,
                water_deficit.mul_add(0.25, exposure_deficit.mul_add(0.25, health_deficit * 0.25)),
            )
        }
    };
    NormalizedDrive::try_from(drive.clamp(0.0, 1.0))
        .expect("bounded physiology must produce a normalized drive")
}

/// Advance or reset one deprivation clock and report one due damage point.
const fn damage_clock_due(active: bool, elapsed_steps: &mut u32, interval: u32) -> bool {
    if !active {
        *elapsed_steps = 0;
        return false;
    }
    *elapsed_steps = elapsed_steps.saturating_add(1);
    if *elapsed_steps < interval {
        return false;
    }
    *elapsed_steps -= interval;
    true
}

/// Return the movement scale derived from the current physics overlap state.
const fn shallow_water_movement_scale(in_shallow_water: bool) -> f32 {
    if in_shallow_water {
        SHALLOW_WATER_MOVEMENT_SCALE
    } else {
        1.0
    }
}

/// Convert Avian's largest approaching contact speed into bounded HP damage.
fn collision_damage_from_impact_speed(impact_speed: f32) -> u8 {
    if !impact_speed.is_finite() || impact_speed < COLLISION_DAMAGE_SPEED_PER_POINT {
        0
    } else if impact_speed < COLLISION_DAMAGE_SPEED_PER_POINT * 2.0 {
        1
    } else if impact_speed < COLLISION_DAMAGE_SPEED_PER_POINT * 3.0 {
        2
    } else {
        3
    }
}

/// Select rotating recipients that can each withdraw one complete drink.
fn complete_drink_recipients(
    candidates: &mut [AgentId],
    available: f32,
    requested: f32,
    rotation: usize,
) -> Vec<AgentId> {
    if candidates.is_empty() || !requested.is_finite() || requested <= 0.0 {
        return Vec::new();
    }
    candidates.sort_unstable();
    let candidate_count = candidates.len();
    candidates.rotate_left(rotation % candidate_count);
    let mut remaining = available.max(0.0);
    let mut recipients = Vec::with_capacity(candidate_count);
    for id in candidates.iter().copied() {
        if remaining + f32::EPSILON < requested {
            break;
        }
        remaining -= requested;
        recipients.push(id);
    }
    recipients
}

/// Encode one decentralized actor observation without global coordinates.
fn encode_local_observation(
    agent: &AgentBody,
    _position: Vec2,
    rotation: Rotation,
    velocity: LinearVelocity,
    angular_velocity: AngularVelocity,
    age_seconds: f32,
    max_age: f32,
    max_linear_speed: f32,
    gaze_yaw_limit: f32,
    maximum_satiation: u8,
    maximum_hydration: u8,
    maximum_hit_points: u8,
    weather_active: bool,
) -> LocalObservation {
    let mut observation = [0.0; LOCAL_OBSERVATION_SIZE];
    write_feature(
        &mut observation,
        0,
        f32::from(agent.physiology.satiation) / f32::from(maximum_satiation),
    );
    write_feature(
        &mut observation,
        1,
        f32::from(agent.physiology.hydration) / f32::from(maximum_hydration),
    );
    let health_fraction = f32::from(agent.physiology.hit_points) / f32::from(maximum_hit_points);
    write_feature(&mut observation, 2, health_fraction);
    write_feature(&mut observation, 3, (age_seconds / max_age).clamp(0.0, 1.0));
    let body_forward = Vec2::new(rotation.cos, rotation.sin);
    let velocity = velocity.0;
    write_feature(
        &mut observation,
        LOCAL_LONGITUDINAL_VELOCITY_INDEX,
        (velocity.dot(body_forward) / max_linear_speed).clamp(-1.0, 1.0),
    );
    write_feature(
        &mut observation,
        LOCAL_LATERAL_VELOCITY_INDEX,
        (velocity.dot(body_forward.perp()) / max_linear_speed).clamp(-1.0, 1.0),
    );
    write_feature(
        &mut observation,
        6,
        (angular_velocity.0 / MAX_ANGULAR_SPEED).clamp(-1.0, 1.0),
    );
    write_feature(
        &mut observation,
        LOCAL_GAZE_YAW_INDEX,
        (agent.gaze_yaw / gaze_yaw_limit).clamp(-1.0, 1.0),
    );
    write_feature(
        &mut observation,
        LOCAL_EXPOSURE_INDEX,
        f32::from(agent.physiology.exposure) / f32::from(MAXIMUM_EXPOSURE),
    );
    write_feature(
        &mut observation,
        LOCAL_IN_SHELTER_INDEX,
        if agent.physiology.in_shelter {
            1.0
        } else if weather_active {
            -1.0
        } else {
            0.0
        },
    );
    write_feature(
        &mut observation,
        LOCAL_INTERACTION_COOLDOWN_INDEX,
        f32::from(agent.interaction_cooldown_steps) / f32::from(INTERACTION_COOLDOWN_STEPS),
    );

    // A semantic bit establishes the hit, so each ray needs no presence bit.
    for (ray_index, ray) in agent.rays.iter().enumerate() {
        let start = PROPRIOCEPTION_SIZE + ray_index * RAY_FEATURE_SIZE;
        if let Some(kind) = ray.kind {
            write_feature(&mut observation, start, ray.normalized_distance);
            write_feature(&mut observation, start + RAY_KIND_START + kind.index(), 1.0);
        }
    }
    observation
}

/// Return nearest visible proximity for one semantic resource kind.
fn nearest_visible_proximity(agent: &AgentBody, kind: PerceptKind) -> Option<f32> {
    agent
        .rays
        .iter()
        .filter(|sample| sample.kind == Some(kind))
        .map(|sample| (1.0 - sample.normalized_distance).clamp(0.0, 1.0))
        .max_by(f32::total_cmp)
}

/// Write one derived tensor feature when its fixed schema index is present.
fn write_feature(features: &mut [f32], index: usize, value: f32) {
    if let Some(feature) = features.get_mut(index) {
        *feature = value;
    } else {
        debug_assert!(false, "fixed tensor feature index is outside its schema");
    }
}

/// Return the most important fixed center-origin slots for the active profile.
fn active_ray_indices(count: PerceptionRayCount) -> impl Iterator<Item = usize> {
    0..usize::from(count.get())
}

/// Find an active sector only when the target lies inside its fixed arc.
fn nearest_active_ray(relative_angle: f32, count: PerceptionRayCount) -> Option<usize> {
    // Keep the original five-degree half-width when rays are removed. This
    // creates real blind gaps instead of silently widening the remaining rays.
    let (ray_index, sector_angle) = active_ray_indices(count)
        .filter_map(|ray_index| RAY_ANGLES.get(ray_index).map(|angle| (ray_index, *angle)))
        .min_by(|(_, left), (_, right)| {
            angular_distance(relative_angle, *left)
                .total_cmp(&angular_distance(relative_angle, *right))
        })?;
    (angular_distance(relative_angle, sector_angle) <= RAY_HALF_WIDTH).then_some(ray_index)
}

/// Sample one forward-facing semantic cone from the resolved world.
fn update_perceptions(
    spatial_query: SpatialQuery<'_, '_>,
    semantics: Query<'_, '_, (Entity, &Position, &SemanticCollider)>,
    mut agents: Query<'_, '_, (Entity, &Position, &Rotation, &mut AgentBody)>,
    active_perception: Res<'_, ActivePerception>,
) {
    for (entity, position, rotation, mut agent) in &mut agents {
        // Clear every reserved slot first so disabled sectors cannot retain a
        // sample from a denser profile or a prior physics frame.
        agent.rays = [RaySample::MISS; RAY_COUNT];
        if !agent.is_alive() {
            continue;
        }

        let filter = SpatialQueryFilter::from_excluded_entities([entity]);
        let body_forward = Vec2::new(rotation.cos, rotation.sin);
        for ray_index in active_ray_indices(active_perception.0) {
            let Some(relative_angle) = RAY_ANGLES.get(ray_index) else {
                continue;
            };
            let origin = perception_origin(position.0, body_forward);
            let direction = Vec2::from_angle(agent.gaze_yaw + *relative_angle).rotate(body_forward);
            let sample = Dir2::new(direction)
                .ok()
                .and_then(|direction| {
                    spatial_query.cast_ray_predicate(
                        origin,
                        direction,
                        SIGHT_RANGE,
                        false,
                        &filter,
                        &|candidate| semantics.contains(candidate),
                    )
                })
                .and_then(|hit| {
                    semantics
                        .get(hit.entity)
                        .ok()
                        .map(|(_, _, semantic)| RaySample {
                            normalized_distance: (hit.distance / SIGHT_RANGE).clamp(0.0, 1.0),
                            kind: Some(semantic.0),
                        })
                })
                .unwrap_or(RaySample::MISS);
            if let Some(ray) = agent.rays.get_mut(ray_index) {
                *ray = sample;
            }
        }

        // Assign each visible semantic center to its nearest sector. The actor
        // receives only egocentric range and kind, never world coordinates.
        let gaze_forward = Vec2::from_angle(agent.gaze_yaw).rotate(body_forward);
        let origin = perception_origin(position.0, body_forward);
        for (target, target_position, semantic) in &semantics {
            if target == entity {
                continue;
            }
            let offset = target_position.0 - origin;
            let distance = offset.length();
            if !(f32::EPSILON..=SIGHT_RANGE).contains(&distance) {
                continue;
            }
            let Ok(direction) = Dir2::new(offset) else {
                continue;
            };
            let visible = spatial_query
                .cast_ray_predicate(
                    origin,
                    direction,
                    distance + 0.01,
                    false,
                    &filter,
                    &|candidate| semantics.contains(candidate),
                )
                .is_some_and(|hit| hit.entity == target);
            if !visible {
                continue;
            }
            let relative_angle = gaze_forward
                .perp_dot(*direction)
                .atan2(gaze_forward.dot(*direction));
            let Some(sector) = nearest_active_ray(relative_angle, active_perception.0) else {
                continue;
            };
            let normalized_distance = (distance / SIGHT_RANGE).clamp(0.0, 1.0);
            let Some(ray) = agent.rays.get_mut(sector) else {
                continue;
            };
            if ray.kind.is_none() || normalized_distance < ray.normalized_distance {
                *ray = RaySample {
                    normalized_distance,
                    kind: Some(semantic.0),
                };
            }
        }
    }
}

/// Return the shared perception origin centered between both visual eyes.
fn perception_origin(position: Vec2, body_forward: Vec2) -> Vec2 {
    position + body_forward * EYE_FORWARD_OFFSET
}

/// Smallest absolute angular separation between two radians.
fn angular_distance(left: f32, right: f32) -> f32 {
    (left - right).sin().atan2((left - right).cos()).abs()
}

/// Return the shortest distance from a point to a finite ordered segment.
fn point_segment_distance(point: Vec2, start: Vec2, end: Vec2) -> f32 {
    let segment = end - start;
    let length_squared = segment.length_squared();
    if length_squared <= f32::EPSILON {
        return point.distance(start);
    }
    let fraction = ((point - start).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.distance(segment.mul_add(Vec2::splat(fraction), start))
}

/// Collision matrix for dynamic agents.
fn agent_layers() -> CollisionLayers {
    CollisionLayers::new(
        [PhysicsGroup::Agent],
        [
            PhysicsGroup::Agent,
            PhysicsGroup::Solid,
            PhysicsGroup::Sensor,
        ],
    )
}

/// Collision matrix for blocking map geometry.
fn solid_layers() -> CollisionLayers {
    CollisionLayers::new([PhysicsGroup::Solid], [PhysicsGroup::Agent])
}

/// Collision matrix for non-blocking interaction areas.
fn sensor_layers() -> CollisionLayers {
    CollisionLayers::new([PhysicsGroup::Sensor], [PhysicsGroup::Agent])
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    /// Survival defaults must use the requested small integer physiology profile.
    #[test]
    fn survival_defaults_use_integer_points_and_seconds() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");

        assert_eq!(config.maximum_hit_points, 5);
        assert_eq!(config.initial_hit_points, 5);
        assert_eq!(config.maximum_satiation, 5);
        assert_eq!(config.initial_satiation, 3);
        assert_eq!(config.maximum_hydration, 5);
        assert_eq!(config.initial_hydration, 3);
        assert_eq!(config.need_loss_interval_seconds, 60);
        assert_eq!(config.starvation_damage_interval_seconds, 10);
        assert_eq!(config.dehydration_damage_interval_seconds, 10);
        assert_eq!(config.episode_seconds, 20);
        assert!(config.solid_obstacles >= 4);
    }

    /// Survival resources must replenish faster than maximum-motion need loss.
    #[test]
    fn survival_resources_support_indefinite_homeostasis() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let maximum_need_loss_per_second = (1.0
            + f32::from(config.movement_need_cost_percent) / 100.0)
            / f32::from(config.need_loss_interval_seconds);
        let food_per_second = 1.0 / (config.food_spawn_interval as f32 * config.time_step);
        let hydration_per_second = config.well_refill_rate / config.well_drink_rate;

        assert!(food_per_second >= maximum_need_loss_per_second);
        assert!(hydration_per_second >= maximum_need_loss_per_second);
    }

    /// Sprint food must expire after five simulated seconds and reward earlier contact more.
    #[test]
    fn sprint_food_has_a_fixed_deadline_and_speed_scaled_reward() {
        // Compare identical worlds at opposite ends of the target lifetime.
        let config = SimulationConfig::for_stage(CurriculumStage::Sprint)
            .expect("sprint defaults are valid");
        let lifetime_steps = config.steps_for_seconds(EPHEMERAL_FOOD_LIFETIME_SECONDS);
        let mut early = Ecosystem::new(config.clone(), 17).expect("sprint world spawns");
        let early_id = early.living_agents()[0];
        let early_food = {
            let world = early.app.world_mut();
            let mut query = world.query::<(Entity, &EphemeralFood)>();
            let (entity, lifetime) = query.single(world).expect("one ephemeral food exists");
            assert_eq!(lifetime.expires_at_step, lifetime_steps);
            entity
        };
        let early_reward = early.resolve_food(&[(early_food, early_id)])[&early_id];

        let mut late = Ecosystem::new(config.clone(), 17).expect("matching sprint world spawns");
        let late_id = late.living_agents()[0];
        late.step = lifetime_steps - 1;
        let late_food = {
            let world = late.app.world_mut();
            let mut query = world.query_filtered::<Entity, With<EphemeralFood>>();
            query.single(world).expect("one ephemeral food exists")
        };
        let late_reward = late.resolve_food(&[(late_food, late_id)])[&late_id];

        let mut expired = Ecosystem::new(config, 17).expect("expiry sprint world spawns");
        let expired_id = expired.living_agents()[0];
        let expired_food = {
            let world = expired.app.world_mut();
            let mut query = world.query_filtered::<Entity, With<EphemeralFood>>();
            query.single(world).expect("one ephemeral food exists")
        };
        expired.step = lifetime_steps;
        let expired_rewards = expired.resolve_food(&[(expired_food, expired_id)]);

        assert!(early_reward > late_reward);
        assert!(late_reward > 0.0);
        assert!(!expired_rewards.contains_key(&expired_id));
        assert_eq!(food_count(&mut expired), 0);
    }

    /// Gorge food alternates banks while only the marked bridge corridor is safe.
    #[test]
    fn gorge_alternates_food_banks_and_kills_off_bridge_crossings() {
        // Verify both target sequencing and the safe-corridor boundary.
        let config =
            SimulationConfig::for_stage(CurriculumStage::Gorge).expect("gorge defaults are valid");
        let mut ecosystem = Ecosystem::new(config.clone(), 29).expect("gorge world spawns");
        let initial_food_x = food_positions(&mut ecosystem)[0].x;
        despawn_one_food(&mut ecosystem);
        ecosystem.spawn_one_food().expect("replacement food spawns");
        let replacement_food_x = food_positions(&mut ecosystem)[0].x;
        assert_eq!(initial_food_x, -replacement_food_x);

        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        position_agent(
            &mut ecosystem,
            entity,
            Vec2::new(0.0, BRIDGE_HALF_WIDTH + 1.0),
            0.0,
        );
        ecosystem.resolve_gorge_falls();
        let fallen = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(fallen.life, LifeState::Dead(DeathCause::Gorge));

        let mut bridge = Ecosystem::new(config, 29).expect("second gorge world spawns");
        let bridge_id = bridge.living_agents()[0];
        let bridge_entity = bridge.agent_entities()[&bridge_id];
        position_agent(&mut bridge, bridge_entity, Vec2::ZERO, 0.0);
        bridge.resolve_gorge_falls();
        assert!(bridge
            .app
            .world()
            .get::<AgentBody>(bridge_entity)
            .expect("agent exists")
            .is_alive());
    }

    /// Shallow water remains traversable and halves translation speed.
    #[test]
    fn shallow_water_is_nonblocking_and_halves_agent_speed() {
        // Check both collider configuration and a resolved body overlap.
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 17).expect("ecosystem initializes");
        let (is_sensor, has_semantic, well_position) = {
            let world = ecosystem.app.world_mut();
            let mut well_query = world.query_filtered::<
                (Has<Sensor>, Has<SemanticCollider>, &Position),
                With<ShallowWater>,
            >();
            let (is_sensor, has_semantic, position) =
                well_query.single(world).expect("one visible well exists");
            (is_sensor, has_semantic, position.0)
        };
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        position_agent(&mut ecosystem, entity, well_position, 0.0);
        set_agent_motion(&mut ecosystem, id, Vec2::X * MAX_LINEAR_SPEED, 0.0);
        ecosystem.app.update();
        let contacts =
            ecosystem.collect_contacts(&BTreeMap::from([(id, Vec2::X * MAX_LINEAR_SPEED)]));
        ecosystem.resolve_shallow_water(&contacts.shallow_water_agents);
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        let speed = ecosystem
            .app
            .world()
            .get::<LinearVelocity>(entity)
            .expect("agent velocity exists")
            .length();

        assert!(is_sensor);
        assert!(has_semantic);
        assert!(agent.in_shallow_water);
        assert!(speed <= MAX_LINEAR_SPEED * SHALLOW_WATER_MOVEMENT_SCALE);
        assert_eq!(shallow_water_movement_scale(false), 1.0);
        assert_eq!(shallow_water_movement_scale(true), 0.5);
    }

    /// Different episode seeds produce different survival obstacle layouts.
    #[test]
    fn survival_obstacles_regenerate_between_episode_seeds() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut first = Ecosystem::new(config.clone(), 31).expect("first world initializes");
        let mut second = Ecosystem::new(config, 32).expect("second world initializes");

        let first_obstacles = solid_obstacle_clearances(&mut first);
        let second_obstacles = solid_obstacle_clearances(&mut second);
        assert!(first_obstacles.len() >= 4);
        assert_eq!(first_obstacles.len(), second_obstacles.len());
        assert_ne!(first_obstacles, second_obstacles);
    }

    /// Avian impact speed maps monotonically to discrete collision damage.
    #[test]
    fn collision_damage_scales_with_physics_impact_speed() {
        assert_eq!(collision_damage_from_impact_speed(0.0), 0);
        assert_eq!(collision_damage_from_impact_speed(2.9), 0);
        assert_eq!(collision_damage_from_impact_speed(3.0), 1);
        assert_eq!(collision_damage_from_impact_speed(6.0), 2);
        assert_eq!(collision_damage_from_impact_speed(9.0), 3);
    }

    /// A fast Avian wall impact must emit contact speed and remove hit points.
    #[test]
    fn wall_impact_uses_avian_contact_speed_for_damage() {
        // Accelerate through normal controls so Avian creates the wall contact.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.solid_obstacles = 0;
        let mut ecosystem = Ecosystem::new(config, 23).expect("ecosystem initializes");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        position_agent(&mut ecosystem, entity, Vec2::new(6.0, 0.0), 0.0);
        let hit_points_before = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists")
            .physiology
            .hit_points;
        let forward =
            LocomotionAction::new(1.0, 0.0, 0.0, 0.0).expect("full-speed action is valid");
        for _ in 0..20 {
            ecosystem
                .step(&[(id, forward)])
                .expect("wall approach remains a valid episode step");
            let collision_damage = ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .metrics
                .collision_damage;
            if collision_damage > 0.0 {
                break;
            }
        }
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent remains after impact");
        let position = ecosystem
            .app
            .world()
            .get::<Position>(entity)
            .expect("agent position exists")
            .0;
        let velocity = ecosystem
            .app
            .world()
            .get::<LinearVelocity>(entity)
            .expect("agent velocity exists")
            .0;

        assert!(
            agent.physiology.hit_points < hit_points_before,
            "wall approach ended at {position:?} with velocity {velocity:?} and collision damage {}",
            agent.metrics.collision_damage
        );
        assert!(agent.metrics.collision_damage > 0.0);
    }

    /// Bunny steering can turn through more than one revolution per second.
    #[test]
    fn full_turn_input_supports_bunny_quick_turns() {
        assert_eq!(MAX_ANGULAR_SPEED, 16.0);
        assert!(std::f32::consts::PI / MAX_ANGULAR_SPEED <= 0.2);
    }

    /// Ground traction removes lateral skating and makes full braking immediate.
    #[test]
    fn ground_traction_controls_slip_and_braking() {
        // Separate full-throttle lateral grip from the zero-throttle brake endpoint.
        let mut moving = Vec2::new(6.0, 4.0);
        apply_ground_traction(&mut moving, Vec2::X, 1.0);
        assert!((moving.x - 6.0).abs() < 1.0e-6);
        assert!((moving.y - 0.8).abs() < 1.0e-6);

        let mut braking = Vec2::new(4.0, 0.0);
        apply_ground_traction(&mut braking, Vec2::X, 0.0);
        assert!((braking.x - 1.0).abs() < 1.0e-6);
    }

    /// Healthy survival must pay continuously and add one bounded horizon objective.
    #[test]
    fn horizon_emits_bounded_live_rewards_and_records_their_sum() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.episode_seconds = 5;
        config.initial_satiation = config.maximum_satiation;
        config.initial_hydration = config.maximum_hydration;
        let expected_steps = config.max_steps();
        let mut ecosystem = Ecosystem::new(config, 73).expect("ecosystem initializes");
        let id = ecosystem.living_agents()[0];
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("idle action is valid");
        let mut rewards = Vec::new();
        let mut final_status = EpisodeStatus::Continuing;
        for _ in 0..expected_steps {
            let step = ecosystem.step(&[(id, idle)]).expect("world advances");
            rewards.push(step.agents[0].reward);
            final_status = step.agents[0].status;
        }

        assert!(rewards.iter().all(|reward| *reward > 0.0));
        let first_reward = rewards[0];
        let final_reward = rewards[rewards.len() - 1];
        assert!(final_reward > first_reward);
        assert_eq!(final_status, EpisodeStatus::Truncated);
        let recorded_return = rewards.iter().sum::<f64>();
        assert!((5.0..6.0).contains(&recorded_return));
        let metrics = ecosystem.episode_metrics();
        assert!((f64::from(metrics[0].episode_return) - recorded_return).abs() < 1e-6);
    }

    /// The invisible drinking radius must not be encoded as visible water.
    #[test]
    fn well_drinking_sensor_is_nonsemantic_and_round() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 91).expect("ecosystem initializes");
        let world = ecosystem.app.world_mut();
        let mut query =
            world.query_filtered::<(&Collider, Has<SemanticCollider>), With<WellSensor>>();
        let (collider, has_semantic) = query.single(world).expect("one drinking sensor exists");

        assert!(!has_semantic);
        assert!(collider.shape().as_ball().is_some());
    }

    /// A drinking sensor must not hide semantic targets beyond its boundary.
    #[test]
    fn well_drinking_sensor_does_not_clip_or_occlude_perception() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 91).expect("ecosystem initializes");
        let well_position = ecosystem.app.world().resource::<WellState>().position;
        let agent_id = ecosystem.living_agents()[0];
        let agent_entity = ecosystem.agent_entities()[&agent_id];
        let removable_entities = {
            let world = ecosystem.app.world_mut();
            let mut query = world.query::<(Entity, Option<&SemanticCollider>, Option<&FoodSlot>)>();
            query
                .iter(world)
                .filter_map(|(entity, semantic, food)| {
                    (food.is_some()
                        || semantic.is_some_and(|semantic| semantic.0 == PerceptKind::Well))
                    .then_some(entity)
                })
                .collect::<Vec<_>>()
        };
        for entity in removable_entities {
            let _despawned = ecosystem.app.world_mut().despawn(entity);
        }
        ecosystem.app.world_mut().entity_mut(agent_entity).insert((
            Position(well_position),
            Rotation::radians(0.0),
            Transform::from_translation(well_position.extend(0.0)),
            LinearVelocity::ZERO,
        ));
        ecosystem.spawn_food_at(well_position + Vec2::X * 4.0);
        ecosystem.app.update();
        ecosystem.app.world_mut().run_schedule(PerceptionSchedule);

        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(agent_entity)
            .expect("agent remains present");
        assert!(agent
            .rays
            .iter()
            .any(|sample| sample.kind == Some(PerceptKind::Food)));
        assert!(agent
            .rays
            .iter()
            .all(|sample| sample.kind != Some(PerceptKind::Well)));
    }

    /// Dynamic agents must use smooth oval colliders instead of cornered boxes.
    #[test]
    fn agent_collider_is_oval() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 91).expect("ecosystem initializes");
        let world = ecosystem.app.world_mut();
        let mut query = world.query_filtered::<&Collider, With<AgentBody>>();
        let collider = query.single(world).expect("one survival agent exists");

        assert!(collider
            .shape()
            .as_shape::<avian2d::collision::collider::EllipseColliderShape>()
            .is_some());
    }

    /// Agents must separate the solid hurtbox from one forward child hitbox.
    #[test]
    fn interaction_colliders_have_hitbox_and_hurtbox_roles() {
        let config = SimulationConfig::for_stage(CurriculumStage::Forage)
            .expect("forage defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 41).expect("ecosystem initializes");
        let world = ecosystem.app.world_mut();
        let mut agent_query =
            world.query_filtered::<(Entity, Has<HitBox>, Has<HurtBox>), With<AgentBody>>();
        let (agent, has_hitbox, has_hurtbox) = agent_query.single(world).expect("one agent exists");
        assert!(!has_hitbox);
        assert!(has_hurtbox);

        let mut hitbox_query =
            world.query_filtered::<(&ChildOf, &Transform, Has<Sensor>), With<HitBox>>();
        let (child_of, transform, is_sensor) =
            hitbox_query.single(world).expect("one child hitbox exists");
        assert_eq!(child_of.parent(), agent);
        assert!(is_sensor);
        assert!(transform.translation.x > AGENT_SIZE.x * 0.5);

        let mut food_query = world.query_filtered::<Has<HurtBox>, With<FoodSlot>>();
        assert!(food_query.single(world).expect("one food exists"));
    }

    /// Cooldown must block immediate reactivation and remain actor-visible.
    #[test]
    fn interaction_cooldown_blocks_reactivation_and_is_observed() {
        let config = SimulationConfig::for_stage(CurriculumStage::Forage)
            .expect("forage defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 43).expect("ecosystem initializes");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("agent exists")
            .interaction_cooldown_steps = 2;
        let active =
            LocomotionAction::new(-1.0, 0.0, 0.0, 1.0).expect("active interaction is valid");

        ecosystem.apply_actions(&BTreeMap::from([(id, active)]));
        let first = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists after first cooldown step");
        assert!(!first.interaction_active);
        assert_eq!(first.interaction_cooldown_steps, 1);

        let (_, observation, _) = ecosystem.agent_output(id).expect("agent output exists");
        assert!(observation[LOCAL_INTERACTION_COOLDOWN_INDEX] > 0.0);
        ecosystem.apply_actions(&BTreeMap::from([(id, active)]));
        ecosystem.apply_actions(&BTreeMap::from([(id, active)]));
        let ready = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists after cooldown");
        assert!(ready.interaction_active);
        assert_eq!(ready.interaction_cooldown_steps, 0);
    }

    /// A food overlap must remain inert until the agent enables its hitbox.
    #[test]
    fn eating_requires_positive_attack_intent() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Forage)
            .expect("forage defaults are valid");
        config.initial_satiation = 0;
        let mut ecosystem = Ecosystem::new(config, 47).expect("ecosystem initializes");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        let food_position = food_positions(&mut ecosystem)[0];
        ecosystem.app.world_mut().entity_mut(entity).insert((
            Position(food_position),
            Transform::from_translation(food_position.extend(0.0)),
            LinearVelocity::ZERO,
        ));

        let inactive =
            LocomotionAction::new(-1.0, 0.0, 0.0, -1.0).expect("inactive interaction is valid");
        for _ in 0..4 {
            ecosystem.step(&[(id, inactive)]).expect("world advances");
            if resource_contacts_agent::<FoodSlot>(&mut ecosystem, entity) {
                break;
            }
        }
        assert!(resource_contacts_agent::<FoodSlot>(&mut ecosystem, entity));
        assert_eq!(food_count(&mut ecosystem), 1);
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .metrics
                .food_eaten,
            0
        );

        let active =
            LocomotionAction::new(-1.0, 0.0, 0.0, 1.0).expect("active interaction is valid");
        ecosystem.apply_actions(&BTreeMap::from([(id, active)]));
        let contacts = ecosystem.collect_contacts(&BTreeMap::new());
        assert_eq!(contacts.food_winners.len(), 1);
        ecosystem.resolve_food(&contacts.food_winners);
        assert_eq!(food_count(&mut ecosystem), 0);
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .metrics
                .food_eaten,
            1
        );
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists after eating");
        assert!(!agent.interaction_active);
        assert_eq!(agent.physiology.satiation, 1);
        assert_eq!(agent.interaction_cooldown_steps, INTERACTION_COOLDOWN_STEPS);
    }

    /// Shelter must restore exposure and prevent exposure death.
    #[test]
    fn shelter_contact_preserves_an_exposed_agent() {
        let config = SimulationConfig::for_stage(CurriculumStage::Shelter)
            .expect("shelter defaults are valid");
        let damage_steps = config.steps_for_seconds(EXPOSURE_DAMAGE_INTERVAL_SECONDS);
        let mut outside = Ecosystem::new(config.clone(), 43).expect("outside world initializes");
        let mut sheltered = Ecosystem::new(config, 43).expect("sheltered world initializes");
        let id = outside.living_agents()[0];
        for ecosystem in [&mut outside, &mut sheltered] {
            ecosystem.step = ecosystem.weather_onset_step;
            let entity = ecosystem.agent_entities()[&id];
            let mut body = ecosystem
                .app
                .world_mut()
                .get_mut::<AgentBody>(entity)
                .expect("agent exists");
            body.physiology.exposure = 0;
            body.physiology.hit_points = 1;
            body.physiology.satiation = ecosystem.config.maximum_satiation;
            body.physiology.hydration = ecosystem.config.maximum_hydration;
        }

        for _ in 0..damage_steps {
            outside.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
            sheltered.resolve_physiology(&BTreeSet::new(), &BTreeSet::from([id]), &BTreeMap::new());
        }

        let outside_entity = outside.agent_entities()[&id];
        let sheltered_entity = sheltered.agent_entities()[&id];
        assert_eq!(
            outside
                .app
                .world()
                .get::<AgentBody>(outside_entity)
                .expect("outside agent exists")
                .life,
            LifeState::Dead(DeathCause::Exposure)
        );
        let sheltered_body = sheltered
            .app
            .world()
            .get::<AgentBody>(sheltered_entity)
            .expect("sheltered agent exists");
        assert_eq!(sheltered_body.life, LifeState::Alive);
        assert!(sheltered_body.physiology.in_shelter);
        assert!(sheltered_body.physiology.exposure > 0);
    }

    /// Seeded weather onset must preserve protection before exposure starts.
    #[test]
    fn weather_onset_delays_exposure_loss() {
        let config = SimulationConfig::for_stage(CurriculumStage::Shelter)
            .expect("shelter defaults are valid");
        let exposure_steps = config.steps_for_seconds(EXPOSURE_LOSS_INTERVAL_SECONDS);
        let mut ecosystem = Ecosystem::new(config, 47).expect("shelter world initializes");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];

        ecosystem.step = ecosystem.weather_onset_step.saturating_sub(1);
        for _ in 0..exposure_steps {
            ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }
        let protected = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists before weather");
        assert_eq!(protected.physiology.exposure, MAXIMUM_EXPOSURE);

        ecosystem.step = ecosystem.weather_onset_step;
        for _ in 0..exposure_steps {
            ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }
        let exposed = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists during weather");
        assert_eq!(exposed.physiology.exposure, MAXIMUM_EXPOSURE - 1);
    }

    /// Shelter lessons must vary weather onset within the two-to-six-second learning window.
    #[test]
    fn weather_onset_uses_the_shelter_lesson_window() {
        let config = SimulationConfig::for_stage(CurriculumStage::Shelter)
            .expect("shelter defaults are valid");
        let steps_per_second = config.steps_for_seconds(1);
        let onsets = (0..32_u64)
            .map(|seed| {
                Ecosystem::new(config.clone(), seed)
                    .expect("shelter world initializes")
                    .weather_onset_step
                    / steps_per_second
            })
            .collect::<BTreeSet<_>>();

        assert!(onsets.iter().all(|seconds| (2..=6).contains(seconds)));
        assert!(onsets.len() > 1);
    }

    /// Visual rays must project every actor sector at its sampled range.
    #[cfg(feature = "render")]
    #[test]
    fn visual_snapshot_projects_actor_perception_rays() {
        // A one-agent stage makes the expected ray count independent of deaths
        // or agent filtering in the presentation layer.
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 91).expect("ecosystem initializes");

        let snapshot = ecosystem.visual_snapshot();

        assert_eq!(snapshot.rays.len(), RAY_COUNT);
        assert!(snapshot.rays.iter().all(|ray| {
            let offset = Vec2::from_array(ray.end) - Vec2::from_array(ray.start);
            (f32::EPSILON..=SIGHT_RANGE + f32::EPSILON).contains(&offset.length())
        }));
        let shared_origin = snapshot.rays.first().expect("perception ray exists").start;
        assert!(snapshot.rays.iter().all(|ray| ray.start == shared_origin));
    }

    /// The rendered mouth must open while the attack action enables its hitbox.
    #[cfg(feature = "render")]
    #[test]
    fn visual_snapshot_exposes_active_mouth_attack() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 93).expect("ecosystem initializes");
        let id = ecosystem.living_agents()[0];
        let attack = LocomotionAction::new(-1.0, 0.0, 0.0, 1.0).expect("attack action is valid");

        ecosystem.apply_actions(&BTreeMap::from([(id, attack)]));
        let snapshot = ecosystem.visual_snapshot();

        assert!(snapshot.agents[0].attack_active);
    }

    /// A reduced ray profile must keep the actor tensor width unchanged.
    #[cfg(feature = "render")]
    #[test]
    fn reduced_perception_uses_only_evenly_spaced_active_rays() {
        // Four rays retain paired frontal samples while unused observation
        // slots remain part of the stable network contract.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        config
            .apply_experiment_tuning(ExperimentTuning {
                perception_ray_count: PerceptionRayCount::try_from(4_u8).expect("four rays fit"),
                ..ExperimentTuning::default()
            })
            .expect("bounded tuning is valid");
        let mut ecosystem = Ecosystem::new(config, 91).expect("ecosystem initializes");

        let snapshot = ecosystem.visual_snapshot();
        let state = ecosystem.state();

        assert_eq!(snapshot.rays.len(), 4);
        assert_eq!(state.agents[0].2.len(), LOCAL_OBSERVATION_SIZE);
        let ray_count = PerceptionRayCount::try_from(4_u8).expect("four rays fit");
        assert_eq!(
            active_ray_indices(ray_count).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(nearest_active_ray(2.0_f32.to_radians(), ray_count), Some(0));
        assert_eq!(
            nearest_active_ray(-(2.0_f32.to_radians()), ray_count),
            Some(1)
        );
        assert_eq!(nearest_active_ray(30.0_f32.to_radians(), ray_count), None);
    }

    /// Reset health and the episode limit must affect the authoritative world.
    #[cfg(feature = "render")]
    #[test]
    fn experiment_tuning_changes_starting_health_and_horizon() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        config
            .apply_experiment_tuning(ExperimentTuning {
                initial_hit_points: 2,
                maximum_satiation: 20,
                initial_satiation: 20,
                maximum_hydration: 20,
                initial_hydration: 20,
                need_loss_interval_seconds: 60,
                episode_seconds: 10,
                ..ExperimentTuning::default()
            })
            .expect("bounded tuning is valid");
        let mut ecosystem = Ecosystem::new(config, 93).expect("ecosystem initializes");
        let agent = *ecosystem
            .visual_snapshot()
            .agents
            .first()
            .expect("solo agent is visible");
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("idle action is valid");
        let mut final_status = EpisodeStatus::Continuing;

        // The minimum timeout remains long enough to exercise real simulation
        // steps while low drain keeps this assertion about truncation stable.
        for _ in 0..100 {
            let id = *ecosystem
                .living_agents()
                .first()
                .expect("solo agent remains alive");
            final_status = ecosystem
                .step(&[(id, idle)])
                .expect("world advances")
                .agents
                .first()
                .expect("solo result exists")
                .status;
        }

        assert_eq!(agent.hit_points, 2);
        assert_eq!(final_status, EpisodeStatus::Truncated);
    }

    /// Equal seeds must reproduce the same initial centralized world state.
    #[test]
    fn reset_seed_reproduces_initial_world() {
        let config = SimulationConfig::for_stage(CurriculumStage::Obstacles)
            .expect("obstacle defaults are valid");
        let first = Ecosystem::new(config.clone(), 17).expect("first world spawns");
        let second = Ecosystem::new(config, 17).expect("second world spawns");
        assert_eq!(initial_state(first), initial_state(second));
    }

    /// Held-out forage resets must span both sides and distinct range bands.
    #[test]
    fn forage_reset_covers_wide_left_right_near_and_far_targets() {
        let mut sides = BTreeSet::new();
        let mut has_near = false;
        let mut has_far = false;
        let mut has_wide_bearing = false;
        for seed in 1..=64 {
            let config = SimulationConfig::for_stage(CurriculumStage::Forage)
                .expect("forage defaults are valid");
            let mut ecosystem = Ecosystem::new(config, seed).expect("world spawns");
            let id = ecosystem.living_agents()[0];
            let entity = ecosystem.agent_entities()[&id];
            let position = ecosystem
                .app
                .world()
                .get::<Position>(entity)
                .expect("position exists")
                .0;
            let rotation = *ecosystem
                .app
                .world()
                .get::<Rotation>(entity)
                .expect("rotation exists");
            let food = food_positions(&mut ecosystem)[0];
            let offset = food - position;
            let bearing = relative_bearing(Vec2::new(rotation.cos, rotation.sin), offset);

            assert!((8.0..=14.0).contains(&offset.length()));
            sides.insert(bearing.is_sign_positive());
            has_near |= offset.length() < 10.0;
            has_far |= offset.length() > 12.0;
            has_wide_bearing |= bearing.abs() > 50.0_f32.to_radians();
        }

        assert_eq!(sides, BTreeSet::from([false, true]));
        assert!(has_near);
        assert!(has_far);
        assert!(has_wide_bearing);
    }

    /// The pre-promotion forage lesson must keep food short and central.
    #[test]
    fn forage_foundation_reset_keeps_targets_short_and_central() {
        for seed in 1..=64 {
            let mut config = SimulationConfig::for_stage(CurriculumStage::Forage)
                .expect("forage defaults are valid");
            config.forage_difficulty = ForageDifficulty::Foundation;
            let mut ecosystem = Ecosystem::new(config, seed).expect("world spawns");
            let id = ecosystem.living_agents()[0];
            let entity = ecosystem.agent_entities()[&id];
            let position = ecosystem
                .app
                .world()
                .get::<Position>(entity)
                .expect("position exists")
                .0;
            let rotation = *ecosystem
                .app
                .world()
                .get::<Rotation>(entity)
                .expect("rotation exists");
            let food = food_positions(&mut ecosystem)[0];
            let offset = food - position;
            let bearing = relative_bearing(Vec2::new(rotation.cos, rotation.sin), offset);

            assert!(offset.length() <= 4.0);
            assert!(bearing.abs() <= 12.0_f32.to_radians());
        }
    }

    /// Survival resets vary resources and place obstacles in actor sightlines.
    #[test]
    fn survival_reset_varies_targets_and_requires_exploration() {
        // Aggregate seeded layouts before asserting coverage of each variation axis.
        let mut food_sides = BTreeSet::new();
        let mut well_sides = BTreeSet::new();
        let mut side_pairs = BTreeSet::new();
        let mut food_is_nearer = BTreeSet::new();
        let mut food_distance_buckets = BTreeSet::new();
        let mut well_distance_buckets = BTreeSet::new();
        let mut obstacle_sightlines = 0_usize;
        let mut fully_visible_resets = 0_usize;
        for seed in 1..=64 {
            let config = SimulationConfig::for_stage(CurriculumStage::Survival)
                .expect("survival defaults are valid");
            let mut ecosystem = Ecosystem::new(config, seed).expect("world spawns");
            let id = ecosystem.living_agents()[0];
            let entity = ecosystem.agent_entities()[&id];
            let position = ecosystem
                .app
                .world()
                .get::<Position>(entity)
                .expect("position exists")
                .0;
            let rotation = *ecosystem
                .app
                .world()
                .get::<Rotation>(entity)
                .expect("rotation exists");
            let well = ecosystem.app.world().resource::<WellState>().position;
            let observation = ecosystem.state().agents[0].2;
            let food_hit_angles = observation_hit_angles(&observation, PerceptKind::Food);
            let well_hit_angles = observation_hit_angles(&observation, PerceptKind::Well);
            let nearest_food = food_positions(&mut ecosystem)
                .into_iter()
                .min_by(|left, right| left.distance(position).total_cmp(&right.distance(position)))
                .expect("survival lesson has food");
            let forward = Vec2::new(rotation.cos, rotation.sin);
            let food_bearing = relative_bearing(forward, nearest_food - position);
            let well_bearing = relative_bearing(forward, well - position);

            let food_is_visible = observation_contains(&observation, PerceptKind::Food);
            let well_is_visible = observation_contains(&observation, PerceptKind::Well);
            obstacle_sightlines = obstacle_sightlines.saturating_add(usize::from(
                observation_contains(&observation, PerceptKind::SolidObstacle),
            ));
            fully_visible_resets = fully_visible_resets
                .saturating_add(usize::from(food_is_visible && well_is_visible));
            assert_eq!(food_is_visible, !food_hit_angles.is_empty());
            assert_eq!(well_is_visible, !well_hit_angles.is_empty());
            assert!(food_bearing.abs() > 0.1);
            assert!(well_bearing.abs() > 0.1);
            assert!(nearest_food.distance(position) > AGENT_RADIUS + FOOD_RADIUS);
            assert!(well.distance(position) > AGENT_RADIUS + WELL_SENSOR_RADIUS);
            assert!(!resource_contacts_agent::<FoodSlot>(&mut ecosystem, entity));
            assert!(!resource_contacts_agent::<WellSensor>(
                &mut ecosystem,
                entity
            ));
            assert_eq!(food_count(&mut ecosystem), 2);
            let food_is_positive = food_bearing.is_sign_positive();
            let well_is_positive = well_bearing.is_sign_positive();
            food_sides.insert(food_is_positive);
            well_sides.insert(well_is_positive);
            side_pairs.insert((food_is_positive, well_is_positive));
            food_is_nearer.insert(nearest_food.distance(position) < well.distance(position));
            food_distance_buckets.insert(nearest_food.distance(position).round() as i32);
            well_distance_buckets.insert(well.distance(position).round() as i32);
        }
        assert_eq!(food_sides, BTreeSet::from([false, true]));
        assert_eq!(well_sides, BTreeSet::from([false, true]));
        assert_eq!(
            side_pairs,
            BTreeSet::from([(false, false), (false, true), (true, false), (true, true)])
        );
        assert_eq!(food_is_nearer, BTreeSet::from([false, true]));
        assert!(food_distance_buckets.len() > 1);
        assert!(well_distance_buckets.len() > 1);
        assert!(obstacle_sightlines > 0);
        assert!(fully_visible_resets > 0);
    }

    /// One action must be supplied for every and only living agent.
    #[test]
    fn joint_action_set_is_exact() {
        let config = SimulationConfig::for_stage(CurriculumStage::Competition)
            .expect("competition defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 3).expect("world spawns");
        let error = ecosystem
            .step(&[])
            .expect_err("missing actions are rejected");
        assert!(matches!(error, SimulationError::ActionSet { .. }));
    }

    /// Evaluation rotations move identities through unchanged procedural slots.
    #[test]
    fn spawn_rotation_is_explicit_and_preserves_map_generation() {
        let config = SimulationConfig::for_stage(CurriculumStage::Competition)
            .expect("competition defaults are valid");
        let mut unrotated = Ecosystem::new_rotated(config.clone(), 7, 0).expect("world spawns");
        let mut rotated = Ecosystem::new_rotated(config, 7, 1).expect("rotated world spawns");
        let unrotated_positions = agent_positions(&mut unrotated);
        let rotated_positions = agent_positions(&mut rotated);

        assert_eq!(
            unrotated_positions[&AgentId(0)],
            rotated_positions[&AgentId(1)]
        );
        assert_eq!(
            unrotated_positions[&AgentId(1)],
            rotated_positions[&AgentId(2)]
        );
        assert_eq!(
            unrotated_positions[&AgentId(3)],
            rotated_positions[&AgentId(0)]
        );
        assert_eq!(
            unrotated.app.world().resource::<WellState>().position,
            rotated.app.world().resource::<WellState>().position
        );
    }

    /// Every emitted local scalar must remain finite and normalized.
    #[test]
    fn local_observations_are_fixed_and_bounded() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 9).expect("world spawns");
        let id = ecosystem
            .living_agents()
            .into_iter()
            .next()
            .expect("solo bunny exists");
        let action = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("idle action is valid");
        let step = ecosystem.step(&[(id, action)]).expect("world steps");
        let observation = &step
            .agents
            .first()
            .expect("bunny result exists")
            .observation;
        assert_eq!(observation.len(), LOCAL_OBSERVATION_SIZE);
        assert!(observation.iter().all(|value| value.is_finite()));
        assert!(observation.iter().all(|value| (-1.0..=1.0).contains(value)));
    }

    /// Translation must accelerate need loss while rotation remains free.
    #[test]
    fn movement_accelerates_need_loss_without_charging_for_looking() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.need_loss_interval_seconds = 1;
        config.movement_need_cost_percent = 25;
        let mut idle = Ecosystem::new(config.clone(), 19).expect("idle world spawns");
        let mut looking = Ecosystem::new(config.clone(), 19).expect("looking world spawns");
        let mut moving = Ecosystem::new(config, 19).expect("moving world spawns");
        let id = idle.living_agents()[0];

        set_agent_motion(&mut looking, id, Vec2::ZERO, MAX_ANGULAR_SPEED);
        set_agent_motion(&mut moving, id, Vec2::X * MAX_LINEAR_SPEED, 0.0);
        for _ in 0..8 {
            idle.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
            looking.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
            moving.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }

        let idle_needs = agent_needs(&mut idle, id);
        let looking_needs = agent_needs(&mut looking, id);
        let moving_needs = agent_needs(&mut moving, id);
        assert_eq!(idle_needs, looking_needs);
        assert_eq!(idle_needs, (3, 3));
        assert_eq!(moving_needs, (2, 2));
    }

    /// Forward throttle must never accelerate behind the body or eye cones.
    #[test]
    fn forward_throttle_matches_body_and_perception_heading() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 31).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        let rotation = *ecosystem
            .app
            .world()
            .get::<Rotation>(entity)
            .expect("agent rotation exists");
        let body_forward = Vec2::new(rotation.cos, rotation.sin);
        let reverse = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("brake action is valid");
        ecosystem.apply_actions(&BTreeMap::from([(id, reverse)]));
        let acceleration = ecosystem
            .app
            .world()
            .get::<ConstantLocalLinearAcceleration>(entity)
            .expect("agent acceleration exists");
        assert_eq!(acceleration.0, Vec2::ZERO);

        let neutral = LocomotionAction::new(0.0, 0.0, 0.0, 0.0).expect("neutral action is valid");
        ecosystem.apply_actions(&BTreeMap::from([(id, neutral)]));
        let acceleration = ecosystem
            .app
            .world()
            .get::<ConstantLocalLinearAcceleration>(entity)
            .expect("agent acceleration exists");
        assert!(acceleration.0.x > 0.0);

        let forward = LocomotionAction::new(1.0, 0.0, 0.0, 0.0).expect("forward action is valid");
        ecosystem.step(&[(id, forward)]).expect("world advances");
        let velocity = ecosystem
            .app
            .world()
            .get::<LinearVelocity>(entity)
            .expect("agent velocity exists");
        assert!(velocity.0.dot(body_forward) > 0.0);
        let gaze_yaw = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists")
            .gaze_yaw;
        let gaze_forward = Vec2::from_angle(gaze_yaw).rotate(body_forward);
        assert!(velocity.0.dot(gaze_forward) > 0.0);
    }

    /// Food sensors must be traversable and consumed by a steering agent.
    #[test]
    fn steering_agent_crosses_and_collects_food_sensor() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.initial_satiation = 0;
        let maximum_turn_per_step = MAX_ANGULAR_SPEED * config.time_step;
        let mut ecosystem = Ecosystem::new(config, 37).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        let start = ecosystem
            .app
            .world()
            .get::<Position>(entity)
            .expect("agent position exists")
            .0;
        // Select the first lesson target before collection can despawn it.
        let target = food_positions(&mut ecosystem)
            .into_iter()
            .min_by(|left, right| left.distance(start).total_cmp(&right.distance(start)))
            .expect("survival lesson has food");
        for _ in 0..40 {
            let position = ecosystem
                .app
                .world()
                .get::<Position>(entity)
                .expect("agent position exists")
                .0;
            let rotation = *ecosystem
                .app
                .world()
                .get::<Rotation>(entity)
                .expect("agent rotation exists");
            let heading = Vec2::new(rotation.cos, rotation.sin);
            let bearing = relative_bearing(heading, target - position);
            let forward = if bearing.abs() < 0.2 { 1.0 } else { -1.0 };
            // Request only the turn needed for this step under the configured cap.
            let turn = (bearing / maximum_turn_per_step).clamp(-1.0, 1.0);
            let action =
                LocomotionAction::new(forward, turn, 0.0, 1.0).expect("steering action is valid");
            ecosystem.step(&[(id, action)]).expect("world advances");
            let has_eaten = ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .is_some_and(|agent| agent.metrics.food_eaten > 0);
            if has_eaten {
                break;
            }
        }
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        let end = ecosystem
            .app
            .world()
            .get::<Position>(entity)
            .expect("agent position exists")
            .0;
        assert!(agent.metrics.food_eaten > 0);
        assert!(agent.metrics.path_length > 2.0);
        assert!(agent.metrics.first_resource_contact_seconds.is_some());
        assert!(agent.metrics.visible_resource_transitions > 0);
        assert!(agent.metrics.resource_approach_transitions > 0);
        assert!(end.distance(target) + 2.0 < start.distance(target));
    }

    /// Dehydration retains terminal attribution at its exact integer boundary.
    #[test]
    fn dehydration_damage_has_explicit_terminal_attribution() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let damage_interval = config.steps_for_seconds(config.dehydration_damage_interval_seconds);
        let mut ecosystem = Ecosystem::new(config, 29).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        {
            let mut agent = ecosystem
                .app
                .world_mut()
                .get_mut::<AgentBody>(entity)
                .expect("agent exists");
            agent.physiology.satiation = 1;
            agent.physiology.hydration = 0;
            agent.physiology.hit_points = 1;
            agent.physiology.dehydration_steps = damage_interval - 1;
        }

        ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());

        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(agent.life, LifeState::Dead(DeathCause::Dehydration));
    }

    /// Concurrent deprivation and thorn damage must not claim one sole cause.
    #[test]
    fn combined_deprivation_and_thorn_damage_has_combined_attribution() {
        let config = SimulationConfig::for_stage(CurriculumStage::Obstacles)
            .expect("obstacle defaults are valid");
        let starvation_interval =
            config.steps_for_seconds(config.starvation_damage_interval_seconds);
        let thorn_interval = config.steps_for_seconds(THORN_DAMAGE_INTERVAL_SECONDS);
        let mut ecosystem = Ecosystem::new(config, 31).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        {
            let mut agent = ecosystem
                .app
                .world_mut()
                .get_mut::<AgentBody>(entity)
                .expect("agent exists");
            agent.physiology.satiation = 0;
            agent.physiology.hydration = 1;
            agent.physiology.hit_points = 2;
            agent.physiology.starvation_steps = starvation_interval - 1;
            agent.physiology.thorn_steps = thorn_interval - 1;
        }

        ecosystem.resolve_physiology(&BTreeSet::from([id]), &BTreeSet::new(), &BTreeMap::new());

        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(agent.life, LifeState::Dead(DeathCause::CombinedDamage));
    }

    /// Need loss and deprivation damage must occur only at exact whole-second boundaries.
    #[test]
    fn integer_physiology_clocks_fire_at_configured_boundaries() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.starvation_damage_interval_seconds = 3;
        config.dehydration_damage_interval_seconds = 2;
        let need_interval = config.steps_for_seconds(config.need_loss_interval_seconds);
        let dehydration_interval =
            config.steps_for_seconds(config.dehydration_damage_interval_seconds);
        let starvation_interval =
            config.steps_for_seconds(config.starvation_damage_interval_seconds);
        let initial_satiation = config.initial_satiation;
        let initial_hydration = config.initial_hydration;
        let mut ecosystem = Ecosystem::new(config, 97).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];

        for _ in 1..need_interval {
            ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }
        let before_need_loss = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(before_need_loss.physiology.satiation, initial_satiation);
        assert_eq!(before_need_loss.physiology.hydration, initial_hydration);

        ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        {
            let mut agent = ecosystem
                .app
                .world_mut()
                .get_mut::<AgentBody>(entity)
                .expect("agent exists");
            assert_eq!(
                agent.physiology.satiation,
                initial_satiation.saturating_sub(1)
            );
            assert_eq!(
                agent.physiology.hydration,
                initial_hydration.saturating_sub(1)
            );
            agent.physiology.satiation = 0;
            agent.physiology.hydration = 0;
            agent.physiology.hit_points = 5;
            agent.physiology.satiation_need_steps = 0;
            agent.physiology.hydration_need_steps = 0;
            agent.physiology.starvation_steps = 0;
            agent.physiology.dehydration_steps = 0;
        }

        for _ in 1..dehydration_interval {
            ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .physiology
                .hit_points,
            5
        );

        ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .physiology
                .hit_points,
            4
        );

        for _ in dehydration_interval..starvation_interval {
            ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .physiology
                .hit_points,
            3
        );
    }

    /// A need that reaches zero starts its damage interval on the next step.
    #[test]
    fn deprivation_interval_starts_after_need_reaches_zero() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let need_interval = config.steps_for_seconds(config.need_loss_interval_seconds);
        let starvation_interval =
            config.steps_for_seconds(config.starvation_damage_interval_seconds);
        let mut ecosystem = Ecosystem::new(config, 98).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        {
            let mut agent = ecosystem
                .app
                .world_mut()
                .get_mut::<AgentBody>(entity)
                .expect("agent exists");
            agent.physiology.satiation = 1;
            agent.physiology.hydration = ecosystem.config.maximum_hydration;
            agent.physiology.hit_points = ecosystem.config.maximum_hit_points;
            agent.physiology.satiation_need_steps = need_interval - 1;
            agent.physiology.starvation_steps = 0;
        }

        ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        let depleted = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(depleted.physiology.satiation, 0);
        assert_eq!(depleted.physiology.starvation_steps, 0);
        let initial_hit_points = depleted.physiology.hit_points;

        for _ in 1..starvation_interval {
            ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        }
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .physiology
                .hit_points,
            initial_hit_points
        );

        ecosystem.resolve_physiology(&BTreeSet::new(), &BTreeSet::new(), &BTreeMap::new());
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("agent exists")
                .physiology
                .hit_points,
            initial_hit_points - 1
        );
    }

    /// Forward center-origin perception must leave a true rear blind area.
    #[test]
    fn local_perception_excludes_rear_resource() {
        let observation = observe_single_food(std::f32::consts::PI, 4.0);
        let food_rays = (0..RAY_COUNT)
            .filter(|ray_index| {
                let start = PROPRIOCEPTION_SIZE + ray_index * RAY_FEATURE_SIZE;
                observation[start + RAY_KIND_START + PerceptKind::Food.index()] > 0.5
            })
            .collect::<Vec<_>>();
        assert!(
            !observation_contains(&observation, PerceptKind::Food),
            "rear food must stay outside both eye cones; hits={food_rays:?}"
        );
    }

    /// Every semantic ray must share one origin centered between the eyes.
    #[test]
    fn perception_origin_is_centered_between_the_eyes() {
        assert_eq!(
            perception_origin(Vec2::new(3.0, -2.0), Vec2::X),
            Vec2::new(3.0 + EYE_FORWARD_OFFSET, -2.0)
        );
    }

    /// Gaze can inspect a peripheral target without rotating the body.
    #[test]
    fn bounded_gaze_moves_both_eye_cones_without_rear_vision() {
        let bearing = 80.0_f32.to_radians();
        let centered = observe_single_food_with_gaze(bearing, 5.0, 0.0);
        let focused = observe_single_food_with_gaze(bearing, 5.0, 1.0);

        assert!(!observation_contains(&centered, PerceptKind::Food));
        assert!(observation_contains(&focused, PerceptKind::Food));
    }

    /// A small resource between sector centerlines must remain perceptible.
    #[test]
    fn local_perception_has_no_needle_gap_between_sectors() {
        let observation = observe_single_food(5.0_f32.to_radians(), 7.0);
        assert!(
            observation_contains(&observation, PerceptKind::Food),
            "off-center food must produce a widened semantic hit"
        );
    }

    /// Body-relative velocity must distinguish approach, retreat, and lateral drift.
    #[test]
    fn local_observation_encodes_signed_body_relative_velocity() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 17).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        *ecosystem
            .app
            .world_mut()
            .get_mut::<Rotation>(entity)
            .expect("agent rotation exists") = Rotation::radians(0.0);

        set_agent_motion(&mut ecosystem, id, Vec2::new(4.0, -2.0), 0.0);
        let forward = ecosystem.agent_output(id).expect("agent output exists").1;
        set_agent_motion(&mut ecosystem, id, Vec2::new(-4.0, 2.0), 0.0);
        let backward = ecosystem.agent_output(id).expect("agent output exists").1;

        assert!(forward[LOCAL_LONGITUDINAL_VELOCITY_INDEX] > 0.0);
        assert!(forward[LOCAL_LATERAL_VELOCITY_INDEX] < 0.0);
        assert!(backward[LOCAL_LONGITUDINAL_VELOCITY_INDEX] < 0.0);
        assert!(backward[LOCAL_LATERAL_VELOCITY_INDEX] > 0.0);
    }

    /// Resources outside the visible ray sectors must provide no actor input.
    #[test]
    fn local_observation_reports_resources_only_through_rays() {
        let left = observe_single_food(25.0_f32.to_radians(), 8.0);
        let right = observe_single_food((-25.0_f32).to_radians(), 8.0);
        let near = observe_single_food(0.0, 4.0);
        let far = observe_single_food(0.0, 10.0);
        let behind = observe_single_food(std::f32::consts::PI, 4.0);

        assert!(observation_hit_angles(&left, PerceptKind::Food)[0] > 0.0);
        assert!(observation_hit_angles(&right, PerceptKind::Food)[0] < 0.0);
        assert!(
            nearest_observed_distance(&near, PerceptKind::Food)
                < nearest_observed_distance(&far, PerceptKind::Food)
        );
        assert!(!observation_contains(&behind, PerceptKind::Food));
    }

    /// Snapshot maxima must remain attached to terminal visual state.
    #[cfg(feature = "render")]
    #[test]
    fn visual_snapshot_includes_configured_status_maxima() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config
            .apply_experiment_tuning(ExperimentTuning {
                maximum_hit_points: 7,
                maximum_satiation: 8,
                maximum_hydration: 9,
                ..ExperimentTuning::default()
            })
            .expect("custom maxima are valid");
        let mut ecosystem = Ecosystem::new(config, 17).expect("world spawns");

        let snapshot = ecosystem.visual_snapshot();

        assert_eq!(snapshot.maximum_hit_points, 7);
        assert_eq!(snapshot.maximum_satiation, 8);
        assert_eq!(snapshot.maximum_hydration, 9);
    }
    /// Repeated spawn and consumption cycles must retain unique critic slots.
    #[test]
    fn live_food_critic_slots_remain_unique_after_reuse() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Competition)
            .expect("competition defaults are valid");
        config.initial_food = config.max_food;
        let mut ecosystem = Ecosystem::new(config, 31).expect("world spawns");

        for _ in 0..MAX_FOOD * 2 {
            let removed = {
                let world = ecosystem.app.world_mut();
                let mut query = world.query::<(Entity, &FoodSlot)>();
                query.iter(world).next().map(|(entity, _)| entity)
            };
            let removed = removed.expect("at least one food entity exists");
            assert!(ecosystem.app.world_mut().despawn(removed));
            ecosystem.spawn_one_food().expect("replacement food spawns");

            let slots = {
                let world = ecosystem.app.world_mut();
                let mut query = world.query::<&FoodSlot>();
                query.iter(world).map(|slot| slot.0).collect::<Vec<_>>()
            };
            let unique = slots.iter().copied().collect::<BTreeSet<_>>();
            assert_eq!(slots.len(), ecosystem.config.max_food);
            assert_eq!(slots.len(), unique.len());
            assert!(
                slots
                    .iter()
                    .all(|slot| usize::from(*slot) < ecosystem.config.max_food),
                "live food must stay inside configured critic slots: {slots:?}"
            );
        }
    }

    /// Survival preserves one food while periodically replenishing toward capacity.
    #[test]
    fn survival_food_supply_replenishes_from_one_toward_six() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        assert_eq!(config.initial_food, 2);
        assert_eq!(config.max_food, 6);
        assert_eq!(config.food_spawn_interval, 20);
        let interval = config.food_spawn_interval;
        let mut ecosystem = Ecosystem::new(config, 37).expect("world spawns");
        assert_eq!(food_count(&mut ecosystem), 2);

        despawn_one_food(&mut ecosystem);
        let id = ecosystem.living_agents()[0];
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("idle action is valid");
        for _ in 1..interval {
            ecosystem.step(&[(id, idle)]).expect("world advances");
            assert_eq!(food_count(&mut ecosystem), 1);
        }
        ecosystem
            .step(&[(id, idle)])
            .expect("spawn interval advances");
        assert_eq!(food_count(&mut ecosystem), 2);

        despawn_all_food(&mut ecosystem);
        ecosystem
            .step(&[(id, idle)])
            .expect("empty food floor advances");
        assert_eq!(food_count(&mut ecosystem), 1);
    }

    /// An immobile bunny must eventually die and lose all physics capability.
    #[test]
    fn unmet_needs_terminate_agent_and_remove_collider() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config
            .apply_experiment_tuning(ExperimentTuning {
                initial_satiation: 0,
                initial_hydration: 0,
                episode_seconds: 60,
                ..ExperimentTuning::default()
            })
            .expect("deprivation profile is valid");
        let mut ecosystem = Ecosystem::new(config, 43).expect("world spawns");
        let id = ecosystem
            .living_agents()
            .into_iter()
            .next()
            .expect("solo bunny exists");
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("idle action is valid");
        let terminal = loop {
            let step = ecosystem.step(&[(id, idle)]).expect("world advances");
            let agent = step.agents.first().expect("bunny result exists");
            if agent.status.is_terminal() {
                break agent.death_cause;
            }
        };
        assert!(matches!(
            terminal,
            Some(DeathCause::Starvation | DeathCause::Dehydration | DeathCause::Deprivation)
        ));

        let world = ecosystem.app.world_mut();
        let mut query = world.query::<(&AgentBody, Has<Collider>)>();
        let (_, has_collider) = query
            .iter(world)
            .find(|(agent, _)| agent.id == id)
            .expect("dead bunny remains available for episode metrics");
        assert!(!has_collider);
    }

    /// One full thorn interval must remove one discrete HP.
    #[test]
    fn thorn_damage_reduces_health_and_hit_points() {
        let config = SimulationConfig::for_stage(CurriculumStage::Obstacles)
            .expect("obstacle defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 47).expect("world spawns");
        let id = ecosystem
            .living_agents()
            .into_iter()
            .next()
            .expect("a bunny exists");
        let before = ecosystem
            .episode_metrics()
            .into_iter()
            .find(|metrics| metrics.id == id)
            .expect("bunny metrics exist");
        for _ in 0..ecosystem
            .config
            .steps_for_seconds(THORN_DAMAGE_INTERVAL_SECONDS)
        {
            ecosystem.resolve_physiology(&BTreeSet::from([id]), &BTreeSet::new(), &BTreeMap::new());
        }
        let entity = ecosystem
            .agent_entities()
            .get(&id)
            .copied()
            .expect("bunny entity exists");
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny body exists");
        assert_eq!(agent.physiology.hit_points, 4);
        assert!(agent.metrics.thorn_damage > before.thorn_damage);
    }

    /// A fully hydrated agent must not remove unusable water from the well.
    #[test]
    fn full_hydration_contact_preserves_well_water() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 51).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        let capacity = ecosystem.config.maximum_hydration;
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("bunny exists")
            .physiology
            .hydration = capacity;
        let before = ecosystem.app.world().resource::<WellState>().water;
        prime_drinking_event(&mut ecosystem, id);

        ecosystem.resolve_drinking(&BTreeSet::from([id]));

        let after = ecosystem.app.world().resource::<WellState>().water;
        assert!((after - before).abs() < f32::EPSILON);
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert!(agent.metrics.water_consumed.abs() < f32::EPSILON);
    }

    /// Hydration and conserved water must advance after one half-second contact.
    #[test]
    fn drinking_requires_one_continuous_half_second_per_event() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let interval = DRINKING_INTERVAL_STEPS;
        let mut ecosystem = Ecosystem::new(config, 61).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("bunny exists")
            .physiology
            .hydration = 0;

        for _ in 1..interval {
            ecosystem.resolve_drinking(&BTreeSet::from([id]));
        }
        let before_event = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert_eq!(before_event.physiology.hydration, 0);
        assert_eq!(before_event.metrics.water_consumed, 0.0);

        ecosystem.resolve_drinking(&BTreeSet::from([id]));
        let after_event = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert_eq!(
            after_event.physiology.hydration,
            ecosystem.config.maximum_hydration
        );
        assert_eq!(
            after_event.metrics.water_consumed,
            ecosystem.config.well_drink_rate
        );
        assert!(!after_event.interaction_active);
        assert_eq!(
            after_event.interaction_cooldown_steps,
            INTERACTION_COOLDOWN_STEPS
        );
    }

    /// One survival drink withdraws one fixed allocation and restores hydration.
    #[test]
    fn partial_hydration_contact_withdraws_only_absorbable_water() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let drink_rate = config.well_drink_rate;
        let mut ecosystem = Ecosystem::new(config, 52).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        let capacity = ecosystem.config.maximum_hydration;
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("bunny exists")
            .physiology
            .hydration = capacity - 1;
        let expected = drink_rate;
        let before = ecosystem.app.world().resource::<WellState>().water;
        prime_drinking_event(&mut ecosystem, id);

        ecosystem.resolve_drinking(&BTreeSet::from([id]));

        let after = ecosystem.app.world().resource::<WellState>().water;
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert!((before - after - expected).abs() < 1e-6);
        assert!((agent.metrics.water_consumed - expected).abs() < 1e-6);
        assert_eq!(agent.physiology.hydration, capacity);
    }

    /// Drinking reward must use hydration before the absorbed water raises it.
    #[test]
    fn dehydrated_drinker_consumes_one_conserved_water_allocation() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let expected_water = config.well_drink_rate;
        let mut ecosystem = Ecosystem::new(config, 59).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("bunny exists")
            .physiology
            .hydration = 0;
        prime_drinking_event(&mut ecosystem, id);

        ecosystem.resolve_drinking(&BTreeSet::from([id]));

        let after = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists")
            .metrics;
        assert!((after.water_consumed - expected_water).abs() < f32::EPSILON);
        assert_eq!(
            ecosystem
                .app
                .world()
                .get::<AgentBody>(entity)
                .expect("bunny exists")
                .physiology
                .hydration,
            ecosystem.config.maximum_hydration
        );
    }

    /// Scarce well water grants only complete drinks and retains the remainder.
    #[test]
    fn finite_well_depletes_and_refills() {
        let config = SimulationConfig::for_stage(CurriculumStage::Competition)
            .expect("competition defaults are valid");
        let requested = config.well_drink_rate;
        let refill = config.well_refill_rate * config.time_step;
        let mut ecosystem = Ecosystem::new(config, 53).expect("world spawns");
        let ids = ecosystem.living_agents();
        let first = ids[0];
        let second = ids[1];
        let entities = ecosystem.agent_entities();
        for id in [first, second] {
            ecosystem
                .app
                .world_mut()
                .get_mut::<AgentBody>(entities[&id])
                .expect("agent exists")
                .physiology
                .hydration = 0;
            prime_drinking_event(&mut ecosystem, id);
        }
        ecosystem.app.world_mut().resource_mut::<WellState>().water = requested * 1.5;

        ecosystem.resolve_drinking(&BTreeSet::from([first, second]));

        let water = ecosystem.app.world().resource::<WellState>().water;
        assert!((water - requested * 0.5).abs() < f32::EPSILON);
        let first_agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entities[&first])
            .expect("first agent exists");
        let second_agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entities[&second])
            .expect("second agent exists");
        assert!((first_agent.metrics.water_consumed - requested).abs() < f32::EPSILON);
        assert_eq!(first_agent.physiology.hydration, 1);
        assert!(second_agent.metrics.water_consumed.abs() < f32::EPSILON);
        assert_eq!(second_agent.physiology.hydration, 0);

        ecosystem.refill_well();
        assert!(
            (ecosystem.app.world().resource::<WellState>().water - requested * 0.5 - refill).abs()
                < f32::EPSILON
        );
    }

    /// A complete priority cycle gives every identity exactly one first choice.
    #[test]
    fn contested_resource_priority_rotates_across_agent_identities() {
        let identities = [AgentId(0), AgentId(1), AgentId(2), AgentId(3)];
        let winners = (0..identities.len())
            .map(|rotation| {
                identities
                    .into_iter()
                    .min_by_key(|id| rotated_priority(*id, 0, identities.len(), rotation))
                    .expect("the population is nonempty")
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(winners, BTreeSet::from(identities));

        let requested = 1.0;
        let mut drinkers = [AgentId(0), AgentId(1)];
        assert_eq!(
            complete_drink_recipients(&mut drinkers, requested, requested, 0),
            vec![AgentId(0)]
        );
        assert_eq!(
            complete_drink_recipients(&mut drinkers, requested, requested, 1),
            vec![AgentId(1)]
        );

        let foxes = [AgentId(6), AgentId(7)];
        let fox_winners = (0..foxes.len())
            .map(|rotation| {
                foxes
                    .into_iter()
                    .min_by_key(|id| rotated_priority(*id, 6, foxes.len(), rotation))
                    .expect("the fox population is nonempty")
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(fox_winners, BTreeSet::from(foxes));
    }

    /// Food appears exactly on the configured periodic simulation step.
    #[test]
    fn food_spawns_periodically_without_exceeding_capacity() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.initial_food = 1;
        config.max_food = 2;
        config.food_spawn_interval = 3;
        let mut ecosystem = Ecosystem::new(config, 57).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0, 0.0).expect("idle action is valid");

        assert_eq!(ecosystem.snapshot().food_count, 1);
        ecosystem.step(&[(id, idle)]).expect("first step advances");
        ecosystem.step(&[(id, idle)]).expect("second step advances");
        assert_eq!(ecosystem.snapshot().food_count, 1);
        ecosystem.step(&[(id, idle)]).expect("spawn step advances");
        assert_eq!(ecosystem.snapshot().food_count, 2);
        for _ in 0..3 {
            ecosystem
                .step(&[(id, idle)])
                .expect("capacity steps advance");
        }
        assert_eq!(ecosystem.snapshot().food_count, 2);
    }

    /// Overlapping dynamic agents are separated and record bilateral contacts.
    #[test]
    fn avian_agents_block_push_and_record_contacts() {
        let config = SimulationConfig::for_stage(CurriculumStage::Competition)
            .expect("competition defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 58).expect("world spawns");
        let entities = ecosystem.agent_entities();
        let first = AgentId(0);
        let second = AgentId(1);
        let well_y = ecosystem.app.world().resource::<WellState>().position.y;
        let collision_y = if well_y > 0.0 { -5.0 } else { 5.0 };
        for (id, position, heading, velocity) in [
            (
                first,
                Vec2::new(-2.0, collision_y),
                0.0,
                Vec2::new(MAX_LINEAR_SPEED, 0.0),
            ),
            (
                second,
                Vec2::new(2.0, collision_y),
                std::f32::consts::PI,
                Vec2::new(-MAX_LINEAR_SPEED, 0.0),
            ),
        ] {
            let mut entity = ecosystem.app.world_mut().entity_mut(entities[&id]);
            *entity.get_mut::<Position>().expect("agent has a position") = Position(position);
            entity
                .get_mut::<Transform>()
                .expect("agent has a transform")
                .translation = position.extend(0.0);
            entity
                .get_mut::<Transform>()
                .expect("agent has a transform")
                .rotation = Quat::from_rotation_z(heading);
            *entity.get_mut::<Rotation>().expect("agent has a rotation") =
                Rotation::radians(heading);
            *entity
                .get_mut::<LinearVelocity>()
                .expect("agent has a velocity") = LinearVelocity(velocity);
        }

        for _ in 0..8 {
            let actions = ecosystem
                .living_agents()
                .into_iter()
                .map(|id| {
                    (
                        id,
                        LocomotionAction::new(1.0, 0.0, 0.0, 0.0).expect("forward action is valid"),
                    )
                })
                .collect::<Vec<_>>();
            ecosystem.step(&actions).expect("contact world advances");
        }
        let world = ecosystem.app.world_mut();
        let first_body = world
            .get::<AgentBody>(entities[&first])
            .expect("first agent exists");
        let second_body = world
            .get::<AgentBody>(entities[&second])
            .expect("second agent exists");
        let first_position = world
            .get::<Position>(entities[&first])
            .expect("first position exists")
            .0;
        let second_position = world
            .get::<Position>(entities[&second])
            .expect("second position exists")
            .0;
        assert!(
            first_body.metrics.collision_contacts > 0,
            "first={first_position:?}, second={second_position:?}"
        );
        assert!(second_body.metrics.collision_contacts > 0);
        assert!(first_body.metrics.contact_displacements > 0);
        assert!(second_body.metrics.contact_displacements > 0);
        assert!(first_position.x <= second_position.x);
    }

    /// Predation terminates only the contacted bunny and feeds the winning fox.
    #[test]
    fn fox_predation_has_species_specific_terminal_effects() {
        let config = SimulationConfig::for_stage(CurriculumStage::PredatorPrey)
            .expect("predator-prey defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 59).expect("world spawns");
        let entities = ecosystem.agent_entities();
        let bunny = AgentId(0);
        let fox = AgentId(6);

        ecosystem.advance_agent_ages();
        ecosystem.resolve_predation(&[(fox, bunny)]);

        let bunny_body = ecosystem
            .app
            .world()
            .get::<AgentBody>(entities[&bunny])
            .expect("bunny exists");
        assert_eq!(bunny_body.life, LifeState::Dead(DeathCause::Predation));
        assert_eq!(bunny_body.physiology.hit_points, 0);
        assert_eq!(bunny_body.age_steps, 1);
        let fox_body = ecosystem
            .app
            .world()
            .get::<AgentBody>(entities[&fox])
            .expect("fox exists");
        assert!(fox_body.is_alive());
        assert_eq!(fox_body.metrics.food_eaten, 1);
        assert_eq!(fox_body.metrics.kills, 1);
        assert!(!fox_body.interaction_active);
        assert_eq!(
            fox_body.interaction_cooldown_steps,
            INTERACTION_COOLDOWN_STEPS
        );

        ecosystem.disable_newly_dead();
        assert!(!ecosystem
            .app
            .world()
            .entity(entities[&bunny])
            .contains::<Collider>());
        let world = ecosystem.app.world_mut();
        let mut hitbox_query = world.query::<(&InteractionHitbox, Has<Collider>)>();
        let (_, bunny_hitbox_enabled) = hitbox_query
            .iter(world)
            .find(|(hitbox, _)| hitbox.owner == bunny)
            .expect("bunny hitbox exists");
        assert!(!bunny_hitbox_enabled);
    }

    /// Side body contact must not kill prey, while forward hitbox contact must.
    #[test]
    fn predation_requires_forward_hitbox_contact() {
        let config = SimulationConfig::for_stage(CurriculumStage::PredatorPrey)
            .expect("predator-prey defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 67).expect("world spawns");
        let entities = ecosystem.agent_entities();
        let bunny = AgentId(0);
        let fox = AgentId(6);
        for (id, entity) in &entities {
            if *id != bunny && *id != fox {
                ecosystem
                    .app
                    .world_mut()
                    .entity_mut(*entity)
                    .remove::<Collider>();
            }
        }
        position_agent(&mut ecosystem, entities[&fox], Vec2::ZERO, 0.0);
        position_agent(&mut ecosystem, entities[&bunny], Vec2::new(0.0, 1.0), 0.0);
        let attack = LocomotionAction::new(-1.0, 0.0, 0.0, 1.0).expect("attack action is valid");
        ecosystem.apply_actions(&BTreeMap::from([(fox, attack)]));
        ecosystem.app.update();

        let side_contacts = ecosystem.collect_contacts(&BTreeMap::new());
        assert!(side_contacts.agent_agents.contains(&fox));
        assert!(side_contacts.agent_agents.contains(&bunny));
        assert!(side_contacts.predation.is_empty());

        position_agent(&mut ecosystem, entities[&fox], Vec2::ZERO, 0.0);
        position_agent(&mut ecosystem, entities[&bunny], Vec2::new(1.3, 0.0), 0.0);
        ecosystem.app.update();
        let forward_contacts = ecosystem.collect_contacts(&BTreeMap::new());
        assert_eq!(forward_contacts.predation, vec![(fox, bunny)]);
    }

    /// Every generated obstacle map keeps a traversable route from each agent to the well.
    #[test]
    fn obstacle_maps_keep_agent_routes_to_the_well() {
        let config = SimulationConfig::for_stage(CurriculumStage::Obstacles)
            .expect("obstacle defaults are valid");
        for seed in 0..16 {
            let mut ecosystem = Ecosystem::new(config.clone(), seed).expect("world spawns");
            let well = ecosystem.app.world().resource::<WellState>().position;
            let blockers = solid_obstacle_clearances(&mut ecosystem);
            for position in agent_positions(&mut ecosystem).into_values() {
                assert!(
                    grid_route_exists(position, well, config.map_half_extent, &blockers),
                    "seed {seed} leaves agent at {position:?} disconnected from well {well:?}"
                );
            }
        }
    }

    /// Avian resolves agent contacts without leaving bodies inside solid obstacles.
    #[test]
    fn solid_obstacles_have_no_unresolved_agent_penetration() {
        let config = SimulationConfig::for_stage(CurriculumStage::Obstacles)
            .expect("obstacle defaults are valid");
        for seed in 16..20 {
            let mut ecosystem = Ecosystem::new(config.clone(), seed).expect("world spawns");
            for step in 0..600 {
                let actions = ecosystem
                    .living_agents()
                    .into_iter()
                    .map(|id| {
                        let turn = match (step + u32::from(id.0)) % 3 {
                            0 => -1.0,
                            1 => 0.0,
                            _ => 1.0,
                        };
                        (
                            id,
                            LocomotionAction::new(1.0, turn, 0.0, 0.0)
                                .expect("scripted action is valid"),
                        )
                    })
                    .collect::<Vec<_>>();
                if actions.is_empty() {
                    break;
                }
                let result = ecosystem.step(&actions).expect("obstacle world advances");
                let penetration = maximum_agent_solid_penetration(&mut ecosystem);
                assert!(
                    penetration <= 0.02,
                    "seed {seed}, step {step} retained {penetration} penetration"
                );
                if result.is_done {
                    break;
                }
            }
            assert!(ecosystem
                .episode_metrics()
                .iter()
                .all(|agent| agent.unresolved_solid_penetrations == 0));
        }
    }

    /// Extract the private initial critic state for deterministic comparison.
    fn initial_state(mut ecosystem: Ecosystem) -> GlobalState {
        ecosystem.global_state()
    }

    /// Count currently live food sensors in a test ecosystem.
    fn food_count(ecosystem: &mut Ecosystem) -> usize {
        let world = ecosystem.app.world_mut();
        let mut query = world.query::<&FoodSlot>();
        query.iter(world).count()
    }

    /// Remove one live food sensor to model one consumption event.
    fn despawn_one_food(ecosystem: &mut Ecosystem) {
        let entity = {
            let world = ecosystem.app.world_mut();
            let mut query = world.query::<(Entity, &FoodSlot)>();
            query.iter(world).next().map(|(entity, _)| entity)
        }
        .expect("food exists");
        assert!(ecosystem.app.world_mut().despawn(entity));
    }

    /// Remove every live food sensor to exercise the one-item floor.
    fn despawn_all_food(ecosystem: &mut Ecosystem) {
        let entities = {
            let world = ecosystem.app.world_mut();
            let mut query = world.query::<(Entity, &FoodSlot)>();
            query
                .iter(world)
                .map(|(entity, _)| entity)
                .collect::<Vec<_>>()
        };
        for entity in entities {
            assert!(ecosystem.app.world_mut().despawn(entity));
        }
    }

    /// Observe one isolated food item at an egocentric bearing and distance.
    fn observe_single_food(relative_angle: f32, distance: f32) -> LocalObservation {
        observe_single_food_with_gaze(relative_angle, distance, 0.0)
    }

    /// Observe one isolated food item with a bounded eye action.
    fn observe_single_food_with_gaze(
        relative_angle: f32,
        distance: f32,
        gaze: f32,
    ) -> LocalObservation {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 17).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let agent_entity = ecosystem.agent_entities()[&id];
        let position = ecosystem
            .app
            .world()
            .get::<Position>(agent_entity)
            .expect("agent position exists")
            .0;
        let inward_angle = (-position).to_angle();
        *ecosystem
            .app
            .world_mut()
            .get_mut::<Rotation>(agent_entity)
            .expect("agent rotation exists") = Rotation::radians(inward_angle);
        ecosystem
            .app
            .world_mut()
            .get_mut::<Transform>(agent_entity)
            .expect("agent transform exists")
            .rotation = Quat::from_rotation_z(inward_angle);
        let semantic_entities = {
            let world = ecosystem.app.world_mut();
            let mut query = world.query_filtered::<Entity, With<SemanticCollider>>();
            query
                .iter(world)
                .filter(|entity| *entity != agent_entity)
                .collect::<Vec<_>>()
        };
        for entity in semantic_entities {
            assert!(ecosystem.app.world_mut().despawn(entity));
        }
        let food_angle = inward_angle + relative_angle;
        let food_position = position + Vec2::from_angle(food_angle) * distance;
        ecosystem.app.world_mut().spawn((
            RigidBody::Static,
            Position(food_position),
            Transform::from_translation(food_position.extend(0.0)),
            Collider::circle(FOOD_RADIUS),
            Sensor,
            sensor_layers(),
            SemanticCollider(PerceptKind::Food),
            FoodSlot(0),
        ));
        let action = LocomotionAction::new(-1.0, 0.0, gaze, 0.0).expect("gaze action is valid");
        ecosystem
            .step(&[(id, action)])
            .expect("gaze step succeeds")
            .agents[0]
            .observation
    }

    /// Return whether one local observation contains the requested semantic hit.
    fn observation_contains(observation: &[f32], kind: PerceptKind) -> bool {
        (0..RAY_COUNT).any(|ray_index| {
            let start = PROPRIOCEPTION_SIZE + ray_index * RAY_FEATURE_SIZE;
            observation[start + RAY_KIND_START + kind.index()] > 0.5
        })
    }

    /// Return the nearest normalized distance for one observed semantic class.
    fn nearest_observed_distance(observation: &[f32], kind: PerceptKind) -> f32 {
        (0..RAY_COUNT)
            .filter_map(|ray_index| {
                let start = PROPRIOCEPTION_SIZE + ray_index * RAY_FEATURE_SIZE;
                (observation[start + RAY_KIND_START + kind.index()] > 0.5)
                    .then_some(observation[start])
            })
            .min_by(f32::total_cmp)
            .unwrap_or(1.0)
    }

    /// Return configured ray bearings containing one semantic class.
    fn observation_hit_angles(observation: &[f32], kind: PerceptKind) -> Vec<f32> {
        // Decode only active semantic hits while retaining their configured bearings.
        (0..RAY_COUNT)
            .filter(|ray_index| {
                let start = PROPRIOCEPTION_SIZE + ray_index * RAY_FEATURE_SIZE;
                observation[start + RAY_KIND_START + kind.index()] > 0.5
            })
            .filter_map(|ray_index| RAY_ANGLES.get(ray_index).copied())
            .collect()
    }

    /// Return the signed angle from one heading to a target offset.
    fn relative_bearing(forward: Vec2, offset: Vec2) -> f32 {
        let direction = offset.normalize();
        forward.perp_dot(direction).atan2(forward.dot(direction))
    }

    /// Return every live food position in query order.
    fn food_positions(ecosystem: &mut Ecosystem) -> Vec<Vec2> {
        let world = ecosystem.app.world_mut();
        let mut query = world.query_filtered::<&Position, With<FoodSlot>>();
        query.iter(world).map(|position| position.0).collect()
    }

    /// Return whether one resource class starts in contact with an agent.
    fn resource_contacts_agent<T: Component>(ecosystem: &mut Ecosystem, agent: Entity) -> bool {
        let world = ecosystem.app.world_mut();
        let mut query = world.query_filtered::<&CollidingEntities, With<T>>();
        query
            .iter(world)
            .any(|contacts| contacts.0.contains(&agent))
    }

    /// Place one test agent with a matching physics and rendering transform.
    fn position_agent(ecosystem: &mut Ecosystem, entity: Entity, position: Vec2, heading: f32) {
        ecosystem.app.world_mut().entity_mut(entity).insert((
            Position(position),
            Rotation::radians(heading),
            Transform {
                translation: position.extend(0.0),
                rotation: Quat::from_rotation_z(heading),
                ..default()
            },
            LinearVelocity::ZERO,
            AngularVelocity(0.0),
        ));
    }

    /// Assign isolated motion components for physiology-cost assertions.
    fn set_agent_motion(ecosystem: &mut Ecosystem, id: AgentId, linear: Vec2, angular: f32) {
        let entity = ecosystem.agent_entities()[&id];
        ecosystem
            .app
            .world_mut()
            .entity_mut(entity)
            .insert((LinearVelocity(linear), AngularVelocity(angular)));
    }

    /// Advance one drink clock to the step before a completed event.
    fn prime_drinking_event(ecosystem: &mut Ecosystem, id: AgentId) {
        let interval = DRINKING_INTERVAL_STEPS;
        let entity = ecosystem.agent_entities()[&id];
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("agent exists")
            .physiology
            .drinking_steps = interval - 1;
    }

    /// Read one agent's food and water reserves.
    fn agent_needs(ecosystem: &mut Ecosystem, id: AgentId) -> (u8, u8) {
        let entity = ecosystem.agent_entities()[&id];
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        (agent.physiology.satiation, agent.physiology.hydration)
    }

    /// Return stable agent positions for spawn-slot assertions.
    fn agent_positions(ecosystem: &mut Ecosystem) -> BTreeMap<AgentId, Vec2> {
        let world = ecosystem.app.world_mut();
        let mut query = world.query::<(&AgentBody, &Position)>();
        query
            .iter(world)
            .map(|(agent, position)| (agent.id, position.0))
            .collect()
    }

    /// Collect conservative radial clearances around solid square obstacles.
    fn solid_obstacle_clearances(ecosystem: &mut Ecosystem) -> Vec<(Vec2, f32)> {
        let world = ecosystem.app.world_mut();
        let mut query = world.query::<(&Position, &ObstacleKind, &SpawnBlocker)>();
        query
            .iter(world)
            .filter(|(_, kind, _)| **kind != ObstacleKind::Thorn)
            .map(|(position, _, radius)| (position.0, radius.0 + AGENT_RADIUS + 0.1))
            .collect()
    }

    /// Search a one-meter occupancy grid for a route into the well sensor annulus.
    fn grid_route_exists(start: Vec2, well: Vec2, extent: f32, blockers: &[(Vec2, f32)]) -> bool {
        let limit = extent.floor() as i32 - 1;
        let start_cell = (start.x.round() as i32, start.y.round() as i32);
        let mut frontier = VecDeque::from([start_cell]);
        let mut visited = BTreeSet::from([start_cell]);

        // The goal is the drinkable ring outside the well's solid base.
        while let Some((x, y)) = frontier.pop_front() {
            let position = Vec2::new(x as f32, y as f32);
            let well_distance = position.distance(well);
            if (WELL_RADIUS + AGENT_RADIUS..=WELL_SENSOR_RADIUS + AGENT_RADIUS)
                .contains(&well_distance)
            {
                return true;
            }
            for neighbor in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                if neighbor.0.abs() > limit || neighbor.1.abs() > limit {
                    continue;
                }
                let neighbor_position = Vec2::new(neighbor.0 as f32, neighbor.1 as f32);
                let blocked = blockers.iter().any(|(center, radius)| {
                    neighbor_position.distance_squared(*center) <= radius.powi(2)
                });
                if !blocked && visited.insert(neighbor) {
                    frontier.push_back(neighbor);
                }
            }
        }
        false
    }

    /// Return the largest positive post-solver agent/tree-or-rock penetration.
    fn maximum_agent_solid_penetration(ecosystem: &mut Ecosystem) -> f32 {
        let world = ecosystem.app.world_mut();
        let agent_entities = {
            let mut query = world.query_filtered::<Entity, With<AgentBody>>();
            query.iter(world).collect::<BTreeSet<_>>()
        };
        let solid_entities = {
            let mut query = world.query::<(Entity, &ObstacleKind)>();
            query
                .iter(world)
                .filter(|(_, kind)| **kind != ObstacleKind::Thorn)
                .map(|(entity, _)| entity)
                .collect::<BTreeSet<_>>()
        };
        let graph = world.resource::<ContactGraph>();
        graph
            .iter_active()
            .chain(graph.iter_sleeping())
            .filter(|pair| {
                (agent_entities.contains(&pair.collider1)
                    && solid_entities.contains(&pair.collider2))
                    || (agent_entities.contains(&pair.collider2)
                        && solid_entities.contains(&pair.collider1))
            })
            .flat_map(|pair| pair.manifolds.iter())
            .flat_map(|manifold| manifold.points.iter())
            .map(|point| point.penetration.max(0.0))
            .fold(0.0, f32::max)
    }
}
