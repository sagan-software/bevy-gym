//! Deterministic `Avian2D` ecosystem mechanics shared by every lesson.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::time::Duration;

use avian2d::prelude::*;
use bevy::ecs::schedule::ScheduleLabel;
use bevy::prelude::*;
use bevy::time::TimeUpdateStrategy;
use bevy_gym::EpisodeStatus;

#[cfg(test)]
use super::domain::ExperimentTuning;
use super::domain::{
    AgentEpisodeMetrics, AgentId, AgentStep, CurriculumStage, DeathCause, EcosystemSnapshot,
    EcosystemState, GlobalState, JointStep, LocalObservation, LocomotionAction, PerceptKind,
    PerceptionRayCount, SimulationConfig, Species, GLOBAL_AGENT_FEATURES, GLOBAL_FOOD_FEATURES,
    GLOBAL_OBSTACLE_FEATURES, LOCAL_OBSERVATION_SIZE, MAX_AGENTS, MAX_FOOD, MAX_OBSTACLES,
    PROPRIOCEPTION_SIZE, RAY_ANGLES, RAY_COUNT, RAY_KIND_COUNT,
};
#[cfg(feature = "render")]
use super::domain::{VisualAgent, VisualObject, VisualObjectKind, VisualRay, VisualWorldSnapshot};
use super::rng::SplitMix64;

/// Radius of every bunny and fox collision body.
const AGENT_RADIUS: f32 = 0.75;

/// Radius of a food interaction sensor.
const FOOD_RADIUS: f32 = 0.55;

/// Solid radius of the well base.
const WELL_RADIUS: f32 = 1.25;

/// Radius in which an agent automatically drinks.
const WELL_SENSOR_RADIUS: f32 = 2.25;

/// Maximum distance represented by every semantic sector.
const SIGHT_RANGE: f32 = 18.0;

/// Half-width of one fixed-capacity semantic sector.
const RAY_HALF_WIDTH: f32 = 5.0_f32.to_radians();

/// Maximum linear speed used for physics and observation normalization.
const MAX_LINEAR_SPEED: f32 = 5.0;

/// Maximum angular speed in radians per second.
const MAX_ANGULAR_SPEED: f32 = 3.0;

/// Local forward acceleration at a unit action.
const FORWARD_ACCELERATION: f32 = 16.0;

/// Hunger reserve lost per simulated second.
const HUNGER_DRAIN_RATE: f32 = 0.025;

/// Thirst reserve lost per simulated second.
const THIRST_DRAIN_RATE: f32 = 0.035;

/// Need level below which health and hit-point damage begins.
const NEED_DAMAGE_THRESHOLD: f32 = 0.25;

/// Health fraction lost per fully depleted need-second.
const HEALTH_DAMAGE_RATE: f32 = 0.12;

/// Hit points lost per fully depleted need-second.
const HIT_POINT_DAMAGE_RATE: f32 = 3.0;

/// Health fraction restored per nourished second.
const HEALTH_RECOVERY_RATE: f32 = 0.02;

/// Hit points restored when a bunny eats food.
const FOOD_HIT_POINT_RECOVERY: f32 = 4.0;

/// Hunger reserve restored by one food item or predation event.
const FOOD_HUNGER_RECOVERY: f32 = 0.7;

/// Thirst reserve restored per full-rate drinking second.
const DRINK_THIRST_RECOVERY: f32 = 0.8;

/// Hit points lost per second inside at least one thorn sensor.
const THORN_HIT_POINT_DAMAGE_RATE: f32 = 8.0;

/// Health fraction lost per second inside thorn sensors.
const THORN_HEALTH_DAMAGE_RATE: f32 = 0.08;

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

/// Hunger, thirst, health, and hit-point state.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Physiology {
    /// Normalized food reserve.
    hunger: f32,

    /// Normalized water reserve.
    thirst: f32,

    /// Normalized general health condition.
    health: f32,

    /// Hit points in `0..=100`.
    hit_points: f32,
}

impl Physiology {
    /// Construct reset physiology from the configured starting fraction.
    fn new(initial_health_fraction: f32) -> Self {
        Self {
            hunger: 1.0,
            thirst: 1.0,
            health: initial_health_fraction,
            hit_points: initial_health_fraction * 100.0,
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
    /// Compatible food or prey consumed.
    food_eaten: u32,

    /// Well-water units consumed.
    water_consumed: f32,

    /// Bunnies consumed by a fox.
    kills: u32,

    /// Hit points lost to thorn contact.
    thorn_damage: f32,

    /// Agent-agent overlap contacts observed across physics steps.
    collision_contacts: u32,
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

    /// Seconds elapsed since episode spawn.
    age_seconds: f32,

    /// Post-physics local ray samples.
    rays: [RaySample; RAY_COUNT],

    /// Behavior diagnostics excluded from reward.
    metrics: AgentMetrics,
}

impl AgentBody {
    /// Construct one clean episode agent.
    fn new(id: AgentId, species: Species, initial_health_fraction: f32) -> Self {
        Self {
            id,
            species,
            life: LifeState::Alive,
            physiology: Physiology::new(initial_health_fraction),
            age_seconds: 0.0,
            rays: [RaySample::MISS; RAY_COUNT],
            metrics: AgentMetrics::default(),
        }
    }

    /// Return whether the agent can still act and earn survival reward.
    const fn is_alive(&self) -> bool {
        matches!(self.life, LifeState::Alive)
    }
}

/// Semantic class attached to every queryable collider.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct SemanticCollider(PerceptKind);

/// Approximate clearance radius used only during bounded spawn rejection.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
struct SpawnBlocker(f32);

/// Stable padded critic slot for a food entity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct FoodSlot(u16);

/// Stable padded critic slot for an obstacle or thorn.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
struct ObstacleSlot(u16);

/// Obstacle semantic needed by the centralized critic.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
enum ObstacleKind {
    /// Circular solid tree.
    Tree,

    /// Rectangular solid rock.
    Rock,

    /// Traversable damaging sensor.
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

    /// Next padded food slot assigned within this episode.
    next_food_slot: u16,

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
            .field("next_food_slot", &self.next_food_slot)
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
        let mut ecosystem = Self {
            app,
            config,
            rng: SplitMix64::new(seed),
            step: 0,
            next_food_slot: 0,
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
        if living_before.is_empty() || self.step >= self.config.max_steps {
            return Err(SimulationError::EpisodeFinished);
        }
        let metrics_before = self.agent_metrics_by_id();

        self.step = self.step.saturating_add(1);
        self.refill_well();

        // All controls are written before Avian advances, preserving joint
        // action semantics independent of entity query order.
        self.apply_actions(&actions);
        self.app.update();

        let contacts = self.collect_contacts();
        self.resolve_food(&contacts.food_winners);
        self.replenish_food()?;
        self.resolve_drinking(&contacts.well_agents);
        self.resolve_predation(&contacts.predation);
        self.resolve_physiology(&contacts.thorn_agents);
        self.disable_newly_dead();

        self.app.world_mut().run_schedule(PerceptionSchedule);

        let horizon = self.step >= self.config.max_steps;
        let metrics_after = self.agent_metrics_by_id();
        let mut results = Vec::with_capacity(living_before.len());
        for id in living_before {
            if let Some((species, observation, life)) = self.agent_output(id) {
                let (status, death_cause) = match life {
                    LifeState::Alive if horizon => (EpisodeStatus::Truncated, None),
                    LifeState::Alive => (EpisodeStatus::Continuing, None),
                    LifeState::Dead(cause) => (EpisodeStatus::Terminated, Some(cause)),
                };
                results.push(AgentStep {
                    id,
                    species,
                    observation,
                    reward: transition_reward(
                        &self.config,
                        metrics_before.get(&id).copied().unwrap_or_default(),
                        metrics_after.get(&id).copied().unwrap_or_default(),
                    ),
                    status,
                    death_cause,
                });
            }
        }
        results.sort_by_key(|result| result.id);

        let is_done = horizon || self.living_agents().is_empty();
        Ok(JointStep {
            agents: results,
            global_state: self.global_state(),
            is_done,
        })
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
    #[cfg(feature = "render")]
    pub(super) fn visual_snapshot(&mut self) -> VisualWorldSnapshot {
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
                is_alive: agent.is_alive(),
                hunger: agent.physiology.hunger,
                thirst: agent.physiology.thirst,
                hit_points: agent.physiology.hit_points,
            });
            if agent.is_alive() {
                // Reconstruct endpoints from the exact stored sector samples
                // used by the actor's current local observation.
                let forward = Vec2::from_angle(heading);
                rays.extend(
                    active_ray_indices(perception_ray_count).filter_map(|ray_index| {
                        let sample = agent.rays.get(ray_index)?;
                        let angle = RAY_ANGLES.get(ray_index)?;
                        let direction = Vec2::from_angle(*angle).rotate(forward);
                        let distance = sample
                            .kind
                            .map_or(SIGHT_RANGE, |_| sample.normalized_distance * SIGHT_RANGE);
                        Some(VisualRay {
                            agent: agent.id,
                            start: position.to_array(),
                            end: (direction * distance + position).to_array(),
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
            agents,
            objects,
            rays,
        }
    }

    /// Return final behavior evidence for every possible agent slot.
    pub(super) fn episode_metrics(&mut self) -> Vec<AgentEpisodeMetrics> {
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        let mut metrics = query
            .iter(world)
            .map(|agent| AgentEpisodeMetrics {
                id: agent.id,
                species: agent.species,
                lifetime_seconds: agent.age_seconds,
                food_eaten: agent.metrics.food_eaten,
                water_consumed: agent.metrics.water_consumed,
                kills: agent.metrics.kills,
                thorn_damage: agent.metrics.thorn_damage,
                collision_contacts: agent.metrics.collision_contacts,
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

        let well_position = self.open_position(WELL_SENSOR_RADIUS + 0.5)?;
        self.spawn_well(well_position);

        for slot in 0..self.config.solid_obstacles {
            let radius = if slot.is_multiple_of(2) { 1.4 } else { 1.1 };
            let position = self.open_position(radius + 0.5)?;
            self.spawn_solid_obstacle(slot, position, radius);
        }
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

    /// Spawn the well's solid base and non-blocking drinking radius.
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
            solid_layers(),
            SemanticCollider(PerceptKind::Well),
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
            SemanticCollider(PerceptKind::Well),
            WellSensor,
        ));
    }

    /// Spawn one dynamic mutually pushable agent body.
    fn spawn_agent(&mut self, id: AgentId, species: Species, position: Vec2) {
        let facing = self
            .rng
            .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let mut entity = self.app.world_mut().spawn((
            AgentBody::new(id, species, self.config.initial_health_fraction),
            RigidBody::Dynamic,
            Position(position),
            Rotation::radians(facing),
            Transform::from_translation(position.extend(0.0)),
            Collider::circle(AGENT_RADIUS),
            agent_layers(),
            LinearVelocity::ZERO,
            AngularVelocity(0.0),
            ConstantLocalLinearAcceleration::default(),
            LinearDamping(2.0),
            AngularDamping(4.0),
        ));
        entity.insert((
            MaxLinearSpeed(MAX_LINEAR_SPEED),
            MaxAngularSpeed(MAX_ANGULAR_SPEED),
            SleepingDisabled,
            CollidingEntities::default(),
            SemanticCollider(match species {
                Species::Bunny => PerceptKind::Bunny,
                Species::Fox => PerceptKind::Fox,
            }),
            SpawnBlocker(AGENT_RADIUS),
        ));
    }

    /// Spawn one tree or rock with a stable centralized-state slot.
    fn spawn_solid_obstacle(&mut self, slot: usize, position: Vec2, radius: f32) {
        let kind = if slot.is_multiple_of(2) {
            ObstacleKind::Tree
        } else {
            ObstacleKind::Rock
        };
        let collider = match kind {
            ObstacleKind::Rock => Collider::rectangle(radius * 1.8, radius * 1.4),
            ObstacleKind::Tree | ObstacleKind::Thorn => Collider::circle(radius),
        };
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
        ));
    }

    /// Spawn one sensor food item when the padded capacity permits it.
    fn spawn_one_food(&mut self) -> Result<(), SimulationError> {
        let used_slots = {
            let world = self.app.world_mut();
            let mut query = world.query::<&FoodSlot>();
            query
                .iter(world)
                .map(|slot| slot.0)
                .collect::<BTreeSet<_>>()
        };
        if used_slots.len() >= self.config.max_food {
            return Ok(());
        }

        // Reuse only a vacant critic slot so two live food entities can never
        // overwrite each other in the padded global state.
        let capacity = u16::try_from(self.config.max_food).unwrap_or(u16::MAX);
        let slot = (0..capacity)
            .map(|offset| self.next_food_slot.wrapping_add(offset) % capacity)
            .find(|candidate| !used_slots.contains(candidate));
        let Some(slot) = slot else {
            debug_assert!(false, "a validated food capacity must have a vacant slot");
            return Ok(());
        };
        let position = self.open_position(FOOD_RADIUS + 0.25)?;
        self.next_food_slot = slot.wrapping_add(1) % capacity;
        self.app.world_mut().spawn((
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
        ));
        Ok(())
    }

    /// Preserve one available item and periodically replenish the second slot.
    fn replenish_food(&mut self) -> Result<(), SimulationError> {
        let live_food = {
            let world = self.app.world_mut();
            let mut query = world.query::<&FoodSlot>();
            query.iter(world).count()
        };
        let is_due = self.step.is_multiple_of(self.config.food_spawn_interval);
        if live_food == 0 || (live_food < self.config.max_food && is_due) {
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
            &AgentBody,
            &mut ConstantLocalLinearAcceleration,
            &mut AngularVelocity,
        )>();
        for (agent, mut acceleration, mut angular_velocity) in query.iter_mut(world) {
            if !agent.is_alive() {
                continue;
            }
            if let Some(action) = actions.get(&agent.id) {
                acceleration.0 = Vec2::X * action.forward * FORWARD_ACCELERATION;
                angular_velocity.0 = action.turn * MAX_ANGULAR_SPEED;
            }
        }
    }

    /// Collect current Avian overlaps before mutating any interaction state.
    fn collect_contacts(&mut self) -> ContactResolution {
        let bunny_count = self.config.bunny_count;
        let fox_count = self.config.fox_count;
        let bunny_priority_rotation = self.step as usize % bunny_count;
        let fox_priority_rotation = if fox_count == 0 {
            0
        } else {
            self.step as usize % fox_count
        };
        let world = self.app.world_mut();
        let agents = {
            let mut query = world.query::<(Entity, &AgentBody)>();
            query
                .iter(world)
                .filter(|(_, agent)| agent.is_alive())
                .map(|(entity, agent)| (entity, agent.id, agent.species))
                .collect::<Vec<_>>()
        };
        let by_entity: BTreeMap<_, _> = agents
            .iter()
            .map(|(entity, id, species)| (*entity, (*id, *species)))
            .collect();

        let mut food_winners = Vec::new();
        {
            let mut query = world.query::<(Entity, &FoodSlot, &CollidingEntities)>();
            for (food, _, collisions) in query.iter(world) {
                let winner = collisions
                    .0
                    .iter()
                    .filter_map(|entity| by_entity.get(entity))
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
                .filter_map(|entity| by_entity.get(entity).map(|(id, _)| *id))
                .collect::<BTreeSet<_>>()
        };

        let thorn_agents = {
            let mut query =
                world.query_filtered::<(&ObstacleKind, &CollidingEntities), With<ObstacleSlot>>();
            query
                .iter(world)
                .filter(|(kind, _)| **kind == ObstacleKind::Thorn)
                .flat_map(|(_, collisions)| collisions.0.iter())
                .filter_map(|entity| by_entity.get(entity).map(|(id, _)| *id))
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
                    let first = by_entity.get(&pair.collider1)?;
                    let second = by_entity.get(&pair.collider2)?;
                    Some((pair.collider1, *first, pair.collider2, *second))
                })
                .collect::<Vec<_>>()
        };
        let mut predation_by_bunny: BTreeMap<AgentId, AgentId> = BTreeMap::new();
        for (first_entity, (first_id, first_species), second_entity, (second_id, second_species)) in
            agent_contacts
        {
            for entity in [first_entity, second_entity] {
                if let Some(mut agent) = world.get_mut::<AgentBody>(entity) {
                    agent.metrics.collision_contacts =
                        agent.metrics.collision_contacts.saturating_add(1);
                }
            }
            let predator_and_prey = match (first_species, second_species) {
                (Species::Fox, Species::Bunny) => Some((first_id, second_id)),
                (Species::Bunny, Species::Fox) => Some((second_id, first_id)),
                (Species::Bunny, Species::Bunny) | (Species::Fox, Species::Fox) => None,
            };
            if let Some((fox_id, bunny_id)) = predator_and_prey {
                predation_by_bunny
                    .entry(bunny_id)
                    .and_modify(|winner| {
                        let winner_rank = rotated_priority(
                            *winner,
                            bunny_count,
                            fox_count,
                            fox_priority_rotation,
                        );
                        let challenger_rank =
                            rotated_priority(fox_id, bunny_count, fox_count, fox_priority_rotation);
                        if challenger_rank < winner_rank {
                            *winner = fox_id;
                        }
                    })
                    .or_insert(fox_id);
            }
        }

        ContactResolution {
            food_winners,
            well_agents,
            thorn_agents,
            predation: predation_by_bunny
                .into_iter()
                .map(|(bunny, fox)| (fox, bunny))
                .collect(),
        }
    }

    /// Consume each contested food entity exactly once.
    fn resolve_food(&mut self, winners: &[(Entity, AgentId)]) {
        let entity_by_id = self.agent_entities();
        for (food, winner) in winners {
            if let Some(agent_entity) = entity_by_id.get(winner).copied() {
                if let Some(mut agent) = self.app.world_mut().get_mut::<AgentBody>(agent_entity) {
                    if agent.is_alive() && agent.species == Species::Bunny {
                        agent.physiology.hunger =
                            (agent.physiology.hunger + FOOD_HUNGER_RECOVERY).min(1.0);
                        agent.physiology.hit_points =
                            (agent.physiology.hit_points + FOOD_HIT_POINT_RECOVERY).min(100.0);
                        agent.metrics.food_eaten = agent.metrics.food_eaten.saturating_add(1);
                    }
                }
            }
            let _despawned = self.app.world_mut().despawn(*food);
        }
    }

    /// Divide scarce well water fairly without withdrawing more than agents absorb.
    fn resolve_drinking(&mut self, drinkers: &BTreeSet<AgentId>) {
        if drinkers.is_empty() {
            return;
        }
        let entity_by_id = self.agent_entities();
        let requested = self.config.well_drink_rate * self.config.time_step;
        let mut demands = drinkers
            .iter()
            .filter_map(|id| {
                let entity = entity_by_id.get(id).copied()?;
                let agent = self.app.world().get::<AgentBody>(entity)?;
                if !agent.is_alive() {
                    return None;
                }
                let absorbable = (1.0 - agent.physiology.thirst).max(0.0)
                    * self.config.well_drink_rate
                    / DRINK_THIRST_RECOVERY;
                Some((*id, requested.min(absorbable)))
            })
            .collect::<Vec<_>>();
        let available = self.app.world().resource::<WellState>().water;
        let allocations = fair_water_allocations(&mut demands, available);
        let withdrawn = allocations.iter().map(|(_, amount)| amount).sum::<f32>();
        self.app.world_mut().resource_mut::<WellState>().water = (available - withdrawn).max(0.0);

        // Apply only the conserved allocation so metrics equal useful withdrawal.
        for (id, allocation) in allocations {
            if allocation <= 0.0 {
                continue;
            }
            if let Some(entity) = entity_by_id.get(&id).copied() {
                if let Some(mut agent) = self.app.world_mut().get_mut::<AgentBody>(entity) {
                    let recovery = allocation / self.config.well_drink_rate * DRINK_THIRST_RECOVERY;
                    agent.physiology.thirst = (agent.physiology.thirst + recovery).min(1.0);
                    agent.metrics.water_consumed += allocation;
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
                bunny.physiology.health = 0.0;
                bunny.physiology.hit_points = 0.0;
            }
            if let Some(mut fox) = self.app.world_mut().get_mut::<AgentBody>(fox_entity) {
                if fox.is_alive() {
                    fox.physiology.hunger = (fox.physiology.hunger + FOOD_HUNGER_RECOVERY).min(1.0);
                    fox.physiology.hit_points =
                        (fox.physiology.hit_points + FOOD_HIT_POINT_RECOVERY).min(100.0);
                    fox.metrics.food_eaten = fox.metrics.food_eaten.saturating_add(1);
                    fox.metrics.kills = fox.metrics.kills.saturating_add(1);
                }
            }
        }
    }

    /// Advance needs and apply starvation, dehydration, and thorn damage.
    fn resolve_physiology(&mut self, thorn_agents: &BTreeSet<AgentId>) {
        let time_step = self.config.time_step;
        let need_time_step = time_step * self.config.need_drain_multiplier;
        let damage_time_step = time_step * self.config.damage_multiplier;
        let extent = self.config.map_half_extent + 2.0;
        let world = self.app.world_mut();
        let mut query = world.query::<(&mut AgentBody, &Position)>();
        for (mut agent, position) in query.iter_mut(world) {
            if !agent.is_alive() {
                continue;
            }
            agent.age_seconds += time_step;
            agent.physiology.hunger = HUNGER_DRAIN_RATE
                .mul_add(-need_time_step, agent.physiology.hunger)
                .max(0.0);
            agent.physiology.thirst = THIRST_DRAIN_RATE
                .mul_add(-need_time_step, agent.physiology.thirst)
                .max(0.0);

            let starvation = need_deficit(agent.physiology.hunger);
            let dehydration = need_deficit(agent.physiology.thirst);
            let deprivation = starvation + dehydration;
            if deprivation > 0.0 {
                agent.physiology.health = (deprivation * HEALTH_DAMAGE_RATE)
                    .mul_add(-damage_time_step, agent.physiology.health)
                    .max(0.0);
                agent.physiology.hit_points = (deprivation * HIT_POINT_DAMAGE_RATE)
                    .mul_add(-damage_time_step, agent.physiology.hit_points)
                    .max(0.0);
            } else {
                agent.physiology.health = HEALTH_RECOVERY_RATE
                    .mul_add(time_step, agent.physiology.health)
                    .min(1.0);
            }

            if thorn_agents.contains(&agent.id) {
                let hit_point_damage = THORN_HIT_POINT_DAMAGE_RATE * damage_time_step;
                agent.physiology.health = THORN_HEALTH_DAMAGE_RATE
                    .mul_add(-damage_time_step, agent.physiology.health)
                    .max(0.0);
                agent.physiology.hit_points =
                    (agent.physiology.hit_points - hit_point_damage).max(0.0);
                agent.metrics.thorn_damage += hit_point_damage;
            }

            if !position.0.is_finite() || position.x.abs() > extent || position.y.abs() > extent {
                agent.life = LifeState::Dead(DeathCause::InvalidPhysics);
                agent.physiology.health = 0.0;
                agent.physiology.hit_points = 0.0;
            } else if agent.physiology.hit_points <= 0.0 {
                let cause = if thorn_agents.contains(&agent.id) {
                    DeathCause::Thorns
                } else if starvation > 0.0 && dehydration > 0.0 {
                    DeathCause::Deprivation
                } else if starvation > 0.0 {
                    DeathCause::Starvation
                } else {
                    DeathCause::Dehydration
                };
                agent.life = LifeState::Dead(cause);
            }
        }
    }

    /// Remove physics and perception capabilities from newly dead bodies.
    fn disable_newly_dead(&mut self) {
        let dead_entities = {
            let world = self.app.world_mut();
            let mut query = world.query::<(Entity, &AgentBody, Has<Collider>)>();
            query
                .iter(world)
                .filter(|(_, agent, has_collider)| !agent.is_alive() && *has_collider)
                .map(|(entity, _, _)| entity)
                .collect::<Vec<_>>()
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
            }
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

    /// Snapshot reward event counters before or after one joint transition.
    fn agent_metrics_by_id(&mut self) -> BTreeMap<AgentId, AgentMetrics> {
        // Stable IDs align pre-contact and post-contact counters without
        // relying on Bevy query iteration order.
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        query
            .iter(world)
            .map(|agent| (agent.id, agent.metrics))
            .collect()
    }

    /// Return one agent's local output after interaction resolution.
    fn agent_output(&mut self, id: AgentId) -> Option<(Species, LocalObservation, LifeState)> {
        let max_age = self.config.max_steps as f32 * self.config.time_step;
        let stage = self.config.stage;
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
                    max_age,
                    stage,
                );
                (agent.species, observation, agent.life)
            })
    }

    /// Encode the fixed padded centralized critic state.
    fn global_state(&mut self) -> GlobalState {
        let mut state = [0.0; super::domain::GLOBAL_STATE_SIZE];
        let extent = self.config.map_half_extent;
        let world = self.app.world_mut();

        let mut agent_query = world.query::<(&AgentBody, &Position, Option<&LinearVelocity>)>();
        for (agent, position, velocity) in agent_query.iter(world) {
            let slot = usize::from(agent.id.0);
            if slot >= MAX_AGENTS {
                continue;
            }
            let start = slot * GLOBAL_AGENT_FEATURES;
            let velocity = velocity.copied().unwrap_or_default().0;
            write_feature(&mut state, start + agent.species.index(), 1.0);
            write_feature(&mut state, start + 2, f32::from(agent.is_alive()));
            write_feature(
                &mut state,
                start + 3,
                (position.x / extent).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 4,
                (position.y / extent).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 5,
                (velocity.x / MAX_LINEAR_SPEED).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 6,
                (velocity.y / MAX_LINEAR_SPEED).clamp(-1.0, 1.0),
            );
            write_feature(&mut state, start + 7, agent.physiology.hunger);
            write_feature(&mut state, start + 8, agent.physiology.thirst);
            write_feature(&mut state, start + 9, agent.physiology.health);
            write_feature(&mut state, start + 10, agent.physiology.hit_points / 100.0);
        }

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

        let food_start = well_start + 4;
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
        write_feature(
            &mut state,
            time_start,
            1.0 - self.step as f32 / self.config.max_steps as f32,
        );
        write_feature(&mut state, time_start + 1 + self.config.stage.index(), 1.0);
        state
    }
}

/// Current overlap sets resolved in stable order after one physics step.
#[derive(Debug, Default)]
struct ContactResolution {
    /// One winner for every contacted food entity.
    food_winners: Vec<(Entity, AgentId)>,

    /// Agents currently inside the well sensor.
    well_agents: BTreeSet<AgentId>,

    /// Agents currently inside at least one thorn sensor.
    thorn_agents: BTreeSet<AgentId>,

    /// Stable `(fox, bunny)` predation winners.
    predation: Vec<(AgentId, AgentId)>,
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

/// Compute one configured transition reward from conserved event deltas.
fn transition_reward(config: &SimulationConfig, before: AgentMetrics, after: AgentMetrics) -> f64 {
    // Saturating counter differences prevent reset or diagnostic mutations
    // from manufacturing negative reward events.
    let food_events = after.food_eaten.saturating_sub(before.food_eaten) as f32;
    let water_consumed = (after.water_consumed - before.water_consumed).max(0.0);
    let survival_and_food = config
        .survival_reward_per_second
        .mul_add(config.time_step, config.food_reward * food_events);
    let reward = config
        .water_reward_per_unit
        .mul_add(water_consumed, survival_and_food);
    f64::from(reward)
}

/// Return normalized need deficit below the damage threshold.
fn need_deficit(value: f32) -> f32 {
    ((NEED_DAMAGE_THRESHOLD - value) / NEED_DAMAGE_THRESHOLD).clamp(0.0, 1.0)
}

/// Max-min fair allocations capped by each drinker's absorbable demand.
fn fair_water_allocations(demands: &mut [(AgentId, f32)], available: f32) -> Vec<(AgentId, f32)> {
    demands.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut remaining = available.max(0.0);
    let count = demands.len();
    demands
        .iter()
        .enumerate()
        .map(|(index, (id, demand))| {
            let equal_share = remaining / (count - index) as f32;
            let allocation = demand.max(0.0).min(equal_share);
            remaining -= allocation;
            (*id, allocation)
        })
        .collect()
}

/// Encode one decentralized actor observation without global coordinates.
fn encode_local_observation(
    agent: &AgentBody,
    _position: Vec2,
    rotation: Rotation,
    velocity: LinearVelocity,
    angular_velocity: AngularVelocity,
    max_age: f32,
    stage: CurriculumStage,
) -> LocalObservation {
    let mut observation = [0.0; LOCAL_OBSERVATION_SIZE];
    write_feature(&mut observation, 0, agent.physiology.hunger);
    write_feature(&mut observation, 1, agent.physiology.thirst);
    write_feature(&mut observation, 2, agent.physiology.health);
    write_feature(&mut observation, 3, agent.physiology.hit_points / 100.0);
    write_feature(
        &mut observation,
        4,
        (agent.age_seconds / max_age).clamp(0.0, 1.0),
    );
    write_feature(
        &mut observation,
        5,
        (velocity.length() / MAX_LINEAR_SPEED).clamp(0.0, 1.0),
    );
    write_feature(
        &mut observation,
        6,
        (angular_velocity.0 / MAX_ANGULAR_SPEED).clamp(-1.0, 1.0),
    );
    write_feature(&mut observation, 7, rotation.sin);
    write_feature(&mut observation, 8, rotation.cos);
    write_feature(&mut observation, 9 + agent.species.index(), 1.0);

    for (ray_index, ray) in agent.rays.iter().enumerate() {
        let start = PROPRIOCEPTION_SIZE + ray_index * (2 + RAY_KIND_COUNT);
        if let Some(kind) = ray.kind {
            write_feature(&mut observation, start, ray.normalized_distance);
            write_feature(&mut observation, start + 1, 1.0);
            write_feature(&mut observation, start + 2 + kind.index(), 1.0);
        }
    }
    let lesson_start = PROPRIOCEPTION_SIZE + RAY_COUNT * (2 + RAY_KIND_COUNT);
    write_feature(&mut observation, lesson_start + stage.index(), 1.0);
    observation
}

/// Write one derived tensor feature when its fixed schema index is present.
fn write_feature(features: &mut [f32], index: usize, value: f32) {
    if let Some(feature) = features.get_mut(index) {
        *feature = value;
    } else {
        debug_assert!(false, "fixed tensor feature index is outside its schema");
    }
}

/// Return fixed tensor slots for one evenly spaced all-around ray profile.
fn active_ray_indices(count: PerceptionRayCount) -> impl Iterator<Item = usize> {
    let active_count = usize::from(count.get());
    (0..active_count).map(move |slot| {
        let canonical_sector = slot * RAY_COUNT / active_count;
        ray_storage_index(canonical_sector)
    })
}

/// Map counter-clockwise sector order to the forward-alternating tensor order.
const fn ray_storage_index(canonical_sector: usize) -> usize {
    // Forward owns slot zero, rear owns the final slot, and paired left/right
    // directions alternate between them in increasing angular distance.
    if canonical_sector == 0 {
        0
    } else if canonical_sector == RAY_COUNT / 2 {
        RAY_COUNT - 1
    } else if canonical_sector < RAY_COUNT / 2 {
        canonical_sector * 2 - 1
    } else {
        (RAY_COUNT - canonical_sector) * 2
    }
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

/// Sample uniform all-around semantic sectors from the resolved world.
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
        let forward = Vec2::new(rotation.cos, rotation.sin);
        for ray_index in active_ray_indices(active_perception.0) {
            let Some(relative_angle) = RAY_ANGLES.get(ray_index) else {
                continue;
            };
            let direction = Vec2::from_angle(*relative_angle).rotate(forward);
            let sample = Dir2::new(direction)
                .ok()
                .and_then(|direction| {
                    spatial_query.cast_ray(position.0, direction, SIGHT_RANGE, false, &filter)
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
        for (target, target_position, semantic) in &semantics {
            if target == entity {
                continue;
            }
            let offset = target_position.0 - position.0;
            let distance = offset.length();
            if !(f32::EPSILON..=SIGHT_RANGE).contains(&distance) {
                continue;
            }
            let Ok(direction) = Dir2::new(offset) else {
                continue;
            };
            let visible = spatial_query
                .cast_ray(position.0, direction, distance + 0.01, false, &filter)
                .is_some_and(|hit| hit.entity == target);
            if !visible {
                continue;
            }
            let relative_angle = forward.perp_dot(*direction).atan2(forward.dot(*direction));
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

/// Smallest absolute angular separation between two radians.
fn angular_distance(left: f32, right: f32) -> f32 {
    (left - right).sin().atan2((left - right).cos()).abs()
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

    /// Reward shaping must combine survival, food, and conserved water events.
    #[test]
    fn configured_reward_weights_combine_exact_event_deltas() {
        // One new food event and half a water unit isolate every configured
        // reward term in a single transition.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        config
            .apply_experiment_tuning(ExperimentTuning {
                survival_reward_per_second: 0.5,
                food_reward: 3.0,
                water_reward_per_unit: 2.0,
                ..ExperimentTuning::default()
            })
            .expect("bounded tuning is valid");
        let before = AgentMetrics {
            food_eaten: 1,
            water_consumed: 0.25,
            ..AgentMetrics::default()
        };
        let after = AgentMetrics {
            food_eaten: 2,
            water_consumed: 0.75,
            ..before
        };

        assert!((transition_reward(&config, before, after) - 4.05).abs() < 1e-6);
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
    }

    /// A reduced ray profile must keep the actor tensor width unchanged.
    #[cfg(feature = "render")]
    #[test]
    fn reduced_perception_uses_only_evenly_spaced_active_rays() {
        // Four rays should cover forward, left, rear, and right while unused
        // observation slots remain part of the stable network contract.
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
            [0, 17, 35, 18]
        );
        assert_eq!(nearest_active_ray(0.0, ray_count), Some(0));
        assert_eq!(
            nearest_active_ray(89.0_f32.to_radians(), ray_count),
            Some(17)
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
                initial_health_fraction: 0.4,
                need_drain_multiplier: 0.25,
                damage_multiplier: 0.25,
                episode_step_limit: 100,
                ..ExperimentTuning::default()
            })
            .expect("bounded tuning is valid");
        let mut ecosystem = Ecosystem::new(config, 93).expect("ecosystem initializes");
        let agent = *ecosystem
            .visual_snapshot()
            .agents
            .first()
            .expect("solo agent is visible");
        let idle = LocomotionAction::new(0.0, 0.0).expect("idle action is valid");
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

        assert!((agent.hit_points - 40.0).abs() < f32::EPSILON);
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
        let action = LocomotionAction::new(0.0, 0.0).expect("idle action is valid");
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

    /// Local semantic perception detects food behind the agent without coordinates.
    #[test]
    fn local_perception_covers_rear_resource_sector() {
        let observation = observe_single_food(std::f32::consts::PI, 4.0);
        assert!(
            observation_contains(&observation, PerceptKind::Food),
            "rear food must produce a local semantic hit"
        );
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
    /// Repeated spawn and consumption cycles must retain unique critic slots.
    #[test]
    fn live_food_critic_slots_remain_unique_after_reuse() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
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

    /// Survival keeps one food available while delaying replenishment to two.
    #[test]
    fn survival_food_supply_stays_between_one_and_two_with_delayed_replenishment() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        assert_eq!(config.initial_food, 2);
        assert_eq!(config.max_food, 2);
        let interval = config.food_spawn_interval;
        let mut ecosystem = Ecosystem::new(config, 37).expect("world spawns");
        assert_eq!(food_count(&mut ecosystem), 2);

        despawn_one_food(&mut ecosystem);
        let id = ecosystem.living_agents()[0];
        let idle = LocomotionAction::new(0.0, 0.0).expect("idle action is valid");
        for _ in 1..interval {
            ecosystem.step(&[(id, idle)]).expect("world advances");
            assert_eq!(food_count(&mut ecosystem), 1);
        }
        ecosystem
            .step(&[(id, idle)])
            .expect("spawn interval advances");
        assert_eq!(food_count(&mut ecosystem), 2);

        despawn_all_food(&mut ecosystem);
        ecosystem.step(&[(id, idle)]).expect("food floor advances");
        assert_eq!(food_count(&mut ecosystem), 1);
    }

    /// An immobile bunny must eventually die and lose all physics capability.
    #[test]
    fn unmet_needs_terminate_agent_and_remove_collider() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 43).expect("world spawns");
        let id = ecosystem
            .living_agents()
            .into_iter()
            .next()
            .expect("solo bunny exists");
        let idle = LocomotionAction::new(0.0, 0.0).expect("idle action is valid");
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

    /// Thorn contact must reduce both health representations by simulated time.
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
        ecosystem.resolve_physiology(&BTreeSet::from([id]));
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
        assert!(agent.physiology.health < 1.0);
        assert!(agent.physiology.hit_points < 100.0);
        assert!(agent.metrics.thorn_damage > before.thorn_damage);
    }

    /// A fully hydrated agent must not remove unusable water from the well.
    #[test]
    fn full_thirst_contact_preserves_well_water() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let mut ecosystem = Ecosystem::new(config, 51).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let before = ecosystem.app.world().resource::<WellState>().water;

        ecosystem.resolve_drinking(&BTreeSet::from([id]));

        let after = ecosystem.app.world().resource::<WellState>().water;
        assert!((after - before).abs() < f32::EPSILON);
        let entity = ecosystem.agent_entities()[&id];
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert!(agent.metrics.water_consumed.abs() < f32::EPSILON);
    }

    /// A partially thirsty agent withdraws exactly the water it can absorb.
    #[test]
    fn partial_thirst_contact_withdraws_only_absorbable_water() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let drink_rate = config.well_drink_rate;
        let mut ecosystem = Ecosystem::new(config, 52).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(entity)
            .expect("bunny exists")
            .physiology
            .thirst = 0.99;
        let expected = 0.01 * drink_rate / DRINK_THIRST_RECOVERY;
        let before = ecosystem.app.world().resource::<WellState>().water;

        ecosystem.resolve_drinking(&BTreeSet::from([id]));

        let after = ecosystem.app.world().resource::<WellState>().water;
        let agent = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert!((before - after - expected).abs() < 1e-6);
        assert!((agent.metrics.water_consumed - expected).abs() < 1e-6);
        assert!((agent.physiology.thirst - 1.0).abs() < f32::EPSILON);
    }

    /// Scarce well water is shared equally and the well refills over time.
    #[test]
    fn finite_well_depletes_and_refills() {
        let config = SimulationConfig::for_stage(CurriculumStage::Competition)
            .expect("competition defaults are valid");
        let requested = config.well_drink_rate * config.time_step;
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
                .thirst = 0.0;
        }
        ecosystem.app.world_mut().resource_mut::<WellState>().water = requested * 1.5;

        ecosystem.resolve_drinking(&BTreeSet::from([first, second]));

        let water = ecosystem.app.world().resource::<WellState>().water;
        assert!(water.abs() < f32::EPSILON);
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
        let expected_share = requested * 0.75;
        assert!((first_agent.metrics.water_consumed - expected_share).abs() < f32::EPSILON);
        assert!((second_agent.metrics.water_consumed - expected_share).abs() < f32::EPSILON);

        ecosystem.refill_well();
        assert!(
            (ecosystem.app.world().resource::<WellState>().water - refill).abs() < f32::EPSILON
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
        let idle = LocomotionAction::new(0.0, 0.0).expect("idle action is valid");

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
        for (id, position, velocity) in [
            (
                first,
                Vec2::new(-2.0, collision_y),
                Vec2::new(MAX_LINEAR_SPEED, 0.0),
            ),
            (
                second,
                Vec2::new(2.0, collision_y),
                Vec2::new(-MAX_LINEAR_SPEED, 0.0),
            ),
        ] {
            let mut entity = ecosystem.app.world_mut().entity_mut(entities[&id]);
            *entity.get_mut::<Position>().expect("agent has a position") = Position(position);
            entity
                .get_mut::<Transform>()
                .expect("agent has a transform")
                .translation = position.extend(0.0);
            *entity
                .get_mut::<LinearVelocity>()
                .expect("agent has a velocity") = LinearVelocity(velocity);
        }

        for _ in 0..8 {
            ecosystem.app.update();
            let contacts = ecosystem.collect_contacts();
            assert!(contacts.predation.is_empty());
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
        assert!(first_position.distance(second_position) >= AGENT_RADIUS * 2.0 - 0.01);
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

        ecosystem.resolve_predation(&[(fox, bunny)]);

        let bunny_body = ecosystem
            .app
            .world()
            .get::<AgentBody>(entities[&bunny])
            .expect("bunny exists");
        assert_eq!(bunny_body.life, LifeState::Dead(DeathCause::Predation));
        assert_eq!(bunny_body.physiology.hit_points, 0.0);
        let fox_body = ecosystem
            .app
            .world()
            .get::<AgentBody>(entities[&fox])
            .expect("fox exists");
        assert!(fox_body.is_alive());
        assert_eq!(fox_body.metrics.food_eaten, 1);
        assert_eq!(fox_body.metrics.kills, 1);
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
                            LocomotionAction::new(1.0, turn).expect("scripted action is valid"),
                        )
                    })
                    .collect::<Vec<_>>();
                if actions.is_empty() {
                    break;
                }
                let result = ecosystem.step(&actions).expect("obstacle world advances");
                assert!(
                    maximum_agent_solid_penetration(&mut ecosystem) <= 0.001,
                    "seed {seed}, step {step} retained a penetrated solid contact"
                );
                if result.is_done {
                    break;
                }
            }
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
        ecosystem
            .app
            .world_mut()
            .get_mut::<AgentBody>(agent_entity)
            .expect("agent body exists")
            .rays = [RaySample::MISS; RAY_COUNT];
        ecosystem.app.update();
        ecosystem.app.world_mut().run_schedule(PerceptionSchedule);
        ecosystem.agent_output(id).expect("agent output exists").1
    }

    /// Return whether one local observation contains the requested semantic hit.
    fn observation_contains(observation: &[f32], kind: PerceptKind) -> bool {
        (0..RAY_COUNT).any(|ray_index| {
            let start = PROPRIOCEPTION_SIZE + ray_index * (2 + RAY_KIND_COUNT);
            observation[start + 1] > 0.5 && observation[start + 2 + kind.index()] > 0.5
        })
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

    /// Collect conservative circular clearances for solid trees and rocks.
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
