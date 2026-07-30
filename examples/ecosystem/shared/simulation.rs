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
    EcosystemState, EyeSide, GlobalState, JointStep, LocalObservation, LocomotionAction,
    PerceptKind, PerceptionRayCount, SimulationConfig, Species, GLOBAL_AGENT_FEATURES,
    GLOBAL_FOOD_FEATURES, GLOBAL_OBSTACLE_FEATURES, LOCAL_OBSERVATION_SIZE, MAX_AGENTS, MAX_FOOD,
    MAX_OBSTACLES, PROPRIOCEPTION_SIZE, RAY_ANGLES, RAY_COUNT, RAY_EYES, RAY_KIND_COUNT,
};
#[cfg(feature = "render")]
use super::domain::{VisualAgent, VisualObject, VisualObjectKind, VisualRay, VisualWorldSnapshot};
use super::rng::SplitMix64;

/// Conservative agent half-extent used for spawn clearance and route tests.
const AGENT_RADIUS: f32 = 0.75;

/// Rectangle dimensions shared by agent physics and rendering.
pub(super) const AGENT_SIZE: Vec2 = Vec2::new(1.5, 1.1);

/// Radius of a food interaction sensor.
const FOOD_RADIUS: f32 = 0.55;

/// Solid radius of the well base.
const WELL_RADIUS: f32 = 1.25;

/// Radius in which an agent automatically drinks.
const WELL_SENSOR_RADIUS: f32 = 2.25;

/// Forward distance from the survival bunny to both initial resource choices.
const SURVIVAL_RESOURCE_FORWARD_DISTANCE: f32 = 5.5;

/// Lateral offset of the initial resource route from the body centerline.
const SURVIVAL_RESOURCE_LATERAL_DISTANCE: f32 = 2.0;

/// Distance that places tutorial food before the well on one learnable route.
const SURVIVAL_FOOD_LEAD_DISTANCE: f32 = 2.4;

/// Maximum distance represented by every semantic sector.
const SIGHT_RANGE: f32 = 18.0;

/// Half-width of one fixed-capacity semantic sector.
const RAY_HALF_WIDTH: f32 = 5.0_f32.to_radians();

/// Forward offset of each eye from the agent center.
const EYE_FORWARD_OFFSET: f32 = 0.48;

/// Lateral offset separating left and right ray origins.
const EYE_LATERAL_OFFSET: f32 = 0.28;

/// Maximum linear speed used for physics and observation normalization.
const MAX_LINEAR_SPEED: f32 = 8.0;

/// Maximum angular speed in radians per second.
const MAX_ANGULAR_SPEED: f32 = 3.0;

/// Local forward acceleration at a unit action.
const FORWARD_ACCELERATION: f32 = 28.0;

/// Simulated thorn-contact seconds between one-point HP losses.
const THORN_DAMAGE_INTERVAL_SECONDS: u16 = 1;

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

    /// Steps elapsed toward the next need loss.
    need_steps: u32,

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
}

impl Physiology {
    /// Construct reset physiology from validated integer settings.
    const fn new(hit_points: u8, satiation: u8, hydration: u8) -> Self {
        Self {
            satiation,
            hydration,
            hit_points,
            need_steps: 0,
            movement_need_progress: 0,
            starvation_steps: 0,
            dehydration_steps: 0,
            thorn_steps: 0,
            drinking_steps: 0,
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

    /// Need-weighted food events used only to calculate transition reward.
    food_reward_units: f32,

    /// Need-weighted drink events used only to calculate transition reward.
    drink_reward_units: f32,

    /// Bunnies consumed by a fox.
    kills: u32,

    /// Hit points lost to thorn contact.
    thorn_damage: f32,

    /// Legacy metric retained at zero for artifact compatibility.
    overconsumption_damage: f32,

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

    /// Fixed simulation steps elapsed since episode spawn.
    age_steps: u16,

    /// Conjugate eye yaw relative to the body heading.
    gaze_yaw: f32,

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
        if living_before.is_empty() || self.step >= self.config.max_steps() {
            return Err(SimulationError::EpisodeFinished);
        }
        let metrics_before = self.agent_metrics_by_id();

        self.step = self.step.saturating_add(1);
        self.refill_well();

        // All controls are written before Avian advances, preserving joint
        // action semantics independent of entity query order.
        self.apply_actions(&actions);
        self.app.update();
        self.advance_agent_ages();

        let contacts = self.collect_contacts();
        self.resolve_food(&contacts.food_winners);
        self.replenish_food()?;
        self.resolve_drinking(&contacts.well_agents);
        self.resolve_predation(&contacts.predation);
        self.resolve_physiology(&contacts.thorn_agents);
        self.disable_newly_dead();

        self.app.world_mut().run_schedule(PerceptionSchedule);

        let horizon = self.step >= self.config.max_steps();
        let metrics_after = self.agent_metrics_by_id();
        let mut results = Vec::with_capacity(living_before.len());
        for id in living_before {
            if let Some((species, observation, life)) = self.agent_output(id) {
                let (status, death_cause) = match life {
                    LifeState::Alive if horizon => (EpisodeStatus::Truncated, None),
                    LifeState::Alive => (EpisodeStatus::Continuing, None),
                    LifeState::Dead(cause) => (EpisodeStatus::Terminated, Some(cause)),
                };
                let event_reward = transition_reward(
                    &self.config,
                    metrics_before.get(&id).copied().unwrap_or_default(),
                    metrics_after.get(&id).copied().unwrap_or_default(),
                );
                let survival_reward = if status == EpisodeStatus::Continuing {
                    0.0
                } else {
                    self.agent_survival_state(id)
                        .map_or(0.0, |(age, hit_points)| {
                            terminal_survival_reward(
                                age,
                                hit_points,
                                self.config.maximum_hit_points,
                            )
                        })
                };
                results.push(AgentStep {
                    id,
                    species,
                    observation,
                    reward: event_reward + f64::from(survival_reward),
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
                gaze_yaw: agent.gaze_yaw,
                is_alive: agent.is_alive(),
                satiation: agent.physiology.satiation,
                hydration: agent.physiology.hydration,
                hit_points: agent.physiology.hit_points,
            });
            if agent.is_alive() {
                // Reconstruct endpoints from the exact stored sector samples
                // used by the actor's current local observation.
                let body_forward = Vec2::from_angle(heading);
                rays.extend(
                    active_ray_indices(perception_ray_count).filter_map(|ray_index| {
                        let sample = agent.rays.get(ray_index)?;
                        let angle = RAY_ANGLES.get(ray_index)?;
                        let eye = *RAY_EYES.get(ray_index)?;
                        let start = eye_origin(position, body_forward, eye);
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
        let config = &self.config;
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        let mut metrics = query
            .iter(world)
            .map(|agent| AgentEpisodeMetrics {
                id: agent.id,
                species: agent.species,
                lifetime_seconds: agent.age_seconds(config.time_step),
                episode_return: resource_reward(config, AgentMetrics::default(), agent.metrics)
                    + terminal_survival_reward(
                        agent.age_seconds(config.time_step),
                        agent.physiology.hit_points,
                        config.maximum_hit_points,
                    ),
                food_eaten: agent.metrics.food_eaten,
                water_consumed: agent.metrics.water_consumed,
                kills: agent.metrics.kills,
                thorn_damage: agent.metrics.thorn_damage,
                overconsumption_damage: agent.metrics.overconsumption_damage,
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

        if self.config.stage == CurriculumStage::Survival {
            self.spawn_survival_lesson();
            return Ok(());
        }

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

    /// Spawn separated visible resource choices for the single-agent lesson.
    fn spawn_survival_lesson(&mut self) {
        // Rotate the whole layout per episode so the actor must use egocentric
        // perception instead of memorizing one world-space direction.
        let route_angle = self
            .rng
            .f32_between(-std::f32::consts::PI, std::f32::consts::PI);
        let forward = Vec2::from_angle(route_angle);
        let lateral = forward.perp();
        let bunny_position = forward * -4.0;
        let resource_center = bunny_position + forward * SURVIVAL_RESOURCE_FORWARD_DISTANCE;
        let route_offset = lateral * SURVIVAL_RESOURCE_LATERAL_DISTANCE;
        let well_position = resource_center + route_offset;
        let first_food_position = well_position - forward * SURVIVAL_FOOD_LEAD_DISTANCE;
        self.spawn_well(well_position);
        self.spawn_agent_facing(AgentId(0), Species::Bunny, bunny_position, route_angle);
        self.spawn_food_at(first_food_position);
        if self.config.initial_food > 1 {
            self.spawn_food_at(bunny_position + forward * 8.0 - lateral * 3.0);
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
            WellSensor,
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
        let position = self.open_position(FOOD_RADIUS + 0.25)?;
        self.spawn_food_at(position);
        Ok(())
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
            &mut AgentBody,
            &mut ConstantLocalLinearAcceleration,
            &mut AngularVelocity,
            &mut MaxLinearSpeed,
        )>();
        for (mut agent, mut acceleration, mut angular_velocity, mut max_speed) in
            query.iter_mut(world)
        {
            if !agent.is_alive() {
                continue;
            }
            if let Some(action) = actions.get(&agent.id) {
                let movement_scale = self.config.movement_speed_multiplier;
                let forward_throttle = action.forward.mul_add(0.5, 0.5).clamp(0.0, 1.0);
                acceleration.0 = Vec2::X * forward_throttle * FORWARD_ACCELERATION * movement_scale;
                max_speed.0 = MAX_LINEAR_SPEED * movement_scale;
                angular_velocity.0 = action.turn * MAX_ANGULAR_SPEED;
                agent.gaze_yaw = action.gaze * self.config.gaze_yaw_limit_degrees.to_radians();
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
        let maximum_satiation = self.config.maximum_satiation;
        for (food, winner) in winners {
            if let Some(agent_entity) = entity_by_id.get(winner).copied() {
                if let Some(mut agent) = self.app.world_mut().get_mut::<AgentBody>(agent_entity) {
                    if agent.is_alive() && agent.species == Species::Bunny {
                        let reward_multiplier =
                            need_reward_multiplier(agent.physiology.satiation, maximum_satiation);
                        agent.physiology.satiation = agent
                            .physiology
                            .satiation
                            .saturating_add(1)
                            .min(maximum_satiation);
                        agent.metrics.food_eaten = agent.metrics.food_eaten.saturating_add(1);
                        agent.metrics.food_reward_units += reward_multiplier;
                    }
                }
            }
            let _despawned = self.app.world_mut().despawn(*food);
        }
    }

    /// Divide scarce well water fairly without withdrawing more than agents absorb.
    fn resolve_drinking(&mut self, drinkers: &BTreeSet<AgentId>) {
        let entity_by_id = self.agent_entities();
        let requested = self.config.well_drink_rate;
        let maximum_hydration = self.config.maximum_hydration;
        let drinking_interval = self.config.steps_for_seconds(1);
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
                    let reward_multiplier =
                        need_reward_multiplier(agent.physiology.hydration, maximum_hydration);
                    agent.physiology.hydration = agent
                        .physiology
                        .hydration
                        .saturating_add(1)
                        .min(maximum_hydration);
                    agent.metrics.water_consumed += requested;
                    agent.metrics.drink_reward_units += reward_multiplier;
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
                    let reward_multiplier = need_reward_multiplier(
                        fox.physiology.satiation,
                        self.config.maximum_satiation,
                    );
                    fox.physiology.satiation = fox
                        .physiology
                        .satiation
                        .saturating_add(1)
                        .min(self.config.maximum_satiation);
                    fox.metrics.food_eaten = fox.metrics.food_eaten.saturating_add(1);
                    fox.metrics.food_reward_units += reward_multiplier;
                    fox.metrics.kills = fox.metrics.kills.saturating_add(1);
                }
            }
        }
    }

    /// Advance needs and apply starvation, dehydration, and thorn damage.
    fn resolve_physiology(&mut self, thorn_agents: &BTreeSet<AgentId>) {
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
        let extent = self.config.map_half_extent + 2.0;
        let world = self.app.world_mut();
        let mut query = world.query::<(&mut AgentBody, &Position, Option<&LinearVelocity>)>();
        for (mut agent, position, velocity) in query.iter_mut(world) {
            if !agent.is_alive() {
                continue;
            }
            let was_starving = agent.physiology.satiation == 0;
            let was_dehydrated = agent.physiology.hydration == 0;
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
            agent.physiology.need_steps = agent
                .physiology
                .need_steps
                .saturating_add(1 + u32::from(extra_need_steps));
            while agent.physiology.need_steps >= need_interval {
                agent.physiology.need_steps -= need_interval;
                agent.physiology.satiation = agent.physiology.satiation.saturating_sub(1);
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
            let deprivation_damage = u8::from(starvation_due) + u8::from(dehydration_due);
            let total_damage = deprivation_damage.saturating_add(u8::from(thorn_due));
            agent.physiology.hit_points = agent.physiology.hit_points.saturating_sub(total_damage);
            if thorn_due {
                agent.metrics.thorn_damage += 1.0;
            }

            let lethal_cause = if thorn_due && (starvation_due || dehydration_due) {
                Some(DeathCause::CombinedDamage)
            } else if starvation_due && dehydration_due {
                Some(DeathCause::Deprivation)
            } else if starvation_due {
                Some(DeathCause::Starvation)
            } else if dehydration_due {
                Some(DeathCause::Dehydration)
            } else if thorn_due {
                Some(DeathCause::Thorns)
            } else {
                None
            };

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

    /// Return elapsed seconds and HP for one terminal reward calculation.
    fn agent_survival_state(&mut self, id: AgentId) -> Option<(f32, u8)> {
        let time_step = self.config.time_step;
        let world = self.app.world_mut();
        let mut query = world.query::<&AgentBody>();
        query
            .iter(world)
            .find(|agent| agent.id == id)
            .map(|agent| (agent.age_seconds(time_step), agent.physiology.hit_points))
    }

    /// Return one agent's local output after interaction resolution.
    fn agent_output(&mut self, id: AgentId) -> Option<(Species, LocalObservation, LifeState)> {
        let max_age = f32::from(self.config.episode_seconds);
        let stage = self.config.stage;
        let max_linear_speed = MAX_LINEAR_SPEED * self.config.movement_speed_multiplier;
        let gaze_yaw_limit = self.config.gaze_yaw_limit_degrees.to_radians();
        let maximum_satiation = self.config.maximum_satiation;
        let maximum_hydration = self.config.maximum_hydration;
        let maximum_hit_points = self.config.maximum_hit_points;
        let time_step = self.config.time_step;
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
                    stage,
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
                (velocity.x / max_linear_speed).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 6,
                (velocity.y / max_linear_speed).clamp(-1.0, 1.0),
            );
            write_feature(
                &mut state,
                start + 7,
                f32::from(agent.physiology.satiation) / f32::from(maximum_satiation),
            );
            write_feature(
                &mut state,
                start + 8,
                f32::from(agent.physiology.hydration) / f32::from(maximum_hydration),
            );
            let health_fraction =
                f32::from(agent.physiology.hit_points) / f32::from(maximum_hit_points);
            write_feature(&mut state, start + 9, health_fraction);
            write_feature(&mut state, start + 10, health_fraction);
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
            1.0 - self.step as f32 / self.config.max_steps() as f32,
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

/// Compute one configured transition reward from need-weighted resource events.
fn transition_reward(config: &SimulationConfig, before: AgentMetrics, after: AgentMetrics) -> f64 {
    f64::from(resource_reward(config, before, after))
}

/// Compute resource-event reward while preserving the trainer's scalar boundary.
fn resource_reward(config: &SimulationConfig, before: AgentMetrics, after: AgentMetrics) -> f32 {
    // Monotonic weighted units prevent reset or diagnostic mutations from
    // manufacturing negative reward events.
    let food_reward_units = (after.food_reward_units - before.food_reward_units).max(0.0);
    let drink_reward_units = (after.drink_reward_units - before.drink_reward_units).max(0.0);
    let food_reward = config.food_reward * food_reward_units;
    config.drink_reward.mul_add(drink_reward_units, food_reward)
}

/// Return terminal seconds weighted by the remaining HP fraction.
fn terminal_survival_reward(age_seconds: f32, hit_points: u8, maximum_hit_points: u8) -> f32 {
    if maximum_hit_points == 0 {
        return 0.0;
    }
    age_seconds * f32::from(hit_points) / f32::from(maximum_hit_points)
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

/// Return the reward multiplier for consuming a resource at one need level.
fn need_reward_multiplier(points: u8, maximum: u8) -> f32 {
    let reserve = f32::from(points) / f32::from(maximum.max(1));
    if reserve <= 0.10 {
        1.25
    } else if reserve <= 0.50 {
        1.0
    } else if reserve <= 0.75 {
        0.9
    } else if reserve < 0.90 {
        0.75
    } else {
        0.0
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
    stage: CurriculumStage,
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
    write_feature(&mut observation, 3, health_fraction);
    write_feature(&mut observation, 4, (age_seconds / max_age).clamp(0.0, 1.0));
    write_feature(
        &mut observation,
        5,
        (velocity.length() / max_linear_speed).clamp(0.0, 1.0),
    );
    write_feature(
        &mut observation,
        6,
        (angular_velocity.0 / MAX_ANGULAR_SPEED).clamp(-1.0, 1.0),
    );
    write_feature(&mut observation, 7, rotation.sin);
    write_feature(&mut observation, 8, rotation.cos);
    write_feature(&mut observation, 9 + agent.species.index(), 1.0);
    write_feature(
        &mut observation,
        11,
        (agent.gaze_yaw / gaze_yaw_limit).clamp(-1.0, 1.0),
    );

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

/// Return the most important fixed binocular slots for the active profile.
fn active_ray_indices(count: PerceptionRayCount) -> impl Iterator<Item = usize> {
    0..usize::from(count.get())
}

/// Find an active sector only when the target lies inside its fixed arc.
fn nearest_active_ray(
    relative_angle: f32,
    count: PerceptionRayCount,
    eye: EyeSide,
) -> Option<usize> {
    // Keep the original five-degree half-width when rays are removed. This
    // creates real blind gaps instead of silently widening the remaining rays.
    let (ray_index, sector_angle) = active_ray_indices(count)
        .filter(|ray_index| RAY_EYES.get(*ray_index).copied() == Some(eye))
        .filter_map(|ray_index| RAY_ANGLES.get(ray_index).map(|angle| (ray_index, *angle)))
        .min_by(|(_, left), (_, right)| {
            angular_distance(relative_angle, *left)
                .total_cmp(&angular_distance(relative_angle, *right))
        })?;
    (angular_distance(relative_angle, sector_angle) <= RAY_HALF_WIDTH).then_some(ray_index)
}

/// Sample two forward-facing semantic cones from the resolved world.
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
            let Some(eye) = RAY_EYES.get(ray_index).copied() else {
                continue;
            };
            let origin = eye_origin(position.0, body_forward, eye);
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
        for eye in [EyeSide::Left, EyeSide::Right] {
            let origin = eye_origin(position.0, body_forward, eye);
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
                let Some(sector) = nearest_active_ray(relative_angle, active_perception.0, eye)
                else {
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
}

/// Return one eye origin in world coordinates from the body pose.
fn eye_origin(position: Vec2, body_forward: Vec2, eye: EyeSide) -> Vec2 {
    position
        + body_forward * EYE_FORWARD_OFFSET
        + body_forward.perp() * EYE_LATERAL_OFFSET * eye.lateral_sign()
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
        assert_eq!(config.need_loss_interval_seconds, 5);
        assert_eq!(config.starvation_damage_interval_seconds, 3);
        assert_eq!(config.dehydration_damage_interval_seconds, 2);
        assert_eq!(config.episode_seconds, 20);
    }

    /// Terminal survival reward must weight elapsed seconds by remaining HP.
    #[test]
    fn terminal_survival_reward_weights_seconds_by_health() {
        assert!((terminal_survival_reward(20.0, 9, 10) - 18.0).abs() < f32::EPSILON);
        assert!((terminal_survival_reward(7.5, 4, 5) - 6.0).abs() < f32::EPSILON);
        assert_eq!(terminal_survival_reward(20.0, 0, 5), 0.0);
    }

    /// The horizon transition and episode metrics must expose the same return.
    #[test]
    fn horizon_emits_health_weighted_return_once() {
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        config.episode_seconds = 5;
        config.initial_hit_points = 4;
        let expected_steps = config.max_steps();
        let mut ecosystem = Ecosystem::new(config, 73).expect("ecosystem initializes");
        let id = ecosystem.living_agents()[0];
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0).expect("idle action is valid");
        let mut rewards = Vec::new();
        let mut final_status = EpisodeStatus::Continuing;
        for _ in 0..expected_steps {
            let step = ecosystem.step(&[(id, idle)]).expect("world advances");
            rewards.push(step.agents[0].reward);
            final_status = step.agents[0].status;
        }

        assert!(rewards[..rewards.len() - 1]
            .iter()
            .all(|reward| reward.abs() < f64::EPSILON));
        assert!((rewards[rewards.len() - 1] - 4.0).abs() < f64::EPSILON);
        assert_eq!(final_status, EpisodeStatus::Truncated);
        let metrics = ecosystem.episode_metrics();
        assert!((metrics[0].episode_return - 4.0).abs() < f32::EPSILON);
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

    /// Reward shaping must combine survival, food, and conserved water events.
    #[test]
    fn configured_reward_weights_combine_exact_event_deltas() {
        // Weighted food and water units isolate every configured reward term.
        let mut config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("stage defaults are valid");
        config
            .apply_experiment_tuning(ExperimentTuning {
                food_reward: 3.0,
                drink_reward: 2.0,
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
            food_reward_units: 1.25,
            drink_reward_units: 0.45,
            ..before
        };

        assert!((transition_reward(&config, before, after) - 4.65).abs() < 1e-6);
    }

    /// Resource reward must decline through the requested reserve zones.
    #[test]
    fn need_reward_multiplier_uses_declared_satiation_zones() {
        let cases = [(0, 1.25), (1, 1.0), (2, 1.0), (3, 0.9), (4, 0.75), (5, 0.0)];

        for (points, expected) in cases {
            assert!((need_reward_multiplier(points, 5) - expected).abs() < f32::EPSILON);
        }
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
        let left_origin = snapshot.rays.first().expect("left-eye ray exists").start;
        let right_origin = snapshot.rays.get(1).expect("right-eye ray exists").start;
        assert_ne!(left_origin, right_origin, "eye origins must be distinct");
        assert!(snapshot
            .rays
            .iter()
            .all(|ray| ray.start == left_origin || ray.start == right_origin));
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
        assert_eq!(
            nearest_active_ray(2.0_f32.to_radians(), ray_count, EyeSide::Left),
            Some(0)
        );
        assert_eq!(
            nearest_active_ray(-(2.0_f32.to_radians()), ray_count, EyeSide::Right),
            Some(3)
        );
        assert_eq!(
            nearest_active_ray(30.0_f32.to_radians(), ray_count, EyeSide::Left),
            None
        );
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
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0).expect("idle action is valid");
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

    /// Survival reset must show food before water on one learnable route.
    #[test]
    fn survival_reset_faces_visible_food_and_water() {
        // Sample several world rotations while preserving the same egocentric contract.
        for seed in [1, 29, 197, 4_099] {
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

            assert!(observation_contains(&observation, PerceptKind::Food));
            assert!(observation_contains(&observation, PerceptKind::Well));
            assert!(food_hit_angles.iter().any(|angle| *angle > 0.1));
            assert!(well_hit_angles.iter().any(|angle| *angle > 0.1));
            assert!(food_bearing > well_bearing + 0.1);
            assert!(well_bearing > 0.1);
            assert!(nearest_food.distance(position) > AGENT_RADIUS + FOOD_RADIUS);
            assert!(well.distance(position) > AGENT_RADIUS + WELL_SENSOR_RADIUS);
            assert!(!resource_contacts_agent::<FoodSlot>(&mut ecosystem, entity));
            assert!(!resource_contacts_agent::<WellSensor>(
                &mut ecosystem,
                entity
            ));
            assert_eq!(food_count(&mut ecosystem), 2);
        }
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
        let action = LocomotionAction::new(-1.0, 0.0, 0.0).expect("idle action is valid");
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
            idle.resolve_physiology(&BTreeSet::new());
            looking.resolve_physiology(&BTreeSet::new());
            moving.resolve_physiology(&BTreeSet::new());
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
        let reverse = LocomotionAction::new(-1.0, 0.0, 0.0).expect("brake action is valid");
        ecosystem.apply_actions(&BTreeMap::from([(id, reverse)]));
        let acceleration = ecosystem
            .app
            .world()
            .get::<ConstantLocalLinearAcceleration>(entity)
            .expect("agent acceleration exists");
        assert_eq!(acceleration.0, Vec2::ZERO);

        let neutral = LocomotionAction::new(0.0, 0.0, 0.0).expect("neutral action is valid");
        ecosystem.apply_actions(&BTreeMap::from([(id, neutral)]));
        let acceleration = ecosystem
            .app
            .world()
            .get::<ConstantLocalLinearAcceleration>(entity)
            .expect("agent acceleration exists");
        assert!(acceleration.0.x > 0.0);

        let forward = LocomotionAction::new(1.0, 0.0, 0.0).expect("forward action is valid");
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
        let mut ecosystem = Ecosystem::new(config, 37).expect("world spawns");
        let id = ecosystem.living_agents()[0];
        let entity = ecosystem.agent_entities()[&id];
        let start = ecosystem
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
        let body_forward = Vec2::new(rotation.cos, rotation.sin);
        // Select the first lesson target before collection can despawn it.
        let target = food_positions(&mut ecosystem)
            .into_iter()
            .min_by(|left, right| left.distance(start).total_cmp(&right.distance(start)))
            .expect("survival lesson has food");
        let mut largest_reward = f64::NEG_INFINITY;
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
            let turn = (bearing / 0.35).clamp(-1.0, 1.0);
            let action =
                LocomotionAction::new(forward, turn, 0.0).expect("steering action is valid");
            let step = ecosystem.step(&[(id, action)]).expect("world advances");
            largest_reward = largest_reward.max(step.agents[0].reward);
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
        assert!(agent.metrics.food_reward_units > 0.0);
        assert!(largest_reward > 7.0);
        let expected_forward_progress = SURVIVAL_RESOURCE_FORWARD_DISTANCE
            - SURVIVAL_FOOD_LEAD_DISTANCE
            - AGENT_RADIUS
            - FOOD_RADIUS
            - 0.25;
        assert!((end - start).dot(body_forward) > expected_forward_progress);
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

        ecosystem.resolve_physiology(&BTreeSet::new());

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

        ecosystem.resolve_physiology(&BTreeSet::from([id]));

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
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
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
            ecosystem.resolve_physiology(&BTreeSet::new());
        }
        let before_need_loss = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(before_need_loss.physiology.satiation, initial_satiation);
        assert_eq!(before_need_loss.physiology.hydration, initial_hydration);

        ecosystem.resolve_physiology(&BTreeSet::new());
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
            agent.physiology.need_steps = 0;
            agent.physiology.starvation_steps = 0;
            agent.physiology.dehydration_steps = 0;
        }

        for _ in 1..dehydration_interval {
            ecosystem.resolve_physiology(&BTreeSet::new());
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

        ecosystem.resolve_physiology(&BTreeSet::new());
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
            ecosystem.resolve_physiology(&BTreeSet::new());
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
            agent.physiology.need_steps = need_interval - 1;
            agent.physiology.starvation_steps = 0;
        }

        ecosystem.resolve_physiology(&BTreeSet::new());
        let depleted = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("agent exists");
        assert_eq!(depleted.physiology.satiation, 0);
        assert_eq!(depleted.physiology.starvation_steps, 0);
        let initial_hit_points = depleted.physiology.hit_points;

        for _ in 1..starvation_interval {
            ecosystem.resolve_physiology(&BTreeSet::new());
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

        ecosystem.resolve_physiology(&BTreeSet::new());
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

    /// Unrewarded movement and physiology diagnostics cannot change reward.
    #[test]
    fn reward_ignores_energy_and_damage_diagnostics() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let before = AgentMetrics::default();
        let after = AgentMetrics {
            thorn_damage: 40.0,
            overconsumption_damage: 20.0,
            collision_contacts: 100,
            ..before
        };

        assert_eq!(transition_reward(&config, before, after), 0.0);
    }

    /// Forward binocular perception must leave a true rear blind area.
    #[test]
    fn local_perception_excludes_rear_resource() {
        let observation = observe_single_food(std::f32::consts::PI, 4.0);
        let food_rays = (0..RAY_COUNT)
            .filter(|ray_index| {
                let start = PROPRIOCEPTION_SIZE + ray_index * (2 + RAY_KIND_COUNT);
                observation[start + 2 + PerceptKind::Food.index()] > 0.5
            })
            .collect::<Vec<_>>();
        assert!(
            !observation_contains(&observation, PerceptKind::Food),
            "rear food must stay outside both eye cones; hits={food_rays:?}"
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
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0).expect("idle action is valid");
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
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0).expect("idle action is valid");
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
            ecosystem.resolve_physiology(&BTreeSet::from([id]));
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

    /// Hydration and reward must advance once per completed contact second.
    #[test]
    fn drinking_requires_one_continuous_second_per_event() {
        let config = SimulationConfig::for_stage(CurriculumStage::Survival)
            .expect("survival defaults are valid");
        let interval = config.steps_for_seconds(1);
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
        assert_eq!(before_event.metrics.drink_reward_units, 0.0);

        ecosystem.resolve_drinking(&BTreeSet::from([id]));
        let after_event = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists");
        assert_eq!(after_event.physiology.hydration, 1);
        assert_eq!(after_event.metrics.drink_reward_units, 1.25);
    }

    /// One drinking event withdraws one fixed allocation and adds one hydration point.
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
    fn dehydrated_drinker_earns_need_weighted_water_reward() {
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
        let before = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists")
            .metrics;
        prime_drinking_event(&mut ecosystem, id);

        ecosystem.resolve_drinking(&BTreeSet::from([id]));

        let after = ecosystem
            .app
            .world()
            .get::<AgentBody>(entity)
            .expect("bunny exists")
            .metrics;
        assert!((after.water_consumed - expected_water).abs() < f32::EPSILON);
        assert!((after.drink_reward_units - 1.25).abs() < f32::EPSILON);
        let expected_reward = ecosystem.config.drink_reward * 1.25;
        assert!(
            (transition_reward(&ecosystem.config, before, after) - f64::from(expected_reward))
                .abs()
                < 1e-6
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
        let idle = LocomotionAction::new(-1.0, 0.0, 0.0).expect("idle action is valid");

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
                            LocomotionAction::new(1.0, turn, 0.0)
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
        let action = LocomotionAction::new(-1.0, 0.0, gaze).expect("gaze action is valid");
        ecosystem
            .step(&[(id, action)])
            .expect("gaze step succeeds")
            .agents[0]
            .observation
    }

    /// Return whether one local observation contains the requested semantic hit.
    fn observation_contains(observation: &[f32], kind: PerceptKind) -> bool {
        (0..RAY_COUNT).any(|ray_index| {
            let start = PROPRIOCEPTION_SIZE + ray_index * (2 + RAY_KIND_COUNT);
            observation[start + 1] > 0.5 && observation[start + 2 + kind.index()] > 0.5
        })
    }

    /// Return configured ray bearings containing one semantic class.
    fn observation_hit_angles(observation: &[f32], kind: PerceptKind) -> Vec<f32> {
        // Decode only active semantic hits while retaining their configured bearings.
        (0..RAY_COUNT)
            .filter(|ray_index| {
                let start = PROPRIOCEPTION_SIZE + ray_index * (2 + RAY_KIND_COUNT);
                observation[start + 1] > 0.5 && observation[start + 2 + kind.index()] > 0.5
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
        let interval = ecosystem.config.steps_for_seconds(1);
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
