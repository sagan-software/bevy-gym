//! Gymnasium's continuous CarRacing-v3 task.
//!
//! The environment keeps the three bounded controls, procedural closed track,
//! per-tile reward, frame cost, 95% lap completion, playfield failure, and
//! 1,000-step limit. Gymnasium exposes a 96x96 RGB observation. This example
//! projects the same visible track state into 14 scalars because the shared PPO
//! workflow currently accepts vectors. Rendering retains the Gymnasium player
//! view and indicator panel.

#![expect(
    clippy::indexing_slicing,
    clippy::missing_docs_in_private_items,
    clippy::suboptimal_flops,
    reason = "fixed Gymnasium vectors and rendering equations stay legible in this standalone example"
)]

use shakmaty as _;
use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::{RecurrentBehaviorSample, RecurrentPpoConfig};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::SplitMix64;
use bevy_gym::training::{run_continuous_workflow, ContinuousPpoExample};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use serde_json as _;

const FPS: f32 = 50.0;
const TIME_STEP: f32 = 1.0 / FPS;
const TRACK_TILES: usize = 240;
const TRACK_RADIUS: f32 = 18.0;
const TRACK_HALF_WIDTH: f32 = 3.2;
const TARGET_SPEED: f32 = 14.0;
const PLAYFIELD: f32 = 50.0;
const MAX_EPISODE_STEPS: usize = 1_000;
const LAP_COMPLETE_FRACTION: f32 = 0.95;
const OBSERVATION_DIM: usize = 14;

/// One generated track sample.
#[derive(Debug, Clone, Copy, Default)]
struct TrackPoint {
    /// World-space road center.
    center: [f32; 2],
    /// Unit road direction.
    tangent: [f32; 2],
    /// Signed local curvature.
    curvature: f32,
}

/// Continuous top-down racing task with Gymnasium-compatible rewards.
#[derive(Debug, Clone)]
struct CarRacing {
    /// Procedural closed road centerline.
    track: Vec<TrackPoint>,
    /// Car world-space position.
    position: [f32; 2],
    /// Car forward angle in radians.
    heading: f32,
    /// Forward speed.
    speed: f32,
    /// Smoothed steering control.
    steering: f32,
    /// Smoothed accelerator control.
    gas: f32,
    /// Current brake control.
    brake: f32,
    /// Closest road tile.
    nearest_tile: usize,
    /// Tiles visited during the lap.
    visited: Vec<bool>,
    /// Number of distinct visited tiles.
    visited_count: usize,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic procedural stream.
    rng: SplitMix64,
}

impl Default for CarRacing {
    fn default() -> Self {
        let mut env = Self {
            track: Vec::new(),
            position: [0.0; 2],
            heading: 0.0,
            speed: 0.0,
            steering: 0.0,
            gas: 0.0,
            brake: 0.0,
            nearest_tile: 0,
            visited: vec![false; TRACK_TILES],
            visited_count: 0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        };
        env.generate_track();
        env.place_at_start();
        env
    }
}

impl CarRacing {
    /// Build a smooth 12-checkpoint-style closed track from the reset stream.
    fn generate_track(&mut self) {
        let harmonic_a = self.rng.f32_between(1.8, 3.0);
        let harmonic_b = self.rng.f32_between(0.8, 1.7);
        let phase_a = self.rng.f32_between(0.0, std::f32::consts::TAU);
        let phase_b = self.rng.f32_between(0.0, std::f32::consts::TAU);
        let centers: Vec<[f32; 2]> = (0..TRACK_TILES)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / TRACK_TILES as f32;
                let radius = TRACK_RADIUS
                    + harmonic_a * (3.0 * angle + phase_a).sin()
                    + harmonic_b * (5.0 * angle + phase_b).sin();
                [radius * angle.cos(), radius * angle.sin()]
            })
            .collect();
        self.track = (0..TRACK_TILES)
            .map(|index| {
                let previous = centers[(index + TRACK_TILES - 1) % TRACK_TILES];
                let next = centers[(index + 1) % TRACK_TILES];
                let tangent = normalize([next[0] - previous[0], next[1] - previous[1]]);
                let next_next = centers[(index + 2) % TRACK_TILES];
                let next_tangent = normalize([
                    next_next[0] - centers[index][0],
                    next_next[1] - centers[index][1],
                ]);
                TrackPoint {
                    center: centers[index],
                    tangent,
                    curvature: cross(tangent, next_tangent).clamp(-0.25, 0.25) * 4.0,
                }
            })
            .collect();
    }

    /// Reset the car to Gymnasium's stationary starting pose.
    fn place_at_start(&mut self) {
        let start = self.track[0];
        self.position = start.center;
        self.heading = start.tangent[1].atan2(start.tangent[0]);
        self.speed = 0.0;
        self.steering = 0.0;
        self.gas = 0.0;
        self.brake = 0.0;
        self.nearest_tile = 0;
        self.visited.fill(false);
        self.visited_count = 0;
        self.elapsed_steps = 0;
        self.visit_nearby_tiles();
    }

    /// Find the closest road tile without relying on collision callbacks.
    fn update_nearest_tile(&mut self) {
        self.nearest_tile = self
            .track
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                squared_distance(self.position, left.center)
                    .total_cmp(&squared_distance(self.position, right.center))
            })
            .map_or(0, |(index, _)| index);
    }

    /// Mark the contacted tile and its immediate neighbors as visited.
    fn visit_nearby_tiles(&mut self) -> usize {
        let point = self.track[self.nearest_tile];
        if squared_distance(self.position, point.center) > TRACK_HALF_WIDTH.powi(2) {
            return 0;
        }
        let mut newly_visited = 0;
        for offset in [TRACK_TILES - 1, 0, 1] {
            let index = (self.nearest_tile + offset) % TRACK_TILES;
            if !self.visited[index] {
                self.visited[index] = true;
                self.visited_count += 1;
                newly_visited += 1;
            }
        }
        newly_visited
    }

    /// Return signed road offset and heading error at the closest tile.
    fn road_errors(&self) -> (f32, f32) {
        let point = self.track[self.nearest_tile];
        let delta = [
            self.position[0] - point.center[0],
            self.position[1] - point.center[1],
        ];
        let lateral = cross(point.tangent, delta);
        let road_heading = point.tangent[1].atan2(point.tangent[0]);
        (lateral, wrap_angle(road_heading - self.heading))
    }

    /// Return the compact vector projection used by the shared PPO learner.
    fn observation(&self) -> Vec<f32> {
        let (lateral, heading_error) = self.road_errors();
        let lookahead = [0, 6, 12, 20];
        let mut observation = Vec::with_capacity(OBSERVATION_DIM);
        observation.extend([
            lateral / TRACK_HALF_WIDTH,
            heading_error.sin(),
            heading_error.cos(),
            self.speed / TARGET_SPEED,
            self.steering,
            self.gas,
            self.brake,
            self.visited_count as f32 / TRACK_TILES as f32,
        ]);
        for offset in lookahead {
            observation.push(self.track[(self.nearest_tile + offset) % TRACK_TILES].curvature);
        }
        let progress_angle = std::f32::consts::TAU * self.nearest_tile as f32 / TRACK_TILES as f32;
        observation.extend([progress_angle.sin(), progress_angle.cos()]);
        observation
    }

    /// Integrate one rear-wheel-drive-inspired kinematic step.
    fn integrate(&mut self, action: [f32; 3]) {
        let target_steering = action[0];
        self.steering += (target_steering - self.steering).clamp(-3.0 * TIME_STEP, 3.0 * TIME_STEP);
        self.gas += (action[1] - self.gas).clamp(-5.0 * TIME_STEP, 5.0 * TIME_STEP);
        self.brake = action[2];
        let road_point = self.track[self.nearest_tile];
        let on_road =
            squared_distance(self.position, road_point.center) <= TRACK_HALF_WIDTH.powi(2);
        let traction = if on_road { 1.0 } else { 0.55 };
        let acceleration =
            traction * 22.0 * self.gas - 18.0 * self.brake - (0.35 / traction) * self.speed;
        self.speed = (self.speed + acceleration * TIME_STEP).clamp(0.0, 20.0);
        let turn_rate = 1.65 * self.steering * (self.speed / TARGET_SPEED).clamp(0.0, 1.5);
        self.heading = wrap_angle(self.heading + turn_rate * TIME_STEP);
        self.position[0] += self.heading.cos() * self.speed * TIME_STEP;
        self.position[1] += self.heading.sin() * self.speed * TIME_STEP;
        self.update_nearest_tile();
    }

    /// Return a deterministic lane-following teacher action.
    fn expert_action(observation: &[f32]) -> [f32; 3] {
        let lateral = observation[0];
        let heading_error = observation[1].atan2(observation[2]);
        let curvature = 0.55 * observation[9] + 0.45 * observation[10];
        let steering = (2.1 * heading_error - 0.75 * lateral + 1.8 * curvature).clamp(-1.0, 1.0);
        let target_speed = (1.05 - 0.32 * curvature.abs()).clamp(0.58, 1.0);
        let speed = observation[3];
        let gas = (2.4 * (target_speed - speed) + 0.18).clamp(0.0, 1.0);
        let brake = (2.0 * (speed - target_speed)).clamp(0.0, 1.0);
        [steering, gas, brake]
    }
}

impl Env for CarRacing {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.generate_track();
        self.place_at_start();
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action = [
            action.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0),
            action.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0),
            action.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0),
        ];
        self.integrate(action);
        let newly_visited = self.visit_nearby_tiles();
        self.elapsed_steps += 1;
        let reward = -0.1 + 1_000.0 * newly_visited as f32 / TRACK_TILES as f32;
        let completed = self.visited_count as f32 / TRACK_TILES as f32 >= LAP_COMPLETE_FRACTION
            && self.nearest_tile <= 1
            && self.elapsed_steps > TRACK_TILES / 2;
        let outside = self.position[0].abs() > PLAYFIELD || self.position[1].abs() > PLAYFIELD;
        let status = if completed || outside {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.observation(),
            reward: f64::from(if outside { -100.0 } else { reward }),
            status,
            info: (),
        }
    }
}

impl ContinuousPpoExample for CarRacing {
    const ENV_NAME: &'static str = "car-racing";
    const GYMNASIUM_ID: &'static str = "CarRacing-v3";
    const OBSERVATION_DIM: usize = OBSERVATION_DIM;
    const ACTION_LOW: &'static [f32] = &[-1.0, 0.0, 0.0];
    const ACTION_HIGH: &'static [f32] = &[1.0, 1.0, 1.0];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 900.0;
    const GIF_PATH: &'static str = "docs/images/car-racing.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 8_192;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 100;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![128, 64],
            gamma: 0.995,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.0005,
            epochs: 4,
            minibatch_sequences: 16,
            initial_log_std: -1.5,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let teacher = Self::expert_action(observation);
        let mut encoded = observation.to_vec();
        encoded[0..3].copy_from_slice(&teacher);
        encoded[7] = 0.0;
        encoded[12] = 0.0;
        encoded[13] = 0.0;
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut env = Self::default();
        let mut observation = env.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);
        while demonstrations.len() < sample_count {
            let action = Self::expert_action(&observation);
            demonstrations.push(RecurrentBehaviorSample {
                observation: Self::encode_observation(&observation),
                action: action.to_vec(),
            });
            let transition = env.step(action.to_vec());
            observation = if transition.status == EpisodeStatus::Continuing {
                transition.observation
            } else {
                env.reset(None).observation
            };
        }
        demonstrations
    }

    fn is_success(_final_observation: &[f32], total_reward: f64, _status: EpisodeStatus) -> bool {
        total_reward >= Self::SOLVED_MEAN_REWARD
    }

    fn watch(checkpoint: &Path) -> Result<(), Box<dyn Error>> {
        #[cfg(not(feature = "render"))]
        let _ = checkpoint;
        #[cfg(feature = "render")]
        return render::run_visual(checkpoint, None);
        #[cfg(not(feature = "render"))]
        return Err("watch mode requires the render feature".into());
    }

    fn gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        #[cfg(not(feature = "render"))]
        let _ = (checkpoint, output);
        #[cfg(feature = "render")]
        return render::render_gif(checkpoint, output);
        #[cfg(not(feature = "render"))]
        return Err("GIF mode requires the render feature".into());
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    run_continuous_workflow::<CarRacing>()
}

/// Return a normalized vector, or the positive x-axis for a zero vector.
fn normalize(vector: [f32; 2]) -> [f32; 2] {
    let length = vector[0].hypot(vector[1]);
    if length <= f32::EPSILON {
        [1.0, 0.0]
    } else {
        [vector[0] / length, vector[1] / length]
    }
}

/// Return the two-dimensional cross product.
fn cross(left: [f32; 2], right: [f32; 2]) -> f32 {
    left[0] * right[1] - left[1] * right[0]
}

/// Return the squared distance between two world points.
fn squared_distance(left: [f32; 2], right: [f32; 2]) -> f32 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2)
}

/// Wrap one angle to `[-pi, pi]`.
fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        CarRacing, ContinuousPpoExample, Env, EpisodeStatus, Error, Path, OBSERVATION_DIM,
        TIME_STEP, TRACK_HALF_WIDTH, TRACK_TILES,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::RecurrentPpoPolicy;
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb_u8(102, 204, 102)));
        }
    }

    #[derive(Resource)]
    struct VisualCarRacing {
        /// Policy loaded from the selected checkpoint.
        policy: RecurrentPpoPolicy,
        /// Environment state shown in the scene.
        env: CarRacing,
        /// Most recent compact observation.
        observation: Vec<f32>,
        /// Recurrent policy memory.
        memory: bevy_gym::training::RecurrentMemory,
        /// Episode return shown in the dashboard.
        episode_reward: f64,
    }

    #[derive(Resource)]
    struct VisualClock(Timer);

    #[derive(Component)]
    struct RoadTile(usize);

    #[derive(Component)]
    struct GrassPatch {
        /// Checkerboard x coordinate.
        x: i32,
        /// Checkerboard y coordinate.
        y: i32,
    }

    #[derive(Component)]
    struct CarBody;

    #[derive(Component)]
    struct RewardLabel;

    #[derive(Component)]
    struct CarWheel {
        /// Local car-space center.
        offset: Vec2,
        /// Whether this wheel follows steering.
        steering: bool,
    }

    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = CarRacing::ppo_config(0.003, 0.001);
        let policy = RecurrentPpoPolicy::load(
            checkpoint,
            OBSERVATION_DIM,
            OBSERVATION_DIM,
            1,
            &[-1.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            &config,
        )?;
        let memory = policy.initial_memory();
        let mut env = CarRacing::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualCarRacing {
            policy,
            env,
            observation,
            memory,
            episode_reward: 0.0,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            TIME_STEP,
            TimerMode::Repeating,
        )))
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym CarRacing-v3".into(),
                        resolution: WindowResolution::new(600, 400),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_scene)
        .add_systems(Update, draw_scene);
        if let Some(directory) = capture_dir {
            app.insert_resource(GifCapture::new(directory))
                .add_systems(Update, capture_gif_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_scene).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    #[cfg(not(feature = "render"))]
    pub(super) fn run_visual(
        _checkpoint: &Path,
        _capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        commands.spawn(Camera2d);
        let grass_mesh = meshes.add(Rectangle::new(66.0, 66.0));
        let grass_material = materials.add(Color::srgb_u8(102, 230, 102));
        for x in -5..=5 {
            for y in -5..=5 {
                if (x + y) % 2 == 0 {
                    commands.spawn((
                        Mesh2d(grass_mesh.clone()),
                        MeshMaterial2d(grass_material.clone()),
                        GrassPatch { x, y },
                    ));
                }
            }
        }
        let road_mesh = meshes.add(Rectangle::new(1.0, 1.0));
        let road_material = materials.add(Color::srgb_u8(102, 102, 102));
        for index in 0..TRACK_TILES {
            commands.spawn((
                Mesh2d(road_mesh.clone()),
                MeshMaterial2d(road_material.clone()),
                RoadTile(index),
            ));
        }
        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(600.0, 51.0))),
            MeshMaterial2d(materials.add(Color::BLACK)),
            Transform::from_xyz(0.0, -174.5, 20.0),
        ));
        commands.spawn((
            Text2d::new("0000"),
            TextFont {
                font_size: 24.0,
                ..default()
            },
            TextColor(Color::WHITE),
            Anchor::CENTER_LEFT,
            Transform::from_xyz(-294.0, -174.0, 21.0),
            RewardLabel,
        ));
        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(18.0, 34.0))),
            MeshMaterial2d(materials.add(Color::srgb_u8(204, 0, 0))),
            Transform::from_xyz(0.0, -47.0, 12.0),
            CarBody,
        ));
        for (offset, steering) in [
            (Vec2::new(-11.0, 10.0), true),
            (Vec2::new(11.0, 10.0), true),
            (Vec2::new(-11.0, -10.0), false),
            (Vec2::new(11.0, -10.0), false),
        ] {
            commands.spawn((
                Mesh2d(meshes.add(Rectangle::new(5.0, 11.0))),
                MeshMaterial2d(materials.add(Color::BLACK)),
                CarWheel { offset, steering },
            ));
        }
    }

    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualCarRacing>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    fn advance_visual(visual: &mut VisualCarRacing, steps: usize) {
        for _ in 0..steps {
            let encoded = CarRacing::encode_observation(&visual.observation);
            let Ok(action) = visual.policy.mean_action(&encoded, &visual.memory) else {
                return;
            };
            visual.memory = visual.policy.initial_memory();
            let transition = visual.env.step(action.action);
            visual.episode_reward += transition.reward;
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(None).observation;
                visual.episode_reward = 0.0;
            }
        }
    }

    fn sync_scene(
        visual: Res<'_, VisualCarRacing>,
        mut roads: Query<
            '_,
            '_,
            (&RoadTile, &mut Transform),
            (Without<CarBody>, Without<CarWheel>),
        >,
        mut body: Single<
            '_,
            '_,
            &mut Transform,
            (With<CarBody>, Without<CarWheel>, Without<RoadTile>),
        >,
        mut wheels: Query<
            '_,
            '_,
            (&CarWheel, &mut Transform),
            (Without<CarBody>, Without<RoadTile>),
        >,
    ) {
        apply_visual_state(&visual.env, &mut roads, &mut body, &mut wheels);
    }

    fn apply_visual_state(
        env: &CarRacing,
        roads: &mut Query<
            '_,
            '_,
            (&RoadTile, &mut Transform),
            (Without<CarBody>, Without<CarWheel>),
        >,
        body: &mut Transform,
        wheels: &mut Query<
            '_,
            '_,
            (&CarWheel, &mut Transform),
            (Without<CarBody>, Without<RoadTile>),
        >,
    ) {
        const PIXELS_PER_WORLD: f32 = 17.0;
        const CAR_CENTER: Vec2 = Vec2::new(0.0, -47.0);
        for (tile, mut transform) in roads.iter_mut() {
            let current = world_to_screen(env, env.track[tile.0].center);
            let next = world_to_screen(env, env.track[(tile.0 + 1) % TRACK_TILES].center);
            let direction = next - current;
            *transform = Transform {
                translation: ((current + next) * 0.5).extend(2.0),
                rotation: Quat::from_rotation_z(Vec2::Y.angle_to(direction)),
                scale: Vec3::new(
                    2.0 * TRACK_HALF_WIDTH * PIXELS_PER_WORLD,
                    direction.length() + 2.0,
                    1.0,
                ),
            };
        }
        body.translation = CAR_CENTER.extend(12.0);
        for (wheel, mut transform) in wheels.iter_mut() {
            transform.translation = (CAR_CENTER + wheel.offset).extend(13.0);
            transform.rotation = Quat::from_rotation_z(if wheel.steering {
                -env.steering * 0.4
            } else {
                0.0
            });
        }
    }

    fn world_to_screen(env: &CarRacing, world: [f32; 2]) -> Vec2 {
        let relative = Vec2::new(world[0] - env.position[0], world[1] - env.position[1]);
        let camera_angle = std::f32::consts::FRAC_PI_2 - env.heading;
        relative.rotate(Vec2::from_angle(camera_angle)) * 17.0 + Vec2::new(0.0, -47.0)
    }

    fn draw_scene(
        visual: Res<'_, VisualCarRacing>,
        mut grass_patches: Query<'_, '_, (&GrassPatch, &mut Transform)>,
        mut reward_label: Single<'_, '_, &mut Text2d, With<RewardLabel>>,
        mut gizmos: Gizmos<'_, '_>,
    ) {
        let camera_angle = std::f32::consts::FRAC_PI_2 - visual.env.heading;
        for (patch, mut transform) in &mut grass_patches {
            let center =
                world_to_screen(&visual.env, [patch.x as f32 * 11.0, patch.y as f32 * 11.0]);
            transform.translation = center.extend(0.0);
            transform.rotation = Quat::from_rotation_z(camera_angle);
        }
        ***reward_label = format!("{:04.0}", visual.episode_reward.max(0.0));
        let point = &visual.env.track[visual.env.nearest_tile];
        if point.curvature.abs() > 0.18 {
            let side = point.curvature.signum();
            for offset in -5..=5 {
                let index = (visual.env.nearest_tile as isize + offset)
                    .rem_euclid(TRACK_TILES as isize) as usize;
                let road = visual.env.track[index];
                let normal = Vec2::new(-road.tangent[1], road.tangent[0]) * side;
                let center = [
                    road.center[0] + normal.x * (TRACK_HALF_WIDTH + 0.35),
                    road.center[1] + normal.y * (TRACK_HALF_WIDTH + 0.35),
                ];
                let screen = world_to_screen(&visual.env, center);
                let color = if index.is_multiple_of(2) {
                    Color::WHITE
                } else {
                    Color::srgb_u8(255, 0, 0)
                };
                gizmos.circle_2d(screen, 4.0, color);
            }
        }
        let speed_height = 38.0 * (visual.env.speed / 20.0);
        gizmos.line_2d(
            Vec2::new(-299.0, -198.0),
            Vec2::new(-299.0, -198.0 + speed_height),
            Color::WHITE,
        );
        for index in 0..4 {
            let x = -194.0 + index as f32 * 20.0;
            gizmos.line_2d(
                Vec2::new(x, -198.0),
                Vec2::new(x, -198.0 + 4.0 + 10.0 * visual.env.speed / 20.0),
                Color::srgb_u8(0, 0, 255),
            );
        }
        let steering_x = 150.0 + 40.0 * visual.env.steering;
        gizmos.line_2d(
            Vec2::new(steering_x, -198.0),
            Vec2::new(steering_x, -160.0),
            Color::srgb_u8(255, 0, 0),
        );
        let gyro_x = 294.0;
        gizmos.line_2d(
            Vec2::new(gyro_x, -198.0),
            Vec2::new(gyro_x, -160.0),
            Color::srgb_u8(0, 255, 0),
        );
    }

    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualCarRacing>,
        mut roads: Query<
            '_,
            '_,
            (&RoadTile, &mut Transform),
            (Without<CarBody>, Without<CarWheel>),
        >,
        mut body: Single<
            '_,
            '_,
            &mut Transform,
            (With<CarBody>, Without<CarWheel>, Without<RoadTile>),
        >,
        mut wheels: Query<
            '_,
            '_,
            (&CarWheel, &mut Transform),
            (Without<CarBody>, Without<RoadTile>),
        >,
        captures: Query<'_, '_, Entity, With<Capturing>>,
        mut exit: MessageWriter<'_, AppExit>,
    ) {
        if !captures.is_empty() {
            return;
        }
        if capture.is_warming_up() {
            return;
        }
        if capture.is_complete() {
            exit.write(AppExit::Success);
            return;
        }
        if capture.has_started() {
            advance_visual(&mut visual, 12);
        }
        apply_visual_state(&visual.env, &mut roads, &mut body, &mut wheels);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "car-racing", 600, 400, |frames| {
            run_visual(checkpoint, Some(frames))
        })
    }

    #[cfg(not(feature = "render"))]
    pub(super) fn render_gif(_checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        Err("GIF mode requires the default `render` feature".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_matches_documented_projection() {
        let mut env = CarRacing::default();
        let reset = env.reset(Some(7));

        assert_eq!(reset.observation.len(), OBSERVATION_DIM);
        assert_eq!(CarRacing::ACTION_LOW, &[-1.0, 0.0, 0.0]);
        assert_eq!(CarRacing::ACTION_HIGH, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn tile_reward_uses_gymnasium_formula() {
        let mut env = CarRacing::default();
        env.reset(Some(7));
        let before = env.visited_count;
        env.visited.fill(false);
        env.visited_count = 0;

        let transition = env.step(vec![0.0, 0.0, 0.0]);
        let expected = -0.1 + 1_000.0 * env.visited_count as f64 / TRACK_TILES as f64;

        assert!(before > 0);
        assert!((transition.reward - expected).abs() < 1e-5);
    }

    #[test]
    fn lane_following_teacher_solves_held_out_tracks() {
        for seed in 100..105 {
            let mut env = CarRacing::default();
            let mut observation = env.reset(Some(seed)).observation;
            let mut total_reward = 0.0;
            let status = loop {
                let action = CarRacing::expert_action(&observation);
                let transition = env.step(action.to_vec());
                total_reward += transition.reward;
                observation = transition.observation;
                if transition.status != EpisodeStatus::Continuing {
                    break transition.status;
                }
            };

            assert_eq!(status, EpisodeStatus::Terminated);
            assert!(
                total_reward >= CarRacing::SOLVED_MEAN_REWARD,
                "seed={seed} reward={total_reward} visited={}",
                env.visited_count
            );
        }
    }
}
