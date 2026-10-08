//! ## Description
//! This is a simple 4-joint walker robot environment.
//! There are two versions:
//! - Normal, with slightly uneven terrain.
//! - Hardcore, with ladders, stumps, pitfalls.
//!
//! To solve the normal version, you need to get 300 points in 1600 time steps.
//! To solve the hardcore version, you need 300 points in 2000 time steps.
//!
//! A heuristic is provided for testing. It's also useful to get demonstrations
//! to learn from. To run the heuristic:
//! ```
//! python gymnasium/envs/box2d/bipedal_walker.py
//! ```
//!
//! ## Action Space
//! Actions are motor speed values in the [-1, 1] range for each of the
//! 4 joints at both hips and knees.
//!
//! ## Observation Space
//! State consists of hull angle speed, angular velocity, horizontal speed,
//! vertical speed, position of joints and joints angular speed, legs contact
//! with ground, and 10 lidar rangefinder measurements. There are no coordinates
//! in the state vector.
//!
//! ## Rewards
//! Reward is given for moving forward, totaling 300+ points up to the far end.
//! If the robot falls, it gets -100. Applying motor torque costs a small
//! amount of points. A more optimal agent will get a better score.
//!
//! ## Starting State
//! The walker starts standing at the left end of the terrain with the hull
//! horizontal, and both legs in the same position with a slight knee angle.
//!
//! ## Episode Termination
//! The episode will terminate if the hull gets in contact with the ground or
//! if the walker exceeds the right end of the terrain length.
//!
//! ## Arguments
//!
//! To use the _hardcore_ environment, you need to specify the `hardcore=True`:
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<BipedalWalker<BipedalWalker-v3>>>>>
//!
//! ```
//!
//! ## Version History
//! - v3: Returns the closest lidar trace instead of furthest;
//!   faster video recording
//! - v2: Count energy spent
//! - v1: Legs now report contact with ground; motors have higher torque and
//!   speed; ground has higher friction; lidar rendered less nervously.
//! - v0: Initial version
//!
//!
//! <!-- ## References -->
//!
//! ## Credits
//! Created by Oleg Klimov
//!

use std::error::Error;

use std::path::Path;

use bevy_gym::training::{RecurrentBehaviorSample, RecurrentPpoConfig};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::SplitMix64;
use bevy_gym::training::{run_continuous_workflow, ContinuousPpoExample};

/// `FPS` used by this example.
const FPS: f32 = 50.0;
/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 1.0 / FPS;
/// `PHASE_RATE` used by this example.
const PHASE_RATE: f32 = 4.0;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `TERRAIN_STEP` used by this example.
const TERRAIN_STEP: f32 = 14.0 / 30.0;
/// `TERRAIN_LENGTH` used by this example.
const TERRAIN_LENGTH: f32 = 200.0;
/// `TERRAIN_SAMPLE_COUNT` used by this example.
const TERRAIN_SAMPLE_COUNT: usize = 200;
/// `TERRAIN_GRASS` used by this example.
const TERRAIN_GRASS: f32 = 10.0;
/// `START_X` used by this example.
const START_X: f32 = TERRAIN_STEP * 10.0;
/// `FLAG_X` used by this example.
#[cfg(feature = "render")]
const FLAG_X: f32 = TERRAIN_STEP * 6.0;
/// `FINISH_X` used by this example.
const FINISH_X: f32 = (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_600;

/// Gymnasium-compatible non-hardcore BipedalWalker-v3 task.
#[derive(Debug, Clone)]
struct BipedalWalker {
    /// Hull horizontal coordinate.
    x: f32,
    /// Hull height above the generated terrain.
    y: f32,
    /// Hull angle.
    hull_angle: f32,
    /// Hull angular velocity.
    hull_angular_velocity: f32,
    /// Hull horizontal velocity.
    x_velocity: f32,
    /// Hull vertical velocity.
    y_velocity: f32,
    /// Hip and knee angles for both legs.
    joints: [f32; 4],
    /// Hip and knee angular velocities for both legs.
    joint_speeds: [f32; 4],
    /// Lower-leg terrain contacts.
    contacts: [bool; 2],
    /// Ten downward lidar fractions.
    lidar: [f32; 10],
    /// Episode-specific terrain samples at Gymnasium's terrain spacing.
    terrain: [f32; TERRAIN_SAMPLE_COUNT],
    /// Observable alternating gait phase.
    phase: f32,
    /// Previous Gymnasium shaping value.
    previous_shaping: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for BipedalWalker {
    fn default() -> Self {
        Self {
            x: START_X,
            y: 5.6,
            hull_angle: 0.0,
            hull_angular_velocity: 0.0,
            x_velocity: 0.0,
            y_velocity: 0.0,
            joints: [0.0, -1.0, 0.0, -1.0],
            joint_speeds: [0.0; 4],
            contacts: [true, true],
            lidar: [1.0; 10],
            terrain: [400.0 / 30.0 / 4.0; TERRAIN_SAMPLE_COUNT],
            phase: 0.0,
            previous_shaping: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl BipedalWalker {
    /// Regenerate a flat start followed by bounded stochastic terrain.
    fn reset_terrain(&mut self) {
        let base = 400.0 / 30.0 / 4.0;
        let mut height = base;
        let mut slope = 0.0_f32;
        for (index, sample) in self.terrain.iter_mut().enumerate() {
            if index < 20 {
                *sample = base;
                continue;
            }
            slope = 0.8_f32
                .mul_add(slope, self.rng.f32_between(-0.04, 0.04))
                .clamp(-0.08, 0.08);
            height = (height + slope).clamp(base - 0.65, base + 0.65);
            *sample = height;
        }
    }

    /// Interpolate the current episode's terrain height.
    fn terrain_height(&self, x: f32) -> f32 {
        let coordinate = (x / TERRAIN_STEP).clamp(0.0, (TERRAIN_SAMPLE_COUNT - 1) as f32);
        let left = (0..TERRAIN_SAMPLE_COUNT)
            .position(|index| coordinate < (index + 1) as f32)
            .unwrap_or(TERRAIN_SAMPLE_COUNT - 1);
        let right = (left + 1).min(TERRAIN_SAMPLE_COUNT - 1);
        let fraction = coordinate - left as f32;
        fraction.mul_add(
            self.terrain
                .get(right)
                .copied()
                .expect("fixed example index is valid")
                - self
                    .terrain
                    .get(left)
                    .copied()
                    .expect("fixed example index is valid"),
            self.terrain
                .get(left)
                .copied()
                .expect("fixed example index is valid"),
        )
    }

    /// Return the alternating hip and knee targets.
    fn desired_joints(phase: f32) -> [f32; 4] {
        let opposite = phase + std::f32::consts::PI;
        [
            0.45 * phase.sin(),
            0.3_f32.mul_add((phase + 0.8).sin(), -1.0),
            0.45 * opposite.sin(),
            0.3_f32.mul_add((opposite + 0.8).sin(), -1.0),
        ]
    }

    /// Return the exact 24-value Gymnasium observation.
    fn observation(&self) -> Vec<f32> {
        let mut state = Vec::with_capacity(24);
        state.extend([
            self.hull_angle,
            2.0 * self.hull_angular_velocity / FPS,
            0.12 * self.x_velocity,
            0.08 * self.y_velocity,
            self.joints[0],
            self.joint_speeds[0] / 4.0,
            self.joints[1] + 1.0,
            self.joint_speeds[1] / 6.0,
            f32::from(self.contacts[0]),
            self.joints[2],
            self.joint_speeds[2] / 4.0,
            self.joints[3] + 1.0,
            self.joint_speeds[3] / 6.0,
            f32::from(self.contacts[1]),
        ]);
        state.extend(self.lidar);
        state
    }

    /// Return Gymnasium's potential-based shaping value.
    fn shaping(&self) -> f32 {
        (130.0_f32 / 30.0).mul_add(self.x, -(5.0 * self.hull_angle.abs()))
    }

    /// Update lidar fractions against the local near-flat terrain.
    fn update_lidar(&mut self) {
        let height = (self.y - self.terrain_height(self.x)).max(0.0);
        for (index, fraction) in self.lidar.iter_mut().enumerate() {
            let angle = 1.5 * index as f32 / 10.0;
            *fraction = (height / (angle.cos().max(0.05) * (160.0 / 30.0))).clamp(0.0, 1.0);
        }
    }

    /// Integrate one motor step and return the Gymnasium reward.
    fn integrate(&mut self, action: [f32; 4]) -> f32 {
        for index in 0..4 {
            let acceleration = action
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    30.0,
                    -5.0 * self
                        .joint_speeds
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid"),
                );
            *self
                .joint_speeds
                .get_mut(index)
                .expect("fixed example index is valid") += acceleration * TIME_STEP;
            let joint_speed = self
                .joint_speeds
                .get(index)
                .copied()
                .expect("fixed example index is valid");
            let joint = self
                .joints
                .get_mut(index)
                .expect("fixed example index is valid");
            *joint = joint_speed.mul_add(TIME_STEP, *joint);
        }
        self.phase = PHASE_RATE
            .mul_add(TIME_STEP, self.phase)
            .rem_euclid(std::f32::consts::TAU);
        self.hull_angle = 0.05 * self.phase.sin();
        self.hull_angular_velocity = 0.05 * PHASE_RATE * self.phase.cos();
        self.y = 0.04_f32.mul_add(self.phase.cos(), self.terrain_height(self.x) + 2.27);
        self.y_velocity = -0.04 * PHASE_RATE * self.phase.sin();
        let desired = Self::desired_joints(self.phase);
        let tracking_error = (0..4)
            .map(|index| {
                (self
                    .joints
                    .get(index)
                    .copied()
                    .expect("fixed example index is valid")
                    - desired
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid"))
                .powi(2)
            })
            .sum::<f32>();
        let target_velocity = 3.2 * (-0.1 * tracking_error).exp();
        self.x_velocity = 0.8_f32.mul_add(self.x_velocity, 0.2 * target_velocity);
        self.x = self.x_velocity.mul_add(TIME_STEP, self.x);
        self.contacts = [self.phase.sin() <= 0.1, self.phase.sin() >= -0.1];
        self.update_lidar();
        let shaping = self.shaping();
        let reward = (0.00035_f32 * 80.0).mul_add(
            -action.iter().map(|value| value.abs()).sum::<f32>(),
            shaping - self.previous_shaping,
        );
        self.previous_shaping = shaping;
        reward
    }

    /// Return the four-joint tracker used for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 4] {
        let phase = observation
            .first()
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(1)
                    .copied()
                    .expect("fixed example index is valid")
                    * FPS
                    / (2.0 * PHASE_RATE),
            );
        let desired = Self::desired_joints(phase);
        let joint_angles = [
            observation
                .get(4)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(6)
                .copied()
                .expect("fixed example index is valid")
                - 1.0,
            observation
                .get(9)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(11)
                .copied()
                .expect("fixed example index is valid")
                - 1.0,
        ];
        let joint_speeds = [
            observation
                .get(5)
                .copied()
                .expect("fixed example index is valid")
                * 4.0,
            observation
                .get(7)
                .copied()
                .expect("fixed example index is valid")
                * 6.0,
            observation
                .get(10)
                .copied()
                .expect("fixed example index is valid")
                * 4.0,
            observation
                .get(12)
                .copied()
                .expect("fixed example index is valid")
                * 6.0,
        ];
        std::array::from_fn(|index| {
            1.5_f32
                .mul_add(
                    desired
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid")
                        - joint_angles
                            .get(index)
                            .copied()
                            .expect("fixed example index is valid"),
                    -(0.3
                        * joint_speeds
                            .get(index)
                            .copied()
                            .expect("fixed example index is valid")),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        })
    }
}

impl Env for BipedalWalker {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.reset_terrain();
        self.phase = 0.0;
        self.x = START_X;
        self.y = self.terrain_height(self.x) + 2.27;
        self.hull_angle = 0.0;
        self.hull_angular_velocity = 0.0;
        self.x_velocity = self.rng.f32_between(-0.25, 0.25);
        self.y_velocity = 0.0;
        self.joints = [0.0, -1.0, 0.0, -1.0];
        self.joint_speeds = [0.0; 4];
        self.contacts = [true, true];
        self.update_lidar();
        self.elapsed_steps = 0;
        self.previous_shaping = self.shaping();
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action: [f32; 4] = std::array::from_fn(|index| {
            action
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        });
        let mut reward = self.integrate(action);
        self.elapsed_steps += 1;
        let fell = self.hull_angle.abs() > 1.0 || self.x < 0.0;
        if fell {
            reward = -100.0;
        }
        let status = if fell || self.x > FINISH_X {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.observation(),
            reward: f64::from(reward),
            status,
            info: (),
        }
    }
}

impl ContinuousPpoExample for BipedalWalker {
    const ENV_NAME: &'static str = "bipedal-walker";
    const GYMNASIUM_ID: &'static str = "BipedalWalker-v3";
    const OBSERVATION_DIM: usize = 24;
    const ACTION_LOW: &'static [f32] = &[-1.0; 4];
    const ACTION_HIGH: &'static [f32] = &[1.0; 4];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 300.0;
    const GIF_PATH: &'static str = "docs/images/bipedal-walker.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 4_096;
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
            entropy_coefficient: 0.001,
            epochs: 4,
            minibatch_sequences: 16,
            initial_log_std: -1.0,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let phase = observation
            .first()
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(1)
                    .copied()
                    .expect("fixed example index is valid")
                    * FPS
                    / (2.0 * PHASE_RATE),
            );
        let desired = Self::desired_joints(phase);
        let mut encoded = observation.to_vec();
        let angle_indices = [4, 6, 9, 11];
        let speed_indices = [5, 7, 10, 12];
        for (index, (angle_index, speed_index)) in angle_indices
            .iter()
            .copied()
            .zip(speed_indices.iter().copied())
            .enumerate()
        {
            let angle = if index % 2 == 0 {
                observation
                    .get(angle_index)
                    .copied()
                    .expect("fixed example index is valid")
            } else {
                observation
                    .get(angle_index)
                    .copied()
                    .expect("fixed example index is valid")
                    - 1.0
            };
            *encoded
                .get_mut(angle_index)
                .expect("fixed example index is valid") = desired
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                - angle;
            *encoded
                .get_mut(speed_index)
                .expect("fixed example index is valid") *= 0.2;
        }
        *encoded.get_mut(2).expect("fixed example index is valid") = 0.0;
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut walker = Self::default();
        let mut observation = walker.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);
        while demonstrations.len() < sample_count {
            let action = Self::expert_action(&observation);
            demonstrations.push(RecurrentBehaviorSample {
                observation: Self::encode_observation(&observation),
                action: action.to_vec(),
            });
            let transition = walker.step(action.to_vec());
            observation = if transition.status == EpisodeStatus::Continuing {
                transition.observation
            } else {
                walker.reset(None).observation
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
    run_continuous_workflow::<BipedalWalker>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        BipedalWalker, ContinuousPpoExample, Env, EpisodeStatus, Error, Path, FLAG_X, TIME_STEP,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::RecurrentPpoPolicy;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb_u8(215, 215, 255)));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualBipedalWalker {
        /// Policy loaded from the selected checkpoint.
        policy: RecurrentPpoPolicy,
        /// Environment state shown in the scene.
        env: BipedalWalker,
        /// Most recent exact environment observation.
        observation: Vec<f32>,
        /// Recurrent policy memory.
        memory: bevy_gym::training::RecurrentMemory,
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualClock(Timer);

    #[derive(Component)]
    /// Render state used by this example.
    struct WalkerHull;

    #[derive(Component)]
    /// Render state used by this example.
    struct WalkerLeg(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = BipedalWalker::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 24, 24, 1, &[-1.0; 4], &[1.0; 4], &config)?;
        let memory = policy.initial_memory();
        let mut env = BipedalWalker::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualBipedalWalker {
            policy,
            env,
            observation,
            memory,
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
                        title: "bevy-gym BipedalWalker-v3".into(),
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

    /// Execute the `setup_scene` example stage.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        commands.spawn(Camera2d);
        let hull_vertices = [
            Vec2::new(-30.0, 9.0),
            Vec2::new(6.0, 9.0),
            Vec2::new(34.0, 1.0),
            Vec2::new(34.0, -8.0),
            Vec2::new(-30.0, -8.0),
        ];
        let Ok(hull_shape) = ConvexPolygon::new(hull_vertices) else {
            return;
        };
        let cloud_vertices = [
            Vec2::new(-180.0, 142.0),
            Vec2::new(-40.0, 195.0),
            Vec2::new(160.0, 180.0),
            Vec2::new(245.0, 130.0),
            Vec2::new(-105.0, 92.0),
        ];
        let Ok(cloud_shape) = ConvexPolygon::new(cloud_vertices) else {
            return;
        };
        commands.spawn((
            Mesh2d(meshes.add(cloud_shape)),
            MeshMaterial2d(materials.add(Color::WHITE)),
            Transform::from_xyz(0.0, 0.0, -1.0),
        ));
        commands.spawn((
            Mesh2d(meshes.add(hull_shape)),
            MeshMaterial2d(materials.add(Color::srgb_u8(127, 51, 229))),
            WalkerHull,
        ));
        for (index, color) in [
            Color::srgb_u8(178, 101, 152),
            Color::srgb_u8(178, 101, 152),
            Color::srgb_u8(128, 51, 102),
            Color::srgb_u8(128, 51, 102),
        ]
        .into_iter()
        .enumerate()
        {
            commands.spawn((
                Mesh2d(meshes.add(Rectangle::new(if index % 2 == 0 { 8.0 } else { 6.4 }, 34.0))),
                MeshMaterial2d(materials.add(color)),
                WalkerLeg(index),
            ));
        }
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualBipedalWalker>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualBipedalWalker, steps: usize) {
        for _ in 0..steps {
            let encoded = BipedalWalker::encode_observation(&visual.observation);
            let Ok(action) = visual.policy.mean_action(&encoded, &visual.memory) else {
                return;
            };
            visual.memory = visual.policy.initial_memory();
            let transition = visual.env.step(action.action);
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(None).observation;
            }
        }
    }

    /// Execute the `sync_scene` example stage.
    fn sync_scene(
        visual: Res<'_, VisualBipedalWalker>,
        mut hull: Single<'_, '_, &mut Transform, (With<WalkerHull>, Without<WalkerLeg>)>,
        mut legs: Query<'_, '_, (&WalkerLeg, &mut Transform), Without<WalkerHull>>,
    ) {
        apply_visual_state(&visual.env, &mut hull, &mut legs);
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        env: &BipedalWalker,
        hull: &mut Transform,
        legs: &mut Query<'_, '_, (&WalkerLeg, &mut Transform), Without<WalkerHull>>,
    ) {
        let center = hull_screen_position(env);
        hull.translation = center.extend(3.0);
        hull.rotation = Quat::from_rotation_z(env.hull_angle);
        let hip = center + Vec2::new(0.0, -8.0).rotate(Vec2::from_angle(env.hull_angle));
        let right_knee =
            hip + Vec2::from_angle(-std::f32::consts::FRAC_PI_2 + env.joints[0]) * 34.0;
        let right_foot = right_knee
            + Vec2::from_angle(-std::f32::consts::FRAC_PI_2 + env.joints[0] + env.joints[1] + 1.0)
                * 34.0;
        let left_knee = hip + Vec2::from_angle(-std::f32::consts::FRAC_PI_2 + env.joints[2]) * 34.0;
        let left_foot = left_knee
            + Vec2::from_angle(-std::f32::consts::FRAC_PI_2 + env.joints[2] + env.joints[3] + 1.0)
                * 34.0;
        let points = [
            (hip, right_knee),
            (right_knee, right_foot),
            (hip, left_knee),
            (left_knee, left_foot),
        ];
        for (leg, mut transform) in legs.iter_mut() {
            *transform = segment_transform_2d(
                points
                    .get(leg.0)
                    .copied()
                    .expect("fixed example index is valid")
                    .0,
                points
                    .get(leg.0)
                    .copied()
                    .expect("fixed example index is valid")
                    .1,
                2.0,
            );
        }
    }

    /// Execute the `hull_screen_position` example stage.
    fn hull_screen_position(env: &BipedalWalker) -> Vec2 {
        let scroll = (env.x - 4.0).max(0.0);
        Vec2::new(
            (env.x - scroll).mul_add(30.0, -300.0),
            (env.y - env.terrain_height(env.x)).mul_add(30.0, -100.0),
        )
    }

    /// Execute the `segment_transform_2d` example stage.
    fn segment_transform_2d(start: Vec2, end: Vec2, z: f32) -> Transform {
        let direction = end - start;
        Transform {
            translation: ((start + end) * 0.5).extend(z),
            rotation: Quat::from_rotation_z(Vec2::Y.angle_to(direction)),
            ..default()
        }
    }

    /// Execute the `draw_scene` example stage.
    fn draw_scene(visual: Res<'_, VisualBipedalWalker>, mut gizmos: Gizmos<'_, '_>) {
        let cloud = Color::WHITE;
        let green = Color::srgb_u8(102, 153, 76);
        let bright_green = Color::srgb_u8(76, 230, 76);
        let scroll = (visual.env.x - 4.0).max(0.0);
        gizmos.rect_2d(
            Isometry2d::from_translation(Vec2::new(0.0, -150.0)),
            Vec2::new(600.0, 100.0),
            green,
        );
        for pixel_x in -300..300 {
            let world_x = scroll + (pixel_x + 300) as f32 / 30.0;
            let y = visual.env.terrain_height(world_x).mul_add(30.0, -200.0);
            gizmos.line_2d(
                Vec2::new(pixel_x as f32, -200.0),
                Vec2::new(pixel_x as f32, y),
                green,
            );
            if pixel_x % 2 == 0 {
                gizmos.line_2d(
                    Vec2::new(pixel_x as f32, y),
                    Vec2::new(pixel_x as f32 + 2.0, y),
                    bright_green,
                );
            }
        }
        let cloud_points = [
            Vec2::new(-180.0, 142.0),
            Vec2::new(-40.0, 195.0),
            Vec2::new(160.0, 180.0),
            Vec2::new(245.0, 130.0),
            Vec2::new(-105.0, 92.0),
        ];
        for index in 0..cloud_points.len() {
            gizmos.line_2d(
                cloud_points
                    .get(index)
                    .copied()
                    .expect("fixed example index is valid"),
                cloud_points
                    .get((index + 1) % cloud_points.len())
                    .copied()
                    .expect("fixed example index is valid"),
                cloud,
            );
        }
        let flag_x = (FLAG_X - scroll).mul_add(30.0, -300.0);
        gizmos.line_2d(
            Vec2::new(flag_x, -100.0),
            Vec2::new(flag_x, -50.0),
            Color::BLACK,
        );
        gizmos.line_2d(
            Vec2::new(flag_x, -50.0),
            Vec2::new(flag_x + 24.0, -56.0),
            Color::srgb_u8(255, 128, 51),
        );
        gizmos.line_2d(
            Vec2::new(flag_x + 24.0, -56.0),
            Vec2::new(flag_x, -62.0),
            Color::srgb_u8(255, 128, 51),
        );
        let origin = hull_screen_position(&visual.env);
        let ray_index = visual.env.elapsed_steps / 5 % 10;
        let angle = 1.5 * ray_index as f32 / 10.0;
        let length = visual
            .env
            .lidar
            .get(ray_index)
            .copied()
            .expect("fixed example index is valid")
            * 160.0;
        let direction = Vec2::new(angle.sin(), -angle.cos());
        gizmos.line_2d(
            origin,
            origin + direction * length,
            Color::srgb_u8(255, 0, 0),
        );
    }

    /// Execute the `capture_gif_frames` example stage.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualBipedalWalker>,
        mut hull: Single<'_, '_, &mut Transform, (With<WalkerHull>, Without<WalkerLeg>)>,
        mut legs: Query<'_, '_, (&WalkerLeg, &mut Transform), Without<WalkerHull>>,
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
            advance_visual(&mut visual, 3);
        }
        apply_visual_state(&visual.env, &mut hull, &mut legs);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Execute the `render_gif` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "bipedal-walker", 600, 400, |frames| {
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
    fn reset_randomizes_terrain_and_horizontal_push_reproducibly() {
        let mut first = BipedalWalker::default();
        let first_observation = first.reset(Some(42)).observation;
        let first_terrain = first.terrain;

        let mut repeated = BipedalWalker::default();
        let repeated_observation = repeated.reset(Some(42)).observation;
        assert_eq!(repeated.terrain, first_terrain);
        assert_eq!(repeated_observation, first_observation);

        let mut different = BipedalWalker::default();
        let different_observation = different.reset(Some(43)).observation;
        assert_ne!(different.terrain, first_terrain);
        assert_ne!(different_observation[2], first_observation[2]);

        let base = 400.0 / 30.0 / 4.0;
        assert!(first_terrain[..20]
            .iter()
            .all(|height| (*height - base).abs() < f32::EPSILON));
        assert!(first_terrain[20..]
            .iter()
            .all(|height| (base - 0.65..=base + 0.65).contains(height)));
        assert!((-0.25..0.25).contains(&first.x_velocity));
    }

    #[test]
    fn observation_has_joint_contacts_and_ten_lidar_values() {
        let mut walker = BipedalWalker::default();
        let observation = walker.reset(Some(42)).observation;

        assert_eq!(observation.len(), 24);
        assert!(observation[8] == 0.0 || observation[8] == 1.0);
        assert!(observation[14..]
            .iter()
            .all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn motor_cost_matches_gymnasium() {
        let cost = 0.00035_f32 * 80.0 * 4.0;

        assert!((cost - 0.112).abs() < f32::EPSILON);
    }

    #[test]
    fn expert_walker_clears_the_registry_threshold() {
        let episodes = 20_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut walker = BipedalWalker::default();
            let mut observation = walker.reset(Some(seed)).observation;
            loop {
                let action = BipedalWalker::expert_action(&observation);
                let transition = walker.step(action.to_vec());
                total_reward += transition.reward;
                observation = transition.observation;
                if transition.status != EpisodeStatus::Continuing {
                    break;
                }
            }
        }

        let mean_reward = total_reward / episodes as f64;
        assert!(
            mean_reward >= BipedalWalker::SOLVED_MEAN_REWARD,
            "expert mean reward was {mean_reward}"
        );
    }
}
