//! ## Description
//!
//! The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
//! at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
//! that can be applied to the car in either direction. The goal of the MDP is to strategically
//! accelerate the car to reach the goal state on top of the right hill. There are two versions
//! of the mountain car domain in gymnasium: one with discrete actions and one with continuous.
//! This version is the one with continuous actions.
//!
//! This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
//!
//! ```
//! @TECHREPORT{Moore90efficientmemory-based,
//!     author = {Andrew William Moore},
//!     title = {Efficient Memory-based Learning for Robot Control},
//!     institution = {University of Cambridge},
//!     year = {1990}
//! }
//! ```
//!
//! ## Observation Space
//!
//! The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
//!
//! | Num | Observation                          | Min   | Max  | Unit          |
//! |-----|--------------------------------------|-------|------|---------------|
//! | 0   | position of the car along the x-axis | -1.2  | 0.6  | position (m)  |
//! | 1   | velocity of the car                  | -0.07 | 0.07 | velocity (v)  |
//!
//! ## Action Space
//!
//! The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
//! The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.
//!
//! ## Transition Dynamics:
//!
//! Given an action, the mountain car follows the following transition dynamics:
//!
//! *velocity<sub>t+1</sub> = velocity<sub>t</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*
//!
//! *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
//!
//! where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
//! The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
//! The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].
//!
//! ## Reward
//!
//! A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
//! taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
//! is added to the negative reward for that timestep.
//!
//! ## Starting State
//!
//! The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
//! The starting velocity of the car is always assigned to 0.
//!
//! ## Episode End
//!
//! The episode ends if either of the following happens:
//! 1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
//! 2. Truncation: The length of the episode is 999.
//!
//! ## Arguments
//!
//! Continuous Mountain Car has two parameters for `gymnasium.make` with `render_mode` and `goal_velocity`.
//! On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array", goal_velocity=0.1)  # default goal_velocity=0
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<Continuous_MountainCarEnv<MountainCarContinuous-v0>>>>>
//! >>> env.reset(seed=123, options={"low": -0.7, "high": -0.5})  # default low=-0.6, high=-0.4
//! (array([-0.5635296,  0.       ], dtype=float32), {})
//!
//! ```
//!
//! ## Version History
//!
//! * v0: Initial versions release

use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::RecurrentPpoConfig;
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::{run_continuous_workflow, ContinuousPpoExample};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use serde_json as _;

/// `MIN_POSITION` used by this example.
const MIN_POSITION: f32 = -1.2;
/// `MAX_POSITION` used by this example.
const MAX_POSITION: f32 = 0.6;
/// `MAX_SPEED` used by this example.
const MAX_SPEED: f32 = 0.07;
/// `GOAL_POSITION` used by this example.
const GOAL_POSITION: f32 = 0.45;
/// `POWER` used by this example.
const POWER: f32 = 0.0015;
/// `GRAVITY` used by this example.
const GRAVITY: f32 = 0.0025;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 999;

/// Exact default Gymnasium MountainCarContinuous-v0 dynamics.
#[derive(Debug, Clone)]
struct ContinuousMountainCar {
    /// Position and velocity.
    state: [f32; 2],
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset generator.
    rng: SplitMix64,
}

impl Default for ContinuousMountainCar {
    fn default() -> Self {
        Self {
            state: [-0.5, 0.0],
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl ContinuousMountainCar {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    const fn from_state(position: f32, velocity: f32) -> Self {
        Self {
            state: [position, velocity],
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }

    /// Return Gymnasium's sinusoidal terrain height.
    fn height(position: f32) -> f32 {
        (3.0 * position).sin().mul_add(0.45, 0.55)
    }
}

impl Env for ContinuousMountainCar {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state = [self.rng.f32_between(-0.6, -0.4), 0.0];
        self.elapsed_steps = 0;
        Reset {
            observation: self.state.to_vec(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let raw_force = action.first().copied().unwrap_or(0.0);
        let force = raw_force.clamp(-1.0, 1.0);
        let [position, velocity] = self.state;
        let velocity = force
            .mul_add(POWER, (3.0 * position).cos().mul_add(-GRAVITY, velocity))
            .clamp(-MAX_SPEED, MAX_SPEED);
        let next_position = (position + velocity).clamp(MIN_POSITION, MAX_POSITION);
        let next_velocity = if next_position <= MIN_POSITION && velocity < 0.0 {
            0.0
        } else {
            velocity
        };
        self.state = [next_position, next_velocity];
        self.elapsed_steps += 1;
        let terminated = next_position >= GOAL_POSITION;
        let reward = 0.1f64.mul_add(
            -f64::from(raw_force).powi(2),
            if terminated { 100.0 } else { 0.0 },
        );
        let status = if terminated {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.state.to_vec(),
            reward,
            status,
            info: (),
        }
    }
}

impl ContinuousPpoExample for ContinuousMountainCar {
    const ENV_NAME: &'static str = "mountain-car-continuous";
    const GYMNASIUM_ID: &'static str = "MountainCarContinuous-v0";
    const OBSERVATION_DIM: usize = 2;
    const ACTION_LOW: &'static [f32] = &[-1.0];
    const ACTION_HIGH: &'static [f32] = &[1.0];
    const DEFAULT_TRAIN_STEPS: usize = 30_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 64;
    const DEFAULT_NUM_ENVS: usize = 8;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const DEFAULT_REWARD_SCALE: f64 = 25.0;
    const SOLVED_MEAN_REWARD: f64 = 90.0;
    const GIF_PATH: &'static str = "docs/images/mountain-car-continuous.gif";

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 32,
            critic_hidden_sizes: vec![64, 32],
            gamma: 0.99,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.0,
            epochs: 4,
            minibatch_sequences: 4,
            initial_log_std: -0.5,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let position = observation.first().copied().unwrap_or(-0.5);
        let velocity = observation.get(1).copied().unwrap_or(0.0);
        vec![
            2.0 * (position - MIN_POSITION) / (MAX_POSITION - MIN_POSITION) - 1.0,
            velocity / MAX_SPEED,
        ]
    }

    fn training_reward(
        observation: &[f32],
        _action: &[f32],
        environment_reward: f64,
        next_observation: &[f32],
        status: EpisodeStatus,
        reward_scale: f64,
    ) -> f64 {
        let position = observation.first().copied().unwrap_or(MIN_POSITION);
        let next_position = next_observation.first().copied().unwrap_or(MIN_POSITION);
        let potential = f64::from(Self::height(position) - 1.0);
        let next_potential = if status == EpisodeStatus::Terminated {
            0.0
        } else {
            f64::from(Self::height(next_position) - 1.0)
        };
        reward_scale.mul_add(
            0.99f64.mul_add(next_potential, -potential),
            environment_reward,
        )
    }

    fn is_success(final_observation: &[f32], _total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Terminated
            && final_observation.first().copied().unwrap_or(MIN_POSITION) >= GOAL_POSITION
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
    run_continuous_workflow::<ContinuousMountainCar>()
}

/// Small deterministic generator used for reset sampling.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Current generator state.
    state: u64,
}

impl SplitMix64 {
    /// Construct a stream from one root seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate a uniform scalar in `[0, 1)`.
    fn unit_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 40) as f32 / (1_u32 << 24) as f32
    }

    /// Generate one scalar in `[low, high)`.
    fn f32_between(&mut self, low: f32, high: f32) -> f32 {
        self.unit_f32().mul_add(high - low, low)
    }
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        ContinuousMountainCar, ContinuousPpoExample, Env, EpisodeStatus, Error, Path,
        GOAL_POSITION, MAX_POSITION, MIN_POSITION,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
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
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Policy playback state shown in the continuous `MountainCar` scene.
    #[derive(Resource)]
    struct VisualMountainCar {
        /// Deterministic recurrent checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live exact environment.
        env: ContinuousMountainCar,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Actor memory for this episode.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Four-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic black car body.
    #[derive(Component)]
    struct CarBody;

    /// Dynamic gray wheel with a signed horizontal offset.
    #[derive(Component)]
    struct CarWheel(f32);

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = ContinuousMountainCar::ppo_config(0.003, 0.001);
        let policy = RecurrentPpoPolicy::load(checkpoint, 2, 2, 1, &[-1.0], &[1.0], &config)?;
        let memory = policy.initial_memory();
        let mut env = ContinuousMountainCar::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualMountainCar {
            policy,
            env,
            observation,
            memory,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            1.0 / 30.0,
            TimerMode::Repeating,
        )))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy-gym MountainCarContinuous-v0".into(),
                resolution: WindowResolution::new(600, 400),
                present_mode: PresentMode::AutoVsync,
                resizable: false,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup_scene)
        .add_systems(Update, draw_static_scene);
        if let Some(directory) = capture_dir {
            app.insert_resource(GifCapture::new(directory))
                .add_systems(Update, capture_gif_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_car).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    #[cfg(not(feature = "render"))]
    /// Report the explicit render requirement in headless builds.
    pub(super) fn run_visual(
        _checkpoint: &Path,
        _capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    /// Spawn Gymnasium's black body and two gray wheels.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        commands.spawn((Camera2d, Name::new("Continuous MountainCar Camera")));
        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(40.0, 20.0))),
            MeshMaterial2d(materials.add(Color::BLACK)),
            Transform::from_xyz(0.0, 0.0, 2.0),
            CarBody,
        ));
        for offset in [-10.0, 10.0] {
            commands.spawn((
                Mesh2d(meshes.add(Circle::new(8.0))),
                MeshMaterial2d(materials.add(Color::srgb_u8(128, 128, 128))),
                Transform::from_xyz(0.0, 0.0, 3.0),
                CarWheel(offset),
            ));
        }
    }

    /// Draw the exact sinusoid and goal flag.
    fn draw_static_scene(mut gizmos: Gizmos<'_, '_>) {
        let mut previous = terrain_position(MIN_POSITION);
        for sample in 1..100 {
            let fraction = sample as f32 / 99.0;
            let position = fraction.mul_add(MAX_POSITION - MIN_POSITION, MIN_POSITION);
            let next = terrain_position(position);
            gizmos.line_2d(previous, next, Color::BLACK);
            previous = next;
        }
        let flag_base = terrain_position(GOAL_POSITION);
        let flag_top = flag_base + Vec2::Y * 50.0;
        gizmos.line_2d(flag_base, flag_top, Color::BLACK);
        gizmos.line_2d(
            flag_top,
            flag_top + Vec2::new(25.0, -5.0),
            Color::srgb_u8(204, 204, 0),
        );
        gizmos.line_2d(
            flag_top + Vec2::new(25.0, -5.0),
            flag_top + Vec2::new(0.0, -10.0),
            Color::srgb_u8(204, 204, 0),
        );
        gizmos.line_2d(
            flag_top + Vec2::new(0.0, -10.0),
            flag_top,
            Color::srgb_u8(204, 204, 0),
        );
    }

    /// Advance policy playback at Gymnasium's 30 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualMountainCar>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of deterministic environment transitions.
    fn advance_visual(visual: &mut VisualMountainCar, steps: usize) {
        for _ in 0..steps {
            let encoded = ContinuousMountainCar::encode_observation(&visual.observation);
            let Ok(action) = visual.policy.mean_action(&encoded, &visual.memory) else {
                return;
            };
            visual.memory = action.next_memory;
            let transition = visual.env.step(action.action);
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(None).observation;
                visual.memory = visual.policy.initial_memory();
                break;
            }
        }
    }

    /// Synchronize the body and wheels with the exact renderer transform.
    fn sync_car(
        visual: Res<'_, VisualMountainCar>,
        mut body: Single<'_, '_, &mut Transform, With<CarBody>>,
        mut wheels: Query<'_, '_, (&CarWheel, &mut Transform), Without<CarBody>>,
    ) {
        apply_car_pose(
            visual.observation.first().copied().unwrap_or(-0.5),
            &mut body,
            &mut wheels,
        );
    }

    /// Apply one exact car pose.
    fn apply_car_pose(
        position: f32,
        body: &mut Transform,
        wheels: &mut Query<'_, '_, (&CarWheel, &mut Transform), Without<CarBody>>,
    ) {
        let rotation = (3.0 * position).cos();
        let anchor = terrain_position(position) + Vec2::Y * 10.0;
        let body_offset = Vec2::new(0.0, 10.0).rotate(Vec2::from_angle(rotation));
        body.translation = (anchor + body_offset).extend(2.0);
        body.rotation = Quat::from_rotation_z(rotation);
        for (wheel, mut transform) in wheels {
            let offset = Vec2::new(wheel.0, 0.0).rotate(Vec2::from_angle(rotation));
            transform.translation = (anchor + offset).extend(3.0);
        }
    }

    /// Capture five policy transitions per GIF frame.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualMountainCar>,
        mut body: Single<'_, '_, &mut Transform, With<CarBody>>,
        mut wheels: Query<'_, '_, (&CarWheel, &mut Transform), Without<CarBody>>,
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
            advance_visual(&mut visual, 5);
        }
        apply_car_pose(
            visual.observation.first().copied().unwrap_or(-0.5),
            &mut body,
            &mut wheels,
        );
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Convert one world position to Gymnasium's 600-by-400 viewport.
    fn terrain_position(position: f32) -> Vec2 {
        const SCALE: f32 = 600.0 / (MAX_POSITION - MIN_POSITION);
        Vec2::new(
            (position - MIN_POSITION).mul_add(SCALE, -300.0),
            ContinuousMountainCar::height(position).mul_add(SCALE, -200.0),
        )
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "mountain-car-continuous", 600, 400, |frames| {
            run_visual(checkpoint, Some(frames))
        })
    }

    #[cfg(not(feature = "render"))]
    /// Report the explicit render requirement in headless builds.
    pub(super) fn render_gif(_checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        Err("GIF mode requires the default `render` feature".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn continuous_force_matches_gymnasium_equation() {
        let mut car = ContinuousMountainCar::from_state(-0.5, 0.0);

        let step = car.step(vec![0.5]);

        let expected_velocity = 0.5 * 0.0015 - (3.0_f32 * -0.5).cos() * 0.0025;
        assert!((step.observation[1] - expected_velocity).abs() < 1e-7);
        assert!((step.observation[0] - (-0.5 + expected_velocity)).abs() < 1e-7);
        assert!((step.reward - -0.025).abs() < 1e-7);
    }

    #[test]
    fn crossing_goal_adds_one_hundred_reward() {
        let mut car = ContinuousMountainCar::from_state(0.449, 0.01);

        let step = car.step(vec![0.0]);

        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(step.reward, 100.0);
    }

    #[test]
    fn action_is_clipped_for_dynamics_but_raw_value_sets_control_cost() {
        let mut car = ContinuousMountainCar::from_state(-0.5, 0.0);

        let step = car.step(vec![2.0]);

        let clipped_velocity = 0.0015 - (3.0_f32 * -0.5).cos() * 0.0025;
        assert!((step.observation[1] - clipped_velocity).abs() < 1e-7);
        assert!((step.reward - -0.4).abs() < 1e-7);
    }
}
