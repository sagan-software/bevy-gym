//! ## Description
//!
//! The inverted pendulum swingup problem is based on the classic problem in control theory.
//! The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
//! The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
//! into an upright position, with its center of gravity right above the fixed point.
//!
//! The diagram below specifies the coordinate system used for the implementation of the pendulum's
//! dynamic equations.
//!
//! ![Pendulum Coordinate System](/_static/diagrams/pendulum.png)
//!
//! - `x-y`: cartesian coordinates of the pendulum's end in meters.
//! - `theta` : angle in radians.
//! - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
//!
//! ## Action Space
//!
//! The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.
//!
//! | Num | Action | Min  | Max |
//! |-----|--------|------|-----|
//! | 0   | Torque | -2.0 | 2.0 |
//!
//! ## Observation Space
//!
//! The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
//! end and its angular velocity.
//!
//! | Num | Observation      | Min  | Max |
//! |-----|------------------|------|-----|
//! | 0   | x = cos(theta)   | -1.0 | 1.0 |
//! | 1   | y = sin(theta)   | -1.0 | 1.0 |
//! | 2   | Angular Velocity | -8.0 | 8.0 |
//!
//! ## Rewards
//!
//! The reward function is defined as:
//!
//! *r = -(theta<sup>2</sup> + 0.1 * `theta_dt`<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
//!
//! where `theta` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
//! Based on the above equation, the minimum reward that can be obtained is
//! *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
//! while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
//!
//! ## Starting State
//!
//! The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
//!
//! ## Episode Truncation
//!
//! The episode truncates at 200 time steps.
//!
//! ## Arguments
//!
//! - `g`: .
//!
//! Pendulum has two parameters for `gymnasium.make` with `render_mode` and `g` representing
//! the acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
//! The default value is `g = 10.0`.
//! On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)  # default g=10.0
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<PendulumEnv<Pendulum-v1>>>>>
//! >>> env.reset(seed=123, options={"low": -0.7, "high": 0.5})  # default low=-0.6, high=-0.5
//! (array([ 0.4123625 ,  0.91101986, -0.89235795], dtype=float32), {})
//!
//! ```
//!
//! ## Version History
//!
//! * v1: Simplify the math equations, no difference in behavior.
//! * v0: Initial versions release

use shakmaty as _;
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

/// `MAX_SPEED` used by this example.
const MAX_SPEED: f32 = 8.0;
/// `MAX_TORQUE` used by this example.
const MAX_TORQUE: f32 = 2.0;
/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 0.05;
/// `GRAVITY` used by this example.
const GRAVITY: f32 = 10.0;
/// `MASS` used by this example.
const MASS: f32 = 1.0;
/// `LENGTH` used by this example.
const LENGTH: f32 = 1.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 200;

/// Exact default Gymnasium Pendulum-v1 dynamics.
#[derive(Debug, Clone)]
struct Pendulum {
    /// Unwrapped angle and angular velocity.
    state: [f32; 2],
    /// Last clipped torque for rendering.
    last_torque: Option<f32>,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset generator.
    rng: SplitMix64,
}

impl Default for Pendulum {
    fn default() -> Self {
        Self {
            state: [std::f32::consts::PI, 0.0],
            last_torque: None,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Pendulum {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    const fn from_state(angle: f32, angular_velocity: f32) -> Self {
        Self {
            state: [angle, angular_velocity],
            last_torque: None,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }

    /// Return Gymnasium's observable cosine, sine, and angular velocity.
    fn observation(&self) -> Vec<f32> {
        vec![self.state[0].cos(), self.state[0].sin(), self.state[1]]
    }
}

impl Env for Pendulum {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state = [
            self.rng
                .f32_between(-std::f32::consts::PI, std::f32::consts::PI),
            self.rng.f32_between(-1.0, 1.0),
        ];
        self.last_torque = None;
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let torque = action
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(-MAX_TORQUE, MAX_TORQUE);
        let [angle, angular_velocity] = self.state;
        let normalized_angle = angle_normalize(f64::from(angle));
        let state_cost =
            normalized_angle.mul_add(normalized_angle, 0.1 * f64::from(angular_velocity).powi(2));
        let reward = (-0.001_f64).mul_add(f64::from(torque).powi(2), -state_cost);
        let acceleration = (3.0 * GRAVITY / (2.0 * LENGTH))
            .mul_add(angle.sin(), (3.0 / (MASS * LENGTH.powi(2))) * torque);
        let next_velocity = acceleration
            .mul_add(TIME_STEP, angular_velocity)
            .clamp(-MAX_SPEED, MAX_SPEED);
        let next_angle = next_velocity.mul_add(TIME_STEP, angle);
        self.state = [next_angle, next_velocity];
        self.last_torque = Some(torque);
        self.elapsed_steps += 1;
        let status = if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.observation(),
            reward,
            status,
            info: (),
        }
    }
}

/// Normalize an angle into Gymnasium's half-open `[-pi, pi)` interval.
fn angle_normalize(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

impl ContinuousPpoExample for Pendulum {
    const ENV_NAME: &'static str = "pendulum";
    const GYMNASIUM_ID: &'static str = "Pendulum-v1";
    const OBSERVATION_DIM: usize = 3;
    const ACTION_LOW: &'static [f32] = &[-2.0];
    const ACTION_HIGH: &'static [f32] = &[2.0];
    const DEFAULT_TRAIN_STEPS: usize = 100_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 64;
    const DEFAULT_NUM_ENVS: usize = 8;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const DEFAULT_REWARD_SCALE: f64 = 0.1;
    const SOLVED_MEAN_REWARD: f64 = -700.0;
    const GIF_PATH: &'static str = "docs/images/pendulum.gif";

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
        vec![
            observation.first().copied().unwrap_or(-1.0),
            observation.get(1).copied().unwrap_or(0.0),
            observation.get(2).copied().unwrap_or(0.0) / MAX_SPEED,
        ]
    }

    fn is_success(_final_observation: &[f32], total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Truncated && total_reward >= Self::SOLVED_MEAN_REWARD
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
    run_continuous_workflow::<Pendulum>()
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
    use super::{ContinuousPpoExample, Env, EpisodeStatus, Error, Path, Pendulum, MAX_TORQUE};

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

    /// Policy playback state shown in the Pendulum scene.
    #[derive(Resource)]
    struct VisualPendulum {
        /// Greedy recurrent checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live exact environment.
        env: Pendulum,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Actor memory for this episode.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Gymnasium's clockwise torque-arrow texture.
    #[derive(Resource)]
    struct TorqueImage(Handle<Image>);

    /// Four-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic red pendulum rod.
    #[derive(Component)]
    struct PendulumRod;

    /// Dynamic red pendulum end cap.
    #[derive(Component)]
    struct PendulumEnd;

    /// Dynamic torque arrow.
    #[derive(Component)]
    struct TorqueArrow;

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Pendulum::ppo_config(3e-4, 3e-4);
        let policy = RecurrentPpoPolicy::load(checkpoint, 3, 3, 1, &[-2.0], &[2.0], &config)?;
        let memory = policy.initial_memory();
        let mut env = Pendulum::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualPendulum {
            policy,
            env,
            observation,
            memory,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            1.0 / 30.0,
            TimerMode::Repeating,
        )))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "ref/gymnasium/gymnasium/envs/classic_control".into(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym Pendulum-v1".into(),
                        resolution: WindowResolution::new(500, 500),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_scene);
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
    /// Report the explicit render requirement in headless builds.
    pub(super) fn run_visual(
        _checkpoint: &Path,
        _capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    /// Spawn Gymnasium's rod, rounded endpoint, torque arrow, and black axle.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        assets: Res<'_, AssetServer>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        const SCALE: f32 = 500.0 / 4.4;
        const ROD_WIDTH: f32 = 0.2 * SCALE;
        let red = Color::srgb_u8(204, 77, 77);
        commands.spawn((Camera2d, Name::new("Pendulum Camera")));
        commands.spawn((
            Mesh2d(meshes.add(Rectangle::new(SCALE, ROD_WIDTH))),
            MeshMaterial2d(materials.add(red)),
            Transform::from_xyz(0.0, 0.0, 1.0),
            PendulumRod,
        ));
        commands.spawn((
            Mesh2d(meshes.add(Circle::new(ROD_WIDTH * 0.5))),
            MeshMaterial2d(materials.add(red)),
            Transform::from_xyz(0.0, 0.0, 2.0),
            PendulumEnd,
        ));
        let torque = assets.load("assets/clockwise.png");
        commands.spawn((
            Sprite {
                image: torque.clone(),
                custom_size: Some(Vec2::ZERO),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 3.0),
            Visibility::Hidden,
            TorqueArrow,
        ));
        commands.spawn((
            Mesh2d(meshes.add(Circle::new(0.05 * SCALE))),
            MeshMaterial2d(materials.add(Color::BLACK)),
            Transform::from_xyz(0.0, 0.0, 4.0),
        ));
        commands.insert_resource(TorqueImage(torque));
    }

    /// Advance policy playback at Gymnasium's 30 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualPendulum>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of deterministic policy transitions.
    fn advance_visual(visual: &mut VisualPendulum, steps: usize) {
        for _ in 0..steps {
            let encoded = Pendulum::encode_observation(&visual.observation);
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

    /// Synchronize rod, end cap, and torque arrow.
    fn sync_scene(
        visual: Res<'_, VisualPendulum>,
        torque_image: Res<'_, TorqueImage>,
        mut rod: Single<'_, '_, &mut Transform, With<PendulumRod>>,
        mut end: Single<'_, '_, &mut Transform, (With<PendulumEnd>, Without<PendulumRod>)>,
        mut arrow: Single<
            '_,
            '_,
            (&mut Sprite, &mut Visibility),
            (
                With<TorqueArrow>,
                Without<PendulumRod>,
                Without<PendulumEnd>,
            ),
        >,
    ) {
        apply_visual_state(&visual.env, &torque_image, &mut rod, &mut end, &mut arrow);
    }

    /// Apply the exact Gymnasium geometry for one state.
    fn apply_visual_state(
        env: &Pendulum,
        torque_image: &TorqueImage,
        rod: &mut Transform,
        end: &mut Transform,
        arrow: &mut (Mut<'_, Sprite>, Mut<'_, Visibility>),
    ) {
        const SCALE: f32 = 500.0 / 4.4;
        let rotation = env.state[0] + std::f32::consts::FRAC_PI_2;
        let direction = Vec2::from_angle(rotation);
        rod.translation = (direction * (SCALE * 0.5)).extend(1.0);
        rod.rotation = Quat::from_rotation_z(rotation);
        end.translation = (direction * SCALE).extend(2.0);
        if let Some(torque) = env.last_torque {
            arrow.0.image = torque_image.0.clone();
            arrow.0.custom_size = Some(Vec2::splat(SCALE * torque.abs() / MAX_TORQUE));
            arrow.0.flip_x = torque > 0.0;
            arrow.0.flip_y = true;
            *arrow.1 = Visibility::Visible;
        } else {
            *arrow.1 = Visibility::Hidden;
        }
    }

    /// Capture five policy transitions per GIF frame.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualPendulum>,
        torque_image: Res<'_, TorqueImage>,
        mut rod: Single<'_, '_, &mut Transform, With<PendulumRod>>,
        mut end: Single<'_, '_, &mut Transform, (With<PendulumEnd>, Without<PendulumRod>)>,
        mut arrow: Single<
            '_,
            '_,
            (&mut Sprite, &mut Visibility),
            (
                With<TorqueArrow>,
                Without<PendulumRod>,
                Without<PendulumEnd>,
            ),
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
            advance_visual(&mut visual, 5);
        }
        apply_visual_state(&visual.env, &torque_image, &mut rod, &mut end, &mut arrow);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "pendulum", 500, 500, |frames| {
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
    fn zero_torque_step_matches_gymnasium_equations() {
        let mut pendulum = Pendulum::from_state(0.5, -0.25);

        let step = pendulum.step(vec![0.0]);

        let expected_velocity = (-0.25 + (15.0 * 0.5_f32.sin()) * 0.05).clamp(-8.0, 8.0);
        let expected_angle = 0.5 + expected_velocity * 0.05;
        assert!((step.observation[0] - expected_angle.cos()).abs() < 1e-6);
        assert!((step.observation[1] - expected_angle.sin()).abs() < 1e-6);
        assert!((step.observation[2] - expected_velocity).abs() < 1e-6);
    }

    #[test]
    fn reward_uses_pre_transition_state_and_clipped_torque() {
        let mut pendulum = Pendulum::from_state(std::f32::consts::PI, 1.0);

        let step = pendulum.step(vec![5.0]);

        let expected = -(std::f64::consts::PI.powi(2) + 0.1 + 0.004);
        assert!((step.reward - expected).abs() < 1e-6);
    }

    #[test]
    fn episode_truncates_after_two_hundred_steps() {
        let mut pendulum = Pendulum::from_state(0.0, 0.0);
        let mut status = EpisodeStatus::Continuing;
        for _ in 0..200 {
            status = pendulum.step(vec![0.0]).status;
        }

        assert_eq!(status, EpisodeStatus::Truncated);
    }
}
