//! ## Description
//! This environment is the Cartpole environment, based on the work of Barto, Sutton, and Anderson in [Neuronlike adaptive elements that can solve difficult learning control problems](https://ieeexplore.ieee.org/document/6313077),
//! just like in the classic environments, but now powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity).
//! This environment consists of a cart that can be moved linearly, with a pole attached to one end and having another end free.
//! The cart can be pushed left or right, and the goal is to balance the pole on top of the cart by applying forces to the cart.
//!
//!
//! ## Action Space
//! The agent take a 1-element vector for actions.
//!
//! The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
//! the numerical force applied to the cart (with magnitude representing the amount of
//! force and sign representing the direction)
//!
//! | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
//! |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
//! | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//! - *qpos (2 element):* Position values of the robot's cart and pole.
//! - *qvel (2 elements):* The velocities of cart and pole (their derivatives).
//!
//! The observation space is a `Box(-Inf, Inf, (4,), float64)` where the elements are as follows:
//!
//! | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
//! | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
//! | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
//! | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
//! | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
//! | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s)  |
//!
//!
//! ## Rewards
//! The goal is to keep the inverted pendulum stand upright (within a certain angle limit) for as long as possible - as such, a reward of +1 is given for each timestep that the pole is upright.
//!
//! The pole is considered upright if:
//! $|angle| < 0.2$.
//!
//! and `info` also contains the reward.
//!
//!
//! ## Starting State
//! The initial position state is $\mathcal{U}_{[-reset\_noise\_scale   imes I_{2}, reset\_noise\_scale         imes I_{2}]}$.
//! The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale   imes I_{2}, reset\_noise\_scale         imes I_{2}]}$.
//!
//! where $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//!
//! ## Episode End
//! ### Termination
//! The environment terminates when the Inverted Pendulum is unhealthy.
//! The Inverted Pendulum is unhealthy if any of the following happens:
//!
//! 1. Any of the state space values is no longer finite.
//! 2. The absolute value of the vertical angle between the pole and the cart is greater than 0.2 radians.
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! `InvertedPendulum` provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1)
//! ```
//!
//! | Parameter               | Type       | Default                 | Description                                                                                   |
//! |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
//! | `xml_file`              | **str**    |`"inverted_pendulum.xml"`| Path to a `MuJoCo` model                                                                        |
//! | `reset_noise_scale`     | **float**  | `0.01`                  | Scale of random perturbations of initial position and velocity (see `Starting State` section) |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the `MuJoCo` environments.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the Pendulum is healthy (not terminated) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
//!     - Added `xml_file` argument.
//!     - Added `reset_noise_scale` argument to set the range of initial states.
//!     - Added `info["reward_survive"]` which contains the reward.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3.
//! * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.5. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks (including inverted pendulum).
//! * v0: Initial versions release.

use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::RecurrentPpoConfig;
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

/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 0.04;
/// `GRAVITY` used by this example.
const GRAVITY: f32 = 9.81;
/// `CART_MASS` used by this example.
const CART_MASS: f32 = 1.0;
/// `POLE_MASS` used by this example.
const POLE_MASS: f32 = 0.1;
/// `HALF_POLE_LENGTH` used by this example.
const HALF_POLE_LENGTH: f32 = 0.3;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 3.0;
/// `FORCE_GAIN` used by this example.
const FORCE_GAIN: f32 = 10.0;
/// `HEALTHY_ANGLE_LIMIT` used by this example.
const HEALTHY_ANGLE_LIMIT: f32 = 0.2;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Gymnasium-compatible InvertedPendulum-v5 task.
#[derive(Debug, Clone)]
struct InvertedPendulum {
    /// Cart position, pole angle, and their velocities.
    state: [f32; 4],
    /// Most recent clipped force control.
    last_action: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for InvertedPendulum {
    fn default() -> Self {
        Self {
            state: [0.0; 4],
            last_action: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl InvertedPendulum {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    const fn from_state(state: [f32; 4]) -> Self {
        Self {
            state,
            last_action: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }

    /// Return continuous cart-pole state derivatives.
    fn derivatives(state: &[f32; 4], action: f32) -> [f32; 4] {
        let [_position, angle, cart_velocity, angular_velocity] = *state;
        let force = action.mul_add(FORCE_GAIN, -cart_velocity);
        let sin_angle = angle.sin();
        let cos_angle = angle.cos();
        let total_mass = CART_MASS + POLE_MASS;
        let pole_mass_length = POLE_MASS * HALF_POLE_LENGTH;
        let temporary =
            (pole_mass_length * angular_velocity.powi(2)).mul_add(sin_angle, force) / total_mass;
        let angular_acceleration = 0.05f32.mul_add(
            -angular_velocity,
            GRAVITY.mul_add(sin_angle, -(cos_angle * temporary))
                / (HALF_POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cos_angle.powi(2) / total_mass)),
        );
        let cart_acceleration =
            temporary - pole_mass_length * angular_acceleration * cos_angle / total_mass;
        [
            cart_velocity,
            angular_velocity,
            cart_acceleration,
            angular_acceleration,
        ]
    }

    /// Integrate two `MuJoCo` 0.02-second RK4 frames.
    fn integrate(&mut self, action: f32) {
        let state = self.state;
        let k_1 = Self::derivatives(&state, action);
        let k_2 = Self::derivatives(&add_scaled(state, k_1, TIME_STEP * 0.5), action);
        let k_3 = Self::derivatives(&add_scaled(state, k_2, TIME_STEP * 0.5), action);
        let k_4 = Self::derivatives(&add_scaled(state, k_3, TIME_STEP), action);
        for index in 0..4 {
            *self
                .state
                .get_mut(index)
                .expect("fixed example index is valid") += TIME_STEP
                * (2.0f32.mul_add(
                    k_3.get(index)
                        .copied()
                        .expect("fixed example index is valid"),
                    2.0f32.mul_add(
                        k_2.get(index)
                            .copied()
                            .expect("fixed example index is valid"),
                        k_1.get(index)
                            .copied()
                            .expect("fixed example index is valid"),
                    ),
                ) + k_4
                    .get(index)
                    .copied()
                    .expect("fixed example index is valid"))
                / 6.0;
        }
        if self.state[0] < -1.0 {
            self.state[0] = -1.0;
            self.state[2] = self.state[2].max(0.0);
        } else if self.state[0] > 1.0 {
            self.state[0] = 1.0;
            self.state[2] = self.state[2].min(0.0);
        }
    }
}

impl Env for InvertedPendulum {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state = std::array::from_fn(|index| {
            if index == 0 {
                self.rng.f64_between(-0.01, 0.01) as f32
            } else {
                self.rng.f32_between(-0.01, 0.01)
            }
        });
        self.last_action = 0.0;
        self.elapsed_steps = 0;
        Reset {
            observation: self.state.to_vec(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action = action
            .first()
            .copied()
            .unwrap_or(0.0)
            .clamp(-ACTION_LIMIT, ACTION_LIMIT);
        self.integrate(action);
        self.last_action = action;
        self.elapsed_steps += 1;
        let healthy = self.state.iter().all(|value| value.is_finite())
            && self.state[1].abs() <= HEALTHY_ANGLE_LIMIT;
        let status = if !healthy {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.state.to_vec(),
            reward: f64::from(healthy),
            status,
            info: (),
        }
    }
}

impl ContinuousPpoExample for InvertedPendulum {
    const ENV_NAME: &'static str = "inverted-pendulum";
    const GYMNASIUM_ID: &'static str = "InvertedPendulum-v5";
    const OBSERVATION_DIM: usize = 4;
    const ACTION_LOW: &'static [f32] = &[-3.0];
    const ACTION_HIGH: &'static [f32] = &[3.0];
    const DEFAULT_TRAIN_STEPS: usize = 300_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 16;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.0003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const DEFAULT_REWARD_SCALE: f64 = 1.0;
    const SOLVED_MEAN_REWARD: f64 = 950.0;
    const GIF_PATH: &'static str = "docs/images/inverted-pendulum.gif";

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 32,
            critic_hidden_sizes: vec![64, 32],
            gamma: 0.995,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.001,
            epochs: 4,
            minibatch_sequences: 8,
            initial_log_std: -1.0,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        vec![
            observation.first().copied().unwrap_or(0.0),
            observation.get(1).copied().unwrap_or(0.0) / HEALTHY_ANGLE_LIMIT,
            observation.get(2).copied().unwrap_or(0.0) / 5.0,
            observation.get(3).copied().unwrap_or(0.0) / 5.0,
        ]
    }

    fn training_reward(
        _observation: &[f32],
        _action: &[f32],
        environment_reward: f64,
        next_observation: &[f32],
        status: EpisodeStatus,
        reward_scale: f64,
    ) -> f64 {
        if status == EpisodeStatus::Terminated {
            return -10.0 * reward_scale;
        }
        let position = f64::from(next_observation.first().copied().unwrap_or(0.0));
        let angle = f64::from(next_observation.get(1).copied().unwrap_or(0.0));
        let cart_velocity = f64::from(next_observation.get(2).copied().unwrap_or(0.0));
        let angular_velocity = f64::from(next_observation.get(3).copied().unwrap_or(0.0));
        reward_scale.mul_add(
            (-0.01_f64).mul_add(
                angular_velocity.powi(2),
                0.01f64.mul_add(
                    cart_velocity.powi(2),
                    25.0f64.mul_add(angle.powi(2), 0.1 * position.powi(2)),
                ),
            ),
            environment_reward,
        )
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
    run_continuous_workflow::<InvertedPendulum>()
}

/// Add one scaled derivative to one state.
fn add_scaled(state: [f32; 4], derivative: [f32; 4], scale: f32) -> [f32; 4] {
    std::array::from_fn(|index| {
        scale.mul_add(
            derivative
                .get(index)
                .copied()
                .expect("fixed example index is valid"),
            state
                .get(index)
                .copied()
                .expect("fixed example index is valid"),
        )
    })
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        ContinuousPpoExample, Env, EpisodeStatus, Error, InvertedPendulum, Path, HALF_POLE_LENGTH,
        TIME_STEP,
    };

    use bevy::prelude::*;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{
        BevyGymRecorderPlugin, RecordingCallbackError, RecordingFrame, RecordingSettings,
    };
    use bevy_gym::training::RecurrentPpoPolicy;
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::BLACK));
        }
    }

    /// Policy playback state shown in the `InvertedPendulum` scene.
    #[derive(Resource)]
    struct VisualInvertedPendulum {
        /// Greedy recurrent checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live environment.
        env: InvertedPendulum,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Actor memory for this episode.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Twenty-five-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic olive cart capsule.
    #[derive(Component)]
    struct PendulumCart;

    /// Dynamic cyan pole capsule.
    #[derive(Component)]
    struct PendulumPole;

    /// Run the MuJoCo-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        recording: Option<RecordingSettings>,
    ) -> Result<(), Box<dyn Error>> {
        let config = InvertedPendulum::ppo_config(0.0003, 0.001);
        let policy = RecurrentPpoPolicy::load(checkpoint, 4, 4, 1, &[-3.0], &[3.0], &config)?;
        let memory = policy.initial_memory();
        let mut env = InvertedPendulum::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualInvertedPendulum {
            policy,
            env,
            observation,
            memory,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            TIME_STEP,
            TimerMode::Repeating,
        )))
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 200.0,
            affects_lightmapped_meshes: true,
        })
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym InvertedPendulum-v5".into(),
                        resolution: WindowResolution::new(480, 480),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_scene);
        if let Some(settings) = recording {
            app.add_plugins(
                BevyGymRecorderPlugin::new(settings).with_frame_driver(drive_recording_frame),
            );
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
        _recording: Option<()>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    /// Spawn the XML rail, cart, pole, camera, and light.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.30, 2.04).looking_at(Vec3::ZERO, Vec3::Y),
            Name::new("InvertedPendulum Camera"),
        ));
        commands.spawn((
            PointLight {
                intensity: 1_200.0,
                range: 8.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_xyz(-1.0, 2.0, 2.0),
        ));
        let capsule_rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.02, 2.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.3, 0.7),
                perceptual_roughness: 0.45,
                ..default()
            })),
            Transform::from_rotation(capsule_rotation),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.1, 0.2))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.7, 0.7, 0.0),
                perceptual_roughness: 0.4,
                ..default()
            })),
            Transform::from_rotation(capsule_rotation),
            PendulumCart,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.049, 0.44))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.0, 0.7, 0.7),
                perceptual_roughness: 0.35,
                ..default()
            })),
            Transform::from_xyz(0.0, HALF_POLE_LENGTH, 0.0),
            PendulumPole,
        ));
    }

    /// Advance policy playback at `MuJoCo`'s 25 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualInvertedPendulum>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of deterministic policy transitions.
    fn advance_visual(visual: &mut VisualInvertedPendulum, steps: usize) {
        for _ in 0..steps {
            let encoded = InvertedPendulum::encode_observation(&visual.observation);
            let Ok(action) = visual.policy.mean_action(&encoded, &visual.memory) else {
                return;
            };
            visual.memory = action.next_memory;
            let transition = visual.env.step(action.action);
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(Some(12_345)).observation;
                visual.memory = visual.policy.initial_memory();
                break;
            }
        }
    }

    /// Synchronize cart and pole transforms.
    fn sync_scene(
        visual: Res<'_, VisualInvertedPendulum>,
        mut cart: Single<'_, '_, &mut Transform, With<PendulumCart>>,
        mut pole: Single<'_, '_, &mut Transform, (With<PendulumPole>, Without<PendulumCart>)>,
    ) {
        apply_visual_state(&visual.env, &mut cart, &mut pole);
    }

    /// Apply one state to the XML-aligned scene geometry.
    fn apply_visual_state(env: &InvertedPendulum, cart: &mut Transform, pole: &mut Transform) {
        let position = env.state[0];
        let angle = env.state[1];
        let direction = Vec2::new(angle.sin(), angle.cos());
        cart.translation = Vec3::new(position, 0.0, 0.0);
        pole.translation = Vec3::new(
            direction.x.mul_add(HALF_POLE_LENGTH, position),
            direction.y * HALF_POLE_LENGTH,
            0.0,
        );
        pole.rotation = Quat::from_rotation_z(-angle);
    }

    /// Advance at `MuJoCo`'s 25-Hz simulation rate and synchronize one captured frame.
    fn drive_recording_frame(
        world: &mut World,
        frame: RecordingFrame,
    ) -> Result<(), RecordingCallbackError> {
        if !frame.is_first_in_segment() {
            let current = frame.frame_in_segment();
            let previous = current - 1;
            let rate = usize::from(frame.frames_per_second());
            let steps = current * 25 / rate - previous * 25 / rate;
            if steps > 0 {
                advance_visual(&mut world.resource_mut::<VisualInvertedPendulum>(), steps);
            }
        }
        world.run_system_cached(sync_scene)?;
        Ok(())
    }

    /// Render five seconds at 20 frames per second through the reusable recorder.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        let settings = RecordingSettings::gif(output.to_path_buf(), checkpoint.to_path_buf())?;
        run_visual(checkpoint, Some(settings))
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
    fn reset_matches_the_four_value_observation_contract() {
        let mut pendulum = InvertedPendulum::default();

        let reset = pendulum.reset(Some(42));

        assert_eq!(reset.observation.len(), 4);
        assert!(reset.observation.iter().all(|value| value.abs() <= 0.01));
    }

    #[test]
    fn falling_past_point_two_radians_terminates_without_reward() {
        let mut pendulum = InvertedPendulum::from_state([0.0, 0.199, 0.0, 1.0]);

        let step = pendulum.step(vec![0.0]);

        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(step.reward, 0.0);
    }

    #[test]
    fn healthy_transition_rewards_one_point() {
        let mut pendulum = InvertedPendulum::from_state([0.0; 4]);

        let step = pendulum.step(vec![0.0]);

        assert_eq!(step.status, EpisodeStatus::Continuing);
        assert_eq!(step.reward, 1.0);
    }
}
