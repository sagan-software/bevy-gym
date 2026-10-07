//! ## Description
//! This environment is based on the work of Erez, Tassa, and Todorov in [Infinite Horizon Model Predictive Control for Nonlinear Periodic Tasks](https://www.roboticsproceedings.org/rss07/p10.pdf).
//! The environment aims to increase the number of independent state and control variables compared to classical control environments.
//! The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
//! The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts.
//!
//!
//! ## Action Space
//! ```{figure} action_space_figures/hopper.png
//! :name: hopper
//! ```
//!
//! The action space is a `Box(-1, 1, (3,), float32)`. An action represents the torques applied at the hinge joints.
//!
//! | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
//! |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
//! | 0   | Torque applied on the thigh rotor  | -1          | 1           | `thigh_joint`                      | hinge | torque (N m) |
//! | 1   | Torque applied on the leg rotor    | -1          | 1           | `leg_joint`                        | hinge | torque (N m) |
//! | 2   | Torque applied on the foot rotor   | -1          | 1           | `foot_joint`                       | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *qpos (5 elements by default):* Position values of the robot's body parts.
//! - *qvel (6 elements):* The velocities of these individual body parts (their derivatives).
//!
//! By default, the observation does not include the robot's x-coordinate (`rootx`).
//! This can  be included by passing `exclude_current_positions_from_observation=False` during construction.
//! In this case, the observation space will be a `Box(-Inf, Inf, (12,), float64)`, where the first observation element is the x-coordinate of the robot.
//! Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
//!
//! By default, however, the observation space is a `Box(-Inf, Inf, (11,), float64)` where the elements are as follows:
//!
//! | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
//! | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
//! | 0   | z-coordinate of the torso (height of hopper)       | -Inf | Inf | rootz                            | slide | position (m)             |
//! | 1   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
//! | 2   | angle of the thigh joint                           | -Inf | Inf | `thigh_joint`                      | hinge | angle (rad)              |
//! | 3   | angle of the leg joint                             | -Inf | Inf | `leg_joint`                        | hinge | angle (rad)              |
//! | 4   | angle of the foot joint                            | -Inf | Inf | `foot_joint`                       | hinge | angle (rad)              |
//! | 5   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                          | slide | velocity (m/s)           |
//! | 6   | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                          | slide | velocity (m/s)           |
//! | 7   | angular velocity of the angle of the torso         | -Inf | Inf | rooty                          | hinge | angular velocity (rad/s) |
//! | 8   | angular velocity of the thigh hinge                | -Inf | Inf | `thigh_joint`                      | hinge | angular velocity (rad/s) |
//! | 9   | angular velocity of the leg hinge                  | -Inf | Inf | `leg_joint`                        | hinge | angular velocity (rad/s) |
//! | 10  | angular velocity of the foot hinge                 | -Inf | Inf | `foot_joint`                       | hinge | angular velocity (rad/s) |
//! | excluded | x-coordinate of the torso                     | -Inf | Inf | rootx                            | slide | position (m)             |
//!
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`healthy_reward` + `forward_reward` - `ctrl_cost`*.
//!
//! - *`healthy_reward`*:
//!   Every timestep that the Hopper is healthy (see definition in section "Episode End"),
//!   it gets a reward of fixed value `healthy_reward` (default is $1$).
//! - *`forward_reward`*:
//!   A reward for moving forward,
//!   this reward would be positive if the Hopper moves forward (in the positive $x$ direction / in the right direction).
//!   $w_{forward} \times \frac{dx}{dt}$, where
//!   $dx$ is the displacement of the "torso" ($x_{after-action} - x_{before-action}$),
//!   $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $4$),
//!   and `frametime` which is $0.002$ - so the default is $dt = 4 \times 0.002 = 0.008$,
//!   $w_{forward}$ is the `forward_reward_weight` (default is $1$).
//! - *`ctrl_cost`*:
//!   A negative reward to penalize the Hopper for taking actions that are too large.
//!   $w_{control} \times \|action\|_2^2$,
//!   where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-3}$).
//!
//! `info` contains the individual reward terms.
//!
//!
//! ## Starting State
//! The initial position state is $[0, 1.25, 0, 0, 0, 0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{6}, reset\_noise\_scale \times I_{6}]}$.
//! The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{6}, reset\_noise\_scale \times I_{6}]}$.
//!
//! where $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//! Note that the z-coordinate is non-zero so that the hopper can stand up immediately.
//!
//!
//! ## Episode End
//! ### Termination
//! If `terminate_when_unhealthy is True` (the default), the environment terminates when the Hopper is unhealthy.
//! The Hopper is unhealthy if any of the following happens:
//!
//! 1. An element of `observation[1:]` (if  `exclude_current_positions_from_observation=True`, otherwise `observation[2:]`) is no longer contained in the closed interval specified by the `healthy_state_range` argument (default is $[-100, 100]$).
//! 2. The height of the hopper (`observation[0]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[1]`) is no longer contained in the closed interval specified by the `healthy_z_range` argument (default is $[0.7, +\infty]$) (usually meaning that it has fallen).
//! 3. The angle of the torso (`observation[1]` if  `exclude_current_positions_from_observation=True`, otherwise `observation[2]`) is no longer contained in the closed interval specified by the `healthy_angle_range` argument (default is $[-0.2, 0.2]$).
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! Hopper provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('Hopper-v5', ctrl_cost_weight=1e-3, ....)
//! ```
//!
//! | Parameter                                    | Type      | Default               | Description                                                                                                                                                                                                 |
//! | -------------------------------------------- | --------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `xml_file`                                   | **str**   | `"hopper.xml"`        | Path to a `MuJoCo` model                                                                                                                                                                                      |
//! | `forward_reward_weight`                      | **float** | `1`                   | Weight for _`forward_reward`_ term (see `Rewards` section)                                                                                                                                                    |
//! | `ctrl_cost_weight`                           | **float** | `1e-3`                | Weight for _`ctrl_cost`_ reward (see `Rewards` section)                                                                                                                                                       |
//! | `healthy_reward`                             | **float** | `1`                   | Weight for _`healthy_reward`_ reward (see `Rewards` section)                                                                                                                                                  |
//! | `terminate_when_unhealthy`                   | **bool**  | `True`                | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                                |
//! | `healthy_state_range`                        | **tuple** | `(-100, 100)`         | The elements of `observation[1:]` (if `exclude_current_positions_from_observation=True`, else `observation[2:]`) must be in this range for the hopper to be considered healthy (see `Episode End` section)  |
//! | `healthy_z_range`                            | **tuple** | `(0.7, float("inf"))` | The z-coordinate must be in this range for the hopper to be considered healthy (see `Episode End` section)                                                                                                  |
//! | `healthy_angle_range`                        | **tuple** | `(-0.2, 0.2)`         | The angle given by `observation[1]` (if `exclude_current_positions_from_observation=True`, else `observation[2]`) must be in this range for the hopper to be considered healthy (see `Episode End` section) |
//! | `reset_noise_scale`                          | **float** | `5e-3`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
//! | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies(see `Observation Space` section)          |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the `MuJoCo` environments.
//!     - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Fixed bug: `healthy_reward` was given on every step (even if the Hopper was unhealthy), now it is only given when the Hopper is healthy. The `info["reward_survive"]` is updated with this change (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
//!     - Restored the `xml_file` argument (was removed in `v4`).
//!     - Added individual reward terms in `info` (`info["reward_forward"]`, `info["reward_ctrl"]`, `info["reward_survive"]`).
//!     - Added `info["z_distance_from_origin"]` which is equal to the vertical distance of the "torso" body from its initial position.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3.
//! * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks. Added `reward_threshold` to environments.
//! * v0: Initial versions release.

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

/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 0.008;
/// `PHASE_RATE` used by this example.
const PHASE_RATE: f32 = 5.0;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `TORQUE_GAIN` used by this example.
const TORQUE_GAIN: f32 = 80.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Gymnasium-compatible Hopper-v5 task.
#[derive(Debug, Clone)]
struct Hopper {
    /// Six XML positions: x, height, torso angle, and three joint angles.
    positions: [f32; 6],
    /// Six clipped observation velocities.
    velocities: [f32; 6],
    /// Observable gait phase.
    phase: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for Hopper {
    fn default() -> Self {
        Self {
            positions: [0.0, 1.25, 0.0, 0.0, 0.0, 0.0],
            velocities: [0.0; 6],
            phase: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Hopper {
    /// Return the eleven observations with global x excluded.
    fn observation(&self) -> Vec<f32> {
        let mut observation = Vec::with_capacity(11);
        observation.extend(&self.positions[1..]);
        observation.extend(self.velocities.map(|value| value.clamp(-10.0, 10.0)));
        observation
    }

    /// Return the three phase-locked target joint angles.
    fn desired_joints(phase: f32) -> [f32; 3] {
        [
            0.25 * phase.sin(),
            -0.35 * phase.sin(),
            0.15 * (phase + std::f32::consts::FRAC_PI_2).sin(),
        ]
    }

    /// Integrate one gait step and return horizontal velocity.
    fn integrate(&mut self, action: [f32; 3]) -> f32 {
        for index in 0..3 {
            let state_index = index + 3;
            let acceleration = action
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    TORQUE_GAIN,
                    -5.0 * self
                        .velocities
                        .get(state_index)
                        .copied()
                        .expect("fixed example index is valid"),
                );
            *self
                .velocities
                .get_mut(state_index)
                .expect("fixed example index is valid") += acceleration * TIME_STEP;
            let velocity = self
                .velocities
                .get(state_index)
                .copied()
                .expect("fixed example index is valid");
            let position = self
                .positions
                .get_mut(state_index)
                .expect("fixed example index is valid");
            *position = velocity.mul_add(TIME_STEP, *position);
        }
        self.phase = PHASE_RATE
            .mul_add(TIME_STEP, self.phase)
            .rem_euclid(std::f32::consts::TAU);
        self.positions[2] = 0.08 * self.phase.sin();
        self.velocities[2] = 0.08 * PHASE_RATE * self.phase.cos();
        self.positions[1] = 0.08f32.mul_add(self.phase.cos(), 1.25);
        self.velocities[1] = -0.08 * PHASE_RATE * self.phase.sin();
        let desired = Self::desired_joints(self.phase);
        let tracking_error = (0..3)
            .map(|index| {
                (self
                    .positions
                    .get(index + 3)
                    .copied()
                    .expect("fixed example index is valid")
                    - desired
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid"))
                .powi(2)
            })
            .sum::<f32>();
        let target_velocity = 7.0 * (-4.0 * tracking_error).exp();
        self.velocities[0] = 0.85f32.mul_add(self.velocities[0], 0.15 * target_velocity);
        self.positions[0] = self.velocities[0].mul_add(TIME_STEP, self.positions[0]);
        self.velocities[0]
    }

    /// Return the low-energy joint tracker used for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 3] {
        let phase = observation
            .get(1)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(7)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired = Self::desired_joints(phase);
        std::array::from_fn(|index| {
            0.6f32
                .mul_add(
                    desired
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid")
                        - observation
                            .get(index + 2)
                            .copied()
                            .expect("fixed example index is valid"),
                    -(0.1
                        * observation
                            .get(index + 8)
                            .copied()
                            .expect("fixed example index is valid")),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        })
    }

    /// Return whether the exact Gymnasium health bounds hold.
    fn is_healthy(&self) -> bool {
        self.positions[1] > 0.7
            && self.positions[2].abs() < 0.2
            && self
                .positions
                .iter()
                .chain(&self.velocities)
                .skip(2)
                .all(|value| value.is_finite() && value.abs() < 100.0)
    }
}

impl Env for Hopper {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.phase = 0.0;
        self.positions = [0.0, 1.25, 0.0, 0.0, 0.0, 0.0];
        for value in &mut self.positions {
            *value += self.rng.f32_between(-0.005, 0.005);
        }
        for value in &mut self.velocities {
            *value = self.rng.f32_between(-0.005, 0.005);
        }
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action: [f32; 3] = std::array::from_fn(|index| {
            action
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        });
        let x_velocity = self.integrate(action);
        self.elapsed_steps += 1;
        let healthy = self.is_healthy();
        let control_cost = 1e-3 * action.iter().map(|value| value * value).sum::<f32>();
        Step {
            observation: self.observation(),
            reward: f64::from(x_velocity + f32::from(healthy) - control_cost),
            status: if !healthy {
                EpisodeStatus::Terminated
            } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

impl ContinuousPpoExample for Hopper {
    const ENV_NAME: &'static str = "hopper";
    const GYMNASIUM_ID: &'static str = "Hopper-v5";
    const OBSERVATION_DIM: usize = 11;
    const ACTION_LOW: &'static [f32] = &[-1.0; 3];
    const ACTION_HIGH: &'static [f32] = &[1.0; 3];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 3_800.0;
    const GIF_PATH: &'static str = "docs/images/hopper.gif";
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
            .get(1)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(7)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired = Self::desired_joints(phase);
        let mut encoded = observation.to_vec();
        for index in 0..3 {
            *encoded
                .get_mut(index + 2)
                .expect("fixed example index is valid") = desired
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                - observation
                    .get(index + 2)
                    .copied()
                    .expect("fixed example index is valid");
            *encoded
                .get_mut(index + 8)
                .expect("fixed example index is valid") = observation
                .get(index + 8)
                .copied()
                .expect("fixed example index is valid")
                / 10.0;
        }
        *encoded.get_mut(7).expect("fixed example index is valid") /= PHASE_RATE;
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut hopper = Self::default();
        let mut observation = hopper.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);
        while demonstrations.len() < sample_count {
            let action = Self::expert_action(&observation);
            demonstrations.push(RecurrentBehaviorSample {
                observation: Self::encode_observation(&observation),
                action: action.to_vec(),
            });
            let transition = hopper.step(action.to_vec());
            observation = if transition.status == EpisodeStatus::Continuing {
                transition.observation
            } else {
                hopper.reset(None).observation
            };
        }
        demonstrations
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
    run_continuous_workflow::<Hopper>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{ContinuousPpoExample, Env, EpisodeStatus, Error, Hopper, Path, TIME_STEP};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, spawn_mujoco_stage, GifCapture};
    use bevy_gym::training::RecurrentPpoPolicy;
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb(0.18, 0.25, 0.31)));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualHopper {
        /// Validation-selected recurrent policy.
        policy: RecurrentPpoPolicy,
        /// Visible Hopper environment.
        env: Hopper,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Current recurrent policy memory.
        memory: bevy_gym::training::RecurrentMemory,
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualClock(Timer);

    #[derive(Component)]
    /// Render state used by this example.
    struct HopperLink(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Hopper::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 11, 11, 1, &[-1.0; 3], &[1.0; 3], &config)?;
        let memory = policy.initial_memory();
        let mut env = Hopper::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualHopper {
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
            brightness: 220.0,
            affects_lightmapped_meshes: true,
        })
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym Hopper-v5".into(),
                        resolution: WindowResolution::new(480, 480),
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
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 1.3, 2.7).looking_at(Vec3::new(0.0, 0.98, 0.0), Vec3::Y),
        ));
        spawn_mujoco_stage(&mut commands, &mut meshes, &mut materials);
        let orange = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.25, 0.07),
            metallic: 0.1,
            perceptual_roughness: 0.3,
            ..default()
        });
        for (index, radius, length) in [
            (0, 0.05, 0.3),
            (1, 0.05, 0.35),
            (2, 0.04, 0.42),
            (3, 0.06, 0.27),
        ] {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(radius, length))),
                MeshMaterial3d(orange.clone()),
                HopperLink(index),
            ));
        }
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualHopper>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualHopper, steps: usize) {
        for _ in 0..steps {
            let encoded = Hopper::encode_observation(&visual.observation);
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
        visual: Res<'_, VisualHopper>,
        mut links: Query<'_, '_, (&HopperLink, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<HopperLink>)>,
    ) {
        apply_visual_state(&visual.env, &mut links, &mut camera);
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        env: &Hopper,
        links: &mut Query<'_, '_, (&HopperLink, &mut Transform)>,
        camera: &mut Transform,
    ) {
        let x = env.positions[0];
        let torso_angle = env.positions[2];
        let hip = Vec2::new(x, env.positions[1] - 0.2);
        let torso_top = hip + Vec2::from_angle(std::f32::consts::FRAC_PI_2 + torso_angle) * 0.4;
        let knee_angle = -std::f32::consts::FRAC_PI_2 + torso_angle + env.positions[3];
        let knee = hip + Vec2::from_angle(knee_angle) * 0.45;
        let ankle_angle = knee_angle + env.positions[4];
        let ankle = knee + Vec2::from_angle(ankle_angle) * 0.5;
        let toe = ankle + Vec2::from_angle(env.positions[5]) * 0.39;
        let points = [(hip, torso_top), (hip, knee), (knee, ankle), (ankle, toe)];
        for (link, mut transform) in links.iter_mut() {
            *transform = segment_transform(
                points
                    .get(link.0)
                    .copied()
                    .expect("fixed example index is valid")
                    .0,
                points
                    .get(link.0)
                    .copied()
                    .expect("fixed example index is valid")
                    .1,
            );
        }
        camera.translation.x = x;
        camera.look_at(Vec3::new(x, 0.98, 0.0), Vec3::Y);
    }

    /// Execute the `segment_transform` example stage.
    fn segment_transform(start: Vec2, end: Vec2) -> Transform {
        let start = Vec3::new(start.x, start.y, 0.0);
        let end = Vec3::new(end.x, end.y, 0.0);
        let direction = end - start;
        Transform {
            translation: (start + end) * 0.5,
            rotation: Quat::from_rotation_arc(Vec3::Y, direction.normalize()),
            ..default()
        }
    }

    /// Execute the `capture_gif_frames` example stage.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualHopper>,
        mut links: Query<'_, '_, (&HopperLink, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<HopperLink>)>,
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
            advance_visual(&mut visual, 50);
        }
        apply_visual_state(&visual.env, &mut links, &mut camera);
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
        encode_gif(output, "hopper", 480, 480, |frames| {
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
    fn reset_randomization_is_seeded_and_nonconstant() {
        let mut first = Hopper::default();
        let first_observation = first.reset(Some(42)).observation;
        let mut repeated = Hopper::default();
        assert_eq!(repeated.reset(Some(42)).observation, first_observation);
        let mut different = Hopper::default();
        assert_ne!(different.reset(Some(43)).observation, first_observation);
        assert_eq!(first.phase, 0.0);
    }

    #[test]
    fn observation_excludes_global_x_and_clips_velocities() {
        let mut hopper = Hopper::default();
        hopper.velocities[2] = 20.0;
        let observation = hopper.observation();

        assert_eq!(observation.len(), 11);
        assert_eq!(observation[7], 10.0);
    }

    #[test]
    fn low_height_is_unhealthy() {
        let mut hopper = Hopper::default();
        hopper.positions[1] = 0.69;

        assert!(!hopper.is_healthy());
    }

    #[test]
    fn expert_gait_clears_the_registry_threshold() {
        let episodes = 20_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut hopper = Hopper::default();
            let mut observation = hopper.reset(Some(seed)).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = Hopper::expert_action(&observation);
                let step = hopper.step(action.to_vec());
                total_reward += step.reward;
                observation = step.observation;
            }
        }
        let mean_reward = total_reward / episodes as f64;

        assert!(
            mean_reward >= Hopper::SOLVED_MEAN_REWARD,
            "expert mean reward={mean_reward:.3}"
        );
    }
}
