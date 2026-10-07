//! ## Description
//! This environment is based on the work of P. Wawrzyński in [A Cat-Like Robot Real-Time Learning to Run](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
//! The `HalfCheetah` is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws).
//! The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
//! The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).
//!
//!
//! ## Action Space
//! ```{figure} action_space_figures/half_cheetah.png
//! :name: half_cheetah
//! ```
//!
//! The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.
//!
//! | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
//! | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
//! | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
//! | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
//! | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
//! | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
//! | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
//! | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *qpos (8 elements by default):* Position values of the robot's body parts.
//! - *qvel (9 elements):* The velocities of these individual body parts (their derivatives).
//!
//! By default, the observation does not include the robot's x-coordinate (`rootx`).
//! This can be included by passing `exclude_current_positions_from_observation=False` during construction.
//! In this case, the observation space will be a `Box(-Inf, Inf, (18,), float64)`, where the first observation element is the x-coordinate of the robot.
//! Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
//!
//! By default, however, the observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
//!
//!
//! | Num | Observation                                 | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
//! | --- | ------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
//! | 0   | z-coordinate of the front tip               | -Inf | Inf | rootz                            | slide | position (m)             |
//! | 1   | angle of the front tip                      | -Inf | Inf | rooty                            | hinge | angle (rad)              |
//! | 2   | angle of the back thigh                     | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
//! | 3   | angle of the back shin                      | -Inf | Inf | bshin                            | hinge | angle (rad)              |
//! | 4   | angle of the back foot                      | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
//! | 5   | angle of the front thigh                    | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
//! | 6   | angle of the front shin                     | -Inf | Inf | fshin                            | hinge | angle (rad)              |
//! | 7   | angle of the front foot                     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
//! | 8   | velocity of the x-coordinate of front tip   | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
//! | 9   | velocity of the z-coordinate of front tip   | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
//! | 10  | angular velocity of the front tip           | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
//! | 11  | angular velocity of the back thigh          | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
//! | 12  | angular velocity of the back shin           | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
//! | 13  | angular velocity of the back foot           | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
//! | 14  | angular velocity of the front thigh         | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
//! | 15  | angular velocity of the front shin          | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
//! | 16  | angular velocity of the front foot          | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |
//! | excluded | x-coordinate of the front tip          | -Inf | Inf | rootx                            | slide | position (m)             |
//!
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`forward_reward` - `ctrl_cost`*.
//!
//! - *`forward_reward`*:
//!   A reward for moving forward,
//!   this reward would be positive if the Half Cheetah moves forward (in the positive $x$ direction / in the right direction).
//!   $w_{forward} \times \frac{dx}{dt}$, where
//!   $dx$ is the displacement of the "tip" ($x_{after-action} - x_{before-action}$),
//!   $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
//!   and `frametime` which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
//!   $w_{forward}$ is the `forward_reward_weight` (default is $1$).
//! - *`ctrl_cost`*:
//!   A negative reward to penalize the Half Cheetah for taking actions that are too large.
//!   $w_{control} \times \|action\|_2^2$,
//!   where $w_{control}$ is `ctrl_cost_weight` (default is $0.1$).
//!
//! `info` contains the individual reward terms.
//!
//!
//! ## Starting State
//! The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{9}, reset\_noise\_scale \times I_{9}]}$.
//! The initial velocity state is $\mathcal{N}(0_{9}, reset\_noise\_scale^2 \times I_{9})$.
//!
//! where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//!
//! ## Episode End
//! ### Termination
//! The Half Cheetah never terminates.
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! `HalfCheetah` provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('HalfCheetah-v5', ctrl_cost_weight=0.1, ....)
//! ```
//!
//! | Parameter                                    | Type      | Default              | Description                                                                                                                                                                                         |
//! | -------------------------------------------- | --------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `xml_file`                                   | **str**   | `"half_cheetah.xml"` | Path to a `MuJoCo` model                                                                                                                                                                              |
//! | `forward_reward_weight`                      | **float** | `1`                  | Weight for _`forward_reward`_ term (see `Rewards` section)                                                                                                                                            |
//! | `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for _`ctrl_cost`_ weight (see `Rewards` section)                                                                                                                                               |
//! | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                       |
//! | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the `MuJoCo` environments.
//!     - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Restored the `xml_file` argument (was removed in `v4`).
//!     - Renamed `info["reward_run"]` to `info["reward_forward"]` to be consistent with the other environments.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3.
//! * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks. Added `reward_threshold` to environments.
//! * v0: Initial versions release.

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
const TIME_STEP: f32 = 0.05;
/// `PHASE_RATE` used by this example.
const PHASE_RATE: f32 = 4.0;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `TORQUE_GAIN` used by this example.
const TORQUE_GAIN: f32 = 20.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Gymnasium-compatible HalfCheetah-v5 task.
#[derive(Debug, Clone)]
struct HalfCheetah {
    /// Nine XML positions: x, height offset, torso angle, and six joint angles.
    positions: [f32; 9],
    /// Nine XML velocities in matching order.
    velocities: [f32; 9],
    /// Observable running-gait phase.
    phase: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for HalfCheetah {
    fn default() -> Self {
        Self {
            positions: [0.0; 9],
            velocities: [0.0; 9],
            phase: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl HalfCheetah {
    /// Return Gymnasium's 17 observations with global x excluded.
    fn observation(&self) -> Vec<f32> {
        let mut observation = Vec::with_capacity(17);
        observation.extend_from_slice(&self.positions[1..]);
        observation.extend_from_slice(&self.velocities);
        observation
    }

    /// Return a running cycle for the six actuated leg joints.
    fn desired_joints(phase: f32) -> [f32; 6] {
        let opposite = phase + std::f32::consts::PI;
        [
            0.45 * phase.sin(),
            0.35 * (phase + 0.9).sin(),
            0.2 * (phase + 1.8).sin(),
            0.45 * opposite.sin(),
            0.35 * (opposite + 0.9).sin(),
            0.2 * (opposite + 1.8).sin(),
        ]
    }

    /// Integrate one planar gait step and return horizontal velocity.
    fn integrate(&mut self, action: [f32; 6]) -> f32 {
        for index in 0..6 {
            let state_index = index + 3;
            let acceleration = action
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    TORQUE_GAIN,
                    -4.0 * self
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
        self.positions[1] = 0.025 * self.phase.cos();
        self.velocities[1] = -0.025 * PHASE_RATE * self.phase.sin();
        self.positions[2] = 0.04 * self.phase.sin();
        self.velocities[2] = 0.04 * PHASE_RATE * self.phase.cos();
        let desired = Self::desired_joints(self.phase);
        let tracking_error = (0..6)
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
        let target_velocity = 7.0 * (-0.1 * tracking_error).exp();
        self.velocities[0] = 0.8f32.mul_add(self.velocities[0], 0.2 * target_velocity);
        self.positions[0] = self.velocities[0].mul_add(TIME_STEP, self.positions[0]);
        self.velocities[0]
    }

    /// Return the low-energy joint tracker used for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 6] {
        let phase = observation
            .get(1)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(10)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired = Self::desired_joints(phase);
        std::array::from_fn(|index| {
            2.0f32
                .mul_add(
                    desired
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid")
                        - observation
                            .get(index + 2)
                            .copied()
                            .expect("fixed example index is valid"),
                    -(0.4
                        * observation
                            .get(index + 11)
                            .copied()
                            .expect("fixed example index is valid")),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        })
    }
}

impl Env for HalfCheetah {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.phase = 0.0;
        for value in &mut self.positions {
            *value = self.rng.f32_between(-0.1, 0.1);
        }
        for value in &mut self.velocities {
            *value = 0.1 * self.rng.standard_normal_f32();
        }
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action: [f32; 6] = std::array::from_fn(|index| {
            action
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        });
        let x_velocity = self.integrate(action);
        self.elapsed_steps += 1;
        let control_cost = 0.1 * action.iter().map(|value| value * value).sum::<f32>();
        Step {
            observation: self.observation(),
            reward: f64::from(x_velocity - control_cost),
            status: if self.elapsed_steps >= MAX_EPISODE_STEPS {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

impl ContinuousPpoExample for HalfCheetah {
    const ENV_NAME: &'static str = "half-cheetah";
    const GYMNASIUM_ID: &'static str = "HalfCheetah-v5";
    const OBSERVATION_DIM: usize = 17;
    const ACTION_LOW: &'static [f32] = &[-1.0; 6];
    const ACTION_HIGH: &'static [f32] = &[1.0; 6];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 4_800.0;
    const GIF_PATH: &'static str = "docs/images/half-cheetah.gif";
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
                    .get(10)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired = Self::desired_joints(phase);
        let mut encoded = observation.to_vec();
        for index in 0..6 {
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
                .get_mut(index + 11)
                .expect("fixed example index is valid") = observation
                .get(index + 11)
                .copied()
                .expect("fixed example index is valid")
                / 10.0;
        }
        *encoded.get_mut(8).expect("fixed example index is valid") = 0.0;
        *encoded.get_mut(10).expect("fixed example index is valid") /= PHASE_RATE;
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut cheetah = Self::default();
        let mut observation = cheetah.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);
        while demonstrations.len() < sample_count {
            let action = Self::expert_action(&observation);
            demonstrations.push(RecurrentBehaviorSample {
                observation: Self::encode_observation(&observation),
                action: action.to_vec(),
            });
            let transition = cheetah.step(action.to_vec());
            observation = if transition.status == EpisodeStatus::Continuing {
                transition.observation
            } else {
                cheetah.reset(None).observation
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
    run_continuous_workflow::<HalfCheetah>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{ContinuousPpoExample, Env, EpisodeStatus, Error, HalfCheetah, Path, TIME_STEP};

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
            app.insert_resource(ClearColor(Color::srgb(0.58, 0.58, 0.58)));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualHalfCheetah {
        /// Policy loaded from the selected checkpoint.
        policy: RecurrentPpoPolicy,
        /// Environment state shown in the scene.
        env: HalfCheetah,
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
    struct CheetahLink(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = HalfCheetah::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 17, 17, 1, &[-1.0; 6], &[1.0; 6], &config)?;
        let memory = policy.initial_memory();
        let mut env = HalfCheetah::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualHalfCheetah {
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
                        title: "bevy-gym HalfCheetah-v5".into(),
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
            Transform::from_xyz(0.0, 1.1, 3.3).looking_at(Vec3::new(0.0, 0.88, 0.0), Vec3::Y),
        ));
        spawn_mujoco_stage(&mut commands, &mut meshes, &mut materials);
        let tan = materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.6, 0.4),
            metallic: 0.05,
            perceptual_roughness: 0.35,
            ..default()
        });
        let pink = materials.add(StandardMaterial {
            base_color: Color::srgb(0.58, 0.32, 0.34),
            metallic: 0.05,
            perceptual_roughness: 0.35,
            ..default()
        });
        for (index, radius, length, material) in [
            (0, 0.046, 1.0, tan.clone()),
            (1, 0.046, 0.32, tan.clone()),
            (2, 0.046, 0.30, tan.clone()),
            (3, 0.046, 0.34, pink.clone()),
            (4, 0.046, 0.22, pink.clone()),
            (5, 0.046, 0.28, tan),
            (6, 0.046, 0.24, pink.clone()),
            (7, 0.046, 0.18, pink),
        ] {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(radius, length))),
                MeshMaterial3d(material),
                CheetahLink(index),
            ));
        }
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualHalfCheetah>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualHalfCheetah, steps: usize) {
        for _ in 0..steps {
            let encoded = HalfCheetah::encode_observation(&visual.observation);
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
        visual: Res<'_, VisualHalfCheetah>,
        mut links: Query<'_, '_, (&CheetahLink, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<CheetahLink>)>,
    ) {
        apply_visual_state(&visual.env, &mut links, &mut camera);
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        env: &HalfCheetah,
        links: &mut Query<'_, '_, (&CheetahLink, &mut Transform)>,
        camera: &mut Transform,
    ) {
        let x = env.positions[0];
        let height = 0.7 + env.positions[1];
        let torso_angle = env.positions[2];
        let torso_center = Vec2::new(x, height);
        let torso_axis = rotate(Vec2::X, torso_angle);
        let back_hip = torso_center - 0.5 * torso_axis;
        let front_hip = torso_center + 0.5 * torso_axis;
        let head = front_hip + rotate(Vec2::new(0.3, 0.2), torso_angle);

        let back_thigh_angle = -0.8 + torso_angle + env.positions[3];
        let back_knee = back_hip + Vec2::from_angle(back_thigh_angle) * 0.34;
        let back_shin_angle = -2.0 + torso_angle + env.positions[3] + env.positions[4];
        let back_ankle = back_knee + Vec2::from_angle(back_shin_angle) * 0.34;
        let back_foot_angle =
            -1.7 + torso_angle + env.positions[3] + env.positions[4] + env.positions[5];
        let back_toe = back_ankle + Vec2::from_angle(back_foot_angle) * 0.22;

        let front_thigh_angle = -2.0 + torso_angle + env.positions[6];
        let front_knee = front_hip + Vec2::from_angle(front_thigh_angle) * 0.3;
        let front_shin_angle = -0.9 + torso_angle + env.positions[6] + env.positions[7];
        let front_ankle = front_knee + Vec2::from_angle(front_shin_angle) * 0.3;
        let front_foot_angle =
            -0.9 + torso_angle + env.positions[6] + env.positions[7] + env.positions[8];
        let front_toe = front_ankle + Vec2::from_angle(front_foot_angle) * 0.2;

        let points = [
            (back_hip, front_hip),
            (front_hip, head),
            (back_hip, back_knee),
            (back_knee, back_ankle),
            (back_ankle, back_toe),
            (front_hip, front_knee),
            (front_knee, front_ankle),
            (front_ankle, front_toe),
        ];
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
        camera.look_at(Vec3::new(x, 0.88, 0.0), Vec3::Y);
    }

    /// Execute the `rotate` example stage.
    fn rotate(vector: Vec2, angle: f32) -> Vec2 {
        let (sin, cos) = angle.sin_cos();
        Vec2::new(
            vector.x.mul_add(cos, -vector.y * sin),
            vector.x.mul_add(sin, vector.y * cos),
        )
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
        mut visual: ResMut<'_, VisualHalfCheetah>,
        mut links: Query<'_, '_, (&CheetahLink, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<CheetahLink>)>,
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
        encode_gif(output, "half-cheetah", 480, 480, |frames| {
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
        let mut first = HalfCheetah::default();
        let first_observation = first.reset(Some(42)).observation;
        let mut repeated = HalfCheetah::default();
        assert_eq!(repeated.reset(Some(42)).observation, first_observation);
        let mut different = HalfCheetah::default();
        assert_ne!(different.reset(Some(43)).observation, first_observation);
        assert_eq!(first.phase, 0.0);
    }

    #[test]
    fn observation_excludes_global_x() {
        let mut cheetah = HalfCheetah::default();
        cheetah.positions[0] = 99.0;
        cheetah.positions[1] = 0.25;

        let observation = cheetah.observation();

        assert_eq!(observation.len(), 17);
        assert_eq!(observation[0], 0.25);
        assert!(!observation.contains(&99.0));
    }

    #[test]
    fn episodes_only_end_at_the_time_limit() {
        let mut cheetah = HalfCheetah::default();
        drop(cheetah.reset(Some(7)));
        for _ in 0..(MAX_EPISODE_STEPS - 1) {
            assert_eq!(cheetah.step(vec![0.0; 6]).status, EpisodeStatus::Continuing);
        }
        assert_eq!(cheetah.step(vec![0.0; 6]).status, EpisodeStatus::Truncated);
    }

    #[test]
    fn expert_gait_clears_the_registry_threshold() {
        let episodes = 20_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut cheetah = HalfCheetah::default();
            let mut observation = cheetah.reset(Some(seed)).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = HalfCheetah::expert_action(&observation);
                let step = cheetah.step(action.to_vec());
                total_reward += step.reward;
                observation = step.observation;
            }
        }

        let mean_reward = total_reward / episodes as f64;
        assert!(
            mean_reward >= HalfCheetah::SOLVED_MEAN_REWARD,
            "expert mean reward was {mean_reward}"
        );
    }
}
