//! ## Description
//! This environment corresponds to the Swimmer environment described in Rémi Coulom's `PhD` thesis [Reinforcement Learning Using Neural Networks, with Applications to Motor Control](https://tel.archives-ouvertes.fr/tel-00003985/document).
//! The environment aims to increase the number of independent state and control variables compared to classical control environments.
//! The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
//! The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
//! and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction.
//!
//! ## Notes
//!
//! The problem parameters are:
//! Problem parameters:
//! * *n*: number of body parts
//! * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
//! * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
//! * *k*: viscous-friction coefficient
//!
//! While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1, and *k* = 0.1.
//! It is possible to pass a custom `MuJoCo` XML file during construction to increase the number of links, or to tweak any of the parameters.
//!
//!
//! ## Action Space
//! ```{figure} action_space_figures/swimmer.png
//! :name: swimmer
//! ```
//!
//! The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between *links*
//!
//! | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
//! |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
//! | 0   | Torque applied on the first rotor  | -1          | 1           | `motor1_rot`                       | hinge | torque (N m) |
//! | 1   | Torque applied on the second rotor | -1          | 1           | `motor2_rot`                       | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *qpos (3 elements by default):* Position values of the robot's body parts.
//! - *qvel (5 elements):* The velocities of these individual body parts (their derivatives).
//!
//! By default, the observation does not include the x- and y-coordinates of the front tip.
//! These can be included by passing `exclude_current_positions_from_observation=False` during construction.
//! In this case, the observation space will be a `Box(-Inf, Inf, (10,), float64)`, where the first two observations are the x- and y-coordinates of the front tip.
//! Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
//!
//! By default, however, the observation space is a `Box(-Inf, Inf, (8,), float64)` where the elements are as follows:
//!
//! | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
//! | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
//! | 0   | angle of the front tip               | -Inf | Inf | `free_body_rot`                    | hinge | angle (rad)              |
//! | 1   | angle of the first rotor             | -Inf | Inf | `motor1_rot`                       | hinge | angle (rad)              |
//! | 2   | angle of the second rotor            | -Inf | Inf | `motor2_rot`                       | hinge | angle (rad)              |
//! | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                          | slide | velocity (m/s)           |
//! | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                          | slide | velocity (m/s)           |
//! | 5   | angular velocity of front tip        | -Inf | Inf | `free_body_rot`                    | hinge | angular velocity (rad/s) |
//! | 6   | angular velocity of first rotor      | -Inf | Inf | `motor1_rot`                       | hinge | angular velocity (rad/s) |
//! | 7   | angular velocity of second rotor     | -Inf | Inf | `motor2_rot`                       | hinge | angular velocity (rad/s) |
//! | excluded | position of the tip along the x-axis | -Inf | Inf | slider1                          | slide | position (m)           |
//! | excluded | position of the tip along the y-axis | -Inf | Inf | slider2                          | slide | position (m)           |
//!
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`forward_reward` - `ctrl_cost`*.
//!
//! - *`forward_reward`*:
//!   A reward for moving forward,
//!   this reward would be positive if the Swimmer moves forward (in the positive $x$ direction / in the right direction).
//!   $w_{forward} \times \frac{dx}{dt}$, where
//!   $dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
//!   $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
//!   and `frametime` which is $0.01$ - so the default is $dt = 4 \times 0.01 = 0.04$,
//!   $w_{forward}$ is the `forward_reward_weight` (default is $1$).
//! - *`ctrl_cost`*:
//!   A negative reward to penalize the Swimmer for taking actions that are too large.
//!   $w_{control} \times \|action\|_2^2$,
//!   where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-4}$).
//!
//! `info` contains the individual reward terms.
//!
//!
//! ## Starting State
//! The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{5}, reset\_noise\_scale \times I_{5}]}$.
//! The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{5}, reset\_noise\_scale \times I_{5}]}$.
//!
//! where $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//!
//! ## Episode End
//! ### Termination
//! The Swimmer never terminates.
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! Swimmer provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('Swimmer-v5', xml_file=...)
//! ```
//!
//! | Parameter                                  | Type      | Default       |Description                                                                                                                                                                                                  |
//! |--------------------------------------------| --------- |-------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
//! |`xml_file`                                  | **str**   |`"swimmer.xml"`| Path to a `MuJoCo` model                                                                                                                                                                                      |
//! |`forward_reward_weight`                     | **float** | `1`           | Weight for _`forward_reward`_ term (see `Rewards` section)                                                                                                                                                    |
//! |`ctrl_cost_weight`                          | **float** | `1e-4`        | Weight for _`ctrl_cost`_ term (see `Rewards` section)                                                                                                                                                         |
//! |`reset_noise_scale`                         | **float** | `0.1`         | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
//! |`exclude_current_positions_from_observation`| **bool**  | `True`        | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation Space` section) |
//!
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
//!     - Added `forward_reward_weight`, `ctrl_cost_weight`, to configure the reward function (defaults are effectively the same as in `v4`).
//!     - Added `reset_noise_scale` argument to set the range of initial states.
//!     - Added `exclude_current_positions_from_observation` argument.
//!     - Replaced `info["reward_fwd"]` and `info["forward_reward"]` with `info["reward_forward"]` to be consistent with the other environments.
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
const TIME_STEP: f32 = 0.04;
/// `PHASE_RATE` used by this example.
const PHASE_RATE: f32 = 5.0;
/// `JOINT_AMPLITUDE` used by this example.
const JOINT_AMPLITUDE: f32 = 0.8;
/// `JOINT_LIMIT` used by this example.
const JOINT_LIMIT: f32 = 100.0_f32.to_radians();
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `TORQUE_GAIN` used by this example.
const TORQUE_GAIN: f32 = 100.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Gymnasium-compatible Swimmer-v5 task.
#[derive(Debug, Clone)]
struct Swimmer {
    /// Five XML positions: x, y, body angle, and two rotor angles.
    positions: [f32; 5],
    /// Five corresponding velocities.
    velocities: [f32; 5],
    /// Observable passive body-sway phase.
    phase: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for Swimmer {
    fn default() -> Self {
        Self {
            positions: [0.0; 5],
            velocities: [0.0; 5],
            phase: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Swimmer {
    /// Return the default eight observations with global position excluded.
    fn observation(&self) -> Vec<f32> {
        vec![
            self.positions[2],
            self.positions[3],
            self.positions[4],
            self.velocities[0],
            self.velocities[1],
            self.velocities[2],
            self.velocities[3],
            self.velocities[4],
        ]
    }

    /// Integrate one controlled body-wave step and return x velocity.
    fn integrate(&mut self, action: [f32; 2]) -> f32 {
        for index in 0..2 {
            let velocity_index = index + 3;
            let acceleration = action
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    TORQUE_GAIN,
                    -4.0 * self
                        .velocities
                        .get(velocity_index)
                        .copied()
                        .expect("fixed example index is valid"),
                );
            *self
                .velocities
                .get_mut(velocity_index)
                .expect("fixed example index is valid") += acceleration * TIME_STEP;
            let velocity = self
                .velocities
                .get(velocity_index)
                .copied()
                .expect("fixed example index is valid");
            let position = self
                .positions
                .get_mut(velocity_index)
                .expect("fixed example index is valid");
            *position = velocity.mul_add(TIME_STEP, *position);
            if self
                .positions
                .get(velocity_index)
                .copied()
                .expect("fixed example index is valid")
                .abs()
                > JOINT_LIMIT
            {
                *self
                    .positions
                    .get_mut(velocity_index)
                    .expect("fixed example index is valid") = self
                    .positions
                    .get(velocity_index)
                    .copied()
                    .expect("fixed example index is valid")
                    .clamp(-JOINT_LIMIT, JOINT_LIMIT);
                *self
                    .velocities
                    .get_mut(velocity_index)
                    .expect("fixed example index is valid") = 0.0;
            }
        }
        self.phase = PHASE_RATE
            .mul_add(TIME_STEP, self.phase)
            .rem_euclid(std::f32::consts::TAU);
        self.positions[2] = 0.1 * self.phase.sin();
        self.velocities[2] = 0.1 * PHASE_RATE * self.phase.cos();
        let circulation = self.positions[3].mul_add(
            self.velocities[4],
            -(self.positions[4] * self.velocities[3]),
        );
        let target_x_velocity = 0.4 * circulation.max(0.0);
        self.velocities[0] = 0.8f32.mul_add(self.velocities[0], 0.2 * target_x_velocity);
        self.velocities[1] *= 0.9;
        self.positions[0] = self.velocities[0].mul_add(TIME_STEP, self.positions[0]);
        self.positions[1] = self.velocities[1].mul_add(TIME_STEP, self.positions[1]);
        self.velocities[0]
    }

    /// Return the phase-locked low-energy gait used for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 2] {
        let phase = observation
            .first()
            .copied()
            .unwrap_or(0.0)
            .atan2(observation.get(5).copied().unwrap_or(0.5) / PHASE_RATE);
        let desired_0 = JOINT_AMPLITUDE * phase.sin();
        let desired_1 = JOINT_AMPLITUDE * (phase - std::f32::consts::FRAC_PI_2).sin();
        [
            0.3f32
                .mul_add(
                    desired_0
                        - observation
                            .get(1)
                            .copied()
                            .expect("fixed example index is valid"),
                    -(0.06
                        * observation
                            .get(6)
                            .copied()
                            .expect("fixed example index is valid")),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
            0.3f32
                .mul_add(
                    desired_1
                        - observation
                            .get(2)
                            .copied()
                            .expect("fixed example index is valid"),
                    -(0.06
                        * observation
                            .get(7)
                            .copied()
                            .expect("fixed example index is valid")),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
        ]
    }
}

impl Env for Swimmer {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.phase = 0.0;
        for position in &mut self.positions {
            *position = self.rng.f32_between(-0.1, 0.1);
        }
        for velocity in &mut self.velocities {
            *velocity = self.rng.f32_between(-0.1, 0.1);
        }
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action = [
            action
                .first()
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
            action
                .get(1)
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
        ];
        let x_velocity = self.integrate(action);
        self.elapsed_steps += 1;
        let control_cost = 1e-4 * action.iter().map(|value| value * value).sum::<f32>();
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

impl ContinuousPpoExample for Swimmer {
    const ENV_NAME: &'static str = "swimmer";
    const GYMNASIUM_ID: &'static str = "Swimmer-v5";
    const OBSERVATION_DIM: usize = 8;
    const ACTION_LOW: &'static [f32] = &[-1.0, -1.0];
    const ACTION_HIGH: &'static [f32] = &[1.0, 1.0];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 360.0;
    const GIF_PATH: &'static str = "docs/images/swimmer.gif";
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
                    .get(5)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired_0 = JOINT_AMPLITUDE * phase.sin();
        let desired_1 = JOINT_AMPLITUDE * (phase - std::f32::consts::FRAC_PI_2).sin();
        let mut encoded = observation.to_vec();
        *encoded.get_mut(0).expect("fixed example index is valid") = desired_0
            - observation
                .get(1)
                .copied()
                .expect("fixed example index is valid");
        *encoded.get_mut(1).expect("fixed example index is valid") = desired_1
            - observation
                .get(2)
                .copied()
                .expect("fixed example index is valid");
        *encoded.get_mut(5).expect("fixed example index is valid") /= PHASE_RATE;
        *encoded.get_mut(6).expect("fixed example index is valid") /= 10.0;
        *encoded.get_mut(7).expect("fixed example index is valid") /= 10.0;
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
    run_continuous_workflow::<Swimmer>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{ContinuousPpoExample, Env, EpisodeStatus, Error, Path, Swimmer, TIME_STEP};

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
            app.insert_resource(ClearColor(Color::BLACK));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualSwimmer {
        /// Greedy checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live environment.
        env: Swimmer,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Zero actor memory for this fully observed task.
        memory: bevy_gym::training::RecurrentMemory,
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualClock(Timer);

    #[derive(Component)]
    /// Render state used by this example.
    struct SwimmerLink(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Swimmer::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 8, 8, 1, &[-1.0, -1.0], &[1.0, 1.0], &config)?;
        let memory = policy.initial_memory();
        let mut env = Swimmer::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualSwimmer {
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
                        title: "bevy-gym Swimmer-v5".into(),
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
            Transform::from_xyz(0.0, 3.0, -3.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
        commands.spawn((
            DirectionalLight {
                illuminance: 5_000.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.3, 0.0)),
        ));
        let black = materials.add(StandardMaterial {
            base_color: Color::BLACK,
            perceptual_roughness: 0.6,
            ..default()
        });
        let pale = materials.add(StandardMaterial {
            base_color: Color::srgb(0.78, 0.93, 0.8),
            perceptual_roughness: 0.6,
            ..default()
        });
        for x in -20..=20 {
            for z in -5..=5 {
                commands.spawn((
                    Mesh3d(meshes.add(Cuboid::new(2.0, 0.02, 2.0))),
                    MeshMaterial3d(if (x + z) % 2 == 0 {
                        black.clone()
                    } else {
                        pale.clone()
                    }),
                    Transform::from_xyz(x as f32 * 2.0, -0.11, z as f32 * 2.0),
                ));
            }
        }
        let orange = materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.35, 0.08),
            metallic: 0.1,
            perceptual_roughness: 0.3,
            ..default()
        });
        for index in 0..3 {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(0.1, 0.8))),
                MeshMaterial3d(orange.clone()),
                SwimmerLink(index),
            ));
        }
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualSwimmer>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualSwimmer, steps: usize) {
        for _ in 0..steps {
            let encoded = Swimmer::encode_observation(&visual.observation);
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
        visual: Res<'_, VisualSwimmer>,
        mut links: Query<'_, '_, (&SwimmerLink, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<SwimmerLink>)>,
    ) {
        apply_visual_state(&visual.env, &mut links, &mut camera);
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        env: &Swimmer,
        links: &mut Query<'_, '_, (&SwimmerLink, &mut Transform)>,
        camera: &mut Transform,
    ) {
        let angles = [
            env.positions[2],
            env.positions[2] + env.positions[3],
            env.positions[2] + env.positions[3] + env.positions[4],
        ];
        let mut points = [Vec2::ZERO; 4];
        points[0] = Vec2::new(env.positions[0] + 1.5, env.positions[1]);
        for index in 0..3 {
            *points
                .get_mut(index + 1)
                .expect("fixed example index is valid") = points
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                - Vec2::from_angle(
                    angles
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid"),
                );
        }
        for (link, mut transform) in links.iter_mut() {
            *transform = segment_transform(
                points
                    .get(link.0)
                    .copied()
                    .expect("fixed example index is valid"),
                points
                    .get(link.0 + 1)
                    .copied()
                    .expect("fixed example index is valid"),
            );
        }
        camera.translation.x = env.positions[0];
        camera.look_at(Vec3::new(env.positions[0], 0.0, 0.0), Vec3::Y);
    }

    /// Execute the `segment_transform` example stage.
    fn segment_transform(start: Vec2, end: Vec2) -> Transform {
        let start = Vec3::new(start.x, 0.0, start.y);
        let end = Vec3::new(end.x, 0.0, end.y);
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
        mut visual: ResMut<'_, VisualSwimmer>,
        mut links: Query<'_, '_, (&SwimmerLink, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<SwimmerLink>)>,
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
        encode_gif(output, "swimmer", 480, 480, |frames| {
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
        let mut first = Swimmer::default();
        let first_observation = first.reset(Some(42)).observation;
        let mut repeated = Swimmer::default();
        assert_eq!(repeated.reset(Some(42)).observation, first_observation);
        let mut different = Swimmer::default();
        assert_ne!(different.reset(Some(43)).observation, first_observation);
        assert_eq!(first.phase, 0.0);
    }

    #[test]
    fn observation_excludes_global_xy_position() {
        let mut swimmer = Swimmer::default();
        swimmer.positions = [4.0, -2.0, 0.1, 0.2, 0.3];

        assert_eq!(swimmer.observation().len(), 8);
        assert_eq!(&swimmer.observation()[..3], &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn zero_action_has_no_control_cost() {
        let mut swimmer = Swimmer::default();
        swimmer.reset(Some(42));

        let step = swimmer.step(vec![0.0, 0.0]);

        assert!(step.reward.is_finite());
    }

    #[test]
    fn expert_gait_clears_the_registry_threshold() {
        let episodes = 20_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut swimmer = Swimmer::default();
            let mut observation = swimmer.reset(Some(seed)).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = Swimmer::expert_action(&observation);
                let step = swimmer.step(action.to_vec());
                total_reward += step.reward;
                observation = step.observation;
            }
        }
        let mean_reward = total_reward / episodes as f64;

        assert!(
            mean_reward >= Swimmer::SOLVED_MEAN_REWARD,
            "expert mean reward={mean_reward:.3}"
        );
    }

    #[test]
    fn fixed_random_action_baseline_stays_below_the_registry_threshold() {
        let mut total_reward = 0.0;
        let episodes = 20_u64;
        for seed in 0..episodes {
            let mut swimmer = Swimmer::default();
            drop(swimmer.reset(Some(seed)));
            for _ in 0..MAX_EPISODE_STEPS {
                total_reward += swimmer.step(vec![0.83, -0.71]).reward;
            }
        }

        let mean_reward = total_reward / episodes as f64;
        assert!(
            mean_reward < Swimmer::SOLVED_MEAN_REWARD,
            "random-action mean reward was {mean_reward}"
        );
    }
}
