//! ## Description
//! "Reacher" is a two-jointed robot arm.
//! The goal is to move the robot's end effector (called *fingertip*) close to a target that is spawned at a random position.
//!
//!
//! ## Action Space
//! ```{figure} action_space_figures/reacher.png
//! :name: reacher
//! ```
//!
//! The action space is a `Box(-1, 1, (2,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.
//!
//! | Num | Action                                                                          | Control Min | Control Max |Name (in corresponding XML file)| Joint | Type (Unit)  |
//! |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------------|-------|--------------|
//! | 0   | Torque applied at the first hinge (connecting the link to the point of fixture) | -1          | 1           | joint0                         | hinge | torque (N m) |
//! | 1   | Torque applied at the second hinge (connecting the two links)                   | -1          | 1           | joint1                         | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *cos(qpos) (2 elements):* The cosine of the angles of the two arms.
//! - *sin(qpos) (2 elements):* The sine of the angles of the two arms.
//! - *qpos (2 elements):* The coordinates of the target.
//! - *qvel (2 elements):* The angular velocities of the arms (their derivatives).
//! - *xpos (2 elements):* The vector between the target and the reacher's.
//!
//! The observation space is a `Box(-Inf, Inf, (10,), float64)` where the elements are as follows:
//!
//! | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
//! | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
//! | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(joint0)                      | hinge | unitless                 |
//! | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(joint1)                      | hinge | unitless                 |
//! | 2   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(joint0)                      | hinge | unitless                 |
//! | 3   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(joint1)                      | hinge | unitless                 |
//! | 4   | x-coordinate of the target                                                                     | -Inf | Inf | `target_x`                         | slide | position (m)             |
//! | 5   | y-coordinate of the target                                                                     | -Inf | Inf | `target_y`                         | slide | position (m)             |
//! | 6   | angular velocity of the first arm                                                              | -Inf | Inf | joint0                           | hinge | angular velocity (rad/s) |
//! | 7   | angular velocity of the second arm                                                             | -Inf | Inf | joint1                           | hinge | angular velocity (rad/s) |
//! | 8   | x-value of `position_fingertip` - `position_target`                                                | -Inf | Inf | NA                               | slide | position (m)             |
//! | 9   | y-value of `position_fingertip` - `position_target`                                                | -Inf | Inf | NA                               | slide | position (m)             |
//! | excluded | z-value of `position_fingertip` - `position_target` (constantly 0 since reacher is 2d)        | -Inf | Inf | NA                               | slide | position (m)             |
//!
//!
//! Most Gymnasium environments just return the positions and velocities of the joints in the `.xml` file as the state of the environment.
//! In reacher, however, the state is created by combining only certain elements of the position and velocity and performing some function transformations on them.
//! The `reacher.xml` contains these 4 joints:
//!
//! | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
//! |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
//! | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
//! | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
//! | 2   | x-coordinate of the target  | -Inf     | Inf      | `target_x`                         | slide | position (m)       |
//! | 3   | y-coordinate of the target  | -Inf     | Inf      | `target_y`                         | slide | position (m)       |
//!
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`reward_distance` + `reward_control`*.
//!
//! - *`reward_distance`*:
//!   This reward is a measure of how far the *fingertip* of the reacher (the unattached end) is from the target,
//!   with a more negative value assigned if the reacher's *fingertip* is further away from the target.
//!   It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
//!   where $w_{near}$ is the `reward_near_weight` (default is $1$).
//! - *`reward_control`*:
//!   A negative reward to penalize the walker for taking actions that are too large.
//!   It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
//!   where $w_{control}$ is the `reward_control_weight`. (default is $0.1$)
//!
//! `info` contains the individual reward terms.
//!
//! ## Starting State
//! The initial position state of the reacher arm is $\mathcal{U}_{[-0.1 \times I_{2}, 0.1 \times I_{2}]}$.
//! The position state of the goal is (permanently) $\mathcal{S}(0.2)$.
//! The initial velocity state of the Reacher arm is $\mathcal{U}_{[-0.005 \times 1_{2}, 0.005 \times 1_{2}]}$.
//! The velocity state of the object is (permanently) $`0_2`$.
//!
//! where $\mathcal{U}$ is the multivariate uniform continuous distribution and $\mathcal{S}$ is the uniform continuous spherical distribution.
//!
//! The default frame rate is $2$, with each frame lasting $0.01$, so *dt = 5 * 0.01 = 0.02*.
//!
//!
//! ## Episode End
//! ### Termination
//! The Reacher never terminates.
//!
//! ### Truncation
//! The default duration of an episode is 50 timesteps.
//!
//!
//! ## Arguments
//! Reacher provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('Reacher-v5', xml_file=...)
//! ```
//!
//! | Parameter               | Type       | Default       | Description                                              |
//! |-------------------------|------------|---------------|----------------------------------------------------------|
//! | `xml_file`              | **str**    |`"reacher.xml"`| Path to a `MuJoCo` model                                   |
//! | `reward_dist_weight`    | **float**  | `1`           | Weight for _`reward_dist`_ term (see `Rewards` section)    |
//! | `reward_control_weight` | **float**  | `0.1`         | Weight for _`reward_control`_ term (see `Rewards` section) |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Fixed bug: `reward_distance` was based on the state before the physics step, now it is based on the state after the physics step (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).
//!     - Removed `"z - position_fingertip"` from the observation space since it is always 0 and therefore provides no useful information to the agent, this should result is slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
//!     - Added `xml_file` argument.
//!     - Added `reward_dist_weight`, `reward_control_weight` arguments to configure the reward function (defaults are effectively the same as in `v4`).
//!     - Fixed `info["reward_ctrl"]`  not being multiplied by the reward weight.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3
//! * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks (not including reacher, which has a `max_time_steps` of 50). Added `reward_threshold` to environments.
//! * v0: Initial versions release

use shakmaty as _;
use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy::math::Vec2;
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
const TIME_STEP: f32 = 0.02;
/// `LINK_0_LENGTH` used by this example.
const LINK_0_LENGTH: f32 = 0.1;
/// `LINK_1_LENGTH` used by this example.
const LINK_1_LENGTH: f32 = 0.11;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `TORQUE_GAIN` used by this example.
const TORQUE_GAIN: f32 = 8_000.0;
/// `JOINT_DAMPING` used by this example.
const JOINT_DAMPING: f32 = 8.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 50;

/// Gymnasium-compatible Reacher-v5 task.
#[derive(Debug, Clone)]
struct Reacher {
    /// Two relative joint angles followed by their angular velocities.
    state: [f32; 4],
    /// Fixed target coordinates for the current episode.
    target: Vec2,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for Reacher {
    fn default() -> Self {
        Self {
            state: [0.0; 4],
            target: Vec2::new(0.1, -0.1),
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Reacher {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    const fn from_state(state: [f32; 4], target: Vec2) -> Self {
        Self {
            state,
            target,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }

    /// Return the endpoint of the XML-defined two-link arm.
    fn fingertip(&self) -> Vec2 {
        let angle_0 = self.state[0];
        let angle_1 = angle_0 + self.state[1];
        Vec2::new(
            LINK_0_LENGTH.mul_add(angle_0.cos(), LINK_1_LENGTH * angle_1.cos()),
            LINK_0_LENGTH.mul_add(angle_0.sin(), LINK_1_LENGTH * angle_1.sin()),
        )
    }

    /// Return Gymnasium's ten transformed observations.
    fn observation(&self) -> Vec<f32> {
        let difference = self.fingertip() - self.target;
        vec![
            self.state[0].cos(),
            self.state[1].cos(),
            self.state[0].sin(),
            self.state[1].sin(),
            self.target.x,
            self.target.y,
            self.state[2],
            self.state[3],
            difference.x,
            difference.y,
        ]
    }

    /// Integrate damped actuator dynamics over one 0.02-second control step.
    fn integrate(&mut self, action: [f32; 2]) {
        for index in 0..2 {
            let acceleration = action
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    TORQUE_GAIN,
                    -JOINT_DAMPING
                        * self
                            .state
                            .get(index + 2)
                            .copied()
                            .expect("fixed example index is valid"),
                );
            *self
                .state
                .get_mut(index + 2)
                .expect("fixed example index is valid") += acceleration * TIME_STEP;
            let velocity = self
                .state
                .get(index + 2)
                .copied()
                .expect("fixed example index is valid");
            let angle = self
                .state
                .get_mut(index)
                .expect("fixed example index is valid");
            *angle = velocity.mul_add(TIME_STEP, *angle);
        }
        self.state[1] = self.state[1].clamp(-3.0, 3.0);
    }

    /// Return a low-energy inverse-kinematics controller for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 2] {
        let angle_0 = observation
            .get(2)
            .copied()
            .unwrap_or(0.0)
            .atan2(observation.first().copied().unwrap_or(1.0));
        let angle_1 = observation
            .get(3)
            .copied()
            .unwrap_or(0.0)
            .atan2(observation.get(1).copied().unwrap_or(1.0));
        let target = Vec2::new(
            observation.get(4).copied().unwrap_or(0.0),
            observation.get(5).copied().unwrap_or(0.0),
        );
        let [desired_0, desired_1] = inverse_kinematic_angles(target);
        let velocity_0 = observation.get(6).copied().unwrap_or(0.0);
        let velocity_1 = observation.get(7).copied().unwrap_or(0.0);
        [
            0.01f32
                .mul_add(
                    angle_difference(desired_0, angle_0),
                    -(0.00125 * velocity_0),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
            0.01f32
                .mul_add(
                    angle_difference(desired_1, angle_1),
                    -(0.00125 * velocity_1),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
        ]
    }
}

/// Return the elbow-up joint solution for one reachable target.
fn inverse_kinematic_angles(target: Vec2) -> [f32; 2] {
    let radius_squared = target.length_squared().clamp(
        (LINK_0_LENGTH - LINK_1_LENGTH).powi(2),
        (LINK_0_LENGTH + LINK_1_LENGTH).powi(2),
    );
    let desired_1 = (LINK_1_LENGTH.mul_add(
        -LINK_1_LENGTH,
        LINK_0_LENGTH.mul_add(-LINK_0_LENGTH, radius_squared),
    ) / (2.0 * LINK_0_LENGTH * LINK_1_LENGTH))
        .clamp(-1.0, 1.0)
        .acos();
    let shoulder_offset = (LINK_1_LENGTH * desired_1.sin())
        .atan2(LINK_1_LENGTH.mul_add(desired_1.cos(), LINK_0_LENGTH));
    let desired_0 = target.y.atan2(target.x) - shoulder_offset;
    [desired_0, desired_1]
}

impl Env for Reacher {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state[0] = self.rng.f32_between(-0.1, 0.1);
        self.state[1] = self.rng.f32_between(-0.1, 0.1);
        self.state[2] = self.rng.f32_between(-0.005, 0.005);
        self.state[3] = self.rng.f32_between(-0.005, 0.005);
        loop {
            self.target = Vec2::new(
                self.rng.f32_between(-0.2, 0.2),
                self.rng.f32_between(-0.2, 0.2),
            );
            if self.target.length() < 0.2 {
                break;
            }
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
        self.integrate(action);
        self.elapsed_steps += 1;
        let reward = -f64::from((self.fingertip() - self.target).length())
            - action
                .iter()
                .map(|value| f64::from(value * value))
                .sum::<f64>();
        Step {
            observation: self.observation(),
            reward,
            status: if self.elapsed_steps >= MAX_EPISODE_STEPS {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

impl ContinuousPpoExample for Reacher {
    const ENV_NAME: &'static str = "reacher";
    const GYMNASIUM_ID: &'static str = "Reacher-v5";
    const OBSERVATION_DIM: usize = 10;
    const ACTION_LOW: &'static [f32] = &[-1.0, -1.0];
    const ACTION_HIGH: &'static [f32] = &[1.0, 1.0];
    const DEFAULT_TRAIN_STEPS: usize = 300_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 50;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 100;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = -3.75;
    const GIF_PATH: &'static str = "docs/images/reacher.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 4_096;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 100;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![128, 64],
            gamma: 0.98,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.001,
            epochs: 4,
            minibatch_sequences: 16,
            initial_log_std: -1.5,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let mut encoded = observation.to_vec();
        let angle_0 = observation
            .get(2)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .first()
                    .copied()
                    .expect("fixed example index is valid"),
            );
        let angle_1 = observation
            .get(3)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(1)
                    .copied()
                    .expect("fixed example index is valid"),
            );
        let [desired_0, desired_1] = inverse_kinematic_angles(Vec2::new(
            observation
                .get(4)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(5)
                .copied()
                .expect("fixed example index is valid"),
        ));
        *encoded.get_mut(4).expect("fixed example index is valid") =
            angle_difference(desired_0, angle_0) / std::f32::consts::PI;
        *encoded.get_mut(5).expect("fixed example index is valid") =
            angle_difference(desired_1, angle_1) / std::f32::consts::PI;
        *encoded.get_mut(6).expect("fixed example index is valid") = observation
            .get(6)
            .copied()
            .expect("fixed example index is valid")
            / 10.0;
        *encoded.get_mut(7).expect("fixed example index is valid") = observation
            .get(7)
            .copied()
            .expect("fixed example index is valid")
            / 10.0;
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut demonstrations = Vec::with_capacity(sample_count);
        let mut episode = 0_u64;
        while demonstrations.len() < sample_count {
            let mut env = Self::default();
            let mut observation = env.reset(Some(seed.wrapping_add(episode))).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = Self::expert_action(&observation);
                demonstrations.push(RecurrentBehaviorSample {
                    observation: Self::encode_observation(&observation),
                    action: action.to_vec(),
                });
                if demonstrations.len() == sample_count {
                    break;
                }
                observation = env.step(action.to_vec()).observation;
            }
            episode = episode.wrapping_add(1);
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
    run_continuous_workflow::<Reacher>()
}

/// Return the shortest signed angular difference.
fn angle_difference(target: f32, current: f32) -> f32 {
    (target - current + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        ContinuousPpoExample, Env, EpisodeStatus, Error, Path, Reacher, LINK_0_LENGTH,
        LINK_1_LENGTH, TIME_STEP,
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
            app.insert_resource(ClearColor(Color::BLACK));
        }
    }

    /// Policy playback state shown in the Reacher scene.
    #[derive(Resource)]
    struct VisualReacher {
        /// Greedy recurrent checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live environment.
        env: Reacher,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Zero actor memory used by this fully observed task.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Fifty-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic arm link indexed from the root.
    #[derive(Component)]
    struct ReacherLink(usize);

    /// Dynamic arm joint indexed from the root.
    #[derive(Component)]
    struct ReacherJoint(usize);

    /// Dynamic target marker.
    #[derive(Component)]
    struct ReacherTarget;

    /// Run the MuJoCo-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Reacher::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 10, 10, 1, &[-1.0, -1.0], &[1.0, 1.0], &config)?;
        let memory = policy.initial_memory();
        let mut env = Reacher::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualReacher {
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
            brightness: 160.0,
            affects_lightmapped_meshes: true,
        })
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym Reacher-v5".into(),
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
    /// Report the explicit render requirement in headless builds.
    pub(super) fn run_visual(
        _checkpoint: &Path,
        _capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    /// Spawn the XML arena, arm, target, camera, and light.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.8, 0.85).looking_at(Vec3::ZERO, Vec3::Y),
            Name::new("Reacher Camera"),
        ));
        commands.spawn((
            DirectionalLight {
                illuminance: 4_000.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, -0.3, 0.0)),
        ));
        let ground = materials.add(StandardMaterial {
            base_color: Color::srgb(0.35, 0.35, 0.35),
            perceptual_roughness: 0.8,
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(2.0, 2.0))),
            MeshMaterial3d(ground),
        ));
        let pink = materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.16, 0.3),
            perceptual_roughness: 0.35,
            ..default()
        });
        for (start, end) in [
            (Vec2::new(-0.3, -0.3), Vec2::new(0.3, -0.3)),
            (Vec2::new(0.3, -0.3), Vec2::new(0.3, 0.3)),
            (Vec2::new(0.3, 0.3), Vec2::new(-0.3, 0.3)),
            (Vec2::new(-0.3, 0.3), Vec2::new(-0.3, -0.3)),
        ] {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(0.02, 0.56))),
                MeshMaterial3d(pink.clone()),
                segment_transform(start, end, 0.02),
            ));
        }
        let blue = materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.18, 0.28),
            perceptual_roughness: 0.35,
            ..default()
        });
        for (index, length) in [LINK_0_LENGTH, LINK_1_LENGTH].into_iter().enumerate() {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(0.01, length - 0.02))),
                MeshMaterial3d(blue.clone()),
                ReacherLink(index),
            ));
        }
        for index in 0..2 {
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(0.011))),
                MeshMaterial3d(pink.clone()),
                ReacherJoint(index),
            ));
        }
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.009))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.45, 0.08, 0.08),
                perceptual_roughness: 0.4,
                ..default()
            })),
            ReacherTarget,
        ));
    }

    /// Advance policy playback at the environment's 50 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualReacher>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of deterministic policy transitions.
    fn advance_visual(visual: &mut VisualReacher, steps: usize) {
        for _ in 0..steps {
            let encoded = Reacher::encode_observation(&visual.observation);
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

    /// Synchronize the arm and target transforms.
    fn sync_scene(
        visual: Res<'_, VisualReacher>,
        mut links: Query<'_, '_, (&ReacherLink, &mut Transform)>,
        mut joints: Query<'_, '_, (&ReacherJoint, &mut Transform), Without<ReacherLink>>,
        mut target: Single<
            '_,
            '_,
            &mut Transform,
            (
                With<ReacherTarget>,
                Without<ReacherLink>,
                Without<ReacherJoint>,
            ),
        >,
    ) {
        apply_visual_state(&visual.env, &mut links, &mut joints, &mut target);
    }

    /// Apply one state to the XML-aligned arm geometry.
    fn apply_visual_state(
        env: &Reacher,
        links: &mut Query<'_, '_, (&ReacherLink, &mut Transform)>,
        joints: &mut Query<'_, '_, (&ReacherJoint, &mut Transform), Without<ReacherLink>>,
        target: &mut Transform,
    ) {
        let root = Vec2::ZERO;
        let angle_0 = env.state[0];
        let angle_1 = angle_0 + env.state[1];
        let elbow = root + Vec2::new(angle_0.cos(), angle_0.sin()) * LINK_0_LENGTH;
        let fingertip = elbow + Vec2::new(angle_1.cos(), angle_1.sin()) * LINK_1_LENGTH;
        for (link, mut transform) in links.iter_mut() {
            *transform = if link.0 == 0 {
                segment_transform(root, elbow, 0.025)
            } else {
                segment_transform(elbow, fingertip, 0.025)
            };
        }
        for (joint, mut transform) in joints.iter_mut() {
            let position = if joint.0 == 0 { root } else { fingertip };
            transform.translation = Vec3::new(position.x, 0.025, position.y);
        }
        target.translation = Vec3::new(env.target.x, 0.025, env.target.y);
    }

    /// Build a capsule transform between two MuJoCo-plane points.
    fn segment_transform(start: Vec2, end: Vec2, height: f32) -> Transform {
        let start = Vec3::new(start.x, height, start.y);
        let end = Vec3::new(end.x, height, end.y);
        let direction = end - start;
        Transform {
            translation: (start + end) * 0.5,
            rotation: Quat::from_rotation_arc(Vec3::Y, direction.normalize()),
            ..default()
        }
    }

    /// Capture ten policy transitions per GIF frame.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualReacher>,
        mut links: Query<'_, '_, (&ReacherLink, &mut Transform)>,
        mut joints: Query<'_, '_, (&ReacherJoint, &mut Transform), Without<ReacherLink>>,
        mut target: Single<
            '_,
            '_,
            &mut Transform,
            (
                With<ReacherTarget>,
                Without<ReacherLink>,
                Without<ReacherJoint>,
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
            advance_visual(&mut visual, 10);
        }
        apply_visual_state(&visual.env, &mut links, &mut joints, &mut target);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "reacher", 480, 480, |frames| {
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
    fn straight_arm_observation_matches_xml_geometry() {
        let reacher = Reacher::from_state([0.0; 4], Vec2::new(0.1, 0.0));

        let observation = reacher.observation();

        assert_eq!(&observation[..8], &[1.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]);
        assert!((observation[8] - 0.11).abs() < 1e-6);
        assert_eq!(observation[9], 0.0);
    }

    #[test]
    fn reward_uses_post_step_distance_and_squared_control() {
        let mut reacher = Reacher::from_state([0.0; 4], Vec2::new(0.21, 0.0));

        let step = reacher.step(vec![0.0, 0.0]);

        assert!(step.reward.abs() < 1e-6);
    }

    #[test]
    fn expert_controller_clears_the_registry_threshold() {
        let episodes = 100_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut reacher = Reacher::default();
            let mut observation = reacher.reset(Some(seed)).observation;
            loop {
                let action = Reacher::expert_action(&observation);
                let step = reacher.step(action.to_vec());
                total_reward += step.reward;
                observation = step.observation;
                if step.status != EpisodeStatus::Continuing {
                    break;
                }
            }
        }
        let mean_reward = total_reward / episodes as f64;

        assert!(
            mean_reward >= Reacher::SOLVED_MEAN_REWARD,
            "expert mean reward={mean_reward:.3}"
        );
    }
}
