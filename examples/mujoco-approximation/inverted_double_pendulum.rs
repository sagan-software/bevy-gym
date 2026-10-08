//! ## Description
//! This environment originates from control theory and builds on the cartpole environment based on the work of Barto, Sutton, and Anderson in [Neuronlike adaptive elements that can solve difficult learning control problems](https://ieeexplore.ieee.org/document/6313077),
//! powered by the Mujoco physics simulator - allowing for more complex experiments (such as varying the effects of gravity or constraints).
//! This environment involves a cart that can be moved linearly, with one pole attached to it and a second pole attached to the other end of the first pole (leaving the second pole as the only one with a free end).
//! The cart can be pushed left or right, and the goal is to balance the second pole on top of the first pole, which is in turn on top of the cart, by applying continuous forces to the cart.
//!
//!
//! ## Action Space
//! The agent take a 1-element vector for actions.
//! The action space is a continuous `(action)` in `[-1, 1]`, where `action` represents the
//! numerical force applied to the cart (with magnitude representing the amount of force and
//! sign representing the direction)
//!
//! | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint |Type (Unit)|
//! |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
//! | 0   | Force applied on the cart | -1          | 1           | slider                           | slide | Force (N) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *qpos (1 element):* Position values of the robot's cart.
//! - *sin(qpos) (2 elements):* The sine of the angles of poles.
//! - *cos(qpos) (2 elements):* The cosine of the angles of poles.
//! - *qvel (3 elements):* The velocities of these individual body parts (their derivatives).
//! - *`qfrc_constraint` (1 element):* Constraint force of the cart.
//!   There is one constraint force for contacts for each degree of freedom (3).
//!   The approach and handling of constraints by `MuJoCo` is unique to the simulator and is based on their research.
//!   More information can be found  in their [*documentation*](https://mujoco.readthedocs.io/en/latest/computation.html) or in their paper [Convex and analytically-invertible dynamics with contacts and constraints: Theory and implementation in MuJoCo](https://roboti.us/lab/papers/TodorovICRA14.pdf).
//!
//! The observation space is a `Box(-Inf, Inf, (9,), float64)` where the elements are as follows:
//!
//! | Num | Observation                                                       | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
//! | --- | ----------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
//! | 0   | position of the cart along the linear surface                     | -Inf | Inf | slider                           | slide | position (m)             |
//! | 1   | sine of the angle between the cart and the first pole             | -Inf | Inf | sin(hinge)                       | hinge | unitless                 |
//! | 2   | sine of the angle between the two poles                           | -Inf | Inf | sin(hinge2)                      | hinge | unitless                 |
//! | 3   | cosine of the angle between the cart and the first pole           | -Inf | Inf | cos(hinge)                       | hinge | unitless                 |
//! | 4   | cosine of the angle between the two poles                         | -Inf | Inf | cos(hinge2)                      | hinge | unitless                 |
//! | 5   | velocity of the cart                                              | -Inf | Inf | slider                           | slide | velocity (m/s)           |
//! | 6   | angular velocity of the angle between the cart and the first pole | -Inf | Inf | hinge                            | hinge | angular velocity (rad/s) |
//! | 7   | angular velocity of the angle between the two poles               | -Inf | Inf | hinge2                           | hinge | angular velocity (rad/s) |
//! | 8   | constraint force - x                                              | -Inf | Inf | slider                           | slide | Force (N)                |
//! | excluded | constraint force - y                                         | -Inf | Inf | slider                           | slide | Force (N)                |
//! | excluded | constraint force - z                                         | -Inf | Inf | slider                           | slide | Force (N)                |
//!
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`alive_bonus` - `distance_penalty` - `velocity_penalty`*.
//!
//! - *`alive_bonus`*:
//!   Every timestep that the Inverted Pendulum is healthy (see definition in section "Episode End"),
//!   it gets a reward of fixed value `healthy_reward` (default is $10$).
//! - *`distance_penalty`*:
//!   This reward is a measure of how far the *tip* of the second pendulum (the only free end) moves,
//!   and it is calculated as $0.01 x_{pole2-tip}^2 + (y_{pole2-tip}-2)^2$,
//!   where $x_{pole2-tip}, y_{pole2-tip}$ are the xy-coordinatesof the tip of the second pole.
//! - *`velocity_penalty`*:
//!   A negative reward to penalize the agent for moving too fast.
//!   The penalty is $10^{-3}$ times `omega_1` plus $5 \times 10^{-3}$ times `omega_2`,
//!   where `omega_1` and `omega_2` are the angular velocities of the hinges.
//!
//! `info` contains the individual reward terms.
//!
//!
//! ## Starting State
//! The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{3}, reset\_noise\_scale \times I_{3}]}$.
//! The initial velocity state is $\mathcal{N}(0_{3}, reset\_noise\_scale^2 \times I_{3})$.
//!
//! where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//!
//! ## Episode End
//! ### Termination
//! The environment terminates when the Inverted Double Pendulum is unhealthy.
//! The Inverted Double Pendulum is unhealthy if any of the following happens:
//!
//! 1.Termination: The `y_coordinate` of the tip of the second pole $\leq 1$.
//!
//! Note: The maximum standing height of the system is 1.2 m when all the parts are perpendicularly vertical on top of each other.
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! `InvertedDoublePendulum` provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('InvertedDoublePendulum-v5', healthy_reward=10, ...)
//! ```
//!
//! | Parameter               | Type       | Default                        | Description                                                                                   |
//! |-------------------------|------------|--------------------------------|-----------------------------------------------------------------------------------------------|
//! | `xml_file`              | **str**    |`"inverted_double_pendulum.xml"`| Path to a `MuJoCo` model                                                                        |
//! | `healthy_reward`        | **float**  | `10`                           | Constant reward given if the pendulum is `healthy` (upright) (see `Rewards` section)          |
//! | `reset_noise_scale`     | **float**  | `0.1`                          | Scale of random perturbations of initial position and velocity (see `Starting State` section) |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Fixed bug: `healthy_reward` was given on every step (even if the Pendulum is unhealthy), now it is only given if the `DoublePendulum` is healthy (not terminated)(related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/500)).
//!     - Excluded the `qfrc_constraint` ("constraint force") of the hinges from the observation space (as it was always 0, thus providing no useful information to the agent, resulting in slightly faster training) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/228)).
//!     - Added `xml_file` argument.
//!     - Added `reset_noise_scale` argument to set the range of initial states.
//!     - Added `healthy_reward` argument to configure the reward function (defaults are effectively the same as in `v4`).
//!     - Added individual reward terms in `info` (`info["reward_survive"]`, `info["distance_penalty"]`, `info["velocity_penalty"]`).
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3.
//! * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks (including inverted pendulum).
//! * v0: Initial versions release.

use std::error::Error;

use std::path::Path;

use bevy_gym::training::{RecurrentBehaviorSample, RecurrentPpoConfig};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::SplitMix64;
use bevy_gym::training::{run_continuous_workflow, ContinuousPpoExample};

/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 0.05;
/// `GRAVITY` used by this example.
const GRAVITY: f32 = 9.81;
/// `LINK_LENGTH` used by this example.
const LINK_LENGTH: f32 = 0.6;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `FORCE_GAIN` used by this example.
const FORCE_GAIN: f32 = 20.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Gymnasium-compatible InvertedDoublePendulum-v5 task.
#[derive(Debug, Clone)]
struct InvertedDoublePendulum {
    /// Cart position, two joint angles, and their velocities.
    state: [f32; 6],
    /// Slider constraint force exposed by the ninth observation.
    constraint_force: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for InvertedDoublePendulum {
    fn default() -> Self {
        Self {
            state: [0.0; 6],
            constraint_force: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl InvertedDoublePendulum {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    const fn from_state(state: [f32; 6]) -> Self {
        Self {
            state,
            constraint_force: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }

    /// Return the nine observations defined by Gymnasium v5.
    fn observation(&self) -> Vec<f32> {
        let [position, angle_1, angle_2, cart_velocity, angular_velocity_1, angular_velocity_2] =
            self.state;
        vec![
            position,
            angle_1.sin(),
            angle_2.sin(),
            angle_1.cos(),
            angle_2.cos(),
            cart_velocity.clamp(-10.0, 10.0),
            angular_velocity_1.clamp(-10.0, 10.0),
            angular_velocity_2.clamp(-10.0, 10.0),
            self.constraint_force.clamp(-10.0, 10.0),
        ]
    }

    /// Return the free tip's horizontal and vertical coordinates.
    fn tip_position(state: &[f32; 6]) -> (f32, f32) {
        let angle_1 = state[1];
        let absolute_angle_2 = angle_1 + state[2];
        (
            LINK_LENGTH.mul_add(
                absolute_angle_2.sin(),
                LINK_LENGTH.mul_add(angle_1.sin(), state[0]),
            ),
            LINK_LENGTH.mul_add(angle_1.cos(), LINK_LENGTH * absolute_angle_2.cos()),
        )
    }

    /// Return coupled cart and link state derivatives.
    fn derivatives(state: &[f32; 6], action: f32) -> [f32; 6] {
        let [_position, angle_1, angle_2, cart_velocity, angular_velocity_1, angular_velocity_2] =
            *state;
        let cart_acceleration = action.mul_add(FORCE_GAIN, -0.1 * cart_velocity);
        let absolute_angle_2 = angle_1 + angle_2;
        let absolute_velocity_2 = angular_velocity_1 + angular_velocity_2;
        let coupling = 0.25
            * angle_2.sin()
            * angular_velocity_2.mul_add(
                angular_velocity_2,
                2.0 * angular_velocity_1 * angular_velocity_2,
            );
        let angular_acceleration_1 = 0.05f32.mul_add(
            -angular_velocity_1,
            (GRAVITY / LINK_LENGTH).mul_add(
                angle_1.sin(),
                -(cart_acceleration / LINK_LENGTH * angle_1.cos()),
            ) + coupling,
        );
        let absolute_acceleration_2 = 0.2f32.mul_add(
            -coupling,
            0.05f32.mul_add(
                -absolute_velocity_2,
                (GRAVITY / LINK_LENGTH).mul_add(
                    absolute_angle_2.sin(),
                    -(cart_acceleration / LINK_LENGTH * absolute_angle_2.cos()),
                ),
            ),
        );
        [
            cart_velocity,
            angular_velocity_1,
            angular_velocity_2,
            cart_acceleration,
            angular_acceleration_1,
            0.5f32.mul_add(-angular_acceleration_1, absolute_acceleration_2),
        ]
    }

    /// Integrate five `MuJoCo` 0.01-second RK4 frames.
    fn integrate(&mut self, action: f32) {
        let state = self.state;
        let k_1 = Self::derivatives(&state, action);
        let k_2 = Self::derivatives(&add_scaled(state, k_1, TIME_STEP * 0.5), action);
        let k_3 = Self::derivatives(&add_scaled(state, k_2, TIME_STEP * 0.5), action);
        let k_4 = Self::derivatives(&add_scaled(state, k_3, TIME_STEP), action);
        for index in 0..6 {
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
        self.constraint_force = 0.0;
        if self.state[0] < -1.0 {
            self.state[0] = -1.0;
            self.state[3] = self.state[3].max(0.0);
            self.constraint_force = action.abs() * FORCE_GAIN;
        } else if self.state[0] > 1.0 {
            self.state[0] = 1.0;
            self.state[3] = self.state[3].min(0.0);
            self.constraint_force = -action.abs() * FORCE_GAIN;
        }
    }

    /// Return Gymnasium's dense reward and health result.
    fn reward(&self) -> (f64, bool) {
        let (tip_x, tip_y) = Self::tip_position(&self.state);
        let healthy = tip_y > 1.0;
        let distance_penalty =
            f64::from(tip_y - 2.0).mul_add(f64::from(tip_y - 2.0), 0.01 * f64::from(tip_x).powi(2));
        let velocity_penalty = 0.001f64.mul_add(
            f64::from(self.state[4]).powi(2),
            0.005 * f64::from(self.state[5]).powi(2),
        );
        (
            if healthy { 10.0 } else { 0.0 } - distance_penalty - velocity_penalty,
            healthy,
        )
    }

    /// Return the discrete-time linear controller used for reward guidance.
    fn stabilizing_action(observation: &[f32]) -> f32 {
        let position = observation.first().copied().unwrap_or(0.0);
        let angle_1 = observation
            .get(1)
            .copied()
            .unwrap_or(0.0)
            .atan2(observation.get(3).copied().unwrap_or(1.0));
        let angle_2 = observation
            .get(2)
            .copied()
            .unwrap_or(0.0)
            .atan2(observation.get(4).copied().unwrap_or(1.0));
        let cart_velocity = observation.get(5).copied().unwrap_or(0.0);
        let angular_velocity_1 = observation.get(6).copied().unwrap_or(0.0);
        let angular_velocity_2 = observation.get(7).copied().unwrap_or(0.0);
        let feedback = 0.036_614_1_f32.mul_add(
            position,
            2.548_29f32.mul_add(
                -angular_velocity_2,
                0.687_094_8f32.mul_add(
                    angular_velocity_1,
                    0.090_126_6f32.mul_add(
                        cart_velocity,
                        0.953_486_1f32.mul_add(angle_1, -(17.747_795 * angle_2)),
                    ),
                ),
            ),
        );
        (-feedback).clamp(-ACTION_LIMIT, ACTION_LIMIT)
    }
}

impl Env for InvertedDoublePendulum {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        for (index, value) in self.state[..3].iter_mut().enumerate() {
            *value = if index == 0 {
                self.rng.f64_between(-0.1, 0.1) as f32
            } else {
                self.rng.f32_between(-0.1, 0.1)
            };
        }
        for value in &mut self.state[3..] {
            *value = self.rng.standard_normal_f32() * 0.1;
        }
        self.constraint_force = 0.0;
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
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
        self.elapsed_steps += 1;
        let (reward, healthy) = self.reward();
        let status = if !healthy {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
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

impl ContinuousPpoExample for InvertedDoublePendulum {
    const ENV_NAME: &'static str = "inverted-double-pendulum";
    const GYMNASIUM_ID: &'static str = "InvertedDoublePendulum-v5";
    const OBSERVATION_DIM: usize = 9;
    const ACTION_LOW: &'static [f32] = &[-1.0];
    const ACTION_HIGH: &'static [f32] = &[1.0];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 16;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.0003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const DEFAULT_REWARD_SCALE: f64 = 1.0;
    const SOLVED_MEAN_REWARD: f64 = 9_100.0;
    const GIF_PATH: &'static str = "docs/images/inverted-double-pendulum.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 2_048;
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
            minibatch_sequences: 8,
            initial_log_std: -1.0,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let mut encoded = vec![0.0; Self::OBSERVATION_DIM];
        *encoded.get_mut(0).expect("fixed example index is valid") =
            Self::stabilizing_action(observation);
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut demonstrations = Vec::with_capacity(sample_count);
        let mut episode = 0_u64;
        while demonstrations.len() < sample_count {
            let mut env = Self::default();
            let mut observation = env.reset(Some(seed.wrapping_add(episode))).observation;
            for _ in 0..32 {
                let action = Self::stabilizing_action(&observation);
                demonstrations.push(RecurrentBehaviorSample {
                    observation: Self::encode_observation(&observation),
                    action: vec![action],
                });
                if demonstrations.len() == sample_count {
                    break;
                }
                let transition = env.step(vec![action]);
                observation = transition.observation;
                if transition.status != EpisodeStatus::Continuing {
                    break;
                }
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
    run_continuous_workflow::<InvertedDoublePendulum>()
}

/// Add one scaled derivative to one state.
fn add_scaled(state: [f32; 6], derivative: [f32; 6], scale: f32) -> [f32; 6] {
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
        ContinuousPpoExample, Env, EpisodeStatus, Error, InvertedDoublePendulum, Path, LINK_LENGTH,
        TIME_STEP,
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
            app.insert_resource(ClearColor(Color::srgb_u8(77, 86, 80)));
        }
    }

    /// Policy playback state shown in the `InvertedDoublePendulum` scene.
    #[derive(Resource)]
    struct VisualInvertedDoublePendulum {
        /// Greedy recurrent checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live environment.
        env: InvertedDoublePendulum,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Actor memory for this episode.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Twenty-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic olive cart.
    #[derive(Component)]
    struct DoublePendulumCart;

    /// Dynamic cyan link indexed from the cart.
    #[derive(Component)]
    struct DoublePendulumLink(usize);

    /// Run the MuJoCo-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = InvertedDoublePendulum::ppo_config(0.0003, 0.001);
        let policy = RecurrentPpoPolicy::load(checkpoint, 9, 9, 1, &[-1.0], &[1.0], &config)?;
        let memory = policy.initial_memory();
        let mut env = InvertedDoublePendulum::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualInvertedDoublePendulum {
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
                        title: "bevy-gym InvertedDoublePendulum-v5".into(),
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

    /// Spawn the XML rail, cart, links, camera, and light.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.30, 8.245).looking_at(Vec3::ZERO, Vec3::Y),
            Name::new("InvertedDoublePendulum Camera"),
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
        let horizontal = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.02, 2.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.3, 0.3, 0.7),
                perceptual_roughness: 0.45,
                ..default()
            })),
            Transform::from_rotation(horizontal),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.1, 0.2))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.7, 0.7, 0.0),
                perceptual_roughness: 0.4,
                ..default()
            })),
            Transform::from_rotation(horizontal),
            DoublePendulumCart,
        ));
        let cyan = materials.add(StandardMaterial {
            base_color: Color::srgb(0.0, 0.7, 0.7),
            perceptual_roughness: 0.35,
            ..default()
        });
        for index in 0..2 {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(0.045, 0.51))),
                MeshMaterial3d(cyan.clone()),
                Transform::from_xyz(0.0, LINK_LENGTH * (index as f32 + 0.5), 0.0),
                DoublePendulumLink(index),
            ));
        }
    }

    /// Advance policy playback at `MuJoCo`'s 20 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualInvertedDoublePendulum>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of deterministic policy transitions.
    fn advance_visual(visual: &mut VisualInvertedDoublePendulum, steps: usize) {
        for _ in 0..steps {
            let encoded = InvertedDoublePendulum::encode_observation(&visual.observation);
            let Ok(action) = visual.policy.mean_action(&encoded, &visual.memory) else {
                return;
            };
            visual.memory = visual.policy.initial_memory();
            let transition = visual.env.step(action.action);
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(Some(12_345)).observation;
                visual.memory = visual.policy.initial_memory();
                break;
            }
        }
    }

    /// Synchronize cart and both pole transforms.
    fn sync_scene(
        visual: Res<'_, VisualInvertedDoublePendulum>,
        mut cart: Single<'_, '_, &mut Transform, With<DoublePendulumCart>>,
        mut links: Query<
            '_,
            '_,
            (&DoublePendulumLink, &mut Transform),
            Without<DoublePendulumCart>,
        >,
    ) {
        apply_visual_state(&visual.env, &mut cart, &mut links);
    }

    /// Apply one state to the XML-aligned double-pole geometry.
    fn apply_visual_state(
        env: &InvertedDoublePendulum,
        cart: &mut Transform,
        links: &mut Query<
            '_,
            '_,
            (&DoublePendulumLink, &mut Transform),
            Without<DoublePendulumCart>,
        >,
    ) {
        let position = env.state[0];
        let angle_1 = env.state[1];
        let absolute_angle_2 = angle_1 + env.state[2];
        let direction_1 = Vec2::new(angle_1.sin(), angle_1.cos());
        let direction_2 = Vec2::new(absolute_angle_2.sin(), absolute_angle_2.cos());
        let joint = Vec2::new(position, 0.0) + direction_1 * LINK_LENGTH;
        cart.translation = Vec3::new(position, 0.0, 0.0);
        for (link, mut transform) in links.iter_mut() {
            let (start, direction, angle) = if link.0 == 0 {
                (Vec2::new(position, 0.0), direction_1, angle_1)
            } else {
                (joint, direction_2, absolute_angle_2)
            };
            let center = start + direction * (LINK_LENGTH * 0.5);
            transform.translation = Vec3::new(center.x, center.y, 0.0);
            transform.rotation = Quat::from_rotation_z(-angle);
        }
    }

    /// Capture five policy transitions per GIF frame.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualInvertedDoublePendulum>,
        mut cart: Single<'_, '_, &mut Transform, With<DoublePendulumCart>>,
        mut links: Query<
            '_,
            '_,
            (&DoublePendulumLink, &mut Transform),
            Without<DoublePendulumCart>,
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
        apply_visual_state(&visual.env, &mut cart, &mut links);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "inverted-double-pendulum", 480, 480, |frames| {
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
    fn upright_observation_has_two_unit_cosines() {
        let pendulum = InvertedDoublePendulum::from_state([0.0; 6]);

        assert_eq!(
            pendulum.observation(),
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn upright_reward_matches_the_reference_formula() {
        let mut pendulum = InvertedDoublePendulum::from_state([0.0; 6]);

        let step = pendulum.step(vec![0.0]);

        assert!((step.reward - 9.36).abs() < 1e-6);
        assert_eq!(step.status, EpisodeStatus::Continuing);
    }

    #[test]
    fn a_low_tip_terminates() {
        let mut pendulum = InvertedDoublePendulum::from_state([0.0, 0.8, 0.8, 0.0, 0.0, 0.0]);

        let step = pendulum.step(vec![0.0]);

        assert_eq!(step.status, EpisodeStatus::Terminated);
    }

    #[test]
    fn linear_controller_clears_the_reward_gate() {
        let mut total = 0.0;
        let mut completed = 0_u32;
        let mut total_steps = 0_u32;
        let episodes = 100_u32;
        for seed in 0..episodes {
            let mut pendulum = InvertedDoublePendulum::default();
            let mut observation = pendulum.reset(Some(u64::from(seed))).observation;
            loop {
                let action = InvertedDoublePendulum::stabilizing_action(&observation);
                let step = pendulum.step(vec![action]);
                total += step.reward;
                total_steps += 1;
                observation = step.observation;
                if step.status != EpisodeStatus::Continuing {
                    completed += u32::from(step.status == EpisodeStatus::Truncated);
                    break;
                }
            }
        }
        let mean = total / f64::from(episodes);

        assert!(
            mean >= 9_100.0,
            "controller mean={mean:.3} completion={completed}/{episodes} mean_steps={:.1}",
            f64::from(total_steps) / f64::from(episodes)
        );
    }
}
