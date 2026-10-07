//! ## Description
//!
//! The Acrobot environment is based on Sutton's work in
//! [Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html)
//! and [Sutton and Barto's book](http://www.incompleteideas.net/book/the-book-2nd.html).
//! The system consists of two links connected linearly to form a chain, with one end of
//! the chain fixed. The joint between the two links is actuated. The goal is to apply
//! torques on the actuated joint to swing the free end of the linear chain above a
//! given height while starting from the initial state of hanging downwards.
//!
//! As seen in the **Gif**: two blue links connected by two green joints. The joint in
//! between the two links is actuated. The goal is to swing the free end of the outer-link
//! to reach the target height (black horizontal line above system) by applying torque on
//! the actuator.
//!
//! ## Action Space
//!
//! The action is discrete, deterministic, and represents the torque applied on the actuated
//! joint between the two links.
//!
//! | Num | Action                                | Unit         |
//! |-----|---------------------------------------|--------------|
//! | 0   | apply -1 torque to the actuated joint | torque (N m) |
//! | 1   | apply 0 torque to the actuated joint  | torque (N m) |
//! | 2   | apply 1 torque to the actuated joint  | torque (N m) |
//!
//! ## Observation Space
//!
//! The observation is a `ndarray` with shape `(6,)` that provides information about the
//! two rotational joint angles as well as their angular velocities:
//!
//! | Num | Observation                  | Min                 | Max               |
//! |-----|------------------------------|---------------------|-------------------|
//! | 0   | Cosine of `theta1`           | -1                  | 1                 |
//! | 1   | Sine of `theta1`             | -1                  | 1                 |
//! | 2   | Cosine of `theta2`           | -1                  | 1                 |
//! | 3   | Sine of `theta2`             | -1                  | 1                 |
//! | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
//! | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |
//!
//! where
//! - `theta1` is the angle of the first joint, where an angle of 0 indicates the first link is pointing directly
//!   downwards.
//! - `theta2` is ***relative to the angle of the first link.***
//!   An angle of 0 corresponds to having the same angle between the two links.
//!
//! The angular velocities of `theta1` and `theta2` are bounded at ±4π, and ±9π rad/s respectively.
//! A state of `[1, 0, 1, 0, ..., ...]` indicates that both links are pointing downwards.
//!
//! ## Rewards
//!
//! The goal is to have the free end reach a designated target height in as few steps as possible,
//! and as such all steps that do not reach the goal incur a reward of -1.
//! Achieving the target height results in termination with a reward of 0. The reward threshold is -100.
//!
//! ## Starting State
//!
//! Each parameter in the underlying state (`theta1`, `theta2`, and the two angular velocities) is initialized
//! uniformly between -0.1 and 0.1. This means both links are pointing downwards with some initial stochasticity.
//!
//! ## Episode End
//!
//! The episode ends if one of the following occurs:
//! 1. Termination: The free end reaches the target height, which is constructed as:
//!    `-cos(theta1) - cos(theta2 + theta1) > 1.0`
//! 2. Truncation: Episode length is greater than 500 (200 for v0)
//!
//! ## Arguments
//!
//! Acrobot only has `render_mode` as a keyword for `gymnasium.make`.
//! On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make('Acrobot-v1', render_mode="rgb_array")
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<AcrobotEnv<Acrobot-v1>>>>>
//! >>> env.reset(seed=123, options={"low": -0.2, "high": 0.2})  # default low=-0.1, high=0.1
//! (array([ 0.997341  ,  0.07287608,  0.9841162 , -0.17752565, -0.11185605,
//!        -0.12625128], dtype=float32), {})
//!
//! ```
//!
//! By default, the dynamics of the acrobot follow those described in Sutton and Barto's book
//! [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/11/node4.html).
//! However, a `book_or_nips` parameter can be modified to change the pendulum dynamics to those described
//! in the original [NeurIPS paper](https://papers.nips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html).
//!
//! ```python
//! # To change the dynamics as described above
//! env.unwrapped.book_or_nips = 'nips'
//! ```
//!
//! See the following note for details:
//!
//! > The dynamics equations were missing some terms in the NIPS paper which are present in the book.
//! > R. Sutton confirmed in personal correspondence that the experimental results shown in the paper and the book were
//! > generated with the equations shown in the book. However, there is the option to run the domain with the paper equations
//! > by setting `book_or_nips = 'nips'`
//!
//! ## Version History
//!
//! - v1: Maximum number of steps increased from 200 to 500. The observation space for v0 provided direct readings of
//!   `theta1` and `theta2` in radians, having a range of `[-pi, pi]`. The v1 observation space as described here provides the
//!   sine and cosine of each angle instead.
//! - v0: Initial versions release
//!
//! ## References
//! - Sutton, R. S. (1996). Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding.
//!   In D. Touretzky, M. C. Mozer, & M. Hasselmo (Eds.), Advances in Neural Information Processing Systems (Vol. 8).
//!   MIT Press. <https://proceedings.neurips.cc/paper/1995/file/8f1d43620bc6bb580df6e80b0dc05c48-Paper.pdf>
//! - Sutton, R. S., Barto, A. G. (2018 ). Reinforcement Learning: An Introduction. The MIT Press.

use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::DqnConfig;
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::{run_discrete_workflow, DiscreteDqnExample, DqnAction};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use serde_json as _;

/// `DT` used by this example.
const DT: f64 = 0.2;
/// `LINK_LENGTH` used by this example.
const LINK_LENGTH: f64 = 1.0;
/// `LINK_MASS` used by this example.
const LINK_MASS: f64 = 1.0;
/// `LINK_COM_POSITION` used by this example.
const LINK_COM_POSITION: f64 = 0.5;
/// `LINK_MOI` used by this example.
const LINK_MOI: f64 = 1.0;
/// `MAX_VELOCITY_1` used by this example.
const MAX_VELOCITY_1: f64 = 4.0 * std::f64::consts::PI;
/// `MAX_VELOCITY_2` used by this example.
const MAX_VELOCITY_2: f64 = 9.0 * std::f64::consts::PI;
/// `GRAVITY` used by this example.
const GRAVITY: f64 = 9.8;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 500;

/// Acrobot-v1 action in Gymnasium order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AcrobotAction {
    /// Apply negative unit torque.
    Negative,
    /// Apply zero torque.
    Coast,
    /// Apply positive unit torque.
    Positive,
}

impl AcrobotAction {
    /// Return the torque represented by the action.
    const fn torque(self) -> f64 {
        match self {
            Self::Negative => -1.0,
            Self::Coast => 0.0,
            Self::Positive => 1.0,
        }
    }
}

impl DqnAction for AcrobotAction {
    fn from_index(index: usize) -> Self {
        match index % 3 {
            0 => Self::Negative,
            1 => Self::Coast,
            _ => Self::Positive,
        }
    }

    fn as_index(self) -> usize {
        match self {
            Self::Negative => 0,
            Self::Coast => 1,
            Self::Positive => 2,
        }
    }
}

/// Exact default Gymnasium Acrobot-v1 dynamics.
#[derive(Debug, Clone)]
struct Acrobot {
    /// Joint angles and angular velocities.
    state: [f64; 4],
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset generator.
    rng: SplitMix64,
}

impl Default for Acrobot {
    fn default() -> Self {
        Self {
            state: [0.0; 4],
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Acrobot {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    const fn from_state(state: [f64; 4]) -> Self {
        Self {
            state,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }

    /// Encode the internal state as Gymnasium's six observations.
    fn observation(&self) -> Vec<f32> {
        let [theta_1, theta_2, velocity_1, velocity_2] = self.state;
        vec![
            theta_1.cos() as f32,
            theta_1.sin() as f32,
            theta_2.cos() as f32,
            theta_2.sin() as f32,
            velocity_1 as f32,
            velocity_2 as f32,
        ]
    }

    /// Return the free-end height used by Gymnasium's terminal condition.
    fn free_end_height(state: &[f64; 4]) -> f64 {
        -state[0].cos() - (state[0] + state[1]).cos()
    }

    /// Return whether the free end crossed the target height.
    fn is_terminal_state(state: &[f64; 4]) -> bool {
        Self::free_end_height(state) > 1.0
    }

    /// Return the four Acrobot state derivatives for one torque.
    fn derivatives(state: &[f64; 4], torque: f64) -> [f64; 4] {
        let [theta_1, theta_2, velocity_1, velocity_2] = *state;
        let d_1 = 2.0f64.mul_add(
            LINK_MOI,
            LINK_MASS.mul_add(
                LINK_COM_POSITION.powi(2),
                LINK_MASS
                    * (2.0 * LINK_LENGTH * LINK_COM_POSITION).mul_add(
                        theta_2.cos(),
                        LINK_COM_POSITION.mul_add(LINK_COM_POSITION, LINK_LENGTH.powi(2)),
                    ),
            ),
        );
        let d_2 = LINK_MASS.mul_add(
            LINK_COM_POSITION.mul_add(
                LINK_COM_POSITION,
                LINK_LENGTH * LINK_COM_POSITION * theta_2.cos(),
            ),
            LINK_MOI,
        );
        let phi_2 = LINK_MASS
            * LINK_COM_POSITION
            * GRAVITY
            * (theta_1 + theta_2 - std::f64::consts::FRAC_PI_2).cos();
        let phi_1 =
            (LINK_MASS.mul_add(LINK_COM_POSITION, LINK_MASS * LINK_LENGTH) * GRAVITY).mul_add(
                (theta_1 - std::f64::consts::FRAC_PI_2).cos(),
                (-LINK_MASS * LINK_LENGTH * LINK_COM_POSITION * velocity_2.powi(2)).mul_add(
                    theta_2.sin(),
                    -(2.0
                        * LINK_MASS
                        * LINK_LENGTH
                        * LINK_COM_POSITION
                        * velocity_2
                        * velocity_1
                        * theta_2.sin()),
                ),
            ) + phi_2;
        let acceleration_2 = ((LINK_MASS * LINK_LENGTH * LINK_COM_POSITION * velocity_1.powi(2))
            .mul_add(-theta_2.sin(), (d_2 / d_1).mul_add(phi_1, torque))
            - phi_2)
            / (LINK_MASS.mul_add(LINK_COM_POSITION.powi(2), LINK_MOI) - d_2.powi(2) / d_1);
        let acceleration_1 = -d_2.mul_add(acceleration_2, phi_1) / d_1;
        [velocity_1, velocity_2, acceleration_1, acceleration_2]
    }

    /// Integrate Gymnasium's equations with one fourth-order Runge-Kutta step.
    fn integrate(&mut self, torque: f64) {
        let state = self.state;
        let k_1 = Self::derivatives(&state, torque);
        let k_2 = Self::derivatives(&add_scaled(state, k_1, DT * 0.5), torque);
        let k_3 = Self::derivatives(&add_scaled(state, k_2, DT * 0.5), torque);
        let k_4 = Self::derivatives(&add_scaled(state, k_3, DT), torque);
        for index in 0..4 {
            *self
                .state
                .get_mut(index)
                .expect("fixed example index is valid") += DT
                * (2.0f64.mul_add(
                    k_3.get(index)
                        .copied()
                        .expect("fixed example index is valid"),
                    2.0f64.mul_add(
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
        self.state[0] = wrap_angle(self.state[0]);
        self.state[1] = wrap_angle(self.state[1]);
        self.state[2] = self.state[2].clamp(-MAX_VELOCITY_1, MAX_VELOCITY_1);
        self.state[3] = self.state[3].clamp(-MAX_VELOCITY_2, MAX_VELOCITY_2);
    }
}

impl Env for Acrobot {
    type Action = AcrobotAction;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state = std::array::from_fn(|_| self.rng.f64_between(-0.1, 0.1));
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        self.integrate(action.torque());
        self.elapsed_steps += 1;
        let terminated = Self::is_terminal_state(&self.state);
        let status = if terminated {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation: self.observation(),
            reward: if terminated { 0.0 } else { -1.0 },
            status,
            info: (),
        }
    }
}

impl DiscreteDqnExample for Acrobot {
    const ENV_NAME: &'static str = "acrobot";
    const GYMNASIUM_ID: &'static str = "Acrobot-v1";
    const OBSERVATION_DIM: usize = 6;
    const ACTION_COUNT: usize = 3;
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_LEARNING_RATE: f64 = 0.0003;
    const DEFAULT_REWARD_SCALE: f64 = 10.0;
    const SOLVED_MEAN_REWARD: f64 = -100.0;
    const GIF_PATH: &'static str = "docs/images/acrobot.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 0.99,
            learning_rate,
            replay_capacity: 200_000,
            min_replay_size: 5_000,
            batch_size: 128,
            target_update_interval: 1_000,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 200_000,
            ..DqnConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let mut encoded = observation.to_vec();
        if encoded.len() >= 6 {
            *encoded.get_mut(4).expect("fixed example index is valid") /= MAX_VELOCITY_1 as f32;
            *encoded.get_mut(5).expect("fixed example index is valid") /= MAX_VELOCITY_2 as f32;
        }
        encoded
    }

    fn training_reward(
        observation: &[f32],
        _action: Self::Action,
        environment_reward: f64,
        next_observation: &[f32],
        status: EpisodeStatus,
        reward_scale: f64,
    ) -> f64 {
        let potential = observation_height(observation) - 2.0;
        let next_potential = if status == EpisodeStatus::Terminated {
            0.0
        } else {
            observation_height(next_observation) - 2.0
        };
        reward_scale.mul_add(
            0.99f64.mul_add(next_potential, -potential),
            environment_reward,
        )
    }

    fn is_success(_final_observation: &[f32], _total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Terminated
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
    run_discrete_workflow::<Acrobot>()
}

/// Add one scaled derivative to one state.
fn add_scaled(state: [f64; 4], derivative: [f64; 4], scale: f64) -> [f64; 4] {
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

/// Wrap one angle into Gymnasium's inclusive `[-pi, pi]` range.
fn wrap_angle(angle: f64) -> f64 {
    angle.sin().atan2(angle.cos())
}

/// Recover free-end height from Gymnasium's trigonometric observation.
fn observation_height(observation: &[f32]) -> f64 {
    let cos_1 = f64::from(observation.first().copied().unwrap_or(1.0));
    let sin_1 = f64::from(observation.get(1).copied().unwrap_or(0.0));
    let cos_2 = f64::from(observation.get(2).copied().unwrap_or(1.0));
    let sin_2 = f64::from(observation.get(3).copied().unwrap_or(0.0));
    -cos_1 - (cos_1 * cos_2 - sin_1 * sin_2)
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
    fn unit_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        (value >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Generate one scalar in `[low, high)`.
    fn f64_between(&mut self, low: f64, high: f64) -> f64 {
        self.unit_f64().mul_add(high - low, low)
    }
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        Acrobot, AcrobotAction, DiscreteDqnExample, DqnAction, Env, EpisodeStatus, Error, Path,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;
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

    /// Policy playback state shown in the Acrobot scene.
    #[derive(Resource)]
    struct VisualAcrobot {
        /// Greedy checkpoint policy.
        policy: DqnPolicy,
        /// Live exact environment.
        env: Acrobot,
        /// Current exact observation.
        observation: Vec<f32>,
    }

    /// Thirty-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic rectangular link with its chain index.
    #[derive(Component)]
    struct AcrobotLink(usize);

    /// Dynamic circular joint with its chain index.
    #[derive(Component)]
    struct AcrobotJoint(usize);

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Acrobot::dqn_config(Acrobot::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(checkpoint, 6, 3, &config.hidden_sizes)?;
        let mut env = Acrobot::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualAcrobot {
            policy,
            env,
            observation,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            1.0 / 30.0,
            TimerMode::Repeating,
        )))
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym Acrobot-v1".into(),
                        resolution: WindowResolution::new(500, 500),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_scene)
        .add_systems(Update, draw_goal_line);
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

    /// Spawn Gymnasium's cyan links and yellow joints.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        const SCALE: f32 = 500.0 / 4.4;
        const LINK_WIDTH: f32 = 0.2 * SCALE;
        let cyan = materials.add(Color::srgb_u8(0, 204, 204));
        let yellow = materials.add(Color::srgb_u8(204, 204, 0));
        commands.spawn((Camera2d, Name::new("Acrobot Camera")));
        for index in 0..2 {
            commands.spawn((
                Mesh2d(meshes.add(Rectangle::new(SCALE, LINK_WIDTH))),
                MeshMaterial2d(cyan.clone()),
                Transform::from_xyz(0.0, 0.0, 1.0),
                AcrobotLink(index),
            ));
            commands.spawn((
                Mesh2d(meshes.add(Circle::new(LINK_WIDTH * 0.5))),
                MeshMaterial2d(yellow.clone()),
                Transform::from_xyz(0.0, 0.0, 2.0),
                AcrobotJoint(index),
            ));
        }
    }

    /// Draw Gymnasium's one-unit target height.
    fn draw_goal_line(mut gizmos: Gizmos<'_, '_>) {
        const SCALE: f32 = 500.0 / 4.4;
        gizmos.line_2d(
            Vec2::new(-250.0, SCALE),
            Vec2::new(250.0, SCALE),
            Color::BLACK,
        );
    }

    /// Advance policy playback at Gymnasium's 30 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualAcrobot>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of greedy environment transitions.
    fn advance_visual(visual: &mut VisualAcrobot, steps: usize) {
        for _ in 0..steps {
            let encoded = Acrobot::encode_observation(&visual.observation);
            let Ok(action_index) = visual.policy.greedy_action(&encoded) else {
                return;
            };
            let transition = visual.env.step(AcrobotAction::from_index(action_index));
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(None).observation;
                break;
            }
        }
    }

    /// Synchronize the two links and joints with Gymnasium's coordinates.
    fn sync_scene(
        visual: Res<'_, VisualAcrobot>,
        mut links: Query<'_, '_, (&AcrobotLink, &mut Transform), Without<AcrobotJoint>>,
        mut joints: Query<'_, '_, (&AcrobotJoint, &mut Transform), Without<AcrobotLink>>,
    ) {
        apply_visual_state(&visual.env, &mut links, &mut joints);
    }

    /// Apply one exact Acrobot geometry state.
    fn apply_visual_state(
        env: &Acrobot,
        links: &mut Query<'_, '_, (&AcrobotLink, &mut Transform), Without<AcrobotJoint>>,
        joints: &mut Query<'_, '_, (&AcrobotJoint, &mut Transform), Without<AcrobotLink>>,
    ) {
        const SCALE: f32 = 500.0 / 4.4;
        let theta_1 = env.state[0] as f32;
        let theta_2 = env.state[1] as f32;
        let direction_1 = Vec2::from_angle(theta_1 - std::f32::consts::FRAC_PI_2);
        let direction_2 = Vec2::from_angle(theta_1 + theta_2 - std::f32::consts::FRAC_PI_2);
        let elbow = direction_1 * SCALE;
        for (link, mut transform) in links.iter_mut() {
            let (anchor, direction) = if link.0 == 0 {
                (Vec2::ZERO, direction_1)
            } else {
                (elbow, direction_2)
            };
            transform.translation = (anchor + direction * (SCALE * 0.5)).extend(1.0);
            transform.rotation = Quat::from_rotation_z(direction.to_angle());
        }
        for (joint, mut transform) in joints.iter_mut() {
            let position = if joint.0 == 0 { Vec2::ZERO } else { elbow };
            transform.translation = position.extend(2.0);
        }
    }

    /// Capture four policy transitions per GIF frame.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualAcrobot>,
        mut links: Query<'_, '_, (&AcrobotLink, &mut Transform), Without<AcrobotJoint>>,
        mut joints: Query<'_, '_, (&AcrobotJoint, &mut Transform), Without<AcrobotLink>>,
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
            advance_visual(&mut visual, 4);
        }
        apply_visual_state(&visual.env, &mut links, &mut joints);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "acrobot", 500, 500, |frames| {
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
    fn hanging_state_with_zero_torque_is_an_equilibrium() {
        let mut acrobot = Acrobot::from_state([0.0; 4]);

        let step = acrobot.step(AcrobotAction::Coast);

        let expected = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        for (actual, expected) in step.observation.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(step.reward, -1.0);
        assert_eq!(step.status, EpisodeStatus::Continuing);
    }

    #[test]
    fn free_end_above_the_goal_terminates() {
        assert!(Acrobot::is_terminal_state(&[
            std::f64::consts::PI,
            0.0,
            0.0,
            0.0
        ]));
    }

    #[test]
    fn episode_truncates_at_five_hundred_transitions() {
        let mut acrobot = Acrobot::from_state([0.0; 4]);
        acrobot.elapsed_steps = MAX_EPISODE_STEPS - 1;

        let step = acrobot.step(AcrobotAction::Coast);

        assert_eq!(step.status, EpisodeStatus::Truncated);
    }
}
