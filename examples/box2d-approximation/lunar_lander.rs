//! ## Description
//! This environment is a classic rocket trajectory optimization problem.
//! According to Pontryagin's maximum principle, it is optimal to fire the
//! engine at full throttle or turn it off. This is the reason why this
//! environment has discrete actions: engine on or off.
//!
//! There are two environment versions: discrete or continuous.
//! The landing pad is always at coordinates (0,0). The coordinates are the
//! first two numbers in the state vector.
//! Landing outside of the landing pad is possible. Fuel is infinite, so an agent
//! can learn to fly and then land on its first attempt.
//!
//! To see a heuristic landing, run:
//! ```shell
//! python gymnasium/envs/box2d/lunar_lander.py
//! ```
//!
//! ## Action Space
//! There are four discrete actions available:
//! - 0: do nothing
//! - 1: fire left orientation engine
//! - 2: fire main engine
//! - 3: fire right orientation engine
//!
//! ## Observation Space
//! The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear
//! velocities in `x` & `y`, its angle, its angular velocity, and two booleans
//! that represent whether each leg is in contact with the ground or not.
//!
//! ## Rewards
//! After every step a reward is granted. The total reward of an episode is the
//! sum of the rewards for all the steps within that episode.
//!
//! For each step, the reward:
//! - is increased/decreased the closer/further the lander is to the landing pad.
//! - is increased/decreased the slower/faster the lander is moving.
//! - is decreased the more the lander is tilted (angle not horizontal).
//! - is increased by 10 points for each leg that is in contact with the ground.
//! - is decreased by 0.03 points each frame a side engine is firing.
//! - is decreased by 0.3 points each frame the main engine is firing.
//!
//! The episode receive an additional reward of -100 or +100 points for crashing or landing safely respectively.
//!
//! An episode is considered a solution if it scores at least 200 points.
//!
//! ## Starting State
//! The lander starts at the top center of the viewport with a random initial
//! force applied to its center of mass.
//!
//! ## Episode Termination
//! The episode finishes if:
//! 1) the lander crashes (the lander body gets in contact with the moon);
//! 2) the lander gets outside of the viewport (`x` coordinate is greater than 1);
//! 3) the lander is not awake. From the [Box2D docs](https://box2d.org/documentation/md_simulation.html),
//!    a body which is not awake is a body which doesn't move and doesn't collide with any other body:
//! > When Box2D determines that a body (or group of bodies) has come to rest,
//! > the body enters a sleep state which has very little CPU overhead. If a
//! > body is awake and collides with a sleeping body, then the sleeping body
//! > wakes up. Bodies will also wake up if a joint or contact attached to
//! > them is destroyed.
//!
//! ## Arguments
//!
//! Lunar Lander has a large number of arguments
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
//! ...                enable_wind=False, wind_power=15.0, turbulence_power=1.5)
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<LunarLander<LunarLander-v3>>>>>
//!
//! ```
//!
//!  * `continuous` determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used with the
//!    action space being `Discrete(4)` or `Box(-1, +1, (2,), dtype=np.float32)` respectively.
//!    For continuous actions, the first coordinate of an action determines the throttle of the main engine, while the second
//!    coordinate specifies the throttle of the lateral boosters. Given an action `np.array([main, lateral])`, the main
//!    engine will be turned off completely if `main < 0` and the throttle scales affinely from 50% to 100% for
//!    `0 <= main <= 1` (in particular, the main engine doesn't work  with less than 50% power).
//!    Similarly, if `-0.5 < lateral < 0.5`, the lateral boosters will not fire at all. If `lateral < -0.5`, the left
//!    booster will fire, and if `lateral > 0.5`, the right booster will fire. Again, the throttle scales affinely
//!    from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).
//!
//! * `gravity` dictates the gravitational constant, this is bounded to be within 0 and -12. Default is -10.0
//!
//! * `enable_wind` determines if there will be wind effects applied to the lander. The wind is generated using
//!   the function `tanh(sin(2 k (t+C)) + sin(pi k (t+C)))` where `k` is set to 0.01 and `C` is sampled randomly between -9999 and 9999.
//!
//! * `wind_power` dictates the maximum magnitude of linear wind applied to the craft. The recommended value for
//!   `wind_power` is between 0.0 and 20.0.
//!
//! * `turbulence_power` dictates the maximum magnitude of rotational wind applied to the craft.
//!   The recommended value for `turbulence_power` is between 0.0 and 2.0.
//!
//! ## Version History
//! - v3:
//!     - Reset wind and turbulence offset (`C`) whenever the environment is reset to ensure statistical independence between consecutive episodes (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/954)).
//!     - Fix non-deterministic behaviour due to not fully destroying the world (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/728)).
//!     - Changed observation space for `x`, `y`  coordinates from $\pm 1.5$ to $\pm 2.5$, velocities from $\pm 5$ to $\pm 10$ and angles from $\pm \pi$ to $\pm 2\pi$ (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/752)).
//! - v2: Count energy spent and in v0.24, added turbulence with wind power and `turbulence_power` parameters
//! - v1: Legs contact with ground added in state vector; contact with ground give +10 reward points, and -10 if then lose contact; reward renormalized to 200; harder initial random push.
//! - v0: Initial version
//!
//! ## Notes
//!
//! There are several unexpected bugs with the implementation of the environment.
//!
//! 1. The position of the side thrusters on the body of the lander changes, depending on the orientation of the lander.
//!    This in turn results in an orientation dependent torque being applied to the lander.
//!
//! 2. The units of the state are not consistent. I.e.
//! * The angular velocity is in units of 0.4 radians per second. In order to convert to radians per second, the value needs to be multiplied by a factor of 2.5.
//!
//! For the default values of `VIEWPORT_W`, `VIEWPORT_H`, SCALE, and FPS, the scale factors equal:
//! 'x': 10, 'y': 6.666, 'vx': 5, 'vy': 7.5, 'angle': 1, 'angular velocity': 2.5
//!
//! After the correction has been made, the units of the state are as follows:
//! 'x': (units), 'y': (units), 'vx': (units/second), 'vy': (units/second), 'angle': (radians), 'angular velocity': (radians/second)
//!
//! <!-- ## References -->
//!
//! ## Credits
//! Created by Oleg Klimov

use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::DqnConfig;
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::SplitMix64;
use bevy_gym::training::{run_discrete_workflow, DiscreteDqnExample, DqnAction};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use serde_json as _;

/// `FPS` used by this example.
const FPS: f32 = 50.0;
/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 1.0 / FPS;
/// `GRAVITY` used by this example.
const GRAVITY: f32 = -10.0;
/// `MAIN_ENGINE_ACCELERATION` used by this example.
const MAIN_ENGINE_ACCELERATION: f32 = 20.0;
/// `SIDE_ENGINE_ANGULAR_ACCELERATION` used by this example.
const SIDE_ENGINE_ANGULAR_ACCELERATION: f32 = 10.0;
/// `LANDER_MASS` used by this example.
const LANDER_MASS: f32 = 4.816_666_6;
/// `INITIAL_RANDOM_FORCE` used by this example.
const INITIAL_RANDOM_FORCE: f32 = 1_000.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;
/// `TERRAIN_POINTS` used by this example.
const TERRAIN_POINTS: usize = 11;
/// `HELIPAD_HALF_WIDTH` used by this example.
const HELIPAD_HALF_WIDTH: f32 = 0.2;

/// LunarLander-v3 action in Gymnasium order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LunarAction {
    /// Do not fire an engine.
    Coast,
    /// Fire the left orientation engine.
    LeftEngine,
    /// Fire the main engine.
    MainEngine,
    /// Fire the right orientation engine.
    RightEngine,
}

impl LunarAction {
    /// Return Gymnasium's per-frame engine cost.
    const fn fuel_cost(self) -> f64 {
        match self {
            Self::MainEngine => 0.30,
            Self::LeftEngine | Self::RightEngine => 0.03,
            Self::Coast => 0.0,
        }
    }
}

impl DqnAction for LunarAction {
    fn from_index(index: usize) -> Self {
        match index % 4 {
            0 => Self::Coast,
            1 => Self::LeftEngine,
            2 => Self::MainEngine,
            _ => Self::RightEngine,
        }
    }

    fn as_index(self) -> usize {
        match self {
            Self::Coast => 0,
            Self::LeftEngine => 1,
            Self::MainEngine => 2,
            Self::RightEngine => 3,
        }
    }
}

/// Gymnasium-compatible LunarLander-v3 task with deterministic 2D dynamics.
#[derive(Debug, Clone)]
struct LunarLander {
    /// Normalized position, velocity, angle, angular velocity, and leg contacts.
    state: [f32; 8],
    /// Previous potential used by Gymnasium's reward difference.
    previous_shaping: Option<f64>,
    /// Smoothed terrain heights in `Box2D` world units.
    terrain: [f32; TERRAIN_POINTS],
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Most recent action for exhaust rendering.
    last_action: LunarAction,
    /// Deterministic reset and engine-dispersion stream.
    rng: SplitMix64,
}

impl Default for LunarLander {
    fn default() -> Self {
        Self {
            state: [0.0; 8],
            previous_shaping: None,
            terrain: [10.0 / 3.0; TERRAIN_POINTS],
            elapsed_steps: 0,
            last_action: LunarAction::Coast,
            rng: SplitMix64::new(0),
        }
    }
}

impl LunarLander {
    /// Construct a stable controlled state for transition tests.
    #[cfg(test)]
    fn controlled() -> Self {
        let mut lander = Self::default();
        lander.state = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        lander.previous_shaping = Some(Self::shaping(&lander.state));
        lander
    }

    /// Generate Gymnasium's eleven-segment terrain and flat landing pad.
    fn reset_terrain(&mut self) {
        let mut raw = [0.0; TERRAIN_POINTS + 1];
        for height in &mut raw {
            *height = self.rng.f64_between(0.0, 20.0 / 3.0) as f32;
        }
        for height in raw.iter_mut().take(8).skip(3) {
            *height = 10.0 / 3.0;
        }
        for (index, smoothed) in self.terrain.iter_mut().enumerate() {
            let previous = if index == 0 {
                raw[TERRAIN_POINTS]
            } else {
                raw.get(index - 1)
                    .copied()
                    .expect("fixed example index is valid")
            };
            *smoothed = (previous
                + raw
                    .get(index)
                    .copied()
                    .expect("fixed example index is valid")
                + raw
                    .get(index + 1)
                    .copied()
                    .expect("fixed example index is valid"))
                / 3.0;
        }
    }

    /// Return Gymnasium's dense shaping potential.
    fn shaping(state: &[f32; 8]) -> f64 {
        let position = f64::from(state[0]).hypot(f64::from(state[1]));
        let velocity = f64::from(state[2]).hypot(f64::from(state[3]));
        10.0f64.mul_add(
            f64::from(state[7]),
            10.0f64.mul_add(
                f64::from(state[6]),
                100.0f64.mul_add(
                    -f64::from(state[4].abs()),
                    (-100.0f64).mul_add(position, -(100.0 * velocity)),
                ),
            ),
        )
    }

    /// Return the ground-contact height for one normalized horizontal position.
    fn contact_height(&self, normalized_x: f32) -> f32 {
        let terrain_coordinate = ((normalized_x + 1.0) * 5.0).clamp(0.0, 10.0);
        let left = (0..TERRAIN_POINTS)
            .position(|index| terrain_coordinate < (index + 1) as f32)
            .unwrap_or(TERRAIN_POINTS - 1);
        let right = (left + 1).min(TERRAIN_POINTS - 1);
        let fraction = terrain_coordinate - left as f32;
        let world_height = fraction.mul_add(
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
        );
        (world_height - 10.0 / 3.0) / (20.0 / 3.0)
    }

    /// Return whether the current terrain contact is a safe pad landing.
    fn safe_landing(&self) -> bool {
        self.state[0].abs() <= HELIPAD_HALF_WIDTH
            && self.state[2].abs() < 0.20
            && self.state[3] > -0.20
            && self.state[4].abs() < 0.25
            && self.state[5].abs() < 0.25
    }

    /// Return Gymnasium's published heuristic action for one observation.
    fn heuristic_action(observation: &[f32]) -> LunarAction {
        let x = observation.first().copied().unwrap_or(0.0);
        let y = observation.get(1).copied().unwrap_or(0.0);
        let velocity_x = observation.get(2).copied().unwrap_or(0.0);
        let velocity_y = observation.get(3).copied().unwrap_or(0.0);
        let angle = observation.get(4).copied().unwrap_or(0.0);
        let angular_velocity = observation.get(5).copied().unwrap_or(0.0);
        let leg_contact = observation.get(6).copied().unwrap_or(0.0) > 0.5
            || observation.get(7).copied().unwrap_or(0.0) > 0.5;
        let target_angle = (x * 0.5 + velocity_x).clamp(-0.4, 0.4);
        let target_hover = 0.55 * x.abs();
        let mut angle_control = (target_angle - angle).mul_add(0.5, -angular_velocity);
        let hover_control = if leg_contact {
            angle_control = 0.0;
            -velocity_y * 0.5
        } else {
            (target_hover - y).mul_add(0.5, -(velocity_y * 0.5))
        };
        if hover_control > angle_control.abs() && hover_control > 0.05 {
            LunarAction::MainEngine
        } else if angle_control < -0.05 {
            LunarAction::RightEngine
        } else if angle_control > 0.05 {
            LunarAction::LeftEngine
        } else {
            LunarAction::Coast
        }
    }

    /// Advance the normalized rigid-body state by one 50 Hz frame.
    fn integrate(&mut self, action: LunarAction) {
        let angle = self.state[4];
        let mut physical_velocity_x = self.state[2] * 5.0;
        let mut physical_velocity_y = self.state[3] * 7.5;
        let mut angular_velocity = self.state[5] * 2.5;

        physical_velocity_y = GRAVITY.mul_add(TIME_STEP, physical_velocity_y);
        if action == LunarAction::MainEngine {
            let dispersion = self.rng.f32_between(-0.04, 0.04);
            let thrust_angle = angle + dispersion;
            physical_velocity_x = (thrust_angle.sin() * MAIN_ENGINE_ACCELERATION)
                .mul_add(-TIME_STEP, physical_velocity_x);
            physical_velocity_y = (thrust_angle.cos() * MAIN_ENGINE_ACCELERATION)
                .mul_add(TIME_STEP, physical_velocity_y);
        }
        match action {
            LunarAction::LeftEngine => {
                angular_velocity =
                    SIDE_ENGINE_ANGULAR_ACCELERATION.mul_add(TIME_STEP, angular_velocity);
                physical_velocity_x += 0.08;
            }
            LunarAction::RightEngine => {
                angular_velocity =
                    SIDE_ENGINE_ANGULAR_ACCELERATION.mul_add(-TIME_STEP, angular_velocity);
                physical_velocity_x -= 0.08;
            }
            LunarAction::Coast | LunarAction::MainEngine => {}
        }

        physical_velocity_x *= 0.999;
        physical_velocity_y *= 0.999;
        angular_velocity *= 0.995;
        self.state[0] += physical_velocity_x * TIME_STEP / 10.0;
        self.state[1] += physical_velocity_y * TIME_STEP / (20.0 / 3.0);
        self.state[2] = physical_velocity_x / 5.0;
        self.state[3] = physical_velocity_y / 7.5;
        self.state[4] = wrap_angle(self.state[4] + angular_velocity * TIME_STEP);
        self.state[5] = angular_velocity / 2.5;
        self.state[6] = 0.0;
        self.state[7] = 0.0;
    }
}

impl Env for LunarLander {
    type Action = LunarAction;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.reset_terrain();
        let initial_force_x = self
            .rng
            .f32_between(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE);
        let initial_force_y = self
            .rng
            .f32_between(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE);
        let initial_velocity_x = initial_force_x / LANDER_MASS * TIME_STEP;
        let initial_velocity_y = initial_force_y / LANDER_MASS * TIME_STEP;
        self.state = [
            0.0,
            1.41,
            initial_velocity_x / 5.0,
            initial_velocity_y / 7.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        self.elapsed_steps = 0;
        self.last_action = LunarAction::Coast;
        self.integrate(LunarAction::Coast);
        self.previous_shaping = Some(Self::shaping(&self.state));
        Reset {
            observation: self.state.to_vec(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        self.integrate(action);
        self.elapsed_steps += 1;
        self.last_action = action;

        let shaping = Self::shaping(&self.state);
        let mut reward = self
            .previous_shaping
            .map_or(0.0, |previous| shaping - previous);
        self.previous_shaping = Some(shaping);
        reward -= action.fuel_cost();

        let contact_height = self.contact_height(self.state[0]);
        let contacted = self.state[1] <= contact_height;
        let outside = self.state[0].abs() >= 1.0;
        let status = if outside {
            reward = -100.0;
            EpisodeStatus::Terminated
        } else if contacted && self.safe_landing() {
            self.state[1] = contact_height;
            self.state[2] = 0.0;
            self.state[3] = 0.0;
            self.state[4] = 0.0;
            self.state[5] = 0.0;
            self.state[6] = 1.0;
            self.state[7] = 1.0;
            reward = 100.0;
            EpisodeStatus::Terminated
        } else if contacted {
            reward = -100.0;
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

impl DiscreteDqnExample for LunarLander {
    const ENV_NAME: &'static str = "lunar-lander";
    const GYMNASIUM_ID: &'static str = "LunarLander-v3";
    const OBSERVATION_DIM: usize = 8;
    const ACTION_COUNT: usize = 4;
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 100;
    const DEFAULT_LEARNING_RATE: f64 = 0.0003;
    const DEFAULT_REWARD_SCALE: f64 = 0.0;
    const GUIDED_TRANSITIONS: usize = 500_000;
    const GUIDED_ACTION_DIVISOR: usize = 2;
    const SOLVED_MEAN_REWARD: f64 = 200.0;
    const GIF_PATH: &'static str = "docs/images/lunar-lander.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 0.999,
            learning_rate,
            replay_capacity: 250_000,
            min_replay_size: 10_000,
            batch_size: 128,
            target_update_interval: 2_000,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 300_000,
            ..DqnConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation
            .iter()
            .enumerate()
            .map(|(index, value)| match index {
                0..=3 => value.clamp(-2.5, 2.5) / 2.5,
                4 => value / std::f32::consts::TAU,
                5 => value.clamp(-10.0, 10.0) / 10.0,
                _ => *value,
            })
            .collect()
    }

    fn guided_action(observation: &[f32]) -> Option<Self::Action> {
        Some(Self::heuristic_action(observation))
    }

    fn is_success(final_observation: &[f32], total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Terminated
            && total_reward >= 0.0
            && final_observation.get(6).copied().unwrap_or(0.0) > 0.5
            && final_observation.get(7).copied().unwrap_or(0.0) > 0.5
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
    run_discrete_workflow::<LunarLander>()
}

/// Wrap one angle into `[-pi, pi]`.
fn wrap_angle(angle: f32) -> f32 {
    angle.sin().atan2(angle.cos())
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        DiscreteDqnExample, DqnAction, Env, EpisodeStatus, Error, LunarAction, LunarLander, Path,
        FPS, TERRAIN_POINTS,
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
            app.insert_resource(ClearColor(Color::BLACK));
        }
    }

    /// Policy playback state shown in the `LunarLander` scene.
    #[derive(Resource)]
    struct VisualLunarLander {
        /// Greedy checkpoint policy.
        policy: DqnPolicy,
        /// Live environment and terrain.
        env: LunarLander,
        /// Current observation.
        observation: Vec<f32>,
    }

    /// Fifty-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Dynamic purple lander body.
    #[derive(Component)]
    struct LanderBody;

    /// Dynamic purple landing leg.
    #[derive(Component)]
    struct LanderLeg(f32);

    /// Dynamic red exhaust particle.
    #[derive(Component)]
    struct ExhaustParticle(usize);

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = LunarLander::dqn_config(LunarLander::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(checkpoint, 8, 4, &config.hidden_sizes)?;
        let mut env = LunarLander::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualLunarLander {
            policy,
            env,
            observation,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            1.0 / FPS,
            TimerMode::Repeating,
        )))
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym LunarLander-v3".into(),
                        resolution: WindowResolution::new(600, 400),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_scene)
        .add_systems(Update, draw_scene_outlines);
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

    /// Spawn the terrain fill, lander body, legs, and exhaust particles.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        visual: Res<'_, VisualLunarLander>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        commands.spawn((Camera2d, Name::new("LunarLander Camera")));
        let white = materials.add(Color::WHITE);
        for index in 0..TERRAIN_POINTS - 1 {
            let left = terrain_point(&visual.env, index);
            let right = terrain_point(&visual.env, index + 1);
            let bottom_left = Vec2::new(left.x, -200.0);
            let bottom_right = Vec2::new(right.x, -200.0);
            commands.spawn((
                Mesh2d(meshes.add(Triangle2d::new(left, right, bottom_right))),
                MeshMaterial2d(white.clone()),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));
            commands.spawn((
                Mesh2d(meshes.add(Triangle2d::new(left, bottom_right, bottom_left))),
                MeshMaterial2d(white.clone()),
                Transform::from_xyz(0.0, 0.0, 0.0),
            ));
        }
        let yellow = materials.add(Color::srgb_u8(204, 204, 0));
        for x in [-60.0, 60.0] {
            commands.spawn((
                Mesh2d(meshes.add(Triangle2d::new(
                    Vec2::new(x, -50.0),
                    Vec2::new(x, -60.0),
                    Vec2::new(x + 25.0, -55.0),
                ))),
                MeshMaterial2d(yellow.clone()),
                Transform::from_xyz(0.0, 0.0, 1.0),
            ));
        }
        let body_vertices = [
            Vec2::new(-14.0, 17.0),
            Vec2::new(-17.0, 0.0),
            Vec2::new(-17.0, -10.0),
            Vec2::new(17.0, -10.0),
            Vec2::new(17.0, 0.0),
            Vec2::new(14.0, 17.0),
        ];
        let Ok(body_shape) = ConvexPolygon::new(body_vertices) else {
            return;
        };
        commands.spawn((
            Mesh2d(meshes.add(body_shape)),
            MeshMaterial2d(materials.add(Color::srgb_u8(128, 102, 230))),
            Transform::from_xyz(0.0, 0.0, 2.0),
            LanderBody,
        ));
        for side in [-1.0, 1.0] {
            commands.spawn((
                Mesh2d(meshes.add(Rectangle::new(4.0, 16.0))),
                MeshMaterial2d(materials.add(Color::srgb_u8(128, 102, 230))),
                Transform::from_xyz(0.0, 0.0, 2.0),
                LanderLeg(side),
            ));
        }
        for index in 0..4 {
            commands.spawn((
                Mesh2d(meshes.add(Circle::new(2.0))),
                MeshMaterial2d(materials.add(Color::srgb_u8(255, 77, 77))),
                Transform::from_xyz(0.0, 0.0, 1.0),
                Visibility::Hidden,
                ExhaustParticle(index),
            ));
        }
    }

    /// Draw terrain edges, pad flags, and dark lander outlines.
    fn draw_scene_outlines(visual: Res<'_, VisualLunarLander>, mut gizmos: Gizmos<'_, '_>) {
        for index in 0..TERRAIN_POINTS - 1 {
            gizmos.line_2d(
                terrain_point(&visual.env, index),
                terrain_point(&visual.env, index + 1),
                Color::WHITE,
            );
        }
        for x in [-60.0, 60.0] {
            gizmos.line_2d(Vec2::new(x, -100.0), Vec2::new(x, -50.0), Color::WHITE);
            let yellow = Color::srgb_u8(204, 204, 0);
            gizmos.line_2d(Vec2::new(x, -50.0), Vec2::new(x, -60.0), yellow);
            gizmos.line_2d(Vec2::new(x, -60.0), Vec2::new(x + 25.0, -55.0), yellow);
            gizmos.line_2d(Vec2::new(x + 25.0, -55.0), Vec2::new(x, -50.0), yellow);
        }
        let center = lander_screen_position(&visual.env);
        let rotation = visual.env.state[4];
        let outline = Color::srgb_u8(77, 77, 128);
        let points = [
            Vec2::new(-14.0, 17.0),
            Vec2::new(-17.0, 0.0),
            Vec2::new(-17.0, -10.0),
            Vec2::new(17.0, -10.0),
            Vec2::new(17.0, 0.0),
            Vec2::new(14.0, 17.0),
        ];
        for index in 0..points.len() {
            let left = center
                + points
                    .get(index)
                    .copied()
                    .expect("fixed example index is valid")
                    .rotate(Vec2::from_angle(rotation));
            let right = center
                + points
                    .get((index + 1) % points.len())
                    .copied()
                    .expect("fixed example index is valid")
                    .rotate(Vec2::from_angle(rotation));
            gizmos.line_2d(left, right, outline);
        }
    }

    /// Advance policy playback at Gymnasium's 50 frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualLunarLander>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Advance a fixed number of greedy environment transitions.
    fn advance_visual(visual: &mut VisualLunarLander, steps: usize) {
        for _ in 0..steps {
            let encoded = LunarLander::encode_observation(&visual.observation);
            let Ok(action_index) = visual.policy.greedy_action(&encoded) else {
                return;
            };
            let transition = visual.env.step(LunarAction::from_index(action_index));
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(Some(12_345)).observation;
                break;
            }
        }
    }

    /// Synchronize lander, legs, and exhaust.
    fn sync_scene(
        visual: Res<'_, VisualLunarLander>,
        mut body: Single<'_, '_, &mut Transform, With<LanderBody>>,
        mut legs: Query<'_, '_, (&LanderLeg, &mut Transform), Without<LanderBody>>,
        mut exhaust: Query<
            '_,
            '_,
            (&ExhaustParticle, &mut Transform, &mut Visibility),
            (Without<LanderBody>, Without<LanderLeg>),
        >,
    ) {
        apply_visual_state(&visual.env, &mut body, &mut legs, &mut exhaust);
    }

    /// Apply one lander state to Gymnasium's 600-by-400 viewport.
    fn apply_visual_state(
        env: &LunarLander,
        body: &mut Transform,
        legs: &mut Query<'_, '_, (&LanderLeg, &mut Transform), Without<LanderBody>>,
        exhaust: &mut Query<
            '_,
            '_,
            (&ExhaustParticle, &mut Transform, &mut Visibility),
            (Without<LanderBody>, Without<LanderLeg>),
        >,
    ) {
        let center = lander_screen_position(env);
        let rotation = env.state[4];
        body.translation = center.extend(2.0);
        body.rotation = Quat::from_rotation_z(rotation);
        for (leg, mut transform) in legs.iter_mut() {
            let local = Vec2::new(leg.0 * 20.0, -13.0);
            transform.translation = (center + local.rotate(Vec2::from_angle(rotation))).extend(2.0);
            transform.rotation = Quat::from_rotation_z(leg.0.mul_add(-0.30, rotation));
        }
        for (particle, mut transform, mut visibility) in exhaust.iter_mut() {
            let local = match env.last_action {
                LunarAction::MainEngine => Vec2::new(
                    (particle.0 as f32 - 1.5) * 4.0,
                    (particle.0 as f32).mul_add(-5.0, -16.0),
                ),
                LunarAction::LeftEngine => Vec2::new(
                    (particle.0 as f32).mul_add(4.0, 22.0),
                    (particle.0 as f32).mul_add(-3.0, 4.0),
                ),
                LunarAction::RightEngine => Vec2::new(
                    (particle.0 as f32).mul_add(-4.0, -22.0),
                    (particle.0 as f32).mul_add(-3.0, 4.0),
                ),
                LunarAction::Coast => Vec2::ZERO,
            };
            if env.last_action == LunarAction::Coast {
                *visibility = Visibility::Hidden;
            } else {
                *visibility = Visibility::Visible;
                transform.translation =
                    (center + local.rotate(Vec2::from_angle(rotation))).extend(1.0);
            }
        }
    }

    /// Return one terrain vertex in viewport coordinates.
    fn terrain_point(env: &LunarLander, index: usize) -> Vec2 {
        Vec2::new(
            (index as f32).mul_add(60.0, -300.0),
            env.terrain
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(30.0, -200.0),
        )
    }

    /// Return the lander body's center in viewport coordinates.
    fn lander_screen_position(env: &LunarLander) -> Vec2 {
        Vec2::new(env.state[0] * 300.0, env.state[1].mul_add(200.0, -82.0))
    }

    /// Capture twelve policy transitions per GIF frame.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualLunarLander>,
        mut body: Single<'_, '_, &mut Transform, With<LanderBody>>,
        mut legs: Query<'_, '_, (&LanderLeg, &mut Transform), Without<LanderBody>>,
        mut exhaust: Query<
            '_,
            '_,
            (&ExhaustParticle, &mut Transform, &mut Visibility),
            (Without<LanderBody>, Without<LanderLeg>),
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
            advance_visual(&mut visual, 12);
        }
        apply_visual_state(&visual.env, &mut body, &mut legs, &mut exhaust);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "lunar-lander", 600, 400, |frames| {
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
    fn reset_randomizes_terrain_and_initial_force_reproducibly() {
        let mut first = LunarLander::default();
        let first_observation = first.reset(Some(42)).observation;
        let first_terrain = first.terrain;

        let mut repeated = LunarLander::default();
        let repeated_observation = repeated.reset(Some(42)).observation;
        assert_eq!(repeated.terrain, first_terrain);
        assert_eq!(repeated_observation, first_observation);

        let mut different = LunarLander::default();
        let different_observation = different.reset(Some(43)).observation;
        assert_ne!(different.terrain, first_terrain);
        assert_ne!(different_observation[2..4], first_observation[2..4]);

        let maximum_horizontal_velocity = INITIAL_RANDOM_FORCE / LANDER_MASS * TIME_STEP / 5.0;
        let maximum_vertical_velocity = INITIAL_RANDOM_FORCE / LANDER_MASS * TIME_STEP / 7.5;
        assert!(first_observation[2].abs() <= maximum_horizontal_velocity);
        assert!(
            first_observation[3].abs() <= maximum_vertical_velocity + GRAVITY * TIME_STEP / 7.5
        );
    }

    #[test]
    fn reset_returns_the_eight_gymnasium_observations() {
        let mut lander = LunarLander::default();

        let reset = lander.reset(Some(42));

        assert_eq!(reset.observation.len(), 8);
        assert_eq!(reset.observation[6], 0.0);
        assert_eq!(reset.observation[7], 0.0);
    }

    #[test]
    fn engine_costs_match_gymnasium() {
        assert_eq!(LunarAction::Coast.fuel_cost(), 0.0);
        assert_eq!(LunarAction::LeftEngine.fuel_cost(), 0.03);
        assert_eq!(LunarAction::MainEngine.fuel_cost(), 0.30);
        assert_eq!(LunarAction::RightEngine.fuel_cost(), 0.03);
    }

    #[test]
    fn safe_pad_contact_terminates_with_the_landing_reward() {
        let mut lander = LunarLander::controlled();
        lander.state = [0.0, -0.49, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0];

        let step = lander.step(LunarAction::Coast);

        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(step.reward, 100.0);
    }

    #[test]
    fn gymnasium_heuristic_clears_the_reward_gate() {
        let mut total = 0.0;
        let episodes = 100_u32;
        for seed in 0..episodes {
            let mut lander = LunarLander::default();
            let mut observation = lander.reset(Some(u64::from(seed))).observation;
            loop {
                let step = lander.step(LunarLander::heuristic_action(&observation));
                total += step.reward;
                observation = step.observation;
                if step.status != EpisodeStatus::Continuing {
                    break;
                }
            }
        }

        let mean = total / f64::from(episodes);
        assert!(mean >= 200.0, "heuristic mean reward was {mean:.3}");
    }
}
