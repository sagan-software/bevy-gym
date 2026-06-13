#![recursion_limit = "256"]

//! ## Description
//!
//! This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
//! ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
//! A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
//! The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
//!  in the left and right direction on the cart.
//!
//! ## Action Space
//!
//! The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
//!  of the fixed force the cart is pushed with.
//!
//! - 0: Push cart to the left
//! - 1: Push cart to the right
//!
//! **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
//!  the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
//!
//! ## Observation Space
//!
//! The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
//!
//! | Num | Observation           | Min                 | Max               |
//! |-----|-----------------------|---------------------|-------------------|
//! | 0   | Cart Position         | -4.8                | 4.8               |
//! | 1   | Cart Velocity         | -Inf                | Inf               |
//! | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
//! | 3   | Pole Angular Velocity | -Inf                | Inf               |
//!
//! **Note:** While the ranges above denote the possible values for observation space of each element,
//!     it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
//! -  The cart x-position (index 0) can take values between `(-4.8, 4.8)`, but the episode terminates
//!    if the cart leaves the `(-2.4, 2.4)` range.
//! -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
//!    if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
//!
//! ## Rewards
//! Since the goal is to keep the pole upright for as long as possible, by default, a reward of `+1` is given for every step taken, including the termination step. The default reward threshold is 500 for v1 and 200 for v0 due to the time limit on the environment.
//!
//! If `sutton_barto_reward=True`, then a reward of `0` is awarded for every non-terminating step and `-1` for the terminating step. As a result, the reward threshold is 0 for v0 and v1.
//!
//! ## Starting State
//! All observations are assigned a uniformly random value in `(-0.05, 0.05)`
//!
//! ## Episode End
//! The episode ends if any one of the following occurs:
//!
//! 1. Termination: Pole Angle is greater than ±12°
//! 2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
//! 3. Truncation: Episode length is greater than 500 (200 for v0)
//!
//! ## Arguments
//!
//! Cartpole only has `render_mode` as a keyword for `gymnasium.make`.
//! On reset, the `options` parameter allows the user to change the bounds used to determine the new random state.
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("CartPole-v1", render_mode="rgb_array")
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
//! >>> env.reset(seed=123, options={"low": -0.1, "high": 0.1})  # default low=-0.05, high=0.05
//! (array([ 0.03647037, -0.0892358 , -0.05592803, -0.06312564], dtype=float32), {})
//!
//! ```
//!
//! | Parameter               | Type       | Default                 | Description                                                                                   |
//! |-------------------------|------------|-------------------------|-----------------------------------------------------------------------------------------------|
//! | `sutton_barto_reward`   | **bool**   | `False`                 | If `True` the reward function matches the original sutton barto implementation                |
//!
//! ## Vectorized environment
//!
//! To increase steps per seconds, users can use a custom vector environment or with an environment vectorizor.
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="vector_entry_point")
//! >>> envs
//! CartPoleVectorEnv(CartPole-v1, num_envs=3)
//! >>> envs = gym.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync")
//! >>> envs
//! SyncVectorEnv(CartPole-v1, num_envs=3)
//!
//! ```
//!
//! ## Version History
//! * v1: `max_time_steps` raised to 500.
//!     - In Gymnasium `1.0.0a2` the `sutton_barto_reward` argument was added (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/790))
//! * v0: Initial versions release.

//! CartPole-v1 -- train or evaluate a DQN agent, with optional live rendering.
//!
//! Demonstrates the full ecosystem integration:
//!   rl-traits (env contract) → ember-rl (DQN + TrainingSession) → bevy-gym (parallel ECS loop)
//!
//! Four CartPole environments step in parallel each ECS tick. A single DQN
//! agent collects experience from all four, fills its replay buffer, and trains.
//! Checkpoints are saved under `runs/bevy_cartpole/v1/<timestamp>/`; `best.mpk`
//! is updated whenever the rolling training mean reward improves.
//!
//! # Train (headless, maximum speed)
//!
//!   cargo run --example cartpole --release
//!
//! # Train with live 2D rendering
//!
//!   cargo run --example cartpole --features render,x11 --release -- --render
//!
//! # Train with live rendering at half speed (easier to observe)
//!
//!   cargo run --example cartpole --features render,x11 --release -- --render --speed 0.5
//!
//! # Evaluate a saved run (headless)
//!
//!   cargo run --example cartpole --release -- --eval runs/bevy_cartpole/v1
//!
//! # Evaluate with live rendering
//!
//!   cargo run --example cartpole --features render,x11 --release -- --eval runs/bevy_cartpole/v1 --render
//!
//! Evaluation loads `best.mpk` from the most recent run under the given path
//! and runs 20 greedy episodes across all parallel environments.
//!
//! With `--render`, environments are visualised as coloured rectangles stacked
//! vertically. The pole shifts green → red as its angle approaches the 12°
//! failure limit; the cart shifts blue → orange as it nears the ±2.4 boundary.
//! Pass `--speed <factor>` to scale simulation speed (e.g. `0.5` for half speed,
//! `2.0` for double). Has no effect in headless mode, which always runs at
//! maximum CPU throughput.

use std::env;

use bevy::prelude::*;
use burn::backend::Autodiff;
#[cfg(all(
    feature = "gpu-metal",
    any(feature = "gpu-vulkan", feature = "gpu-webgpu")
))]
compile_error!("Choose only one CartPole GPU backend feature.");
#[cfg(all(feature = "gpu-vulkan", feature = "gpu-webgpu"))]
compile_error!("Choose only one CartPole GPU backend feature.");
#[cfg(feature = "gpu-metal")]
type BackendImpl = burn::backend::Metal<f32, i64>;
#[cfg(not(feature = "gpu"))]
type BackendImpl = burn::backend::NdArray;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-metal")))]
type BackendImpl = burn::backend::Vulkan<f32, i64>;
#[cfg(all(
    feature = "gpu-webgpu",
    not(any(feature = "gpu-metal", feature = "gpu-vulkan"))
))]
type BackendImpl = burn::backend::WebGpu<f32, i64>;
#[cfg(all(
    feature = "gpu",
    not(any(feature = "gpu-metal", feature = "gpu-vulkan", feature = "gpu-webgpu"))
))]
type BackendImpl = burn::backend::Wgpu<f32, i64>;
use ember_rl::{
    algorithms::dqn::{DqnAgent, DqnConfig, DqnPolicy},
    encoding::{UsizeActionMapper, VecEncoder},
    envs::cartpole::CartPoleEnv,
    training::{TrainingRun, TrainingSession},
    traits::{ActMode, LearningAgent},
};
use rand::Rng;
use rl_traits::{Environment, Policy, StepResult};

use bevy_gym::{
    ActionRequestEvent, BevyGymPlugin, EpisodeEndEvent, ExperienceEvent, GymSet,
    components::{CurrentObservation, PendingAction},
};

#[cfg(feature = "render")]
use bevy_gym::render::{GymRender, GymRenderPlugin, SpawnCtx};

type B = Autodiff<BackendImpl>;
type InferB = BackendImpl;
type Session =
    TrainingSession<CartPoleEnv, DqnAgent<CartPoleEnv, VecEncoder, UsizeActionMapper, B>>;

#[cfg(feature = "gpu-metal")]
const BACKEND_NAME: &str = "burn-wgpu-metal";
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-metal")))]
const BACKEND_NAME: &str = "burn-wgpu-vulkan";
#[cfg(all(
    feature = "gpu-webgpu",
    not(any(feature = "gpu-metal", feature = "gpu-vulkan"))
))]
const BACKEND_NAME: &str = "burn-wgpu-webgpu";
#[cfg(all(
    feature = "gpu",
    not(any(feature = "gpu-metal", feature = "gpu-vulkan", feature = "gpu-webgpu"))
))]
const BACKEND_NAME: &str = "burn-wgpu-auto";
#[cfg(all(not(feature = "gpu"), feature = "cpu-accelerate"))]
const BACKEND_NAME: &str = "burn-ndarray-accelerate";
#[cfg(all(not(feature = "gpu"), not(feature = "cpu-accelerate")))]
const BACKEND_NAME: &str = "burn-ndarray";

const NUM_ENVS: usize = 4;
const MAX_STEPS: usize = 200_000;
const CHECKPOINT_FREQ: usize = 25_000;
const LOG_INTERVAL: usize = 5_000;
const EVAL_EPISODES: usize = 20;

// ── CartPole newtype ──────────────────────────────────────────────────────────
//
// The orphan rule prevents `impl GymRender for CartPoleEnv` in an example
// binary -- both the trait (bevy-gym) and the type (ember-rl) are foreign.
// Wrapping it in a local newtype solves this at zero runtime cost.
// CartPoleViz is used as the ECS environment type in all modes for uniformity.

struct CartPoleViz(CartPoleEnv);

impl Environment for CartPoleViz {
    type Observation = Vec<f32>;
    type Action = usize;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> (Vec<f32>, ()) {
        self.0.reset(seed)
    }
    fn step(&mut self, action: usize) -> StepResult<Vec<f32>, ()> {
        self.0.step(action)
    }
    fn sample_action(&self, rng: &mut impl Rng) -> usize {
        self.0.sample_action(rng)
    }
}

// ── Visual constants & GymRender impl (render feature only) ──────────────────

#[cfg(feature = "render")]
const SCALE: f32 = 100.0;
#[cfg(feature = "render")]
const CART_W: f32 = 60.0;
#[cfg(feature = "render")]
const CART_H: f32 = 30.0;
#[cfg(feature = "render")]
const POLE_W: f32 = 8.0;
#[cfg(feature = "render")]
const POLE_H: f32 = 100.0;
#[cfg(feature = "render")]
const ENV_SPACING: f32 = 160.0;

#[cfg(feature = "render")]
#[derive(Component)]
struct CartPoleVisuals {
    cart: Entity,
    pole: Entity,
    cart_mat: Handle<ColorMaterial>,
    pole_mat: Handle<ColorMaterial>,
    y_offset: f32,
}

#[cfg(feature = "render")]
impl GymRender for CartPoleViz {
    type Visuals = CartPoleVisuals;

    fn setup_visuals(entity: Entity, env_id: usize, ctx: &mut SpawnCtx<'_, '_, '_>) {
        let y_offset = (NUM_ENVS as f32 - 1.0) / 2.0 * ENV_SPACING - env_id as f32 * ENV_SPACING;

        // Track -- static grey bar.
        ctx.commands.spawn((
            Mesh2d(ctx.meshes.add(Rectangle::new(500.0, 4.0))),
            MeshMaterial2d(
                ctx.materials
                    .add(ColorMaterial::from_color(Color::srgb(0.45, 0.45, 0.45))),
            ),
            Transform::from_xyz(0.0, y_offset - CART_H / 2.0, -1.0),
        ));

        // Failure-zone boundary markers at x = ±2.4 units = ±240 px.
        let boundary_mat = ctx
            .materials
            .add(ColorMaterial::from_color(Color::srgba(0.8, 0.2, 0.2, 0.5)));
        for sign in [-1.0f32, 1.0] {
            ctx.commands.spawn((
                Mesh2d(ctx.meshes.add(Rectangle::new(3.0, ENV_SPACING * 0.7))),
                MeshMaterial2d(boundary_mat.clone()),
                Transform::from_xyz(sign * 2.4 * SCALE, y_offset, -0.5),
            ));
        }

        // Cart -- starts blue, shifts toward orange near the boundary.
        let cart_mat = ctx
            .materials
            .add(ColorMaterial::from_color(Color::srgb(0.25, 0.45, 0.85)));
        let cart = ctx
            .commands
            .spawn((
                Mesh2d(ctx.meshes.add(Rectangle::new(CART_W, CART_H))),
                MeshMaterial2d(cart_mat.clone()),
                Transform::from_xyz(0.0, y_offset, 0.0),
            ))
            .id();

        // Pole -- starts green, shifts to red as angle approaches failure (12°).
        let pole_mat = ctx
            .materials
            .add(ColorMaterial::from_color(Color::srgb(0.2, 0.8, 0.2)));
        let pole = ctx
            .commands
            .spawn((
                Mesh2d(ctx.meshes.add(Rectangle::new(POLE_W, POLE_H))),
                MeshMaterial2d(pole_mat.clone()),
                Transform::from_xyz(0.0, y_offset + CART_H / 2.0 + POLE_H / 2.0, 1.0),
            ))
            .id();

        ctx.commands.entity(entity).insert(CartPoleVisuals {
            cart,
            pole,
            cart_mat,
            pole_mat,
            y_offset,
        });
    }

    fn sync_visuals(
        obs: &Vec<f32>,
        visuals: &CartPoleVisuals,
        transforms: &mut Query<&mut Transform>,
    ) {
        let x = obs[0];
        let theta = obs[2];

        let cart_x = x * SCALE;
        let cart_y = visuals.y_offset;

        if let Ok(mut t) = transforms.get_mut(visuals.cart) {
            t.translation = Vec3::new(cart_x, cart_y, 0.0);
        }

        let pivot_y = cart_y + CART_H / 2.0;
        let half_pole = POLE_H / 2.0;

        if let Ok(mut t) = transforms.get_mut(visuals.pole) {
            t.translation = Vec3::new(
                cart_x + half_pole * theta.sin(),
                pivot_y + half_pole * theta.cos(),
                1.0,
            );
            // Bevy z-rotation is CCW-positive; CartPole theta is CW-positive.
            t.rotation = Quat::from_rotation_z(-theta);
        }
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

fn cartpole_config() -> DqnConfig {
    DqnConfig {
        gamma: 0.99,
        learning_rate: 3e-4,
        batch_size: 64,
        buffer_capacity: 100_000,
        min_replay_size: 1_000,
        target_update_freq: 500,
        hidden_sizes: vec![64, 64],
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay_steps: 10_000,
    }
}

// ── Eval resource ─────────────────────────────────────────────────────────────

struct DqnPolicyResource {
    policy: DqnPolicy<CartPoleEnv, VecEncoder, UsizeActionMapper, InferB>,
}

// ── Train systems ─────────────────────────────────────────────────────────────

fn policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(
        &CurrentObservation<CartPoleViz>,
        &mut PendingAction<CartPoleViz>,
    )>,
    mut session: NonSendMut<Session>,
) {
    for req in requests.read() {
        if let Ok((obs, mut pending)) = query.get_mut(req.entity) {
            pending.action = Some(session.act(&obs.observation, ActMode::Explore));
        }
    }
}

fn learn_system(
    mut events: MessageReader<ExperienceEvent<Vec<f32>, usize>>,
    mut session: NonSendMut<Session>,
) {
    for event in events.read() {
        session.observe(event.experience.clone());
    }
}

fn log_and_stop_system(
    mut events: MessageReader<EpisodeEndEvent>,
    mut session: NonSendMut<Session>,
    mut last_logged: Local<usize>,
    mut last_eval_step: Local<usize>,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }

    for event in events.read() {
        session.on_episode(
            event.total_reward,
            event.episode_steps,
            event.status.clone(),
            event.extras.clone(),
        );
    }

    let steps = session.total_steps();
    let bucket = steps / LOG_INTERVAL;
    if bucket > *last_logged {
        *last_logged = bucket;
        let summary = session.stats_summary();
        let mean = summary.get("episode_reward").copied().unwrap_or(f64::NAN);
        let len = summary.get("episode_length").copied().unwrap_or(f64::NAN);
        println!(
            "step {:>6}  reward mean: {:>6.1}  len: {:>5.1}  steps/sec: {:.0}  ε {:.3}",
            steps,
            mean,
            len,
            session.steps_per_sec(),
            session.agent().epsilon(),
        );
    }

    if steps >= CHECKPOINT_FREQ && steps.saturating_sub(*last_eval_step) >= CHECKPOINT_FREQ {
        *last_eval_step = steps;
        let eval_mean = evaluate_agent(session.agent_mut(), EVAL_EPISODES, steps as u64);
        println!(
            "           greedy eval: {:>6.1} over {EVAL_EPISODES} episodes",
            eval_mean
        );
        session.maybe_save_best(eval_mean);
    }

    if !*done && session.total_steps() >= MAX_STEPS {
        *done = true;
        let summary = session.stats_summary();
        let eval_mean = evaluate_agent(session.agent_mut(), EVAL_EPISODES, steps as u64);
        session.maybe_save_best(eval_mean);
        println!("\nTraining complete. Checkpoints saved to run directory.");
        println!(
            "Final stats -- train mean reward: {:.1}  greedy eval: {:.1}  steps/sec: {:.0}",
            summary.get("episode_reward").copied().unwrap_or(f64::NAN),
            eval_mean,
            session.steps_per_sec(),
        );
        std::process::exit(0);
    }
}

fn evaluate_agent(agent: &mut impl LearningAgent<CartPoleEnv>, episodes: usize, seed: u64) -> f64 {
    let mut total_reward = 0.0;

    for episode in 0..episodes {
        let mut env = CartPoleEnv::new();
        let (mut obs, _) = env.reset(Some(seed.wrapping_add(episode as u64)));
        let mut episode_reward = 0.0;

        loop {
            let action = agent.act(&obs, ActMode::Exploit);
            let result = env.step(action);
            episode_reward += result.reward;

            if result.is_done() {
                break;
            }

            obs = result.observation;
        }

        total_reward += episode_reward;
    }

    total_reward / episodes as f64
}

// ── Eval systems ──────────────────────────────────────────────────────────────

fn eval_policy_system(
    mut requests: MessageReader<ActionRequestEvent>,
    mut query: Query<(
        &CurrentObservation<CartPoleViz>,
        &mut PendingAction<CartPoleViz>,
    )>,
    policy: NonSend<DqnPolicyResource>,
) {
    for req in requests.read() {
        if let Ok((obs, mut pending)) = query.get_mut(req.entity) {
            pending.action = Some(policy.policy.act(&obs.observation));
        }
    }
}

fn eval_log_and_stop_system(
    mut events: MessageReader<EpisodeEndEvent>,
    mut total_episodes: Local<usize>,
    mut total_reward: Local<f64>,
    mut total_length: Local<usize>,
) {
    for event in events.read() {
        *total_episodes += 1;
        *total_reward += event.total_reward;
        *total_length += event.episode_steps;
        println!(
            "ep {:>3}  env {}  steps {:>3}  reward {:>6.1}",
            *total_episodes, event.env_id, event.episode_steps, event.total_reward,
        );
    }

    if *total_episodes >= 20 {
        println!(
            "\nmean reward: {:.1}  mean length: {:.1}",
            *total_reward / *total_episodes as f64,
            *total_length as f64 / *total_episodes as f64,
        );
        std::process::exit(0);
    }
}

// ── Render systems (render feature only) ─────────────────────────────────────

#[cfg(feature = "render")]
fn update_danger_colors(
    query: Query<(&CurrentObservation<CartPoleViz>, &CartPoleVisuals)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (obs, visuals) in query.iter() {
        let angle_danger = (obs.observation[2].abs() / 0.2094).clamp(0.0, 1.0);
        let pos_danger = (obs.observation[0].abs() / 2.4).clamp(0.0, 1.0);
        let danger = angle_danger.max(pos_danger);

        if let Some(mat) = materials.get_mut(&visuals.pole_mat) {
            mat.color = Color::srgb(danger, 1.0 - danger * 0.5, 0.0);
        }
        if let Some(mat) = materials.get_mut(&visuals.cart_mat) {
            mat.color = Color::srgb(
                0.25 + pos_danger * 0.6,
                0.45 - pos_danger * 0.2,
                0.85 - pos_danger * 0.6,
            );
        }
    }
}

#[cfg(feature = "render")]
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

// ── App helpers ───────────────────────────────────────────────────────────────

/// Adds rendering plugins and systems. Only callable when the `render` feature
/// is enabled -- call sites are gated with `#[cfg(feature = "render")]`.
#[cfg(feature = "render")]
fn add_render_plugins(app: &mut App, speed: f32) {
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "CartPole-v1 -- bevy-gym".into(),
                    resolution: (800u32, 780u32).into(),
                    ..default()
                }),
                ..default()
            })
            .set(bevy::log::LogPlugin {
                filter: "warn,burn_core::record::file=error".to_string(),
                ..default()
            }),
    )
    .add_plugins(GymRenderPlugin::<CartPoleViz>::new())
    .add_systems(Startup, setup_camera)
    .add_systems(Update, update_danger_colors);

    if speed != 1.0 {
        app.add_systems(Startup, move |mut time: ResMut<Time<Virtual>>| {
            time.set_relative_speed(speed);
        });
    }
}

fn add_headless_plugins(app: &mut App) {
    app.add_plugins(
        MinimalPlugins.set(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        )),
    )
    .add_plugins(bevy::log::LogPlugin {
        filter: "warn,burn_core::record::file=error".to_string(),
        ..default()
    })
    // Virtual time advances at f32::MAX × real time, so FixedUpdate runs as
    // many steps as possible each app-loop iteration instead of at real-time 64 Hz.
    .add_systems(Startup, |mut time: ResMut<Time<Virtual>>| {
        time.set_relative_speed(1_000_000.0);
    });
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = env::args().collect();
    let render = args.contains(&"--render".to_string());

    #[cfg(not(feature = "render"))]
    if render {
        eprintln!("error: --render requires compiling with --features render");
        std::process::exit(1);
    }

    #[cfg(all(feature = "render", not(feature = "winit")))]
    if render {
        eprintln!("error: --render requires a winit backend, e.g. --features render,x11");
        std::process::exit(1);
    }

    let speed: f32 = args
        .iter()
        .position(|a| a == "--speed")
        .and_then(|pos| args.get(pos + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);

    if let Some(pos) = args.iter().position(|a| a == "--eval") {
        let run_path = args
            .get(pos + 1)
            .expect("--eval requires a path argument, e.g. --eval runs/bevy_cartpole/v1");
        run_eval(run_path, render, speed);
    } else {
        run_train(render, speed);
    }
}

#[cfg_attr(not(feature = "render"), allow(unused_variables))]
fn run_train(render: bool, speed: f32) {
    println!(
        "Training DQN on CartPole-v1 with {BACKEND_NAME} ({NUM_ENVS} parallel envs, {MAX_STEPS} steps)...\n"
    );

    let device = Default::default();
    let agent = DqnAgent::<CartPoleEnv, _, _, B>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        cartpole_config(),
        device,
        2,
    );

    let run = TrainingRun::create("bevy_cartpole", "v1").expect("failed to create training run");
    println!("Run directory: {}", run.dir().display());

    let session: Session = TrainingSession::new(agent)
        .with_run(run)
        .with_checkpoint_freq(CHECKPOINT_FREQ)
        .with_keep_checkpoints(3);

    let gym = BevyGymPlugin::new(|_| CartPoleViz(CartPoleEnv::new()), NUM_ENVS);

    let mut app = App::new();
    if render {
        app.add_plugins(gym);
        #[cfg(feature = "render")]
        add_render_plugins(&mut app, speed);
    } else {
        add_headless_plugins(&mut app);
        app.add_plugins(gym.headless());
    }
    app.insert_non_send_resource(session)
        .add_systems(
            FixedUpdate,
            (
                policy_system.after(GymSet::ManualReset),
                learn_system.after(GymSet::ManualReset),
                log_and_stop_system.after(GymSet::ManualReset),
            ),
        )
        .run();
}

#[cfg_attr(not(feature = "render"), allow(unused_variables))]
fn run_eval(run_path: &str, render: bool, speed: f32) {
    println!("Loading checkpoint from {run_path}...\n");

    let run = TrainingRun::resume(run_path)
        .unwrap_or_else(|e| panic!("failed to resume run at {run_path}: {e}"));

    println!(
        "Run: {} {} (step {})",
        run.metadata.name, run.metadata.version, run.metadata.total_steps
    );

    let config = cartpole_config();
    let device: <InferB as burn::prelude::Backend>::Device = Default::default();

    let policy = DqnPolicy::<CartPoleEnv, _, _, InferB>::new(
        VecEncoder::new(4),
        UsizeActionMapper::new(2),
        &config,
        device,
    )
    .load(run.best_checkpoint_path().with_extension(""))
    .expect("failed to load best.mpk -- run `cargo run --example cartpole --release` first");

    println!("Running greedy evaluation ({NUM_ENVS} envs)...\n");

    let gym = BevyGymPlugin::new(|_| CartPoleViz(CartPoleEnv::new()), NUM_ENVS);

    let mut app = App::new();
    if render {
        app.add_plugins(gym);
        #[cfg(feature = "render")]
        add_render_plugins(&mut app, speed);
    } else {
        add_headless_plugins(&mut app);
        app.add_plugins(gym.headless());
    }
    app.insert_non_send_resource(DqnPolicyResource { policy })
        .add_systems(
            FixedUpdate,
            (
                eval_policy_system.after(GymSet::ManualReset),
                eval_log_and_stop_system.after(GymSet::ManualReset),
            ),
        )
        .run();
}
