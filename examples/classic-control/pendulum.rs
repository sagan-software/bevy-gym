//! Gymnasium Pendulum-v1 with shared dynamics and recurrent PPO.
//!
//! The pendulum starts with angle in [-pi, pi) radians and angular velocity
//! in [-1, 1) radians per second. Torque lies in [-2, 2] newton metres.
//! Observations are angle cosine, angle sine, and angular velocity.
//! The policy divides angular velocity by its 8-radian-per-second limit.
//!
//! Reward is the negative pre-transition cost: squared normalized angle,
//! plus 0.1 times squared angular velocity, plus 0.001 times squared torque.
//! Zero angle is upright. Each transition advances 0.05 seconds; the external
//! time limit truncates after 200 transitions. Gravity is fixed at 10 m/s².
//! Custom gravity and reset bounds are unsupported. `SplitMix64` resets follow
//! the original uniform bounds without reproducing `NumPy`'s seed stream.
//!
//! The default recipe uses eight environments, 64 transitions per rollout,
//! actor rate 0.003, critic rate 0.001, and reward scale 0.1 for optimization.
//! Evaluation reports unscaled rewards and uses deterministic mean actions.
//!
//! Source: [Gymnasium Pendulum at revision 7a119138](https://github.com/Farama-Foundation/Gymnasium/blob/7a1191388aa4aa973d3a5e4b039899cd99cc991f/gymnasium/envs/classic_control/pendulum.py).
//! See `gymnasium-web/PENDULUM.md` for browser controls and qualification gates.

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
use serde as _;
use serde_json as _;

/// Angular-velocity observation scale, in radians per second.
const MAX_SPEED: f32 = 8.0;
/// Source torque limit, in newton metres, used by the arrow renderer.
#[cfg(feature = "render")]
const MAX_TORQUE: f32 = 2.0;
/// Default external episode cap.
const MAX_EPISODE_STEPS: usize = 200;

/// Native workflow adapter around the shared Gymnasium dynamics.
#[derive(Debug, Clone)]
struct Pendulum {
    /// Original dynamics with the external 200-transition limit.
    inner: bevy_gym::TimeLimit<bevy_gym::environments::Pendulum>,
}

impl Default for Pendulum {
    fn default() -> Self {
        Self {
            inner: bevy_gym::TimeLimit::new(
                bevy_gym::environments::Pendulum::default(),
                MAX_EPISODE_STEPS,
            )
            .expect("positive episode cap"),
        }
    }
}

impl Pendulum {
    /// Construct one controlled state for transition tests.
    #[cfg(test)]
    fn from_state(angle: f64, angular_velocity: f64) -> Self {
        let state = bevy_gym::environments::PendulumState::try_from([angle, angular_velocity])
            .expect("finite diagnostic state");
        Self {
            inner: bevy_gym::TimeLimit::new(
                bevy_gym::environments::Pendulum::from_state(state),
                MAX_EPISODE_STEPS,
            )
            .expect("positive episode cap"),
        }
    }
}

impl Env for Pendulum {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        let result = self.inner.reset(seed);
        Reset {
            observation: result.observation.to_vec(),
            info: result.info,
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let raw = action.first().copied().unwrap_or(0.0);
        let torque = bevy_gym::environments::PendulumAction::try_from(raw)
            .expect("policy torque must be finite");
        let result = self.inner.step(torque);
        Step {
            observation: result.observation.to_vec(),
            reward: result.reward,
            status: result.status,
            info: result.info,
        }
    }
}

impl ContinuousPpoExample for Pendulum {
    const ENV_NAME: &'static str = "pendulum";
    const GYMNASIUM_ID: &'static str = "Pendulum-v1";
    const OBSERVATION_DIM: usize = 3;
    const ACTION_LOW: &'static [f32] = &[-2.0];
    const ACTION_HIGH: &'static [f32] = &[2.0];
    const DEFAULT_TRAIN_STEPS: usize = 2_000_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 64;
    const DEFAULT_NUM_ENVS: usize = 8;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const DEFAULT_REWARD_SCALE: f64 = 0.1;
    const SOLVED_MEAN_REWARD: f64 = -200.0;
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
        let checkpoint = checkpoint.display();
        println!("watching checkpoint={checkpoint}");
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
        let rotation = env.inner.inner().state()[0] as f32 + std::f32::consts::FRAC_PI_2;
        let direction = Vec2::from_angle(rotation);
        rod.translation = (direction * (SCALE * 0.5)).extend(1.0);
        rod.rotation = Quat::from_rotation_z(rotation);
        end.translation = (direction * SCALE).extend(2.0);
        if let Some(torque) = env.inner.inner().last_torque() {
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
    fn adapter_preserves_seeded_resets_and_empty_action_coasts() {
        let mut native = Pendulum::default();
        let mut shared = bevy_gym::environments::Pendulum::default();
        assert_eq!(
            native.reset(Some(42)).observation,
            shared.reset(Some(42)).observation
        );
        let torque = bevy_gym::environments::PendulumAction::try_from(0.0).expect("zero torque");
        let result = native.step(vec![]);
        let expected = shared.step(torque);
        assert_eq!(result.observation, expected.observation);
        assert_eq!(result.reward.to_bits(), expected.reward.to_bits());
        assert_eq!(result.status, expected.status);
    }

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
        let mut pendulum = Pendulum::from_state(std::f64::consts::PI, 1.0);

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
