//! Train and evaluate Acrobot-v1 with the shared Gymnasium book dynamics.
//!
//! Run `cargo run --release --example acrobot -- train --steps 1000000`.
//! The `eval`, `watch`, and `gif` subcommands consume a saved policy.
//!
//! Three discrete actions apply -1, 0, or 1 N m at the elbow joint. Observations
//! contain both angle cosine/sine pairs followed by both angular velocities.
//! Each 0.2-second transition returns -1 until the free end exceeds 1 metre
//! above the fixed joint; that terminal transition returns 0. The external
//! time limit truncates after 500 transitions when the goal has not been reached.
//!
//! DQN normalizes angular velocities by 4 pi and 9 pi radians per second.
//! Potential shaping affects optimizer rewards only; evaluation retains the
//! original rewards. The browser uses the same architecture and normalization.
//!
//! The default model and local restrictions are documented on
//! [`bevy_gym::environments::Acrobot`]. Reset uses a reproducible `SplitMix64`
//! stream; seed values do not identify the same samples as `NumPy` PCG64.
//! The Gymnasium source derives from `RLPy` under BSD-3-Clause; see
//! `LICENSES/ACROBOT-BSD-3-Clause.txt`.

use std::error::Error;

use std::path::Path;

use bevy_gym::training::DqnConfig;
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::{run_discrete_workflow, DiscreteDqnExample, DqnAction};

/// `MAX_VELOCITY_1` used by this example.
const MAX_VELOCITY_1: f64 = 4.0 * std::f64::consts::PI;
/// `MAX_VELOCITY_2` used by this example.
const MAX_VELOCITY_2: f64 = 9.0 * std::f64::consts::PI;
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

/// Native workflow adapter around the shared Gymnasium dynamics.
#[derive(Debug, Clone)]
struct Acrobot {
    /// Original book dynamics with the external 500-transition limit.
    inner: bevy_gym::TimeLimit<bevy_gym::environments::Acrobot>,
}

impl Default for Acrobot {
    fn default() -> Self {
        Self {
            inner: bevy_gym::TimeLimit::new(
                bevy_gym::environments::Acrobot::default(),
                MAX_EPISODE_STEPS,
            )
            .expect("positive episode cap"),
        }
    }
}

impl Acrobot {
    /// Construct a bounded state for transition tests.
    #[cfg(test)]
    fn from_state(values: [f64; 4]) -> Self {
        let state = bevy_gym::environments::AcrobotState::try_from(values)
            .expect("bounded diagnostic state");
        Self {
            inner: bevy_gym::TimeLimit::new(
                bevy_gym::environments::Acrobot::from_state(state),
                MAX_EPISODE_STEPS,
            )
            .expect("positive episode cap"),
        }
    }
}

impl Env for Acrobot {
    type Action = AcrobotAction;
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
        let action = bevy_gym::environments::AcrobotAction::try_from(action.as_index())
            .expect("native action has one of the three valid indices");
        let result = self.inner.step(action);
        Step {
            observation: result.observation.to_vec(),
            reward: result.reward,
            status: result.status,
            info: result.info,
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
    // A -90 validation stop leaves margin for the separate -100 qualification gate.
    const SOLVED_MEAN_REWARD: f64 = -90.0;
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

/// Recover free-end height from Gymnasium's trigonometric observation.
fn observation_height(observation: &[f32]) -> f64 {
    let cos_1 = f64::from(observation.first().copied().unwrap_or(1.0));
    let sin_1 = f64::from(observation.get(1).copied().unwrap_or(0.0));
    let cos_2 = f64::from(observation.get(2).copied().unwrap_or(1.0));
    let sin_2 = f64::from(observation.get(3).copied().unwrap_or(0.0));
    -cos_1 - (cos_1 * cos_2 - sin_1 * sin_2)
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
            1.0 / 15.0,
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

    /// Advance policy playback at Gymnasium's 15 frames per second.
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
        let theta_1 = env.inner.inner().state()[0] as f32;
        let theta_2 = env.inner.inner().state()[1] as f32;
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
    fn native_adapter_matches_shared_resets_transitions_and_time_limit() {
        let mut native = Acrobot::default();
        let mut shared = bevy_gym::TimeLimit::new(bevy_gym::environments::Acrobot::default(), 500)
            .expect("positive time limit");
        for seed in 0..33 {
            assert_eq!(
                native.reset(Some(seed)).observation,
                shared.reset(Some(seed)).observation
            );
            for index in 0..500 {
                let action = index % 3;
                let native_step = native.step(AcrobotAction::from_index(action));
                let shared_step = shared.step(
                    bevy_gym::environments::AcrobotAction::try_from(action).expect("three actions"),
                );
                assert_eq!(native_step.observation, shared_step.observation);
                assert_eq!(native_step.reward.to_bits(), shared_step.reward.to_bits());
                assert_eq!(native_step.status, shared_step.status);
            }
        }
    }

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
        let mut acrobot = Acrobot::from_state([std::f64::consts::PI, 0.0, 0.0, 0.0]);
        let step = acrobot.step(AcrobotAction::Coast);
        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(step.reward, 0.0);
    }

    #[test]
    fn episode_truncates_at_five_hundred_transitions() {
        let mut acrobot = Acrobot::from_state([0.0; 4]);
        for _ in 0..MAX_EPISODE_STEPS - 1 {
            assert_eq!(
                acrobot.step(AcrobotAction::Coast).status,
                EpisodeStatus::Continuing
            );
        }

        let step = acrobot.step(AcrobotAction::Coast);

        assert_eq!(step.status, EpisodeStatus::Truncated);
    }
}
