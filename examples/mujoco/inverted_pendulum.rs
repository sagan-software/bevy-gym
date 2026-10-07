//! MuJoCo-backed `InvertedPendulum-v5` with the official XML, reset, reward,
//! observation, termination, frame skip, and 1,000-step time limit.

#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;

use std::error::Error;
use std::path::Path;

use bevy_gym::mujoco::MujocoSimulation;
use bevy_gym::training::{
    run_continuous_workflow, ContinuousPpoExample, RecurrentBehaviorSample, RecurrentPpoConfig,
    SplitMix64,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
use burn as _;
use clap as _;
use mujoco_rs as _;
use serde_json as _;
use shakmaty as _;
use tokio as _;

/// `MODEL_XML` used by this example.
const MODEL_XML: &str = include_str!("assets/inverted_pendulum.xml");
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 3.0;
/// `HEALTHY_ANGLE_LIMIT` used by this example.
const HEALTHY_ANGLE_LIMIT: f32 = 0.2;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Exact native environment state plus deterministic reset sampling.
#[derive(Debug)]
struct InvertedPendulum {
    /// Owned `MuJoCo` model and data.
    simulation: MujocoSimulation,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for InvertedPendulum {
    fn default() -> Self {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        Self {
            simulation: MujocoSimulation::from_xml_string(MODEL_XML)
                .expect("bundled InvertedPendulum XML must compile"),
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl InvertedPendulum {
    /// Return Gymnasium's `qpos` followed by `qvel` observation.
    fn observation(&self) -> Vec<f32> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        self.simulation
            .qpos()
            .iter()
            .chain(self.simulation.qvel())
            .map(|value| *value as f32)
            .collect()
    }

    /// Return a stabilizing cart controller for behavior initialization.
    fn expert_action(observation: &[f32]) -> f32 {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let position = observation.first().copied().unwrap_or(0.0);
        let angle = observation.get(1).copied().unwrap_or(0.0);
        let cart_velocity = observation.get(2).copied().unwrap_or(0.0);
        let angular_velocity = observation.get(3).copied().unwrap_or(0.0);
        1.520_59f32
            .mul_add(
                angular_velocity,
                0.723_768f32.mul_add(
                    cart_velocity,
                    0.327_802f32.mul_add(position, 7.537_257 * angle),
                ),
            )
            .clamp(-ACTION_LIMIT, ACTION_LIMIT)
    }
}

impl Env for InvertedPendulum {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        let qpos = [
            self.rng.f64_between(-0.01, 0.01),
            self.rng.f64_between(-0.01, 0.01),
        ];
        let qvel = [
            self.rng.f64_between(-0.01, 0.01),
            self.rng.f64_between(-0.01, 0.01),
        ];
        self.simulation
            .set_state(&qpos, &qvel)
            .expect("bundled model dimensions are fixed");
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let action = f64::from(
            action
                .first()
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT),
        );
        self.simulation
            .step(&[action], 2)
            .expect("bundled model has one actuator");
        self.elapsed_steps = self.elapsed_steps.saturating_add(1);
        let observation = self.observation();
        let healthy = observation.iter().all(|value| value.is_finite())
            && observation
                .get(1)
                .is_some_and(|angle| angle.abs() <= HEALTHY_ANGLE_LIMIT);
        let status = if !healthy {
            EpisodeStatus::Terminated
        } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Continuing
        };
        Step {
            observation,
            reward: f64::from(healthy),
            status,
            info: (),
        }
    }
}

impl ContinuousPpoExample for InvertedPendulum {
    const ENV_NAME: &'static str = "mujoco-inverted-pendulum";
    const GYMNASIUM_ID: &'static str = "InvertedPendulum-v5";
    const OBSERVATION_DIM: usize = 4;
    const ACTION_LOW: &'static [f32] = &[-3.0];
    const ACTION_HIGH: &'static [f32] = &[3.0];
    const DEFAULT_TRAIN_STEPS: usize = 300_000;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 3e-4;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 1e-3;
    const SOLVED_MEAN_REWARD: f64 = 950.0;
    const GIF_PATH: &'static str = "docs/images/mujoco-inverted-pendulum.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 8_192;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 80;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        RecurrentPpoConfig {
            actor_hidden_size: 32,
            critic_hidden_sizes: vec![64, 32],
            gamma: 0.995,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.001,
            initial_log_std: -1.5,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        vec![
            observation.first().copied().unwrap_or(0.0),
            observation.get(1).copied().unwrap_or(0.0) / HEALTHY_ANGLE_LIMIT,
            observation.get(2).copied().unwrap_or(0.0) / 5.0,
            observation.get(3).copied().unwrap_or(0.0) / 5.0,
        ]
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut rng = SplitMix64::new(seed);
        (0..sample_count)
            .map(|_| {
                let observation = vec![
                    rng.f32_between(-0.8, 0.8),
                    rng.f32_between(-0.19, 0.19),
                    rng.f32_between(-1.5, 1.5),
                    rng.f32_between(-2.5, 2.5),
                ];
                let action = Self::expert_action(&observation);
                RecurrentBehaviorSample {
                    observation: Self::encode_observation(&observation),
                    action: vec![action],
                }
            })
            .collect()
    }

    fn training_reward(
        _observation: &[f32],
        _action: &[f32],
        environment_reward: f64,
        next_observation: &[f32],
        status: EpisodeStatus,
        reward_scale: f64,
    ) -> f64 {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if status == EpisodeStatus::Terminated {
            return -10.0 * reward_scale;
        }
        let angle = f64::from(next_observation.get(1).copied().unwrap_or(0.0));
        8.0f64.mul_add(-angle.powi(2), environment_reward)
    }

    fn is_success(_observation: &[f32], reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Truncated && reward >= Self::SOLVED_MEAN_REWARD
    }

    fn watch(checkpoint: &Path) -> Result<(), Box<dyn Error>> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        #[cfg(not(feature = "render"))]
        let _ = checkpoint;
        #[cfg(feature = "render")]
        {
            render::watch(checkpoint);
            Ok(())
        }
        #[cfg(not(feature = "render"))]
        {
            Err("watch mode requires the `render` feature".into())
        }
    }

    fn gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        #[cfg(not(feature = "render"))]
        let _ = (checkpoint, output);
        #[cfg(feature = "render")]
        return render::gif(checkpoint, output);
        #[cfg(not(feature = "render"))]
        return Err("GIF mode requires the `render` feature".into());
    }

    fn video(run_directory: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        #[cfg(not(feature = "render"))]
        let _ = (run_directory, output);
        #[cfg(feature = "render")]
        return render::video(run_directory, output);
        #[cfg(not(feature = "render"))]
        return Err("video mode requires the `render` feature".into());
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    run_continuous_workflow::<InvertedPendulum>()
}

/// Render-only scene and reusable recorder integration.
#[cfg(feature = "render")]
mod render {
    use bevy::prelude::*;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{
        BevyGymRecorderPlugin, RecordingCallbackError, RecordingFrame, RecordingSettings,
    };
    use bevy_gym::training::RecurrentPpoPolicy;
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    use super::{ContinuousPpoExample, Env, EpisodeStatus, Error, InvertedPendulum, Path};

    /// Scene policy and native environment state.
    #[derive(Resource)]
    struct VisualState {
        /// Current checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Native environment.
        environment: InvertedPendulum,
        /// Exact current observation.
        observation: Vec<f32>,
        /// Actor memory.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Interactive 25-Hz policy clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Rendered cart marker.
    #[derive(Component)]
    struct Cart;

    /// Rendered pole marker.
    #[derive(Component)]
    struct Pole;

    /// Environment-specific renderer registered with one Bevy application.
    pub(super) struct ExampleRendererPlugin {
        /// Initial checkpoint.
        checkpoint: std::path::PathBuf,
        /// Optional finite recorder settings.
        recording: Option<RecordingSettings>,
    }

    impl ExampleRendererPlugin {
        /// Construct interactive or finite playback.
        fn new(checkpoint: &Path, recording: Option<RecordingSettings>) -> Self {
            Self {
                checkpoint: checkpoint.to_path_buf(),
                recording,
            }
        }
    }

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
            let policy = load_policy(&self.checkpoint)
                .expect("checkpoint was validated before renderer construction");
            let memory = policy.initial_memory();
            let mut environment = InvertedPendulum::default();
            let observation = environment.reset(Some(12_345)).observation;
            app.insert_resource(VisualState {
                policy,
                environment,
                observation,
                memory,
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.04, TimerMode::Repeating)))
            .insert_resource(ClearColor(Color::BLACK))
            .insert_resource(GlobalAmbientLight {
                color: Color::WHITE,
                brightness: 200.0,
                affects_lightmapped_meshes: true,
            })
            .add_systems(Startup, setup_scene);
            if let Some(settings) = self.recording.clone() {
                app.add_plugins(
                    BevyGymRecorderPlugin::new(settings)
                        .with_checkpoint_loader(load_checkpoint)
                        .with_frame_driver(drive_recording_frame),
                );
            } else {
                app.add_systems(Update, (advance_watch, sync_scene).chain());
            }
        }
    }

    /// Start a rendered application with one renderer plugin.
    fn run(checkpoint: &Path, recording: Option<RecordingSettings>) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym MuJoCo InvertedPendulum-v5".into(),
                        resolution: WindowResolution::new(720, 720),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(ExampleRendererPlugin::new(checkpoint, recording));
        app.run();
    }

    /// Open interactive best-checkpoint playback.
    pub(super) fn watch(checkpoint: &Path) {
        run(checkpoint, None);
    }

    /// Record the five-second best-checkpoint preview.
    pub(super) fn gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        let settings = RecordingSettings::gif(output.to_path_buf(), checkpoint.to_path_buf())?;
        run(checkpoint, Some(settings));
        Ok(())
    }

    /// Record the 30-second dynamic-checkpoint progression.
    pub(super) fn video(run_directory: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let settings = RecordingSettings::training_video(output.to_path_buf(), run_directory)?;
        let checkpoint = settings
            .timeline()
            .first_checkpoint()
            .expect("a training timeline always has a first checkpoint")
            .to_path_buf();
        run(&checkpoint, Some(settings));
        Ok(())
    }

    /// Load a policy with the example's stable runtime profile.
    fn load_policy(checkpoint: &Path) -> Result<RecurrentPpoPolicy, Box<dyn Error>> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        RecurrentPpoPolicy::load(
            checkpoint,
            4,
            4,
            1,
            &[-3.0],
            &[3.0],
            &InvertedPendulum::ppo_config(3e-4, 1e-3),
        )
        .map_err(Into::into)
    }

    /// Spawn the official rail, cart, pole, light, and camera geometry.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.3, 2.04).looking_at(Vec3::ZERO, Vec3::Y),
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
            MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.7))),
            Transform::from_rotation(horizontal),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.1, 0.2))),
            MeshMaterial3d(materials.add(Color::srgb(0.7, 0.7, 0.0))),
            Transform::from_rotation(horizontal),
            Cart,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.049, 0.6))),
            MeshMaterial3d(materials.add(Color::srgb(0.0, 0.7, 0.7))),
            Pole,
        ));
    }

    /// Advance interactive playback at the official 25-Hz control rate.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut state: ResMut<'_, VisualState>,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if clock.0.tick(time.delta()).just_finished() {
            advance(&mut state, 1);
        }
    }

    /// Run deterministic policy transitions and reset completed episodes.
    fn advance(state: &mut VisualState, steps: usize) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        for _ in 0..steps {
            let encoded = InvertedPendulum::encode_observation(&state.observation);
            let Ok(action) = state.policy.mean_action(&encoded, &state.memory) else {
                return;
            };
            state.memory = action.next_memory;
            let transition = state.environment.step(action.action);
            state.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                state.observation = state.environment.reset(Some(12_345)).observation;
                state.memory = state.policy.initial_memory();
            }
        }
    }

    /// Synchronize native qpos with the Bevy geometry.
    fn sync_scene(
        state: Res<'_, VisualState>,
        mut cart: Single<'_, '_, &mut Transform, With<Cart>>,
        mut pole: Single<'_, '_, &mut Transform, (With<Pole>, Without<Cart>)>,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let qpos = state.environment.simulation.qpos();
        let position = qpos.first().copied().unwrap_or(0.0) as f32;
        let angle = qpos.get(1).copied().unwrap_or(0.0) as f32;
        let direction = Vec2::new(angle.sin(), angle.cos());
        cart.translation = Vec3::new(position, 0.0, 0.0);
        pole.translation = Vec3::new(direction.x.mul_add(0.3, position), direction.y * 0.3, 0.0);
        pole.rotation = Quat::from_rotation_z(-angle);
    }

    /// Replace the policy and reset the episode at a video segment boundary.
    fn load_checkpoint(world: &mut World, checkpoint: &Path) -> Result<(), RecordingCallbackError> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let policy = load_policy(checkpoint)
            .map_err(|error| -> RecordingCallbackError { error.to_string().into() })?;
        let memory = policy.initial_memory();
        let mut state = world.resource_mut::<VisualState>();
        state.policy = policy;
        state.memory = memory;
        state.observation = state.environment.reset(Some(12_345)).observation;
        Ok(())
    }

    /// Advance at 25 Hz relative to the encoded frame rate and synchronize.
    fn drive_recording_frame(
        world: &mut World,
        frame: RecordingFrame,
    ) -> Result<(), RecordingCallbackError> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if !frame.is_first_in_segment() {
            let current = frame.frame_in_segment();
            let previous = current.saturating_sub(1);
            let rate = usize::from(frame.frames_per_second());
            let steps = current * 25 / rate - previous * 25 / rate;
            advance(&mut world.resource_mut::<VisualState>(), steps);
        }
        world.run_system_cached(sync_scene)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_and_reward_match_the_official_contract() {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut environment = InvertedPendulum::default();
        let reset = environment.reset(Some(42));
        assert_eq!(reset.observation.len(), 4);
        assert!(reset.observation.iter().all(|value| value.abs() <= 0.01));

        let transition = environment.step(vec![0.0]);
        assert_eq!(transition.reward, 1.0);
        assert_eq!(transition.status, EpisodeStatus::Continuing);
    }

    #[test]
    fn expert_controller_exceeds_the_registry_threshold() {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut mean_reward = 0.0;
        for seed in 0..20 {
            let mut environment = InvertedPendulum::default();
            let mut observation = environment.reset(Some(seed)).observation;
            let mut reward = 0.0;
            loop {
                let transition =
                    environment.step(vec![InvertedPendulum::expert_action(&observation)]);
                observation = transition.observation;
                reward += transition.reward;
                if transition.status != EpisodeStatus::Continuing {
                    break;
                }
            }
            mean_reward += reward;
        }
        mean_reward /= 20.0;
        assert!(mean_reward >= InvertedPendulum::SOLVED_MEAN_REWARD);
    }
}
