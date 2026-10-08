//! MuJoCo-backed `InvertedDoublePendulum-v5` using Gymnasium's exact model,
//! observation, reset distribution, reward, termination, and time limit.

use std::error::Error;
use std::path::Path;

use bevy_gym::mujoco::MujocoSimulation;
use bevy_gym::training::{
    run_continuous_workflow, ContinuousPpoExample, RecurrentBehaviorSample, RecurrentPpoConfig,
    SplitMix64,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// `MODEL_XML` used by this example.
const MODEL_XML: &str = include_str!("assets/inverted_double_pendulum.xml");
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;

/// Exact native double-pendulum task.
#[derive(Debug)]
struct InvertedDoublePendulum {
    /// Owned `MuJoCo` model and data.
    simulation: MujocoSimulation,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for InvertedDoublePendulum {
    fn default() -> Self {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        Self {
            simulation: MujocoSimulation::from_xml_string(MODEL_XML)
                .expect("bundled InvertedDoublePendulum XML must compile"),
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl InvertedDoublePendulum {
    /// Return Gymnasium's nine transformed observation values.
    fn observation(&self) -> Vec<f32> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let qpos = self.simulation.qpos();
        let qvel = self.simulation.qvel();
        let force = self.simulation.constraint_forces();
        vec![
            qpos.first().copied().unwrap_or(0.0) as f32,
            qpos.get(1).copied().unwrap_or(0.0).sin() as f32,
            qpos.get(2).copied().unwrap_or(0.0).sin() as f32,
            qpos.get(1).copied().unwrap_or(0.0).cos() as f32,
            qpos.get(2).copied().unwrap_or(0.0).cos() as f32,
            qvel.first().copied().unwrap_or(0.0).clamp(-10.0, 10.0) as f32,
            qvel.get(1).copied().unwrap_or(0.0).clamp(-10.0, 10.0) as f32,
            qvel.get(2).copied().unwrap_or(0.0).clamp(-10.0, 10.0) as f32,
            force.first().copied().unwrap_or(0.0).clamp(-10.0, 10.0) as f32,
        ]
    }

    /// Return the LQR-style controller used for actor initialization.
    fn expert_action(observation: &[f32]) -> f32 {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
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
        let feedback = 0.030_859_3_f32.mul_add(
            position,
            0.539_115f32.mul_add(
                angular_velocity_2,
                0.405_1f32.mul_add(
                    angular_velocity_1,
                    0.076_782f32.mul_add(
                        cart_velocity,
                        0.417_91f32.mul_add(angle_1, 3.695_299 * angle_2),
                    ),
                ),
            ),
        );
        (-feedback).clamp(-1.0, 1.0)
    }
}

impl Env for InvertedDoublePendulum {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        let qpos: [f64; 3] = std::array::from_fn(|_| self.rng.f64_between(-0.1, 0.1));
        let qvel: [f64; 3] = std::array::from_fn(|_| self.rng.standard_normal_f64() * 0.1);
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
        let action = f64::from(action.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0));
        self.simulation
            .step(&[action], 5)
            .expect("bundled model has one actuator");
        self.elapsed_steps = self.elapsed_steps.saturating_add(1);
        let [tip_x, _, tip_height] = self
            .simulation
            .site_position("tip")
            .expect("bundled model defines the tip site");
        let qvel = self.simulation.qvel();
        let angular_velocity_1 = qvel.get(1).copied().unwrap_or(0.0);
        let angular_velocity_2 = qvel.get(2).copied().unwrap_or(0.0);
        let healthy = tip_height > 1.0;
        let distance_penalty = (tip_height - 2.0).mul_add(tip_height - 2.0, 0.01 * tip_x.powi(2));
        let velocity_penalty = 0.001f64.mul_add(
            angular_velocity_1.powi(2),
            0.005 * angular_velocity_2.powi(2),
        );
        let reward = if healthy { 10.0 } else { 0.0 } - distance_penalty - velocity_penalty;
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
    const ENV_NAME: &'static str = "mujoco-inverted-double-pendulum";
    const GYMNASIUM_ID: &'static str = "InvertedDoublePendulum-v5";
    const OBSERVATION_DIM: usize = 9;
    const ACTION_LOW: &'static [f32] = &[-1.0];
    const ACTION_HIGH: &'static [f32] = &[1.0];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const DEFAULT_EVAL_INTERVAL: usize = 25_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 3e-4;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 1e-3;
    const SOLVED_MEAN_REWARD: f64 = 9_100.0;
    const GIF_PATH: &'static str = "docs/images/mujoco-inverted-double-pendulum.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 16_384;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 100;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        RecurrentPpoConfig {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![128, 64],
            gamma: 0.995,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.0005,
            initial_log_std: -2.0,
            ..RecurrentPpoConfig::default()
        }
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        const STEPS_PER_RESET: usize = 128;
        let mut environment = Self::default();
        let mut episode = 0_u64;
        let mut observation = environment.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);

        // Train on states the controller actually visits. Frequent seeded
        // resets retain broad recovery coverage instead of overfitting the
        // long near-upright portion of a successful 1,000-step rollout.
        for sample_index in 0..sample_count {
            let action = Self::expert_action(&observation);
            demonstrations.push(RecurrentBehaviorSample {
                observation: observation.clone(),
                action: vec![action],
            });
            let transition = environment.step(vec![action]);
            observation = transition.observation;
            let reached_reset_boundary = (sample_index + 1).is_multiple_of(STEPS_PER_RESET);
            if transition.status != EpisodeStatus::Continuing || reached_reset_boundary {
                episode = episode.wrapping_add(1);
                observation = environment
                    .reset(Some(seed.wrapping_add(episode)))
                    .observation;
            }
        }
        demonstrations
    }

    fn training_reward(
        _observation: &[f32],
        _action: &[f32],
        environment_reward: f64,
        _next_observation: &[f32],
        status: EpisodeStatus,
        reward_scale: f64,
    ) -> f64 {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if status == EpisodeStatus::Terminated {
            -25.0 * reward_scale
        } else {
            environment_reward * reward_scale
        }
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
    run_continuous_workflow::<InvertedDoublePendulum>()
}

/// Render-only scene and recorder integration.
#[cfg(feature = "render")]
mod render {
    use bevy::prelude::*;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{
        BevyGymRecorderPlugin, RecordingCallbackError, RecordingFrame, RecordingSettings,
    };
    use bevy_gym::training::RecurrentPpoPolicy;

    use super::{ContinuousPpoExample, Env, EpisodeStatus, Error, InvertedDoublePendulum, Path};

    /// Current policy and native environment.
    #[derive(Resource)]
    struct VisualState {
        /// Current checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Native environment.
        environment: InvertedDoublePendulum,
        /// Exact current observation.
        observation: Vec<f32>,
        /// Actor memory.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Interactive 20-Hz control clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Cart geometry marker.
    #[derive(Component)]
    struct Cart;

    /// First pole geometry marker.
    #[derive(Component)]
    struct FirstPole;

    /// Second pole geometry marker.
    #[derive(Component)]
    struct SecondPole;

    /// Environment-specific renderer plugin.
    pub(super) struct ExampleRendererPlugin {
        /// Initial checkpoint.
        checkpoint: std::path::PathBuf,
        /// Optional recorder settings.
        recording: Option<RecordingSettings>,
    }

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
            let policy = load_policy(&self.checkpoint)
                .expect("checkpoint was validated before renderer construction");
            let memory = policy.initial_memory();
            let mut environment = InvertedDoublePendulum::default();
            let observation = environment.reset(Some(12_345)).observation;
            app.insert_resource(VisualState {
                policy,
                environment,
                observation,
                memory,
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.05, TimerMode::Repeating)))
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

    /// Run interactive or finite playback.
    fn run(checkpoint: &Path, recording: Option<RecordingSettings>) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        App::new()
            .add_plugins(
                DefaultPlugins
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "bevy-gym MuJoCo InvertedDoublePendulum-v5".into(),
                            resolution: WindowResolution::new(720, 720),
                            present_mode: PresentMode::AutoVsync,
                            resizable: false,
                            ..default()
                        }),
                        ..default()
                    }),
            )
            .add_plugins(ExampleRendererPlugin {
                checkpoint: checkpoint.to_path_buf(),
                recording,
            })
            .run();
    }

    /// Open interactive playback.
    pub(super) fn watch(checkpoint: &Path) {
        run(checkpoint, None);
    }

    /// Record a five-second GIF.
    pub(super) fn gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        let settings = RecordingSettings::gif(output.to_path_buf(), checkpoint.to_path_buf())?;
        run(checkpoint, Some(settings));
        Ok(())
    }

    /// Record the standard 30-second progression.
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

    /// Load one checkpoint policy.
    fn load_policy(checkpoint: &Path) -> Result<RecurrentPpoPolicy, Box<dyn Error>> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        RecurrentPpoPolicy::load(
            checkpoint,
            9,
            9,
            1,
            &[-1.0],
            &[1.0],
            &InvertedDoublePendulum::ppo_config(3e-4, 1e-3),
        )
        .map_err(Into::into)
    }

    /// Spawn geometry matching Gymnasium's `MuJoCo` model.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.45, 4.1225).looking_at(Vec3::new(0.0, 0.6, 0.0), Vec3::Y),
        ));
        commands.spawn((
            PointLight {
                intensity: 1_500.0,
                range: 10.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_xyz(-1.0, 3.0, 3.0),
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
            Mesh3d(meshes.add(Capsule3d::new(0.045, 0.51))),
            MeshMaterial3d(materials.add(Color::srgb(0.0, 0.7, 0.7))),
            FirstPole,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.045, 0.51))),
            MeshMaterial3d(materials.add(Color::srgb(0.0, 0.7, 0.7))),
            SecondPole,
        ));
    }

    /// Advance interactive playback.
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

    /// Advance deterministic policy steps.
    fn advance(state: &mut VisualState, steps: usize) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        for _ in 0..steps {
            let Ok(action) = state.policy.mean_action(&state.observation, &state.memory) else {
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

    /// Synchronize qpos with cart and two linked poles.
    fn sync_scene(
        state: Res<'_, VisualState>,
        mut cart: Single<'_, '_, &mut Transform, With<Cart>>,
        mut first: Single<
            '_,
            '_,
            &mut Transform,
            (With<FirstPole>, Without<Cart>, Without<SecondPole>),
        >,
        mut second: Single<
            '_,
            '_,
            &mut Transform,
            (With<SecondPole>, Without<Cart>, Without<FirstPole>),
        >,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let qpos = state.environment.simulation.qpos();
        let position = qpos.first().copied().unwrap_or(0.0) as f32;
        let angle_1 = qpos.get(1).copied().unwrap_or(0.0) as f32;
        let angle_2 = angle_1 + qpos.get(2).copied().unwrap_or(0.0) as f32;
        let direction_1 = Vec2::new(angle_1.sin(), angle_1.cos());
        let direction_2 = Vec2::new(angle_2.sin(), angle_2.cos());
        let joint = Vec2::new(position, 0.0) + direction_1 * 0.6;
        cart.translation = Vec3::new(position, 0.0, 0.0);
        first.translation = Vec3::new(
            direction_1.x.mul_add(0.3, position),
            direction_1.y * 0.3,
            0.0,
        );
        first.rotation = Quat::from_rotation_z(-angle_1);
        second.translation = Vec3::new(
            direction_2.x.mul_add(0.3, joint.x),
            direction_2.y.mul_add(0.3, joint.y),
            0.0,
        );
        second.rotation = Quat::from_rotation_z(-angle_2);
    }

    /// Load a segment checkpoint and reset playback.
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

    /// Advance at the official 20-Hz control rate and synchronize.
    fn drive_recording_frame(
        world: &mut World,
        frame: RecordingFrame,
    ) -> Result<(), RecordingCallbackError> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if !frame.is_first_in_segment() {
            let current = frame.frame_in_segment();
            let previous = current.saturating_sub(1);
            let rate = usize::from(frame.frames_per_second());
            let steps = current * 20 / rate - previous * 20 / rate;
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
    fn reset_matches_the_nine_value_observation_contract() {
        let mut environment = InvertedDoublePendulum::default();
        let reset = environment.reset(Some(42));
        assert_eq!(reset.observation.len(), 9);
        assert!(reset.observation.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn expert_controller_exceeds_the_registry_threshold_on_held_out_resets() {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut mean_reward = 0.0;
        for seed in 9_001..9_101 {
            let mut environment = InvertedDoublePendulum::default();
            let mut observation = environment.reset(Some(seed)).observation;
            let mut reward = 0.0;
            loop {
                let transition =
                    environment.step(vec![InvertedDoublePendulum::expert_action(&observation)]);
                observation = transition.observation;
                reward += transition.reward;
                if transition.status != EpisodeStatus::Continuing {
                    break;
                }
            }
            mean_reward += reward;
        }
        mean_reward /= 100.0;
        assert!(
            mean_reward >= InvertedDoublePendulum::SOLVED_MEAN_REWARD,
            "mean reward was {mean_reward}"
        );
    }
}
