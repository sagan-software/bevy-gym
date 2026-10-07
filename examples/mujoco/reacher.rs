//! MuJoCo-backed `Reacher-v5` using Gymnasium's exact model, randomized target,
//! transformed observation, action cost, frame skip, and 50-step time limit.

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
use tokio as _;

/// `MODEL_XML` used by this example.
const MODEL_XML: &str = include_str!("assets/reacher.xml");
/// `LINK_0_LENGTH` used by this example.
const LINK_0_LENGTH: f32 = 0.1;
/// `LINK_1_LENGTH` used by this example.
const LINK_1_LENGTH: f32 = 0.11;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 50;

/// Exact native two-joint reacher task.
#[derive(Debug)]
struct Reacher {
    /// Owned `MuJoCo` model and data.
    simulation: MujocoSimulation,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for Reacher {
    fn default() -> Self {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        Self {
            simulation: MujocoSimulation::from_xml_string(MODEL_XML)
                .expect("bundled Reacher XML must compile"),
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Reacher {
    /// Return Gymnasium's ten transformed observation values.
    fn observation(&self) -> Vec<f32> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let qpos = self.simulation.qpos();
        let qvel = self.simulation.qvel();
        let fingertip = self
            .simulation
            .body_position("fingertip")
            .expect("bundled model defines the fingertip body");
        let target = self
            .simulation
            .body_position("target")
            .expect("bundled model defines the target body");
        let angle_0 = qpos.first().copied().unwrap_or(0.0);
        let angle_1 = qpos.get(1).copied().unwrap_or(0.0);
        vec![
            angle_0.cos() as f32,
            angle_1.cos() as f32,
            angle_0.sin() as f32,
            angle_1.sin() as f32,
            qpos.get(2).copied().unwrap_or(0.0) as f32,
            qpos.get(3).copied().unwrap_or(0.0) as f32,
            qvel.first().copied().unwrap_or(0.0) as f32,
            qvel.get(1).copied().unwrap_or(0.0) as f32,
            (fingertip[0] - target[0]) as f32,
            (fingertip[1] - target[1]) as f32,
        ]
    }

    /// Return a low-energy inverse-kinematics controller for actor initialization.
    fn expert_action(observation: &[f32]) -> [f32; 2] {
        Self::expert_action_with_joint_gains(observation, [0.29, 0.25], [0.05, 0.05])
    }

    /// Apply independent gains to each inverse-kinematics joint target.
    fn expert_action_with_joint_gains(
        observation: &[f32],
        position_gain: [f32; 2],
        velocity_gain: [f32; 2],
    ) -> [f32; 2] {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
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
        let target = [
            observation.get(4).copied().unwrap_or(0.0),
            observation.get(5).copied().unwrap_or(0.0),
        ];
        let [desired_0, desired_1] = inverse_kinematic_angles(target, [angle_0, angle_1]);
        let velocity_0 = observation.get(6).copied().unwrap_or(0.0);
        let velocity_1 = observation.get(7).copied().unwrap_or(0.0);
        [
            position_gain[0]
                .mul_add(
                    angle_difference(desired_0, angle_0),
                    -(velocity_gain[0] * velocity_0),
                )
                .clamp(-1.0, 1.0),
            position_gain[1]
                .mul_add(
                    angle_difference(desired_1, angle_1),
                    -(velocity_gain[1] * velocity_1),
                )
                .clamp(-1.0, 1.0),
        ]
    }
}

/// Return the reachable two-link solution closest to the current joint pose.
fn inverse_kinematic_angles(target: [f32; 2], current: [f32; 2]) -> [f32; 2] {
    // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
    let radius_squared = target[1].mul_add(target[1], target[0].powi(2)).clamp(
        (LINK_0_LENGTH - LINK_1_LENGTH).powi(2),
        (LINK_0_LENGTH + LINK_1_LENGTH).powi(2),
    );
    let elbow_magnitude = (LINK_1_LENGTH.mul_add(
        -LINK_1_LENGTH,
        LINK_0_LENGTH.mul_add(-LINK_0_LENGTH, radius_squared),
    ) / (2.0 * LINK_0_LENGTH * LINK_1_LENGTH))
        .clamp(-1.0, 1.0)
        .acos();
    [elbow_magnitude, -elbow_magnitude]
        .map(|desired_1| {
            let shoulder_offset = (LINK_1_LENGTH * desired_1.sin())
                .atan2(LINK_1_LENGTH.mul_add(desired_1.cos(), LINK_0_LENGTH));
            [target[1].atan2(target[0]) - shoulder_offset, desired_1]
        })
        .into_iter()
        .min_by(|left, right| {
            let left_distance = angle_difference(left[0], current[0]).abs()
                + angle_difference(left[1], current[1]).abs();
            let right_distance = angle_difference(right[0], current[0]).abs()
                + angle_difference(right[1], current[1]).abs();
            left_distance.total_cmp(&right_distance)
        })
        .expect("two inverse-kinematics solutions are always present")
}

/// Return the shortest signed angular difference.
fn angle_difference(target: f32, current: f32) -> f32 {
    let difference = target - current;
    difference.sin().atan2(difference.cos())
}

impl Env for Reacher {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.simulation.reset();
        let mut qpos = self.simulation.qpos().to_vec();
        for value in &mut qpos {
            *value += self.rng.f64_between(-0.1, 0.1);
        }
        loop {
            let target_x = self.rng.f64_between(-0.2, 0.2);
            let target_y = self.rng.f64_between(-0.2, 0.2);
            if target_x.mul_add(target_x, target_y * target_y) < 0.04 {
                let length = qpos.len();
                *qpos
                    .get_mut(length - 2)
                    .expect("fixed example index is valid") = target_x;
                *qpos
                    .get_mut(length - 1)
                    .expect("fixed example index is valid") = target_y;
                break;
            }
        }
        let mut qvel = self.simulation.qvel().to_vec();
        for value in &mut qvel {
            *value += self.rng.f64_between(-0.005, 0.005);
        }
        let length = qvel.len();
        *qvel
            .get_mut(length - 2)
            .expect("fixed example index is valid") = 0.0;
        *qvel
            .get_mut(length - 1)
            .expect("fixed example index is valid") = 0.0;
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
        let action = [
            action.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0),
            action.get(1).copied().unwrap_or(0.0).clamp(-1.0, 1.0),
        ];
        self.simulation
            .step(&[f64::from(action[0]), f64::from(action[1])], 2)
            .expect("bundled model has two actuators");
        self.elapsed_steps = self.elapsed_steps.saturating_add(1);
        let fingertip = self
            .simulation
            .body_position("fingertip")
            .expect("bundled model defines the fingertip body");
        let target = self
            .simulation
            .body_position("target")
            .expect("bundled model defines the target body");
        let distance = fingertip
            .iter()
            .zip(target)
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt();
        let control_cost = action
            .iter()
            .map(|value| f64::from(value.powi(2)))
            .sum::<f64>();
        Step {
            observation: self.observation(),
            reward: -distance - control_cost,
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
    const ENV_NAME: &'static str = "mujoco-reacher";
    const GYMNASIUM_ID: &'static str = "Reacher-v5";
    const OBSERVATION_DIM: usize = 10;
    const ACTION_LOW: &'static [f32] = &[-1.0, -1.0];
    const ACTION_HIGH: &'static [f32] = &[1.0, 1.0];
    const DEFAULT_TRAIN_STEPS: usize = 300_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 50;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 100;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 3e-3;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 1e-3;
    const SOLVED_MEAN_REWARD: f64 = -3.75;
    const GIF_PATH: &'static str = "docs/images/mujoco-reacher.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 8_192;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 200;
    const BEHAVIOR_PRETRAINING_BATCH_SIZE: usize = 512;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
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
        let target = [
            observation.get(4).copied().unwrap_or(0.0),
            observation.get(5).copied().unwrap_or(0.0),
        ];
        let desired = inverse_kinematic_angles(target, [angle_0, angle_1]);
        vec![
            angle_difference(desired[0], angle_0) / std::f32::consts::PI,
            angle_difference(desired[1], angle_1) / std::f32::consts::PI,
            observation.get(6).copied().unwrap_or(0.0),
            observation.get(7).copied().unwrap_or(0.0),
            observation.first().copied().unwrap_or(1.0),
            observation.get(1).copied().unwrap_or(1.0),
            target[0] / 0.2,
            target[1] / 0.2,
            observation.get(8).copied().unwrap_or(0.0) / 0.21,
            observation.get(9).copied().unwrap_or(0.0) / 0.21,
        ]
    }

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        RecurrentPpoConfig {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![128, 64],
            gamma: 0.98,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.0001,
            initial_log_std: -2.5,
            ..RecurrentPpoConfig::default()
        }
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut samples = Vec::with_capacity(sample_count);
        let mut episode = 0_u64;
        while samples.len() < sample_count {
            let mut environment = Self::default();
            let mut observation = environment
                .reset(Some(seed.wrapping_add(episode)))
                .observation;
            loop {
                let action = Self::expert_action(&observation).to_vec();
                samples.push(RecurrentBehaviorSample {
                    observation: Self::encode_observation(&observation),
                    action: action.clone(),
                });
                if samples.len() >= sample_count {
                    break;
                }
                let transition = environment.step(action);
                observation = transition.observation;
                if transition.status != EpisodeStatus::Continuing {
                    break;
                }
            }
            episode = episode.saturating_add(1);
        }
        samples
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
    run_continuous_workflow::<Reacher>()
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
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    use super::{
        ContinuousPpoExample, Env, EpisodeStatus, Error, Path, Reacher, LINK_0_LENGTH,
        LINK_1_LENGTH,
    };

    /// Current policy and native environment.
    #[derive(Resource)]
    struct VisualState {
        /// Current checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Native environment.
        environment: Reacher,
        /// Exact current observation.
        observation: Vec<f32>,
        /// Actor memory.
        memory: bevy_gym::training::RecurrentMemory,
    }

    /// Interactive 50-Hz control clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// First arm link.
    #[derive(Component)]
    struct FirstLink;

    /// Second arm link.
    #[derive(Component)]
    struct SecondLink;

    /// Target marker.
    #[derive(Component)]
    struct Target;

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
            let mut environment = Reacher::default();
            let observation = environment.reset(Some(12_345)).observation;
            app.insert_resource(VisualState {
                policy,
                environment,
                observation,
                memory,
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.02, TimerMode::Repeating)))
            .insert_resource(ClearColor(Color::srgb(0.9, 0.9, 0.9)))
            .insert_resource(GlobalAmbientLight {
                color: Color::WHITE,
                brightness: 500.0,
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
                            title: "bevy-gym MuJoCo Reacher-v5".into(),
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
            10,
            10,
            1,
            &[-1.0, -1.0],
            &[1.0, 1.0],
            &Reacher::ppo_config(3e-3, 1e-3),
        )
        .map_err(Into::into)
    }

    /// Spawn a top-down arena matching Gymnasium's Reacher model.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 0.0, 0.75).looking_at(Vec3::ZERO, Vec3::Y),
        ));
        commands.spawn((
            DirectionalLight {
                illuminance: 5_000.0,
                ..default()
            },
            Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, -0.5, 0.0)),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.6, 0.6, 0.005))),
            MeshMaterial3d(materials.add(Color::srgb(0.9, 0.9, 0.9))),
            Transform::from_xyz(0.0, 0.0, -0.01),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.01, 0.08))),
            MeshMaterial3d(materials.add(Color::srgb(0.0, 0.4, 0.6))),
            FirstLink,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.01, 0.09))),
            MeshMaterial3d(materials.add(Color::srgb(0.0, 0.4, 0.6))),
            SecondLink,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.012))),
            MeshMaterial3d(materials.add(Color::srgb(0.9, 0.2, 0.2))),
            Target,
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

    /// Synchronize native qpos with the arm and target geometry.
    fn sync_scene(
        state: Res<'_, VisualState>,
        mut first: Single<'_, '_, &mut Transform, With<FirstLink>>,
        mut second: Single<'_, '_, &mut Transform, (With<SecondLink>, Without<FirstLink>)>,
        mut target: Single<
            '_,
            '_,
            &mut Transform,
            (With<Target>, Without<FirstLink>, Without<SecondLink>),
        >,
    ) {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let qpos = state.environment.simulation.qpos();
        let angle_0 = qpos.first().copied().unwrap_or(0.0) as f32;
        let angle_1 = angle_0 + qpos.get(1).copied().unwrap_or(0.0) as f32;
        let direction_0 = Vec2::new(angle_0.cos(), angle_0.sin());
        let direction_1 = Vec2::new(angle_1.cos(), angle_1.sin());
        let elbow = direction_0 * LINK_0_LENGTH;
        first.translation = Vec3::new(
            direction_0.x * LINK_0_LENGTH * 0.5,
            direction_0.y * LINK_0_LENGTH * 0.5,
            0.0,
        );
        first.rotation = Quat::from_rotation_z(angle_0 - std::f32::consts::FRAC_PI_2);
        second.translation = Vec3::new(
            (direction_1.x * LINK_1_LENGTH).mul_add(0.5, elbow.x),
            (direction_1.y * LINK_1_LENGTH).mul_add(0.5, elbow.y),
            0.0,
        );
        second.rotation = Quat::from_rotation_z(angle_1 - std::f32::consts::FRAC_PI_2);
        let target_position = state
            .environment
            .simulation
            .body_position("target")
            .expect("bundled model defines target");
        target.translation = Vec3::new(target_position[0] as f32, target_position[1] as f32, 0.0);
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

    /// Advance at the official 50-Hz control rate and synchronize.
    fn drive_recording_frame(
        world: &mut World,
        frame: RecordingFrame,
    ) -> Result<(), RecordingCallbackError> {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        if !frame.is_first_in_segment() {
            let current = frame.frame_in_segment();
            let previous = current.saturating_sub(1);
            let rate = usize::from(frame.frames_per_second());
            let steps = current * 50 / rate - previous * 50 / rate;
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
    fn reset_matches_the_target_and_observation_contract() {
        let mut environment = Reacher::default();
        let reset = environment.reset(Some(42));
        assert_eq!(reset.observation.len(), 10);
        let target_radius = reset.observation[4].hypot(reset.observation[5]);
        assert!(target_radius < 0.2);
    }

    #[test]
    fn expert_controller_exceeds_the_registry_threshold_on_held_out_resets() {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let mut mean_reward = 0.0;
        for seed in 9_001..9_101 {
            let mut environment = Reacher::default();
            let mut observation = environment.reset(Some(seed)).observation;
            let mut reward = 0.0;
            loop {
                let transition = environment.step(Reacher::expert_action(&observation).to_vec());
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
            mean_reward >= Reacher::SOLVED_MEAN_REWARD,
            "mean reward was {mean_reward}"
        );
    }

    #[test]
    fn private_encoding_preserves_both_joint_errors() {
        // Keep native state, Gymnasium semantics, and rendered playback on the same environment contract.
        let angle_0 = 0.2_f32;
        let angle_1 = -0.4_f32;
        let observation = vec![
            angle_0.cos(),
            angle_1.cos(),
            angle_0.sin(),
            angle_1.sin(),
            0.12,
            0.08,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let desired = inverse_kinematic_angles([0.12, 0.08], [angle_0, angle_1]);

        let encoded = Reacher::encode_observation(&observation);

        let expected = angle_difference(desired[1], angle_1) / std::f32::consts::PI;
        assert!((encoded[1] - expected).abs() < 1e-6);
    }
}
