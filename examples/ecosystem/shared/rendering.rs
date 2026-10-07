//! Interactive top-down policy playback for ecosystem curriculum stages.

use std::collections::BTreeMap;
use std::error::Error;
use std::ops::DerefMut as _;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};
use std::time::Instant;

use bevy::input::common_conditions::input_toggle_active;
use bevy::input::mouse::{AccumulatedMouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::sprite::Anchor;
use bevy::window::{PresentMode, WindowResolution};
use bevy_gym::training::{RecurrentMemory, RecurrentPpoPolicy};
use bevy_gym::EpisodeStatus;
use bevy_inspector_egui::bevy_egui::{
    input::EguiWantsInput, EguiContext, EguiPlugin, EguiPostUpdateSet, EguiPrimaryContextPass,
    PrimaryEguiContext,
};
use bevy_inspector_egui::egui;
use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_inspector_egui::DefaultInspectorConfigPlugin;

use super::demo::{DemoTrainingControl, DemoTrainingEvent};
use super::domain::{
    AgentEpisodeMetrics, AgentId, CurriculumStage, EcosystemState, ExperimentTuning,
    LocomotionAction, PerceptKind, PerceptionRayCount, SimulationConfig, Species, VisualAgent,
    VisualObject, VisualObjectKind, VisualWorldSnapshot, ACTION_HIGH, ACTION_LOW,
    GLOBAL_STATE_SIZE, LOCAL_OBSERVATION_SIZE, MAX_AGENTS, SURVIVAL_BATCH_ENVIRONMENTS,
};
use super::simulation::{
    Ecosystem, AGENT_SIZE, INTERACTION_HITBOX_FORWARD_OFFSET, INTERACTION_HITBOX_RADIUS,
    WELL_SENSOR_RADIUS,
};
use super::survival_batch::{SurvivalBatchPurpose, SurvivalBatchTrace};
use super::training::{
    checkpoint_experiment_tuning, ecosystem_algorithm, resolve_bunny_checkpoint,
    sibling_fox_checkpoint, validate_checkpoint_stage, validate_checkpoint_tuning_match,
    TrainerActivity, TrainingProgress,
};

/// Simulation units to rendered world units.
const WORLD_SCALE: f32 = 20.0;

/// Wall-clock seconds that a terminal frame remains visible.
const EPISODE_PAUSE_SECONDS: f32 = 1.0;

/// Wall-clock seconds that one signed reward delta remains visible.
const REWARD_POPUP_SECONDS: f32 = 1.0;

/// Minimum reward change that produces a floating delta label.
const REWARD_DELTA_EPSILON: f64 = 1.0e-6;

/// Return the centered row-major origin for one lane in the survival grid.
fn batch_grid_offset(lane: usize, spacing: f32) -> Vec2 {
    debug_assert!(lane < SURVIVAL_BATCH_ENVIRONMENTS);
    let column = lane % 3;
    let row = lane / 3;
    Vec2::new(column as f32 - 1.0, 1.0 - row as f32) * spacing
}

/// Return whether a completed batch should replay while the worker prepares its successor.
const fn should_loop_batch(_purpose: SurvivalBatchPurpose, has_pending: bool) -> bool {
    !has_pending
}

/// Mutually exclusive viewer playback lifecycle.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PlaybackState {
    /// Wall time advances the fixed-step accumulator.
    Running,

    /// The user paused playback.
    Paused,

    /// A terminal frame remains visible for the remaining wall time.
    EpisodePause {
        /// Wall-clock seconds before reset becomes ready.
        remaining_seconds: f32,
    },

    /// The terminal pause completed and the next episode can start.
    ResetReady,
}

impl PlaybackState {
    /// Toggle only user-controlled running and paused states.
    const fn toggle_user_pause(&mut self) {
        *self = match self {
            Self::Running => Self::Paused,
            Self::Paused => Self::Running,
            Self::EpisodePause { .. } | Self::ResetReady => *self,
        };
    }

    /// Consume wall time from an active terminal pause.
    fn advance_episode_pause(&mut self, delta_seconds: f32) {
        let Self::EpisodePause { remaining_seconds } = self else {
            return;
        };
        // Change state only after the complete terminal-frame interval elapses.
        *remaining_seconds = (*remaining_seconds - delta_seconds).max(0.0);
        if *remaining_seconds == 0.0 {
            *self = Self::ResetReady;
        }
    }
}

/// One signed per-step reward change shown above an agent.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RewardPopup {
    /// Signed change from the preceding step reward.
    value: f64,

    /// Wall-clock lifetime remaining before removal.
    remaining_seconds: f32,
}

impl RewardPopup {
    /// Start one fully opaque signed reward delta.
    const fn new(value: f64) -> Self {
        Self {
            value,
            remaining_seconds: REWARD_POPUP_SECONDS,
        }
    }

    /// Advance the popup fade in wall-clock time.
    fn advance(&mut self, delta_seconds: f32) {
        self.remaining_seconds = (self.remaining_seconds - delta_seconds).max(0.0);
    }

    /// Return the normalized opacity left in this fade.
    fn alpha(self) -> f32 {
        (self.remaining_seconds / REWARD_POPUP_SECONDS).clamp(0.0, 1.0)
    }

    /// Return whether the popup no longer needs rendering.
    fn is_expired(self) -> bool {
        self.remaining_seconds == 0.0
    }
}

/// One renderer-space collision outline and its interaction role.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DebugCollisionShape {
    /// A circular object that can receive an interaction.
    HurtboxCircle {
        /// Circle center in renderer world units.
        center: Vec2,

        /// Circle radius in renderer world units.
        radius: f32,
    },

    /// A living agent's solid oval body.
    HurtboxEllipse {
        /// Ellipse center in renderer world units.
        center: Vec2,

        /// Axis-aligned half extents before rotation.
        half_size: Vec2,

        /// Counterclockwise rotation in radians.
        rotation: f32,
    },

    /// A living agent's forward interaction sensor.
    HitboxCircle {
        /// Circle center in renderer world units.
        center: Vec2,

        /// Circle radius in renderer world units.
        radius: f32,

        /// Whether contacts can trigger an interaction this step.
        is_active: bool,
    },
}

/// Validated interactive playback arguments.
#[derive(Debug, Clone, PartialEq)]
struct WatchOptions {
    /// Bunny checkpoint or run directory.
    checkpoint: PathBuf,

    /// Optional explicit fox checkpoint.
    fox_checkpoint: Option<PathBuf>,

    /// Deterministic environment seed.
    seed: u64,

    /// Simulated seconds advanced per wall-clock second.
    speed: f32,

    /// Optional per-episode simulated-second horizon.
    episode_seconds: Option<u16>,
}

/// Visual trainer lifecycle displayed in the Inspector-egui panel.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TrainingStatus {
    /// The worker is collecting experience or optimizing.
    Running,

    /// The user paused the worker between complete updates.
    Paused,

    /// The worker will stop after its current complete update.
    Stopping,

    /// Every configured update completed.
    Complete,

    /// Training stopped with a diagnostic message.
    Failed(String),
}

/// Live training metrics and worker controls owned by the visual session.
struct DemoDashboard {
    /// Typed progress stream from the background trainer.
    receiver: mpsc::Receiver<DemoTrainingEvent>,

    /// Pause and shutdown flags shared with the background trainer.
    control: Arc<DemoTrainingControl>,

    /// Most recent policy and optimizer snapshot.
    progress: TrainingProgress,

    /// Whether this curriculum stage contains a fox policy.
    has_fox: bool,

    /// One bunny return point for every completed training episode.
    bunny_episode_curve: Vec<[f64; 2]>,

    /// One fox return point for every completed training episode.
    fox_episode_curve: Vec<[f64; 2]>,

    /// Bunny fixed-seed return curve in iteration order.
    bunny_eval_curve: Vec<[f64; 2]>,

    /// Fox fixed-seed return curve when predators exist.
    fox_eval_curve: Vec<[f64; 2]>,

    /// Bunny fixed-seed survival-time curve.
    bunny_survival_curve: Vec<[f64; 2]>,

    /// Fox fixed-seed survival-time curve when predators exist.
    fox_survival_curve: Vec<[f64; 2]>,

    /// Change in bunny fixed-seed reward per optimizer iteration.
    learning_velocity_curve: Vec<[f64; 2]>,

    /// Effective actor learning-rate curve.
    actor_learning_rate_curve: Vec<[f64; 2]>,

    /// Effective critic learning-rate curve.
    critic_learning_rate_curve: Vec<[f64; 2]>,

    /// Iterations whose rollout profile differs from the preceding point.
    tuning_markers: Vec<f64>,

    /// Current worker lifecycle.
    status: TrainingStatus,

    /// Exact work currently performed by the background trainer.
    activity: TrainerActivity,

    /// Wall-clock origin for the visible speed claim.
    started_at: Instant,
}

impl DemoDashboard {
    /// Seed the dashboard with the random-policy evaluation baseline.
    fn new(
        progress: TrainingProgress,
        receiver: mpsc::Receiver<DemoTrainingEvent>,
        control: Arc<DemoTrainingControl>,
        has_fox: bool,
    ) -> Self {
        // Start with empty curves because the first event may precede evaluation.
        let mut dashboard = Self {
            receiver,
            control,
            progress,
            has_fox,
            bunny_episode_curve: Vec::new(),
            fox_episode_curve: Vec::new(),
            bunny_eval_curve: Vec::new(),
            fox_eval_curve: Vec::new(),
            bunny_survival_curve: Vec::new(),
            fox_survival_curve: Vec::new(),
            learning_velocity_curve: Vec::new(),
            actor_learning_rate_curve: Vec::new(),
            critic_learning_rate_curve: Vec::new(),
            tuning_markers: Vec::new(),
            status: TrainingStatus::Running,
            activity: TrainerActivity::InitializationPreview,
            started_at: Instant::now(),
        };
        if dashboard.progress.has_evaluation {
            dashboard.record_evaluation_point();
        }
        dashboard.record_update_point();
        dashboard
    }

    /// Retain one point per completed optimizer iteration.
    fn record_evaluation_point(&mut self) {
        let episode = self.bunny_episode_curve.len() as f64;
        let iteration = self.progress.iteration as f64;
        // Keep evaluation means on the same episode axis as raw training returns.
        if let Some(previous) = self.bunny_eval_curve.last() {
            let previous_iteration = self
                .bunny_survival_curve
                .last()
                .map_or(0.0, |point| point[0]);
            let iteration_delta = (iteration - previous_iteration).max(1.0);
            let return_delta =
                (f64::from(self.progress.bunny_eval_return) - previous[1]) / iteration_delta;
            self.learning_velocity_curve.push([iteration, return_delta]);
        }
        self.bunny_eval_curve
            .push([episode, f64::from(self.progress.bunny_eval_return)]);
        if self.has_fox {
            self.fox_eval_curve
                .push([episode, f64::from(self.progress.fox_eval_return)]);
        }
        self.bunny_survival_curve
            .push([iteration, f64::from(self.progress.bunny_eval_lifetime)]);
        if self.has_fox {
            self.fox_survival_curve
                .push([iteration, f64::from(self.progress.fox_eval_lifetime)]);
        }
    }

    /// Retain optimizer and rollout values for every completed update.
    fn record_update_point(&mut self) {
        let iteration = self.progress.iteration as f64;
        // Append the complete rollout batch before recording optimizer scalars.
        append_episode_returns(
            &mut self.bunny_episode_curve,
            &self.progress.bunny_episode_returns,
        );
        append_episode_returns(
            &mut self.fox_episode_curve,
            &self.progress.fox_episode_returns,
        );
        self.actor_learning_rate_curve
            .push([iteration, self.progress.actor_learning_rate]);
        self.critic_learning_rate_curve
            .push([iteration, self.progress.critic_learning_rate]);
    }
}

/// Append raw batch returns on one continuous one-based episode axis.
fn append_episode_returns(curve: &mut Vec<[f64; 2]>, returns: &[f32]) {
    let first_episode = curve.last().map_or(1.0, |point| point[0] + 1.0);
    curve.extend(returns.iter().enumerate().map(|(offset, value)| {
        let offset = u32::try_from(offset).unwrap_or(u32::MAX);
        [first_episode + f64::from(offset), f64::from(*value)]
    }));
}

/// Interpolate position and the shortest wrapped heading arc.
fn interpolate_agent_pose(
    previous_position: Vec2,
    previous_heading: f32,
    current_position: Vec2,
    current_heading: f32,
    alpha: f32,
) -> (Vec2, f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    // Wrap the heading delta so interpolation never takes the long rotation arc.
    let heading_delta = (current_heading - previous_heading)
        .sin()
        .atan2((current_heading - previous_heading).cos());
    (
        previous_position.lerp(current_position, alpha),
        heading_delta.mul_add(alpha, previous_heading),
    )
}

/// Interpolate every matching agent while retaining the earlier exact world state.
fn interpolate_visual_snapshot(
    previous: &VisualWorldSnapshot,
    current: &VisualWorldSnapshot,
    alpha: f32,
) -> VisualWorldSnapshot {
    let mut snapshot = previous.clone();
    // Stable agent identities align poses without changing physics or observations.
    for agent in &mut snapshot.agents {
        let Some(current_agent) = current
            .agents
            .iter()
            .find(|candidate| candidate.id == agent.id)
        else {
            continue;
        };
        let (position, heading) = interpolate_agent_pose(
            Vec2::from_array(agent.position),
            agent.heading,
            Vec2::from_array(current_agent.position),
            current_agent.heading,
            alpha,
        );
        agent.position = position.to_array();
        agent.heading = heading;
    }
    snapshot
}

/// Return whether physiology changed on a reward-producing transition.
const fn reward_event_changed(previous: &VisualAgent, current: &VisualAgent) -> bool {
    // Ignore pose-only changes so ordinary movement cannot replace an event popup.
    previous.is_alive != current.is_alive
        || previous.satiation != current.satiation
        || previous.hydration != current.hydration
        || previous.hit_points != current.hit_points
        || previous.exposure != current.exposure
        || previous.in_shelter != current.in_shelter
}

impl Drop for DemoDashboard {
    fn drop(&mut self) {
        self.control.should_stop.store(true, Ordering::Relaxed);
    }
}

/// Mutable deterministic policy playback independent from Bevy rendering ECS.
pub(super) struct WatchSession {
    /// Curriculum lesson shown in the window.
    stage: CurriculumStage,

    /// Episode settings reused across resets.
    config: SimulationConfig,

    /// Bunny actor and recurrent memory model.
    bunny: RecurrentPpoPolicy,

    /// Fox actor when the stage includes predators.
    fox: Option<RecurrentPpoPolicy>,

    /// Authoritative headless Avian environment.
    ecosystem: Ecosystem,

    /// Actor-visible local observations at the current world instant.
    state: EcosystemState,

    /// Per-agent recurrent memories, cleared between episodes.
    memories: BTreeMap<AgentId, RecurrentMemory>,

    /// Root used to derive deterministic reset seeds.
    seed: u64,

    /// Number of completed episode resets.
    episode: u64,

    /// Simulated seconds advanced per wall-clock second.
    speed: f32,

    /// Fractional simulated time awaiting a fixed environment step.
    accumulator: f32,

    /// Current user and between-episode playback lifecycle.
    playback: PlaybackState,

    /// Prior exact physics snapshot used for render interpolation.
    previous_snapshot: VisualWorldSnapshot,

    /// Latest exact physics snapshot used for rendering and HUD values.
    current_snapshot: VisualWorldSnapshot,

    /// Whether exact actor perception sectors are drawn over the world.
    show_perception_rays: bool,

    /// Whether hitbox and hurtbox outlines are drawn over the world.
    show_collision_shapes: bool,

    /// Optional stable agent identity whose sectors are isolated.
    perception_ray_agent: Option<AgentId>,

    /// Slider profile requested for the next visible reset and PPO iteration.
    pending_tuning: ExperimentTuning,

    /// Checkpoint label displayed in the HUD.
    checkpoint_label: String,

    /// Live trainer data when the session runs in demo mode.
    dashboard: Option<DemoDashboard>,

    /// Survival reward accumulated in the current visible episode.
    episode_return: f64,

    /// Survival reward earned by the latest visible joint step.
    latest_reward: f64,

    /// Per-agent return accumulated in the current visible episode.
    agent_episode_returns: BTreeMap<AgentId, f64>,

    /// Previous per-step reward used to detect signed reward changes.
    previous_agent_rewards: BTreeMap<AgentId, f64>,

    /// Active signed reward changes fading above each agent.
    reward_popups: BTreeMap<AgentId, RewardPopup>,

    /// Actual survival training trajectories replayed in the 3x3 grid.
    training_batch: Option<SurvivalBatchPlayback>,

    /// Whether the camera must refit after the first batch replaces single-world playback.
    batch_camera_needs_fit: bool,
}

/// Playback cursor and newest pending trace for the survival training grid.
#[derive(Debug)]
struct SurvivalBatchPlayback {
    /// Trace currently advancing at the requested visual speed.
    current: SurvivalBatchTrace,

    /// Newest complete trace waiting behind the current terminal pause.
    pending: Option<SurvivalBatchTrace>,

    /// Exact current trace frame.
    frame: usize,
}

/// Marker for scene entities recreated from each read-only snapshot.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SceneVisual;

/// Marker for the movable top-down camera.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorldCamera;

/// Cached unit mesh and solid-color materials reused by every scene redraw.
#[derive(Resource)]
pub(super) struct SceneAssets {
    /// Unit-radius circle scaled into every circular or oval scene shape.
    circle: Handle<Mesh>,

    /// Food material.
    food: Handle<ColorMaterial>,

    /// Well material.
    well: Handle<ColorMaterial>,

    /// Tree material.
    tree: Handle<ColorMaterial>,

    /// Rock material.
    rock: Handle<ColorMaterial>,

    /// Thorn material.
    thorn: Handle<ColorMaterial>,

    /// Passive shelter-zone material.
    shelter: Handle<ColorMaterial>,

    /// Living bunny material.
    bunny: Handle<ColorMaterial>,

    /// Living fox material.
    fox: Handle<ColorMaterial>,

    /// Dead-agent material.
    dead: Handle<ColorMaterial>,

    /// Eye material shared by live circles and dead face marks.
    black: Handle<ColorMaterial>,

    /// Positive-reward pulse material.
    reward: Handle<ColorMaterial>,
}

/// Launch a top-down deterministic checkpoint viewer.
pub(super) fn run_watch(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    // Watch mode resolves one immutable checkpoint before creating a window.
    let options = parse_watch_options(arguments)?;
    let bunny_path = resolve_bunny_checkpoint(&options.checkpoint);
    validate_checkpoint_stage(&bunny_path, stage)?;
    let config = {
        let mut config = SimulationConfig::for_stage(stage)?;
        let tuning = checkpoint_experiment_tuning(&bunny_path)?
            .ok_or("bunny checkpoint lacks experiment tuning")?;
        config.apply_experiment_tuning(tuning)?;
        if let Some(episode_seconds) = options.episode_seconds {
            config.episode_seconds = episode_seconds;
        }
        config
    };
    let algorithm = ecosystem_algorithm();
    let bunny = load_policy(&bunny_path, &algorithm)?;
    let fox = if config.fox_count > 0 {
        let path = options
            .fox_checkpoint
            .unwrap_or_else(|| sibling_fox_checkpoint(&bunny_path));
        validate_checkpoint_stage(&path, stage)?;
        validate_checkpoint_tuning_match(&bunny_path, &path)?;
        Some(load_policy(&path, &algorithm)?)
    } else {
        None
    };
    let session = WatchSession::new(
        stage,
        config,
        bunny,
        fox,
        options.seed,
        options.speed,
        bunny_path.display().to_string(),
    )?;
    launch_viewer(session);
    Ok(())
}

/// Launch visual training from its random-policy evaluation baseline.
pub(super) fn run_training_demo(
    stage: CurriculumStage,
    initial: TrainingProgress,
    initial_batch: Option<SurvivalBatchTrace>,
    receiver: mpsc::Receiver<DemoTrainingEvent>,
    control: Arc<DemoTrainingControl>,
    playback_speed: f32,
) -> Result<(), Box<dyn Error>> {
    // Demo playback begins with the same profile as its first rollout batch.
    let mut config = SimulationConfig::for_stage(stage)?;
    config.apply_experiment_tuning(control.tuning())?;
    let label = initial.run_dir.display().to_string();
    let bunny = initial.bunny.clone();
    let fox = initial.fox.clone();
    let has_fox = config.fox_count > 0;
    let mut session = WatchSession::new(stage, config, bunny, fox, 101, playback_speed, label)?;
    if let Some(batch) = initial_batch {
        session.queue_training_batch(batch);
    }
    session.dashboard = Some(DemoDashboard::new(initial, receiver, control, has_fox));
    launch_viewer(session);
    Ok(())
}

/// Configure the shared Bevy world, Inspector-egui HUD, and camera controls.
fn launch_viewer(session: WatchSession) {
    let stage = session.stage;
    // One app setup keeps watch, demo, and video presentation behavior aligned.
    let mut app = App::new();
    app.insert_non_send_resource(session)
        .insert_resource(ClearColor(Color::srgb_u8(15, 23, 18)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: format!("bevy-gym ecosystem: {}", stage.as_key()),
                resolution: WindowResolution::new(1280, 720),
                present_mode: PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_plugins(DefaultInspectorConfigPlugin)
        .add_plugins(
            WorldInspectorPlugin::default().run_if(input_toggle_active(false, KeyCode::F1)),
        )
        .add_systems(Startup, setup_viewer)
        .add_systems(
            Update,
            (
                toggle_playback,
                apply_training_progress,
                advance_policy,
                redraw_scene,
                draw_perception_rays,
                draw_collision_shapes,
            )
                .chain(),
        )
        .add_systems(
            PostUpdate,
            (fit_training_batch_camera, camera_controls)
                .chain()
                .after(EguiPostUpdateSet::ProcessOutput),
        )
        .add_systems(EguiPrimaryContextPass, ecosystem_hud_ui);
    app.run();
}

/// Load one policy with the stable ecosystem tensor and action contracts.
pub(super) fn load_policy(
    path: &std::path::Path,
    algorithm: &bevy_gym::training::RecurrentPpoConfig,
) -> Result<RecurrentPpoPolicy, Box<dyn Error>> {
    // Keep load dimensions identical to every ecosystem training stage.
    Ok(RecurrentPpoPolicy::load(
        path,
        LOCAL_OBSERVATION_SIZE,
        GLOBAL_STATE_SIZE,
        MAX_AGENTS,
        &ACTION_LOW,
        &ACTION_HIGH,
        algorithm,
    )?)
}

/// Parse and validate the compact watch command surface.
fn parse_watch_options(
    mut arguments: impl Iterator<Item = String>,
) -> Result<WatchOptions, Box<dyn Error>> {
    // Watch flags use the same one-value grammar as demo and training modes.
    let mut checkpoint = None;
    let mut fox_checkpoint = None;
    let mut environment_seed = 101;
    let mut playback_speed: f32 = 1.0;
    let mut episode_seconds = None;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--fox-checkpoint" => fox_checkpoint = Some(PathBuf::from(value)),
            "--seed" => environment_seed = value.parse()?,
            "--speed" => playback_speed = value.parse()?,
            "--episode-seconds" => episode_seconds = Some(value.parse()?),
            _ => return Err(format!("unknown watch option {flag:?}").into()),
        }
    }
    if !playback_speed.is_finite()
        || playback_speed <= 0.0
        || episode_seconds.is_some_and(|seconds| !(5..=300).contains(&seconds))
    {
        return Err(
            "--speed must be finite and positive; --episode-seconds must be in 5..=300".into(),
        );
    }
    Ok(WatchOptions {
        checkpoint: checkpoint.ok_or("watch requires --checkpoint <run-dir-or-mpk>")?,
        fox_checkpoint,
        seed: environment_seed,
        speed: playback_speed,
        episode_seconds,
    })
}

/// Spawn the camera used by both checkpoint playback and visual training.
pub(super) fn setup_viewer(
    mut commands: Commands<'_, '_>,
    session: NonSend<'_, WatchSession>,
    mut meshes: ResMut<'_, Assets<Mesh>>,
    mut materials: ResMut<'_, Assets<ColorMaterial>>,
) {
    // Fit the complete fixed map before the user applies pan or zoom controls.
    let snapshot = session.current_snapshot.clone();
    let initial_scale = if session.stage == CurriculumStage::Survival && session.dashboard.is_some()
    {
        batch_camera_scale(snapshot.map_half_extent)
    } else {
        (snapshot.map_half_extent * WORLD_SCALE * 2.0 / 650.0).max(0.7)
    };
    commands.spawn((
        Camera2d,
        Projection::from(OrthographicProjection {
            scale: initial_scale,
            ..OrthographicProjection::default_2d()
        }),
        IsDefaultUiCamera,
        WorldCamera,
        Name::new("Ecosystem Camera"),
    ));
    commands.insert_resource(SceneAssets {
        circle: meshes.add(Circle::new(1.0)),
        food: materials.add(Color::srgb_u8(244, 211, 94)),
        well: materials.add(Color::srgb_u8(54, 162, 235)),
        tree: materials.add(Color::srgb_u8(25, 94, 55)),
        rock: materials.add(Color::srgb_u8(123, 130, 137)),
        thorn: materials.add(Color::srgb_u8(191, 64, 128)),
        shelter: materials.add(Color::srgba_u8(117, 92, 184, 150)),
        bunny: materials.add(Color::srgb_u8(238, 238, 232)),
        fox: materials.add(Color::srgb_u8(229, 111, 47)),
        dead: materials.add(Color::srgb_u8(70, 70, 70)),
        black: materials.add(Color::BLACK),
        reward: materials.add(Color::srgba(0.3, 1.0, 0.4, 0.32)),
    });
}

/// Pause, resume, or restart the deterministic evaluation episode.
fn toggle_playback(keys: Res<'_, ButtonInput<KeyCode>>, mut session: NonSendMut<'_, WatchSession>) {
    // Keyboard shortcuts mirror the visible Inspector-egui controls.
    if keys.just_pressed(KeyCode::Space) {
        session.playback.toggle_user_pause();
    }
    if keys.just_pressed(KeyCode::KeyR) {
        let reset_failed = if session.training_batch.is_some() {
            session.restart_training_batch();
            false
        } else {
            session.reset_episode().is_err()
        };
        if reset_failed {
            session.playback = PlaybackState::Paused;
        }
    }
}

/// Advance the policy in exact fixed simulation increments.
fn advance_policy(time: Res<'_, Time>, mut session: NonSendMut<'_, WatchSession>) {
    let delta_seconds = time.delta_secs();
    session.advance_reward_popups(delta_seconds);
    session.playback.advance_episode_pause(delta_seconds);
    if session.advance_training_batch(delta_seconds) {
        return;
    }
    if session.playback == PlaybackState::ResetReady {
        if session.reset_episode().is_err() {
            session.playback = PlaybackState::Paused;
        }
        return;
    }
    // Accumulation preserves fixed simulation time at every playback speed.
    if session.playback != PlaybackState::Running {
        return;
    }
    session.accumulator = time
        .delta_secs()
        .mul_add(session.speed, session.accumulator);
    let time_step = session.config.time_step;
    loop {
        if session.accumulator < time_step {
            break;
        }
        session.accumulator -= time_step;
        match session.step_once() {
            Ok(did_reset) if did_reset => break,
            Ok(_) => {}
            Err(_) => {
                session.playback = PlaybackState::Paused;
                break;
            }
        }
    }
}

/// Apply completed optimizer updates without blocking the render schedule.
fn apply_training_progress(mut session: NonSendMut<'_, WatchSession>) {
    // Temporarily detach the dashboard so policy replacement can borrow session.
    let Some(mut dashboard) = session.dashboard.take() else {
        return;
    };
    let events: Vec<_> = dashboard.receiver.try_iter().collect();
    for event in events {
        match event {
            DemoTrainingEvent::Activity(activity) => {
                dashboard.activity = activity;
            }
            DemoTrainingEvent::BatchTrace(trace) => {
                session.queue_training_batch(*trace);
            }
            DemoTrainingEvent::Progress(progress) => {
                let progress = *progress;
                let label = format!(
                    "{} iteration {}/{}",
                    progress.run_dir.display(),
                    progress.iteration,
                    progress.total_iterations
                );
                if let Err(error) =
                    session.replace_policies(progress.bunny.clone(), progress.fox.clone(), label)
                {
                    dashboard.status = TrainingStatus::Failed(error.to_string());
                    break;
                }
                if progress.experiment_tuning != dashboard.progress.experiment_tuning {
                    let marker = dashboard.bunny_episode_curve.len() as f64;
                    if dashboard.tuning_markers.last().copied() != Some(marker) {
                        dashboard.tuning_markers.push(marker);
                    }
                    dashboard.bunny_eval_curve.clear();
                    dashboard.fox_eval_curve.clear();
                    dashboard.learning_velocity_curve.clear();
                    dashboard.bunny_survival_curve.clear();
                    dashboard.fox_survival_curve.clear();
                }
                dashboard.progress = progress;
                dashboard.record_update_point();
                if dashboard.progress.has_evaluation {
                    dashboard.record_evaluation_point();
                }
                dashboard.status = TrainingStatus::Running;
            }
            DemoTrainingEvent::Finished => {
                dashboard.status = TrainingStatus::Complete;
            }
            DemoTrainingEvent::Failed(message) => {
                dashboard.status = TrainingStatus::Failed(message);
            }
        }
    }
    session.dashboard = Some(dashboard);
}

impl WatchSession {
    /// Construct a clean deterministic playback session for one policy set.
    pub(super) fn new(
        stage: CurriculumStage,
        config: SimulationConfig,
        bunny: RecurrentPpoPolicy,
        fox: Option<RecurrentPpoPolicy>,
        environment_seed: u64,
        playback_speed: f32,
        checkpoint_label: String,
    ) -> Result<Self, Box<dyn Error>> {
        // The renderer owns a separate deterministic world from training lanes.
        let pending_tuning = config.experiment_tuning();
        let mut ecosystem = Ecosystem::new(config.clone(), environment_seed)?;
        let initial_state = ecosystem.state();
        let initial_snapshot = ecosystem.visual_snapshot();
        Ok(Self {
            stage,
            config,
            bunny,
            fox,
            ecosystem,
            state: initial_state,
            memories: BTreeMap::new(),
            seed: environment_seed,
            episode: 0,
            speed: playback_speed,
            accumulator: 0.0,
            playback: PlaybackState::Running,
            previous_snapshot: initial_snapshot.clone(),
            current_snapshot: initial_snapshot,
            show_perception_rays: true,
            show_collision_shapes: false,
            perception_ray_agent: Some(AgentId(0)),
            pending_tuning,
            checkpoint_label,
            dashboard: None,
            episode_return: 0.0,
            latest_reward: 0.0,
            agent_episode_returns: BTreeMap::new(),
            previous_agent_rewards: BTreeMap::new(),
            reward_popups: BTreeMap::new(),
            training_batch: None,
            batch_camera_needs_fit: false,
        })
    }

    /// Queue one completed optimizer batch for real-time grid replay.
    fn queue_training_batch(&mut self, trace: SurvivalBatchTrace) {
        let starts_immediately =
            self.training_batch.is_none() || self.playback == PlaybackState::ResetReady;
        if starts_immediately {
            self.training_batch = Some(SurvivalBatchPlayback {
                current: trace,
                pending: None,
                frame: 0,
            });
            self.accumulator = 0.0;
            self.playback = PlaybackState::Running;
            self.batch_camera_needs_fit = true;
            return;
        }
        if let Some(batch) = self.training_batch.as_mut() {
            // Keep only the newest waiting trace so rendering cannot slow training.
            batch.pending = Some(trace);
        }
    }

    /// Advance actual training trajectories and report whether batch mode owns playback.
    #[expect(
        clippy::while_float,
        reason = "the fixed-step accumulator must consume every complete simulation step"
    )]
    fn advance_training_batch(&mut self, delta_seconds: f32) -> bool {
        let Some(batch) = self.training_batch.as_mut() else {
            return false;
        };
        if self.playback == PlaybackState::ResetReady {
            let has_pending = batch.pending.is_some();
            if let Some(next) = batch.pending.take() {
                batch.current = next;
                batch.frame = 0;
                self.accumulator = 0.0;
                self.playback = PlaybackState::Running;
            } else if should_loop_batch(batch.current.purpose(), has_pending) {
                batch.frame = 0;
                self.accumulator = 0.0;
                self.playback = PlaybackState::Running;
            }
            return true;
        }
        if self.playback != PlaybackState::Running {
            return true;
        }

        // Replay stays smooth and bounded while the worker trains without waiting.
        self.accumulator = delta_seconds.mul_add(self.speed, self.accumulator);
        let time_step = batch.current.time_step();
        while self.accumulator >= time_step {
            self.accumulator -= time_step;
            let next_frame = batch.frame.saturating_add(1);
            if next_frame >= batch.current.frame_count() {
                self.accumulator = 0.0;
                self.playback = PlaybackState::EpisodePause {
                    remaining_seconds: EPISODE_PAUSE_SECONDS,
                };
                break;
            }
            batch.frame = next_frame;
        }
        true
    }

    /// Replay the current nine-lane trace from its exact initial states.
    const fn restart_training_batch(&mut self) {
        if let Some(batch) = self.training_batch.as_mut() {
            batch.frame = 0;
            self.accumulator = 0.0;
            self.playback = PlaybackState::Running;
        }
    }

    /// Run one deterministic mean-action joint step and retain recurrent state.
    pub(super) fn step_once(&mut self) -> Result<bool, Box<dyn Error>> {
        let mut actions = Vec::with_capacity(self.state.agents.len());
        let mut next_memories = BTreeMap::new();
        for (id, species, observation) in &self.state.agents {
            let policy = match species {
                Species::Bunny => &self.bunny,
                Species::Fox => self.fox.as_ref().ok_or("fox policy is missing")?,
            };
            let memory = self
                .memories
                .entry(*id)
                .or_insert_with(|| policy.initial_memory());
            let output = policy.mean_action(observation, memory)?;
            let [forward, turn, gaze, attack] = output.action.as_slice() else {
                return Err("ecosystem policy must emit four action axes".into());
            };
            actions.push((*id, LocomotionAction::new(*forward, *turn, *gaze, *attack)?));
            next_memories.insert(*id, output.next_memory);
        }

        // Replace memory only for continuing trajectories; termination cannot
        // leak recurrent state into the next episode or another identity.
        let previous_snapshot = self.current_snapshot.clone();
        let result = self.ecosystem.step(&actions)?;
        let next_snapshot = self.ecosystem.visual_snapshot();
        self.latest_reward = result.agents.iter().map(|agent| agent.reward).sum();
        self.episode_return += self.latest_reward;
        for agent in &result.agents {
            *self.agent_episode_returns.entry(agent.id).or_default() += agent.reward;
            let previous_reward = self
                .previous_agent_rewards
                .insert(agent.id, agent.reward)
                .unwrap_or_default();
            let reward_delta = agent.reward - previous_reward;
            let physiology_changed = previous_snapshot
                .agents
                .iter()
                .find(|candidate| candidate.id == agent.id)
                .zip(
                    next_snapshot
                        .agents
                        .iter()
                        .find(|candidate| candidate.id == agent.id),
                )
                .is_some_and(|(previous, current)| reward_event_changed(previous, current));
            if physiology_changed && reward_delta.abs() > REWARD_DELTA_EPSILON {
                self.reward_popups
                    .insert(agent.id, RewardPopup::new(reward_delta));
            }
        }
        self.memories.clear();
        let mut next_agents = Vec::new();
        for agent in result.agents {
            if agent.status == EpisodeStatus::Continuing {
                if let Some(memory) = next_memories.remove(&agent.id) {
                    self.memories.insert(agent.id, memory);
                }
                next_agents.push((agent.id, agent.species, agent.observation));
            }
        }
        self.previous_snapshot = previous_snapshot;
        self.current_snapshot = next_snapshot;
        if result.is_done {
            self.accumulator = self.config.time_step;
            self.playback = PlaybackState::EpisodePause {
                remaining_seconds: EPISODE_PAUSE_SECONDS,
            };
            Ok(true)
        } else {
            self.state.agents = next_agents;
            self.state.global_state = result.global_state;
            Ok(false)
        }
    }

    /// Advance and remove signed reward deltas without changing simulation time.
    fn advance_reward_popups(&mut self, delta_seconds: f32) {
        for popup in self.reward_popups.values_mut() {
            popup.advance(delta_seconds);
        }
        self.reward_popups.retain(|_, popup| !popup.is_expired());
    }

    /// Interpolate agent poses between the two latest exact physics snapshots.
    fn presentation_snapshot(&self) -> VisualWorldSnapshot {
        let mut snapshot = self.current_snapshot.clone();
        // Terminal frames show exact state; running frames fill fixed-step gaps.
        let alpha = if matches!(self.playback, PlaybackState::EpisodePause { .. }) {
            1.0
        } else {
            (self.accumulator / self.config.time_step).clamp(0.0, 1.0)
        };
        for agent in &mut snapshot.agents {
            let Some(previous) = self
                .previous_snapshot
                .agents
                .iter()
                .find(|candidate| candidate.id == agent.id)
            else {
                continue;
            };
            let (position, heading) = interpolate_agent_pose(
                Vec2::from_array(previous.position),
                previous.heading,
                Vec2::from_array(agent.position),
                agent.heading,
                alpha,
            );
            agent.position = position.to_array();
            agent.heading = heading;
        }
        snapshot
    }

    /// Create the next deterministic map and clear all actor memories.
    fn reset_episode(&mut self) -> Result<(), Box<dyn Error>> {
        // Each reset advances one deterministic seed without reusing LSTM state.
        self.episode = self.episode.saturating_add(1);
        let seed = self.seed.wrapping_add(self.episode);
        self.ecosystem = Ecosystem::new(self.config.clone(), seed)?;
        self.state = self.ecosystem.state();
        let snapshot = self.ecosystem.visual_snapshot();
        self.previous_snapshot = snapshot.clone();
        self.current_snapshot = snapshot;
        self.memories.clear();
        self.accumulator = 0.0;
        self.playback = PlaybackState::Running;
        self.episode_return = 0.0;
        self.latest_reward = 0.0;
        self.agent_episode_returns.clear();
        self.previous_agent_rewards.clear();
        self.reward_popups.clear();
        Ok(())
    }

    /// Apply pending demo settings and restart the current visible seed.
    fn apply_pending_tuning(&mut self) -> Result<(), Box<dyn Error>> {
        // Build the replacement world before committing the profile so a
        // failed procedural reset preserves the current playable episode.
        let mut config = self.config.clone();
        config.apply_experiment_tuning(self.pending_tuning)?;
        let seed = self.seed.wrapping_add(self.episode);
        let mut ecosystem = Ecosystem::new(config.clone(), seed)?;
        let state = ecosystem.state();
        self.config = config;
        self.ecosystem = ecosystem;
        self.state = state;
        let snapshot = self.ecosystem.visual_snapshot();
        self.previous_snapshot = snapshot.clone();
        self.current_snapshot = snapshot;
        self.memories.clear();
        self.accumulator = 0.0;
        self.playback = PlaybackState::Running;
        self.episode_return = 0.0;
        self.latest_reward = 0.0;
        self.agent_episode_returns.clear();
        self.previous_agent_rewards.clear();
        self.reward_popups.clear();
        Ok(())
    }

    /// Replace both species policies and restart the same evaluation map.
    pub(super) fn replace_policies(
        &mut self,
        bunny: RecurrentPpoPolicy,
        fox: Option<RecurrentPpoPolicy>,
        checkpoint_label: String,
    ) -> Result<(), Box<dyn Error>> {
        // Policy changes restart one seed so behavior changes remain comparable.
        self.bunny = bunny;
        self.fox = fox;
        self.checkpoint_label = checkpoint_label;
        if self.training_batch.is_some() {
            // The grid continues replaying the batch that produced this policy.
            return Ok(());
        }
        self.episode = 0;
        self.ecosystem = Ecosystem::new(self.config.clone(), self.seed)?;
        self.state = self.ecosystem.state();
        let snapshot = self.ecosystem.visual_snapshot();
        self.previous_snapshot = snapshot.clone();
        self.current_snapshot = snapshot;
        self.memories.clear();
        self.accumulator = 0.0;
        self.playback = PlaybackState::Running;
        self.episode_return = 0.0;
        self.latest_reward = 0.0;
        self.agent_episode_returns.clear();
        self.previous_agent_rewards.clear();
        self.reward_popups.clear();
        Ok(())
    }
}

/// Replace transient scene primitives from the authoritative world snapshot.
pub(super) fn redraw_scene(
    mut commands: Commands<'_, '_>,
    session: NonSend<'_, WatchSession>,
    assets: Res<'_, SceneAssets>,
    visuals: Query<'_, '_, Entity, With<SceneVisual>>,
) {
    // Scene primitives are presentation projections, never simulation entities.
    for entity in &visuals {
        commands.entity(entity).despawn();
    }
    if let Some(batch) = session.training_batch.as_ref() {
        let frame = batch.frame;
        let alpha = (session.accumulator / batch.current.time_step()).clamp(0.0, 1.0);
        let spacing = training_batch_grid_spacing(batch);
        for lane in batch.current.lanes() {
            let current = lane.snapshot(frame);
            let next = lane.snapshot(frame.saturating_add(1));
            let snapshot = interpolate_visual_snapshot(current, next, alpha);
            let offset = batch_grid_offset(lane.lane(), spacing);
            let episode_return = lane.episode_return(frame);
            let previous_return = lane.episode_return(frame.saturating_sub(1));
            let reward_delta = episode_return - previous_return;
            let episode_returns = snapshot
                .agents
                .iter()
                .map(|agent| (agent.id, episode_return))
                .collect::<BTreeMap<_, _>>();
            let reward_popups = if reward_delta.abs() > REWARD_DELTA_EPSILON {
                snapshot
                    .agents
                    .iter()
                    .map(|agent| (agent.id, RewardPopup::new(reward_delta)))
                    .collect::<BTreeMap<_, _>>()
            } else {
                BTreeMap::new()
            };
            spawn_snapshot(
                &mut commands,
                &assets,
                &snapshot,
                offset,
                reward_delta > 0.0,
                &episode_returns,
                &reward_popups,
            );
            spawn_batch_lane_label(
                &mut commands,
                lane.lane(),
                lane.stage(),
                batch.current.purpose(),
                snapshot.map_half_extent,
                offset,
            );
        }
        return;
    }
    let snapshot = session.presentation_snapshot();
    spawn_snapshot(
        &mut commands,
        &assets,
        &snapshot,
        Vec2::ZERO,
        session.latest_reward > 0.0,
        &session.agent_episode_returns,
        &session.reward_popups,
    );
}

/// Draw the exact post-physics semantic sectors encoded for every living actor.
pub(super) fn draw_perception_rays(mut gizmos: Gizmos<'_, '_>, session: NonSend<'_, WatchSession>) {
    if !session.show_perception_rays {
        return;
    }
    if let Some(batch) = session.training_batch.as_ref() {
        let spacing = training_batch_grid_spacing(batch);
        for lane in batch.current.lanes() {
            let snapshot = lane.snapshot(batch.frame);
            let offset = batch_grid_offset(lane.lane(), spacing);
            draw_snapshot_rays(&mut gizmos, &session, snapshot, offset);
        }
        return;
    }
    draw_snapshot_rays(&mut gizmos, &session, &session.current_snapshot, Vec2::ZERO);
}

/// Draw hitbox and hurtbox outlines for every visible physics role.
pub(super) fn draw_collision_shapes(
    mut gizmos: Gizmos<'_, '_>,
    session: NonSend<'_, WatchSession>,
) {
    if !session.show_collision_shapes {
        return;
    }
    // Reuse each lane's exact physics frame so the overlay follows the same
    // replay state as its perception rays.
    if let Some(batch) = session.training_batch.as_ref() {
        let spacing = training_batch_grid_spacing(batch);
        for lane in batch.current.lanes() {
            let snapshot = lane.snapshot(batch.frame);
            let offset = batch_grid_offset(lane.lane(), spacing);
            draw_snapshot_collision_shapes(&mut gizmos, snapshot, offset);
        }
        return;
    }
    draw_snapshot_collision_shapes(&mut gizmos, &session.current_snapshot, Vec2::ZERO);
}

/// Draw one snapshot's collision roles at its renderer-space origin.
fn draw_snapshot_collision_shapes(
    gizmos: &mut Gizmos<'_, '_>,
    snapshot: &VisualWorldSnapshot,
    offset: Vec2,
) {
    // Object sensors and agent colliders have different geometry sources but
    // share the same renderer-space projection.
    for object in &snapshot.objects {
        if let Some(shape) = object_collision_shape(object, offset) {
            draw_collision_shape(gizmos, shape);
        }
    }
    for agent in &snapshot.agents {
        if let Some(shapes) = agent_collision_shapes(agent, offset) {
            for shape in shapes {
                draw_collision_shape(gizmos, shape);
            }
        }
    }
}

/// Project one living agent's oval hurtbox and forward mouth hitbox.
fn agent_collision_shapes(agent: &VisualAgent, offset: Vec2) -> Option<[DebugCollisionShape; 2]> {
    if !agent.is_alive {
        return None;
    }
    // Rotate the child hitbox offset with the body exactly as Avian's child
    // transform does in the authoritative simulation.
    let center = Vec2::from_array(agent.position) * WORLD_SCALE + offset;
    let forward = Vec2::from_angle(agent.heading);
    Some([
        DebugCollisionShape::HurtboxEllipse {
            center,
            half_size: AGENT_SIZE * WORLD_SCALE * 0.5,
            rotation: agent.heading,
        },
        DebugCollisionShape::HitboxCircle {
            center: center + forward * (INTERACTION_HITBOX_FORWARD_OFFSET * WORLD_SCALE),
            radius: INTERACTION_HITBOX_RADIUS * WORLD_SCALE,
            is_active: agent.attack_active,
        },
    ])
}

/// Project one object's interaction-receiving sensor when it has one.
fn object_collision_shape(object: &VisualObject, offset: Vec2) -> Option<DebugCollisionShape> {
    // The well's drinking hurtbox is larger than its visible shallow-water
    // circle; other object hurtboxes use their displayed collision radius.
    let radius = match object.kind {
        VisualObjectKind::Food | VisualObjectKind::Thorn => object.radius,
        VisualObjectKind::Well => WELL_SENSOR_RADIUS,
        VisualObjectKind::Tree | VisualObjectKind::Rock | VisualObjectKind::Shelter => return None,
    };
    Some(DebugCollisionShape::HurtboxCircle {
        center: Vec2::from_array(object.position) * WORLD_SCALE + offset,
        radius: radius * WORLD_SCALE,
    })
}

/// Render one collision outline with stable role and activity colors.
fn draw_collision_shape(gizmos: &mut Gizmos<'_, '_>, shape: DebugCollisionShape) {
    // Preserve geometry type so rotated oval bodies are not approximated by
    // their spawn-clearance circles.
    match shape {
        DebugCollisionShape::HurtboxCircle { center, radius } => {
            gizmos.circle_2d(center, radius, hurtbox_color());
        }
        DebugCollisionShape::HurtboxEllipse {
            center,
            half_size,
            rotation,
        } => {
            gizmos.ellipse_2d(
                Isometry2d::new(center, Rot2::radians(rotation)),
                half_size,
                hurtbox_color(),
            );
        }
        DebugCollisionShape::HitboxCircle {
            center,
            radius,
            is_active,
        } => {
            gizmos.circle_2d(center, radius, hitbox_color(is_active));
        }
    }
}

/// Return the shared cyan hurtbox outline color.
const fn hurtbox_color() -> Color {
    Color::srgba(0.2, 0.85, 1.0, 0.95)
}

/// Return orange for an active hitbox and a subdued outline otherwise.
const fn hitbox_color(is_active: bool) -> Color {
    if is_active {
        Color::srgba(1.0, 0.25, 0.1, 1.0)
    } else {
        Color::srgba(1.0, 0.68, 0.2, 0.3)
    }
}

/// Use one cell spacing derived from the largest retained curriculum map.
fn training_batch_grid_spacing(batch: &SurvivalBatchPlayback) -> f32 {
    let maximum_half_extent = batch
        .current
        .lanes()
        .iter()
        .map(|lane| lane.snapshot(batch.frame).map_half_extent)
        .fold(0.0_f32, f32::max);
    (maximum_half_extent * WORLD_SCALE).mul_add(2.0, 28.0)
}

/// Draw one snapshot's actor-visible rays at its grid origin.
fn draw_snapshot_rays(
    gizmos: &mut Gizmos<'_, '_>,
    session: &WatchSession,
    snapshot: &VisualWorldSnapshot,
    offset: Vec2,
) {
    // Hits stop at their sampled distance. Misses extend to the full sight range.
    for ray in snapshot.rays.iter().copied() {
        if session
            .perception_ray_agent
            .is_some_and(|agent| agent != ray.agent)
        {
            continue;
        }
        let start = Vec2::from_array(ray.start) * WORLD_SCALE + offset;
        let end = Vec2::from_array(ray.end) * WORLD_SCALE + offset;
        gizmos.line_2d(start, end, perception_ray_color(ray.kind));
    }
}

/// Map actor semantic channels to stable diagnostic colors.
const fn perception_ray_color(kind: Option<PerceptKind>) -> Color {
    // Keep missed sectors subdued while semantic hits remain recognizable over
    // every stage's common background and programmer art.
    match kind {
        None => Color::srgba(0.7, 0.75, 0.7, 0.12),
        Some(PerceptKind::Food) => Color::srgba(0.96, 0.83, 0.37, 0.9),
        Some(PerceptKind::Well) => Color::srgba(0.21, 0.64, 0.92, 0.9),
        Some(PerceptKind::Bunny) => Color::srgba(0.93, 0.93, 0.9, 0.9),
        Some(PerceptKind::Fox) => Color::srgba(0.9, 0.44, 0.18, 0.9),
        Some(PerceptKind::SolidObstacle) => Color::srgba(0.56, 0.61, 0.58, 0.9),
        Some(PerceptKind::Thorn) => Color::srgba(0.82, 0.27, 0.62, 0.9),
        Some(PerceptKind::Shelter) => Color::srgba(0.55, 0.42, 0.85, 0.9),
        Some(PerceptKind::Boundary) => Color::srgba(0.72, 0.52, 0.29, 0.9),
        Some(PerceptKind::Gorge) => Color::srgba(0.3, 0.34, 0.42, 0.95),
        Some(PerceptKind::Bridge) => Color::srgba(0.78, 0.57, 0.3, 0.95),
    }
}

/// One primitive used to draw a live or dead agent face.
#[derive(Debug, Clone, Copy, PartialEq)]
enum EyeGlyph {
    /// Circular living eye at one body-local offset.
    Circle {
        /// Body-local eye center.
        offset: Vec2,
    },

    /// One stroke of a dead X at one body-local offset and rotation.
    Stroke {
        /// Body-local stroke center.
        offset: Vec2,
        /// Body-local stroke angle in radians.
        rotation: f32,
    },
}

/// Return the complete body-local face geometry for one life state.
fn agent_eye_geometry(is_alive: bool) -> Vec<EyeGlyph> {
    const FORWARD: f32 = 7.0;
    const LATERAL: f32 = 4.5;
    let offsets = [Vec2::new(FORWARD, LATERAL), Vec2::new(FORWARD, -LATERAL)];
    if is_alive {
        return offsets
            .into_iter()
            .map(|offset| EyeGlyph::Circle { offset })
            .collect();
    }
    offsets
        .into_iter()
        .flat_map(|offset| {
            [
                EyeGlyph::Stroke {
                    offset,
                    rotation: std::f32::consts::FRAC_PI_4,
                },
                EyeGlyph::Stroke {
                    offset,
                    rotation: -std::f32::consts::FRAC_PI_4,
                },
            ]
        })
        .collect()
}

/// Return one fill flag for each configured status point.
fn status_segments(current: u8, maximum: u8) -> Vec<bool> {
    (0..maximum).map(|index| index < current).collect()
}

/// Spawn a fixed-width segmented status bar above one agent.
fn spawn_status_bar(
    commands: &mut Commands<'_, '_>,
    agent: &VisualAgent,
    current: u8,
    maximum: u8,
    vertical_offset: f32,
    filled_color: Color,
    label: &'static str,
) {
    const BAR_WIDTH: f32 = 32.0;
    const BAR_HEIGHT: f32 = 3.0;
    const SEGMENT_GAP: f32 = 1.0;
    let segments = status_segments(current, maximum);
    let gap_width = SEGMENT_GAP * f32::from(maximum.saturating_sub(1));
    let segment_width = (BAR_WIDTH - gap_width) / f32::from(maximum);
    let left = agent.position[0].mul_add(WORLD_SCALE, -(BAR_WIDTH * 0.5)) + segment_width * 0.5;
    for (index, is_filled) in segments.into_iter().enumerate() {
        let x = (index as f32).mul_add(segment_width + SEGMENT_GAP, left);
        let color = if is_filled {
            filled_color
        } else {
            Color::srgba(0.08, 0.08, 0.08, 0.72)
        };
        commands.spawn((
            Sprite::from_color(color, Vec2::new(segment_width, BAR_HEIGHT)),
            Transform::from_xyz(
                x,
                agent.position[1].mul_add(WORLD_SCALE, vertical_offset),
                2.5,
            ),
            SceneVisual,
            Name::new(format!("Agent {} {label} segment {index}", agent.id.0)),
        ));
    }
}

/// Spawn circular live eyes or crossed-out dead eyes in body-local space.
fn spawn_agent_eyes(commands: &mut Commands<'_, '_>, assets: &SceneAssets, agent: &VisualAgent) {
    let body_forward = Vec2::from_angle(agent.heading);
    let body_left = body_forward.perp();
    let center = Vec2::from_array(agent.position) * WORLD_SCALE;
    for (index, glyph) in agent_eye_geometry(agent.is_alive).into_iter().enumerate() {
        let (offset, stroke_rotation) = match glyph {
            EyeGlyph::Circle { offset } => (offset, None),
            EyeGlyph::Stroke { offset, rotation } => (offset, Some(rotation)),
        };
        let position = center + body_forward * offset.x + body_left * offset.y;
        if let Some(stroke_rotation) = stroke_rotation {
            commands.spawn((
                Sprite::from_color(Color::BLACK, Vec2::new(5.5, 1.4)),
                Transform::from_xyz(position.x, position.y, 2.4)
                    .with_rotation(Quat::from_rotation_z(agent.heading + stroke_rotation)),
                SceneVisual,
                Name::new(format!("Agent {} dead eye stroke {index}", agent.id.0)),
            ));
        } else {
            commands.spawn((
                Mesh2d(assets.circle.clone()),
                MeshMaterial2d(assets.black.clone()),
                Transform::from_xyz(position.x, position.y, 2.4)
                    .with_scale(Vec3::new(2.2, 2.2, 1.0)),
                SceneVisual,
                Name::new(format!("Agent {} live eye {index}", agent.id.0)),
            ));
        }
    }
}

/// Draw the forward mouth and open it while the attack action is active.
fn spawn_agent_mouth(commands: &mut Commands<'_, '_>, agent: &VisualAgent) {
    let forward = Vec2::from_angle(agent.heading);
    let center = Vec2::from_array(agent.position) * WORLD_SCALE;
    // Keep the visible mouth aligned with the forward physics hitbox.
    let position = center + forward * 11.0;
    let size = if agent.attack_active {
        Vec2::new(7.0, 4.0)
    } else {
        Vec2::new(7.0, 1.5)
    };
    commands.spawn((
        Sprite::from_color(Color::BLACK, size),
        Transform::from_xyz(position.x, position.y, 2.45)
            .with_rotation(Quat::from_rotation_z(agent.heading)),
        SceneVisual,
        Name::new(format!("Agent {} mouth", agent.id.0)),
    ));
}

/// Draw current episode return and one fading signed step-reward change.
fn spawn_agent_reward_hud(
    commands: &mut Commands<'_, '_>,
    agent: &VisualAgent,
    episode_return: f64,
    popup: Option<&RewardPopup>,
) {
    let center = Vec2::from_array(agent.position) * WORLD_SCALE;
    // Anchor the cumulative return above the existing physiology bars.
    commands.spawn((
        Sprite::from_color(Color::srgba(0.03, 0.03, 0.03, 0.82), Vec2::new(48.0, 12.0)),
        Transform::from_xyz(center.x, center.y + 36.0, 2.5),
        SceneVisual,
        Name::new(format!("Agent {} reward backing", agent.id.0)),
    ));
    commands.spawn((
        Text2d::new(format!("R {episode_return:+.3}")),
        TextFont {
            font_size: 10.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Anchor::CENTER,
        Transform::from_xyz(center.x, center.y + 36.0, 2.6),
        SceneVisual,
        Name::new(format!("Agent {} episode reward", agent.id.0)),
    ));
    let Some(popup) = popup else {
        return;
    };
    // Move the signed event delta upward while its opacity decreases.
    let alpha = popup.alpha();
    let color = if popup.value >= 0.0 {
        Color::srgba(0.35, 1.0, 0.45, alpha)
    } else {
        Color::srgba(1.0, 0.3, 0.3, alpha)
    };
    let rise = (1.0 - alpha) * 12.0;
    commands.spawn((
        Text2d::new(format!("{:+.4}", popup.value)),
        TextFont {
            font_size: 13.0,
            ..default()
        },
        TextColor(color),
        Anchor::CENTER,
        Transform::from_xyz(center.x, center.y + 51.0 + rise, 2.7),
        SceneVisual,
        Name::new(format!("Agent {} reward delta", agent.id.0)),
    ));
}

/// Spawn programmer-art primitives for one complete world snapshot.
fn spawn_snapshot(
    commands: &mut Commands<'_, '_>,
    assets: &SceneAssets,
    snapshot: &VisualWorldSnapshot,
    offset: Vec2,
    reward_is_active: bool,
    episode_returns: &BTreeMap<AgentId, f64>,
    reward_popups: &BTreeMap<AgentId, RewardPopup>,
) {
    spawn_world_background(commands, snapshot, offset);

    for object in &snapshot.objects {
        let (material, depth) = match object.kind {
            VisualObjectKind::Food => (assets.food.clone(), 1.0),
            VisualObjectKind::Well => (assets.well.clone(), 0.5),
            VisualObjectKind::Tree => (assets.tree.clone(), 0.2),
            VisualObjectKind::Rock => (assets.rock.clone(), 0.2),
            VisualObjectKind::Thorn => (assets.thorn.clone(), 0.3),
            VisualObjectKind::Shelter => (assets.shelter.clone(), 0.15),
        };
        let scale = object.radius * WORLD_SCALE;
        let position = Vec2::from_array(object.position) * WORLD_SCALE + offset;
        commands.spawn((
            Mesh2d(assets.circle.clone()),
            MeshMaterial2d(material),
            Transform::from_xyz(position.x, position.y, depth)
                .with_scale(Vec3::new(scale, scale, 1.0)),
            SceneVisual,
        ));
    }

    for agent in &snapshot.agents {
        let mut translated = *agent;
        translated.position =
            (Vec2::from_array(translated.position) + offset / WORLD_SCALE).to_array();
        spawn_agent(
            commands,
            assets,
            snapshot,
            &translated,
            reward_is_active,
            episode_returns.get(&agent.id).copied().unwrap_or_default(),
            reward_popups.get(&agent.id),
        );
    }
}

/// Spawn the world floor and four fixed border strips.
fn spawn_world_background(
    commands: &mut Commands<'_, '_>,
    snapshot: &VisualWorldSnapshot,
    offset: Vec2,
) {
    let map_half_extent = snapshot.map_half_extent;
    let diameter = map_half_extent * WORLD_SCALE * 2.0;
    commands.spawn((
        Sprite::from_color(Color::srgb_u8(40, 74, 43), Vec2::splat(diameter)),
        Transform::from_xyz(offset.x, offset.y, -2.0),
        SceneVisual,
    ));
    if let Some(gorge) = snapshot.gorge {
        // Draw one continuous void, then cover its safe corridor with a bridge.
        let gorge_size = Vec2::new(gorge.half_width * WORLD_SCALE * 2.0, diameter);
        commands.spawn((
            Sprite::from_color(Color::srgb_u8(20, 24, 30), gorge_size),
            Transform::from_xyz(offset.x, offset.y, -1.9),
            SceneVisual,
        ));
        let bridge_size = Vec2::new(
            gorge.half_width * WORLD_SCALE * 2.0,
            gorge.bridge_half_width * WORLD_SCALE * 2.0,
        );
        commands.spawn((
            Sprite::from_color(Color::srgb_u8(164, 126, 72), bridge_size),
            Transform::from_xyz(offset.x, offset.y, -1.8),
            SceneVisual,
        ));
    }
    let border = Color::srgb_u8(159, 123, 71);
    for (size, position) in [
        (
            Vec2::new(diameter + 8.0, 8.0),
            Vec2::new(0.0, diameter * 0.5),
        ),
        (
            Vec2::new(diameter + 8.0, 8.0),
            Vec2::new(0.0, -diameter * 0.5),
        ),
        (Vec2::new(8.0, diameter), Vec2::new(diameter * 0.5, 0.0)),
        (Vec2::new(8.0, diameter), Vec2::new(-diameter * 0.5, 0.0)),
    ] {
        commands.spawn((
            Sprite::from_color(border, size),
            Transform::from_translation((position + offset).extend(-1.0)),
            SceneVisual,
        ));
    }
}

/// Label one grid cell with its lane, curriculum stage, and replay purpose.
fn spawn_batch_lane_label(
    commands: &mut Commands<'_, '_>,
    lane: usize,
    stage: CurriculumStage,
    purpose: SurvivalBatchPurpose,
    map_half_extent: f32,
    offset: Vec2,
) {
    let corner = map_half_extent * WORLD_SCALE;
    let environment_number = lane.saturating_add(1);
    let stage_key = stage.as_key();
    let purpose_label = batch_purpose_label(purpose);
    // Keep the role visible because initialization previews are not optimizer data.
    commands.spawn((
        Text2d::new(format!(
            "Environment {environment_number} · {stage_key} · {purpose_label}"
        )),
        TextFont {
            font_size: 15.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Anchor::TOP_LEFT,
        Transform::from_xyz(offset.x - corner + 10.0, offset.y + corner - 8.0, 3.0),
        SceneVisual,
        Name::new(format!("Training environment {environment_number} label")),
    ));
}

/// Format the distinct startup-preview and optimizer-update roles.
fn batch_purpose_label(purpose: SurvivalBatchPurpose) -> String {
    match purpose {
        SurvivalBatchPurpose::InitializationPreview => "initialization preview".to_owned(),
        SurvivalBatchPurpose::TrainingUpdate { iteration } => format!("update {iteration}"),
    }
}

/// Spawn one agent body, reward pulse, eyes, and segmented status bars.
fn spawn_agent(
    commands: &mut Commands<'_, '_>,
    assets: &SceneAssets,
    snapshot: &VisualWorldSnapshot,
    agent: &VisualAgent,
    reward_is_active: bool,
    episode_return: f64,
    reward_popup: Option<&RewardPopup>,
) {
    if agent.is_alive && reward_is_active {
        // A green backing pulse makes positive healthy-survival reward visible
        // at the exact transition that produced it.
        commands.spawn((
            Mesh2d(assets.circle.clone()),
            MeshMaterial2d(assets.reward.clone()),
            Transform::from_xyz(
                agent.position[0] * WORLD_SCALE,
                agent.position[1] * WORLD_SCALE,
                1.8,
            )
            .with_rotation(Quat::from_rotation_z(agent.heading))
            .with_scale(Vec3::new(17.0, 14.0, 1.0)),
            SceneVisual,
        ));
    }
    let material = if agent.is_alive {
        match agent.species {
            Species::Bunny => assets.bunny.clone(),
            Species::Fox => assets.fox.clone(),
        }
    } else {
        assets.dead.clone()
    };
    commands.spawn((
        Mesh2d(assets.circle.clone()),
        MeshMaterial2d(material),
        Transform::from_xyz(
            agent.position[0] * WORLD_SCALE,
            agent.position[1] * WORLD_SCALE,
            2.0,
        )
        .with_rotation(Quat::from_rotation_z(agent.heading))
        .with_scale((AGENT_SIZE * WORLD_SCALE * 0.5).extend(1.0)),
        SceneVisual,
    ));
    spawn_agent_eyes(commands, assets, agent);
    spawn_agent_mouth(commands, agent);
    spawn_status_bar(
        commands,
        agent,
        agent.satiation,
        snapshot.maximum_satiation,
        16.0,
        Color::srgb_u8(139, 90, 43),
        "satiation",
    );
    spawn_status_bar(
        commands,
        agent,
        agent.hydration,
        snapshot.maximum_hydration,
        20.0,
        Color::srgb_u8(45, 133, 230),
        "hydration",
    );
    spawn_status_bar(
        commands,
        agent,
        agent.hit_points,
        snapshot.maximum_hit_points,
        24.0,
        Color::srgb_u8(220, 48, 48),
        "health",
    );
    spawn_status_bar(
        commands,
        agent,
        agent.exposure,
        snapshot.maximum_exposure,
        28.0,
        if agent.in_shelter {
            Color::srgb_u8(126, 230, 145)
        } else {
            Color::srgb_u8(151, 101, 214)
        },
        "exposure",
    );
    spawn_agent_reward_hud(commands, agent, episode_return, reward_popup);
}

/// Draw the interactive HUD through Bevy Inspector's egui integration.
pub(super) fn ecosystem_hud_ui(world: &mut World) {
    let Ok(mut context) = world
        .query_filtered::<&mut EguiContext, With<PrimaryEguiContext>>()
        .single_mut(world)
    else {
        return;
    };
    let mut egui_context = context.deref_mut().clone();
    let Some(mut session) = world.get_non_send_resource_mut::<WatchSession>() else {
        return;
    };

    // Snapshot all displayed values before egui mutates playback controls.
    let snapshot = session
        .training_batch
        .as_ref()
        .and_then(|batch| batch.current.lanes().first().map(|lane| (batch, lane)))
        .map_or_else(
            || session.current_snapshot.clone(),
            |(batch, lane)| lane.snapshot(batch.frame).clone(),
        );
    let episode_metrics = if session.training_batch.is_some() {
        Vec::new()
    } else {
        session.ecosystem.episode_metrics()
    };

    egui::Window::new("Ecosystem learning HUD")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .default_width(390.0)
        .max_height(690.0)
        .vscroll(true)
        .resizable(true)
        .show(egui_context.get_mut(), |ui| {
            ui.heading(session.stage.title());
            if let Some(batch) = session.training_batch.as_ref() {
                let purpose = batch.current.purpose();
                let purpose_label = batch_purpose_label(purpose);
                let current_frame = batch.frame.saturating_add(1);
                let frame_count = batch.current.frame_count();
                ui.colored_label(
                    egui::Color32::LIGHT_BLUE,
                    format!(
                        "3x3 {purpose_label} · frame {current_frame}/{frame_count}"
                    ),
                );
                match purpose {
                    SurvivalBatchPurpose::InitializationPreview => {
                        ui.small("This prerecorded nine-world preview loops while survival initialization runs. Optimizer batches replace it.");
                    }
                    SurvivalBatchPurpose::TrainingUpdate { .. } => {
                        ui.small("All nine prerecorded environments contributed to this policy update. The trace loops while the worker prepares its successor.");
                    }
                }
            }
            ui.label("Reward: bounded healthy survival plus drive reduction");
            ui.small(
                "Food, water, shelter, and HP define physiological drive. Death emits no reward.",
            );
            ui.separator();
            show_debug_overlay_controls(ui, &mut session, &snapshot);

            if let Some(dashboard) = session.dashboard.as_mut() {
                show_training_dashboard(ui, dashboard);
            } else {
                ui.separator();
                ui.label("Checkpoint playback");
                ui.small(&session.checkpoint_label);
            }

            show_world_dashboard(ui, &mut session, &snapshot, &episode_metrics);
            ui.separator();
            ui.small(
                "WASD/arrows or left/middle drag pan | wheel/+/- zoom | space pause | R reset | F1 inspector",
            );
        });
}

/// Draw world controls and exact reward inputs for the current episode.
fn show_world_dashboard(
    ui: &mut egui::Ui,
    session: &mut WatchSession,
    snapshot: &VisualWorldSnapshot,
    episode_metrics: &[AgentEpisodeMetrics],
) {
    let batch_status = session.training_batch.as_ref().and_then(|batch| {
        batch.current.lanes().first().map(|lane| {
            (
                batch.current.purpose(),
                batch.frame,
                batch.current.frame_count(),
                lane.episode_return(batch.frame),
                lane.episode_return(batch.frame.saturating_sub(1)),
            )
        })
    });
    // Aggregate only living physiology because dead bodies are presentation data.
    let (count, satiation, hydration, hit_points) =
        snapshot.agents.iter().filter(|agent| agent.is_alive).fold(
            (0_usize, 0.0_f32, 0.0_f32, 0.0_f32),
            |(count, satiation, hydration, hit_points), agent| {
                (
                    count + 1,
                    satiation + f32::from(agent.satiation),
                    hydration + f32::from(agent.hydration),
                    hit_points + f32::from(agent.hit_points),
                )
            },
        );
    let divisor = count.max(1) as f32;
    let food_eaten = episode_metrics
        .iter()
        .map(|metric| metric.food_eaten)
        .sum::<u32>();
    let water_consumed = episode_metrics
        .iter()
        .map(|metric| metric.water_consumed)
        .sum::<f32>();
    let kills = episode_metrics
        .iter()
        .map(|metric| metric.kills)
        .sum::<u32>();
    let thorn_damage = episode_metrics
        .iter()
        .map(|metric| metric.thorn_damage)
        .sum::<f32>();
    let collision_damage = episode_metrics
        .iter()
        .map(|metric| metric.collision_damage)
        .sum::<f32>();
    let overconsumption_damage = episode_metrics
        .iter()
        .map(|metric| metric.overconsumption_damage)
        .sum::<f32>();

    show_experiment_tuning(ui, session);
    show_world_controls(ui, session);
    ui.add(egui::Slider::new(&mut session.speed, 0.25..=128.0).text("Playback speed"));
    egui::Grid::new("ecosystem-world-metrics")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            if let Some((purpose, frame, frame_count, _, _)) = batch_status {
                ui.label("Batch / replay frame");
                let purpose_label = batch_purpose_label(purpose);
                let current_frame = frame.saturating_add(1);
                ui.label(format!("{purpose_label} / {current_frame}/{frame_count}"));
            } else {
                ui.label("Episode / time");
                ui.label(format!(
                    "{} / {:.1} s",
                    session.episode,
                    snapshot.summary.step as f32 * session.config.time_step
                ));
            }
            ui.end_row();
            ui.label(if batch_status.is_some() {
                "Lane 1 alive"
            } else {
                "Alive"
            });
            ui.label(format!("{count} / {}", snapshot.agents.len()));
            ui.end_row();
            ui.label(if batch_status.is_some() {
                "Lane 1 reward"
            } else {
                "Episode reward"
            });
            let (episode_return, latest_reward) = batch_status.map_or(
                (session.episode_return, session.latest_reward),
                |(_, _, _, episode_return, previous_return)| {
                    (episode_return, episode_return - previous_return)
                },
            );
            ui.colored_label(
                egui::Color32::LIGHT_GREEN,
                format!("{episode_return:.2}  ({latest_reward:+.2})"),
            );
            ui.end_row();
            ui.label("Food / well");
            ui.label(format!(
                "{} / {:.1}",
                snapshot.summary.food_count, snapshot.summary.well_water
            ));
            ui.end_row();
            if matches!(
                session.stage,
                CurriculumStage::Shelter
                    | CurriculumStage::Competition
                    | CurriculumStage::PredatorPrey
                    | CurriculumStage::Obstacles
            ) {
                ui.label("Weather");
                ui.label(if snapshot.weather_active {
                    "active".to_owned()
                } else {
                    format!("starts at {:.1} s", snapshot.weather_onset_seconds)
                });
                ui.end_row();
            }
            ui.label("Ate / drank");
            ui.label(format!("{food_eaten} / {water_consumed:.1}"));
            ui.end_row();
            if matches!(
                session.stage,
                CurriculumStage::PredatorPrey | CurriculumStage::Obstacles
            ) {
                ui.label("Predations");
                ui.label(kills.to_string());
                ui.end_row();
            }
            if session.stage == CurriculumStage::Obstacles {
                ui.label("Thorn damage");
                ui.label(format!("{thorn_damage:.1}"));
                ui.end_row();
            }
            ui.label("Collision damage");
            ui.label(format!("{collision_damage:.1}"));
            ui.end_row();
            if overconsumption_damage > 0.0 {
                ui.label("Overfull damage");
                ui.label(format!("{overconsumption_damage:.1}"));
                ui.end_row();
            }
            ui.label("Satiation / hydration / HP");
            ui.label(format!(
                "{:.1} / {:.1} / {:.1}",
                satiation / divisor,
                hydration / divisor,
                hit_points / divisor
            ));
            ui.end_row();
        });
}

/// Draw collision and perception overlay controls.
fn show_debug_overlay_controls(
    ui: &mut egui::Ui,
    session: &mut WatchSession,
    snapshot: &VisualWorldSnapshot,
) {
    // Filter choices come from the same snapshot as the rendered rays so a
    // dead or reset agent cannot remain selectable for another frame.
    if session.perception_ray_agent.is_some_and(|selected| {
        !snapshot
            .agents
            .iter()
            .any(|agent| agent.is_alive && agent.id == selected)
    }) {
        session.perception_ray_agent = snapshot
            .agents
            .iter()
            .find(|agent| agent.is_alive)
            .map(|agent| agent.id);
    }
    ui.checkbox(&mut session.show_perception_rays, "Show perception rays");
    ui.checkbox(
        &mut session.show_collision_shapes,
        "Show hitbox/hurtbox shapes",
    );
    if session.show_collision_shapes {
        ui.small("Cyan outlines are hurtboxes; orange mouth outlines are hitboxes.");
        ui.small("An active mouth hitbox turns bright red-orange.");
    }
    if !session.show_perception_rays {
        return;
    }
    ui.small("Dim lines are empty sectors; colored lines stop at the perceived hit.");
    ui.small(
        "Hit colors: food yellow, water blue, agents white/orange, thorns magenta, gorge gray, bridge brown.",
    );
    egui::ComboBox::from_label("Ray source")
        .selected_text(session.perception_ray_agent.map_or_else(
            || "All agents".to_owned(),
            |agent| format!("Agent {}", agent.0),
        ))
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut session.perception_ray_agent, None, "All agents");
            for agent in snapshot.agents.iter().filter(|agent| agent.is_alive) {
                ui.selectable_value(
                    &mut session.perception_ray_agent,
                    Some(agent.id),
                    format!("Agent {} ({:?})", agent.id.0, agent.species),
                );
            }
        });
}

/// Draw the live playground profile and its explicit application boundaries.
fn show_experiment_tuning(ui: &mut egui::Ui, session: &mut WatchSession) {
    let training_control = session
        .dashboard
        .as_ref()
        .map(|dashboard| Arc::clone(&dashboard.control));
    let mut tuning = session.pending_tuning;
    let mut changed = false;
    let mut apply_to_world = false;
    egui::CollapsingHeader::new("Experiment tuning")
        .default_open(false)
        .show(ui, |ui| {
            ui.small("Training applies changes at the next PPO iteration.");
            changed |= show_perception_tuning(ui, &mut tuning);
            show_reward_model(ui);
            changed |= show_dynamics_tuning(ui, &mut tuning);
            ui.horizontal(|ui| {
                if ui.button("Reset defaults").clicked() {
                    tuning = ExperimentTuning::default();
                    changed = true;
                }
                let is_world_dirty = tuning != session.config.experiment_tuning();
                if ui
                    .add_enabled(
                        is_world_dirty,
                        egui::Button::new("Apply + restart visible world"),
                    )
                    .clicked()
                {
                    apply_to_world = true;
                }
            });
            if let Some(dashboard) = session.dashboard.as_ref() {
                let applied = dashboard.progress.experiment_tuning;
                if applied == tuning {
                    ui.small("Training is using these settings.");
                } else {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "Pending for training after the current PPO iteration.",
                    );
                }
            }
            ui.small("Vertical graph markers show when training settings changed.");
            ui.small("A settings change starts a new survival-learning curve.");
        });

    // Publish only profiles that pass the same domain validation used by the
    // simulation, even though every slider already has bounded values.
    if changed {
        let mut validation = session.config.clone();
        if validation.apply_experiment_tuning(tuning).is_ok() {
            session.pending_tuning = tuning;
            if let Some(control) = training_control {
                control.replace_tuning(tuning);
            }
        }
    }
    if apply_to_world && session.apply_pending_tuning().is_err() {
        session.playback = PlaybackState::Paused;
    }
}

/// Draw the fixed-capacity active perception-sector control.
fn show_perception_tuning(ui: &mut egui::Ui, tuning: &mut ExperimentTuning) -> bool {
    // Edit a primitive copy because the validated domain count cannot represent
    // the temporary out-of-range states that a general slider API permits.
    let mut ray_count = tuning.perception_ray_count.get();
    let ray_count_changed = ui
        .add(
            egui::Slider::new(
                &mut ray_count,
                PerceptionRayCount::MIN..=PerceptionRayCount::MAX,
            )
            .step_by(2.0)
            .text("Perception rays"),
        )
        .on_hover_text("Fewer rays retain frontal detail first; inactive network inputs stay zero.")
        .changed();
    if ray_count_changed {
        let Ok(validated_count) = PerceptionRayCount::try_from(ray_count) else {
            return false;
        };
        tuning.perception_ray_count = validated_count;
    }
    let mut changed = ray_count_changed;
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.gaze_yaw_limit_degrees, 10.0..=35.0)
                .text("Eye motion limit"),
        )
        .on_hover_text("Both eyes move together without rotating the body or seeing behind it.")
        .changed();
    changed
}
/// Explain the fixed reward contract used by every curriculum stage.
fn show_reward_model(ui: &mut egui::Ui) {
    ui.label("Fixed healthy-survival reward");
    ui.small("Each live step earns time-scaled reward reduced by current physiological drive.");
    ui.small("Reducing drive adds bounded feedback; dying never pays a terminal bonus.");
}

/// Draw reset physiology, need pressure, damage, and time-limit controls.
fn show_dynamics_tuning(ui: &mut egui::Ui, tuning: &mut ExperimentTuning) -> bool {
    // Use bounded sliders because these values enter physics and episode
    // lifecycle calculations on the next safe boundary.
    let mut changed = false;
    ui.label("Episode dynamics");
    changed |= ui
        .add(egui::Slider::new(&mut tuning.maximum_hit_points, 1..=20).text("Maximum HP"))
        .changed();
    tuning.initial_hit_points = tuning.initial_hit_points.min(tuning.maximum_hit_points);
    changed |= ui
        .add(
            egui::Slider::new(
                &mut tuning.initial_hit_points,
                1..=tuning.maximum_hit_points,
            )
            .text("Starting HP"),
        )
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.maximum_satiation, 1..=20).text("Maximum satiation"))
        .changed();
    tuning.initial_satiation = tuning.initial_satiation.min(tuning.maximum_satiation);
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.initial_satiation, 0..=tuning.maximum_satiation)
                .text("Starting satiation"),
        )
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.maximum_hydration, 1..=20).text("Maximum hydration"))
        .changed();
    tuning.initial_hydration = tuning.initial_hydration.min(tuning.maximum_hydration);
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.initial_hydration, 0..=tuning.maximum_hydration)
                .text("Starting hydration"),
        )
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.need_loss_interval_seconds, 1..=60)
                .text("Need loss seconds"),
        )
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.starvation_damage_interval_seconds, 1..=60)
                .text("Starvation damage seconds"),
        )
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.dehydration_damage_interval_seconds, 1..=60)
                .text("Dehydration damage seconds"),
        )
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.movement_need_cost_percent, 0..=100)
                .text("Movement need cost %"),
        )
        .on_hover_text("Translation advances the need clock by this extra percentage; rotation and eye motion remain free.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.movement_speed_multiplier, 0.5..=2.0)
                .text("Movement speed"),
        )
        .on_hover_text("Scales both forward acceleration and maximum translation speed.")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.episode_seconds, 5..=300).text("Episode seconds"))
        .on_hover_text("Short limits produce updates sooner but truncate delayed consequences.")
        .changed();
    changed
}

/// Draw playback controls that only mutate the visual evaluation world.
fn show_world_controls(ui: &mut egui::Ui, session: &mut WatchSession) {
    // Reset failures pause playback instead of leaving controls unresponsive.
    ui.horizontal(|ui| {
        let (playback_label, can_toggle) = match session.playback {
            PlaybackState::Running => ("Pause world", true),
            PlaybackState::Paused => ("Resume world", true),
            PlaybackState::EpisodePause { .. } | PlaybackState::ResetReady => {
                ("Showing episode result", false)
            }
        };
        if ui
            .add_enabled(can_toggle, egui::Button::new(playback_label))
            .clicked()
        {
            session.playback.toggle_user_pause();
        }
        if ui.button("Reset episode").clicked() && session.reset_episode().is_err() {
            session.playback = PlaybackState::Paused;
        }
    });
}

/// Draw optimizer controls, wall-clock evidence, and return curves.
fn show_training_dashboard(ui: &mut egui::Ui, dashboard: &mut DemoDashboard) {
    // All wall-clock and optimizer claims come from the latest typed event.
    let fraction =
        dashboard.progress.iteration as f32 / dashboard.progress.total_iterations.max(1) as f32;
    ui.separator();
    ui.heading("Live PPO training");
    ui.add(
        egui::ProgressBar::new(fraction)
            .show_percentage()
            .text(format!(
                "iteration {} of {}",
                dashboard.progress.iteration, dashboard.progress.total_iterations
            )),
    );

    let status = match &dashboard.status {
        TrainingStatus::Running => "running".to_owned(),
        TrainingStatus::Paused => "paused".to_owned(),
        TrainingStatus::Stopping => "stopping".to_owned(),
        TrainingStatus::Complete => "complete".to_owned(),
        TrainingStatus::Failed(message) => format!("failed: {message}"),
    };
    ui.label(format!(
        "{status} | {} steps | {:.1} s elapsed",
        dashboard.progress.global_steps,
        dashboard.started_at.elapsed().as_secs_f32()
    ));
    ui.label(trainer_activity_label(dashboard.activity));

    show_training_controls(ui, dashboard);
    let progress = &dashboard.progress;

    if !progress.has_evaluation {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Initialization is in progress; the worker phase is shown above.");
        });
        ui.small(format!("Artifacts: {}", progress.run_dir.display()));
        return;
    }

    show_training_metrics(ui, dashboard);
    ui.label("Every training episode return");
    draw_learning_graph(
        ui,
        &dashboard.bunny_episode_curve,
        &dashboard.bunny_eval_curve,
        &dashboard.fox_eval_curve,
        &dashboard.tuning_markers,
    );
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(244, 196, 74), "training episodes");
        ui.colored_label(egui::Color32::from_rgb(95, 210, 125), "fixed-seed mean");
        if !dashboard.fox_eval_curve.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(235, 120, 70), "fox evaluation");
        }
        if !dashboard.tuning_markers.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(190, 125, 235), "settings changed");
        }
    });
    if !dashboard.fox_episode_curve.is_empty() {
        ui.label("Every fox training episode return");
        draw_two_series_graph(
            ui,
            &dashboard.fox_episode_curve,
            &[],
            egui::Color32::from_rgb(235, 120, 70),
            egui::Color32::TRANSPARENT,
            false,
        );
    }
    ui.label("Learning velocity (fixed-seed reward points per iteration)");
    draw_two_series_graph(
        ui,
        &dashboard.learning_velocity_curve,
        &[],
        egui::Color32::from_rgb(105, 195, 245),
        egui::Color32::TRANSPARENT,
        false,
    );
    ui.colored_label(
        egui::Color32::from_rgb(105, 195, 245),
        "bunny evaluation change",
    );
    ui.label("Mean fixed-seed survival time");
    draw_two_series_graph(
        ui,
        &dashboard.bunny_survival_curve,
        &dashboard.fox_survival_curve,
        egui::Color32::from_rgb(95, 210, 125),
        egui::Color32::from_rgb(235, 120, 70),
        false,
    );
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(95, 210, 125), "bunny survival");
        if !dashboard.fox_survival_curve.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(235, 120, 70), "fox survival");
        }
    });
    ui.label("Bunny optimizer learning rates");
    draw_two_series_graph(
        ui,
        &dashboard.actor_learning_rate_curve,
        &dashboard.critic_learning_rate_curve,
        egui::Color32::from_rgb(244, 196, 74),
        egui::Color32::from_rgb(190, 125, 235),
        true,
    );
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(244, 196, 74), "actor");
        ui.colored_label(egui::Color32::from_rgb(190, 125, 235), "critic");
    });
    ui.small(format!("Artifacts: {}", progress.run_dir.display()));
}

/// Format worker activity without describing prerecorded trajectories as live worlds.
fn trainer_activity_label(activity: TrainerActivity) -> String {
    match activity {
        TrainerActivity::InitializationPreview => {
            "Worker: collecting the initialization preview".to_owned()
        }
        TrainerActivity::Evaluation {
            iteration,
            episodes,
        } => format!("Worker: evaluating iteration {iteration} across {episodes} episodes"),
        TrainerActivity::SurvivalTeacher => {
            "Worker: fitting the observation-only survival teacher".to_owned()
        }
        TrainerActivity::MemoryInvariance => {
            "Worker: regularizing recurrent policy memory".to_owned()
        }
        TrainerActivity::Collection {
            iteration,
            episodes,
        } => format!("Worker: collecting {episodes} episodes for PPO iteration {iteration}"),
        TrainerActivity::Optimization { iteration } => {
            format!("Worker: optimizing PPO iteration {iteration}")
        }
    }
}

/// Format absolute reward gain and omit undefined relative gain from zero.
fn format_reward_gain(current: f64, baseline: f64) -> String {
    let gain = current - baseline;
    if baseline.abs() < f64::EPSILON {
        format!("{gain:+.2} (n/a)")
    } else {
        let gain_percent = gain / baseline * 100.0;
        format!("{gain:+.2} ({gain_percent:+.1}%)")
    }
}

/// Draw typed return, gain, loss, entropy, and selection metrics.
fn show_training_metrics(ui: &mut egui::Ui, dashboard: &DemoDashboard) {
    // Compare current fixed-seed returns to the iteration-zero policy.
    let progress = &dashboard.progress;
    egui::Grid::new("ecosystem-training-metrics")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            ui.label("Train return");
            ui.label(format!("{:.2}", progress.bunny_train_return));
            ui.end_row();
            if dashboard.has_fox {
                ui.label("Fox train return");
                ui.label(format!("{:.2}", progress.fox_train_return));
                ui.end_row();
            }
            ui.label("Fixed-seed return");
            ui.label(format!("{:.2}", progress.bunny_eval_return));
            ui.end_row();
            if let Some([_, baseline]) = dashboard.bunny_eval_curve.first() {
                let gain = f64::from(progress.bunny_eval_return) - baseline;
                ui.label("Gain from start");
                ui.colored_label(
                    if gain >= 0.0 {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::LIGHT_RED
                    },
                    format_reward_gain(f64::from(progress.bunny_eval_return), *baseline),
                );
                ui.end_row();
            }
            if dashboard.has_fox {
                ui.label("Fox fixed-seed return");
                ui.label(format!("{:.2}", progress.fox_eval_return));
                ui.end_row();
            }
            ui.label("Fixed-seed survival");
            ui.label(format!("{:.2} s", progress.bunny_eval_lifetime));
            ui.end_row();
            if dashboard.has_fox {
                ui.label("Fox fixed-seed survival");
                ui.label(format!("{:.2} s", progress.fox_eval_lifetime));
                ui.end_row();
            }
            ui.label("Actor / critic loss");
            ui.label(format!(
                "{:.4} / {:.4}",
                progress.actor_loss, progress.critic_loss
            ));
            ui.end_row();
            ui.label("Entropy");
            ui.label(format!("{:.4}", progress.entropy));
            ui.end_row();
            ui.label("Actor / critic learning rate");
            ui.label(format!(
                "{:.2e} / {:.2e}",
                progress.actor_learning_rate, progress.critic_learning_rate
            ));
            ui.end_row();
            ui.label("Selected as best");
            ui.label(if progress.is_best { "yes" } else { "no" });
            ui.end_row();
        });
}

/// Draw controls that take effect between complete optimizer updates.
fn show_training_controls(ui: &mut egui::Ui, dashboard: &mut DemoDashboard) {
    // Atomics let the egui render pass communicate without blocking the worker.
    ui.horizontal(|ui| {
        let is_paused = dashboard.control.is_paused.load(Ordering::Relaxed);
        let pause_label = if is_paused {
            "Resume training"
        } else {
            "Pause training"
        };
        if ui.button(pause_label).clicked() {
            dashboard
                .control
                .is_paused
                .store(!is_paused, Ordering::Relaxed);
            dashboard.status = if is_paused {
                TrainingStatus::Running
            } else {
                TrainingStatus::Paused
            };
        }
        if ui.button("Stop after update").clicked() {
            dashboard.control.should_stop.store(true, Ordering::Relaxed);
            dashboard.status = TrainingStatus::Stopping;
        }
    });
}

/// Paint a compact dependency-free line graph in the Inspector-egui HUD.
fn draw_learning_graph(
    ui: &mut egui::Ui,
    training: &[[f64; 2]],
    bunny_evaluation: &[[f64; 2]],
    fox_evaluation: &[[f64; 2]],
    tuning_markers: &[f64],
) {
    // Scale every series together so visual comparisons preserve magnitude.
    let desired_size = egui::vec2(ui.available_width(), 150.0);
    let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
    let graph = response.rect.shrink2(egui::vec2(8.0, 8.0));
    painter.rect_filled(graph, 4.0, egui::Color32::from_rgb(20, 28, 23));

    let max_iteration = training
        .iter()
        .chain(bunny_evaluation)
        .chain(fox_evaluation)
        .map(|point| point[0])
        .fold(1.0_f64, f64::max);
    let max_return = training
        .iter()
        .chain(bunny_evaluation)
        .chain(fox_evaluation)
        .map(|point| point[1])
        .fold(1.0_f64, f64::max);
    painter.line_segment(
        [graph.left_bottom(), graph.right_bottom()],
        egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
    );
    painter.line_segment(
        [graph.left_bottom(), graph.left_top()],
        egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
    );
    // Vertical markers separate rollout regimes without implying that reward
    // totals on opposite sides use the same scale.
    for marker in tuning_markers {
        let x = graph
            .width()
            .mul_add((*marker / max_iteration) as f32, graph.left());
        painter.line_segment(
            [egui::pos2(x, graph.top()), egui::pos2(x, graph.bottom())],
            egui::Stroke::new(1.5_f32, egui::Color32::from_rgb(190, 125, 235)),
        );
    }
    paint_curve(
        &painter,
        graph,
        training,
        max_iteration,
        max_return,
        egui::Color32::from_rgb(244, 196, 74),
    );
    paint_curve(
        &painter,
        graph,
        bunny_evaluation,
        max_iteration,
        max_return,
        egui::Color32::from_rgb(95, 210, 125),
    );
    paint_curve(
        &painter,
        graph,
        fox_evaluation,
        max_iteration,
        max_return,
        egui::Color32::from_rgb(235, 120, 70),
    );
    painter.text(
        graph.left_top() + egui::vec2(4.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!("{max_return:.0} s"),
        egui::FontId::monospace(10.0),
        egui::Color32::LIGHT_GRAY,
    );
}

/// Project one metric series into the graph rectangle.
fn paint_curve(
    painter: &egui::Painter,
    graph: egui::Rect,
    points: &[[f64; 2]],
    max_iteration: f64,
    max_return: f64,
    color: egui::Color32,
) {
    // Stream adjacent segments without allocating a second point collection.
    let positions = points.iter().map(|point| {
        egui::pos2(
            graph
                .width()
                .mul_add((point[0] / max_iteration) as f32, graph.left()),
            graph
                .height()
                .mul_add(-((point[1] / max_return) as f32), graph.bottom()),
        )
    });
    let mut previous = None;
    for position in positions {
        painter.circle_filled(position, 2.5, color);
        if let Some(start) = previous {
            painter.line_segment([start, position], egui::Stroke::new(2.0_f32, color));
        }
        previous = Some(position);
    }
}

/// Draw two metric series with a shared signed or positive vertical range.
fn draw_two_series_graph(
    ui: &mut egui::Ui,
    first: &[[f64; 2]],
    second: &[[f64; 2]],
    first_color: egui::Color32,
    second_color: egui::Color32,
    scientific_labels: bool,
) {
    let desired_size = egui::vec2(ui.available_width(), 105.0);
    let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::hover());
    let graph = response.rect.shrink2(egui::vec2(8.0, 8.0));
    painter.rect_filled(graph, 4.0, egui::Color32::from_rgb(20, 28, 23));
    let max_iteration = first
        .iter()
        .chain(second)
        .map(|point| point[0])
        .fold(1.0_f64, f64::max);
    let mut minimum = first
        .iter()
        .chain(second)
        .map(|point| point[1])
        .fold(0.0_f64, f64::min);
    let mut maximum = first
        .iter()
        .chain(second)
        .map(|point| point[1])
        .fold(0.0_f64, f64::max);
    if (maximum - minimum).abs() < f64::EPSILON {
        let padding = maximum.abs().max(1e-6) * 0.1;
        minimum -= padding;
        maximum += padding;
    }
    let zero_fraction = ((0.0 - minimum) / (maximum - minimum)).clamp(0.0, 1.0) as f32;
    let zero_y = graph.height().mul_add(-zero_fraction, graph.bottom());
    painter.line_segment(
        [
            egui::pos2(graph.left(), zero_y),
            egui::pos2(graph.right(), zero_y),
        ],
        egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
    );
    paint_curve_in_range(
        &painter,
        graph,
        first,
        max_iteration,
        minimum,
        maximum,
        first_color,
    );
    paint_curve_in_range(
        &painter,
        graph,
        second,
        max_iteration,
        minimum,
        maximum,
        second_color,
    );
    let label = if scientific_labels {
        format!("{maximum:.1e}")
    } else {
        format!("{maximum:+.2}")
    };
    painter.text(
        graph.left_top() + egui::vec2(4.0, 4.0),
        egui::Align2::LEFT_TOP,
        label,
        egui::FontId::monospace(10.0),
        egui::Color32::LIGHT_GRAY,
    );
}

/// Project one metric series into an arbitrary shared vertical range.
fn paint_curve_in_range(
    painter: &egui::Painter,
    graph: egui::Rect,
    points: &[[f64; 2]],
    max_iteration: f64,
    minimum: f64,
    maximum: f64,
    color: egui::Color32,
) {
    let mut previous = None;
    for point in points {
        let x = graph
            .width()
            .mul_add((point[0] / max_iteration) as f32, graph.left());
        let normalized = ((point[1] - minimum) / (maximum - minimum)) as f32;
        let position = egui::pos2(x, graph.height().mul_add(-normalized, graph.bottom()));
        painter.circle_filled(position, 2.5, color);
        if let Some(start) = previous {
            painter.line_segment([start, position], egui::Stroke::new(2.0_f32, color));
        }
        previous = Some(position);
    }
}

/// Recenter and fit all nine cells when the first real training trace arrives.
fn fit_training_batch_camera(
    mut session: NonSendMut<'_, WatchSession>,
    mut camera: Single<'_, '_, (&mut Transform, &mut Projection), With<WorldCamera>>,
) {
    if !session.batch_camera_needs_fit {
        return;
    }
    let Some(batch) = session.training_batch.as_ref() else {
        return;
    };
    let map_half_extent = batch
        .current
        .lanes()
        .iter()
        .map(|lane| lane.snapshot(batch.frame).map_half_extent)
        .fold(0.0_f32, f32::max);
    let (transform, projection) = &mut *camera;
    let Projection::Orthographic(orthographic) = &mut **projection else {
        return;
    };
    // Fit the complete grid once; later pan and zoom remain under user control.
    transform.translation.x = map_half_extent * WORLD_SCALE * 2.0;
    transform.translation.y = 0.0;
    orthographic.scale = batch_camera_scale(map_half_extent);
    session.batch_camera_needs_fit = false;
}

/// Compute the vertical fit used by the fixed 3x3 survival grid.
fn batch_camera_scale(map_half_extent: f32) -> f32 {
    (map_half_extent * WORLD_SCALE * 2.0 * 3.0 / 650.0).max(0.7)
}

/// Pan and zoom the top-down camera without affecting simulation state.
fn camera_controls(
    keys: Res<'_, ButtonInput<KeyCode>>,
    mouse_buttons: Res<'_, ButtonInput<MouseButton>>,
    mouse_motion: Res<'_, AccumulatedMouseMotion>,
    egui_wants_input: Res<'_, EguiWantsInput>,
    mut wheel: MessageReader<'_, '_, MouseWheel>,
    time: Res<'_, Time>,
    mut camera: Single<'_, '_, (&mut Transform, &mut Projection), With<WorldCamera>>,
) {
    // Camera movement changes presentation only; policy observations stay fixed.
    let keyboard_captured = egui_wants_input.wants_any_keyboard_input();
    let mut direction = Vec2::ZERO;
    if !keyboard_captured && (keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft)) {
        direction.x -= 1.0;
    }
    if !keyboard_captured && (keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight)) {
        direction.x += 1.0;
    }
    if !keyboard_captured && (keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown)) {
        direction.y -= 1.0;
    }
    if !keyboard_captured && (keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp)) {
        direction.y += 1.0;
    }
    let (transform, projection) = &mut *camera;
    let Projection::Orthographic(orthographic) = &mut **projection else {
        return;
    };
    let wheel_delta = wheel.read().map(|event| event.y).sum::<f32>();
    let pointer_captured = egui_wants_input.wants_any_pointer_input();
    let dragging =
        mouse_buttons.pressed(MouseButton::Left) || mouse_buttons.pressed(MouseButton::Middle);
    let pointer = camera_pointer_input(pointer_captured, dragging, mouse_motion.delta, wheel_delta);
    let keyboard_delta = if !keyboard_captured && keys.pressed(KeyCode::Minus) {
        1.0
    } else if !keyboard_captured && keys.pressed(KeyCode::Equal) {
        -1.0
    } else {
        0.0
    };
    transform.translation +=
        (direction.normalize_or_zero() * 420.0 * time.delta_secs()).extend(0.0);
    transform.translation +=
        Vec2::new(-pointer.drag_delta.x, pointer.drag_delta.y).extend(0.0) * orthographic.scale;
    let zoom = pointer
        .wheel_delta
        .mul_add(-0.12, keyboard_delta * time.delta_secs())
        .exp();
    orthographic.scale = (orthographic.scale * zoom).clamp(0.25, 5.0);
}

/// Pointer input accepted by the scene camera after HUD capture filtering.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct CameraPointerInput {
    /// Screen-space drag delta for hand-style panning.
    drag_delta: Vec2,

    /// Mouse-wheel zoom delta.
    wheel_delta: f32,
}

/// Filter scene pointer input when egui owns the pointer.
fn camera_pointer_input(
    hud_captured: bool,
    dragging: bool,
    motion: Vec2,
    wheel_delta: f32,
) -> CameraPointerInput {
    if hud_captured {
        return CameraPointerInput::default();
    }
    CameraPointerInput {
        drag_delta: if dragging { motion } else { Vec2::ZERO },
        wheel_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Status bars must reserve one stable segment for every configured point.
    #[test]
    fn status_segments_preserve_configured_capacity_and_fill() {
        assert_eq!(status_segments(3, 5), [true, true, true, false, false]);
        assert_eq!(status_segments(8, 5), [true, true, true, true, true]);
    }

    /// Living and dead faces must use distinct readable eye geometry.
    #[test]
    fn eye_geometry_switches_from_circles_to_two_stroke_crosses() {
        assert_eq!(agent_eye_geometry(true).len(), 2);
        assert!(agent_eye_geometry(true)
            .iter()
            .all(|glyph| matches!(glyph, EyeGlyph::Circle { .. })));
        assert_eq!(agent_eye_geometry(false).len(), 4);
        assert!(agent_eye_geometry(false)
            .iter()
            .all(|glyph| matches!(glyph, EyeGlyph::Stroke { .. })));
    }

    /// Playback requires a checkpoint and rejects unsafe timing values.
    #[test]
    fn watch_options_validate_checkpoint_and_speed() {
        // Cover the required checkpoint plus one valid and one invalid speed.
        assert!(parse_watch_options(Vec::<String>::new().into_iter()).is_err());
        assert!(parse_watch_options(
            ["--checkpoint", "best.mpk", "--speed", "0"]
                .map(str::to_owned)
                .into_iter()
        )
        .is_err());
        let parsed = parse_watch_options(
            ["--checkpoint", "best.mpk", "--seed", "7", "--speed", "8"]
                .map(str::to_owned)
                .into_iter(),
        )
        .expect("valid watch arguments parse");
        assert_eq!(parsed.seed, 7);
        assert_eq!(parsed.speed, 8.0);
        let defaults =
            parse_watch_options(["--checkpoint", "best.mpk"].map(str::to_owned).into_iter())
                .expect("default watch arguments parse");
        assert_eq!(defaults.speed, 1.0);
        for value in ["1", "301"] {
            assert!(parse_watch_options(
                ["--checkpoint", "best.mpk", "--episode-seconds", value]
                    .map(str::to_owned)
                    .into_iter(),
            )
            .is_err());
        }
    }

    /// HUD pointer capture must block wheel zoom and drag panning together.
    #[test]
    fn hud_capture_blocks_all_scene_pointer_input() {
        let motion = Vec2::new(12.0, -4.0);
        assert_eq!(
            camera_pointer_input(true, true, motion, 3.0),
            CameraPointerInput::default()
        );
        assert_eq!(
            camera_pointer_input(false, true, motion, 3.0),
            CameraPointerInput {
                drag_delta: motion,
                wheel_delta: 3.0,
            }
        );
        assert_eq!(
            camera_pointer_input(false, false, motion, 0.0).drag_delta,
            Vec2::ZERO
        );
    }

    /// A sparse zero baseline has no defined percentage gain.
    #[test]
    fn zero_reward_baseline_formats_absolute_gain_only() {
        assert_eq!(format_reward_gain(69.869, 0.0), "+69.87 (n/a)");
        assert_eq!(format_reward_gain(15.0, 10.0), "+5.00 (+50.0%)");
    }

    /// The return graph must retain one point for every collected episode.
    #[test]
    fn episode_return_curve_keeps_each_episode_in_order() {
        let mut curve = vec![[1.0, 4.0]];

        append_episode_returns(&mut curve, &[8.0, -2.0, 5.5]);

        assert_eq!(curve, [[1.0, 4.0], [2.0, 8.0], [3.0, -2.0], [4.0, 5.5]]);
    }

    /// Nine environment origins must form a centered row-major 3x3 grid.
    #[test]
    fn survival_batch_layout_is_a_centered_three_by_three_grid() {
        let offsets = (0..SURVIVAL_BATCH_ENVIRONMENTS)
            .map(|lane| batch_grid_offset(lane, 100.0))
            .collect::<Vec<_>>();

        assert_eq!(
            offsets,
            [
                Vec2::new(-100.0, 100.0),
                Vec2::new(0.0, 100.0),
                Vec2::new(100.0, 100.0),
                Vec2::new(-100.0, 0.0),
                Vec2::ZERO,
                Vec2::new(100.0, 0.0),
                Vec2::new(-100.0, -100.0),
                Vec2::new(0.0, -100.0),
                Vec2::new(100.0, -100.0),
            ]
        );
    }

    /// The visible grid must keep moving until a newer optimizer batch replaces it.
    #[test]
    fn completed_batch_loops_without_a_pending_training_batch() {
        assert!(should_loop_batch(
            SurvivalBatchPurpose::InitializationPreview,
            false
        ));
        assert!(!should_loop_batch(
            SurvivalBatchPurpose::InitializationPreview,
            true
        ));
        assert!(should_loop_batch(
            SurvivalBatchPurpose::TrainingUpdate { iteration: 1 },
            false
        ));
        assert!(!should_loop_batch(
            SurvivalBatchPurpose::TrainingUpdate { iteration: 1 },
            true
        ));
    }

    /// Presentation interpolation must fill the gap between fixed physics poses.
    #[test]
    fn agent_pose_interpolation_uses_shortest_heading_arc() {
        let (position, heading) = interpolate_agent_pose(
            Vec2::new(0.0, 2.0),
            170.0_f32.to_radians(),
            Vec2::new(10.0, 6.0),
            (-170.0_f32).to_radians(),
            0.5,
        );

        assert_eq!(position, Vec2::new(5.0, 4.0));
        assert!((heading.abs() - std::f32::consts::PI).abs() < 1.0e-5);
    }

    /// A terminal frame remains visible for one wall-clock second.
    #[test]
    fn episode_pause_counts_down_in_wall_clock_time() {
        let mut state = PlaybackState::EpisodePause {
            remaining_seconds: EPISODE_PAUSE_SECONDS,
        };

        state.advance_episode_pause(0.4);
        assert!(matches!(state, PlaybackState::EpisodePause { .. }));
        state.advance_episode_pause(0.6);
        assert_eq!(state, PlaybackState::ResetReady);
    }

    /// Reward spikes retain their sign and expire after the fade duration.
    #[test]
    fn reward_popup_tracks_signed_delta_and_fades() {
        let mut popup = RewardPopup::new(-0.25);

        assert_eq!(popup.value, -0.25);
        assert_eq!(popup.alpha(), 1.0);
        popup.advance(REWARD_POPUP_SECONDS * 0.5);
        assert!((popup.alpha() - 0.5).abs() < 1.0e-6);
        popup.advance(REWARD_POPUP_SECONDS * 0.5);
        assert!(popup.is_expired());
    }

    /// Movement does not create reward labels, while physiology changes do.
    #[test]
    fn reward_popups_are_limited_to_physiology_events() {
        // Reuse one complete visual agent so the comparison varies one concern.
        let baseline = VisualAgent {
            id: AgentId(0),
            species: Species::Bunny,
            position: [0.0, 0.0],
            heading: 0.0,
            gaze_yaw: 0.0,
            is_alive: true,
            attack_active: false,
            satiation: 2,
            hydration: 2,
            hit_points: 5,
            exposure: 5,
            in_shelter: false,
        };
        let movement = VisualAgent {
            position: [1.0, 0.0],
            ..baseline
        };
        let drinking = VisualAgent {
            hydration: 3,
            ..movement
        };

        assert!(!reward_event_changed(&baseline, &movement));
        assert!(reward_event_changed(&movement, &drinking));
    }

    /// Collision overlays must use the exact agent collider sizes and heading.
    #[test]
    fn collision_overlay_matches_agent_hitbox_and_hurtbox_geometry() {
        // A quarter turn exercises both the child offset rotation and the
        // renderer-space lane translation.
        let agent = VisualAgent {
            id: AgentId(0),
            species: Species::Bunny,
            position: [2.0, 3.0],
            heading: std::f32::consts::FRAC_PI_2,
            gaze_yaw: 0.0,
            is_alive: true,
            attack_active: true,
            satiation: 2,
            hydration: 2,
            hit_points: 5,
            exposure: 5,
            in_shelter: false,
        };

        let [hurtbox, hitbox] = agent_collision_shapes(&agent, Vec2::new(5.0, -5.0))
            .expect("a living agent has collision shapes");
        assert_eq!(
            hurtbox,
            DebugCollisionShape::HurtboxEllipse {
                center: Vec2::new(45.0, 55.0),
                half_size: AGENT_SIZE * WORLD_SCALE * 0.5,
                rotation: std::f32::consts::FRAC_PI_2,
            }
        );
        let DebugCollisionShape::HitboxCircle {
            center,
            radius,
            is_active,
        } = hitbox
        else {
            panic!("second agent shape must be a hitbox circle");
        };
        assert!((center - Vec2::new(45.0, 73.0)).length() < 1.0e-5);
        assert_eq!(radius, INTERACTION_HITBOX_RADIUS * WORLD_SCALE);
        assert!(is_active);
    }

    /// Object overlays must include interaction sensors and exclude passive geometry.
    #[test]
    fn collision_overlay_includes_only_object_hurtboxes() {
        // The well proves sensor radius selection while the tree proves that
        // unrelated solid colliders stay outside this role overlay.
        let well = VisualObject {
            kind: VisualObjectKind::Well,
            position: [1.0, -2.0],
            radius: 1.25,
        };
        let tree = VisualObject {
            kind: VisualObjectKind::Tree,
            position: [0.0, 0.0],
            radius: 1.4,
        };

        assert_eq!(
            object_collision_shape(&well, Vec2::new(2.0, 3.0)),
            Some(DebugCollisionShape::HurtboxCircle {
                center: Vec2::new(22.0, -37.0),
                radius: WELL_SENSOR_RADIUS * WORLD_SCALE,
            })
        );
        assert_eq!(object_collision_shape(&tree, Vec2::ZERO), None);
    }
}
