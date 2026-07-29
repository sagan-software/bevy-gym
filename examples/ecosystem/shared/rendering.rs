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
    LocomotionAction, PerceptKind, PerceptionRayCount, SimulationConfig, Species, VisualObjectKind,
    VisualWorldSnapshot, GLOBAL_STATE_SIZE, LOCAL_OBSERVATION_SIZE, MAX_AGENTS,
};
use super::simulation::{Ecosystem, AGENT_SIZE};
use super::training::{
    checkpoint_experiment_tuning, ecosystem_algorithm, resolve_bunny_checkpoint,
    sibling_fox_checkpoint, TrainingProgress,
};

/// Simulation units to rendered world units.
const WORLD_SCALE: f32 = 20.0;

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

    /// Per-episode horizon.
    max_steps: Option<u32>,
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

    /// Bunny batch-return curve in iteration order.
    bunny_train_curve: Vec<[f64; 2]>,

    /// Bunny fixed-seed return curve in iteration order.
    bunny_eval_curve: Vec<[f64; 2]>,

    /// Fox fixed-seed return curve when predators exist.
    fox_eval_curve: Vec<[f64; 2]>,

    /// Change in bunny fixed-seed survival per optimizer iteration.
    learning_velocity_curve: Vec<[f64; 2]>,

    /// Effective actor learning-rate curve.
    actor_learning_rate_curve: Vec<[f64; 2]>,

    /// Effective critic learning-rate curve.
    critic_learning_rate_curve: Vec<[f64; 2]>,

    /// Iterations whose rollout profile differs from the preceding point.
    tuning_markers: Vec<f64>,

    /// Current worker lifecycle.
    status: TrainingStatus,

    /// Wall-clock origin for the visible speed claim.
    started_at: Instant,
}

impl DemoDashboard {
    /// Seed the dashboard with the random-policy evaluation baseline.
    fn new(
        progress: TrainingProgress,
        receiver: mpsc::Receiver<DemoTrainingEvent>,
        control: Arc<DemoTrainingControl>,
    ) -> Self {
        // Start with empty curves because the first event may precede evaluation.
        let mut dashboard = Self {
            receiver,
            control,
            progress,
            bunny_train_curve: Vec::new(),
            bunny_eval_curve: Vec::new(),
            fox_eval_curve: Vec::new(),
            learning_velocity_curve: Vec::new(),
            actor_learning_rate_curve: Vec::new(),
            critic_learning_rate_curve: Vec::new(),
            tuning_markers: Vec::new(),
            status: TrainingStatus::Running,
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
        let iteration = self.progress.iteration as f64;
        if let Some(previous) = self.bunny_eval_curve.last() {
            let iteration_delta = (iteration - previous[0]).max(1.0);
            let survival_delta =
                (f64::from(self.progress.bunny_eval_return) - previous[1]) / iteration_delta;
            self.learning_velocity_curve
                .push([iteration, survival_delta]);
        }
        self.bunny_eval_curve
            .push([iteration, f64::from(self.progress.bunny_eval_return)]);
        if self.progress.fox_eval_return > 0.0 {
            self.fox_eval_curve
                .push([iteration, f64::from(self.progress.fox_eval_return)]);
        }
    }

    /// Retain optimizer and rollout values for every completed update.
    fn record_update_point(&mut self) {
        let iteration = self.progress.iteration as f64;
        if self.progress.iteration > 0 {
            self.bunny_train_curve
                .push([iteration, f64::from(self.progress.bunny_train_return)]);
        }
        self.actor_learning_rate_curve
            .push([iteration, self.progress.actor_learning_rate]);
        self.critic_learning_rate_curve
            .push([iteration, self.progress.critic_learning_rate]);
    }
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

    /// Whether spacebar has paused policy playback.
    is_paused: bool,

    /// Whether exact actor perception sectors are drawn over the world.
    show_perception_rays: bool,

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
}

/// Marker for scene entities recreated from each read-only snapshot.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SceneVisual;

/// Marker for the movable top-down camera.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WorldCamera;

/// Launch a top-down deterministic checkpoint viewer.
pub(super) fn run_watch(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    // Watch mode resolves one immutable checkpoint before creating a window.
    let options = parse_watch_options(arguments)?;
    let bunny_path = resolve_bunny_checkpoint(&options.checkpoint);
    let config = {
        let mut config = SimulationConfig::for_stage(stage)?;
        if let Some(tuning) = checkpoint_experiment_tuning(&bunny_path)? {
            config.apply_experiment_tuning(tuning)?;
        }
        if let Some(max_steps) = options.max_steps {
            config.max_steps = max_steps;
        }
        config
    };
    let algorithm = ecosystem_algorithm();
    let bunny = load_policy(&bunny_path, &algorithm)?;
    let fox = if config.fox_count > 0 {
        let path = options
            .fox_checkpoint
            .unwrap_or_else(|| sibling_fox_checkpoint(&bunny_path));
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
    let mut session = WatchSession::new(stage, config, bunny, fox, 101, playback_speed, label)?;
    session.dashboard = Some(DemoDashboard::new(initial, receiver, control));
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
            )
                .chain(),
        )
        .add_systems(
            PostUpdate,
            camera_controls.after(EguiPostUpdateSet::ProcessOutput),
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
        &[-1.0, -1.0, -1.0],
        &[1.0, 1.0, 1.0],
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
    let mut playback_speed: f32 = 4.0;
    let mut max_steps = None;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--fox-checkpoint" => fox_checkpoint = Some(PathBuf::from(value)),
            "--seed" => environment_seed = value.parse()?,
            "--speed" => playback_speed = value.parse()?,
            "--max-steps" => max_steps = Some(value.parse()?),
            _ => return Err(format!("unknown watch option {flag:?}").into()),
        }
    }
    if !playback_speed.is_finite() || playback_speed <= 0.0 || max_steps == Some(0) {
        return Err("--speed must be finite and positive; --max-steps must exceed zero".into());
    }
    Ok(WatchOptions {
        checkpoint: checkpoint.ok_or("watch requires --checkpoint <run-dir-or-mpk>")?,
        fox_checkpoint,
        seed: environment_seed,
        speed: playback_speed,
        max_steps,
    })
}

/// Spawn the camera used by both checkpoint playback and visual training.
pub(super) fn setup_viewer(
    mut commands: Commands<'_, '_>,
    mut session: NonSendMut<'_, WatchSession>,
) {
    // Fit the complete fixed map before the user applies pan or zoom controls.
    let snapshot = session.ecosystem.visual_snapshot();
    let initial_scale = (snapshot.map_half_extent * WORLD_SCALE * 2.0 / 650.0).max(0.7);
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
}

/// Pause, resume, or restart the deterministic evaluation episode.
fn toggle_playback(keys: Res<'_, ButtonInput<KeyCode>>, mut session: NonSendMut<'_, WatchSession>) {
    // Keyboard shortcuts mirror the visible Inspector-egui controls.
    if keys.just_pressed(KeyCode::Space) {
        session.is_paused = !session.is_paused;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        let reset_failed = session.reset_episode().is_err();
        if reset_failed {
            session.is_paused = true;
        }
    }
}

/// Advance the policy in exact fixed simulation increments.
fn advance_policy(time: Res<'_, Time>, mut session: NonSendMut<'_, WatchSession>) {
    // Accumulation preserves fixed simulation time at every playback speed.
    if session.is_paused {
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
                session.is_paused = true;
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
                    let marker = progress.iteration as f64;
                    if dashboard.tuning_markers.last().copied() != Some(marker) {
                        dashboard.tuning_markers.push(marker);
                    }
                    dashboard.bunny_eval_curve.clear();
                    dashboard.fox_eval_curve.clear();
                    dashboard.learning_velocity_curve.clear();
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
            is_paused: false,
            show_perception_rays: true,
            perception_ray_agent: Some(AgentId(0)),
            pending_tuning,
            checkpoint_label,
            dashboard: None,
            episode_return: 0.0,
            latest_reward: 0.0,
        })
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
            let [forward, turn, gaze] = output.action.as_slice() else {
                return Err("ecosystem policy must emit three action axes".into());
            };
            actions.push((*id, LocomotionAction::new(*forward, *turn, *gaze)?));
            next_memories.insert(*id, output.next_memory);
        }

        // Replace memory only for continuing trajectories; termination cannot
        // leak recurrent state into the next episode or another identity.
        let result = self.ecosystem.step(&actions)?;
        self.latest_reward = result.agents.iter().map(|agent| agent.reward).sum();
        self.episode_return += self.latest_reward;
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
        if result.is_done {
            self.reset_episode()?;
            Ok(true)
        } else {
            self.state.agents = next_agents;
            self.state.global_state = result.global_state;
            Ok(false)
        }
    }

    /// Create the next deterministic map and clear all actor memories.
    fn reset_episode(&mut self) -> Result<(), Box<dyn Error>> {
        // Each reset advances one deterministic seed without reusing LSTM state.
        self.episode = self.episode.saturating_add(1);
        let seed = self.seed.wrapping_add(self.episode);
        self.ecosystem = Ecosystem::new(self.config.clone(), seed)?;
        self.state = self.ecosystem.state();
        self.memories.clear();
        self.accumulator = 0.0;
        self.episode_return = 0.0;
        self.latest_reward = 0.0;
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
        self.memories.clear();
        self.accumulator = 0.0;
        self.is_paused = false;
        self.episode_return = 0.0;
        self.latest_reward = 0.0;
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
        self.episode = 0;
        self.ecosystem = Ecosystem::new(self.config.clone(), self.seed)?;
        self.state = self.ecosystem.state();
        self.memories.clear();
        self.accumulator = 0.0;
        self.is_paused = false;
        self.episode_return = 0.0;
        self.latest_reward = 0.0;
        Ok(())
    }
}

/// Replace transient scene primitives from the authoritative world snapshot.
pub(super) fn redraw_scene(
    mut commands: Commands<'_, '_>,
    mut session: NonSendMut<'_, WatchSession>,
    visuals: Query<'_, '_, Entity, With<SceneVisual>>,
) {
    // Scene primitives are presentation projections, never simulation entities.
    for entity in &visuals {
        commands.entity(entity).despawn();
    }
    let snapshot = session.ecosystem.visual_snapshot();
    spawn_snapshot(&mut commands, &snapshot, session.latest_reward > 0.0);
}

/// Draw the exact post-physics semantic sectors encoded for every living actor.
pub(super) fn draw_perception_rays(
    mut gizmos: Gizmos<'_, '_>,
    mut session: NonSendMut<'_, WatchSession>,
) {
    if !session.show_perception_rays {
        return;
    }
    // Hits stop at their sampled distance. Misses extend to the full sight
    // range, matching the observation that the sector contains no target.
    for ray in session.ecosystem.visual_snapshot().rays {
        if session
            .perception_ray_agent
            .is_some_and(|agent| agent != ray.agent)
        {
            continue;
        }
        let start = Vec2::from_array(ray.start) * WORLD_SCALE;
        let end = Vec2::from_array(ray.end) * WORLD_SCALE;
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
        Some(PerceptKind::Boundary) => Color::srgba(0.72, 0.52, 0.29, 0.9),
    }
}

/// Spawn programmer-art primitives for one complete world snapshot.
fn spawn_snapshot(
    commands: &mut Commands<'_, '_>,
    snapshot: &VisualWorldSnapshot,
    reward_is_active: bool,
) {
    let diameter = snapshot.map_half_extent * WORLD_SCALE * 2.0;
    commands.spawn((
        Sprite::from_color(Color::srgb_u8(40, 74, 43), Vec2::splat(diameter)),
        Transform::from_xyz(0.0, 0.0, -2.0),
        SceneVisual,
    ));
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
            Transform::from_translation(position.extend(-1.0)),
            SceneVisual,
        ));
    }

    for object in &snapshot.objects {
        let (color, depth) = match object.kind {
            VisualObjectKind::Food => (Color::srgb_u8(244, 211, 94), 1.0),
            VisualObjectKind::Well => (Color::srgb_u8(54, 162, 235), 0.5),
            VisualObjectKind::Tree => (Color::srgb_u8(25, 94, 55), 0.2),
            VisualObjectKind::Rock => (Color::srgb_u8(123, 130, 137), 0.2),
            VisualObjectKind::Thorn => (Color::srgb_u8(191, 64, 128), 0.3),
        };
        commands.spawn((
            Sprite::from_color(color, Vec2::splat(object.radius * WORLD_SCALE * 2.0)),
            Transform::from_xyz(
                object.position[0] * WORLD_SCALE,
                object.position[1] * WORLD_SCALE,
                depth,
            ),
            SceneVisual,
        ));
    }

    for agent in &snapshot.agents {
        if agent.is_alive && reward_is_active {
            // A green backing pulse makes the exact per-step survival reward
            // visible without implying that food or water is shaping reward.
            commands.spawn((
                Sprite::from_color(Color::srgba(0.3, 1.0, 0.4, 0.32), Vec2::new(34.0, 28.0)),
                Transform::from_xyz(
                    agent.position[0] * WORLD_SCALE,
                    agent.position[1] * WORLD_SCALE,
                    1.8,
                )
                .with_rotation(Quat::from_rotation_z(agent.heading)),
                SceneVisual,
            ));
        }
        let color = if agent.is_alive {
            match agent.species {
                Species::Bunny => Color::srgb_u8(238, 238, 232),
                Species::Fox => Color::srgb_u8(229, 111, 47),
            }
        } else {
            Color::srgb_u8(70, 70, 70)
        };
        commands.spawn((
            Sprite::from_color(color, AGENT_SIZE * WORLD_SCALE),
            Transform::from_xyz(
                agent.position[0] * WORLD_SCALE,
                agent.position[1] * WORLD_SCALE,
                2.0,
            )
            .with_rotation(Quat::from_rotation_z(agent.heading)),
            SceneVisual,
        ));
    }
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
    let snapshot = session.ecosystem.visual_snapshot();
    let episode_metrics = session.ecosystem.episode_metrics();

    egui::Window::new("Ecosystem learning HUD")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .default_width(390.0)
        .max_height(690.0)
        .vscroll(true)
        .resizable(true)
        .show(egui_context.get_mut(), |ui| {
            ui.heading(session.stage.title());
            ui.label(format!(
                "Reward: {:.2}/s alive + up to {:.2}/food or prey + up to {:.2}/water unit",
                session.config.survival_reward_per_second,
                session.config.food_reward * 1.25,
                session.config.water_reward_per_unit * 1.25,
            ));
            ui.small(
                "Need multiplier: <=10% 125%, <=50% 100%, <=75% 90%, <90% 75%, >=90% 0%",
            );
            ui.separator();
            show_perception_controls(ui, &mut session, &snapshot);

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
    // Aggregate only living physiology because dead bodies are presentation data.
    let (count, hunger, thirst, hit_points) =
        snapshot.agents.iter().filter(|agent| agent.is_alive).fold(
            (0_usize, 0.0_f32, 0.0_f32, 0.0_f32),
            |(count, hunger, thirst, hit_points), agent| {
                (
                    count + 1,
                    hunger + agent.hunger,
                    thirst + agent.thirst,
                    hit_points + agent.hit_points,
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
            ui.label("Episode / time");
            ui.label(format!(
                "{} / {:.1} s",
                session.episode,
                snapshot.summary.step as f32 * session.config.time_step
            ));
            ui.end_row();
            ui.label("Alive");
            ui.label(format!("{count} / {}", snapshot.agents.len()));
            ui.end_row();
            ui.label("Episode reward");
            ui.colored_label(
                egui::Color32::LIGHT_GREEN,
                format!(
                    "{:.2}  ({:+.2})",
                    session.episode_return, session.latest_reward
                ),
            );
            ui.end_row();
            ui.label("Food / well");
            ui.label(format!(
                "{} / {:.1}",
                snapshot.summary.food_count, snapshot.summary.well_water
            ));
            ui.end_row();
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
            if overconsumption_damage > 0.0 {
                ui.label("Overfull damage");
                ui.label(format!("{overconsumption_damage:.1}"));
                ui.end_row();
            }
            ui.label("Hunger / thirst / HP");
            ui.label(format!(
                "{:.2} / {:.2} / {:.1}",
                hunger / divisor,
                thirst / divisor,
                hit_points / divisor
            ));
            ui.end_row();
        });
}

/// Draw the ray-overlay toggle and living-agent filter.
fn show_perception_controls(
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
    if !session.show_perception_rays {
        return;
    }
    ui.small("Dim lines are empty sectors; colored lines stop at the perceived hit.");
    ui.small("Hit colors: food yellow, water blue, agents white/orange, thorns magenta.");
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
            changed |= show_reward_tuning(ui, &mut tuning);
            changed |= show_dynamics_tuning(ui, &mut tuning, session.config.time_step);
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
        session.is_paused = true;
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
/// Draw reward shaping controls and concise learning tradeoffs.
fn show_reward_tuning(ui: &mut egui::Ui, tuning: &mut ExperimentTuning) -> bool {
    // Fold every slider response into one change flag before publishing the
    // complete validated profile to the worker.
    let mut changed = false;
    ui.label("Reward weights");
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.survival_reward_per_second, 0.0..=5.0)
                .text("Alive / second"),
        )
        .on_hover_text("Dense reward improves early credit but can favor passive survival.")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.food_reward, 0.0..=20.0).text("Food / prey"))
        .on_hover_text("Base bonus multiplied by hunger need; reward reaches zero at 90% reserve.")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.water_reward_per_unit, 0.0..=10.0).text("Water unit"))
        .on_hover_text(
            "Base bonus multiplied by thirst need; only absorbed water earns reward, and reward reaches zero at 90% reserve.",
        )
        .changed();
    changed
}

/// Draw reset physiology, need pressure, damage, and time-limit controls.
fn show_dynamics_tuning(ui: &mut egui::Ui, tuning: &mut ExperimentTuning, time_step: f32) -> bool {
    // Use bounded sliders because these values enter physics and episode
    // lifecycle calculations on the next safe boundary.
    let mut changed = false;
    ui.label("Episode dynamics");
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.initial_health_fraction, 0.1..=1.0)
                .text("Starting health"),
        )
        .on_hover_text("Lower health reduces exploration time before the first useful behavior.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.initial_reserve_fraction, 0.1..=1.0)
                .text("Starting food/water"),
        )
        .on_hover_text("Lower reserves shorten random-policy survival and create earlier urgency.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.need_drain_multiplier, 0.25..=4.0)
                .logarithmic(true)
                .text("Hunger/thirst speed"),
        )
        .on_hover_text("Faster drain creates urgency but makes resource discovery less forgiving.")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.movement_need_drain, 0.0..=3.0).text("Movement cost"))
        .on_hover_text("Translation drains extra food and water; body rotation and gaze stay free.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.movement_speed_multiplier, 0.5..=2.0)
                .text("Movement speed"),
        )
        .on_hover_text("Scales both forward acceleration and maximum translation speed.")
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut tuning.reserve_capacity, 1.05..=2.0).text("Reserve capacity"))
        .on_hover_text("Food and water can exceed comfortable fullness up to this cap.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.fullness_slow_threshold, 0.5..=1.0)
                .text("Slowdown starts"),
        )
        .on_hover_text("High food or water reserves begin reducing translation at this level.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.overfull_speed_multiplier, 0.1..=1.0)
                .text("Overfull speed"),
        )
        .on_hover_text("Translation uses this multiplier at comfortable fullness and above.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.overfull_damage_rate, 0.0..=50.0).text("Overfull damage"),
        )
        .on_hover_text("Excess above 100% causes up to this many hit points of damage per second.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.damage_multiplier, 0.25..=4.0)
                .logarithmic(true)
                .text("Health damage speed"),
        )
        .on_hover_text("Higher damage sharpens failure but increases return variance.")
        .changed();
    changed |= ui
        .add(
            egui::Slider::new(&mut tuning.episode_step_limit, 100..=3_000)
                .text("Episode step limit"),
        )
        .on_hover_text("Short limits produce updates sooner but truncate delayed consequences.")
        .changed();
    ui.small(format!(
        "Current timeout: {:.1} simulated seconds",
        tuning.episode_step_limit as f32 * time_step
    ));
    changed
}

/// Draw playback controls that only mutate the visual evaluation world.
fn show_world_controls(ui: &mut egui::Ui, session: &mut WatchSession) {
    // Reset failures pause playback instead of leaving controls unresponsive.
    ui.horizontal(|ui| {
        let playback_label = if session.is_paused {
            "Resume world"
        } else {
            "Pause world"
        };
        if ui.button(playback_label).clicked() {
            session.is_paused = !session.is_paused;
        }
        if ui.button("Reset episode").clicked() && session.reset_episode().is_err() {
            session.is_paused = true;
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

    show_training_controls(ui, dashboard);
    let progress = &dashboard.progress;

    if !progress.has_evaluation {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Evaluating the random-policy baseline...");
        });
        ui.small(format!("Artifacts: {}", progress.run_dir.display()));
        return;
    }

    show_training_metrics(ui, dashboard);
    ui.label("Mean survival time (comparable across reward changes)");
    draw_learning_graph(
        ui,
        &dashboard.bunny_train_curve,
        &dashboard.bunny_eval_curve,
        &dashboard.fox_eval_curve,
        &dashboard.tuning_markers,
    );
    ui.horizontal(|ui| {
        ui.colored_label(egui::Color32::from_rgb(244, 196, 74), "training");
        ui.colored_label(egui::Color32::from_rgb(95, 210, 125), "bunny evaluation");
        if !dashboard.fox_eval_curve.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(235, 120, 70), "fox evaluation");
        }
        if !dashboard.tuning_markers.is_empty() {
            ui.colored_label(egui::Color32::from_rgb(190, 125, 235), "settings changed");
        }
    });
    ui.label("Learning velocity (fixed-seed survival seconds per iteration)");
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

/// Draw typed return, gain, loss, entropy, and selection metrics.
fn show_training_metrics(ui: &mut egui::Ui, dashboard: &DemoDashboard) {
    // Compare current fixed-seed returns to the iteration-zero policy.
    let progress = &dashboard.progress;
    egui::Grid::new("ecosystem-training-metrics")
        .num_columns(2)
        .striped(true)
        .show(ui, |ui| {
            ui.label("Train survival");
            ui.label(format!("{:.2} s", progress.bunny_train_return));
            ui.end_row();
            if progress.fox_train_return > 0.0 {
                ui.label("Fox train survival");
                ui.label(format!("{:.2} s", progress.fox_train_return));
                ui.end_row();
            }
            ui.label("Fixed-seed survival");
            ui.label(format!("{:.2} s", progress.bunny_eval_return));
            ui.end_row();
            if let Some([_, baseline]) = dashboard.bunny_eval_curve.first() {
                let gain = f64::from(progress.bunny_eval_return) - baseline;
                let gain_percent = if *baseline > 0.0 {
                    gain / baseline * 100.0
                } else {
                    0.0
                };
                ui.label("Gain from start");
                ui.colored_label(
                    if gain >= 0.0 {
                        egui::Color32::LIGHT_GREEN
                    } else {
                        egui::Color32::LIGHT_RED
                    },
                    format!("{gain:+.2} s ({gain_percent:+.1}%)"),
                );
                ui.end_row();
            }
            if progress.fox_eval_return > 0.0 {
                ui.label("Fox fixed-seed survival");
                ui.label(format!("{:.2} s", progress.fox_eval_return));
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
}
