//! Visual-first ecosystem training orchestration.

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::domain::{
    CurriculumStage, ExperimentTuning, SimulationConfig, SURVIVAL_BATCH_ENVIRONMENTS,
};
use super::rendering;
use super::survival_batch::SurvivalBatchTrace;
use super::training::{self, TrainerActivity, TrainingProgress};

/// Safe bounded settings for a shareable local training demonstration.
#[derive(Debug, Clone, PartialEq)]
struct DemoOptions {
    /// PPO updates before the demo reports completion.
    iterations: usize,

    /// Independent ecosystems pooled into one PPO update.
    rollout_episodes: usize,

    /// Simulated seconds allowed in one episode.
    episode_seconds: u16,

    /// Fixed-seed episodes used for each visible evaluation point.
    eval_episodes: usize,

    /// Root for every deterministic random stream.
    seed: u64,

    /// Initial simulation playback rate.
    playback_speed: f32,

    /// Unique artifact directory leaf.
    run_id: String,
}

impl DemoOptions {
    /// Build defaults for visual training with a responsive renderer.
    fn for_stage(stage: CurriculumStage) -> Result<Self, Box<dyn Error>> {
        // Nanoseconds keep repeated launches collision-safe without user input.
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let run_id = format!(
            "demo-{}-{}-{:03}",
            stage.as_key(),
            now.as_secs(),
            now.subsec_millis()
        );
        Ok(Self {
            iterations: if stage == CurriculumStage::Survival {
                16
            } else {
                8
            },
            rollout_episodes: if stage == CurriculumStage::Survival {
                SURVIVAL_BATCH_ENVIRONMENTS
            } else {
                2
            },
            episode_seconds: if stage == CurriculumStage::Survival {
                120
            } else {
                20
            },
            eval_episodes: if stage == CurriculumStage::Survival {
                16
            } else {
                4
            },
            seed: if stage == CurriculumStage::Survival {
                157
            } else {
                42
            },
            playback_speed: 1.0,
            run_id,
        })
    }

    /// Convert visual settings into the existing headless trainer boundary.
    fn training_arguments(&self) -> Vec<String> {
        // Raw returns still graph every episode. Fixed-seed selection runs less
        // often because survival qualification evaluates at least 100 worlds.
        [
            ("--iterations", self.iterations.to_string()),
            ("--rollout-episodes", self.rollout_episodes.to_string()),
            ("--episode-seconds", self.episode_seconds.to_string()),
            ("--eval-episodes", self.eval_episodes.to_string()),
            ("--eval-interval", "4".to_owned()),
            ("--seed", self.seed.to_string()),
            ("--run-id", self.run_id.clone()),
        ]
        .into_iter()
        .flat_map(|(flag, value)| [flag.to_owned(), value])
        .collect()
    }
}

/// Messages crossing from the Burn trainer to the Bevy render thread.
#[derive(Debug)]
pub(super) enum DemoTrainingEvent {
    /// Current work performed by the background trainer.
    Activity(TrainerActivity),

    /// Current inference policies and their training metrics.
    Progress(Box<TrainingProgress>),

    /// Exact trajectories from the nine environments used by one update.
    BatchTrace(Box<SurvivalBatchTrace>),

    /// Training ended and left its durable run directory.
    Finished,

    /// Training stopped because a recoverable error reached the boundary.
    Failed(String),
}

/// Shared pause and shutdown state for the background trainer.
#[derive(Debug)]
pub(super) struct DemoTrainingControl {
    /// Pause between complete optimizer iterations.
    pub(super) is_paused: AtomicBool,

    /// Stop after the current complete optimizer iteration.
    pub(super) should_stop: AtomicBool,

    /// Latest validated settings requested by the Inspector-egui controls.
    tuning: Mutex<ExperimentTuning>,
}

impl DemoTrainingControl {
    /// Construct controls with the command-line horizon as the initial profile.
    const fn new(tuning: ExperimentTuning) -> Self {
        Self {
            is_paused: AtomicBool::new(false),
            should_stop: AtomicBool::new(false),
            tuning: Mutex::new(tuning),
        }
    }

    /// Copy the latest requested profile at a safe trainer boundary.
    pub(super) fn tuning(&self) -> ExperimentTuning {
        // Recover poisoned state because the profile is a copy-only value and
        // retaining the latest sliders is safer than stopping the render loop.
        *self
            .tuning
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Replace the requested profile after bounded UI validation.
    pub(super) fn replace_tuning(&self, tuning: ExperimentTuning) {
        let mut current = self
            .tuning
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *current = tuning;
    }
}

/// Start the visual demo and train policies away from Bevy's render thread.
pub(super) fn run_demo(
    stage: CurriculumStage,
    arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let options = parse_options(stage, arguments)?;
    let playback_speed = options.playback_speed;
    let training_arguments = options.training_arguments();
    let tuning = demo_tuning(stage, options.episode_seconds)?;
    let control = Arc::new(DemoTrainingControl::new(tuning));
    let worker_control = Arc::clone(&control);
    let tuning_control = Arc::clone(&control);
    let (sender, receiver) = mpsc::channel();

    // Burn training owns a worker thread so rendering, controls, and the HUD
    // remain responsive while an optimizer update is in progress.
    std::thread::Builder::new()
        .name(format!("ecosystem-{}-trainer", stage.as_key()))
        .spawn(move || {
            let progress_sender = sender.clone();
            let batch_sender = sender.clone();
            let activity_sender = sender.clone();
            let result = training::run_training_with_progress(
                stage,
                training_arguments.into_iter(),
                || tuning_control.tuning(),
                |activity| {
                    activity_sender
                        .send(DemoTrainingEvent::Activity(activity))
                        .is_ok()
                },
                |progress| {
                    if progress_sender
                        .send(DemoTrainingEvent::Progress(Box::new(progress)))
                        .is_err()
                    {
                        return false;
                    }
                    while worker_control.is_paused.load(Ordering::Relaxed)
                        && !worker_control.should_stop.load(Ordering::Relaxed)
                    {
                        std::thread::park_timeout(Duration::from_millis(50));
                    }
                    !worker_control.should_stop.load(Ordering::Relaxed)
                },
                |trace| {
                    batch_sender
                        .send(DemoTrainingEvent::BatchTrace(Box::new(trace)))
                        .is_ok()
                },
            );
            let event = match result {
                Ok(()) => DemoTrainingEvent::Finished,
                Err(error) => DemoTrainingEvent::Failed(error.to_string()),
            };
            let _send_result = sender.send(event);
        })?;

    let (initial, initial_batch) = receive_demo_startup(stage, &receiver)?;
    rendering::run_training_demo(
        stage,
        initial,
        initial_batch,
        receiver,
        control,
        playback_speed,
    )
}

/// Preserve stage-specific dynamics when constructing the live HUD profile.
fn demo_tuning(
    stage: CurriculumStage,
    episode_seconds: u16,
) -> Result<ExperimentTuning, Box<dyn Error>> {
    let mut tuning = SimulationConfig::for_stage(stage)?.experiment_tuning();
    tuning.episode_seconds = episode_seconds;
    Ok(tuning)
}

/// Return whether the viewer has every stage-specific startup artifact.
fn startup_is_ready(stage: CurriculumStage, has_progress: bool, has_batch: bool) -> bool {
    has_progress && (stage != CurriculumStage::Survival || has_batch)
}

/// Receive startup state without ever presenting a single survival world.
fn receive_demo_startup(
    stage: CurriculumStage,
    receiver: &mpsc::Receiver<DemoTrainingEvent>,
) -> Result<(TrainingProgress, Option<SurvivalBatchTrace>), Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut initial = None;
    let mut initial_batch = None;
    // Survival waits for both messages so its first rendered frame is a grid.
    while !startup_is_ready(stage, initial.is_some(), initial_batch.is_some()) {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or("visual trainer startup timed out")?;
        let event = receiver.recv_timeout(remaining)?;
        match event {
            DemoTrainingEvent::Activity(_) => {}
            DemoTrainingEvent::Progress(progress) => initial = Some(*progress),
            DemoTrainingEvent::BatchTrace(trace) => initial_batch = Some(*trace),
            DemoTrainingEvent::Failed(message) => return Err(message.into()),
            DemoTrainingEvent::Finished => {
                return Err("visual trainer finished before publishing startup state".into());
            }
        }
    }
    Ok((
        initial.ok_or("visual trainer did not publish its initial policy")?,
        initial_batch,
    ))
}

/// Parse the deliberately small visual-demo command surface.
fn parse_options(
    stage: CurriculumStage,
    mut arguments: impl Iterator<Item = String>,
) -> Result<DemoOptions, Box<dyn Error>> {
    let mut options = DemoOptions::for_stage(stage)?;
    // Every demo option takes one value to keep the shared command predictable.
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--iterations" => options.iterations = value.parse()?,
            "--rollout-episodes" => options.rollout_episodes = value.parse()?,
            "--episode-seconds" => options.episode_seconds = value.parse()?,
            "--eval-episodes" => options.eval_episodes = value.parse()?,
            "--seed" => options.seed = value.parse()?,
            "--speed" => options.playback_speed = value.parse()?,
            "--run-id" => options.run_id = value,
            _ => return Err(format!("unknown demo option {flag:?}").into()),
        }
    }
    let minimum_rollout_episodes = if stage == CurriculumStage::Survival {
        SURVIVAL_BATCH_ENVIRONMENTS
    } else {
        1
    };
    if options.iterations == 0
        || options.rollout_episodes < minimum_rollout_episodes
        || !(5..=300).contains(&options.episode_seconds)
        || options.eval_episodes == 0
        || !options.playback_speed.is_finite()
        || options.playback_speed <= 0.0
    {
        return Err(
            "demo counts and playback speed must be positive; survival requires at least 9 rollout environments; --episode-seconds must be in 5..=300"
                .into(),
        );
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Demo overrides stay bounded and preserve the generated artifact id.
    #[test]
    fn demo_options_accept_safe_runtime_overrides() {
        // Exercise the public-facing knobs without depending on the generated id.
        let options = parse_options(
            CurriculumStage::Survival,
            ["--iterations", "3", "--speed", "16", "--run-id", "visual"]
                .map(str::to_owned)
                .into_iter(),
        )
        .expect("valid demo controls parse");
        assert_eq!(options.iterations, 3);
        assert_eq!(options.playback_speed, 16.0);
        assert_eq!(options.run_id, "visual");
    }

    /// Visual playback starts in real time unless the command overrides it.
    #[test]
    fn demo_playback_defaults_to_realtime() {
        let options = DemoOptions::for_stage(CurriculumStage::Survival)
            .expect("survival demo defaults are valid");

        assert_eq!(options.playback_speed, 1.0);
    }

    /// No-argument survival must use the headless-verified learning profile.
    #[test]
    fn survival_demo_defaults_match_the_verified_learning_profile() {
        // Keep visual training identical to the verified headless command.
        let options = DemoOptions::for_stage(CurriculumStage::Survival)
            .expect("survival demo defaults are valid");

        assert_eq!(options.iterations, 16);
        assert_eq!(options.rollout_episodes, SURVIVAL_BATCH_ENVIRONMENTS);
        assert_eq!(options.episode_seconds, 120);
        assert_eq!(options.eval_episodes, 16);
        assert_eq!(options.seed, 157);

        let arguments = options.training_arguments();
        let interval = arguments
            .windows(2)
            .find(|pair| pair[0] == "--eval-interval")
            .map(|pair| pair[1].as_str());
        assert_eq!(interval, Some("4"));
    }

    /// Visual training must preserve survival-specific physiology defaults.
    #[test]
    fn survival_demo_tuning_matches_the_survival_environment() {
        let tuning =
            demo_tuning(CurriculumStage::Survival, 37).expect("survival demo tuning is valid");

        assert_eq!(tuning.initial_satiation, 3);
        assert_eq!(tuning.initial_hydration, 3);
        assert_eq!(tuning.need_loss_interval_seconds, 60);
        assert_eq!(tuning.starvation_damage_interval_seconds, 10);
        assert_eq!(tuning.dehydration_damage_interval_seconds, 10);
        assert_eq!(tuning.episode_seconds, 37);
    }

    /// Visual survival training must retain the complete 3x3 environment batch.
    #[test]
    fn survival_demo_rejects_fewer_than_nine_training_environments() {
        assert!(parse_options(
            CurriculumStage::Survival,
            ["--rollout-episodes", "8"].map(str::to_owned).into_iter(),
        )
        .is_err());
        assert!(parse_options(
            CurriculumStage::Competition,
            ["--rollout-episodes", "8"].map(str::to_owned).into_iter(),
        )
        .is_ok());
    }

    /// Survival must not launch its viewer until a complete 3x3 trace is ready.
    #[test]
    fn survival_demo_startup_requires_nine_environment_trace() {
        assert!(!startup_is_ready(CurriculumStage::Survival, true, false));
        assert!(startup_is_ready(CurriculumStage::Survival, true, true));
        assert!(startup_is_ready(CurriculumStage::Competition, true, false));
    }

    /// Zero work cannot masquerade as a training demonstration.
    #[test]
    fn demo_options_reject_zero_iterations() {
        assert!(parse_options(
            CurriculumStage::Survival,
            ["--iterations", "0"].map(str::to_owned).into_iter()
        )
        .is_err());
    }

    /// Demo horizons use the same bounds as live Inspector tuning.
    #[test]
    fn demo_options_reject_out_of_range_episode_seconds() {
        for value in ["1", "301"] {
            assert!(parse_options(
                CurriculumStage::Survival,
                ["--episode-seconds", value].map(str::to_owned).into_iter(),
            )
            .is_err());
        }
    }

    /// The render thread must publish one complete tuning profile atomically.
    #[test]
    fn training_control_retains_latest_tuning_profile() {
        // The worker must observe one whole replacement profile, never a mix
        // of fields from separate UI frames.
        let control = DemoTrainingControl::new(ExperimentTuning::default());
        let changed = ExperimentTuning {
            dehydration_damage_interval_seconds: 4,
            ..ExperimentTuning::default()
        };

        control.replace_tuning(changed);

        assert_eq!(control.tuning(), changed);
    }
}
