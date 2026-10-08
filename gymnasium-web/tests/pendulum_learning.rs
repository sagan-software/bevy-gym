//! Pendulum qualification selects checkpoints before inspecting held-out scores.
use bevy_gym::environments::{Pendulum, PendulumAction};
use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoPolicy};
use bevy_gym::{Env, TimeLimit};
use bevy_gym_browser::{Session, Task};
use serde::Serialize;
use std::path::Path;

/// Original return and correlated dwell samples from one complete episode.
#[derive(Debug, Serialize)]
struct EpisodeScore {
    /// Original return over 200 transitions.
    reward: f64,
    /// Upright states among transitions 50 through 199.
    upright_steps: usize,
}

/// Ordered evidence derives every aggregate from its episodes.
#[derive(Debug)]
struct Scores(Vec<EpisodeScore>);

impl Scores {
    /// Mean original return across independent resets.
    fn mean(&self) -> f64 {
        self.0.iter().map(|episode| episode.reward).sum::<f64>() / self.0.len() as f64
    }

    /// Upright states divided by 150 eligible states per episode.
    fn dwell_rate(&self) -> f64 {
        self.0
            .iter()
            .map(|episode| episode.upright_steps)
            .sum::<usize>() as f64
            / (self.0.len() as f64 * 150.0)
    }

    /// Both independent task requirements must pass.
    fn passes(&self) -> bool {
        passes(self.mean(), self.dwell_rate())
    }

    /// Serialize derived aggregates beside the complete episode evidence.
    fn report(&self) -> serde_json::Value {
        serde_json::json!({"mean":self.mean(),"dwell_rate":self.dwell_rate(),"episodes":self.0})
    }
}

/// Inclusive return and dwell thresholds are independent release requirements.
fn passes(mean: f64, dwell_rate: f64) -> bool {
    mean >= -200.0 && dwell_rate >= 0.70
}

/// Test normalized angle through cosine, retaining both signs and full turns.
fn upright(angle: f64, velocity: f64) -> bool {
    angle.cos() >= (std::f64::consts::PI / 12.0).cos() && velocity.abs() <= 1.0
}

/// Evaluate frozen mean actions without resetting before the final dwell sample.
fn evaluate(bytes: Vec<u8>, seeds: std::ops::Range<u64>) -> Scores {
    let config = RecurrentPpoConfig {
        actor_hidden_size: 32,
        critic_hidden_sizes: vec![64, 32],
        ..RecurrentPpoConfig::default()
    };
    let policy = RecurrentPpoPolicy::load_bytes(bytes, 3, 3, 1, &[-2.0], &[2.0], &config)
        .expect("Pendulum policy record");
    let mut episodes = Vec::new();
    for seed in seeds {
        let mut env = TimeLimit::new(Pendulum::default(), 200).expect("episode cap");
        let mut observation = env.reset(Some(seed)).observation;
        let mut memory = policy.initial_memory();
        let mut total = 0.0;
        let mut dwell = 0;
        for index in 0..200 {
            let encoded = [observation[0], observation[1], observation[2] / 8.0];
            let sampled = policy.mean_action(&encoded, &memory).expect("mean torque");
            memory = sampled.next_memory;
            let action = *sampled.action.first().expect("one torque coordinate");
            let result = env.step(PendulumAction::try_from(action).expect("finite torque"));
            observation = result.observation;
            total += result.reward;
            let [angle, velocity] = env.inner().state();
            if index >= 50 && upright(angle, velocity) {
                dwell += 1;
            }
            assert_eq!(result.is_done(), index == 199);
        }
        episodes.push(EpisodeScore {
            reward: total,
            upright_steps: dwell,
        });
    }
    Scores(episodes)
}

/// A native checkpoint chosen without reading the qualification test seeds.
struct Candidate {
    /// Independent initialization and training stream.
    seed: u64,
    /// Collected transitions at checkpoint selection, within the fixed budget.
    transitions: u64,
    /// Inference parameters from the selected native checkpoint.
    bytes: Vec<u8>,
    /// Separate common validation seeds used for final model selection.
    validation: Scores,
}

/// Independently trained recipes retain separate validation and test reports.
#[derive(Clone, Copy)]
enum NativeRecipe {
    /// Actor rate 0.003, declared before the original three runs.
    Original,
    /// Actor rate 0.0003 with the same budget and other settings.
    ReducedActorRate,
}

impl NativeRecipe {
    /// Version names are fixed local artifact paths, not user input.
    const fn version(self) -> &'static str {
        match self {
            Self::Original => "v1",
            Self::ReducedActorRate => "v2",
        }
    }
}

/// Require a completed native run and select only budget-eligible checkpoints.
async fn candidate(seed: u64, recipe: NativeRecipe) -> Candidate {
    let version = recipe.version();
    let run = format!("../runs/pendulum-ppo/pendulum-oracle-{version}-seed{seed}-durable");
    let root = Path::new(&run);
    tokio::fs::read(root.join("summary.json"))
        .await
        .expect("native training must finish before qualification");
    let metrics = tokio::fs::read_to_string(root.join("metrics.jsonl"))
        .await
        .expect("native validation history");
    let evaluations: Vec<(u64, f64)> = metrics
        .lines()
        .filter_map(|line| {
            let row: serde_json::Value = serde_json::from_str(line).expect("metric JSON");
            Some((
                row.get("global_step")?.as_u64()?,
                row.get("eval/mean_reward")?.as_f64()?,
            ))
        })
        .filter(|(step, _mean)| *step <= 2_000_000)
        .collect();
    let (transitions, _mean) = evaluations
        .iter()
        .copied()
        .max_by(|left, right| left.1.total_cmp(&right.1).then(right.0.cmp(&left.0)))
        .expect("initial and validation checkpoints");
    let initial = evaluations.first().expect("initial validation").1;
    assert!(evaluations
        .iter()
        .any(|(step, mean)| *step > 0 && *step <= 500_000 && *mean > initial));
    let bytes = tokio::fs::read(
        root.join("checkpoints")
            .join(format!("step-{transitions}.mpk")),
    )
    .await
    .expect("selected native checkpoint");
    let validation = evaluate(bytes.clone(), 10_000..10_100);
    Candidate {
        seed,
        transitions,
        bytes,
        validation,
    }
}

#[tokio::test]
#[ignore = "requires all three completed native Pendulum training runs"]
async fn qualify_three_native_seeds_before_bundling() {
    qualify(NativeRecipe::Original).await;
}

#[tokio::test]
#[ignore = "requires all three completed native Pendulum v2 training runs"]
async fn qualify_three_reduced_actor_rate_seeds_before_bundling() {
    qualify(NativeRecipe::ReducedActorRate).await;
}

/// Every seed must pass common validation before any held-out scores are opened.
async fn qualify(recipe: NativeRecipe) {
    let mut candidates = Vec::new();
    for seed in [42, 43, 44] {
        candidates.push(candidate(seed, recipe).await);
    }
    let version = recipe.version();
    let directory = format!("../runs/browser-pendulum/native-{version}");
    let output = Path::new(&directory);
    tokio::fs::create_dir_all(output)
        .await
        .expect("qualification directory");
    record_validation(&candidates, output).await;
    assert!(
        candidates.iter().all(|value| value.validation.passes()),
        "every seed must pass return and dwell validation"
    );
    let selected = candidates
        .iter()
        .max_by(|left, right| {
            left.validation
                .mean()
                .total_cmp(&right.validation.mean())
                .then(right.seed.cmp(&left.seed))
        })
        .expect("three candidates")
        .seed;
    let mut failed_seeds = Vec::new();
    for value in candidates {
        let failed_seed = record_test(value, selected, output, recipe).await;
        if let Some(seed) = failed_seed {
            failed_seeds.push(seed);
        }
    }
    assert!(
        failed_seeds.is_empty(),
        "failed test seeds: {failed_seeds:?}"
    );
}

/// Retain validation failures before any held-out test cohort is opened.
async fn record_validation(candidates: &[Candidate], output: &Path) {
    let validation = candidates
        .iter()
        .map(|value| {
            serde_json::json!({
                "seed": value.seed, "transitions": value.transitions, "scores": value.validation.report(),
            })
        })
        .collect::<Vec<_>>();
    tokio::fs::write(
        output.join("validation.json"),
        serde_json::to_vec_pretty(&validation).expect("JSON"),
    )
    .await
    .expect("retain validation evidence");
}

/// Open test seeds only after every candidate passes validation, retaining failures too.
async fn record_test(
    value: Candidate,
    selected: u64,
    output: &Path,
    recipe: NativeRecipe,
) -> Option<u64> {
    let test = evaluate(value.bytes.clone(), 400_000..400_200);
    let version = recipe.version();
    let report = serde_json::json!({
        "profile": format!("pendulum-native-ppo-{version}"), "seed": value.seed,
        "selected_training_seed": selected, "transitions": value.transitions,
        "validation_seeds": [10_000, 10_100], "test_seeds": [400_000, 400_200],
        "validation": value.validation.report(), "test": test.report(),
        "dwell_window": {"first_post_transition_index":50,"last_post_transition_index":199,"maximum_absolute_angle_degrees":15,"maximum_absolute_velocity_radians_per_second":1},
    });
    let stem = format!("seed-{seed}", seed = value.seed);
    tokio::fs::write(
        output.join(format!("{stem}.json")),
        serde_json::to_vec_pretty(&report).expect("JSON"),
    )
    .await
    .expect("retain test evidence");
    tokio::fs::write(output.join(format!("{stem}.mpk")), value.bytes)
        .await
        .expect("retain selected policy");
    (!test.passes()).then_some(value.seed)
}

#[test]
fn dwell_includes_angle_and_velocity_boundaries_but_rejects_near_misses() {
    let angle_limit = std::f64::consts::PI / 12.0;
    for angle in [0.0, angle_limit, -angle_limit, std::f64::consts::TAU] {
        for velocity in [-1.0, 0.0, 1.0] {
            assert!(upright(angle, velocity));
        }
    }
    for angle in [
        angle_limit + 1e-9,
        -angle_limit - 1e-9,
        std::f64::consts::PI,
    ] {
        assert!(!upright(angle, 0.0));
    }
    for velocity in [1.0 + 1e-9, -1.0 - 1e-9] {
        assert!(!upright(0.0, velocity));
    }
}

#[test]
fn return_and_dwell_are_independent_inclusive_gates() {
    for (mean, dwell_rate, expected) in [
        (-200.0, 0.70, true),
        (-200.000_001, 0.70, false),
        (-200.0, 0.699_999, false),
        (-300.0, 0.5, false),
    ] {
        assert_eq!(passes(mean, dwell_rate), expected);
    }
}

#[test]
fn aggregates_use_complete_episodes_and_all_one_hundred_fifty_dwell_samples() {
    let scores = Scores(vec![
        EpisodeScore {
            reward: -100.0,
            upright_steps: 150,
        },
        EpisodeScore {
            reward: -300.0,
            upright_steps: 60,
        },
    ]);
    assert_eq!(scores.mean().to_bits(), (-200.0_f64).to_bits());
    assert_eq!(scores.dwell_rate().to_bits(), 0.70_f64.to_bits());
    assert!(scores.passes());
    let report = scores.report();
    assert_eq!(report["mean"], -200.0);
    assert_eq!(report["dwell_rate"], 0.70);
    assert_eq!(
        report["episodes"]
            .as_array()
            .expect("episode evidence")
            .len(),
        2
    );
}

#[test]
fn evaluation_replays_complete_episodes_from_a_browser_session_record() {
    let bytes = Session::train_task(Task::Pendulum, 42)
        .expect("initial browser model")
        .export_policy()
        .expect("inference record");
    let scores = evaluate(bytes, 10_000..10_002);
    assert_eq!(scores.0.len(), 2);
    assert!(scores.mean().is_finite());
    assert!(scores.mean() < -200.0);
    assert!(!scores.passes());
    assert!(scores.0.iter().all(|episode| episode.upright_steps <= 150));
}

#[tokio::test]
#[ignore = "requires the offline browser qualification artifacts under runs/browser-pendulum/browser-v1"]
async fn browser_trained_pendulum_policy_replays_held_out_episodes_natively() {
    let root = Path::new("../runs/browser-pendulum/browser-v1");
    let bytes = tokio::fs::read(root.join("pendulum-policy.mpk"))
        .await
        .expect("browser-trained policy");
    let report: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(root.join("pendulum-learning.json"))
            .await
            .expect("browser qualification evidence"),
    )
    .expect("browser report JSON");
    let scores = evaluate(bytes, 600_000..600_200);
    assert!(scores.passes());
    let episodes = report["test"]["episodes"]
        .as_array()
        .expect("200 held-out episodes");
    assert_eq!(episodes.len(), scores.0.len());
    for (native, browser) in scores.0.iter().zip(episodes) {
        let reward = browser["reward"].as_f64().expect("browser return");
        let native_reward = native.reward;
        assert!(
            (native_reward - reward).abs() < 0.001,
            "native {native_reward} differs from browser {reward}"
        );
        assert_eq!(
            native.upright_steps as u64,
            browser["upright_steps"]
                .as_u64()
                .expect("browser dwell count")
        );
    }
}

#[tokio::test]
#[ignore = "requires the completed native seed 43; reads validation seeds only"]
async fn inspect_seed_43_validation_without_opening_test_seeds() {
    let value = candidate(43, NativeRecipe::Original).await;
    let scores = value.validation.report();
    println!("seed 43 validation: {scores}");
}

#[tokio::test]
#[ignore = "requires completed reduced-rate seeds 42 and 43; reads validation seeds only"]
async fn inspect_completed_reduced_rate_validation_without_opening_test_seeds() {
    for seed in [42, 43] {
        let value = candidate(seed, NativeRecipe::ReducedActorRate).await;
        let scores = value.validation.report();
        println!("seed {seed} reduced-rate validation: {scores}");
    }
}
