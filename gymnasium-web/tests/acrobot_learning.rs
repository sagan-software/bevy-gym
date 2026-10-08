//! Acrobot qualification keeps checkpoint selection separate from held-out tests.
use bevy_gym::environments::{Acrobot, AcrobotAction};
use bevy_gym::training::DqnPolicy;
use bevy_gym::{Env, EpisodeStatus, TimeLimit};
use bevy_gym_browser::{Session, Task};
use serde::Serialize;
use std::path::Path;

/// Original reward and actual goal attainment for one complete episode.
#[derive(Debug, Serialize)]
struct EpisodeScore {
    /// Sum of the environment's original rewards.
    reward: f64,
    /// True only when the engine reports natural termination.
    reached: bool,
}

/// Retain each episode so aggregates and uncertainty can be recomputed.
#[derive(Debug)]
struct Scores(Vec<EpisodeScore>);

impl Scores {
    /// Mean original return across independent resets.
    fn mean(&self) -> f64 {
        self.0.iter().map(|episode| episode.reward).sum::<f64>() / self.0.len() as f64
    }

    /// Fraction of episodes that end above the target height.
    fn goal_rate(&self) -> f64 {
        self.0.iter().filter(|episode| episode.reached).count() as f64 / self.0.len() as f64
    }

    /// Apply both predeclared, inclusive qualification thresholds.
    fn passes(&self) -> bool {
        self.mean() >= -100.0 && self.goal_rate() >= 0.95
    }

    /// Report raw evidence and a diagnostic normal-approximation 95% mean interval.
    fn report(&self) -> serde_json::Value {
        let mean = self.mean();
        let count = self.0.len() as f64;
        let variance = self
            .0
            .iter()
            .map(|episode| (episode.reward - mean).powi(2))
            .sum::<f64>()
            / (count - 1.0);
        let margin = 1.959_963_984_540_054 * (variance / count).sqrt();
        serde_json::json!({"mean":mean,"goal_rate":self.goal_rate(),"mean_95_percent_normal_interval":[mean-margin,mean+margin],"episodes":self.0})
    }
}

/// Evaluate greedy native-compatible actions, retaining the terminal zero reward.
fn evaluate(bytes: Vec<u8>, seeds: std::ops::Range<u64>) -> Scores {
    let policy = DqnPolicy::load_bytes(bytes, 6, 3, &[128, 128]).expect("Acrobot policy");
    let mut episodes = Vec::new();
    for seed in seeds {
        let mut env = TimeLimit::new(Acrobot::default(), 500).expect("positive cap");
        let mut observation = env.reset(Some(seed)).observation;
        let mut reward = 0.0;
        loop {
            observation[4] /= (4.0 * std::f64::consts::PI) as f32;
            observation[5] /= (9.0 * std::f64::consts::PI) as f32;
            let index = policy.greedy_action(&observation).expect("greedy action");
            let result = env.step(AcrobotAction::try_from(index).expect("three actions"));
            reward += result.reward;
            if result.is_done() {
                episodes.push(EpisodeScore {
                    reward,
                    reached: result.status == EpisodeStatus::Terminated,
                });
                break;
            }
            observation = result.observation;
        }
    }
    Scores(episodes)
}

/// A validation-selected native checkpoint, before opening any test seeds.
struct Candidate {
    /// Independent training initialization.
    seed: u64,
    /// Transition count at the selected checkpoint.
    transitions: u64,
    /// Native inference parameters.
    bytes: Vec<u8>,
    /// Common validation episodes, disjoint from the final tests.
    validation: Scores,
}

/// Load a completed run and retain the earliest checkpoint with the best validation mean.
async fn candidate(seed: u64) -> Candidate {
    let run = format!("../runs/acrobot-dqn/acrobot-oracle-v2-seed{seed}-durable");
    let root = Path::new(&run);
    tokio::fs::read(root.join("summary.json"))
        .await
        .expect("completed native run");
    let metrics = tokio::fs::read_to_string(root.join("metrics.jsonl"))
        .await
        .expect("native metrics");
    let evaluations: Vec<(u64, f64)> = metrics
        .lines()
        .filter_map(|line| {
            let row: serde_json::Value = serde_json::from_str(line).expect("metric JSON");
            Some((
                row.get("global_step")?.as_u64()?,
                row.get("eval/mean_reward")?.as_f64()?,
            ))
        })
        .filter(|(step, _)| *step <= 1_000_000)
        .collect();
    let (transitions, _) = evaluations
        .iter()
        .copied()
        .max_by(|left, right| left.1.total_cmp(&right.1).then(right.0.cmp(&left.0)))
        .expect("initial and validation checkpoints");
    let initial = evaluations.first().expect("initial validation").1;
    assert!(evaluations
        .iter()
        .any(|(step, mean)| *step > 0 && *step <= 250_000 && *mean > initial));
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
#[ignore = "requires three completed native Acrobot runs"]
async fn qualify_three_native_seeds_before_bundling() {
    let output = Path::new("../runs/browser-acrobot/native-v2");
    tokio::fs::create_dir_all(output)
        .await
        .expect("qualification directory");
    let mut candidates = Vec::new();
    for seed in [42, 43, 44] {
        candidates.push(candidate(seed).await);
    }
    let validation: Vec<_> = candidates.iter().map(|candidate| serde_json::json!({
        "seed":candidate.seed,"transitions":candidate.transitions,"scores":candidate.validation.report(),
    })).collect();
    write_json(
        &output.join("validation.json"),
        &serde_json::json!(validation),
    )
    .await;
    // A validation failure must not disclose any held-out score.
    assert!(
        candidates
            .iter()
            .all(|candidate| candidate.validation.passes()),
        "every seed must pass validation"
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
    let mut failures = Vec::new();
    for candidate in candidates {
        let test = evaluate(candidate.bytes.clone(), 700_000..700_200);
        let seed = candidate.seed;
        let report = serde_json::json!({
            "profile":"acrobot-native-dqn-v2","seed":seed,"selected_training_seed":selected,
            "transitions":candidate.transitions,"validation_seeds":[10_000,10_100],"test_seeds":[700_000,700_200],
            "validation":candidate.validation.report(),"test":test.report(),
        });
        write_json(&output.join(format!("seed-{seed}.json")), &report).await;
        tokio::fs::write(output.join(format!("seed-{seed}.mpk")), candidate.bytes)
            .await
            .expect("selected policy");
        if !test.passes() {
            failures.push(seed);
        }
    }
    assert!(failures.is_empty(), "failed test seeds: {failures:?}");
}

/// Keep all evaluation evidence, including failing cohorts.
async fn write_json(path: &Path, report: &serde_json::Value) {
    tokio::fs::write(
        path,
        serde_json::to_vec_pretty(report).expect("report JSON"),
    )
    .await
    .expect("retain evidence");
}

#[test]
fn qualification_requires_both_inclusive_thresholds() {
    for (reward, successes, expected) in [
        (-100.0, 19, true),
        (-100.01, 19, false),
        (-100.0, 18, false),
    ] {
        let scores = Scores(
            (0..20)
                .map(|index| EpisodeScore {
                    reward,
                    reached: index < successes,
                })
                .collect(),
        );
        assert_eq!(scores.passes(), expected);
    }
    let scores = Scores(vec![
        EpisodeScore {
            reward: -90.0,
            reached: true,
        },
        EpisodeScore {
            reward: -110.0,
            reached: false,
        },
    ]);
    let report = scores.report();
    assert_eq!(report["mean"], -100.0);
    assert_eq!(report["goal_rate"], 0.5);
    assert!(
        report["mean_95_percent_normal_interval"][0]
            .as_f64()
            .expect("lower interval")
            < -100.0
    );
}

#[test]
fn evaluator_accepts_browser_records_and_retains_complete_episodes() {
    let bytes = Session::train_task(Task::Acrobot, 42)
        .expect("fresh session")
        .export_policy()
        .expect("record");
    let scores = evaluate(bytes, 10_000..10_002);
    assert_eq!(scores.0.len(), 2);
    assert!(scores
        .0
        .iter()
        .all(|episode| (-500.0..=0.0).contains(&episode.reward)));
}

#[tokio::test]
#[ignore = "requires offline browser artifacts under runs/browser-acrobot/browser-v1"]
async fn browser_trained_policy_passes_the_native_held_out_gate() {
    let root = Path::new("../runs/browser-acrobot/browser-v1");
    let bytes = tokio::fs::read(root.join("acrobot-policy.mpk"))
        .await
        .expect("browser-trained policy");
    let report: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(root.join("acrobot-learning.json"))
            .await
            .expect("browser qualification report"),
    )
    .expect("report JSON");
    let scores = evaluate(bytes, 800_000..800_200);
    let episodes = report["test"]["episodes"]
        .as_array()
        .expect("200 held-out episodes");
    assert_eq!(episodes.len(), scores.0.len());
    // Floating-point differences can change decisions near an action boundary.
    // Retain every difference; numerical transition checks use separate probe seeds.
    let differences: Vec<_> =
        episodes
            .iter()
            .zip(&scores.0)
            .enumerate()
            .filter_map(|(index, (expected, actual))| {
                let reward = expected["reward"].as_f64().expect("browser reward");
                (reward.to_bits() != actual.reward.to_bits()).then(|| serde_json::json!({
            "seed":800_000 + index,"browser_reward":reward,"native_reward":actual.reward,
        }))
            })
            .collect();
    write_json(
        &root.join("native-replay.json"),
        &serde_json::json!({"scores":scores.report(),"episode_differences":differences}),
    )
    .await;
    assert!(scores.passes());
}

#[tokio::test]
#[ignore = "requires qualified native and browser Acrobot records"]
async fn record_native_transition_probes_for_browser_compatibility() {
    for (name, policy_path) in [
        ("native", "../runs/browser-acrobot/native-v2/seed-43.mpk"),
        (
            "browser",
            "../runs/browser-acrobot/browser-v1/acrobot-policy.mpk",
        ),
    ] {
        let bytes = tokio::fs::read(policy_path)
            .await
            .expect("qualified policy");
        // File and byte recorders write different format metadata. Preserve the
        // byte recorder's canonical export for an exact browser comparison.
        let canonical = Session::inference_task(Task::Acrobot, bytes.clone(), 0)
            .expect("native policy loader")
            .export_policy()
            .expect("canonical byte record");
        let canonical_path = format!("../runs/browser-acrobot/{name}-canonical.mpk");
        tokio::fs::write(canonical_path, canonical)
            .await
            .expect("canonical record");
        let mut traces = Vec::new();
        for seed in [20_000, 20_001, 20_002] {
            let mut session = Session::inference_task(Task::Acrobot, bytes.clone(), seed)
                .expect("frozen session");
            let mut states = Vec::new();
            for _ in 0..32 {
                let snapshot = session
                    .advance(bevy_gym_browser::AdvanceSteps::try_from(1).expect("one transition"))
                    .expect("frozen transition");
                assert_eq!(snapshot.optimizer_steps, 0);
                states.push(snapshot.state);
            }
            traces.push(serde_json::json!({"seed":seed,"states":states}));
        }
        let path = format!("../runs/browser-acrobot/{name}-transition-probes.json");
        write_json(
            Path::new(&path),
            &serde_json::json!({"model":name,"traces":traces,"maximum_absolute_state_error":1e-9}),
        )
        .await;
    }
}
