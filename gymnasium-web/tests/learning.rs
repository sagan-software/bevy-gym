//! Reproducible learning gate for the exact training session used by the browser.
use bevy_gym as _;
use bevy_gym_browser::{AdvanceSteps, Session};
use serde as _;

/// Evaluate independent reset seeds without optimizer state.
fn evaluate(bytes: &[u8], seeds: std::ops::Range<u64>) -> (f64, usize) {
    let mut rewards = Vec::new();
    for seed in seeds {
        let mut inference = Session::inference(bytes.to_vec(), seed).expect("policy loads");
        loop {
            let snapshot = inference
                .advance(AdvanceSteps::try_from(256).expect("valid budget"))
                .expect("evaluation advances");
            if let Some(episode) = snapshot.completed.first() {
                rewards.push(episode.reward);
                break;
            }
        }
    }
    (
        rewards.iter().sum::<f64>() / rewards.len() as f64,
        rewards.iter().filter(|reward| **reward >= 500.0).count(),
    )
}

#[tokio::test]
#[ignore = "three training seeds, up to 200000 transitions each; run explicitly for release qualification"]
async fn cartpole_learns_within_the_transition_budget_for_three_seeds() {
    for seed in [42, 43, 44] {
        let mut session = Session::train(seed).expect("fresh learner");
        let initial = session.export_policy().expect("initial model");
        let (initial_mean, _) = evaluate(&initial, 10_000..10_020);
        let mut best = initial;
        let mut best_mean = initial_mean;
        let mut transitions = 0;
        let mut history = Vec::new();
        for checkpoint in 1..=20 {
            for _ in 0..40 {
                transitions = session
                    .advance(AdvanceSteps::try_from(250).expect("valid budget"))
                    .expect("training advances")
                    .transitions;
            }
            let candidate = session.export_policy().expect("trained model");
            let (mean, full) = evaluate(&candidate, 10_000..10_020);
            println!("seed={seed} transitions={transitions} validation_mean={mean} full={full}/20");
            history.push(serde_json::json!({"transitions": transitions, "validation_mean": mean, "full_length": full}));
            if mean > best_mean {
                best_mean = mean;
                best = candidate;
            }
            if mean >= 475.0 && full >= 18 {
                break;
            }
            assert!(
                checkpoint < 20,
                "seed {seed} did not learn within 200000 transitions"
            );
        }
        let (mean, full) = evaluate(&best, 100_000..100_200);
        let report = serde_json::json!({"seed":seed,"transitions":transitions,"initial_validation_mean":initial_mean,
            "best_validation_mean":best_mean,"test_mean":mean,"test_full_length":full,"test_episodes":200,
            "history":history});
        record_qualification(seed, &report, best).await;
        assert!(
            mean >= 475.0 && full >= 180,
            "seed {seed}: held-out mean {mean}, full={full}/200"
        );
        assert!(
            mean - initial_mean >= 400.0,
            "seed {seed}: insufficient learning improvement"
        );
    }
}

/// Preserve each candidate and its evidence before asserting the release thresholds.
async fn record_qualification(seed: u64, report: &serde_json::Value, best: Vec<u8>) {
    let destination = std::path::Path::new("../runs/browser-cartpole");
    tokio::fs::create_dir_all(destination)
        .await
        .expect("qualification output directory");
    tokio::fs::write(
        destination.join(format!("seed-{seed}.json")),
        serde_json::to_vec_pretty(report).expect("report JSON"),
    )
    .await
    .expect("write qualification report");
    tokio::fs::write(destination.join(format!("seed-{seed}.mpk")), best)
        .await
        .expect("write qualified candidate");
}
