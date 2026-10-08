//! Continuous `MountainCar` qualification through the browser worker's public session API.
use bevy_gym as _;
use bevy_gym_browser::{AdvanceSteps, Session, Task};
use serde as _;

/// Score original rewards on held-out resets using a frozen policy.
fn evaluate(bytes: &[u8], seeds: std::ops::Range<u64>) -> (f64, usize) {
    let count = seeds.end - seeds.start;
    let mut total = 0.0;
    let mut successes = 0;
    for seed in seeds {
        let mut session =
            Session::inference_task(Task::MountainCarContinuous, bytes.to_vec(), seed)
                .expect("frozen policy");
        loop {
            let snapshot = session
                .advance(AdvanceSteps::try_from(256).expect("budget"))
                .expect("inference");
            assert_eq!(snapshot.optimizer_steps, 0);
            if let Some(episode) = snapshot.completed.first() {
                total += episode.reward;
                // With force in [-1, 1], 999 penalties total at most 99.9.
                // Only reaching the goal adds 100 and produces a positive return.
                successes += usize::from(episode.reward > 0.0);
                break;
            }
        }
    }
    (total / count as f64, successes)
}

#[test]
fn bundled_continuous_policy_passes_disjoint_evaluation() {
    let (mean, successes) = evaluate(
        include_bytes!("../models/mountain-car-continuous.mpk"),
        400_000..400_200,
    );
    assert!(mean >= 90.0, "bundled mean {mean} below 90");
    assert!(
        successes >= 190,
        "only {successes}/200 episodes reached the goal"
    );
}

/// Select on validation resets only; test the selected checkpoint once.
async fn qualify(seed: u64) {
    let started = std::time::Instant::now();
    let mut session = Session::train_task(Task::MountainCarContinuous, seed).expect("PPO learner");
    let initial = session.export_policy().expect("initial policy");
    let (initial_mean, _) = evaluate(&initial, 10_000..10_020);
    println!("seed={seed} initial_validation_mean={initial_mean}");
    let mut best = initial;
    let mut best_mean = initial_mean;
    let mut selected_transitions = 0;
    let mut history = Vec::new();
    // Each checkpoint is 20 complete 512-transition rollouts. The budget is below two million.
    for checkpoint in 1..=195 {
        for _ in 0..40 {
            session
                .advance(AdvanceSteps::try_from(256).expect("budget"))
                .expect("PPO update");
        }
        let candidate = session.export_policy().expect("candidate policy");
        let (mean, successes) = evaluate(&candidate, 10_000..10_020);
        let transitions = checkpoint * 10_240;
        let seconds = started.elapsed().as_secs_f64();
        println!("seed={seed} transitions={transitions} validation_mean={mean} successes={successes}/20 seconds={seconds:.1}");
        history.push(serde_json::json!({"transitions":transitions,"mean":mean,"successes":successes,"seconds":seconds}));
        if mean > best_mean {
            best_mean = mean;
            best = candidate;
            selected_transitions = transitions;
        }
        if mean >= 90.0 && successes == 20 {
            break;
        }
    }
    let (mean, successes) = evaluate(&best, 400_000..400_200);
    let report = serde_json::json!({
        "seed":seed,"transitions":selected_transitions,"initial_validation_mean":initial_mean,
        "best_validation_mean":best_mean,"test_mean":mean,"test_successes":successes,"test_episodes":200,
        "validation_seeds":[10_000,10_020],"test_seeds":[400_000,400_200],
        "actor_learning_rate":0.003,"critic_learning_rate":0.001,"reward_potential_scale":25,
        "parallel_environments":8,"rollout_steps_per_environment":64,"budget_transitions":1_996_800,"qualification_profile":"mcc-ppo-selection-v2",
        "seconds":started.elapsed().as_secs_f64(),"history":history,
    });
    save(seed, report, best).await;
    assert!(
        mean >= 90.0 && successes >= 190,
        "seed {seed}: mean={mean}, successes={successes}/200"
    );
    assert!(
        best_mean > initial_mean,
        "seed {seed}: insufficient learning improvement"
    );
}

/// Retain the selected model and full report before checking its release gate.
async fn save(seed: u64, report: serde_json::Value, bytes: Vec<u8>) {
    let root = std::path::Path::new("../runs/browser-mountain-car-continuous");
    tokio::fs::create_dir_all(root)
        .await
        .expect("output directory");
    tokio::fs::write(
        root.join(format!("seed-{seed}.json")),
        serde_json::to_vec_pretty(&report).expect("report"),
    )
    .await
    .expect("write report");
    tokio::fs::write(root.join(format!("seed-{seed}.mpk")), bytes)
        .await
        .expect("write policy");
}

#[tokio::test]
#[ignore = "two-million-transition qualification budget"]
async fn continuous_seed_42() {
    qualify(42).await;
}

#[tokio::test]
#[ignore = "two-million-transition qualification budget"]
async fn continuous_seed_43() {
    qualify(43).await;
}

#[tokio::test]
#[ignore = "two-million-transition qualification budget"]
async fn continuous_seed_44() {
    qualify(44).await;
}

#[tokio::test]
#[ignore = "requires the preserved historical training run"]
async fn historical_best_passes_current_browser_environment() {
    let run = "tuned-potential25-actor3e3-critic1e3-seed907-20260730";
    let path = std::path::Path::new("../runs/mountain-car-continuous-ppo")
        .join(run)
        .join("best.mpk");
    let bytes = tokio::fs::read(path).await.expect("preserved best model");
    let (validation_mean, validation_successes) = evaluate(&bytes, 10_000..10_020);
    assert!(validation_mean >= 90.0 && validation_successes == 20);
    let (test_mean, test_successes) = evaluate(&bytes, 400_000..400_200);
    println!("historical run={run} validation_mean={validation_mean} test_mean={test_mean} successes={test_successes}/200");
    let report = serde_json::json!({
        "run_id":run,"seed":907,"source":"preserved native PPO training run; requalified on the current browser session",
        "best_validation_mean":validation_mean,"test_mean":test_mean,"test_successes":test_successes,"test_episodes":200,
        "validation_seeds":[10_000,10_020],"test_seeds":[400_000,400_200],
    });
    save(907, report, bytes).await;
    assert!(test_mean >= 90.0 && test_successes >= 190);
}

#[tokio::test]
#[ignore = "requires the retained Chromium qualification artifacts"]
async fn browser_trained_continuous_policy_loads_natively() {
    let root = std::path::Path::new("../runs/browser-mountain-car-continuous/browser-seed-42");
    let bytes = tokio::fs::read(root.join("mountain-car-continuous-policy.mpk"))
        .await
        .expect("retained browser policy");
    let report: serde_json::Value = serde_json::from_slice(
        &tokio::fs::read(root.join("mountain-car-continuous-learning.json"))
            .await
            .expect("retained browser report"),
    )
    .expect("browser qualification JSON");
    let (mean, successes) = evaluate(&bytes, 500_000..500_200);
    let browser_mean = report["test"]["mean"].as_f64().expect("browser score");
    assert!((mean - browser_mean).abs() < 1e-5);
    assert_eq!(Some(successes as u64), report["test"]["full"].as_u64());
    assert!(mean >= 90.0 && successes >= 190);
}
