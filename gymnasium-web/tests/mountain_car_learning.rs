//! Held-out qualification for the shared `MountainCar` browser learner.
use bevy_gym::environments::{MountainCar, MountainCarAction};
use bevy_gym::training::DqnPolicy;
use bevy_gym::{Env, EpisodeStatus};
use bevy_gym_browser::{AdvanceSteps, Session, Task};
use serde as _;

/// Evaluate original rewards and actual termination, independent of shaping.
fn evaluate(bytes: &[u8], seeds: std::ops::Range<u64>) -> (f64, usize) {
    let policy = DqnPolicy::load_bytes(bytes.to_vec(), 2, 3, &[64, 64]).expect("policy");
    let count = seeds.end - seeds.start;
    let mut sum = 0.0;
    let mut successes = 0;
    for seed in seeds {
        let mut env = MountainCar::default();
        let mut observation = env.reset(Some(seed)).observation;
        for _ in 0..200 {
            let [position, velocity] = observation;
            let encoded = [
                2.0 * (position - (-1.2_f32)) / (0.6_f32 - (-1.2_f32)) - 1.0,
                velocity / 0.07,
            ];
            let action =
                MountainCarAction::try_from(policy.greedy_action(&encoded).expect("inference"))
                    .expect("action");
            let step = env.step(action);
            sum += step.reward;
            observation = step.observation;
            if step.status == EpisodeStatus::Terminated {
                successes += 1;
                break;
            }
        }
    }
    (sum / count as f64, successes)
}

/// Qualify an independent learner using validation-only checkpoint selection.
async fn qualify(seed: u64) {
    let mut session = Session::train_task(Task::MountainCar, seed).expect("learner");
    let initial = session.export_policy().expect("initial model");
    let (initial_mean, _) = evaluate(&initial, 10_000..10_100);
    let mut best_mean = initial_mean;
    let mut best = initial;
    let mut history = Vec::new();
    let mut selected_transitions = 0;
    for checkpoint in 1..=100 {
        for _ in 0..40 {
            session
                .advance(AdvanceSteps::try_from(250).expect("budget"))
                .expect("train");
        }
        let candidate = session.export_policy().expect("candidate");
        let (mean, successes) = evaluate(&candidate, 10_000..10_100);
        let transitions = checkpoint * 10_000;
        println!("seed={seed} transitions={transitions} mean={mean} successes={successes}/100");
        history
            .push(serde_json::json!({"transitions":transitions,"mean":mean,"successes":successes}));
        if mean > best_mean {
            best_mean = mean;
            best = candidate;
            selected_transitions = transitions;
        }
        if mean >= -105.0 && successes >= 95 {
            break;
        }
    }
    let (mean, successes) = evaluate(&best, 200_000..200_200);
    let report = serde_json::json!({"seed":seed,"transitions":selected_transitions,"initial_validation_mean":initial_mean,"best_validation_mean":best_mean,"test_mean":mean,"test_successes":successes,"test_episodes":200,"validation_seeds":[10_000,10_100],"test_seeds":[200_000,200_200],"reward_potential_scale":25,"learning_rate":0.001,"history":history});
    save(seed, report, best).await;
    assert!(
        mean >= -110.0 && successes >= 190,
        "seed {seed}: mean={mean}, successes={successes}/200"
    );
    assert!(
        mean - initial_mean >= 80.0,
        "seed {seed}: insufficient learning improvement"
    );
}

#[tokio::test]
#[ignore = "one-million-transition qualification budget"]
async fn mountain_car_seed_42() {
    qualify(42).await;
}

#[tokio::test]
#[ignore = "one-million-transition qualification budget"]
async fn mountain_car_seed_43() {
    qualify(43).await;
}

#[tokio::test]
#[ignore = "one-million-transition qualification budget"]
async fn mountain_car_seed_44() {
    qualify(44).await;
}

/// Write qualification evidence before enforcing the release threshold.
async fn save(seed: u64, report: serde_json::Value, bytes: Vec<u8>) {
    let root = std::path::Path::new("../runs/browser-mountain-car");
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
