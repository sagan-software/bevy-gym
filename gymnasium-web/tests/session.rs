//! Behavioral contracts at the session boundary used by the browser worker.
use bevy_gym::training::DqnPolicy;
use bevy_gym_browser::{AdvanceSteps, Session};
use serde as _;
use serde_json as _;
use tokio as _;

#[test]
fn training_progresses_and_exported_inference_never_optimizes() {
    let mut training = Session::train(42).expect("valid training session");
    let before = training.export_policy().expect("initial policy");
    let observation = [0.01, -0.02, 0.03, -0.04];
    let before_values = DqnPolicy::load_bytes(before.clone(), 4, 2, &[64, 64])
        .expect("initial policy loads")
        .q_values(&observation)
        .expect("valid observation");
    for _ in 0..5 {
        training
            .advance(AdvanceSteps::try_from(256).expect("bounded work"))
            .expect("training advances");
    }
    let after = training.export_policy().expect("trained policy");
    assert_ne!(before, after, "optimizer must change the policy");
    let after_values = DqnPolicy::load_bytes(after.clone(), 4, 2, &[64, 64])
        .expect("trained policy loads")
        .q_values(&observation)
        .expect("valid observation");
    assert_ne!(
        before_values, after_values,
        "optimizer changes actual Q-values"
    );
    let mut inference = Session::inference(after.clone(), 42).expect("valid inference policy");
    let snapshot = inference
        .advance(AdvanceSteps::try_from(32).expect("bounded work"))
        .expect("inference advances");
    assert_eq!(snapshot.transitions, 32);
    assert_eq!(snapshot.optimizer_steps, 0);
    assert!(snapshot.learning_rate.is_none());
    assert_eq!(inference.export_policy().expect("frozen policy"), after);
}

#[test]
fn bundled_cartpole_policy_passes_disjoint_evaluation() {
    let bytes = include_bytes!("../models/cartpole.mpk");
    let mut rewards = Vec::new();
    for seed in 100_000..100_200 {
        let mut session = Session::inference(bytes.to_vec(), seed).expect("bundled policy loads");
        loop {
            let snapshot = session
                .advance(AdvanceSteps::try_from(256).expect("bounded work"))
                .expect("inference advances");
            if let Some(episode) = snapshot.completed.first() {
                rewards.push(episode.reward);
                break;
            }
        }
    }
    let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let full = rewards.iter().filter(|reward| **reward >= 500.0).count();
    println!("CartPole held-out seeds 100000..100200: mean={mean}, full_length={full}/200");
    assert!(mean >= 475.0, "held-out mean {mean} below 475");
    assert!(full >= 180, "only {full}/200 episodes reached 500 steps");
}

#[test]
fn budgets_and_invalid_policies_are_rejected() {
    for invalid in [0, 257, u16::MAX] {
        assert!(matches!(
            AdvanceSteps::try_from(invalid),
            Err(bevy_gym_browser::SessionError::InvalidBudget)
        ));
    }
    for valid in [1, 256] {
        assert_eq!(
            AdvanceSteps::try_from(valid).expect("valid budget").get(),
            valid
        );
    }
    assert!(matches!(
        Session::inference(vec![0, 1, 2], 42),
        Err(bevy_gym_browser::SessionError::Learner(_))
    ));
}

#[test]
fn batching_preserves_fixed_step_dynamics_and_learning() {
    let mut small = Session::train(73).expect("small-batch learner");
    let mut large = Session::train(73).expect("large-batch learner");
    let mut small_snapshot = small
        .advance(AdvanceSteps::try_from(1).expect("one step"))
        .expect("one transition");
    for _ in 1..1280 {
        small_snapshot = small
            .advance(AdvanceSteps::try_from(1).expect("one step"))
            .expect("one transition");
    }
    let mut large_snapshot = large
        .advance(AdvanceSteps::try_from(256).expect("large batch"))
        .expect("batch advances");
    for _ in 1..5 {
        large_snapshot = large
            .advance(AdvanceSteps::try_from(256).expect("large batch"))
            .expect("batch advances");
    }
    assert_eq!(small_snapshot.state, large_snapshot.state);
    assert_eq!(small_snapshot.transitions, large_snapshot.transitions);
    assert_eq!(
        small_snapshot.optimizer_steps,
        large_snapshot.optimizer_steps
    );
    let policy = |session: &Session| {
        DqnPolicy::load_bytes(
            session.export_policy().expect("policy bytes"),
            4,
            2,
            &[64, 64],
        )
        .expect("policy loads")
        .q_values(&[0.01, -0.02, 0.03, -0.04])
        .expect("valid observation")
    };
    assert_eq!(policy(&small), policy(&large));
}

#[test]
fn errors_preserve_the_learner_source_and_describe_budget_limits() {
    use std::error::Error;
    let budget = AdvanceSteps::try_from(0).expect_err("empty batch");
    assert_eq!(
        budget.to_string(),
        "worker batch must contain 1 through 256 steps"
    );
    assert!(budget.source().is_none());
    let policy = Session::inference(vec![0], 42).expect_err("corrupt model");
    assert!(policy.source().is_some());
    assert!(policy.to_string().contains("<memory>"));
}
