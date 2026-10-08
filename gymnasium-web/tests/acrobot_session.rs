//! Acrobot's browser session uses the native dynamics and the same DQN profile.
use bevy_gym::environments::{Acrobot, AcrobotAction};
use bevy_gym::training::DqnPolicy;
use bevy_gym::{Env, TimeLimit};
use bevy_gym_browser::{AdvanceSteps, Session, Task};
use serde as _;
use serde_json as _;
use tokio as _;

#[test]
fn acrobot_training_changes_q_values_and_exports_a_frozen_policy() {
    assert_eq!("acrobot".parse::<Task>(), Ok(Task::Acrobot));
    let mut session = Session::train_task(Task::Acrobot, 42).expect("fresh learner");
    let before = q_values(session.export_policy().expect("initial parameters"));
    // Replay starts optimizing at 5,000 transitions; eight updates prove learning.
    for _ in 0..19 {
        session.advance(batch(256)).expect("collect replay");
    }
    let snapshot = session.advance(batch(144)).expect("optimizer updates");
    assert_eq!(snapshot.transitions, 5008);
    assert!(snapshot.optimizer_steps > 0);
    assert_eq!(snapshot.learning_rate, Some(0.0003));
    assert_eq!(snapshot.critic_learning_rate, None);
    assert_eq!(snapshot.parallel_environments, 1);
    assert!(snapshot.loss.expect("updated loss").is_finite());
    assert!(snapshot.epsilon.expect("training exploration") < 1.0);
    let bytes = session.export_policy().expect("trained parameters");
    assert_ne!(before, q_values(bytes.clone()));
    assert_frozen(bytes);
}

/// An exported training policy retains no optimizer or exploration state in inference.
fn assert_frozen(bytes: Vec<u8>) {
    let mut frozen =
        Session::inference_task(Task::Acrobot, bytes.clone(), 42).expect("compatible architecture");
    let snapshot = frozen.advance(batch(256)).expect("frozen inference");
    assert_eq!(snapshot.optimizer_steps, 0);
    assert_eq!(snapshot.learning_rate, None);
    assert_eq!(snapshot.epsilon, None);
    assert_eq!(snapshot.loss, None);
    assert_eq!(frozen.export_policy().expect("unchanged parameters"), bytes);
}

#[test]
fn inference_matches_native_actions_resets_and_unshaped_returns() {
    let bytes = Session::train_task(Task::Acrobot, 43)
        .expect("fresh learner")
        .export_policy()
        .expect("parameters");
    let policy = DqnPolicy::load_bytes(bytes.clone(), 6, 3, &[128, 128]).expect("native profile");
    let mut native = TimeLimit::new(Acrobot::default(), 500).expect("positive cap");
    let mut observation = native.reset(Some(17)).observation;
    let mut session = Session::inference_task(Task::Acrobot, bytes, 17).expect("frozen session");
    let mut reward = 0.0;
    let mut episodes = 0;
    for index in 1..=1100 {
        observation[4] /= (4.0 * std::f64::consts::PI) as f32;
        observation[5] /= (9.0 * std::f64::consts::PI) as f32;
        let action = policy.greedy_action(&observation).expect("discrete action");
        let result = native.step(AcrobotAction::try_from(action).expect("three actions"));
        reward += result.reward;
        let snapshot = session.advance(batch(1)).expect("one transition");
        if result.is_done() {
            assert_eq!(snapshot.completed.len(), 1);
            assert_eq!(snapshot.completed[0].reward.to_bits(), reward.to_bits());
            assert_eq!(snapshot.completed[0].transition, index);
            observation = native.reset(None).observation;
            reward = 0.0;
            episodes += 1;
        } else {
            assert!(snapshot.completed.is_empty());
            observation = result.observation;
        }
        assert_eq!(snapshot.episode_count, episodes);
        assert_eq!(snapshot.episode_return.to_bits(), reward.to_bits());
        assert_eq!(
            snapshot.state.map(f64::to_bits),
            native.inner().state().map(f64::to_bits)
        );
    }
    assert!(episodes >= 2);
}

#[test]
fn corrupt_and_other_environment_policies_cannot_load_as_acrobot() {
    for bytes in [
        vec![0, 1, 2],
        include_bytes!("../models/cartpole.mpk").to_vec(),
        include_bytes!("../models/mountain-car.mpk").to_vec(),
    ] {
        Session::inference_task(Task::Acrobot, bytes, 42).expect_err("incompatible policy");
    }
    let bytes = Session::train_task(Task::Acrobot, 42)
        .expect("fresh learner")
        .export_policy()
        .expect("parameters");
    Session::inference(bytes, 42).expect_err("Acrobot cannot load as CartPole");
}

/// Independently inspect the learned values through the native policy loader.
fn q_values(bytes: Vec<u8>) -> Vec<f32> {
    DqnPolicy::load_bytes(bytes, 6, 3, &[128, 128])
        .expect("native profile")
        .q_values(&[1.0, 0.0, 1.0, 0.0, 0.05, -0.05])
        .expect("valid encoded observation")
}

/// Make each test's transition budget explicit at the public boundary.
fn batch(steps: u16) -> AdvanceSteps {
    AdvanceSteps::try_from(steps).expect("bounded batch")
}
