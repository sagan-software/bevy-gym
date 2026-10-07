//! `MountainCar` uses the same bounded browser session and real learner as `CartPole`.
use bevy_gym as _;
use bevy_gym_browser::{AdvanceSteps, Session, Task};
use serde as _;
use serde_json as _;
use tokio as _;

#[test]
fn mountain_car_training_exports_a_frozen_environment_specific_policy() {
    let mut session = Session::train_task(Task::MountainCar, 42).expect("valid task");
    let initial = session.export_policy().expect("initial policy");
    for _ in 0..10 {
        session
            .advance(AdvanceSteps::try_from(256).expect("bounded"))
            .expect("training");
    }
    let snapshot = session
        .advance(AdvanceSteps::try_from(1).expect("bounded"))
        .expect("training");
    assert_eq!(snapshot.transitions, 2561);
    assert!(snapshot.optimizer_steps > 0);
    assert!(snapshot.episode_count >= 12);
    assert_eq!(
        snapshot.learning_rate.map(f64::to_bits),
        Some(0.001_f64.to_bits())
    );
    let trained = session.export_policy().expect("trained policy");
    assert_ne!(initial, trained);
    Session::inference(trained.clone(), 42)
        .expect_err("MountainCar parameters cannot load as CartPole");
    let mut frozen =
        Session::inference_task(Task::MountainCar, trained.clone(), 42).expect("compatible policy");
    let snapshot = frozen
        .advance(AdvanceSteps::try_from(256).expect("bounded"))
        .expect("inference");
    assert_eq!(snapshot.optimizer_steps, 0);
    assert_eq!(snapshot.learning_rate, None);
    assert_eq!(frozen.export_policy().expect("unchanged policy"), trained);
    assert!(snapshot
        .completed
        .iter()
        .all(|episode| (-200.0..=-1.0).contains(&episode.reward)));
}

#[test]
fn bundled_mountain_car_policy_meets_original_reward_and_goal_gates() {
    let bytes = include_bytes!("../models/mountain-car.mpk");
    let mut sum = 0.0;
    let mut goals_before_limit = 0;
    for seed in 200_000..200_200 {
        let mut session = Session::inference_task(Task::MountainCar, bytes.to_vec(), seed)
            .expect("qualified record");
        let snapshot = session
            .advance(AdvanceSteps::try_from(256).expect("budget"))
            .expect("inference");
        let episode = snapshot.completed.first().expect("200-step cap");
        sum += episode.reward;
        if episode.reward > -200.0 {
            goals_before_limit += 1;
        }
    }
    assert!(sum / 200.0 >= -110.0);
    // A goal on the final allowed step is conservatively excluded here.
    assert!(goals_before_limit >= 190);
}
