//! Travel-specific tests through the shared learning example seam.

#[path = "../../examples/robots/travel/encoding.rs"]
mod encoding;

use crate::learning::encoding as motor_encoding;
use bevy::math::{Vec2, Vec3};
use bevy_gym::robots::{DroneDestination, DroneHover};

/// Canonical geometry fixes feature order, scales and turn direction independently.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn travel_features_use_body_frame_metres_and_signed_heading_error() {
    let body = DroneHover::default().observation();
    let destination = DroneDestination::try_from((Vec3::new(2.0, 4.0, -4.0), Vec2::X))
        .expect("interior destination");
    let features = encoding::encode_goal(body, destination);
    for (actual, expected) in features.into_iter().zip([
        1.0, 1.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5,
    ]) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (heading, expected) in [(Vec2::NEG_Y, 0.0), (Vec2::NEG_X, 0.5)] {
        let destination =
            DroneDestination::try_from((body.position(), heading)).expect("valid heading");
        let features = encoding::encode_goal(body, destination);
        assert!((features.last().expect("heading feature") - expected).abs() < 1e-6);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[path = "../../examples/robots/travel/model.rs"]
mod model;

/// Appending a zero-weight heading input preserves the qualified RL actor before PPO updates.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn qualified_recovery_actor_transfers_before_heading_training() {
    use bevy_gym::Env;
    let source = crate::learning::load_policy(
        include_bytes!("../../docs/progress/drone-recovery-transfer.mpk").to_vec(),
    )
    .expect("qualified RL source");
    let transferred = model::new_agent(11).expect("transfer RL weights").policy();
    let transferred = model::load_policy(transferred.to_bytes().expect("serialize transfer"))
        .expect("validate thirteen-input actor and fresh critic");
    let mut drone = DroneHover::disturbed();
    let mut source_memory = source.initial_memory();
    let mut travel_memory = transferred.initial_memory();
    for seed in [0, 1, 2, 42, u64::MAX] {
        let body = drone.reset(Some(seed)).observation;
        let source_inputs = crate::learning::encode(body);
        let destination = DroneDestination::try_from((Vec3::new(0.0, 2.0, 0.0), Vec2::X))
            .expect("original position, extra heading");
        let inputs = encoding::encode_goal(body, destination);
        let before = source
            .mean_action(&source_inputs, &source_memory)
            .expect("source action");
        let after = transferred
            .mean_action(&inputs, &travel_memory)
            .expect("transferred action");
        for (left, right) in before.action.iter().zip(&after.action) {
            assert!((left - right).abs() < 1e-6);
        }
        assert_eq!(before.next_memory, after.next_memory);
        source_memory = before.next_memory;
        travel_memory = after.next_memory;
    }
}

#[path = "../../examples/robots/travel/environment.rs"]
mod environment;
#[path = "../../examples/robots/travel/evaluation.rs"]
mod evaluation;
#[path = "../../examples/robots/travel/progress.rs"]
mod progress;
#[path = "../../examples/robots/travel/stage.rs"]
mod stage;

/// The exact inclusive selection boundaries apply to every named travel stage.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn promotion_requires_arrival_survival_settling_and_return_for_every_seed() {
    use evaluation::Score;
    use std::num::NonZeroU16;
    let seeds = crate::learning::SELECTION_SEEDS;
    for stage in stage::Stage::ALL {
        let scores = seeds.map(|seed| Score {
            seed,
            steps: 1_000,
            reward: stage.minimum_return(),
            survived: true,
            first_arrival: NonZeroU16::new(stage.deadline()),
            settled_actions: 100,
            final_distance: 0.5,
            final_heading_error: std::f32::consts::PI / 12.0,
            final_speed: 0.5,
        });
        assert!(evaluation::passes(stage, &scores, &seeds));
        assert!(!evaluation::passes(stage, &[], &[]));
        assert!(!evaluation::passes(
            stage,
            scores.get(..4).expect("four episodes"),
            &seeds
        ));
        for invalid in 0..11 {
            let mut failed = scores.clone();
            let score = failed.first_mut().expect("five selection episodes");
            match invalid {
                0 => score.seed = 999,
                1 => score.steps = 999,
                2 => score.survived = false,
                3 => score.first_arrival = None,
                4 => score.first_arrival = NonZeroU16::new(stage.deadline() + 1),
                5 => score.settled_actions = 99,
                6 => score.settled_actions = 1_001,
                7 => score.final_distance = 0.500_001,
                8 => score.final_heading_error += 0.000_001,
                9 => score.final_speed = 0.500_001,
                _ => score.reward -= 0.001,
            }
            assert!(
                !evaluation::passes(stage, &failed, &seeds),
                "{stage:?}, invalid gate {invalid}"
            );
        }
    }
}

/// Destination sampling and continuing resets reproduce without sharing physics RNG state.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn sampled_destinations_follow_each_stage_and_repeat_reset_streams() {
    use bevy_gym::Env;
    for stage in stage::Stage::ALL {
        let mut task = environment::TravelTask::new(stage);
        let mut replay = environment::TravelTask::factory(stage)();
        for seed in (0..32).chain([u64::MAX]) {
            let start = task.reset(Some(seed));
            assert_eq!(start, replay.reset(Some(seed)));
            let destination = start.observation.destination();
            let position = destination.position();
            let radius = Vec2::new(position.x, position.z).length();
            let ((low, high), (bottom, top)) = stage.bounds();
            assert!(radius >= low - 1e-5 && radius <= high + 1e-5);
            assert!((bottom..=top).contains(&position.y));
            assert!(destination.heading().is_normalized());
            let next = task.reset(None);
            assert_ne!(start, next);
            assert_eq!(next, replay.reset(None));
            assert_eq!(start, task.reset(Some(seed)));
        }
    }
}

/// Evaluation uses real policy actions, rejects incompatible inference, and never updates weights.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn travel_evaluation_is_frozen_and_rejects_wrong_input_width() {
    use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};
    let policy = RecurrentPpoAgent::new(
        13,
        13,
        1,
        &[0.0; 4],
        &[1.0; 4],
        crate::learning::learning_config(),
        SeedConfig::from_root(7),
    )
    .expect("test policy")
    .policy();
    let before = policy.to_bytes().expect("checkpoint before evaluation");
    for stage in stage::Stage::ALL {
        assert!(!stage.name().is_empty());
        let scores = evaluation::evaluate(stage, &policy, &[42]).expect("evaluate actual policy");
        assert_eq!(
            scores,
            evaluation::evaluate(stage, &policy, &[42]).expect("repeat episode")
        );
        assert_eq!(scores.len(), 1);
        let score = scores.first().expect("one score");
        assert!((1..=1_000).contains(&score.steps));
        assert!(score.reward.is_finite());
        assert!(evaluation::evaluate(stage, &policy, &[])
            .expect("no episodes")
            .is_empty());
    }
    assert_eq!(before, policy.to_bytes().expect("unchanged checkpoint"));
    let wrong = crate::learning::new_agent(7).expect("hover width").policy();
    evaluation::evaluate(stage::Stage::Near, &wrong, &[42]).expect_err("wrong width cannot act");
}

/// Preparatory endurance holds the original position and initial heading for the full travel horizon.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn endurance_precedes_travel_without_relaxing_the_final_stages() {
    use bevy_gym::Env;
    assert_eq!(stage::Stage::ALL.first(), Some(&stage::Stage::Endurance));
    let mut environment = environment::TravelTask::new(stage::Stage::Endurance);
    for seed in [0, 1, 2, 42, u64::MAX] {
        let observation = environment.reset(Some(seed)).observation;
        let destination = observation.destination();
        assert_eq!(destination.position(), Vec3::new(0.0, 2.0, 0.0));
        assert!(encoding::heading_error(observation.body(), destination).abs() < 1e-6);
        assert_eq!(observation, environment.reset(Some(seed)).observation);
    }
    assert_eq!(stage::Stage::Near.bounds(), ((0.5, 1.0), (1.5, 2.5)));
    assert_eq!(stage::Stage::Far.bounds(), ((2.0, 4.0), (1.0, 4.0)));
    assert_eq!(stage::Stage::Fast.bounds(), ((4.0, 6.0), (1.0, 5.0)));
    assert_eq!(
        stage::Stage::Near.minimum_return().to_bits(),
        600.0_f64.to_bits()
    );
    assert_eq!(
        stage::Stage::Far.minimum_return().to_bits(),
        500.0_f64.to_bits()
    );
    assert_eq!(
        stage::Stage::Fast.minimum_return().to_bits(),
        500.0_f64.to_bits()
    );
    assert_eq!(stage::Stage::Near.deadline(), 500);
    assert_eq!(stage::Stage::Far.deadline(), 500);
    assert_eq!(stage::Stage::Fast.deadline(), 250);
}
