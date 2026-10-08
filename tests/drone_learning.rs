//! Exercise the lesson through the public environment and optimizer APIs.
#![cfg(all(
    feature = "robots",
    any(not(target_arch = "wasm32"), feature = "browser-training")
))]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[path = "../examples/robots/learning/mod.rs"]
mod learning;

use bevy_gym::robots::{DroneAction, DroneHover};
use bevy_gym::Env;
use learning::{
    baseline, decode_action, encode, evaluate, load_policy, new_agent, RecoveryBatch,
    SELECTION_SEEDS,
};

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn encoding_preserves_the_documented_body_frame_and_units() {
    let mut environment = DroneHover::disturbed();
    let observation = environment.reset(Some(42)).observation;
    let features = encode(observation);
    let inverse = observation.orientation().inverse();
    let target = bevy::math::Vec3::new(0.0, 2.0, 0.0);
    let expected = [
        inverse * (target - observation.position()) / 2.0,
        inverse * bevy::math::Vec3::Y,
        inverse * observation.linear_velocity() / 2.0,
        inverse * observation.angular_velocity() / 2.0,
    ];
    for (actual, expected) in features.chunks_exact(3).zip(expected) {
        assert_eq!(actual, expected.to_array());
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn policy_outputs_must_pass_the_typed_action_boundary() {
    for width in [0, 3, 5] {
        decode_action(&vec![0.5; width]).expect_err("policy width must be four");
    }
    for invalid in [
        -f32::EPSILON,
        1.0 + f32::EPSILON,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        decode_action(&[0.5, invalid, 0.5, 0.5]).expect_err("invalid motor value is rejected");
    }
    assert_eq!(
        decode_action(&[0.0, 1.0, 0.5, 0.5]).expect("inclusive motor bounds"),
        DroneAction::try_from([0.0, 1.0, 0.5, 0.5]).expect("inclusive motor bounds")
    );
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[expect(
    clippy::panic_in_result_fn,
    reason = "Assertions fail by panicking; returned errors report test setup failures."
)]
fn seeded_batches_repeat_and_real_updates_change_policy_actions(
) -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = new_agent(7)?;
    let policy = agent.policy();
    let mut left = RecoveryBatch::new(7, &policy);
    let mut right = RecoveryBatch::new(7, &policy);
    let sequences = left.collect(&policy)?;
    assert_eq!(sequences, right.collect(&policy)?);
    assert_eq!(
        sequences
            .iter()
            .map(|sequence| sequence.observations.len())
            .sum::<usize>(),
        512
    );
    let update = agent.update(&sequences)?;
    assert_eq!(update.valid_samples, 512);
    assert!(update.optimizer_updates > 0);
    let mut environment = DroneHover::disturbed();
    let features = encode(environment.reset(Some(42)).observation);
    let before = policy.mean_action(&features, &policy.initial_memory())?;
    let trained = agent.policy();
    let after = trained.mean_action(&features, &trained.initial_memory())?;
    assert_ne!(before.action, after.action);
    decode_action(&after.action)?;
    let restored = load_policy(trained.to_bytes()?)?;
    assert_eq!(
        after,
        restored.mean_action(&features, &restored.initial_memory())?
    );
    Ok(())
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[expect(
    clippy::panic_in_result_fn,
    reason = "Assertions fail by panicking; returned errors report test setup failures."
)]
fn evaluation_is_repeatable_and_never_mutates_the_policy() -> Result<(), Box<dyn std::error::Error>>
{
    let policy = new_agent(7)?.policy();
    let first = evaluate(&policy, &SELECTION_SEEDS[..1])?;
    assert_eq!(first, evaluate(&policy, &SELECTION_SEEDS[..1])?);
    assert_eq!(first.len(), 1);
    let episode = first.first().expect("one evaluation episode");
    assert!((1..=500).contains(&episode.steps));
    assert!(episode.reward.is_finite());
    assert!(episode.final_distance.is_finite());
    assert!(evaluate(&policy, &[])?.is_empty());
    load_policy(vec![]).expect_err("empty checkpoint cannot load");
    let constant = baseline(&SELECTION_SEEDS)?;
    assert!(constant
        .iter()
        .all(|episode| !episode.survived && episode.steps < 500));
    assert_eq!(constant.get(3).expect("seed 42 result").steps, 274);
    Ok(())
}

/// Run only after selecting a checkpoint without consulting these 32 seeds.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires BEVY_GYM_DRONE_CHECKPOINT pointing to a selected trained model"]
#[expect(
    clippy::disallowed_methods,
    reason = "This synchronous qualification reads one local checkpoint."
)]
fn selected_checkpoint_recovers_on_final_seeds() {
    let path = std::env::var_os("BEVY_GYM_DRONE_CHECKPOINT").expect("selected checkpoint path");
    let policy = load_policy(std::fs::read(path).expect("read selected checkpoint"))
        .expect("load selected policy");
    qualify(&policy);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn bundled_checkpoint_recovers_on_held_out_seeds() {
    let policy = load_policy(include_bytes!("../assets/robots/recovery.mpk").to_vec())
        .expect("load bundled recovery policy");
    qualify(&policy);
}

/// Apply the frozen success gates to the final seed partition.
fn qualify(policy: &bevy_gym::training::RecurrentPpoPolicy) {
    let seeds: Vec<_> = (1..=32).map(|offset| u64::MAX - offset).collect();
    let episodes = evaluate(policy, &seeds).expect("final evaluation");
    let constant = baseline(&seeds).expect("constant-thrust comparison");
    let mean_reward = episodes.iter().map(|episode| episode.reward).sum::<f64>() / 32.0;
    let mean_distance = episodes
        .iter()
        .map(|episode| f64::from(episode.final_distance))
        .sum::<f64>()
        / 32.0;
    let survived = episodes.iter().filter(|episode| episode.survived).count();
    let record = serde_json::json!({"episodes": episodes, "baseline": constant, "mean_reward": mean_reward, "mean_final_distance": mean_distance, "survived": survived});
    println!("{record}");
    assert!(
        mean_reward >= 400.0,
        "mean return {mean_reward} must reach 400"
    );
    assert!(
        mean_distance <= 0.5,
        "mean final distance {mean_distance} m must be at most 0.5 m"
    );
    assert!(
        survived >= 30,
        "at least 30 of 32 episodes must survive; observed {survived}"
    );
}
