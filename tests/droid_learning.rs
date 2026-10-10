//! Standing policy collection and updates through the physical environment and PPO APIs.
#![cfg(all(
    feature = "robots",
    any(not(target_arch = "wasm32"), feature = "browser-training")
))]

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[expect(
    unused_imports,
    dead_code,
    reason = "Standing reuses the collector from the shared drone tutorial facade."
)]
#[path = "../examples/robots/learning/mod.rs"]
mod learning;
#[expect(
    dead_code,
    reason = "CLI entry points are exercised by the separate executable tests."
)]
#[path = "../examples/robots/standing/mod.rs"]
mod standing;

use bevy_gym::{robots::DroidStanding, Env};

/// Real physical rollouts drive PPO updates and survive a frozen checkpoint round trip.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn standing_rollout_updates_all_joint_actions_and_round_trips() {
    let mut agent = standing::model::new_agent(7).expect("standing recipe");
    let policy = agent.policy();
    let mut batch = standing::batch(7, &policy);
    let sequences = batch.collect(&policy).expect("physical standing rollout");
    assert_eq!(
        sequences
            .iter()
            .map(|sequence| sequence.observations.len())
            .sum::<usize>(),
        512
    );
    assert!(sequences
        .iter()
        .flat_map(|sequence| &sequence.observations)
        .all(|row| row.len() == 204));
    assert!(sequences
        .iter()
        .flat_map(|sequence| &sequence.pre_tanh_actions)
        .all(|row| row.len() == 26));
    let observation = DroidStanding::default().reset(Some(42)).observation;
    let features = standing::encoding::encode(&observation);
    let before = policy
        .mean_action(&features, &policy.initial_memory())
        .expect("inference");
    let metrics = agent.update(&sequences).expect("real PPO update");
    assert_eq!(metrics.valid_samples, 512);
    assert!(metrics.optimizer_updates > 0);
    let trained = agent.policy();
    let after = trained
        .mean_action(&features, &trained.initial_memory())
        .expect("trained inference");
    assert!(before
        .action
        .iter()
        .zip(&after.action)
        .all(|(before, after)| before.to_bits() != after.to_bits()));
    standing::encoding::decode(&after.action).expect("valid torque commands");
    let restored =
        standing::model::load_policy(trained.to_bytes().expect("serialize")).expect("restore");
    assert_eq!(
        after,
        restored
            .mean_action(&features, &restored.initial_memory())
            .expect("restored inference")
    );
}

/// Promotion requires complete physical success for the exact ordered seed set.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn promotion_requires_a_complete_stable_episode() {
    use standing::evaluation::{passes, Score};
    let success = Score {
        seed: 42,
        steps: 1_000,
        reward: 900.0,
        survived: true,
        stable_actions: 100,
        final_height: 0.99,
        final_upright: 1.0,
        final_distance: 0.0,
        final_speed: 0.0,
        supported: true,
    };
    assert!(passes(std::slice::from_ref(&success), &[42]));
    let mut failed = success.clone();
    failed.stable_actions = 99;
    assert!(!passes(&[failed], &[42]));
    assert!(!passes(&[], &[]));
    assert!(!passes(&[success], &[43]));
}

/// Frozen inference requires a positive PPO update record bound to the exact weight bytes.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn checkpoint_provenance_binds_weights_and_derived_sample_count() {
    use standing::checkpoint::{from_bytes, Record};
    use std::num::{NonZeroU32, NonZeroU64};
    let bytes = standing::model::new_agent(7)
        .expect("fixture")
        .policy()
        .to_bytes()
        .expect("serialize");
    // This fixture exercises metadata validation; it is not evidence of trained behavior.
    let record = Record::new(7, NonZeroU32::MIN, NonZeroU64::MIN, &bytes);
    let metadata = serde_json::to_vec(&record).expect("metadata");
    from_bytes(bytes.clone(), &metadata).expect("matching fixture");
    let mut wrong = bytes.clone();
    *wrong.first_mut().expect("nonempty weights") ^= 1;
    from_bytes(wrong, &metadata).expect_err("mismatched digest");
    let mut value = serde_json::to_value(record).expect("wire record");
    *value.get_mut("transitions").expect("required field") = serde_json::json!(513);
    from_bytes(bytes, &serde_json::to_vec(&value).expect("modified record"))
        .expect_err("contradictory sample count");
}

/// Bind-pose body features preserve fixed geometry across independently sampled root poses.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn body_encoding_preserves_geometry_and_units() {
    use bevy_gym::robots::DroidAction;
    for seed in [0, 42, u64::MAX] {
        let mut environment = DroidStanding::default();
        let observation = environment.reset(Some(seed)).observation;
        let values = standing::encoding::encode(&observation);
        assert!(values.iter().all(|value| value.is_finite()));
        let torso = values.get(12..28).expect("torso features");
        let expected = [
            0.0, 0.26, -0.02, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for (actual, expected) in torso.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let head = values.get(28..31).expect("head displacement");
        for (actual, expected) in head.iter().zip([0.0, 0.59, -0.028]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let moved = environment
            .step(DroidAction::try_from([0.1; 26]).expect("isolated fixture"))
            .observation;
        let actual = standing::encoding::encode(&moved);
        let pelvis = moved.body(bevy_gym::robots::DroidBody::Pelvis);
        let velocity = pelvis.orientation().inverse() * pelvis.linear_velocity() / 2.0;
        let angular = pelvis.orientation().inverse() * pelvis.angular_velocity() / 4.0;
        for (actual, expected) in actual
            .get(6..12)
            .expect("root motion")
            .iter()
            .zip(velocity.to_array().into_iter().chain(angular.to_array()))
        {
            assert!((actual - expected).abs() < 1e-5);
        }
    }
}

/// Decoder errors keep width precedence and the original domain-validation source.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn joint_decoder_rejects_every_malformed_output() {
    use bevy_gym::robots::{DroidActuator, InvalidDroidAction};
    use std::error::Error;
    for width in [0, 25, 27] {
        let error =
            standing::encoding::decode(&vec![f32::NAN; width]).expect_err("wrong width first");
        assert_eq!(
            error.to_string(),
            format!("expected 26 joint values, received {width}")
        );
        assert!(error.source().is_none());
    }
    for (index, actuator) in DroidActuator::ALL.into_iter().enumerate() {
        for value in [-1.0, -0.0, 0.0, 1.0] {
            let mut values = [0.0; 26];
            *values.get_mut(index).expect("axis") = value;
            let action = standing::encoding::decode(&values).expect("valid boundary");
            assert_eq!(action.fraction(actuator).to_bits(), value.to_bits());
        }
        for value in [
            -1.000_000_1,
            1.000_000_1,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut values = [0.0; 26];
            *values.get_mut(index).expect("axis") = value;
            let error = standing::encoding::decode(&values).expect_err("invalid torque");
            assert_eq!(error.to_string(), InvalidDroidAction.to_string());
            assert!(error
                .source()
                .expect("domain cause")
                .is::<InvalidDroidAction>());
        }
    }
}

/// Repeated seeds have independent memory, and evaluation cannot modify frozen weights.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn frozen_evaluation_is_repeatable_and_held_out_roots_are_disjoint() {
    let policy = standing::model::new_agent(7).expect("fixture").policy();
    let before = policy.to_bytes().expect("snapshot");
    let scores =
        standing::evaluation::evaluate(&policy, &[42, 0, 42]).expect("frozen fixture evaluation");
    assert_eq!(scores.first(), scores.last());
    assert_eq!(before, policy.to_bytes().expect("unchanged weights"));
    assert!(!standing::evaluation::passes(&scores, &[42, 0, 42]));
    let roots = standing::evaluation::held_out_seeds();
    assert_eq!(roots.len(), 32);
    for (index, root) in roots.into_iter().enumerate() {
        assert_eq!(root, u64::MAX - index as u64 - 1);
        assert!(root > (1 << 63));
        assert!(!standing::evaluation::SELECTION_SEEDS.contains(&root));
    }
}

/// Encoded relative motion reconstructs measured world motion and exposes real foot contacts.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn segment_features_reconstruct_physical_state() {
    use bevy::math::Vec3;
    use bevy_gym::robots::{DroidAction, DroidBody};
    let mut environment = DroidStanding::default();
    environment.reset(Some(42));
    let mut contact_seen = false;
    for _ in 0..100 {
        let step =
            environment.step(DroidAction::try_from([0.0; 26]).expect("isolated passive fixture"));
        let observation = step.observation;
        let features = standing::encoding::encode(&observation);
        let pelvis = observation.body(DroidBody::Pelvis);
        for (identity, chunk) in DroidBody::ALL.into_iter().skip(1).zip(
            features
                .get(12..)
                .expect("segment features")
                .chunks_exact(16),
        ) {
            let body = observation.body(identity);
            let vector = |start| Vec3::from_slice(chunk.get(start..start + 3).expect("XYZ vector"));
            let position = pelvis.orientation() * vector(0) + pelvis.position();
            let up = pelvis.orientation() * vector(3);
            let forward = pelvis.orientation() * vector(6);
            let velocity = pelvis.orientation() * (vector(9) * 2.0) + pelvis.linear_velocity();
            let angular = pelvis.orientation() * (vector(12) * 4.0) + pelvis.angular_velocity();
            assert!(position.distance(body.position()) < 1e-5);
            assert!(up.distance(body.orientation() * Vec3::Y) < 1e-5);
            assert!(forward.distance(body.orientation() * Vec3::NEG_Z) < 1e-5);
            assert!(velocity.distance(body.linear_velocity()) < 1e-5);
            assert!(angular.distance(body.angular_velocity()) < 1e-5);
            assert_eq!(
                chunk.last().expect("contact").to_bits(),
                f32::from(body.floor_contact()).to_bits()
            );
            contact_seen |= body.floor_contact();
        }
        if step.status.is_done() {
            break;
        }
    }
    assert!(contact_seen);
}

/// Matching metadata cannot turn malformed or incompatible network bytes into valid inference.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn checkpoint_rejects_corrupt_and_incompatible_networks_after_identity() {
    use std::num::{NonZeroU32, NonZeroU64};
    for bytes in [
        b"invalid weights".to_vec(),
        learning::new_agent(7)
            .expect("drone fixture")
            .policy()
            .to_bytes()
            .expect("serialize drone"),
    ] {
        let record = standing::checkpoint::Record::new(7, NonZeroU32::MIN, NonZeroU64::MIN, &bytes);
        let metadata = serde_json::to_vec(&record).expect("identity fixture");
        let error = standing::checkpoint::from_bytes(bytes, &metadata)
            .expect_err("invalid architecture or record");
        assert!(!error.to_string().contains("digest mismatch"));
    }
}

/// Training reward shaping cannot alter resets, torque application or terminal freezing.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn posture_reward_preserves_the_authoritative_physical_episode() {
    use bevy_gym::robots::{DroidAction, DroidBody};
    for seed in [0, 42, u64::MAX] {
        let mut original = DroidStanding::default();
        let mut shaped = standing::reward::TrainingTask::posture();
        assert_eq!(original.reset(Some(seed)), shaped.reset(Some(seed)));
        let mut changed_reward = false;
        for _ in 0..1_000 {
            let action = DroidAction::try_from([0.0; 26]).expect("isolated test fixture");
            let expected = original.step(action);
            let actual = shaped.step(action);
            assert_eq!(expected.observation, actual.observation);
            assert_eq!(expected.status, actual.status);
            assert!((0.0..=expected.reward).contains(&actual.reward));
            changed_reward |= actual.reward < expected.reward;
            if actual.status.is_done() {
                assert_eq!(shaped.step(action), actual);
                assert_eq!(actual.reward.to_bits(), 0.0_f64.to_bits());
                break;
            }
            let observation = actual.observation;
            assert!(observation.body(DroidBody::Torso).position().is_finite());
        }
        assert!(changed_reward);
        assert_eq!(original.reset(None), shaped.reset(None));
    }
}

/// The original profile preserves exact collector samples and optimizer inputs.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn original_training_profile_preserves_all_rollout_values() {
    use standing::reward::Recipe;
    let policy = standing::model::new_agent(7).expect("fixture").policy();
    let mut original = standing::batch(7, &policy);
    let mut profiled = standing::batch_with_recipe(7, &policy, Recipe::Original);
    for _ in 0..2 {
        assert_eq!(
            original.collect(&policy).expect("original rollout"),
            profiled.collect(&policy).expect("profiled rollout")
        );
    }
}

/// A frozen RL actor supplies identical physical samples while shaping changes learning targets.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn posture_rollout_changes_reward_targets_without_selecting_actions() {
    use standing::reward::Recipe;
    let mut agent = standing::model::new_agent(7).expect("seeded actor");
    let policy = agent.policy();
    let mut original = standing::batch(7, &policy);
    let mut shaped = standing::batch_with_recipe(7, &policy, Recipe::PostureV1);
    let expected = original.collect(&policy).expect("original rollout");
    let actual = shaped.collect(&policy).expect("posture rollout");
    assert_eq!(expected.len(), actual.len());
    let mut targets_changed = false;
    for (original, shaped) in expected.iter().zip(&actual) {
        assert_eq!(original.observations, shaped.observations);
        assert_eq!(original.global_states, shaped.global_states);
        assert_eq!(original.pre_tanh_actions, shaped.pre_tanh_actions);
        assert_eq!(original.old_log_probabilities, shaped.old_log_probabilities);
        targets_changed |= original.returns != shaped.returns;
    }
    assert!(targets_changed);
    let metrics = agent.update(&actual).expect("real shaped PPO update");
    assert_eq!(metrics.valid_samples, 512);
    assert!(metrics.optimizer_updates > 0);
}
