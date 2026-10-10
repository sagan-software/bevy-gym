//! Frozen six-agent inference through the same shared physical world.

use crate::{learning, standing};
use bevy::math::Vec3;
use bevy_gym::robots::{
    DroidAction, DroidBody, DroneAction, RobotActions, RobotSlot, RobotSnapshot, RobotWorld,
};
use sha2::Digest;
use std::time::Duration;

/// Qualified standalone hover weights; shared-world qualification remains separate.
const DRONE: &[u8] = include_bytes!("../../docs/progress/drone-hover.mpk");
/// Highest original standing selection candidate; held-out standing failed.
const DROID: &[u8] =
    include_bytes!("../../docs/progress/standing-seed17-update22940/checkpoint.mpk");
/// PPO provenance bound to the retained droid weights.
const RECORD: &[u8] =
    include_bytes!("../../docs/progress/standing-seed17-update22940/checkpoint.json");

/// Native references replay identical actor inputs with separate memories under WASM.
#[path = "world_parity.rs"]
mod parity;

/// One controller per robot produces bounded requests before one common interval.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn six_frozen_controllers_advance_one_common_frame() {
    let mut session =
        standing::world_session::Session::load(DRONE.to_vec(), DROID.to_vec(), RECORD)
            .expect("retained RL checkpoints");
    assert_eq!(session.snapshot().elapsed(), Duration::ZERO);
    for slot in RobotSlot::ALL {
        assert!(session.last_drone_action(slot).is_none());
        assert!(session.last_droid_action(slot).is_none());
    }
    session.step();
    assert!(session.error().is_none());
    assert_eq!(session.snapshot().elapsed(), Duration::from_millis(20));
    for slot in RobotSlot::ALL {
        assert!(session.last_drone_action(slot).is_some());
        assert!(session.last_droid_action(slot).is_some());
    }
}

/// All applied requests follow the original actor encoders and separate robot memories.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn complete_clip_matches_six_direct_recurrent_policies_and_resets() {
    let mut scene = standing::world_session::Session::load(DRONE.to_vec(), DROID.to_vec(), RECORD)
        .expect("retained RL checkpoints");
    let drone = learning::load_policy(DRONE.to_vec()).expect("hover actor");
    let (droid, _) =
        standing::checkpoint::from_bytes(DROID.to_vec(), RECORD).expect("standing actor");
    let mut drone_memory = RobotSlot::ALL.map(|_| drone.initial_memory());
    let mut droid_memory = RobotSlot::ALL.map(|_| droid.initial_memory());
    let mut world = RobotWorld::default();
    let mut snapshot = world.snapshot().expect("direct reset");
    for step in 1..=1_000_u64 {
        let drones = std::array::from_fn(|index| {
            let slot = RobotSlot::ALL[index];
            let input = drone_input(&snapshot, slot);
            let inferred = drone
                .mean_action(&input, &drone_memory[index])
                .expect("own drone memory");
            drone_memory[index] = inferred.next_memory;
            learning::decode_action(&inferred.action).expect("four bounded rotors")
        });
        let droids = std::array::from_fn(|index| {
            let slot = RobotSlot::ALL[index];
            let input = droid_input(&snapshot, slot);
            let inferred = droid
                .mean_action(&input, &droid_memory[index])
                .expect("own droid memory");
            droid_memory[index] = inferred.next_memory;
            standing::encoding::decode(&inferred.action).expect("26 bounded torques")
        });
        snapshot = world
            .advance(RobotActions::new(&snapshot, drones, droids))
            .expect("common step");
        scene.step();
        assert_eq!(scene.snapshot().elapsed(), Duration::from_millis(step * 20));
        assert_frame(&scene, &snapshot, &drones, &droids);
        assert_eq!(scene.finished(), step == 1_000);
    }
    let final_frame = scene.snapshot().frame().clone();
    scene.step();
    assert_eq!(scene.snapshot().frame(), &final_frame);
    assert_restarted_session(&mut scene);
}

/// Compare every applied request and physical body against the independent direct replay.
fn assert_frame(
    scene: &standing::world_session::Session,
    snapshot: &RobotSnapshot,
    drones: &[DroneAction; 3],
    droids: &[DroidAction; 3],
) {
    assert!(scene.error().is_none(), "policy clip stays usable");
    for ((slot, &drone), &droid) in RobotSlot::ALL.into_iter().zip(drones).zip(droids) {
        assert_eq!(scene.last_drone_action(slot), Some(drone));
        assert_eq!(scene.last_droid_action(slot), Some(droid));
        assert_eq!(scene.snapshot().drone(slot), snapshot.drone(slot));
        assert_eq!(scene.snapshot().droid(slot), snapshot.droid(slot));
    }
}

/// Reset must discard the old frame and all six memories before matching a fresh session.
fn assert_restarted_session(scene: &mut standing::world_session::Session) {
    let final_frame = scene.snapshot().frame().clone();
    scene.reset();
    assert_ne!(scene.snapshot().frame(), &final_frame);
    assert!(!scene.finished());
    assert!(scene.error().is_none());
    assert_eq!(scene.snapshot().elapsed(), Duration::ZERO);
    let mut fresh = standing::world_session::Session::load(DRONE.to_vec(), DROID.to_vec(), RECORD)
        .expect("fresh retained policies");
    for slot in RobotSlot::ALL {
        assert!(scene.last_drone_action(slot).is_none());
        assert!(scene.last_droid_action(slot).is_none());
    }
    scene.step();
    fresh.step();
    for slot in RobotSlot::ALL {
        assert_eq!(scene.last_drone_action(slot), fresh.last_drone_action(slot));
        assert_eq!(scene.last_droid_action(slot), fresh.last_droid_action(slot));
        assert_eq!(scene.snapshot().drone(slot), fresh.snapshot().drone(slot));
        assert_eq!(scene.snapshot().droid(slot), fresh.snapshot().droid(slot));
    }
}

/// The specified lanes are fixed task setup, in metres, in closed slot order.
const fn lane(slot: RobotSlot) -> f32 {
    match slot {
        RobotSlot::First => -2.0,
        RobotSlot::Second => 0.0,
        RobotSlot::Third => 2.0,
    }
}

/// Independently project the original hover task into this lane's passive target.
fn drone_input(snapshot: &RobotSnapshot, slot: RobotSlot) -> [f32; 12] {
    let target = Vec3::new(-3.0, 2.0, lane(slot));
    learning::encoding::encode_target(snapshot.drone(slot), target)
}

/// Preserve original standing inputs with the specified translated task origin.
fn droid_input(snapshot: &RobotSnapshot, slot: RobotSlot) -> [f32; standing::encoding::FEATURES] {
    let observation = snapshot.droid(slot);
    let mut input = standing::encoding::encode(&observation);
    let pelvis = observation.body(DroidBody::Pelvis);
    let target = Vec3::new(3.0, 0.99, lane(slot));
    // Position difference is in metres divided by the original 1 m input scale.
    let displacement = pelvis.orientation().inverse() * (target - pelvis.position());
    input[..3].copy_from_slice(&displacement.to_array());
    input
}

/// Invalid identities and malformed model bytes cannot create physical playback.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn checkpoint_identity_and_architecture_precede_playback() {
    for bytes in [Vec::new(), vec![0, 1, 2], DROID.to_vec()] {
        let error = standing::world_session::Session::load(bytes, DROID.to_vec(), RECORD)
            .err()
            .expect("hover identity mismatch");
        assert_eq!(
            error.to_string(),
            "shared-world drone weights must match the recorded PPO hover reference"
        );
    }
    for metadata in [b"".as_slice(), b"{}", b"null"] {
        assert!(
            standing::world_session::Session::load(DRONE.to_vec(), DROID.to_vec(), metadata)
                .is_err()
        );
    }
    let mut changed = DROID.to_vec();
    changed[0] ^= 1;
    let error = standing::world_session::Session::load(DRONE.to_vec(), changed, RECORD)
        .err()
        .expect("standing digest mismatch");
    assert_eq!(error.to_string(), "standing checkpoint digest mismatch");
    let record = standing::checkpoint::Record::new(
        42,
        std::num::NonZeroU32::MIN,
        std::num::NonZeroU64::MIN,
        b"malformed model",
    );
    let metadata = serde_json::to_vec(&record).expect("internally consistent test record");
    assert!(standing::world_session::Session::load(
        DRONE.to_vec(),
        b"malformed model".to_vec(),
        &metadata
    )
    .is_err());
    let scene = standing::world_session::Session::load(DRONE.to_vec(), DROID.to_vec(), RECORD)
        .expect("exact retained checkpoints");
    assert_eq!(scene.drone_digest(), &sha2::Sha256::digest(DRONE));
    assert_eq!(scene.droid_record().update().get(), 22_940);
    let actual = serde_json::to_value(scene.droid_record()).expect("validated record");
    let expected: serde_json::Value = serde_json::from_slice(RECORD).expect("source record");
    assert_eq!(actual, expected);
}
