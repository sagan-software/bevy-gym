//! Native parallel collection retains the existing physical rollout contract.
#![cfg(all(feature = "robots", not(target_arch = "wasm32")))]
#![expect(
    dead_code,
    unused_imports,
    reason = "Imported example modules retain independent CLI and viewer entry points."
)]

#[path = "../examples/robots/learning/mod.rs"]
#[expect(
    clippy::type_complexity,
    reason = "The imported collector retains its existing function-pointer contract."
)]
mod learning;
#[path = "../examples/robots/standing/mod.rs"]
mod standing;

use bevy_gym::training::{RecurrentPpoConfig, RecurrentPpoPolicy};
use std::num::NonZeroU16;

/// Load an actual failed PPO research checkpoint solely as a rollout regression fixture.
fn policy() -> RecurrentPpoPolicy {
    RecurrentPpoPolicy::load_bytes(
        include_bytes!(
            "../docs/progress/standing-gentle-trials-20261011/calibrated-motion/standing-540.mpk"
        )
        .to_vec(),
        204,
        204,
        1,
        &[-1.0; 26],
        &[1.0; 26],
        &RecurrentPpoConfig {
            log_std_min: -9.0,
            ..Default::default()
        },
    )
    .expect("recorded PPO fixture")
}

/// Compare complete transitions, recurrent memories and reset boundaries over successive batches.
#[test]
fn parallel_droid_collection_matches_existing_serial_sequences() {
    let policy = policy();
    let mut serial =
        standing::batch_with_recipe(227, &policy, standing::reward::Recipe::PostureHeightV1);
    let mut parallel = standing::parallel_rollout::ParallelRecoveryBatch::with_parallel_actions(
        227,
        &policy,
        standing::reward::TrainingTask::posture_height,
        standing::encoding::encode,
        |values| Ok(standing::encoding::decode(values)?),
        NonZeroU16::new(1000).expect("existing horizon"),
    );
    for _ in 0..6 {
        let expected = serial.collect(&policy).expect("serial rollout");
        let actual = parallel
            .collect_parallel(&policy)
            .expect("parallel rollout");
        assert_eq!(
            actual
                .iter()
                .map(|sequence| sequence.observations.len())
                .sum::<usize>(),
            512
        );
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    }
}

/// Controlled worker events are isolated concurrency fixtures, never example agents.
mod workers {
    use super::*;
    use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};
    use bevy_gym::{Env, EpisodeStatus, Reset, Step};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    };
    use std::time::Duration;

    /// Serialize ownership of the constructor fixture shared by these tests only.
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    /// Supply each freshly created lane with the current isolated test's state.
    static CURRENT: Mutex<Option<Arc<Control>>> = Mutex::new(None);

    /// Closed fixture events exercised at the worker boundary.
    enum Mode {
        /// Stall lane zero while lane one reaches its operation.
        Concurrent,
        /// Return distinct invalid observations from lanes zero and two.
        Error,
        /// Panic in two workers while valid workers finish.
        Panic,
        /// Keep every operation valid.
        Valid,
    }
    /// Observe independent worker progress without changing learned policy outputs.
    struct Control {
        /// Assign constructor order before any worker starts.
        next: AtomicUsize,
        /// Count each lane's completed or attempted physical fixture operations.
        calls: [AtomicUsize; 8],
        /// The selected test event.
        mode: Mode,
        /// Lane-zero and lane-one progress markers under one condition-variable lock.
        progress: Mutex<(bool, bool)>,
        /// Wake a stalled worker when its peer reaches the operation.
        changed: Condvar,
    }
    /// One fixture lane with its own observation and shared progress observer.
    struct Fixture {
        /// Closed constructor index in 0..8.
        lane: usize,
        /// Observer retained until every scoped worker finishes.
        control: Arc<Control>,
    }
    impl Default for Fixture {
        fn default() -> Self {
            let control = CURRENT
                .lock()
                .expect("fixture registry")
                .as_ref()
                .expect("active fixture")
                .clone();
            let lane = control.next.fetch_add(1, Ordering::SeqCst);
            Self { lane, control }
        }
    }
    impl Env for Fixture {
        type Observation = [f32; 2];
        type Action = ();
        type Info = ();
        fn reset(&mut self, _seed: Option<u64>) -> Reset<Self::Observation> {
            Reset {
                observation: [0.0; 2],
                info: (),
            }
        }
        fn step(&mut self, (): ()) -> Step<Self::Observation, ()> {
            self.control
                .calls
                .get(self.lane)
                .expect("closed lane index")
                .fetch_add(1, Ordering::SeqCst);
            if matches!(self.control.mode, Mode::Concurrent) && self.lane < 2 {
                self.coordinate();
            }
            assert!(
                !(matches!(self.control.mode, Mode::Panic) && (self.lane == 0 || self.lane == 2)),
                "isolated worker panic fixture"
            );
            let observation = if matches!(self.control.mode, Mode::Error) {
                match self.lane {
                    0 => [f32::NAN, 0.0],
                    2 => [0.0, f32::NAN],
                    _ => [0.0; 2],
                }
            } else {
                [0.0; 2]
            };
            Step {
                observation,
                reward: 1.0,
                status: EpisodeStatus::Continuing,
                info: (),
            }
        }
    }
    impl Fixture {
        /// Stall the first operation until the other lane enters its operation.
        fn coordinate(&self) {
            if self.lane == 0 {
                let mut state = self.control.progress.lock().expect("progress lock");
                state.0 = true;
                drop(state);
                self.control.changed.notify_all();
                let (waited, timeout) = self
                    .control
                    .changed
                    .wait_timeout_while(
                        self.control.progress.lock().expect("progress lock"),
                        Duration::from_secs(2),
                        |flags| !flags.1,
                    )
                    .expect("peer wait");
                let second_entered = waited.1;
                drop(waited);
                assert!(
                    !timeout.timed_out() && second_entered,
                    "lane one must advance while lane zero is stalled"
                );
            } else {
                let (mut waited, timeout) = self
                    .control
                    .changed
                    .wait_timeout_while(
                        self.control.progress.lock().expect("progress lock"),
                        Duration::from_secs(2),
                        |flags| !flags.0,
                    )
                    .expect("first operation wait");
                let first_entered = waited.0;
                waited.1 = true;
                drop(waited);
                self.control.changed.notify_all();
                assert!(
                    !timeout.timed_out() && first_entered,
                    "lane zero must enter first"
                );
            }
        }
    }
    /// Install one test observer and return a two-feature learned policy.
    fn setup(mode: Mode, width: usize) -> (Arc<Control>, RecurrentPpoPolicy) {
        let control = Arc::new(Control {
            next: AtomicUsize::new(0),
            calls: std::array::from_fn(|_| AtomicUsize::new(0)),
            mode,
            progress: Mutex::new((false, false)),
            changed: Condvar::new(),
        });
        *CURRENT.lock().expect("fixture registry") = Some(control.clone());
        let policy = RecurrentPpoAgent::new(
            width,
            width,
            1,
            &[-1.0],
            &[1.0],
            RecurrentPpoConfig {
                actor_hidden_size: 4,
                critic_hidden_sizes: vec![4],
                ..Default::default()
            },
            SeedConfig::from_root(257),
        )
        .expect("fixture policy")
        .policy();
        (control, policy)
    }
    /// Keep fixture decoding separate from physical droid actuation.
    fn batch(
        policy: &RecurrentPpoPolicy,
    ) -> standing::parallel_rollout::ParallelRecoveryBatch<Fixture, 2> {
        standing::parallel_rollout::ParallelRecoveryBatch::with_parallel_actions(
            257,
            policy,
            Fixture::default,
            |observation| *observation,
            |_values| Ok(()),
            NonZeroU16::new(1000).expect("fixture horizon"),
        )
    }
    /// Read all eight counters after the collector has joined every worker.
    fn counts(control: &Control) -> [usize; 8] {
        std::array::from_fn(|index| {
            control
                .calls
                .get(index)
                .expect("closed lane index")
                .load(Ordering::SeqCst)
        })
    }
    #[test]
    fn another_lane_advances_while_the_first_operation_is_stalled() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, policy) = setup(Mode::Concurrent, 2);
        let sequences = batch(&policy)
            .collect_parallel(&policy)
            .expect("concurrent progress");
        assert_eq!(counts(&control), [64; 8]);
        assert_eq!(
            sequences
                .iter()
                .map(|sequence| sequence.observations.len())
                .sum::<usize>(),
            512
        );
    }
    #[test]
    fn inference_errors_retain_lane_order_and_join_valid_workers() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, policy) = setup(Mode::Error, 2);
        let error = batch(&policy)
            .collect_parallel(&policy)
            .expect_err("invalid next observation");
        let error = error
            .downcast_ref::<bevy_gym::training::RecurrentPpoError>()
            .expect("original typed inference error");
        assert!(matches!(
            error,
            bevy_gym::training::RecurrentPpoError::NonFiniteValue { dimension: 0, .. }
        ));
        assert_eq!(counts(&control), [1, 64, 1, 64, 64, 64, 64, 64]);
    }
    #[test]
    fn two_worker_panics_return_an_error_after_other_workers_finish() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, policy) = setup(Mode::Panic, 2);
        let error = batch(&policy)
            .collect_parallel(&policy)
            .expect_err("isolated worker panic");
        let error = error
            .downcast_ref::<std::io::Error>()
            .expect("worker I/O error");
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
        assert_eq!(error.to_string(), "rollout worker 0 panicked");
        assert_eq!(counts(&control), [1, 64, 1, 64, 64, 64, 64, 64]);
    }
    /// Decoder rejection occurs before any lane receives a physical request.
    #[test]
    fn decoder_errors_preserve_the_source_and_advance_no_lane() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, policy) = setup(Mode::Valid, 2);
        let mut batch = standing::parallel_rollout::ParallelRecoveryBatch::with_parallel_actions(
            257,
            &policy,
            Fixture::default,
            |observation| *observation,
            |_values| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "fixture decoder rejection",
                )
                .into())
            },
            NonZeroU16::new(1000).expect("fixture horizon"),
        );
        let error = batch
            .collect_parallel(&policy)
            .expect_err("decoder rejection");
        let error = error
            .downcast_ref::<std::io::Error>()
            .expect("original decoder source");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(error.to_string(), "fixture decoder rejection");
        assert_eq!(counts(&control), [0; 8]);
    }
    /// A valid actor with incompatible critic width fails before any actuator request.
    #[test]
    fn critic_width_errors_advance_no_lane() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, _policy) = setup(Mode::Valid, 2);
        let policy = RecurrentPpoAgent::new(
            2,
            1,
            1,
            &[-1.0],
            &[1.0],
            RecurrentPpoConfig {
                actor_hidden_size: 4,
                critic_hidden_sizes: vec![4],
                ..Default::default()
            },
            SeedConfig::from_root(263),
        )
        .expect("separate critic contract")
        .policy();
        let error = batch(&policy)
            .collect_parallel(&policy)
            .expect_err("critic width mismatch");
        assert!(matches!(
            error.downcast_ref::<bevy_gym::training::RecurrentPpoError>(),
            Some(bevy_gym::training::RecurrentPpoError::DimensionMismatch {
                expected: 1,
                actual: 2,
                ..
            })
        ));
        assert_eq!(counts(&control), [0; 8]);
    }
    /// Time-limit resets preserve bootstrapping while clearing every lane's recurrent memory.
    #[test]
    fn truncation_bootstraps_and_resets_each_recurrent_sequence() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, policy) = setup(Mode::Valid, 2);
        let mut batch = standing::parallel_rollout::ParallelRecoveryBatch::with_parallel_actions(
            257,
            &policy,
            Fixture::default,
            |observation| *observation,
            |_values| Ok(()),
            NonZeroU16::MIN,
        );
        let sequences = batch.collect_parallel(&policy).expect("truncated rollout");
        assert_eq!(sequences.len(), 512);
        let value = policy.value(&[0.0; 2], 0).expect("constant fixture value");
        let expected_return = learning::GAMMA.mul_add(value, 1.0);
        for sequence in sequences {
            assert_eq!(sequence.observations.len(), 1);
            assert_eq!(sequence.initial_memory, policy.initial_memory());
            assert!(
                (sequence.returns.first().expect("one transition") - expected_return).abs() < 1e-6
            );
        }
        assert_eq!(counts(&control), [64; 8]);
    }
    #[test]
    fn incompatible_policy_advances_no_lane() {
        let _guard = TEST_LOCK.lock().expect("isolated fixture");
        let (control, policy) = setup(Mode::Valid, 1);
        batch(&policy)
            .collect_parallel(&policy)
            .expect_err("observation width mismatch");
        assert_eq!(counts(&control), [0; 8]);
    }
}

/// The standing factory propagates a typed width error before decoded torques can advance physics.
#[test]
fn standing_factory_rejects_an_incompatible_motor_count() {
    let policy = bevy_gym::training::RecurrentPpoAgent::new(
        204,
        204,
        1,
        &[-1.0; 25],
        &[1.0; 25],
        RecurrentPpoConfig {
            actor_hidden_size: 4,
            critic_hidden_sizes: vec![4],
            ..Default::default()
        },
        bevy_gym::training::SeedConfig::from_root(271),
    )
    .expect("invalid-for-standing fixture policy")
    .policy();
    let error = standing::parallel_batch_with_recipe(
        271,
        &policy,
        standing::reward::Recipe::PostureHeightV1,
    )
    .collect_parallel(&policy)
    .expect_err("twenty-six torques required");
    assert!(matches!(
        error.downcast_ref::<standing::encoding::DecodeError>(),
        Some(standing::encoding::DecodeError::Width(25))
    ));
}
