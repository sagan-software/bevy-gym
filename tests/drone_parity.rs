//! Exact native/browser replay of public drone observations and episode results.

#![cfg(feature = "robots")]

use bevy_gym::robots::{DroneAction, DroneHover, DroneObservation};
use bevy_gym::{Env, EpisodeStatus, Step, TimeLimit};
use serde::{Deserialize, Serialize};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test_configure!(run_in_browser);

/// A reproducible motor sequence, independent of platform and wall-clock time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum FlightCase {
    /// Equal thrust balances weight for the full ten-second episode.
    Hover,
    /// No thrust leads to ground contact.
    Fall,
    /// Full thrust leads to ceiling exit.
    Climb,
    /// Left motors produce a roll moment.
    Roll,
    /// Front motors produce a pitch moment.
    Pitch,
    /// Opposing diagonal pairs produce a yaw moment.
    Yaw,
    /// Each action changes the motor receiving extra thrust.
    Alternating,
    /// Fixed hover thrust after a disturbed reset.
    DisturbedHover,
    /// Full thrust after a disturbed reset with seed zero.
    DisturbedClimb,
    /// Changing motor commands after a disturbed reset with the largest seed.
    DisturbedAlternating,
}

impl FlightCase {
    /// Cases appear in this stable order in the fixture.
    const ALL: [Self; 10] = [
        Self::Hover,
        Self::Fall,
        Self::Climb,
        Self::Roll,
        Self::Pitch,
        Self::Yaw,
        Self::Alternating,
        Self::DisturbedHover,
        Self::DisturbedClimb,
        Self::DisturbedAlternating,
    ];

    /// Cover zero, an ordinary seed, and the largest accepted seed.
    const fn seed(self) -> u64 {
        match self {
            Self::Hover | Self::Fall | Self::DisturbedHover => 42,
            Self::Climb | Self::Roll | Self::DisturbedClimb => 0,
            Self::Pitch | Self::Yaw | Self::Alternating | Self::DisturbedAlternating => u64::MAX,
        }
    }

    /// Keep the original calm traces while adding the disturbed reset profile.
    fn environment(self) -> DroneHover {
        match self {
            Self::DisturbedHover | Self::DisturbedClimb | Self::DisturbedAlternating => {
                DroneHover::disturbed()
            }
            Self::Hover
            | Self::Fall
            | Self::Climb
            | Self::Roll
            | Self::Pitch
            | Self::Yaw
            | Self::Alternating => DroneHover::default(),
        }
    }

    /// Construct each command through the public validation boundary.
    fn action(self, tick: usize) -> DroneAction {
        let fractions = match self {
            Self::Hover | Self::DisturbedHover => [0.5; 4],
            Self::Fall => [0.0; 4],
            Self::Climb | Self::DisturbedClimb => [1.0; 4],
            Self::Roll => [0.55, 0.45, 0.45, 0.55],
            Self::Pitch => [0.55, 0.55, 0.45, 0.45],
            Self::Yaw => [0.55, 0.45, 0.55, 0.45],
            Self::Alternating | Self::DisturbedAlternating => {
                let mut motors = [0.45; 4];
                *motors.get_mut(tick % 4).expect("one of four motors") = 0.65;
                motors
            }
        };
        DroneAction::try_from(fractions).expect("fixture commands are valid")
    }
}

/// Reset has no reward; each step retains its exact reward and lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Outcome {
    /// A reset observation precedes the next action.
    Reset,
    /// The episode can receive another action; payload is the reward's f64 bits.
    Continuing(u64),
    /// Physics has ended; payload is the reward's f64 bits.
    Terminated(u64),
    /// The external action limit was reached; payload is the reward's f64 bits.
    Truncated(u64),
}

/// One complete public result, stored as integers to preserve every float bit.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Record {
    /// Motor sequence that produced this result.
    case: FlightCase,
    /// World position in metres, encoded as three f32 bit patterns.
    position: [u32; 3],
    /// Body-to-world quaternion in XYZW order, encoded as four f32 bit patterns.
    orientation: [u32; 4],
    /// World velocity in metres per second, encoded as three f32 bit patterns.
    linear_velocity: [u32; 3],
    /// World angular velocity in radians per second, encoded as three f32 bits.
    angular_velocity: [u32; 3],
    /// Reset or step result, including the reward for each step.
    outcome: Outcome,
}

impl Record {
    /// Copy every scalar without rounding or quaternion sign normalization.
    fn observation(case: FlightCase, state: DroneObservation, outcome: Outcome) -> Self {
        Self {
            case,
            position: state.position().to_array().map(f32::to_bits),
            orientation: state.orientation().to_array().map(f32::to_bits),
            linear_velocity: state.linear_velocity().to_array().map(f32::to_bits),
            angular_velocity: state.angular_velocity().to_array().map(f32::to_bits),
            outcome,
        }
    }

    /// Retain termination and truncation as distinct outcomes.
    fn step(case: FlightCase, result: Step<DroneObservation>) -> Self {
        let reward = result.reward.to_bits();
        let outcome = match result.status {
            EpisodeStatus::Continuing => Outcome::Continuing(reward),
            EpisodeStatus::Terminated => Outcome::Terminated(reward),
            EpisodeStatus::Truncated => Outcome::Truncated(reward),
        };
        Self::observation(case, result.observation, outcome)
    }
}

/// Record bounded flights, absorbing termination, and both forms of reset.
fn recordings() -> Vec<Record> {
    let mut records = Vec::new();
    for case in FlightCase::ALL {
        let mut drone = TimeLimit::new(case.environment(), 500).expect("positive limit");
        let initial = drone.reset(Some(case.seed())).observation;
        records.push(Record::observation(case, initial, Outcome::Reset));
        for tick in 0..500 {
            let step = drone.step(case.action(tick));
            let status = step.status;
            records.push(Record::step(case, step));
            if status == EpisodeStatus::Terminated {
                // Different commands must leave a naturally ended body unchanged.
                for repeated in 0..3 {
                    records.push(Record::step(case, drone.step(case.action(repeated))));
                }
            }
            if status.is_done() {
                break;
            }
        }

        // Continuing the random stream and restarting it must both match native.
        for seed in [None, Some(case.seed())] {
            let reset = drone.reset(seed).observation;
            records.push(Record::observation(case, reset, Outcome::Reset));
            records.push(Record::step(case, drone.step(case.action(0))));
        }
    }
    records
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn all_drone_results_match_native_float_bits() {
    let fixture = include_str!("fixtures/robots/drone-native.jsonl");
    let expected: Vec<Record> = serde_json::Deserializer::from_str(fixture)
        .into_iter()
        .collect::<Result<_, _>>()
        .expect("committed native fixture parses");
    let actual = recordings();
    assert_eq!(
        actual.len(),
        expected.len(),
        "fixture must contain every result"
    );
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(actual, expected, "native/browser result {index}");
    }
    let count = actual.len();
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!("Matched {count} complete drone results, bit for bit.");
    #[cfg(not(target_arch = "wasm32"))]
    println!("Matched {count} complete drone results, bit for bit.");
}

/// Regenerate only after reviewing an intentional change to drone dynamics.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "explicitly regenerates the committed native physics fixture"]
fn regenerate_native_fixture() {
    use std::io::{BufWriter, Write};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/robots/drone-native.jsonl"
    );
    let file = std::fs::File::create(path).expect("create native fixture");
    let mut output = BufWriter::new(file);
    for record in recordings() {
        serde_json::to_writer(&mut output, &record).expect("serialize exact result");
        writeln!(output).expect("separate JSON records");
    }
    output.flush().expect("persist native fixture");
}
