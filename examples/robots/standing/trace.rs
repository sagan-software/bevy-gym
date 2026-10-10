//! Frozen-session JSONL evidence: RFC 8259 <https://www.rfc-editor.org/rfc/rfc8259>; local profile in docs/DROID_STANDING.md.

use super::{checkpoint, session};
use bevy_gym::{
    robots::{DroidActuator, DroidBody, DroidBodyState, DroidStanding},
    EpisodeStatus,
};
use serde::Serialize;
use std::{error::Error, io::Write};

/// Closed JSONL profile, emitted without accepting alternate input representations.
#[derive(Serialize)]
enum Schema {
    /// One origin followed by the reset frame and every applied action's resulting frame.
    #[serde(rename = "droid-standing-trace-v1")]
    V1,
}

/// Mutually exclusive records; borrowed payloads avoid retaining previous frames.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
enum Row<'a> {
    /// Validated source identity and the independent episode root.
    Origin(&'a Origin<'a>),
    /// Read-only state from one authoritative physics boundary.
    Frame(&'a Frame<'a>),
}

/// Source and nominal time contract; neither establishes qualification.
#[derive(Serialize)]
struct Origin<'a> {
    /// Exact closed emitted profile.
    schema: Schema,
    /// Independent physical reset root; no optimization occurs.
    seed: u64,
    /// Validated seven-member checkpoint identity and claimed training counters.
    source: &'a checkpoint::Record,
    /// Maximum policy actions in the episode.
    horizon: usize,
    /// Nominal milliseconds per action; a terminal substep can stop earlier.
    policy_interval_ms: u128,
}

/// Closed lifecycle spellings at this serialization boundary.
#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
enum Status {
    /// More physical actions remain permitted.
    Continuing,
    /// Physics ended the episode.
    Terminated,
    /// The declared horizon ended the episode.
    Truncated,
}

impl From<EpisodeStatus> for Status {
    fn from(status: EpisodeStatus) -> Self {
        match status {
            EpisodeStatus::Continuing => Self::Continuing,
            EpisodeStatus::Terminated => Self::Terminated,
            EpisodeStatus::Truncated => Self::Truncated,
        }
    }
}

/// One post-action physical boundary; the reset frame has no preceding action.
#[derive(Serialize)]
struct Frame<'a> {
    /// Completed action count; zero identifies the reset boundary.
    step: usize,
    /// Complete episode lifecycle after this boundary.
    status: Status,
    /// Previous policy-selected fractions in `DroidActuator::ALL` order, or reset null.
    torques: Option<[f32; 26]>,
    /// Actor encoding of this boundary; it supplies the next action, when continuing.
    observation: &'a [f32],
    /// All physical segments in `DroidBody::ALL` order.
    segments: [Segment; 13],
}

/// World physical quantities in metres, seconds and radians; every float must be finite.
#[derive(Serialize)]
struct Segment {
    /// Collider centre in world metres, XYZ.
    position: [f32; 3],
    /// Rotation from segment coordinates to world coordinates, quaternion XYZW.
    orientation: [f32; 4],
    /// World linear velocity in metres per second, XYZ.
    linear_velocity: [f32; 3],
    /// World angular velocity in radians per second, XYZ.
    angular_velocity: [f32; 3],
    /// Whether this segment has active contact with the floor.
    floor_contact: bool,
}

/// Validate before emitting evidence; retain only the current frame, O(1) episode storage.
pub(crate) fn write(
    bytes: Vec<u8>,
    metadata: &[u8],
    seed: u64,
    writer: &mut impl Write,
) -> Result<(), Box<dyn Error>> {
    // Validate the checkpoint and architecture before writing even the origin record.
    let mut playback = session::load(bytes, metadata, seed)?;
    let origin = Origin {
        schema: Schema::V1,
        seed,
        source: playback.checkpoint_record(),
        horizon: playback.horizon(),
        policy_interval_ms: DroidStanding::POLICY_INTERVAL.as_millis(),
    };
    emit(writer, &Row::Origin(&origin))?;
    loop {
        // Emit the reset or resulting physical boundary before requesting another action.
        emit_frame(writer, &playback)?;
        if playback.status().is_done() {
            return Ok(());
        }
        playback.step();
        if let Some(error) = playback.error() {
            return Err(error.into());
        }
    }
}

impl From<DroidBodyState> for Segment {
    fn from(body: DroidBodyState) -> Self {
        Self {
            position: body.position().to_array(),
            orientation: body.orientation().to_array(),
            linear_velocity: body.linear_velocity().to_array(),
            angular_velocity: body.angular_velocity().to_array(),
            floor_contact: body.floor_contact(),
        }
    }
}

impl Segment {
    /// Prevent JSON null from concealing a nonfinite physical float.
    fn is_finite(&self) -> bool {
        self.position
            .iter()
            .chain(&self.orientation)
            .chain(&self.linear_velocity)
            .chain(&self.angular_velocity)
            .all(|value| value.is_finite())
    }
}

/// Capture borrowed current-state features and fixed-size physical arrays, without episode buffering.
fn emit_frame(writer: &mut impl Write, playback: &session::Session) -> Result<(), Box<dyn Error>> {
    let observation = playback.observation();
    let features = super::encoding::encode(&observation);
    let segments = DroidBody::ALL.map(|body| Segment::from(observation.body(body)));
    let frame = Frame {
        step: playback.steps(),
        status: playback.status().into(),
        torques: playback
            .last_action()
            .map(|action| DroidActuator::ALL.map(|axis| action.fraction(axis))),
        observation: &features,
        segments,
    };
    emit_snapshot(writer, &frame)
}

/// Reject nonfinite physical evidence before serialization can replace it with JSON null.
fn emit_snapshot(writer: &mut impl Write, frame: &Frame<'_>) -> Result<(), Box<dyn Error>> {
    if !frame.observation.iter().all(|value| value.is_finite())
        || !frame.segments.iter().all(Segment::is_finite)
    {
        return Err("standing trace contains a nonfinite physical state".into());
    }
    emit(writer, &Row::Frame(frame))
}

/// Write exactly one UTF-8 JSON value and LF; stop immediately if the sink rejects bytes.
fn emit(writer: &mut impl Write, row: &Row<'_>) -> Result<(), Box<dyn Error>> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A serialization fixture; it is not an environment controller or qualification result.
    fn fixture(features: &[f32]) -> Frame<'_> {
        Frame {
            step: 0,
            status: Status::Continuing,
            torques: None,
            observation: features,
            segments: std::array::from_fn(|_| Segment {
                position: [0.0; 3],
                orientation: [0.0, 0.0, 0.0, 1.0],
                linear_velocity: [0.0; 3],
                angular_velocity: [0.0; 3],
                floor_contact: false,
            }),
        }
    }

    /// Every scalar field rejects NaN and either infinity before writing any frame bytes.
    #[test]
    fn nonfinite_evidence_cannot_become_json_null() {
        let mut output = Vec::new();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut features = [0.0; 204];
            features[0] = value;
            assert_eq!(
                emit_snapshot(&mut output, &fixture(&features))
                    .unwrap_err()
                    .to_string(),
                "standing trace contains a nonfinite physical state",
            );
            for coordinate in 0..13 {
                let features = [0.0; 204];
                let mut frame = fixture(&features);
                let segment = &mut frame.segments[0];
                let values: Vec<_> = segment
                    .position
                    .iter_mut()
                    .chain(&mut segment.orientation)
                    .chain(&mut segment.linear_velocity)
                    .chain(&mut segment.angular_velocity)
                    .collect();
                *values
                    .into_iter()
                    .nth(coordinate)
                    .expect("thirteen scalars") = value;
                assert!(emit_snapshot(&mut output, &frame).is_err());
                assert!(output.is_empty());
            }
        }
    }

    /// Closed lifecycle variants retain their distinct emitted spellings and exact LF framing.
    #[test]
    fn statuses_and_line_framing_are_exact() {
        let features = [0.0; 204];
        for (status, expected) in [
            (EpisodeStatus::Continuing, "continuing"),
            (EpisodeStatus::Terminated, "terminated"),
            (EpisodeStatus::Truncated, "truncated"),
        ] {
            let mut frame = fixture(&features);
            frame.status = status.into();
            let mut output = Vec::new();
            emit_snapshot(&mut output, &frame).expect("finite fixture");
            assert_eq!(
                std::str::from_utf8(&output)
                    .expect("UTF-8 row")
                    .lines()
                    .count(),
                1
            );
            assert_eq!(output.last(), Some(&b'\n'));
            let value: serde_json::Value = serde_json::from_slice(&output).expect("JSON row");
            assert_eq!(value["status"], expected);
            assert_eq!(value["kind"], "frame");
            assert!(value["torques"].is_null());
        }
    }

    /// Finite f32 formatting retains signed zero, subnormal and extreme bit patterns.
    #[test]
    fn scalar_round_trips_preserve_f32_bits() {
        for scalar in [
            0.0,
            -0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            f32::MIN,
            f32::MAX,
        ] {
            let mut features = [0.0; 204];
            features[0] = scalar;
            let mut output = Vec::new();
            emit_snapshot(&mut output, &fixture(&features)).expect("finite scalar");
            let row: serde_json::Value = serde_json::from_slice(&output).expect("JSON row");
            let observations: Vec<f32> =
                serde_json::from_value(row.get("observation").expect("observation field").clone())
                    .expect("f32 consumer");
            assert_eq!(
                observations.first().expect("first scalar").to_bits(),
                scalar.to_bits()
            );
        }
    }

    /// Independently reject JSON bytes, the final LF, or a flush without hiding sink errors.
    #[test]
    fn sink_failures_stop_emission() {
        for point in [Failure::Json, Failure::Newline, Failure::Flush] {
            let features = [0.0; 204];
            let mut sink = RejectingSink(point);
            let error = emit_snapshot(&mut sink, &fixture(&features)).expect_err("sink failure");
            assert_eq!(error.to_string(), "trace sink failed");
        }
    }

    /// The public streaming seam stops on either the origin or reset frame's rejected bytes.
    #[test]
    fn write_preserves_completed_rows_before_a_sink_failure() {
        let bytes = include_bytes!("../../../docs/progress/droid-standing-trial.mpk");
        let metadata = include_bytes!("../../../docs/progress/droid-standing-trial.json");
        for fail_on_line in [0, 1] {
            let mut sink = RejectingLine {
                fail_on_line,
                completed: 0,
                bytes: Vec::new(),
            };
            assert_eq!(
                write(bytes.to_vec(), metadata, 42, &mut sink)
                    .expect_err("rejected row")
                    .to_string(),
                "trace sink failed",
            );
            assert_eq!(sink.completed, fail_on_line);
            let output = std::str::from_utf8(&sink.bytes).expect("complete UTF-8 rows");
            assert_eq!(output.lines().count(), fail_on_line);
            if fail_on_line == 1 {
                let origin: serde_json::Value = serde_json::from_str(output).expect("origin row");
                assert_eq!(origin["kind"], "origin");
            }
        }
    }

    /// Retain accepted rows and reject the next row before any of its bytes are accepted.
    struct RejectingLine {
        /// Zero-based row to reject.
        fail_on_line: usize,
        /// Rows completely written and flushed.
        completed: usize,
        /// Accepted bytes, used only for the failure-boundary assertion.
        bytes: Vec<u8>,
    }

    impl Write for RejectingLine {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.completed == self.fail_on_line {
                return Err(std::io::Error::other("trace sink failed"));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.completed += 1;
            Ok(())
        }
    }

    /// Disjoint sink failure points for the streaming boundary test.
    #[derive(Clone, Copy)]
    enum Failure {
        /// Reject the first JSON byte.
        Json,
        /// Accept JSON and reject its terminating LF.
        Newline,
        /// Accept the complete line and reject flushing it.
        Flush,
    }

    /// Writer used only to exercise serialization and flush error propagation.
    struct RejectingSink(Failure);

    impl Write for RejectingSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if matches!(self.0, Failure::Json)
                || (matches!(self.0, Failure::Newline) && bytes == b"\n")
            {
                return Err(std::io::Error::other("trace sink failed"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            if matches!(self.0, Failure::Flush) {
                return Err(std::io::Error::other("trace sink failed"));
            }
            Ok(())
        }
    }
}
