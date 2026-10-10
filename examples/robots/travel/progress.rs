//! Persist each completed PPO update before selection or the next rollout starts.

use std::{io::Write, num::NonZeroU32};

use bevy_gym::training::RecurrentPpoUpdate;

use super::stage::Stage;

/// Flush one optimizer measurement without storing rollout observations or selecting actions.
pub(crate) fn record(
    writer: &mut impl Write,
    stage: Stage,
    update: NonZeroU32,
    metrics: &RecurrentPpoUpdate,
) -> std::io::Result<()> {
    // JSON represents non-finite floats as null; reject them before any output is written.
    for (name, value) in [
        ("actor_loss", metrics.actor_loss),
        ("critic_loss", metrics.critic_loss),
        ("entropy", metrics.entropy),
        ("approximate_kl", metrics.approximate_kl),
        ("actor_learning_rate", metrics.actor_learning_rate),
        ("critic_learning_rate", metrics.critic_learning_rate),
    ] {
        if !value.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("non-finite PPO metric: {name}"),
            ));
        }
    }
    let record = serde_json::json!({
        "lesson": stage.name(), "update": update.get(),
        "optimizer_steps": metrics.optimizer_steps,
        "optimizer_updates": metrics.optimizer_updates,
        "valid_samples": metrics.valid_samples,
        "actor_loss": metrics.actor_loss, "critic_loss": metrics.critic_loss,
        "entropy": metrics.entropy, "approximate_kl": metrics.approximate_kl,
        "actor_learning_rate": metrics.actor_learning_rate,
        "critic_learning_rate": metrics.critic_learning_rate,
    });
    writeln!(writer, "{record}")?;
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct values expose swapped fields and preserve signed loss diagnostics.
    fn measurements() -> RecurrentPpoUpdate {
        RecurrentPpoUpdate {
            optimizer_steps: 12,
            optimizer_updates: 4,
            valid_samples: 512,
            actor_loss: -0.125,
            critic_loss: 2.5,
            entropy: -2.0,
            approximate_kl: -0.01,
            actor_learning_rate: 0.0003,
            critic_learning_rate: 0.001,
        }
    }

    /// Each call appends one complete record, including signed metrics and the stage-local index.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn records_preserve_measurements_and_append_newlines() {
        let mut bytes = Vec::new();
        record(
            &mut bytes,
            Stage::Endurance,
            NonZeroU32::MIN,
            &measurements(),
        )
        .expect("first record");
        record(
            &mut bytes,
            Stage::Near,
            NonZeroU32::new(2).expect("positive"),
            &measurements(),
        )
        .expect("second record");
        let text = String::from_utf8(bytes).expect("UTF-8 JSON");
        assert!(text.ends_with('\n'));
        let rows = text
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).expect("complete JSON record")
            })
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        let expected = serde_json::json!({
            "lesson": "travel-endurance", "update": 1,
            "optimizer_steps": 12, "optimizer_updates": 4, "valid_samples": 512,
            "actor_loss": -0.125, "critic_loss": 2.5, "entropy": -2.0,
            "approximate_kl": -0.01, "actor_learning_rate": 0.0003,
            "critic_learning_rate": 0.001,
        });
        assert_eq!(rows.first(), Some(&expected));
        assert_eq!(rows.last().expect("second record")["lesson"], "travel-near");
        assert_eq!(rows.last().expect("second record")["update"], 2);
    }

    /// Disk write and flush errors must reach the caller instead of losing optimizer evidence.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn output_errors_stop_recording() {
        /// Select the failing I/O operation without involving policy or environment state.
        enum Failure {
            /// Refuse bytes before a complete record can reach the output.
            Write,
            /// Accept bytes but fail to flush the completed record.
            Flush,
        }
        impl Write for Failure {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                match self {
                    Self::Write => Err(std::io::Error::other("write failed")),
                    Self::Flush => Ok(bytes.len()),
                }
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("flush failed"))
            }
        }
        for (mut writer, expected) in [
            (Failure::Write, "write failed"),
            (Failure::Flush, "flush failed"),
        ] {
            let error = record(
                &mut writer,
                Stage::Endurance,
                NonZeroU32::MIN,
                &measurements(),
            )
            .expect_err("I/O failure propagates");
            assert_eq!(error.to_string(), expected);
        }
    }

    /// Non-finite diagnostics must stop logging before JSON can silently replace them with null.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn rejects_each_nonfinite_metric_before_writing() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for (name, metrics) in [
                (
                    "actor_loss",
                    RecurrentPpoUpdate {
                        actor_loss: invalid,
                        ..measurements()
                    },
                ),
                (
                    "critic_loss",
                    RecurrentPpoUpdate {
                        critic_loss: invalid,
                        ..measurements()
                    },
                ),
                (
                    "entropy",
                    RecurrentPpoUpdate {
                        entropy: invalid,
                        ..measurements()
                    },
                ),
                (
                    "approximate_kl",
                    RecurrentPpoUpdate {
                        approximate_kl: invalid,
                        ..measurements()
                    },
                ),
                (
                    "actor_learning_rate",
                    RecurrentPpoUpdate {
                        actor_learning_rate: invalid,
                        ..measurements()
                    },
                ),
                (
                    "critic_learning_rate",
                    RecurrentPpoUpdate {
                        critic_learning_rate: invalid,
                        ..measurements()
                    },
                ),
            ] {
                let mut bytes = Vec::new();
                let error = record(&mut bytes, Stage::Endurance, NonZeroU32::MIN, &metrics)
                    .expect_err("non-finite metric fails visibly");
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
                assert_eq!(error.to_string(), format!("non-finite PPO metric: {name}"));
                assert!(bytes.is_empty());
            }
        }
    }
}
