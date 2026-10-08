//! Browser worker request protocol.
/// Version-one same-origin worker messages, encoded as JSON objects.
///
/// `start_training` and `start_inference` replace the complete session. Callers
/// create a new worker for each run, so responses cannot cross run boundaries.
#[derive(Debug, serde::Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    /// Construct a fresh learner.
    StartTraining {
        /// Reproducible root seed, restricted to the JavaScript integer range.
        seed: u32,
    },
    /// Load a policy without constructing an optimizer.
    StartInference {
        /// Reproducible reset seed.
        seed: u32,
        /// Named `MessagePack` policy parameters.
        bytes: Vec<u8>,
    },
    /// Advance a bounded number of fixed-duration environment steps.
    Advance {
        /// Validated as 1 through 256 before advancing.
        steps: u16,
    },
    /// Export inference parameters; this is not a resumable training checkpoint.
    Export,
    /// Evaluate one complete Pendulum episode without modifying the active session.
    EvaluatePendulum {
        /// Reproducible reset seed.
        seed: u32,
        /// Frozen recurrent PPO parameters with the Pendulum architecture.
        bytes: Vec<u8>,
    },
}

#[cfg(test)]
mod tests {
    use super::Command;

    #[test]
    fn command_json_rejects_unknown_fields_and_lossy_seeds() {
        for text in [
            r#"{"command":"advance","steps":1,"speed":16}"#,
            r#"{"command":"start_training","seed":4294967296}"#,
            r#"{"command":"start_training","seed":1.5}"#,
            r#"{"command":"start_training","seed":-1}"#,
            r#"{"command":"delete"}"#,
            r#"{"command":"advance"}"#,
            r#"{"command":"evaluate_pendulum","seed":0}"#,
            r#"{"command":"evaluate_pendulum","seed":4294967296,"bytes":[]}"#,
            r#"{"command":"evaluate_pendulum","seed":1.5,"bytes":[]}"#,
            r#"{"command":"evaluate_pendulum","seed":-1,"bytes":[]}"#,
            r#"{"command":"evaluate_pendulum","seed":0,"bytes":[],"steps":200}"#,
        ] {
            serde_json::from_str::<Command>(text).expect_err(text);
        }
        assert!(matches!(
            serde_json::from_str::<Command>(r#"{"command":"start_training","seed":4294967295}"#),
            Ok(Command::StartTraining { seed: u32::MAX })
        ));
        assert!(matches!(
            serde_json::from_str::<Command>(r#"{"command":"export"}"#),
            Ok(Command::Export)
        ));
        assert!(matches!(
            serde_json::from_str::<Command>(
                r#"{"command":"evaluate_pendulum","seed":4294967295,"bytes":[]}"#
            ),
            Ok(Command::EvaluatePendulum { seed: u32::MAX, .. })
        ));
        assert!(matches!(
            serde_json::from_str::<Command>(
                r#"{"command":"start_inference","seed":0,"bytes":[1,2]}"#
            ),
            Ok(Command::StartInference { seed: 0, .. })
        ));
    }
}
