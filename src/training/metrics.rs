//! Metrics JSONL writer boundary.

use std::error::Error;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Metric value emitted by trainer code.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricValue {
    /// Floating-point metric value.
    Number(f64),

    /// Boolean metric value.
    Bool(bool),

    /// String metric value.
    Text(String),
}

impl MetricValue {
    /// Write this metric value as JSON into an existing buffer.
    fn write_json(&self, output: &mut String) {
        match self {
            Self::Number(value) if value.is_finite() => output.push_str(&value.to_string()),
            Self::Number(_) => output.push_str("null"),
            Self::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
            Self::Text(value) => {
                output.push('"');
                output.push_str(&json_escape(value));
                output.push('"');
            }
        }
    }
}

/// One JSONL metrics row.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricRecord {
    /// Global trainer step.
    pub global_step: u64,

    /// Slash-separated metric fields, for example `train/loss`.
    pub fields: Vec<(String, MetricValue)>,
}

impl MetricRecord {
    /// Create an empty metrics record for one global step.
    #[must_use]
    pub const fn new(global_step: u64) -> Self {
        Self {
            global_step,
            fields: Vec::new(),
        }
    }

    /// Add a field to this metrics record.
    #[must_use]
    pub fn with_field(mut self, name: impl Into<String>, value: MetricValue) -> Self {
        self.fields.push((name.into(), value));
        self
    }

    /// Encode this metrics record as one JSON object line.
    #[must_use]
    pub fn to_json_line(&self) -> String {
        let mut output = format!("{{\"global_step\":{}", self.global_step);

        for (name, value) in &self.fields {
            output.push_str(",\"");
            output.push_str(&json_escape(name));
            output.push_str("\":");
            value.write_json(&mut output);
        }

        output.push('}');
        output
    }
}

/// Append-only JSONL metrics writer.
#[derive(Debug)]
pub struct MetricsWriter {
    /// Path receiving JSONL records.
    path: PathBuf,

    /// Open append handle.
    file: File,
}

impl MetricsWriter {
    /// Open a metrics writer in append mode.
    ///
    /// # Errors
    ///
    /// Returns [`MetricsError`] when the file cannot be opened.
    pub fn append(path: impl AsRef<Path>) -> Result<Self, MetricsError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| MetricsError::io(path.clone(), source))?;

        Ok(Self { path, file })
    }

    /// Return the writer path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one record and trailing newline.
    ///
    /// # Errors
    ///
    /// Returns [`MetricsError`] when writing fails.
    pub fn write_record(&mut self, record: &MetricRecord) -> Result<(), MetricsError> {
        let line = record.to_json_line();
        writeln!(self.file, "{line}").map_err(|source| MetricsError::io(self.path.clone(), source))
    }
}

/// Metrics writer error.
#[derive(Debug)]
pub struct MetricsError {
    /// Path involved in the metrics operation.
    pub path: PathBuf,

    /// Underlying I/O error.
    pub source: io::Error,
}

impl MetricsError {
    /// Wrap an I/O error with the path being written.
    const fn io(path: PathBuf, source: io::Error) -> Self {
        Self { path, source }
    }
}

impl fmt::Display for MetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "metrics writer failed for {}: {}",
            self.path.display(),
            self.source
        )
    }
}

impl Error for MetricsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// Escape a JSON object key or string metric value.
fn json_escape(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());

    for ch in input.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => push_json_control_escape(&mut escaped, ch),
            _ => escaped.push(ch),
        }
    }

    escaped
}

/// Append a JSON `\u00xx` escape for a control character.
fn push_json_control_escape(output: &mut String, ch: char) {
    let code = ch as u32;
    output.push_str("\\u");
    for shift in [12, 8, 4, 0] {
        let nibble = (code >> shift) & 0x0f;
        output.push(char::from_digit(nibble, 16).unwrap_or('0'));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_record_writes_jsonl_shape() {
        let record = MetricRecord::new(100)
            .with_field("train/loss", MetricValue::Number(0.25))
            .with_field("checkpoint/best", MetricValue::Bool(true))
            .with_field("run/id", MetricValue::Text("ci-smoke".into()));

        assert_eq!(
            record.to_json_line(),
            "{\"global_step\":100,\"train/loss\":0.25,\"checkpoint/best\":true,\"run/id\":\"ci-smoke\"}"
        );
    }

    #[test]
    fn metric_strings_escape_json_control_characters() {
        let record = MetricRecord::new(2)
            .with_field("note", MetricValue::Text("ok\u{001f}done\u{007f}".into()));

        assert_eq!(
            record.to_json_line(),
            "{\"global_step\":2,\"note\":\"ok\\u001fdone\\u007f\"}"
        );
    }

    #[test]
    fn metrics_writer_appends_record() {
        let path = std::env::temp_dir().join(format!(
            "bevy-gym-metrics-{}-{}.jsonl",
            std::process::id(),
            100
        ));
        drop(std::fs::remove_file(&path));

        let record = MetricRecord::new(1).with_field("eval/mean_reward", MetricValue::Number(42.0));
        let mut writer = MetricsWriter::append(&path).expect("metrics writer opens");
        writer.write_record(&record).expect("metrics record writes");

        let contents = std::fs::read_to_string(&path).expect("metrics file is readable");
        assert_eq!(contents, "{\"global_step\":1,\"eval/mean_reward\":42}\n");

        drop(std::fs::remove_file(path));
    }
}
