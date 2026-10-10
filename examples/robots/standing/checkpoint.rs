//! Standing checkpoint identity and claimed PPO update provenance.

use super::model;
use bevy_gym::training::RecurrentPpoPolicy;
use serde::{Deserialize, Serialize};
use sha2::{digest::Output, Digest, Sha256};
use std::{
    error::Error,
    num::{NonZeroU32, NonZeroU64},
};

/// A positive PPO update record bound to immutable checkpoint bytes.
///
/// A matching record establishes identity and internal consistency, not independent
/// proof of training history or standing competence. Run evidence supplies that proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "WireRecord", into = "WireRecord")]
pub(crate) struct Record {
    /// Root used for initialization and disjoint training streams.
    seed: u64,
    /// Completed batches, each containing exactly 512 environment transitions.
    update: NonZeroU32,
    /// Actual cumulative actor/critic optimizer steps reported by PPO.
    optimizer_steps: NonZeroU64,
    /// Digest of the exact serialized actor and critic bytes.
    digest: Output<Sha256>,
}

impl Record {
    /// Read the validated positive number of completed PPO batches.
    pub(crate) const fn update(&self) -> NonZeroU32 {
        self.update
    }

    /// Bind an actual positive update to the exact bytes written by the trainer.
    pub(crate) fn new(
        seed: u64,
        update: NonZeroU32,
        optimizer_steps: NonZeroU64,
        bytes: &[u8],
    ) -> Self {
        Self {
            seed,
            update,
            optimizer_steps,
            digest: Sha256::digest(bytes),
        }
    }
}

/// Closed observation, action, model and evaluation profile.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "String")]
enum Schema {
    /// 204 body-state inputs, 26 torque outputs and the documented standing gates.
    #[serde(rename = "droid-standing-v1")]
    StandingV1,
}

/// The only accepted training algorithm for these records.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(try_from = "String")]
enum Algorithm {
    /// Reinforcement learning through the recurrent PPO implementation.
    #[serde(rename = "PPO")]
    Ppo,
}

impl std::str::FromStr for Schema {
    type Err = &'static str;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "droid-standing-v1" => Ok(Self::StandingV1),
            _ => Err("unsupported standing checkpoint schema"),
        }
    }
}
impl TryFrom<String> for Schema {
    type Error = &'static str;
    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}
impl std::str::FromStr for Algorithm {
    type Err = &'static str;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "PPO" => Ok(Self::Ppo),
            _ => Err("standing checkpoint algorithm must be PPO"),
        }
    }
}
impl TryFrom<String> for Algorithm {
    type Error = &'static str;
    fn try_from(text: String) -> Result<Self, Self::Error> {
        text.parse()
    }
}

/// All members are required; unknown members and closed-vocabulary extensions fail.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRecord {
    /// Exact profile identifier.
    schema: Schema,
    /// Exact training-method identifier.
    algorithm: Algorithm,
    /// Initialization and training root.
    seed: u64,
    /// Positive completed update count.
    update: NonZeroU32,
    /// Derived sample count, checked on ingress instead of retained independently.
    transitions: u64,
    /// Positive cumulative optimizer step count.
    optimizer_steps: NonZeroU64,
    /// Exactly 64 lowercase hexadecimal ASCII characters.
    sha256: String,
}

impl From<Record> for WireRecord {
    fn from(record: Record) -> Self {
        let digest = record.digest;
        let update = record.update();
        Self {
            schema: Schema::StandingV1,
            algorithm: Algorithm::Ppo,
            seed: record.seed,
            update,
            transitions: u64::from(update.get()) * 512,
            optimizer_steps: record.optimizer_steps,
            sha256: format!("{digest:x}"),
        }
    }
}

impl TryFrom<WireRecord> for Record {
    type Error = Box<dyn Error>;
    fn try_from(wire: WireRecord) -> Result<Self, Self::Error> {
        // Resolve derived counters before accepting this record into the domain.
        if wire.transitions != u64::from(wire.update.get()) * 512 {
            return Err("standing checkpoint transitions must equal update * 512".into());
        }
        Ok(Self {
            seed: wire.seed,
            update: wire.update,
            optimizer_steps: wire.optimizer_steps,
            digest: parse_digest(&wire.sha256)?,
        })
    }
}

/// Parse into the ecosystem's SHA-256 output type at the JSON boundary.
fn parse_digest(text: &str) -> Result<Output<Sha256>, Box<dyn Error>> {
    if text.len() != 64
        || !text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "standing checkpoint sha256 must be 64 lowercase hexadecimal ASCII characters".into(),
        );
    }
    let mut digest = Output::<Sha256>::default();
    for (pair, byte) in text.as_bytes().chunks_exact(2).zip(digest.iter_mut()) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    Ok(digest)
}

/// Verify provenance and the same owned bytes before any inference environment is created.
pub(crate) fn from_bytes(
    bytes: Vec<u8>,
    metadata: &[u8],
) -> Result<(RecurrentPpoPolicy, Record), Box<dyn Error>> {
    let record: Record = serde_json::from_slice(metadata)?;
    if Sha256::digest(&bytes) != record.digest {
        return Err("standing checkpoint digest mismatch".into());
    }
    Ok((model::load_policy(bytes)?, record))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Produce a valid metadata fixture without claiming that its bytes are a trained policy.
    fn wire() -> serde_json::Value {
        serde_json::to_value(Record::new(7, NonZeroU32::MIN, NonZeroU64::MIN, b"fixture"))
            .expect("wire fixture")
    }

    /// Required fields, scalar types and closed vocabularies have no implicit defaults.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn record_rejects_missing_null_and_malformed_fields() {
        for field in [
            "schema",
            "algorithm",
            "seed",
            "update",
            "transitions",
            "optimizer_steps",
            "sha256",
        ] {
            let mut value = wire();
            value.as_object_mut().expect("object").remove(field);
            serde_json::from_value::<Record>(value).expect_err("required field");
            let mut value = wire();
            value[field] = serde_json::Value::Null;
            serde_json::from_value::<Record>(value).expect_err("non-null field");
        }
        for (field, invalid) in [
            ("schema", serde_json::json!("droid-standing-v2")),
            ("schema", serde_json::json!({"droid-standing-v1":null})),
            ("algorithm", serde_json::json!("imitation")),
            ("algorithm", serde_json::json!({"PPO":null})),
            ("seed", serde_json::json!(-1)),
            ("update", serde_json::json!(0)),
            ("update", serde_json::json!(u64::from(u32::MAX) + 1)),
            ("optimizer_steps", serde_json::json!(0)),
            ("transitions", serde_json::json!(511)),
            ("sha256", serde_json::json!("a".repeat(63))),
            ("sha256", serde_json::json!("A".repeat(64))),
            ("sha256", serde_json::json!("g".repeat(64))),
            ("sha256", serde_json::json!("é".repeat(32))),
        ] {
            let mut value = wire();
            value[field] = invalid;
            serde_json::from_value::<Record>(value).expect_err(field);
        }
        let mut value = wire();
        value["extra"] = serde_json::json!(true);
        serde_json::from_value::<Record>(value).expect_err("unknown member");
    }
    /// Canonical output fixes the profile spelling, numeric types and lowercase digest.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn canonical_record_and_duplicate_rejection() {
        let record = Record::new(7, NonZeroU32::MIN, NonZeroU64::MIN, b"fixture");
        let text = serde_json::to_string(&record).expect("canonical record");
        assert_eq!(
            text,
            r#"{"schema":"droid-standing-v1","algorithm":"PPO","seed":7,"update":1,"transitions":512,"optimizer_steps":1,"sha256":"f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"}"#
        );
        let parsed: Record = serde_json::from_str(&text).expect("round trip");
        assert_eq!(
            serde_json::to_string(&parsed).expect("serialize again"),
            text
        );
        for (field, value) in wire().as_object().expect("object") {
            let tail = text.strip_prefix('{').expect("object prefix");
            let duplicate = format!("{{\"{field}\":{value},{tail}");
            serde_json::from_str::<Record>(&duplicate).expect_err("duplicate field");
        }
        let maximum = Record::new(u64::MAX, NonZeroU32::MAX, NonZeroU64::MAX, b"fixture");
        let value = serde_json::to_value(&maximum).expect("maximum counters");
        assert_eq!(value["transitions"], u64::from(u32::MAX) * 512);
        serde_json::from_value::<Record>(value).expect("all counters fit without overflow");
    }
}
