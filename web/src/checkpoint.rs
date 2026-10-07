//! Browser checkpoint fetching, upload validation, and activation.

use std::error::Error;
use std::fmt;

#[cfg(target_arch = "wasm32")]
use js_sys::Uint8Array;
use sha2::{Digest, Sha256};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;
#[cfg(target_arch = "wasm32")]
use web_sys::{File, Response};

#[cfg(target_arch = "wasm32")]
use crate::manifest::CheckpointSidecar;
use crate::manifest::{CheckpointAsset, ManifestError, MAX_CHECKPOINT_BYTES};

/// Maximum accepted local configuration sidecar size.
#[cfg(target_arch = "wasm32")]
const MAX_SIDECAR_BYTES: u32 = 64 * 1024;

/// One integrity-labelled local checkpoint record.
#[derive(Clone)]
#[cfg(target_arch = "wasm32")]
pub(crate) struct UploadedRecord {
    /// Local filename displayed as untrusted provenance.
    pub(crate) name: Box<str>,
    /// Full lowercase SHA-256 of the selected bytes.
    pub(crate) sha256: Box<str>,
    /// Named MessagePack record bytes.
    pub(crate) bytes: Vec<u8>,
}

/// Fetch one curated asset and enforce its exact content contract.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn fetch_asset(asset: CheckpointAsset) -> Result<Vec<u8>, CheckpointError> {
    let window = web_sys::window().ok_or(CheckpointError::MissingWindow)?;
    let response_value = JsFuture::from(window.fetch_with_str(asset.path.as_ref()))
        .await
        .map_err(|error| CheckpointError::Fetch(js_message(error)))?;
    let response = response_value
        .dyn_into::<Response>()
        .map_err(|error| CheckpointError::Fetch(js_message(error)))?;
    if !response.ok() {
        return Err(CheckpointError::Http(response.status()));
    }
    let buffer_promise = response
        .array_buffer()
        .map_err(|error| CheckpointError::Read(js_message(error)))?;
    let buffer = JsFuture::from(buffer_promise)
        .await
        .map_err(|error| CheckpointError::Read(js_message(error)))?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    validate_curated_bytes(&bytes, &asset)?;
    Ok(bytes)
}

/// Read one local MPK after validating its name and bounded size.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn read_upload(file: File) -> Result<UploadedRecord, CheckpointError> {
    validate_upload_name(&file.name())?;
    let declared_size = file.size();
    if !(1.0..=f64::from(MAX_CHECKPOINT_BYTES)).contains(&declared_size) {
        return Err(CheckpointError::UploadSize(declared_size));
    }
    let buffer = JsFuture::from(file.array_buffer())
        .await
        .map_err(|error| CheckpointError::Read(js_message(error)))?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    validate_upload_bytes(&bytes)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok(UploadedRecord {
        name: file.name().into(),
        sha256: digest.into(),
        bytes,
    })
}

/// Read and decode one optional local checkpoint configuration sidecar.
#[cfg(target_arch = "wasm32")]
pub(crate) async fn read_upload_sidecar(file: File) -> Result<CheckpointSidecar, CheckpointError> {
    validate_sidecar_name(&file.name())?;
    let declared_size = file.size();
    if !(1.0..=f64::from(MAX_SIDECAR_BYTES)).contains(&declared_size) {
        return Err(CheckpointError::SidecarSize(declared_size));
    }
    let buffer = JsFuture::from(file.array_buffer())
        .await
        .map_err(|error| CheckpointError::Read(js_message(error)))?;
    let bytes = Uint8Array::new(&buffer).to_vec();
    let text = std::str::from_utf8(&bytes).map_err(CheckpointError::SidecarUtf8)?;
    Ok(CheckpointSidecar::from_json(text)?)
}

/// Validate response length and digest before Burn sees curated bytes.
fn validate_curated_bytes(bytes: &[u8], asset: &CheckpointAsset) -> Result<(), CheckpointError> {
    let actual = bytes.len();
    let expected = asset.bytes.get() as usize;
    if actual != expected {
        return Err(CheckpointError::Length { expected, actual });
    }
    let digest = Sha256::digest(bytes);
    let actual_digest = format!("{digest:x}");
    if actual_digest != asset.sha256.as_ref() {
        return Err(CheckpointError::Digest {
            expected: asset.sha256.as_ref().into(),
            actual: actual_digest.into(),
        });
    }
    Ok(())
}

/// Enforce the configuration filename before starting an asynchronous read.
fn validate_sidecar_name(name: &str) -> Result<(), CheckpointError> {
    if name.to_ascii_lowercase().ends_with(".config.json") {
        Ok(())
    } else {
        Err(CheckpointError::SidecarName(name.into()))
    }
}

/// Enforce the local file extension before starting an asynchronous read.
fn validate_upload_name(name: &str) -> Result<(), CheckpointError> {
    if name.to_ascii_lowercase().ends_with(".mpk") {
        Ok(())
    } else {
        Err(CheckpointError::UploadName(name.into()))
    }
}

/// Enforce the byte limit again after the browser file read.
fn validate_upload_bytes(bytes: &[u8]) -> Result<(), CheckpointError> {
    if (1..=MAX_CHECKPOINT_BYTES as usize).contains(&bytes.len()) {
        Ok(())
    } else {
        Err(CheckpointError::UploadSize(bytes.len() as f64))
    }
}

/// Convert an opaque JavaScript exception into stable diagnostic text.
#[cfg(target_arch = "wasm32")]
fn js_message(value: JsValue) -> Box<str> {
    value
        .as_string()
        .unwrap_or_else(|| format!("{value:?}"))
        .into()
}

/// Browser fetch, integrity, or upload failure.
#[derive(Debug)]
pub(crate) enum CheckpointError {
    /// The browser global is unavailable.
    #[cfg(target_arch = "wasm32")]
    MissingWindow,
    /// Fetch rejected before an HTTP response arrived.
    #[cfg(target_arch = "wasm32")]
    Fetch(Box<str>),
    /// The server returned a non-success response.
    #[cfg(target_arch = "wasm32")]
    Http(u16),
    /// The response or local file body could not be read.
    #[cfg(target_arch = "wasm32")]
    Read(Box<str>),
    /// The fetched response length differs from the manifest.
    Length { expected: usize, actual: usize },
    /// The fetched response digest differs from the manifest.
    Digest {
        expected: Box<str>,
        actual: Box<str>,
    },
    /// The selected file does not use the MPK extension.
    UploadName(Box<str>),
    /// The selected file is empty or larger than 8 MiB.
    UploadSize(f64),
    /// The selected configuration does not use the sidecar extension.
    SidecarName(Box<str>),
    /// The selected configuration is empty or larger than 64 KiB.
    #[cfg(target_arch = "wasm32")]
    SidecarSize(f64),
    /// The selected configuration is not UTF-8 JSON text.
    #[cfg(target_arch = "wasm32")]
    SidecarUtf8(std::str::Utf8Error),
    /// The selected configuration violates the typed sidecar contract.
    Sidecar(ManifestError),
    /// Burn rejected an integrity-checked checkpoint record.
    #[cfg(target_arch = "wasm32")]
    Decode(Box<str>),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::MissingWindow => formatter.write_str("browser window is unavailable"),
            #[cfg(target_arch = "wasm32")]
            Self::Fetch(message) => write!(formatter, "checkpoint fetch failed: {message}"),
            #[cfg(target_arch = "wasm32")]
            Self::Http(status) => write!(formatter, "checkpoint server returned HTTP {status}"),
            #[cfg(target_arch = "wasm32")]
            Self::Read(message) => write!(formatter, "checkpoint body read failed: {message}"),
            Self::Length { expected, actual } => write!(
                formatter,
                "checkpoint length is {actual} bytes; expected {expected}"
            ),
            Self::Digest { expected, actual } => write!(
                formatter,
                "checkpoint SHA-256 is {actual}; expected {expected}"
            ),
            Self::UploadName(name) => write!(formatter, "upload {name:?} must end in .mpk"),
            Self::UploadSize(size) => write!(
                formatter,
                "upload size is {size} bytes; expected 1 through {MAX_CHECKPOINT_BYTES}"
            ),
            Self::SidecarName(name) => {
                write!(formatter, "upload {name:?} must end in .config.json")
            }
            #[cfg(target_arch = "wasm32")]
            Self::SidecarSize(size) => write!(
                formatter,
                "sidecar size is {size} bytes; expected 1 through {MAX_SIDECAR_BYTES}"
            ),
            #[cfg(target_arch = "wasm32")]
            Self::SidecarUtf8(error) => write!(formatter, "sidecar is not UTF-8: {error}"),
            Self::Sidecar(error) => write!(formatter, "sidecar validation failed: {error}"),
            #[cfg(target_arch = "wasm32")]
            Self::Decode(message) => write!(formatter, "checkpoint decode failed: {message}"),
        }
    }
}

impl Error for CheckpointError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            #[cfg(target_arch = "wasm32")]
            Self::SidecarUtf8(error) => Some(error),
            Self::Sidecar(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ManifestError> for CheckpointError {
    fn from(error: ManifestError) -> Self {
        Self::Sidecar(error)
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::{
        validate_curated_bytes, validate_sidecar_name, validate_upload_bytes, validate_upload_name,
        CheckpointError,
    };
    use crate::manifest::{
        AssetPath, CheckpointAsset, CheckpointSize, ConfigPath, Sha256Digest, MAX_CHECKPOINT_BYTES,
    };

    /// Build one exact static asset contract around the supplied bytes.
    fn asset(bytes: &[u8]) -> CheckpointAsset {
        let digest = format!("{:x}", Sha256::digest(bytes));
        CheckpointAsset {
            path: AssetPath::try_from("checkpoints/test-bunny-000000000000.mpk".to_owned())
                .expect("test path is valid"),
            sha256: Sha256Digest::try_from(digest.clone()).expect("test digest is valid"),
            bytes: CheckpointSize::try_from(u32::try_from(bytes.len()).expect("length fits"))
                .expect("test size is valid"),
            config_path: ConfigPath::try_from(
                "checkpoints/test-bunny-000000000000.config.json".to_owned(),
            )
            .expect("test config path is valid"),
            config_sha256: Sha256Digest::try_from(digest).expect("test digest is valid"),
            config_bytes: CheckpointSize::try_from(1).expect("test size is valid"),
            source_run: "runs/test".into(),
        }
    }

    #[test]
    fn upload_boundary_accepts_mpk_name_and_bounded_bytes() {
        validate_upload_name("policy.MPK").expect("an MPK suffix must be accepted");
        validate_upload_bytes(&[1]).expect("one byte must fit the upload boundary");
    }

    #[test]
    fn upload_boundary_rejects_wrong_extension() {
        let error = validate_upload_name("policy.json").expect_err("JSON must be rejected");
        assert!(matches!(error, CheckpointError::UploadName(_)));
    }

    #[test]
    fn upload_boundary_rejects_empty_content() {
        let error = validate_upload_bytes(&[]).expect_err("empty upload must be rejected");
        assert!(matches!(error, CheckpointError::UploadSize(0.0)));
    }

    #[test]
    fn upload_boundary_rejects_oversized_content() {
        let bytes = vec![0; MAX_CHECKPOINT_BYTES as usize + 1];
        let error = validate_upload_bytes(&bytes).expect_err("oversized upload must be rejected");
        assert!(matches!(error, CheckpointError::UploadSize(_)));
    }

    #[test]
    fn sidecar_boundary_requires_the_config_suffix() {
        validate_sidecar_name("best.config.json").expect("config sidecar suffix is valid");
        let error = validate_sidecar_name("best.json").expect_err("plain JSON must be rejected");
        assert!(matches!(error, CheckpointError::SidecarName(_)));
    }

    #[test]
    fn curated_boundary_rejects_length_and_digest_mismatches() {
        let expected = asset(b"expected");
        let length_error =
            validate_curated_bytes(b"short", &expected).expect_err("length mismatch must fail");
        assert!(matches!(length_error, CheckpointError::Length { .. }));

        let same_length = b"mismatch";
        let digest_error = validate_curated_bytes(same_length, &expected)
            .expect_err("same-length digest mismatch must fail");
        assert!(matches!(digest_error, CheckpointError::Digest { .. }));
    }
}
