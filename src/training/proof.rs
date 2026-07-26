//! Runtime primitives for building and validating durable learning proof.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

/// A SHA-256 digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    /// Parse the canonical lowercase hexadecimal representation used by schemas.
    ///
    /// # Errors
    ///
    /// Returns [`DigestParseError`] when `hex` is not exactly 64 lowercase
    /// hexadecimal ASCII characters.
    pub fn from_hex(hex: &str) -> Result<Self, DigestParseError> {
        if hex.len() != 64 {
            return Err(DigestParseError::InvalidLength { actual: hex.len() });
        }

        let mut bytes = [0; 32];
        for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
            let high = hex_nibble(pair[0], index * 2)?;
            let low = hex_nibble(pair[1], index * 2 + 1)?;
            bytes[index] = (high << 4) | low;
        }
        Ok(Self(bytes))
    }

    /// Borrow the digest bytes in network order.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Encode the digest as the lowercase 64-character form used by proof artifacts.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            output.push(char::from(HEX[usize::from(byte >> 4)]));
            output.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        output
    }
}

/// A malformed SHA-256 hexadecimal representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestParseError {
    /// The representation is not 64 bytes long.
    InvalidLength {
        /// Observed UTF-8 byte length.
        actual: usize,
    },

    /// A byte is not lowercase hexadecimal ASCII.
    InvalidCharacter {
        /// Zero-based byte index.
        index: usize,

        /// Rejected byte.
        byte: u8,
    },
}

impl fmt::Display for DigestParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { actual } => {
                write!(formatter, "SHA-256 text must be 64 bytes, got {actual}")
            }
            Self::InvalidCharacter { index, byte } => write!(
                formatter,
                "SHA-256 text contains non-lowercase-hex byte {byte:#04x} at index {index}"
            ),
        }
    }
}

impl Error for DigestParseError {}

/// Hash bytes with SHA-256.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize()
}

/// Hash the complete contents of a file with SHA-256.
///
/// # Errors
///
/// Returns the underlying I/O error when the file cannot be opened or read.
pub fn sha256_file(path: impl AsRef<Path>) -> io::Result<Sha256Digest> {
    hash_file(path.as_ref()).map(|(digest, _byte_length)| digest)
}

fn hash_file(path: &Path) -> io::Result<(Sha256Digest, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut byte_length = 0_u64;

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        byte_length = byte_length
            .checked_add(u64::try_from(read).unwrap_or(u64::MAX))
            .ok_or_else(|| io::Error::other("file length exceeds u64"))?;
    }

    Ok((hasher.finalize(), byte_length))
}

fn hex_nibble(byte: u8, index: usize) -> Result<u8, DigestParseError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(DigestParseError::InvalidCharacter { index, byte }),
    }
}

/// Immutable checkpoint bytes and the trainer counts at which they were saved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointEvidence {
    /// File path whose exact bytes produced `sha256`.
    pub path: PathBuf,

    /// SHA-256 of the checkpoint bytes.
    pub sha256: Sha256Digest,

    /// Checkpoint byte length.
    pub byte_length: u64,

    /// Environment steps completed when the checkpoint was saved.
    pub global_step: u64,

    /// Episodes completed when the checkpoint was saved.
    pub episode_count: u64,

    /// Optimizer plus Q-table updates completed when the checkpoint was saved.
    pub update_count: u64,
}

impl CheckpointEvidence {
    /// Hash an existing checkpoint and bind it to its save-time counters.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactConsistencyError::Io`] when the checkpoint cannot be
    /// opened/read and [`ArtifactConsistencyError::EmptyCheckpoint`] when it
    /// contains no bytes.
    pub fn from_file(
        path: impl AsRef<Path>,
        global_step: u64,
        episode_count: u64,
        update_count: u64,
    ) -> Result<Self, ArtifactConsistencyError> {
        let path = path.as_ref().to_path_buf();
        let (sha256, byte_length) = hash_file(&path)
            .map_err(|source| ArtifactConsistencyError::io(path.clone(), source))?;
        if byte_length == 0 {
            return Err(ArtifactConsistencyError::EmptyCheckpoint { path });
        }
        Ok(Self {
            path,
            sha256,
            byte_length,
            global_step,
            episode_count,
            update_count,
        })
    }

    /// Rehash the file and prove its bytes still match this evidence.
    ///
    /// # Errors
    ///
    /// Returns [`ArtifactConsistencyError`] when the file is unreadable or its
    /// byte length or SHA-256 changed.
    pub fn verify_file(&self) -> Result<(), ArtifactConsistencyError> {
        let (actual_hash, actual_length) = hash_file(&self.path)
            .map_err(|source| ArtifactConsistencyError::io(self.path.clone(), source))?;
        if actual_length != self.byte_length {
            return Err(ArtifactConsistencyError::ByteLengthMismatch {
                path: self.path.clone(),
                expected: self.byte_length,
                actual: actual_length,
            });
        }
        if actual_hash != self.sha256 {
            return Err(ArtifactConsistencyError::HashMismatch {
                path: self.path.clone(),
                expected: self.sha256,
                actual: actual_hash,
            });
        }
        Ok(())
    }
}

/// Artifact bytes no longer agree with recorded evidence.
#[derive(Debug)]
pub enum ArtifactConsistencyError {
    /// No artifact evidence was provided.
    NoArtifacts,

    /// A policy checkpoint contains no bytes.
    EmptyCheckpoint {
        /// Empty checkpoint path.
        path: PathBuf,
    },

    /// An artifact path is not a normalized relative JSON path.
    InvalidRelativePath {
        /// Rejected path.
        path: PathBuf,
    },

    /// The same artifact path appears more than once.
    DuplicatePath {
        /// Repeated path.
        path: PathBuf,
    },

    /// An artifact path resolves outside the run directory through a symlink.
    PathEscapesRunDirectory {
        /// Escaping path.
        path: PathBuf,
    },

    /// An artifact path does not resolve to a regular file.
    NotRegularFile {
        /// Invalid path.
        path: PathBuf,
    },

    /// An artifact contains malformed SHA-256 text.
    MalformedHash {
        /// Artifact path.
        path: PathBuf,

        /// Parse failure.
        source: DigestParseError,
    },

    /// A file could not be opened or read.
    Io {
        /// Path involved in the failed operation.
        path: PathBuf,

        /// Underlying I/O error.
        source: io::Error,
    },

    /// The current file length differs from the recorded length.
    ByteLengthMismatch {
        /// Path that changed.
        path: PathBuf,

        /// Recorded byte length.
        expected: u64,

        /// Current byte length.
        actual: u64,
    },

    /// The current file hash differs from the recorded hash.
    HashMismatch {
        /// Path that changed.
        path: PathBuf,

        /// Recorded digest.
        expected: Sha256Digest,

        /// Current digest.
        actual: Sha256Digest,
    },
}

impl ArtifactConsistencyError {
    const fn io(path: PathBuf, source: io::Error) -> Self {
        Self::Io { path, source }
    }
}

impl fmt::Display for ArtifactConsistencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoArtifacts => formatter.write_str("artifact set must not be empty"),
            Self::EmptyCheckpoint { path } => {
                write!(formatter, "checkpoint is empty: {}", path.display())
            }
            Self::InvalidRelativePath { path } => write!(
                formatter,
                "artifact path must be normalized and relative: {}",
                path.display()
            ),
            Self::DuplicatePath { path } => {
                write!(formatter, "artifact path is duplicated: {}", path.display())
            }
            Self::PathEscapesRunDirectory { path } => write!(
                formatter,
                "artifact path escapes the run directory: {}",
                path.display()
            ),
            Self::NotRegularFile { path } => {
                write!(
                    formatter,
                    "artifact is not a regular file: {}",
                    path.display()
                )
            }
            Self::MalformedHash { path, source } => write!(
                formatter,
                "artifact {} has malformed SHA-256: {source}",
                path.display()
            ),
            Self::Io { path, source } => {
                write!(formatter, "failed to hash {}: {source}", path.display())
            }
            Self::ByteLengthMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "artifact {} byte length changed: expected {expected}, got {actual}",
                path.display()
            ),
            Self::HashMismatch {
                path,
                expected,
                actual,
            } => write!(
                formatter,
                "artifact {} SHA-256 changed: expected {}, got {}",
                path.display(),
                expected.to_hex(),
                actual.to_hex()
            ),
        }
    }
}

impl Error for ArtifactConsistencyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MalformedHash { source, .. } => Some(source),
            Self::Io { source, .. } => Some(source),
            Self::NoArtifacts
            | Self::EmptyCheckpoint { .. }
            | Self::InvalidRelativePath { .. }
            | Self::DuplicatePath { .. }
            | Self::PathEscapesRunDirectory { .. }
            | Self::NotRegularFile { .. }
            | Self::ByteLengthMismatch { .. }
            | Self::HashMismatch { .. } => None,
        }
    }
}

/// Recorded hash and length for one run-relative artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactBinding<'a> {
    /// Normalized path relative to the run directory.
    pub relative_path: &'a Path,

    /// Canonical lowercase SHA-256 text.
    pub sha256: &'a str,

    /// Recorded artifact length.
    pub byte_length: u64,
}

/// Parsed and byte-verified artifact evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedArtifactBinding {
    /// Normalized path relative to the run directory.
    pub relative_path: PathBuf,

    /// Verified artifact digest.
    pub sha256: Sha256Digest,

    /// Verified byte length.
    pub byte_length: u64,
}

/// Verify a closed set of artifact hashes under one run directory.
///
/// Paths must be normalized relative UTF-8 paths. Canonicalization additionally
/// rejects symlinks that escape the run directory. Every listed file is hashed
/// before a validated binding is returned.
///
/// # Errors
///
/// Returns [`ArtifactConsistencyError`] for an empty set, unsafe or duplicate
/// paths, malformed hashes, non-files, I/O failures, or byte/hash mismatches.
pub fn validate_artifact_consistency(
    run_directory: impl AsRef<Path>,
    artifacts: &[ArtifactBinding<'_>],
) -> Result<Vec<ValidatedArtifactBinding>, ArtifactConsistencyError> {
    if artifacts.is_empty() {
        return Err(ArtifactConsistencyError::NoArtifacts);
    }

    let run_directory = run_directory.as_ref();
    let canonical_run = std::fs::canonicalize(run_directory)
        .map_err(|source| ArtifactConsistencyError::io(run_directory.to_path_buf(), source))?;
    let mut seen = HashSet::with_capacity(artifacts.len());
    let mut validated = Vec::with_capacity(artifacts.len());

    for artifact in artifacts {
        validate_relative_artifact_path(artifact.relative_path)?;
        let relative_path = artifact.relative_path.to_path_buf();
        if !seen.insert(relative_path.clone()) {
            return Err(ArtifactConsistencyError::DuplicatePath {
                path: relative_path,
            });
        }

        let expected_hash = Sha256Digest::from_hex(artifact.sha256).map_err(|source| {
            ArtifactConsistencyError::MalformedHash {
                path: relative_path.clone(),
                source,
            }
        })?;
        let joined = canonical_run.join(&relative_path);
        let canonical_path = std::fs::canonicalize(&joined)
            .map_err(|source| ArtifactConsistencyError::io(relative_path.clone(), source))?;
        if !canonical_path.starts_with(&canonical_run) {
            return Err(ArtifactConsistencyError::PathEscapesRunDirectory {
                path: relative_path,
            });
        }
        let metadata = canonical_path
            .metadata()
            .map_err(|source| ArtifactConsistencyError::io(relative_path.clone(), source))?;
        if !metadata.is_file() {
            return Err(ArtifactConsistencyError::NotRegularFile {
                path: relative_path,
            });
        }

        let (actual_hash, actual_length) = hash_file(&canonical_path)
            .map_err(|source| ArtifactConsistencyError::io(relative_path.clone(), source))?;
        if actual_length != artifact.byte_length {
            return Err(ArtifactConsistencyError::ByteLengthMismatch {
                path: relative_path,
                expected: artifact.byte_length,
                actual: actual_length,
            });
        }
        if actual_hash != expected_hash {
            return Err(ArtifactConsistencyError::HashMismatch {
                path: relative_path,
                expected: expected_hash,
                actual: actual_hash,
            });
        }

        validated.push(ValidatedArtifactBinding {
            relative_path,
            sha256: expected_hash,
            byte_length: actual_length,
        });
    }

    Ok(validated)
}

fn validate_relative_artifact_path(path: &Path) -> Result<(), ArtifactConsistencyError> {
    let text = path
        .to_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ArtifactConsistencyError::InvalidRelativePath {
            path: path.to_path_buf(),
        })?;
    let components_are_normal = path
        .components()
        .all(|component| matches!(component, Component::Normal(_)));
    if !components_are_normal || text.contains('\\') || text.chars().any(char::is_control) {
        return Err(ArtifactConsistencyError::InvalidRelativePath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

/// Caller-selected behavior when a run directory already exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionPolicy {
    /// Refuse to reuse any existing run directory.
    RejectExisting,

    /// Resume a specifically identified prior run.
    Resume,

    /// Destructively replace a specifically identified prior run.
    Overwrite,
}

impl fmt::Display for CollisionPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::RejectExisting => "reject-existing",
            Self::Resume => "resume",
            Self::Overwrite => "overwrite",
        })
    }
}

/// Human and collision-safe identity fields for one run directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunIdentityEvidence<'a> {
    /// Human-readable path segment.
    pub run_id: &'a str,

    /// Canonical UUID for this distinct run attempt.
    pub run_uuid: &'a str,

    /// Directory whose basename must equal `run_id`.
    pub run_directory: &'a Path,

    /// Explicit collision behavior.
    pub collision_policy: CollisionPolicy,

    /// UUID of the prior run when resuming or overwriting.
    pub prior_run_uuid: Option<&'a str>,
}

/// Parsed canonical UUID bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalUuid([u8; 16]);

impl CanonicalUuid {
    /// Borrow the 16 decoded UUID bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    fn parse(value: &str) -> Result<Self, UuidParseError> {
        let source = value.as_bytes();
        if source.len() != 36 {
            return Err(UuidParseError::InvalidLength {
                actual: source.len(),
            });
        }

        let mut bytes = [0_u8; 16];
        let mut nibble_index = 0_usize;
        for (index, &byte) in source.iter().enumerate() {
            if matches!(index, 8 | 13 | 18 | 23) {
                if byte != b'-' {
                    return Err(UuidParseError::InvalidHyphen { index });
                }
                continue;
            }
            let nibble = match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => return Err(UuidParseError::InvalidCharacter { index, byte }),
            };
            let output_index = nibble_index / 2;
            if nibble_index.is_multiple_of(2) {
                bytes[output_index] = nibble << 4;
            } else {
                bytes[output_index] |= nibble;
            }
            nibble_index += 1;
        }
        Ok(Self(bytes))
    }
}

/// Validate run identity and collision-policy joins.
///
/// # Errors
///
/// Returns [`RunIdentityError`] for unsafe IDs, mismatched directories,
/// malformed UUIDs, or inconsistent prior-run/collision fields.
pub fn validate_run_identity(
    evidence: RunIdentityEvidence<'_>,
) -> Result<ValidatedRunIdentity, RunIdentityError> {
    let run_id_path = Path::new(evidence.run_id);
    if evidence.run_id.is_empty()
        || evidence.run_id.contains('\\')
        || evidence.run_id.chars().any(char::is_control)
        || run_id_path.components().count() != 1
        || !matches!(run_id_path.components().next(), Some(Component::Normal(_)))
    {
        return Err(RunIdentityError::InvalidRunId);
    }
    if evidence
        .run_directory
        .file_name()
        .and_then(|name| name.to_str())
        != Some(evidence.run_id)
    {
        return Err(RunIdentityError::DirectoryRunIdMismatch);
    }

    let run_uuid = CanonicalUuid::parse(evidence.run_uuid).map_err(|source| {
        RunIdentityError::MalformedUuid {
            field: RunUuidField::Current,
            source,
        }
    })?;
    let prior_run_uuid =
        match evidence.prior_run_uuid {
            Some(value) => Some(CanonicalUuid::parse(value).map_err(|source| {
                RunIdentityError::MalformedUuid {
                    field: RunUuidField::Prior,
                    source,
                }
            })?),
            None => None,
        };

    match evidence.collision_policy {
        CollisionPolicy::RejectExisting if prior_run_uuid.is_some() => {
            return Err(RunIdentityError::UnexpectedPriorRunUuid);
        }
        CollisionPolicy::Resume | CollisionPolicy::Overwrite if prior_run_uuid.is_none() => {
            return Err(RunIdentityError::MissingPriorRunUuid {
                policy: evidence.collision_policy,
            });
        }
        _ => {}
    }
    if prior_run_uuid == Some(run_uuid) {
        return Err(RunIdentityError::ReusedRunUuid);
    }

    Ok(ValidatedRunIdentity {
        run_uuid,
        prior_run_uuid,
        collision_policy: evidence.collision_policy,
    })
}

/// Parsed collision-safe run identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedRunIdentity {
    /// Current run UUID.
    pub run_uuid: CanonicalUuid,

    /// Prior run UUID for resume/overwrite.
    pub prior_run_uuid: Option<CanonicalUuid>,

    /// Validated collision behavior.
    pub collision_policy: CollisionPolicy,
}

/// UUID field involved in run identity validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunUuidField {
    /// Current run UUID.
    Current,

    /// Prior run UUID.
    Prior,
}

impl fmt::Display for RunUuidField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Current => "current run",
            Self::Prior => "prior run",
        })
    }
}

/// Canonical UUID text is malformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UuidParseError {
    /// UUID text is not 36 bytes long.
    InvalidLength {
        /// Observed byte length.
        actual: usize,
    },

    /// A canonical hyphen is missing.
    InvalidHyphen {
        /// Expected hyphen index.
        index: usize,
    },

    /// A non-hyphen byte is not lowercase hexadecimal ASCII.
    InvalidCharacter {
        /// Rejected byte index.
        index: usize,

        /// Rejected byte.
        byte: u8,
    },
}

impl fmt::Display for UuidParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { actual } => {
                write!(formatter, "canonical UUID must be 36 bytes, got {actual}")
            }
            Self::InvalidHyphen { index } => {
                write!(
                    formatter,
                    "canonical UUID is missing a hyphen at index {index}"
                )
            }
            Self::InvalidCharacter { index, byte } => write!(
                formatter,
                "canonical UUID contains invalid byte {byte:#04x} at index {index}"
            ),
        }
    }
}

impl Error for UuidParseError {}

/// Run identity does not safely encode its collision behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunIdentityError {
    /// The run ID is not a safe single path segment.
    InvalidRunId,

    /// The run directory basename differs from the run ID.
    DirectoryRunIdMismatch,

    /// A run UUID is malformed.
    MalformedUuid {
        /// Invalid UUID field.
        field: RunUuidField,

        /// Parse failure.
        source: UuidParseError,
    },

    /// Reject-existing unexpectedly names a prior run.
    UnexpectedPriorRunUuid,

    /// Resume or overwrite did not name a prior run.
    MissingPriorRunUuid {
        /// Policy that requires a prior UUID.
        policy: CollisionPolicy,
    },

    /// The new attempt reused the prior run UUID.
    ReusedRunUuid,
}

impl fmt::Display for RunIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRunId => formatter.write_str("run ID must be one safe path segment"),
            Self::DirectoryRunIdMismatch => {
                formatter.write_str("run directory basename must equal the run ID")
            }
            Self::MalformedUuid { field, source } => {
                write!(formatter, "malformed {field} UUID: {source}")
            }
            Self::UnexpectedPriorRunUuid => {
                formatter.write_str("reject-existing must not name a prior run UUID")
            }
            Self::MissingPriorRunUuid { policy } => {
                write!(formatter, "{policy} requires a prior run UUID")
            }
            Self::ReusedRunUuid => formatter.write_str("current and prior run UUIDs must differ"),
        }
    }
}

impl Error for RunIdentityError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MalformedUuid { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Self-reported conditions and hashes required for a qualifying training run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualificationEvidence<'a> {
    /// Whether observations, actions, rewards, dynamics, and limits are official.
    pub official_semantics: bool,

    /// Whether the run altered the official reward.
    pub reward_shaping: bool,

    /// Whether a non-learned fallback chose actions.
    pub heuristic_fallback: bool,

    /// Whether behavior cloning contributed policy updates.
    pub behavior_cloning: bool,

    /// Number of warmup imitation updates.
    pub warmup_imitation_updates: u64,

    /// Whether every seed recorded by the run was actually applied.
    pub all_recorded_seeds_applied: bool,

    /// Number of neural optimizer updates.
    pub optimizer_updates: u64,

    /// Number of tabular Q-value updates.
    pub q_table_updates: u64,

    /// Canonical SHA-256 text for the saved step-zero policy.
    pub step_zero_policy_sha256: &'a str,

    /// Canonical SHA-256 text for the validation-selected best policy.
    pub best_policy_sha256: &'a str,
}

impl QualificationEvidence<'_> {
    /// Validate that the evidence proves real, unassisted policy updates.
    ///
    /// # Errors
    ///
    /// Returns [`QualificationError`] for any disallowed training shortcut,
    /// missing seed application or update, malformed hash, or unchanged policy.
    pub fn validate(self) -> Result<ValidatedQualification, QualificationError> {
        if !self.official_semantics {
            return Err(QualificationError::OfficialSemanticsDisabled);
        }
        if self.reward_shaping {
            return Err(QualificationError::RewardShapingEnabled);
        }
        if self.heuristic_fallback {
            return Err(QualificationError::HeuristicFallbackEnabled);
        }
        if self.behavior_cloning {
            return Err(QualificationError::BehaviorCloningEnabled);
        }
        if self.warmup_imitation_updates != 0 {
            return Err(QualificationError::WarmupImitationEnabled {
                update_count: self.warmup_imitation_updates,
            });
        }
        if !self.all_recorded_seeds_applied {
            return Err(QualificationError::RecordedSeedsNotApplied);
        }

        let combined_update_count = self
            .optimizer_updates
            .checked_add(self.q_table_updates)
            .ok_or(QualificationError::UpdateCountOverflow)?;
        if combined_update_count == 0 {
            return Err(QualificationError::ZeroUpdates);
        }

        let step_zero_policy_sha256 = Sha256Digest::from_hex(self.step_zero_policy_sha256)
            .map_err(|source| QualificationError::MalformedHash {
                field: QualificationHashField::StepZeroPolicy,
                source,
            })?;
        let best_policy_sha256 =
            Sha256Digest::from_hex(self.best_policy_sha256).map_err(|source| {
                QualificationError::MalformedHash {
                    field: QualificationHashField::BestPolicy,
                    source,
                }
            })?;
        if step_zero_policy_sha256 == best_policy_sha256 {
            return Err(QualificationError::UnchangedPolicyHash);
        }

        Ok(ValidatedQualification {
            combined_update_count,
            step_zero_policy_sha256,
            best_policy_sha256,
        })
    }
}

/// Parsed qualification facts safe to place in a proof receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedQualification {
    /// Optimizer plus Q-table updates.
    pub combined_update_count: u64,

    /// Parsed step-zero policy hash.
    pub step_zero_policy_sha256: Sha256Digest,

    /// Parsed validation-selected best policy hash.
    pub best_policy_sha256: Sha256Digest,
}

/// Policy hash field involved in qualification validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationHashField {
    /// Step-zero policy hash.
    StepZeroPolicy,

    /// Validation-selected best policy hash.
    BestPolicy,
}

impl fmt::Display for QualificationHashField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::StepZeroPolicy => "step-zero policy",
            Self::BestPolicy => "best policy",
        })
    }
}

/// A run cannot qualify as genuine learning proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationError {
    /// Official environment semantics were not used.
    OfficialSemanticsDisabled,

    /// Reward shaping was enabled.
    RewardShapingEnabled,

    /// A heuristic action fallback was enabled.
    HeuristicFallbackEnabled,

    /// Behavior cloning was enabled.
    BehaviorCloningEnabled,

    /// Warmup imitation performed policy updates.
    WarmupImitationEnabled {
        /// Number of disallowed updates.
        update_count: u64,
    },

    /// At least one recorded seed was not applied.
    RecordedSeedsNotApplied,

    /// Optimizer and Q-table update counts sum to zero.
    ZeroUpdates,

    /// Combining optimizer and Q-table update counts overflowed `u64`.
    UpdateCountOverflow,

    /// A policy hash is not canonical lowercase SHA-256 text.
    MalformedHash {
        /// Invalid field.
        field: QualificationHashField,

        /// Parse failure.
        source: DigestParseError,
    },

    /// Step-zero and best policy hashes are identical.
    UnchangedPolicyHash,
}

impl fmt::Display for QualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OfficialSemanticsDisabled => {
                formatter.write_str("qualification requires official environment semantics")
            }
            Self::RewardShapingEnabled => {
                formatter.write_str("qualification forbids reward shaping")
            }
            Self::HeuristicFallbackEnabled => {
                formatter.write_str("qualification forbids heuristic fallback")
            }
            Self::BehaviorCloningEnabled => {
                formatter.write_str("qualification forbids behavior cloning")
            }
            Self::WarmupImitationEnabled { update_count } => write!(
                formatter,
                "qualification forbids {update_count} warmup imitation updates"
            ),
            Self::RecordedSeedsNotApplied => {
                formatter.write_str("qualification requires every recorded seed to be applied")
            }
            Self::ZeroUpdates => {
                formatter.write_str("qualification requires at least one real policy update")
            }
            Self::UpdateCountOverflow => {
                formatter.write_str("combined policy update count overflows u64")
            }
            Self::MalformedHash { field, source } => {
                write!(formatter, "malformed {field} hash: {source}")
            }
            Self::UnchangedPolicyHash => {
                formatter.write_str("step-zero and best policy hashes must differ after training")
            }
        }
    }
}

impl Error for QualificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MalformedHash { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Fixed evaluation-suite identity and ordered seed values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuiteEvidence<'a> {
    /// Stable suite identifier.
    pub id: &'a str,

    /// Canonical hash of the complete suite artifact.
    pub sha256: &'a str,

    /// Exact reset seeds in evaluation order.
    pub seeds: &'a [u64],
}

/// Evaluation split involved in proof validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofSuite {
    /// Validation selects the best checkpoint.
    Validation,

    /// Test is used only for final held-out evaluation.
    Test,
}

impl fmt::Display for ProofSuite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Validation => "validation",
            Self::Test => "test",
        })
    }
}

/// Parsed evidence that validation and test suites are different and disjoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedSuiteSeparation {
    /// Validation-suite artifact hash.
    pub validation_sha256: Sha256Digest,

    /// Test-suite artifact hash.
    pub test_sha256: Sha256Digest,

    /// Number of validation seeds.
    pub validation_seed_count: usize,

    /// Number of test seeds.
    pub test_seed_count: usize,
}

/// Validate fixed validation/test identity, hashes, and seed disjointness.
///
/// # Errors
///
/// Returns [`SuiteSeparationError`] when either suite is empty or malformed,
/// contains duplicate seeds, aliases the other suite, or shares a seed.
pub fn validate_suite_separation(
    validation: SuiteEvidence<'_>,
    test: SuiteEvidence<'_>,
) -> Result<ValidatedSuiteSeparation, SuiteSeparationError> {
    validate_one_suite(ProofSuite::Validation, validation)?;
    validate_one_suite(ProofSuite::Test, test)?;
    if validation.id == test.id {
        return Err(SuiteSeparationError::SameSuiteId);
    }

    let validation_sha256 = Sha256Digest::from_hex(validation.sha256).map_err(|source| {
        SuiteSeparationError::MalformedHash {
            suite: ProofSuite::Validation,
            source,
        }
    })?;
    let test_sha256 = Sha256Digest::from_hex(test.sha256).map_err(|source| {
        SuiteSeparationError::MalformedHash {
            suite: ProofSuite::Test,
            source,
        }
    })?;
    if validation_sha256 == test_sha256 {
        return Err(SuiteSeparationError::SameSuiteHash);
    }

    let validation_seeds: HashSet<u64> = validation.seeds.iter().copied().collect();
    if let Some(seed) = test
        .seeds
        .iter()
        .copied()
        .find(|seed| validation_seeds.contains(seed))
    {
        return Err(SuiteSeparationError::OverlappingSeed { seed });
    }

    Ok(ValidatedSuiteSeparation {
        validation_sha256,
        test_sha256,
        validation_seed_count: validation.seeds.len(),
        test_seed_count: test.seeds.len(),
    })
}

fn validate_one_suite(
    suite: ProofSuite,
    evidence: SuiteEvidence<'_>,
) -> Result<(), SuiteSeparationError> {
    if evidence.id.is_empty() {
        return Err(SuiteSeparationError::EmptySuiteId { suite });
    }
    if evidence.seeds.is_empty() {
        return Err(SuiteSeparationError::EmptySuite { suite });
    }

    let mut unique = HashSet::with_capacity(evidence.seeds.len());
    for &seed in evidence.seeds {
        if !unique.insert(seed) {
            return Err(SuiteSeparationError::DuplicateSeed { suite, seed });
        }
    }
    Ok(())
}

/// Fixed validation and test suites are not demonstrably separate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuiteSeparationError {
    /// A suite identifier is empty.
    EmptySuiteId {
        /// Invalid suite.
        suite: ProofSuite,
    },

    /// A suite contains no seeds.
    EmptySuite {
        /// Invalid suite.
        suite: ProofSuite,
    },

    /// A suite repeats a seed internally.
    DuplicateSeed {
        /// Invalid suite.
        suite: ProofSuite,

        /// Repeated seed.
        seed: u64,
    },

    /// Validation and test use the same suite identifier.
    SameSuiteId,

    /// A suite artifact hash is malformed.
    MalformedHash {
        /// Invalid suite.
        suite: ProofSuite,

        /// Parse failure.
        source: DigestParseError,
    },

    /// Validation and test use the same suite artifact hash.
    SameSuiteHash,

    /// Validation and test share a reset seed.
    OverlappingSeed {
        /// Shared seed.
        seed: u64,
    },
}

impl fmt::Display for SuiteSeparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySuiteId { suite } => write!(formatter, "{suite} suite ID is empty"),
            Self::EmptySuite { suite } => write!(formatter, "{suite} suite contains no seeds"),
            Self::DuplicateSeed { suite, seed } => {
                write!(formatter, "{suite} suite repeats seed {seed}")
            }
            Self::SameSuiteId => formatter.write_str("validation and test suite IDs must differ"),
            Self::MalformedHash { suite, source } => {
                write!(formatter, "malformed {suite} suite hash: {source}")
            }
            Self::SameSuiteHash => {
                formatter.write_str("validation and test suite hashes must differ")
            }
            Self::OverlappingSeed { seed } => {
                write!(formatter, "validation and test suites share seed {seed}")
            }
        }
    }
}

impl Error for SuiteSeparationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MalformedHash { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Evidence emitted by an evaluation command that reloaded a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreshProcessReloadEvidence<'a> {
    /// Whether evaluation ran in a newly started process.
    pub fresh_process: bool,

    /// Evaluation process exit code.
    pub exit_code: i32,

    /// Whether the process loaded checkpoint bytes from disk.
    pub checkpoint_loaded_from_disk: bool,

    /// Hash reported for the bytes loaded by the evaluation process.
    pub checkpoint_sha256: &'a str,
}

/// Parsed successful reload evidence bound to one checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidatedFreshProcessReload {
    /// Exact checkpoint digest loaded by the fresh process.
    pub checkpoint_sha256: Sha256Digest,
}

/// Validate successful fresh-process reload of exact, unchanged checkpoint bytes.
///
/// # Errors
///
/// Returns [`FreshProcessReloadError`] unless evaluation ran successfully in a
/// fresh process, loaded from disk, reported the selected checkpoint hash, and
/// the checkpoint file still matches its recorded byte evidence.
pub fn validate_fresh_process_reload(
    checkpoint: &CheckpointEvidence,
    evidence: FreshProcessReloadEvidence<'_>,
) -> Result<ValidatedFreshProcessReload, FreshProcessReloadError> {
    if !evidence.fresh_process {
        return Err(FreshProcessReloadError::NotFreshProcess);
    }
    if evidence.exit_code != 0 {
        return Err(FreshProcessReloadError::NonZeroExit {
            exit_code: evidence.exit_code,
        });
    }
    if !evidence.checkpoint_loaded_from_disk {
        return Err(FreshProcessReloadError::CheckpointNotLoaded);
    }

    let checkpoint_sha256 = Sha256Digest::from_hex(evidence.checkpoint_sha256)
        .map_err(|source| FreshProcessReloadError::MalformedCheckpointHash { source })?;
    if checkpoint_sha256 != checkpoint.sha256 {
        return Err(FreshProcessReloadError::CheckpointHashMismatch {
            expected: checkpoint.sha256,
            actual: checkpoint_sha256,
        });
    }
    checkpoint
        .verify_file()
        .map_err(|source| FreshProcessReloadError::ArtifactChanged { source })?;

    Ok(ValidatedFreshProcessReload { checkpoint_sha256 })
}

/// Final evaluation did not prove an exact fresh-process checkpoint reload.
#[derive(Debug)]
pub enum FreshProcessReloadError {
    /// Evaluation reused the training process.
    NotFreshProcess,

    /// The evaluation process failed.
    NonZeroExit {
        /// Observed process exit code.
        exit_code: i32,
    },

    /// The evaluator did not load a checkpoint from disk.
    CheckpointNotLoaded,

    /// The evaluator reported malformed SHA-256 text.
    MalformedCheckpointHash {
        /// Parse failure.
        source: DigestParseError,
    },

    /// The evaluator loaded different checkpoint bytes.
    CheckpointHashMismatch {
        /// Validation-selected checkpoint hash.
        expected: Sha256Digest,

        /// Evaluator-reported checkpoint hash.
        actual: Sha256Digest,
    },

    /// The selected checkpoint file changed or became unreadable.
    ArtifactChanged {
        /// Artifact verification failure.
        source: ArtifactConsistencyError,
    },
}

impl fmt::Display for FreshProcessReloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFreshProcess => {
                formatter.write_str("held-out evaluation must run in a fresh process")
            }
            Self::NonZeroExit { exit_code } => {
                write!(
                    formatter,
                    "held-out evaluation exited with code {exit_code}"
                )
            }
            Self::CheckpointNotLoaded => {
                formatter.write_str("held-out evaluation did not load checkpoint bytes from disk")
            }
            Self::MalformedCheckpointHash { source } => {
                write!(
                    formatter,
                    "held-out evaluation reported a malformed hash: {source}"
                )
            }
            Self::CheckpointHashMismatch { expected, actual } => write!(
                formatter,
                "held-out evaluation loaded {}, expected {}",
                actual.to_hex(),
                expected.to_hex()
            ),
            Self::ArtifactChanged { source } => {
                write!(
                    formatter,
                    "selected checkpoint no longer verifies: {source}"
                )
            }
        }
    }
}

impl Error for FreshProcessReloadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::MalformedCheckpointHash { source } => Some(source),
            Self::ArtifactChanged { source } => Some(source),
            _ => None,
        }
    }
}

/// Incremental SHA-256 state used for file hashing.
struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    length_bytes: u64,
}

impl Sha256 {
    const fn new() -> Self {
        Self {
            state: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            length_bytes: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        self.length_bytes = self
            .length_bytes
            .wrapping_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));

        for &byte in bytes {
            self.buffer[self.buffer_len] = byte;
            self.buffer_len += 1;
            if self.buffer_len == self.buffer.len() {
                let block = self.buffer;
                self.compress(&block);
                self.buffer_len = 0;
            }
        }
    }

    fn finalize(mut self) -> Sha256Digest {
        let bit_length = self.length_bytes.wrapping_mul(8);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..].fill(0);
            let block = self.buffer;
            self.compress(&block);
            self.buffer_len = 0;
        }

        self.buffer[self.buffer_len..56].fill(0);
        self.buffer[56..].copy_from_slice(&bit_length.to_be_bytes());
        let block = self.buffer;
        self.compress(&block);

        let mut digest = [0; 32];
        for (chunk, word) in digest.chunks_exact_mut(4).zip(self.state) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        Sha256Digest(digest)
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const ROUND_CONSTANTS: [u32; 64] = [
            0x428a_2f98,
            0x7137_4491,
            0xb5c0_fbcf,
            0xe9b5_dba5,
            0x3956_c25b,
            0x59f1_11f1,
            0x923f_82a4,
            0xab1c_5ed5,
            0xd807_aa98,
            0x1283_5b01,
            0x2431_85be,
            0x550c_7dc3,
            0x72be_5d74,
            0x80de_b1fe,
            0x9bdc_06a7,
            0xc19b_f174,
            0xe49b_69c1,
            0xefbe_4786,
            0x0fc1_9dc6,
            0x240c_a1cc,
            0x2de9_2c6f,
            0x4a74_84aa,
            0x5cb0_a9dc,
            0x76f9_88da,
            0x983e_5152,
            0xa831_c66d,
            0xb003_27c8,
            0xbf59_7fc7,
            0xc6e0_0bf3,
            0xd5a7_9147,
            0x06ca_6351,
            0x1429_2967,
            0x27b7_0a85,
            0x2e1b_2138,
            0x4d2c_6dfc,
            0x5338_0d13,
            0x650a_7354,
            0x766a_0abb,
            0x81c2_c92e,
            0x9272_2c85,
            0xa2bf_e8a1,
            0xa81a_664b,
            0xc24b_8b70,
            0xc76c_51a3,
            0xd192_e819,
            0xd699_0624,
            0xf40e_3585,
            0x106a_a070,
            0x19a4_c116,
            0x1e37_6c08,
            0x2748_774c,
            0x34b0_bcb5,
            0x391c_0cb3,
            0x4ed8_aa4a,
            0x5b9c_ca4f,
            0x682e_6ff3,
            0x748f_82ee,
            0x78a5_636f,
            0x84c8_7814,
            0x8cc7_0208,
            0x90be_fffa,
            0xa450_6ceb,
            0xbef9_a3f7,
            0xc671_78f2,
        ];

        let mut schedule = [0_u32; 64];
        for (index, chunk) in block.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for index in 16..64 {
            let small_sigma_0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let small_sigma_1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(small_sigma_0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(small_sigma_1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for index in 0..64 {
            let big_sigma_1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary_1 = h
                .wrapping_add(big_sigma_1)
                .wrapping_add(choose)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let big_sigma_0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary_2 = big_sigma_0.wrapping_add(majority);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary_1);
            d = c;
            c = b;
            b = a;
            a = temporary_1.wrapping_add(temporary_2);
        }

        for (state, working) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(working);
        }
    }
}

#[cfg(test)]
#[expect(
    clippy::disallowed_methods,
    reason = "std-only tests create and remove isolated temporary proof files"
)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256_bytes(b"").to_hex(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_bytes(b"abc").to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_bytes(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").to_hex(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_bytes(&vec![b'a'; 1_000_000]).to_hex(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn digest_parser_rejects_noncanonical_hashes() {
        let canonical = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(
            Sha256Digest::from_hex(canonical)
                .expect("standard vector is canonical")
                .to_hex(),
            canonical
        );
        assert_eq!(
            Sha256Digest::from_hex("abc"),
            Err(DigestParseError::InvalidLength { actual: 3 })
        );
        assert_eq!(
            Sha256Digest::from_hex(
                "BA7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            ),
            Err(DigestParseError::InvalidCharacter {
                index: 0,
                byte: b'B'
            })
        );
    }

    #[test]
    fn file_hash_matches_standard_vector() {
        let path = test_path("sha256-file");
        drop(fs::remove_file(&path));
        fs::write(&path, b"abc").expect("test file writes");

        let digest = sha256_file(&path).expect("test file hashes");

        assert_eq!(
            digest.to_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        drop(fs::remove_file(path));
    }

    #[test]
    fn checkpoint_evidence_hashes_and_verifies_exact_file_bytes() {
        let path = test_path("checkpoint-evidence");
        drop(fs::remove_file(&path));
        fs::write(&path, b"checkpoint-bytes").expect("checkpoint writes");

        let evidence = CheckpointEvidence::from_file(&path, 80_000, 710, 79_000)
            .expect("checkpoint evidence builds");

        assert_eq!(evidence.path, path);
        assert_eq!(evidence.byte_length, 16);
        assert_eq!(evidence.global_step, 80_000);
        assert_eq!(evidence.episode_count, 710);
        assert_eq!(evidence.update_count, 79_000);
        evidence
            .verify_file()
            .expect("unchanged checkpoint verifies");

        fs::write(&evidence.path, b"checkpoInt-bytes").expect("same-length mutation writes");
        assert!(matches!(
            evidence.verify_file(),
            Err(ArtifactConsistencyError::HashMismatch { .. })
        ));

        fs::write(&evidence.path, b"short").expect("short mutation writes");
        assert!(matches!(
            evidence.verify_file(),
            Err(ArtifactConsistencyError::ByteLengthMismatch { .. })
        ));
        drop(fs::remove_file(path));
    }

    #[test]
    fn checkpoint_evidence_rejects_an_empty_checkpoint() {
        let path = test_path("empty-checkpoint");
        drop(fs::remove_file(&path));
        fs::write(&path, b"").expect("empty checkpoint writes");

        assert!(matches!(
            CheckpointEvidence::from_file(&path, 0, 0, 0),
            Err(ArtifactConsistencyError::EmptyCheckpoint { .. })
        ));
        drop(fs::remove_file(path));
    }

    #[test]
    fn qualification_accepts_only_real_unassisted_policy_updates() {
        let valid = valid_qualification();
        let qualification = valid.validate().expect("qualifying evidence passes");

        assert_eq!(qualification.combined_update_count, 79_000);
        assert_eq!(
            qualification.step_zero_policy_sha256.to_hex(),
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
        assert_eq!(
            qualification.best_policy_sha256.to_hex(),
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        );
    }

    #[test]
    fn qualification_rejects_each_disallowed_training_shortcut() {
        let mut evidence = valid_qualification();
        evidence.official_semantics = false;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::OfficialSemanticsDisabled)
        );

        let mut evidence = valid_qualification();
        evidence.reward_shaping = true;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::RewardShapingEnabled)
        );

        let mut evidence = valid_qualification();
        evidence.heuristic_fallback = true;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::HeuristicFallbackEnabled)
        );

        let mut evidence = valid_qualification();
        evidence.behavior_cloning = true;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::BehaviorCloningEnabled)
        );

        let mut evidence = valid_qualification();
        evidence.warmup_imitation_updates = 1;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::WarmupImitationEnabled { update_count: 1 })
        );

        let mut evidence = valid_qualification();
        evidence.all_recorded_seeds_applied = false;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::RecordedSeedsNotApplied)
        );
    }

    #[test]
    fn qualification_rejects_missing_updates_and_invalid_policy_hashes() {
        let mut evidence = valid_qualification();
        evidence.optimizer_updates = 0;
        assert_eq!(evidence.validate(), Err(QualificationError::ZeroUpdates));

        let mut evidence = valid_qualification();
        evidence.optimizer_updates = u64::MAX;
        evidence.q_table_updates = 1;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::UpdateCountOverflow)
        );

        let mut evidence = valid_qualification();
        evidence.step_zero_policy_sha256 = "bad";
        assert!(matches!(
            evidence.validate(),
            Err(QualificationError::MalformedHash {
                field: QualificationHashField::StepZeroPolicy,
                ..
            })
        ));

        let mut evidence = valid_qualification();
        evidence.best_policy_sha256 = evidence.step_zero_policy_sha256;
        assert_eq!(
            evidence.validate(),
            Err(QualificationError::UnchangedPolicyHash)
        );
    }

    #[test]
    fn validation_and_test_suites_must_be_fixed_and_disjoint() {
        let validation = SuiteEvidence {
            id: "cartpole-validation-v1",
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            seeds: &[100, 101, 102],
        };
        let test = SuiteEvidence {
            id: "cartpole-test-v1",
            sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            seeds: &[200, 201, 202],
        };

        let separation = validate_suite_separation(validation, test)
            .expect("separate fixed suites are accepted");

        assert_eq!(separation.validation_seed_count, 3);
        assert_eq!(separation.test_seed_count, 3);
        assert_ne!(separation.validation_sha256, separation.test_sha256);
    }

    #[test]
    fn validation_and_test_suites_reject_aliases_and_seed_overlap() {
        let validation = SuiteEvidence {
            id: "validation",
            sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            seeds: &[1, 2, 3],
        };
        let mut test = SuiteEvidence {
            id: "test",
            sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            seeds: &[4, 5, 6],
        };

        test.id = validation.id;
        assert_eq!(
            validate_suite_separation(validation, test),
            Err(SuiteSeparationError::SameSuiteId)
        );

        test.id = "test";
        test.sha256 = validation.sha256;
        assert_eq!(
            validate_suite_separation(validation, test),
            Err(SuiteSeparationError::SameSuiteHash)
        );

        test.sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        test.seeds = &[3, 4, 5];
        assert_eq!(
            validate_suite_separation(validation, test),
            Err(SuiteSeparationError::OverlappingSeed { seed: 3 })
        );

        test.seeds = &[4, 4];
        assert_eq!(
            validate_suite_separation(validation, test),
            Err(SuiteSeparationError::DuplicateSeed {
                suite: ProofSuite::Test,
                seed: 4
            })
        );
    }

    #[test]
    fn fresh_process_reload_binds_the_exact_checkpoint_bytes() {
        let path = test_path("fresh-reload");
        drop(fs::remove_file(&path));
        fs::write(&path, b"selected-best-checkpoint").expect("checkpoint writes");
        let checkpoint = CheckpointEvidence::from_file(&path, 80_000, 710, 79_000)
            .expect("checkpoint evidence builds");
        let checkpoint_hash = checkpoint.sha256.to_hex();
        let evidence = FreshProcessReloadEvidence {
            fresh_process: true,
            exit_code: 0,
            checkpoint_loaded_from_disk: true,
            checkpoint_sha256: &checkpoint_hash,
        };

        let validated = validate_fresh_process_reload(&checkpoint, evidence)
            .expect("matching fresh reload passes");

        assert_eq!(validated.checkpoint_sha256, checkpoint.sha256);

        fs::write(&path, b"changed-best-checkpoint").expect("mutation writes");
        assert!(matches!(
            validate_fresh_process_reload(&checkpoint, evidence),
            Err(FreshProcessReloadError::ArtifactChanged { .. })
        ));
        drop(fs::remove_file(path));
    }

    #[test]
    fn fresh_process_reload_rejects_process_and_hash_mismatches() {
        let path = test_path("fresh-reload-errors");
        drop(fs::remove_file(&path));
        fs::write(&path, b"checkpoint").expect("checkpoint writes");
        let checkpoint =
            CheckpointEvidence::from_file(&path, 10, 2, 9).expect("checkpoint evidence builds");
        let checkpoint_hash = checkpoint.sha256.to_hex();
        let mut evidence = FreshProcessReloadEvidence {
            fresh_process: true,
            exit_code: 0,
            checkpoint_loaded_from_disk: true,
            checkpoint_sha256: &checkpoint_hash,
        };

        evidence.fresh_process = false;
        assert!(matches!(
            validate_fresh_process_reload(&checkpoint, evidence),
            Err(FreshProcessReloadError::NotFreshProcess)
        ));

        evidence.fresh_process = true;
        evidence.exit_code = 1;
        assert!(matches!(
            validate_fresh_process_reload(&checkpoint, evidence),
            Err(FreshProcessReloadError::NonZeroExit { exit_code: 1 })
        ));

        evidence.exit_code = 0;
        evidence.checkpoint_loaded_from_disk = false;
        assert!(matches!(
            validate_fresh_process_reload(&checkpoint, evidence),
            Err(FreshProcessReloadError::CheckpointNotLoaded)
        ));

        evidence.checkpoint_loaded_from_disk = true;
        evidence.checkpoint_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert!(matches!(
            validate_fresh_process_reload(&checkpoint, evidence),
            Err(FreshProcessReloadError::CheckpointHashMismatch { .. })
        ));

        evidence.checkpoint_sha256 = "bad";
        assert!(matches!(
            validate_fresh_process_reload(&checkpoint, evidence),
            Err(FreshProcessReloadError::MalformedCheckpointHash { .. })
        ));
        drop(fs::remove_file(path));
    }

    #[test]
    fn run_identity_enforces_collision_policy_and_directory_binding() {
        let run_dir =
            std::path::Path::new("runs/cartpole-dqn/cartpole-proof-20260711t120000z-a1b2c3d4e5f6");
        let valid = RunIdentityEvidence {
            run_id: "cartpole-proof-20260711t120000z-a1b2c3d4e5f6",
            run_uuid: "018f1a56-3779-7701-93ef-fa1398226d17",
            run_directory: run_dir,
            collision_policy: CollisionPolicy::RejectExisting,
            prior_run_uuid: None,
        };

        validate_run_identity(valid).expect("new collision-safe identity passes");

        let mut invalid = valid;
        invalid.prior_run_uuid = Some("018f1a56-3779-7701-93ef-fa1398226d18");
        assert!(matches!(
            validate_run_identity(invalid),
            Err(RunIdentityError::UnexpectedPriorRunUuid)
        ));

        invalid = valid;
        invalid.collision_policy = CollisionPolicy::Resume;
        assert!(matches!(
            validate_run_identity(invalid),
            Err(RunIdentityError::MissingPriorRunUuid { .. })
        ));

        invalid.prior_run_uuid = Some(valid.run_uuid);
        assert!(matches!(
            validate_run_identity(invalid),
            Err(RunIdentityError::ReusedRunUuid)
        ));

        invalid = valid;
        invalid.run_id = "different-id";
        assert!(matches!(
            validate_run_identity(invalid),
            Err(RunIdentityError::DirectoryRunIdMismatch)
        ));
    }

    #[test]
    fn artifact_set_stays_under_run_directory_and_verifies_all_bytes() {
        let run_dir = test_path("artifact-run");
        drop(fs::remove_dir_all(&run_dir));
        fs::create_dir_all(&run_dir).expect("run directory creates");
        fs::write(run_dir.join("config.json"), b"config").expect("config writes");
        fs::create_dir_all(run_dir.join("checkpoints")).expect("checkpoint directory creates");
        fs::write(run_dir.join("checkpoints/best.mpk"), b"best").expect("checkpoint writes");

        let config_hash = sha256_bytes(b"config").to_hex();
        let checkpoint_hash = sha256_bytes(b"best").to_hex();
        let artifacts = [
            ArtifactBinding {
                relative_path: std::path::Path::new("config.json"),
                sha256: &config_hash,
                byte_length: 6,
            },
            ArtifactBinding {
                relative_path: std::path::Path::new("checkpoints/best.mpk"),
                sha256: &checkpoint_hash,
                byte_length: 4,
            },
        ];

        let validated = validate_artifact_consistency(&run_dir, &artifacts)
            .expect("consistent in-run artifacts pass");
        assert_eq!(validated.len(), 2);

        let duplicate = [artifacts[0], artifacts[0]];
        assert!(matches!(
            validate_artifact_consistency(&run_dir, &duplicate),
            Err(ArtifactConsistencyError::DuplicatePath { .. })
        ));

        let escaped = [ArtifactBinding {
            relative_path: std::path::Path::new("../outside.json"),
            sha256: &config_hash,
            byte_length: 6,
        }];
        assert!(matches!(
            validate_artifact_consistency(&run_dir, &escaped),
            Err(ArtifactConsistencyError::InvalidRelativePath { .. })
        ));

        let wrong_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let mismatched = [ArtifactBinding {
            relative_path: std::path::Path::new("config.json"),
            sha256: wrong_hash,
            byte_length: 6,
        }];
        assert!(matches!(
            validate_artifact_consistency(&run_dir, &mismatched),
            Err(ArtifactConsistencyError::HashMismatch { .. })
        ));
        drop(fs::remove_dir_all(run_dir));
    }

    fn valid_qualification() -> QualificationEvidence<'static> {
        QualificationEvidence {
            official_semantics: true,
            reward_shaping: false,
            heuristic_fallback: false,
            behavior_cloning: false,
            warmup_imitation_updates: 0,
            all_recorded_seeds_applied: true,
            optimizer_updates: 79_000,
            q_table_updates: 0,
            step_zero_policy_sha256:
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            best_policy_sha256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
        }
    }

    fn test_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("bevy-gym-proof-{}-{name}", std::process::id()))
    }
}
