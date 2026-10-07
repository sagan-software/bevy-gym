//! Typed manifest for curated browser checkpoints.

use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Maximum accepted checkpoint size in bytes.
pub(super) const MAX_CHECKPOINT_BYTES: u32 = 8 * 1024 * 1024;

/// Supported browser manifest format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(into = "u8")]
pub(super) enum ManifestVersion {
    /// Initial six-stage ecosystem manifest.
    V1,
}

impl From<ManifestVersion> for u8 {
    fn from(_value: ManifestVersion) -> Self {
        1
    }
}

impl<'de> Deserialize<'de> for ManifestVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u8::deserialize(deserializer)?;
        if value == 1 {
            Ok(Self::V1)
        } else {
            Err(serde::de::Error::custom("manifest_version must be 1"))
        }
    }
}

/// Closed ecosystem navigation and checkpoint vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Stage {
    /// Single bunny learning food acquisition.
    Forage,
    /// Single bunny balancing food and water.
    Survival,
    /// Single bunny using shelter during weather.
    Shelter,
    /// Multiple bunnies sharing limited resources.
    Competition,
    /// Bunnies and foxes using separate policies.
    PredatorPrey,
    /// Predator-prey with solid obstacles and thorns.
    Obstacles,
}

impl Stage {
    /// Every supported route in curriculum order.
    pub(super) const ALL: [Self; 6] = [
        Self::Forage,
        Self::Survival,
        Self::Shelter,
        Self::Competition,
        Self::PredatorPrey,
        Self::Obstacles,
    ];

    /// Return the stable hash-route key.
    pub(super) const fn as_key(self) -> &'static str {
        match self {
            Self::Forage => "forage",
            Self::Survival => "survival",
            Self::Shelter => "shelter",
            Self::Competition => "competition",
            Self::PredatorPrey => "predator-prey",
            Self::Obstacles => "obstacles",
        }
    }

    /// Return the short interface title.
    pub(super) const fn title(self) -> &'static str {
        match self {
            Self::Forage => "Forage",
            Self::Survival => "Survival",
            Self::Shelter => "Shelter",
            Self::Competition => "Competition",
            Self::PredatorPrey => "Predator-prey",
            Self::Obstacles => "Obstacles",
        }
    }

    /// Return whether this stage needs a separate fox checkpoint.
    pub(super) const fn requires_fox(self) -> bool {
        matches!(self, Self::PredatorPrey | Self::Obstacles)
    }

    /// Return the one-based curriculum position.
    pub(super) const fn position(self) -> usize {
        match self {
            Self::Forage => 1,
            Self::Survival => 2,
            Self::Shelter => 3,
            Self::Competition => 4,
            Self::PredatorPrey => 5,
            Self::Obstacles => 6,
        }
    }
}

impl FromStr for Stage {
    type Err = ManifestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|stage| stage.as_key() == value)
            .ok_or_else(|| ManifestError::UnknownStage(value.into()))
    }
}

/// Evidence boundary displayed beside a curated policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Qualification {
    /// Meets every declared stage gate across the required seeds.
    Qualified,
    /// Strongest load-compatible checkpoint when full qualification is absent.
    BestCompatibleAvailable,
}

impl fmt::Display for Qualification {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Qualified => "qualified",
            Self::BestCompatibleAvailable => "best compatible available",
        })
    }
}

impl FromStr for Qualification {
    type Err = ManifestError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "qualified" => Ok(Self::Qualified),
            "best-compatible-available" => Ok(Self::BestCompatibleAvailable),
            _ => Err(ManifestError::UnknownQualification(value.into())),
        }
    }
}

/// Inference algorithm encoded by every curated policy record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Algorithm {
    /// Burn-backed recurrent proximal policy optimization.
    RecurrentPpo,
}

/// Policy roles required by one ecosystem stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum PolicyRole {
    /// Herbivore policy shared by every bunny.
    Bunny,
    /// Predator policy shared by every fox.
    Fox,
}

/// Burn release that controls named `MessagePack` record compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(super) enum BurnVersion {
    /// Burn 0.21.0 record and Flex inference behavior.
    #[serde(rename = "0.21.0")]
    V0_21_0,
}

/// Git commit recorded when the static release manifest was exported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(super) struct SourceCommit(Box<str>);

impl TryFrom<String> for SourceCommit {
    type Error = ManifestError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 40
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value.into()))
        } else {
            Err(ManifestError::InvalidSourceCommit(value))
        }
    }
}

impl From<SourceCommit> for String {
    fn from(value: SourceCommit) -> Self {
        value.0.into()
    }
}

/// Record profile controlling ecosystem tensor and tuning compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u64", into = "u64")]
pub(super) struct CheckpointProfile(u64);

impl TryFrom<u64> for CheckpointProfile {
    type Error = ManifestError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if matches!(value, 15 | 16 | 17 | 18 | 28) {
            Ok(Self(value))
        } else {
            Err(ManifestError::UnsupportedCheckpointProfile(value))
        }
    }
}

impl From<CheckpointProfile> for u64 {
    fn from(value: CheckpointProfile) -> Self {
        value.0
    }
}

/// Exact recurrent actor and centralized critic dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Architecture {
    /// Per-agent observation width.
    pub(super) local_observation_size: u16,
    /// Centralized critic state width.
    pub(super) global_state_size: u16,
    /// Maximum padded agent count.
    pub(super) max_agents: u8,
    /// Continuous actor output width.
    pub(super) action_size: u8,
    /// LSTM hidden-state width.
    pub(super) actor_hidden_size: u16,
    /// Ordered critic MLP hidden widths.
    pub(super) critic_hidden_sizes: [u16; 2],
}

impl Architecture {
    /// Current loadable ecosystem architecture.
    pub(super) const CURRENT: Self = Self {
        local_observation_size: 267,
        global_state_size: 363,
        max_agents: 12,
        action_size: 4,
        actor_hidden_size: 64,
        critic_hidden_sizes: [128, 64],
    };
}

/// Closed forage reset distribution stored in checkpoint sidecars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum ForageDifficulty {
    /// Short targets near the body centerline.
    Foundation,
    /// Held-out left, right, near, and far target distribution.
    Expanded,
}

/// Observable ecosystem tuning paired with a policy record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ExperimentTuning {
    /// Forage reset distribution.
    pub(super) forage_difficulty: ForageDifficulty,
    /// Active center-origin perception sectors.
    pub(super) perception_ray_count: u8,
    /// Maximum hit points.
    pub(super) maximum_hit_points: u8,
    /// Initial hit points.
    pub(super) initial_hit_points: u8,
    /// Maximum satiation points.
    pub(super) maximum_satiation: u8,
    /// Initial satiation points.
    pub(super) initial_satiation: u8,
    /// Maximum hydration points.
    pub(super) maximum_hydration: u8,
    /// Initial hydration points.
    pub(super) initial_hydration: u8,
    /// Seconds between need losses.
    pub(super) need_loss_interval_seconds: u16,
    /// Seconds between starvation damage.
    pub(super) starvation_damage_interval_seconds: u16,
    /// Seconds between dehydration damage.
    pub(super) dehydration_damage_interval_seconds: u16,
    /// Extra moving need-clock percentage.
    pub(super) movement_need_cost_percent: u8,
    /// Movement speed multiplier.
    pub(super) movement_speed_multiplier: f32,
    /// Gaze yaw limit in degrees.
    pub(super) gaze_yaw_limit_degrees: f32,
    /// Episode horizon in seconds.
    pub(super) episode_seconds: u16,
}

/// Typed configuration sidecar supplied with a checkpoint record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckpointSidecar {
    /// Mechanics and tensor compatibility profile.
    pub(super) checkpoint_profile: CheckpointProfile,
    /// Curriculum stage recorded during training.
    pub(super) stage: Stage,
    /// Observable simulation tuning recorded with the policy.
    pub(super) experiment_tuning: ExperimentTuning,
}

impl CheckpointSidecar {
    /// Parse a local sidecar through the same typed schema used by the exporter.
    pub(super) fn from_json(json: &str) -> Result<Self, ManifestError> {
        serde_json::from_str(json).map_err(ManifestError::SidecarJson)
    }

    /// Require the selected route's exact profile and experiment tuning.
    pub(super) fn validate_for(&self, entry: &StageManifest) -> Result<(), ManifestError> {
        if self.stage == entry.stage
            && self.checkpoint_profile == entry.checkpoint_profile
            && self.experiment_tuning == entry.experiment_tuning
        {
            Ok(())
        } else {
            Err(ManifestError::SidecarMismatch(entry.stage))
        }
    }
}

/// Summary values that justified selecting one compatible checkpoint set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SelectionSummary {
    /// Native environment steps completed by the source run.
    pub(super) global_steps: u64,
    /// Initial bunny validation return.
    pub(super) initial_bunny_mean_return: f32,
    /// Selected bunny validation return.
    pub(super) best_bunny_mean_return: f32,
    /// Initial bunny validation lifetime in seconds.
    pub(super) initial_bunny_mean_lifetime: f32,
    /// Selected bunny validation lifetime in seconds.
    pub(super) best_bunny_mean_lifetime: f32,
    /// Selected bunny lower 95% lifetime bound.
    pub(super) best_bunny_ci95_lower: f32,
    /// Selected bunny upper 95% lifetime bound.
    pub(super) best_bunny_ci95_upper: f32,
    /// Validation fraction where bunnies ate.
    pub(super) best_bunny_food_fraction: f32,
    /// Validation fraction where bunnies ate and drank.
    pub(super) best_bunny_food_and_water_fraction: f32,
    /// Initial fox validation return.
    pub(super) initial_fox_mean_return: f32,
    /// Selected fox validation return.
    pub(super) best_fox_mean_return: f32,
    /// Initial fox validation lifetime in seconds.
    pub(super) initial_fox_mean_lifetime: f32,
    /// Selected fox validation lifetime in seconds.
    pub(super) best_fox_mean_lifetime: f32,
    /// Selected fox lower 95% lifetime bound.
    pub(super) best_fox_ci95_lower: f32,
    /// Selected fox upper 95% lifetime bound.
    pub(super) best_fox_ci95_upper: f32,
    /// Validation fraction where foxes hunted and drank.
    pub(super) best_fox_prey_and_water_fraction: f32,
    /// Source summary's explicit evidence boundary.
    pub(super) qualification_evidence: Box<str>,
}

impl ExperimentTuning {
    /// Reject tuning that cannot construct the current ecosystem profile.
    fn validate(&self) -> Result<(), ManifestError> {
        let discrete_valid = (2..=24).contains(&self.perception_ray_count)
            && self.perception_ray_count.is_multiple_of(2)
            && self.maximum_hit_points > 0
            && (1..=self.maximum_hit_points).contains(&self.initial_hit_points)
            && self.maximum_satiation > 0
            && self.initial_satiation <= self.maximum_satiation
            && self.maximum_hydration > 0
            && self.initial_hydration <= self.maximum_hydration
            && self.need_loss_interval_seconds > 0
            && self.starvation_damage_interval_seconds > 0
            && self.dehydration_damage_interval_seconds > 0
            && self.movement_need_cost_percent <= 100
            && self.episode_seconds > 0;
        let continuous_valid = self.movement_speed_multiplier.is_finite()
            && self.movement_speed_multiplier > 0.0
            && self.gaze_yaw_limit_degrees.is_finite()
            && (0.0..=180.0).contains(&self.gaze_yaw_limit_degrees);
        if discrete_valid && continuous_valid {
            Ok(())
        } else {
            Err(ManifestError::InvalidExperimentTuning)
        }
    }
}

impl SelectionSummary {
    /// Reject missing, non-finite, inverted, or out-of-range release evidence.
    fn validate(&self, stage: Stage) -> Result<(), ManifestError> {
        let finite_values = [
            self.initial_bunny_mean_return,
            self.best_bunny_mean_return,
            self.initial_bunny_mean_lifetime,
            self.best_bunny_mean_lifetime,
            self.best_bunny_ci95_lower,
            self.best_bunny_ci95_upper,
            self.best_bunny_food_fraction,
            self.best_bunny_food_and_water_fraction,
            self.initial_fox_mean_return,
            self.best_fox_mean_return,
            self.initial_fox_mean_lifetime,
            self.best_fox_mean_lifetime,
            self.best_fox_ci95_lower,
            self.best_fox_ci95_upper,
            self.best_fox_prey_and_water_fraction,
        ];
        let proportions = [
            self.best_bunny_food_fraction,
            self.best_bunny_food_and_water_fraction,
            self.best_fox_prey_and_water_fraction,
        ];
        let bunny_interval_valid = self.best_bunny_ci95_lower <= self.best_bunny_ci95_upper;
        let fox_interval_valid =
            !stage.requires_fox() || self.best_fox_ci95_lower <= self.best_fox_ci95_upper;
        if self.global_steps > 0
            && finite_values.into_iter().all(f32::is_finite)
            && proportions
                .into_iter()
                .all(|value| (0.0..=1.0).contains(&value))
            && bunny_interval_valid
            && fox_interval_valid
            && !self.qualification_evidence.trim().is_empty()
        {
            Ok(())
        } else {
            Err(ManifestError::InvalidSelectionSummary(stage))
        }
    }
}

/// Validated relative path under the static checkpoint directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(super) struct AssetPath(Box<str>);

impl AsRef<str> for AssetPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for AssetPath {
    type Error = ManifestError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.starts_with("checkpoints/")
            && Path::new(&value).extension() == Some(OsStr::new("mpk"))
            && !value.contains("..")
            && !value.contains('\\')
        {
            Ok(Self(value.into()))
        } else {
            Err(ManifestError::InvalidAssetPath(value))
        }
    }
}

impl From<AssetPath> for String {
    fn from(value: AssetPath) -> Self {
        value.0.into()
    }
}

/// Validated relative path to a checkpoint configuration sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(super) struct ConfigPath(Box<str>);

impl AsRef<str> for ConfigPath {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for ConfigPath {
    type Error = ManifestError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.starts_with("checkpoints/")
            && value.ends_with(".config.json")
            && !value.contains("..")
            && !value.contains('\\')
        {
            Ok(Self(value.into()))
        } else {
            Err(ManifestError::InvalidConfigPath(value))
        }
    }
}

impl From<ConfigPath> for String {
    fn from(value: ConfigPath) -> Self {
        value.0.into()
    }
}

/// Lowercase SHA-256 digest used to verify fetched checkpoint bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub(super) struct Sha256Digest(Box<str>);

impl AsRef<str> for Sha256Digest {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = ManifestError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        if value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            Ok(Self(value.into()))
        } else {
            Err(ManifestError::InvalidDigest(value))
        }
    }
}

impl From<Sha256Digest> for String {
    fn from(value: Sha256Digest) -> Self {
        value.0.into()
    }
}

/// Nonzero checkpoint byte count within the browser upload limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub(super) struct CheckpointSize(u32);

impl CheckpointSize {
    /// Return the validated count.
    pub(super) const fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for CheckpointSize {
    type Error = ManifestError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if (1..=MAX_CHECKPOINT_BYTES).contains(&value) {
            Ok(Self(value))
        } else {
            Err(ManifestError::InvalidSize(value))
        }
    }
}

impl From<CheckpointSize> for u32 {
    fn from(value: CheckpointSize) -> Self {
        value.0
    }
}

/// One immutable checkpoint asset and its provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct CheckpointAsset {
    /// Content-addressed URL relative to the site root.
    pub(super) path: AssetPath,
    /// Expected SHA-256 of the response body.
    pub(super) sha256: Sha256Digest,
    /// Expected response size.
    pub(super) bytes: CheckpointSize,
    /// Versioned configuration sidecar URL.
    pub(super) config_path: ConfigPath,
    /// Expected SHA-256 of the configuration sidecar.
    pub(super) config_sha256: Sha256Digest,
    /// Expected configuration sidecar size.
    pub(super) config_bytes: CheckpointSize,
    /// Repository-relative source run retained for audit.
    pub(super) source_run: Box<str>,
}

/// Curated policy set for one route.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct StageManifest {
    /// Closed route and simulation stage.
    pub(super) stage: Stage,
    /// Human title retained in the wire contract.
    pub(super) title: Box<str>,
    /// Mechanics and tensor compatibility profile.
    pub(super) checkpoint_profile: CheckpointProfile,
    /// Exact experiment tuning used to produce the record.
    pub(super) experiment_tuning: ExperimentTuning,
    /// Training algorithm encoded by the record.
    pub(super) algorithm: Algorithm,
    /// Actor and critic dimensions used to decode the record.
    pub(super) architecture: Architecture,
    /// Exact ordered policy-role requirement.
    pub(super) required_policy_roles: Vec<PolicyRole>,
    /// Root training seed from the source run.
    pub(super) training_seed: u64,
    /// Evidence values used to select this policy set.
    pub(super) selection_summary: SelectionSummary,
    /// Honest evidence classification for this policy set.
    pub(super) qualification: Qualification,
    /// Default bunny checkpoint.
    pub(super) bunny: CheckpointAsset,
    /// Default fox checkpoint for predator stages.
    pub(super) fox: Option<CheckpointAsset>,
}

/// Complete static checkpoint manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct CheckpointManifest {
    /// Closed schema version.
    pub(super) manifest_version: ManifestVersion,
    /// Source tree commit recorded during export.
    pub(super) source_commit: SourceCommit,
    /// Burn version controlling record compatibility.
    pub(super) burn_version: BurnVersion,
    /// Exactly six unique stages in curriculum order.
    pub(super) stages: Vec<StageManifest>,
}

impl CheckpointManifest {
    /// Parse and validate the complete manifest boundary.
    pub(super) fn from_json(json: &str) -> Result<Self, ManifestError> {
        let manifest: Self = serde_json::from_str(json).map_err(ManifestError::Json)?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Return one stage entry.
    pub(super) fn stage(&self, stage: Stage) -> Result<&StageManifest, ManifestError> {
        self.stages
            .iter()
            .find(|entry| entry.stage == stage)
            .ok_or(ManifestError::MissingStage(stage))
    }

    /// Enforce ordering, uniqueness, role cardinality, and provenance.
    fn validate(&self) -> Result<(), ManifestError> {
        if self.stages.len() != Stage::ALL.len() {
            return Err(ManifestError::StageCount(self.stages.len()));
        }
        for (expected, entry) in Stage::ALL.into_iter().zip(&self.stages) {
            if entry.stage != expected {
                return Err(ManifestError::StageOrder {
                    expected,
                    actual: entry.stage,
                });
            }
            if entry.title.as_ref() != entry.stage.title() {
                return Err(ManifestError::TitleMismatch(entry.stage));
            }
            if entry.stage.requires_fox() != entry.fox.is_some() {
                return Err(ManifestError::FoxCardinality(entry.stage));
            }
            let expected_roles = if entry.stage.requires_fox() {
                &[PolicyRole::Bunny, PolicyRole::Fox][..]
            } else {
                &[PolicyRole::Bunny][..]
            };
            if entry.required_policy_roles != expected_roles {
                return Err(ManifestError::PolicyRoles(entry.stage));
            }
            if entry.algorithm != Algorithm::RecurrentPpo
                || entry.architecture != Architecture::CURRENT
            {
                return Err(ManifestError::Architecture(entry.stage));
            }
            if entry.training_seed == 0 {
                return Err(ManifestError::InvalidTrainingSeed(entry.stage));
            }
            entry.experiment_tuning.validate()?;
            entry.selection_summary.validate(entry.stage)?;
            validate_asset(entry.stage, PolicyRole::Bunny, &entry.bunny)?;
            if let Some(fox) = entry.fox.as_ref() {
                validate_asset(entry.stage, PolicyRole::Fox, fox)?;
                if fox.source_run != entry.bunny.source_run {
                    return Err(ManifestError::UnsynchronizedPolicies(entry.stage));
                }
            }
        }
        Ok(())
    }
}

/// Enforce content addressing, stage-role naming, and source provenance.
fn validate_asset(
    stage: Stage,
    role: PolicyRole,
    asset: &CheckpointAsset,
) -> Result<(), ManifestError> {
    let digest_prefix = asset.sha256.as_ref().chars().take(12).collect::<String>();
    let role_key = match role {
        PolicyRole::Bunny => "bunny",
        PolicyRole::Fox => "fox",
    };
    let stem = format!("checkpoints/{}-{role_key}-{digest_prefix}", stage.as_key());
    let record_path = format!("{stem}.mpk");
    let config_path = format!("{stem}.config.json");
    if asset.path.as_ref() != record_path || asset.config_path.as_ref() != config_path {
        return Err(ManifestError::ContentAddress(stage, role));
    }
    if asset.source_run.trim().is_empty() {
        return Err(ManifestError::MissingProvenance(stage));
    }
    Ok(())
}

/// Manifest parse or invariant failure.
#[derive(Debug)]
pub(super) enum ManifestError {
    /// JSON could not be decoded into the typed schema.
    Json(serde_json::Error),
    /// A local checkpoint sidecar could not be decoded.
    SidecarJson(serde_json::Error),
    /// A route key is outside the closed stage vocabulary.
    UnknownStage(Box<str>),
    /// An evidence label is outside the closed release vocabulary.
    UnknownQualification(Box<str>),
    /// An asset path can escape or does not name an MPK checkpoint.
    InvalidAssetPath(String),
    /// A sidecar path can escape or does not name a configuration JSON file.
    InvalidConfigPath(String),
    /// A digest is not 64 lowercase hexadecimal characters.
    InvalidDigest(String),
    /// A release source commit is not a full lowercase Git object ID.
    InvalidSourceCommit(String),
    /// A checkpoint profile cannot load through the current ecosystem architecture.
    UnsupportedCheckpointProfile(u64),
    /// A checkpoint is empty or exceeds the upload limit.
    InvalidSize(u32),
    /// The manifest does not contain six stages.
    StageCount(usize),
    /// Stage entries are not in curriculum order.
    StageOrder {
        /// Stage required at this manifest position.
        expected: Stage,
        /// Stage found at this manifest position.
        actual: Stage,
    },
    /// A fox checkpoint is missing or present for the wrong stage.
    FoxCardinality(Stage),
    /// A stage title differs from its closed identifier.
    TitleMismatch(Stage),
    /// Required policy roles differ from the stage contract.
    PolicyRoles(Stage),
    /// Algorithm or architecture differs from the current decoder contract.
    Architecture(Stage),
    /// Root training seed is zero.
    InvalidTrainingSeed(Stage),
    /// Experiment tuning cannot construct the current simulation profile.
    InvalidExperimentTuning,
    /// Selection summary is absent, non-finite, or internally inconsistent.
    InvalidSelectionSummary(Stage),
    /// Static asset names do not contain the declared digest prefix.
    ContentAddress(Stage, PolicyRole),
    /// Predator policies do not come from one synchronized source run.
    UnsynchronizedPolicies(Stage),
    /// A local sidecar differs from the selected route profile or tuning.
    SidecarMismatch(Stage),
    /// A curated checkpoint lacks its repository source path.
    MissingProvenance(Stage),
    /// A requested supported stage is absent.
    MissingStage(Stage),
}

impl fmt::Display for ManifestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "checkpoint manifest JSON is invalid: {error}"),
            Self::SidecarJson(error) => {
                write!(formatter, "checkpoint sidecar JSON is invalid: {error}")
            }
            Self::UnknownStage(stage) => write!(formatter, "unknown ecosystem stage {stage:?}"),
            Self::UnknownQualification(value) => {
                write!(formatter, "unknown checkpoint qualification {value:?}")
            }
            Self::InvalidAssetPath(path) => write!(formatter, "invalid checkpoint path {path:?}"),
            Self::InvalidConfigPath(path) => {
                write!(formatter, "invalid checkpoint config path {path:?}")
            }
            Self::InvalidDigest(digest) => write!(formatter, "invalid SHA-256 digest {digest:?}"),
            Self::InvalidSourceCommit(commit) => {
                write!(formatter, "invalid source commit {commit:?}")
            }
            Self::UnsupportedCheckpointProfile(profile) => {
                write!(formatter, "unsupported checkpoint profile {profile}")
            }
            Self::InvalidSize(bytes) => write!(formatter, "invalid checkpoint size {bytes}"),
            Self::StageCount(count) => {
                write!(formatter, "manifest contains {count} stages; expected 6")
            }
            Self::StageOrder { expected, actual } => write!(
                formatter,
                "manifest stage order expected {} but found {}",
                expected.as_key(),
                actual.as_key()
            ),
            Self::FoxCardinality(stage) => {
                write!(
                    formatter,
                    "fox checkpoint cardinality is invalid for {}",
                    stage.as_key()
                )
            }
            Self::TitleMismatch(stage) => {
                write!(formatter, "manifest title differs for {}", stage.as_key())
            }
            Self::PolicyRoles(stage) => {
                write!(formatter, "policy roles differ for {}", stage.as_key())
            }
            Self::Architecture(stage) => {
                write!(
                    formatter,
                    "policy architecture differs for {}",
                    stage.as_key()
                )
            }
            Self::InvalidTrainingSeed(stage) => {
                write!(formatter, "training seed is invalid for {}", stage.as_key())
            }
            Self::InvalidExperimentTuning => formatter.write_str("experiment tuning is invalid"),
            Self::InvalidSelectionSummary(stage) => {
                write!(
                    formatter,
                    "selection summary is invalid for {}",
                    stage.as_key()
                )
            }
            Self::ContentAddress(stage, role) => write!(
                formatter,
                "content-addressed {:?} asset names are invalid for {}",
                role,
                stage.as_key()
            ),
            Self::UnsynchronizedPolicies(stage) => write!(
                formatter,
                "bunny and fox policies are not synchronized for {}",
                stage.as_key()
            ),
            Self::SidecarMismatch(stage) => write!(
                formatter,
                "checkpoint sidecar differs from the {} route profile",
                stage.as_key()
            ),
            Self::MissingProvenance(stage) => {
                write!(
                    formatter,
                    "checkpoint provenance is missing for {}",
                    stage.as_key()
                )
            }
            Self::MissingStage(stage) => {
                write!(formatter, "manifest is missing {}", stage.as_key())
            }
        }
    }
}

impl Error for ManifestError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) | Self::SidecarJson(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build one valid stage row for manifest boundary tests.
    fn entry(stage: Stage) -> StageManifest {
        let digest = "a".repeat(64);
        let prefix = digest.chars().take(12).collect::<String>();
        let asset = CheckpointAsset {
            path: AssetPath::try_from(format!("checkpoints/{}-bunny-{prefix}.mpk", stage.as_key()))
                .expect("test path is valid"),
            sha256: Sha256Digest::try_from(digest.clone()).expect("test digest is valid"),
            bytes: CheckpointSize::try_from(512).expect("test size is valid"),
            config_path: ConfigPath::try_from(format!(
                "checkpoints/{}-bunny-{prefix}.config.json",
                stage.as_key()
            ))
            .expect("test config path is valid"),
            config_sha256: Sha256Digest::try_from(digest).expect("test digest is valid"),
            config_bytes: CheckpointSize::try_from(256).expect("test size is valid"),
            source_run: format!("runs/{}/best.mpk", stage.as_key()).into(),
        };
        let fox = stage.requires_fox().then(|| CheckpointAsset {
            path: AssetPath::try_from(format!("checkpoints/{}-fox-{prefix}.mpk", stage.as_key()))
                .expect("test path is valid"),
            config_path: ConfigPath::try_from(format!(
                "checkpoints/{}-fox-{prefix}.config.json",
                stage.as_key()
            ))
            .expect("test config path is valid"),
            ..asset.clone()
        });
        StageManifest {
            stage,
            title: stage.title().into(),
            checkpoint_profile: CheckpointProfile::try_from(18).expect("profile is compatible"),
            experiment_tuning: tuning(),
            algorithm: Algorithm::RecurrentPpo,
            architecture: Architecture::CURRENT,
            required_policy_roles: if stage.requires_fox() {
                vec![PolicyRole::Bunny, PolicyRole::Fox]
            } else {
                vec![PolicyRole::Bunny]
            },
            training_seed: 157,
            selection_summary: summary(),
            qualification: Qualification::BestCompatibleAvailable,
            bunny: asset,
            fox,
        }
    }

    /// Build valid profile tuning for manifest boundary tests.
    fn tuning() -> ExperimentTuning {
        ExperimentTuning {
            forage_difficulty: ForageDifficulty::Expanded,
            perception_ray_count: 24,
            maximum_hit_points: 5,
            initial_hit_points: 5,
            maximum_satiation: 5,
            initial_satiation: 2,
            maximum_hydration: 5,
            initial_hydration: 2,
            need_loss_interval_seconds: 4,
            starvation_damage_interval_seconds: 2,
            dehydration_damage_interval_seconds: 2,
            movement_need_cost_percent: 25,
            movement_speed_multiplier: 1.0,
            gaze_yaw_limit_degrees: 30.0,
            episode_seconds: 20,
        }
    }

    /// Build finite selection evidence for manifest boundary tests.
    fn summary() -> SelectionSummary {
        SelectionSummary {
            global_steps: 100,
            initial_bunny_mean_return: 0.1,
            best_bunny_mean_return: 0.2,
            initial_bunny_mean_lifetime: 10.0,
            best_bunny_mean_lifetime: 12.0,
            best_bunny_ci95_lower: 11.0,
            best_bunny_ci95_upper: 13.0,
            best_bunny_food_fraction: 0.8,
            best_bunny_food_and_water_fraction: 0.6,
            initial_fox_mean_return: 0.1,
            best_fox_mean_return: 0.2,
            initial_fox_mean_lifetime: 10.0,
            best_fox_mean_lifetime: 12.0,
            best_fox_ci95_lower: 11.0,
            best_fox_ci95_upper: 13.0,
            best_fox_prey_and_water_fraction: 0.5,
            qualification_evidence: "held-out validation".into(),
        }
    }

    #[test]
    fn complete_ordered_manifest_is_accepted() {
        let manifest = CheckpointManifest {
            manifest_version: ManifestVersion::V1,
            source_commit: SourceCommit::try_from("a".repeat(40)).expect("commit is valid"),
            burn_version: BurnVersion::V0_21_0,
            stages: Stage::ALL.into_iter().map(entry).collect(),
        };
        let json = serde_json::to_string(&manifest).expect("manifest serializes");
        let parsed = CheckpointManifest::from_json(&json).expect("manifest parses");

        let obstacles = parsed.stage(Stage::Obstacles).expect("stage exists");
        assert_eq!(obstacles.stage, Stage::Obstacles);
        assert_eq!(obstacles.stage.title(), "Obstacles");
        assert_eq!(obstacles.stage.position(), 6);
        assert_eq!(obstacles.bunny.bytes.get(), 512);
    }

    #[test]
    fn missing_fox_checkpoint_is_rejected() {
        let mut stages = Stage::ALL.into_iter().map(entry).collect::<Vec<_>>();
        stages
            .iter_mut()
            .find(|entry| entry.stage == Stage::PredatorPrey)
            .expect("predator-prey stage exists")
            .fox = None;
        let manifest = CheckpointManifest {
            manifest_version: ManifestVersion::V1,
            source_commit: SourceCommit::try_from("a".repeat(40)).expect("commit is valid"),
            burn_version: BurnVersion::V0_21_0,
            stages,
        };
        let json = serde_json::to_string(&manifest).expect("manifest serializes");

        assert!(matches!(
            CheckpointManifest::from_json(&json),
            Err(ManifestError::FoxCardinality(Stage::PredatorPrey))
        ));
    }

    #[test]
    fn traversal_asset_path_is_rejected() {
        assert!(matches!(
            AssetPath::try_from("checkpoints/../secret.mpk".to_owned()),
            Err(ManifestError::InvalidAssetPath(_))
        ));
    }

    #[test]
    fn release_qualification_uses_the_plan_vocabulary() {
        let qualification: Qualification = serde_json::from_str(r#""best-compatible-available""#)
            .expect("the release fallback qualification must parse");

        assert_eq!(qualification, Qualification::BestCompatibleAvailable);
    }

    #[test]
    fn bundled_release_manifest_carries_the_complete_contract() {
        let manifest = CheckpointManifest::from_json(include_str!("../checkpoints/manifest.json"))
            .expect("bundled release manifest must validate");

        assert_eq!(manifest.burn_version, BurnVersion::V0_21_0);
        assert_eq!(manifest.stages.len(), Stage::ALL.len());
        for entry in &manifest.stages {
            assert_eq!(entry.title.as_ref(), entry.stage.title());
            assert_eq!(entry.algorithm, Algorithm::RecurrentPpo);
            assert_eq!(entry.architecture, Architecture::CURRENT);
            assert!(entry.training_seed > 0);
            assert!(entry.selection_summary.global_steps > 0);
            assert_eq!(entry.qualification, Qualification::BestCompatibleAvailable);
        }
    }

    #[test]
    fn duplicate_stage_and_forbidden_fox_are_rejected() {
        let mut duplicate = Stage::ALL.into_iter().map(entry).collect::<Vec<_>>();
        duplicate
            .iter_mut()
            .find(|entry| entry.stage == Stage::Survival)
            .expect("survival stage exists")
            .stage = Stage::Forage;
        let manifest = CheckpointManifest {
            manifest_version: ManifestVersion::V1,
            source_commit: SourceCommit::try_from("a".repeat(40)).expect("commit is valid"),
            burn_version: BurnVersion::V0_21_0,
            stages: duplicate,
        };
        assert!(matches!(
            manifest.validate(),
            Err(ManifestError::StageOrder { .. })
        ));

        let mut forbidden = Stage::ALL.into_iter().map(entry).collect::<Vec<_>>();
        let forage = forbidden
            .iter_mut()
            .find(|entry| entry.stage == Stage::Forage)
            .expect("forage stage exists");
        forage.fox = Some(forage.bunny.clone());
        let manifest = CheckpointManifest {
            manifest_version: ManifestVersion::V1,
            source_commit: SourceCommit::try_from("a".repeat(40)).expect("commit is valid"),
            burn_version: BurnVersion::V0_21_0,
            stages: forbidden,
        };
        assert!(matches!(
            manifest.validate(),
            Err(ManifestError::FoxCardinality(Stage::Forage))
        ));
    }

    #[test]
    fn sidecar_must_match_stage_profile_and_tuning() {
        let survival = entry(Stage::Survival);
        let mut sidecar = CheckpointSidecar {
            checkpoint_profile: survival.checkpoint_profile,
            stage: survival.stage,
            experiment_tuning: survival.experiment_tuning.clone(),
        };
        sidecar
            .validate_for(&survival)
            .expect("matching sidecar must validate");
        sidecar.stage = Stage::Shelter;
        assert!(matches!(
            sidecar.validate_for(&survival),
            Err(ManifestError::SidecarMismatch(Stage::Survival))
        ));
        assert!(matches!(
            CheckpointSidecar::from_json("{}"),
            Err(ManifestError::SidecarJson(_))
        ));
    }
}
