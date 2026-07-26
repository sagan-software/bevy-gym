//! Deterministic tabular Q-learning for finite discrete environments.
//!
//! The trainer owns no environment semantics. Callers provide finite state and
//! action indices, optional fixed per-state action masks, and transitions with
//! separate termination and truncation flags.

use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// On-disk format version written by [`TabularQTrainer::save`].
pub const TABULAR_Q_CHECKPOINT_VERSION: u32 = 1;

/// Fixed checkpoint header identifying bevy-gym tabular Q data.
const CHECKPOINT_MAGIC: &[u8; 8] = b"BGYMTQ\0\0";

/// FNV-1a offset basis used for stable table fingerprints and file checksums.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a prime used for stable table fingerprints and file checksums.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hyperparameters for finite-state tabular Q-learning.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularQConfig {
    /// Q-value interpolation factor in `(0, 1]`.
    pub learning_rate: f64,

    /// Discount factor in `0..=1`.
    pub gamma: f64,

    /// Exploration probability before any action decisions.
    pub epsilon_start: f64,

    /// Exploration probability after the linear decay interval.
    pub epsilon_end: f64,

    /// Action decisions over which epsilon decays linearly.
    pub epsilon_decay_steps: u64,

    /// Seed for the trainer-owned deterministic action RNG.
    pub seed: u64,
}

impl Default for TabularQConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay_steps: 10_000,
            seed: 0,
        }
    }
}

impl TabularQConfig {
    /// Validate dimensions and all numeric hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`TabularQError::InvalidConfig`] for zero or overflowing table
    /// dimensions, non-finite values, invalid probability ranges, or a zero
    /// epsilon decay interval.
    pub fn validate(&self, state_count: usize, action_count: usize) -> Result<(), TabularQError> {
        if state_count == 0 {
            return Err(TabularQError::invalid_config(
                "state_count",
                "must be greater than zero",
            ));
        }
        if action_count == 0 {
            return Err(TabularQError::invalid_config(
                "action_count",
                "must be greater than zero",
            ));
        }
        if state_count.checked_mul(action_count).is_none() {
            return Err(TabularQError::invalid_config(
                "table_dimensions",
                "state_count * action_count must fit usize",
            ));
        }
        if u64::try_from(action_count).is_err() {
            return Err(TabularQError::invalid_config(
                "action_count",
                "must fit u64 for deterministic sampling",
            ));
        }
        if !self.learning_rate.is_finite() || !(0.0..=1.0).contains(&self.learning_rate) {
            return Err(TabularQError::invalid_config(
                "learning_rate",
                "must be finite and in (0, 1]",
            ));
        }
        if self.learning_rate == 0.0 {
            return Err(TabularQError::invalid_config(
                "learning_rate",
                "must be greater than zero",
            ));
        }
        if !self.gamma.is_finite() || !(0.0..=1.0).contains(&self.gamma) {
            return Err(TabularQError::invalid_config(
                "gamma",
                "must be finite and in 0..=1",
            ));
        }
        if !self.epsilon_start.is_finite()
            || !self.epsilon_end.is_finite()
            || !(0.0..=1.0).contains(&self.epsilon_start)
            || !(0.0..=1.0).contains(&self.epsilon_end)
            || self.epsilon_start < self.epsilon_end
        {
            return Err(TabularQError::invalid_config(
                "epsilon",
                "start and end must be finite in 0..=1 with start >= end",
            ));
        }
        if self.epsilon_decay_steps == 0 {
            return Err(TabularQError::invalid_config(
                "epsilon_decay_steps",
                "must be greater than zero",
            ));
        }

        Ok(())
    }

    /// Return the linearly decayed epsilon for an action-decision count.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "u64 schedule counters intentionally map to a bounded f64 interpolation ratio"
    )]
    pub fn epsilon_at(&self, action_decisions: u64) -> f64 {
        let elapsed = action_decisions.min(self.epsilon_decay_steps);
        let progress = elapsed as f64 / self.epsilon_decay_steps as f64;
        (self.epsilon_end - self.epsilon_start).mul_add(progress, self.epsilon_start)
    }
}

/// One finite-state transition consumed by [`TabularQTrainer::update`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabularTransition {
    /// State before the action.
    pub state: usize,

    /// Discrete action selected in `state`.
    pub action: usize,

    /// Scalar reward observed after the action.
    pub reward: f64,

    /// State observed after the action.
    pub next_state: usize,

    /// Whether the environment ended naturally.
    pub terminated: bool,

    /// Whether an external time or step limit ended the episode.
    pub truncated: bool,
}

/// Evidence returned for one successful Q-value update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabularQUpdate {
    /// One-based count of successful updates including this update.
    pub update_index: u64,

    /// Q-value before interpolation.
    pub previous_value: f64,

    /// Bellman target used for interpolation.
    pub target: f64,

    /// Q-value after interpolation.
    pub new_value: f64,

    /// Whether the target included a next-state value.
    pub bootstrapped: bool,

    /// Whether the input transition represented truncation.
    pub transition_was_truncated: bool,
}

/// Compact training evidence suitable for metrics and proof receipts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabularQEvidence {
    /// Successful Bellman updates applied to the table.
    pub update_count: u64,

    /// Epsilon-greedy action decisions made by the trainer.
    pub action_decision_count: u64,

    /// Stable non-cryptographic fingerprint of table dimensions and Q-values.
    pub table_fingerprint: String,
}

/// Immutable deterministic greedy policy derived from a Q-table.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularQPolicy {
    /// Number of encoded states.
    state_count: usize,

    /// Number of encoded actions.
    action_count: usize,

    /// Row-major Q-values.
    q_values: Vec<f64>,

    /// Optional row-major legality mask.
    action_masks: Option<Vec<bool>>,
}

impl TabularQPolicy {
    /// Select the highest-valued legal action, breaking ties by lowest index.
    ///
    /// # Errors
    ///
    /// Returns [`TabularQError::StateOutOfRange`] for an invalid state or
    /// [`TabularQError::NoAllowedActions`] for an invalid policy mask.
    pub fn greedy_action(&self, state: usize) -> Result<usize, TabularQError> {
        greedy_action(
            state,
            self.state_count,
            self.action_count,
            &self.q_values,
            self.action_masks.as_deref(),
        )
    }

    /// Return one Q-value.
    ///
    /// # Errors
    ///
    /// Returns an index error when `state` or `action` is outside the policy
    /// dimensions.
    pub fn q_value(&self, state: usize, action: usize) -> Result<f64, TabularQError> {
        let index = checked_table_index(state, action, self.state_count, self.action_count)?;
        self.q_values.get(index).copied().ok_or_else(|| {
            TabularQError::invalid_internal_table(self.state_count, self.action_count)
        })
    }

    /// Number of encoded states.
    #[must_use]
    pub const fn state_count(&self) -> usize {
        self.state_count
    }

    /// Number of encoded actions.
    #[must_use]
    pub const fn action_count(&self) -> usize {
        self.action_count
    }

    /// Borrow all row-major Q-values.
    #[must_use]
    pub fn q_values(&self) -> &[f64] {
        &self.q_values
    }

    /// Return the same stable table fingerprint used by trainer evidence.
    #[must_use]
    pub fn table_fingerprint(&self) -> String {
        table_fingerprint(self.state_count, self.action_count, &self.q_values)
    }
}

/// First-party tabular Q-learning trainer for finite discrete spaces.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularQTrainer {
    /// Validated hyperparameters.
    config: TabularQConfig,

    /// Number of encoded states.
    state_count: usize,

    /// Number of encoded actions.
    action_count: usize,

    /// Row-major Q-values.
    q_values: Vec<f64>,

    /// Optional row-major legality mask.
    action_masks: Option<Vec<bool>>,

    /// Deterministic exploration RNG.
    rng: SplitMix64,

    /// Successful Bellman update count.
    update_count: u64,

    /// Epsilon-greedy action decision count.
    action_decision_count: u64,
}

impl TabularQTrainer {
    /// Create a zero-initialized Q-table with optional fixed per-state masks.
    ///
    /// Each mask row must contain exactly `action_count` entries and at least
    /// one allowed action. `None` means every action is allowed in every state.
    ///
    /// # Errors
    ///
    /// Returns a configuration or mask error when dimensions, hyperparameters,
    /// or masks cannot define a valid finite trainer.
    pub fn new(
        state_count: usize,
        action_count: usize,
        config: TabularQConfig,
        action_masks: Option<Vec<Vec<bool>>>,
    ) -> Result<Self, TabularQError> {
        config.validate(state_count, action_count)?;
        let table_len = state_count
            .checked_mul(action_count)
            .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))?;
        let action_masks = flatten_action_masks(state_count, action_count, action_masks)?;
        let rng = SplitMix64::new(config.seed);

        Ok(Self {
            config,
            state_count,
            action_count,
            q_values: vec![0.0; table_len],
            action_masks,
            rng,
            update_count: 0,
            action_decision_count: 0,
        })
    }

    /// Select an epsilon-greedy legal action and advance the decision schedule.
    ///
    /// Greedy ties resolve to the lowest legal action index. Exploration is
    /// sampled only from legal actions.
    ///
    /// # Errors
    ///
    /// Returns a state, mask, or counter error without incrementing the
    /// decision count.
    pub fn select_action(&mut self, state: usize) -> Result<usize, TabularQError> {
        validate_state(state, self.state_count)?;
        let next_count =
            self.action_decision_count
                .checked_add(1)
                .ok_or(TabularQError::CounterOverflow {
                    counter: "action_decision_count",
                })?;
        let epsilon = self.current_epsilon();
        let explore = epsilon > 0.0 && self.rng.next_unit_f64() < epsilon;
        let action = if explore {
            self.sample_allowed_action(state)?
        } else {
            self.greedy_action(state)?
        };
        self.action_decision_count = next_count;
        Ok(action)
    }

    /// Select the highest-valued legal action without exploration.
    ///
    /// # Errors
    ///
    /// Returns a state or mask error without changing trainer state.
    pub fn greedy_action(&self, state: usize) -> Result<usize, TabularQError> {
        greedy_action(
            state,
            self.state_count,
            self.action_count,
            &self.q_values,
            self.action_masks.as_deref(),
        )
    }

    /// Apply one tabular Q-learning update.
    ///
    /// Natural termination disables bootstrapping. Truncation alone preserves
    /// bootstrapping so a time limit is not treated as an absorbing state.
    /// Illegal masked actions are rejected and illegal next-state actions are
    /// excluded from the maximum next Q-value.
    ///
    /// # Errors
    ///
    /// Returns an index, mask, finite-value, or counter error without modifying
    /// the Q-table or update count.
    pub fn update(
        &mut self,
        transition: TabularTransition,
    ) -> Result<TabularQUpdate, TabularQError> {
        let index = checked_table_index(
            transition.state,
            transition.action,
            self.state_count,
            self.action_count,
        )?;
        validate_state(transition.next_state, self.state_count)?;
        if !self.is_action_allowed_index(index) {
            return Err(TabularQError::IllegalAction {
                state: transition.state,
                action: transition.action,
            });
        }
        if !transition.reward.is_finite() {
            return Err(TabularQError::NonFiniteValue { field: "reward" });
        }

        let previous_value = self.q_values.get(index).copied().ok_or_else(|| {
            TabularQError::invalid_internal_table(self.state_count, self.action_count)
        })?;
        let bootstrapped = !transition.terminated;
        let next_value = if bootstrapped {
            self.max_allowed_q(transition.next_state)?
        } else {
            0.0
        };
        let target = self.config.gamma.mul_add(next_value, transition.reward);
        if !target.is_finite() {
            return Err(TabularQError::NonFiniteValue {
                field: "bellman_target",
            });
        }
        let new_value = self
            .config
            .learning_rate
            .mul_add(target - previous_value, previous_value);
        if !new_value.is_finite() {
            return Err(TabularQError::NonFiniteValue {
                field: "updated_q_value",
            });
        }
        let update_index =
            self.update_count
                .checked_add(1)
                .ok_or(TabularQError::CounterOverflow {
                    counter: "update_count",
                })?;

        let slot = self.q_values.get_mut(index).ok_or_else(|| {
            TabularQError::invalid_internal_table(self.state_count, self.action_count)
        })?;
        *slot = new_value;
        self.update_count = update_index;

        Ok(TabularQUpdate {
            update_index,
            previous_value,
            target,
            new_value,
            bootstrapped,
            transition_was_truncated: transition.truncated,
        })
    }

    /// Return one Q-value.
    ///
    /// # Errors
    ///
    /// Returns an index error when `state` or `action` is outside the trainer
    /// dimensions.
    pub fn q_value(&self, state: usize, action: usize) -> Result<f64, TabularQError> {
        let index = checked_table_index(state, action, self.state_count, self.action_count)?;
        self.q_values.get(index).copied().ok_or_else(|| {
            TabularQError::invalid_internal_table(self.state_count, self.action_count)
        })
    }

    /// Borrow all row-major Q-values.
    #[must_use]
    pub fn q_values(&self) -> &[f64] {
        &self.q_values
    }

    /// Return the current exploration probability before the next decision.
    #[must_use]
    pub fn current_epsilon(&self) -> f64 {
        self.config.epsilon_at(self.action_decision_count)
    }

    /// Return immutable inference data for deterministic greedy evaluation.
    #[must_use]
    pub fn policy(&self) -> TabularQPolicy {
        TabularQPolicy {
            state_count: self.state_count,
            action_count: self.action_count,
            q_values: self.q_values.clone(),
            action_masks: self.action_masks.clone(),
        }
    }

    /// Return update counters and a stable table fingerprint.
    #[must_use]
    pub fn evidence(&self) -> TabularQEvidence {
        TabularQEvidence {
            update_count: self.update_count,
            action_decision_count: self.action_decision_count,
            table_fingerprint: self.table_fingerprint(),
        }
    }

    /// Return a stable non-cryptographic fingerprint of dimensions and Q-values.
    #[must_use]
    pub fn table_fingerprint(&self) -> String {
        table_fingerprint(self.state_count, self.action_count, &self.q_values)
    }

    /// Borrow the validated configuration.
    #[must_use]
    pub const fn config(&self) -> &TabularQConfig {
        &self.config
    }

    /// Number of encoded states.
    #[must_use]
    pub const fn state_count(&self) -> usize {
        self.state_count
    }

    /// Number of encoded actions.
    #[must_use]
    pub const fn action_count(&self) -> usize {
        self.action_count
    }

    /// Number of successful Bellman updates.
    #[must_use]
    pub const fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Number of epsilon-greedy decisions.
    #[must_use]
    pub const fn action_decision_count(&self) -> u64 {
        self.action_decision_count
    }

    /// Save a versioned checkpoint without replacing an existing path.
    ///
    /// Parent directories are created as needed. The final file is opened with
    /// `create_new`, so an existing caller-supplied `best.mpk` produces
    /// [`io::ErrorKind::AlreadyExists`] and remains untouched.
    ///
    /// # Errors
    ///
    /// Returns a checkpoint I/O error, a finite-value error, or a dimension
    /// conversion error. A failed write removes the newly created partial file.
    #[expect(
        clippy::disallowed_methods,
        reason = "the requested std-only checkpoint core is synchronous and performs bounded file I/O"
    )]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<PathBuf, TabularQError> {
        let path = path.as_ref().to_path_buf();
        let bytes = self.encode_checkpoint()?;
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent).map_err(|error| {
                TabularQError::checkpoint_io(TabularCheckpointOperation::Save, &path, &error)
            })?;
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| {
                TabularQError::checkpoint_io(TabularCheckpointOperation::Save, &path, &error)
            })?;

        if let Err(error) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
            drop(file);
            drop(fs::remove_file(&path));
            return Err(TabularQError::checkpoint_io(
                TabularCheckpointOperation::Save,
                &path,
                &error,
            ));
        }

        Ok(path)
    }

    /// Load a trainer, including masks, counters, and exploration RNG state.
    ///
    /// # Errors
    ///
    /// Returns a checkpoint I/O error or rejects unknown versions, checksum
    /// mismatches, malformed dimensions, invalid masks, non-finite values, or
    /// trailing bytes.
    #[expect(
        clippy::disallowed_methods,
        reason = "the requested std-only checkpoint core is synchronous and performs bounded file I/O"
    )]
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TabularQError> {
        let path = path.as_ref().to_path_buf();
        let bytes = fs::read(&path).map_err(|error| {
            TabularQError::checkpoint_io(TabularCheckpointOperation::Load, &path, &error)
        })?;
        Self::decode_checkpoint(&path, &bytes)
    }

    /// Sample uniformly from the actions allowed in one state.
    fn sample_allowed_action(&mut self, state: usize) -> Result<usize, TabularQError> {
        let allowed_count = (0..self.action_count)
            .filter(|&action| self.is_action_allowed(state, action).unwrap_or(false))
            .count();
        if allowed_count == 0 {
            return Err(TabularQError::NoAllowedActions { state });
        }
        let selected_ordinal = self.rng.index(allowed_count)?;
        (0..self.action_count)
            .filter(|&action| self.is_action_allowed(state, action).unwrap_or(false))
            .nth(selected_ordinal)
            .ok_or(TabularQError::NoAllowedActions { state })
    }

    /// Return the maximum legal next-state Q-value.
    fn max_allowed_q(&self, state: usize) -> Result<f64, TabularQError> {
        let action = self.greedy_action(state)?;
        self.q_value(state, action)
    }

    /// Return whether one checked state/action table slot is legal.
    fn is_action_allowed_index(&self, index: usize) -> bool {
        self.action_masks
            .as_ref()
            .is_none_or(|masks| masks.get(index).copied().unwrap_or(false))
    }

    /// Return whether one state/action pair is legal.
    fn is_action_allowed(&self, state: usize, action: usize) -> Result<bool, TabularQError> {
        let index = checked_table_index(state, action, self.state_count, self.action_count)?;
        Ok(self.is_action_allowed_index(index))
    }

    /// Encode the complete resumable trainer state and append a checksum.
    fn encode_checkpoint(&self) -> Result<Vec<u8>, TabularQError> {
        if self.q_values.iter().any(|value| !value.is_finite()) {
            return Err(TabularQError::NonFiniteValue { field: "q_values" });
        }
        let state_count = u64::try_from(self.state_count).map_err(|_| {
            TabularQError::invalid_config("state_count", "must fit u64 for checkpointing")
        })?;
        let action_count = u64::try_from(self.action_count).map_err(|_| {
            TabularQError::invalid_config("action_count", "must fit u64 for checkpointing")
        })?;
        let q_count = u64::try_from(self.q_values.len())
            .map_err(|_| TabularQError::invalid_config("q_values", "table length must fit u64"))?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(CHECKPOINT_MAGIC);
        push_u32(&mut bytes, TABULAR_Q_CHECKPOINT_VERSION);
        push_u64(&mut bytes, state_count);
        push_u64(&mut bytes, action_count);
        push_f64(&mut bytes, self.config.learning_rate);
        push_f64(&mut bytes, self.config.gamma);
        push_f64(&mut bytes, self.config.epsilon_start);
        push_f64(&mut bytes, self.config.epsilon_end);
        push_u64(&mut bytes, self.config.epsilon_decay_steps);
        push_u64(&mut bytes, self.config.seed);
        push_u64(&mut bytes, self.update_count);
        push_u64(&mut bytes, self.action_decision_count);
        push_u64(&mut bytes, self.rng.state);

        match &self.action_masks {
            Some(masks) => {
                bytes.push(1);
                bytes.extend(masks.iter().map(|allowed| u8::from(*allowed)));
            }
            None => bytes.push(0),
        }

        push_u64(&mut bytes, q_count);
        for &value in &self.q_values {
            push_f64(&mut bytes, value);
        }
        let checksum = fnv1a64(&bytes);
        push_u64(&mut bytes, checksum);
        Ok(bytes)
    }

    /// Decode and validate a complete checkpoint.
    fn decode_checkpoint(path: &Path, bytes: &[u8]) -> Result<Self, TabularQError> {
        let checksum_start = bytes.len().checked_sub(8).ok_or_else(|| {
            TabularQError::invalid_checkpoint(path, "checkpoint is shorter than its checksum")
        })?;
        let (payload, checksum_bytes) = bytes.split_at(checksum_start);
        let expected_checksum = u64::from_le_bytes(checksum_bytes.try_into().map_err(|_| {
            TabularQError::invalid_checkpoint(path, "checkpoint checksum has invalid width")
        })?);
        let actual_checksum = fnv1a64(payload);
        if actual_checksum != expected_checksum {
            return Err(TabularQError::invalid_checkpoint(
                path,
                "checkpoint checksum mismatch",
            ));
        }

        let mut decoder = CheckpointDecoder::new(path, payload);
        if decoder.read_exact(CHECKPOINT_MAGIC.len())? != CHECKPOINT_MAGIC {
            return Err(TabularQError::invalid_checkpoint(
                path,
                "checkpoint magic does not identify tabular Q data",
            ));
        }
        let version = decoder.read_u32()?;
        if version != TABULAR_Q_CHECKPOINT_VERSION {
            return Err(TabularQError::invalid_checkpoint(
                path,
                format!(
                    "unsupported tabular Q checkpoint version {version}; expected {TABULAR_Q_CHECKPOINT_VERSION}"
                ),
            ));
        }
        let state_count = usize::try_from(decoder.read_u64()?).map_err(|_| {
            TabularQError::invalid_checkpoint(path, "state_count does not fit usize")
        })?;
        let action_count = usize::try_from(decoder.read_u64()?).map_err(|_| {
            TabularQError::invalid_checkpoint(path, "action_count does not fit usize")
        })?;
        let config = TabularQConfig {
            learning_rate: decoder.read_f64()?,
            gamma: decoder.read_f64()?,
            epsilon_start: decoder.read_f64()?,
            epsilon_end: decoder.read_f64()?,
            epsilon_decay_steps: decoder.read_u64()?,
            seed: decoder.read_u64()?,
        };
        config
            .validate(state_count, action_count)
            .map_err(|error| {
                TabularQError::invalid_checkpoint(path, format!("invalid saved config: {error}"))
            })?;
        let update_count = decoder.read_u64()?;
        let action_decision_count = decoder.read_u64()?;
        let rng_state = decoder.read_u64()?;
        let table_len = state_count.checked_mul(action_count).ok_or_else(|| {
            TabularQError::invalid_checkpoint(path, "saved table dimensions overflow usize")
        })?;

        let action_masks = match decoder.read_u8()? {
            0 => None,
            1 => {
                let raw_masks = decoder.read_exact(table_len)?;
                if raw_masks.iter().any(|value| !matches!(value, 0 | 1)) {
                    return Err(TabularQError::invalid_checkpoint(
                        path,
                        "saved action mask contains a value other than 0 or 1",
                    ));
                }
                let nested_masks = raw_masks
                    .chunks(action_count)
                    .map(|row| row.iter().map(|value| *value == 1).collect())
                    .collect();
                Some(nested_masks)
            }
            flag => {
                return Err(TabularQError::invalid_checkpoint(
                    path,
                    format!("invalid action-mask presence flag {flag}"),
                ));
            }
        };

        let saved_q_count = usize::try_from(decoder.read_u64()?).map_err(|_| {
            TabularQError::invalid_checkpoint(path, "saved Q-value count does not fit usize")
        })?;
        if saved_q_count != table_len {
            return Err(TabularQError::invalid_checkpoint(
                path,
                format!(
                    "saved Q-value count {saved_q_count} does not match dimensions {state_count}x{action_count}"
                ),
            ));
        }
        let expected_q_bytes = saved_q_count.checked_mul(8).ok_or_else(|| {
            TabularQError::invalid_checkpoint(path, "saved Q-value byte count overflows usize")
        })?;
        if decoder.remaining() != expected_q_bytes {
            return Err(TabularQError::invalid_checkpoint(
                path,
                format!(
                    "saved Q-values require {expected_q_bytes} bytes, but {} remain",
                    decoder.remaining()
                ),
            ));
        }
        let mut q_values = Vec::with_capacity(saved_q_count);
        for _ in 0..saved_q_count {
            let value = decoder.read_f64()?;
            if !value.is_finite() {
                return Err(TabularQError::invalid_checkpoint(
                    path,
                    "saved Q-table contains a non-finite value",
                ));
            }
            q_values.push(value);
        }
        decoder.finish()?;

        let mut trainer = Self::new(state_count, action_count, config, action_masks)
            .map_err(|error| TabularQError::invalid_checkpoint(path, error.to_string()))?;
        trainer.q_values = q_values;
        trainer.update_count = update_count;
        trainer.action_decision_count = action_decision_count;
        trainer.rng.state = rng_state;
        Ok(trainer)
    }
}

/// Checkpoint operation included in tabular Q I/O errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabularCheckpointOperation {
    /// Saving a trainer checkpoint.
    Save,

    /// Loading a trainer checkpoint.
    Load,
}

impl fmt::Display for TabularCheckpointOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Save => "save",
            Self::Load => "load",
        })
    }
}

/// Tabular Q configuration, transition, mask, and checkpoint error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TabularQError {
    /// A trainer configuration field is unusable.
    InvalidConfig {
        /// Invalid field name.
        field: &'static str,

        /// Validation failure.
        message: String,
    },

    /// Per-state action masks do not match the trainer dimensions.
    InvalidActionMasks {
        /// Validation failure.
        message: String,
    },

    /// A state index is outside the finite state space.
    StateOutOfRange {
        /// Rejected state index.
        state: usize,

        /// Number of valid states.
        state_count: usize,
    },

    /// An action index is outside the finite action space.
    ActionOutOfRange {
        /// Rejected action index.
        action: usize,

        /// Number of valid actions.
        action_count: usize,
    },

    /// A masked action was supplied for an update.
    IllegalAction {
        /// Encoded state.
        state: usize,

        /// Rejected action.
        action: usize,
    },

    /// A state mask contains no selectable actions.
    NoAllowedActions {
        /// State whose mask is empty.
        state: usize,
    },

    /// A transition or table value is not finite.
    NonFiniteValue {
        /// Non-finite field name.
        field: &'static str,
    },

    /// An evidence counter cannot be incremented.
    CounterOverflow {
        /// Overflowing counter name.
        counter: &'static str,
    },

    /// A checkpoint filesystem operation failed.
    CheckpointIo {
        /// Attempted operation.
        operation: TabularCheckpointOperation,

        /// Caller-supplied checkpoint path.
        path: PathBuf,

        /// Standard I/O error classification.
        kind: io::ErrorKind,

        /// Human-readable I/O error.
        message: String,
    },

    /// A checkpoint exists but violates the versioned format.
    InvalidCheckpoint {
        /// Caller-supplied checkpoint path.
        path: PathBuf,

        /// Validation failure.
        message: String,
    },
}

impl TabularQError {
    /// Construct a configuration error.
    fn invalid_config(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            field,
            message: message.into(),
        }
    }

    /// Construct an internal table-shape error.
    fn invalid_internal_table(state_count: usize, action_count: usize) -> Self {
        Self::InvalidConfig {
            field: "table_dimensions",
            message: format!(
                "Q-table storage does not match dimensions {state_count}x{action_count}"
            ),
        }
    }

    /// Construct a checkpoint I/O error.
    fn checkpoint_io(
        operation: TabularCheckpointOperation,
        path: &Path,
        error: &io::Error,
    ) -> Self {
        Self::CheckpointIo {
            operation,
            path: path.to_path_buf(),
            kind: error.kind(),
            message: error.to_string(),
        }
    }

    /// Construct a malformed-checkpoint error.
    fn invalid_checkpoint(path: &Path, message: impl Into<String>) -> Self {
        Self::InvalidCheckpoint {
            path: path.to_path_buf(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TabularQError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, message } => {
                write!(formatter, "invalid tabular Q config {field}: {message}")
            }
            Self::InvalidActionMasks { message } => {
                write!(formatter, "invalid tabular Q action masks: {message}")
            }
            Self::StateOutOfRange { state, state_count } => write!(
                formatter,
                "state {state} is outside tabular state count {state_count}"
            ),
            Self::ActionOutOfRange {
                action,
                action_count,
            } => write!(
                formatter,
                "action {action} is outside tabular action count {action_count}"
            ),
            Self::IllegalAction { state, action } => {
                write!(formatter, "action {action} is masked out in state {state}")
            }
            Self::NoAllowedActions { state } => {
                write!(formatter, "state {state} has no allowed actions")
            }
            Self::NonFiniteValue { field } => {
                write!(formatter, "tabular Q field {field} must be finite")
            }
            Self::CounterOverflow { counter } => {
                write!(formatter, "tabular Q counter {counter} overflowed")
            }
            Self::CheckpointIo {
                operation,
                path,
                message,
                ..
            } => write!(
                formatter,
                "tabular Q checkpoint {operation} failed for {}: {message}",
                path.display()
            ),
            Self::InvalidCheckpoint { path, message } => write!(
                formatter,
                "invalid tabular Q checkpoint {}: {message}",
                path.display()
            ),
        }
    }
}

impl Error for TabularQError {}

/// Deterministic `SplitMix64` stream owned by one trainer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SplitMix64 {
    /// Current stream state.
    state: u64,
}

impl SplitMix64 {
    /// Start a stream from a caller-provided seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return the next mixed 64-bit value.
    const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Return a uniformly distributed `f64` in `[0, 1)` using 53 bits.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the upper 53 random bits intentionally populate the f64 mantissa"
    )]
    fn next_unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Sample an unbiased index in `0..upper` with rejection sampling.
    fn index(&mut self, upper: usize) -> Result<usize, TabularQError> {
        let upper = u64::try_from(upper).map_err(|_| {
            TabularQError::invalid_config("action_count", "must fit u64 for sampling")
        })?;
        if upper == 0 {
            return Err(TabularQError::invalid_config(
                "action_count",
                "cannot sample from an empty action set",
            ));
        }
        let rejection_floor = upper.wrapping_neg() % upper;
        loop {
            let value = self.next_u64();
            if value >= rejection_floor {
                return usize::try_from(value % upper).map_err(|_| {
                    TabularQError::invalid_config(
                        "action_count",
                        "sampled action does not fit usize",
                    )
                });
            }
        }
    }
}

/// Validate and flatten optional per-state masks.
fn flatten_action_masks(
    state_count: usize,
    action_count: usize,
    masks: Option<Vec<Vec<bool>>>,
) -> Result<Option<Vec<bool>>, TabularQError> {
    let Some(masks) = masks else {
        return Ok(None);
    };
    if masks.len() != state_count {
        return Err(TabularQError::InvalidActionMasks {
            message: format!(
                "expected {state_count} state rows, received {}",
                masks.len()
            ),
        });
    }

    let capacity = state_count
        .checked_mul(action_count)
        .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))?;
    let mut flattened = Vec::with_capacity(capacity);
    for (state, row) in masks.into_iter().enumerate() {
        if row.len() != action_count {
            return Err(TabularQError::InvalidActionMasks {
                message: format!(
                    "state {state} expected {action_count} actions, received {}",
                    row.len()
                ),
            });
        }
        if !row.iter().any(|allowed| *allowed) {
            return Err(TabularQError::NoAllowedActions { state });
        }
        flattened.extend(row);
    }
    Ok(Some(flattened))
}

/// Return the deterministic lowest-index greedy legal action.
fn greedy_action(
    state: usize,
    state_count: usize,
    action_count: usize,
    q_values: &[f64],
    action_masks: Option<&[bool]>,
) -> Result<usize, TabularQError> {
    validate_state(state, state_count)?;
    let row_start = state
        .checked_mul(action_count)
        .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))?;
    let row_end = row_start
        .checked_add(action_count)
        .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))?;
    let q_row = q_values
        .get(row_start..row_end)
        .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))?;
    let mask_row = match action_masks {
        Some(masks) => Some(
            masks
                .get(row_start..row_end)
                .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))?,
        ),
        None => None,
    };

    let mut best: Option<(usize, f64)> = None;
    for (action, &value) in q_row.iter().enumerate() {
        let allowed = mask_row.is_none_or(|row| row.get(action).copied().unwrap_or(false));
        if allowed && best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((action, value));
        }
    }
    best.map(|(action, _)| action)
        .ok_or(TabularQError::NoAllowedActions { state })
}

/// Validate a state index.
const fn validate_state(state: usize, state_count: usize) -> Result<(), TabularQError> {
    if state >= state_count {
        return Err(TabularQError::StateOutOfRange { state, state_count });
    }
    Ok(())
}

/// Validate state/action indices and return the row-major table index.
fn checked_table_index(
    state: usize,
    action: usize,
    state_count: usize,
    action_count: usize,
) -> Result<usize, TabularQError> {
    validate_state(state, state_count)?;
    if action >= action_count {
        return Err(TabularQError::ActionOutOfRange {
            action,
            action_count,
        });
    }
    state
        .checked_mul(action_count)
        .and_then(|offset| offset.checked_add(action))
        .ok_or_else(|| TabularQError::invalid_internal_table(state_count, action_count))
}

/// Return a stable dimensions-and-values fingerprint.
fn table_fingerprint(state_count: usize, action_count: usize, q_values: &[f64]) -> String {
    let mut hash = FNV_OFFSET_BASIS;
    hash = fnv1a64_continue(hash, &(state_count as u128).to_le_bytes());
    hash = fnv1a64_continue(hash, &(action_count as u128).to_le_bytes());
    for value in q_values {
        hash = fnv1a64_continue(hash, &value.to_bits().to_le_bytes());
    }
    format!("fnv1a64:{hash:016x}")
}

/// Hash bytes with FNV-1a.
fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_continue(FNV_OFFSET_BASIS, bytes)
}

/// Continue an FNV-1a hash from an existing state.
fn fnv1a64_continue(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Append one little-endian `u32`.
fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Append one little-endian `u64`.
fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Append one little-endian `f64` bit pattern.
fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    push_u64(bytes, value.to_bits());
}

/// Bounds-checked decoder for a checkpoint payload without its checksum.
struct CheckpointDecoder<'a> {
    /// Caller-supplied path used in validation errors.
    path: PathBuf,

    /// Payload bytes.
    bytes: &'a [u8],

    /// Next unread byte.
    cursor: usize,
}

impl<'a> CheckpointDecoder<'a> {
    /// Create a decoder at byte zero.
    fn new(path: &Path, bytes: &'a [u8]) -> Self {
        Self {
            path: path.to_path_buf(),
            bytes,
            cursor: 0,
        }
    }

    /// Read an exact byte slice.
    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], TabularQError> {
        let end = self.cursor.checked_add(len).ok_or_else(|| {
            TabularQError::invalid_checkpoint(&self.path, "decoder offset overflow")
        })?;
        let slice = self.bytes.get(self.cursor..end).ok_or_else(|| {
            TabularQError::invalid_checkpoint(&self.path, "unexpected end of checkpoint")
        })?;
        self.cursor = end;
        Ok(slice)
    }

    /// Read one byte.
    fn read_u8(&mut self) -> Result<u8, TabularQError> {
        self.read_exact(1)?
            .first()
            .copied()
            .ok_or_else(|| TabularQError::invalid_checkpoint(&self.path, "missing byte"))
    }

    /// Read one little-endian `u32`.
    fn read_u32(&mut self) -> Result<u32, TabularQError> {
        let bytes: [u8; 4] = self
            .read_exact(4)?
            .try_into()
            .map_err(|_| TabularQError::invalid_checkpoint(&self.path, "invalid u32 width"))?;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Read one little-endian `u64`.
    fn read_u64(&mut self) -> Result<u64, TabularQError> {
        let bytes: [u8; 8] = self
            .read_exact(8)?
            .try_into()
            .map_err(|_| TabularQError::invalid_checkpoint(&self.path, "invalid u64 width"))?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Read one little-endian `f64` bit pattern.
    fn read_f64(&mut self) -> Result<f64, TabularQError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    /// Return unread payload bytes.
    const fn remaining(&self) -> usize {
        self.bytes.len() - self.cursor
    }

    /// Reject unread trailing payload bytes.
    fn finish(self) -> Result<(), TabularQError> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(TabularQError::invalid_checkpoint(
                &self.path,
                format!(
                    "checkpoint contains {} trailing payload bytes",
                    self.bytes.len() - self.cursor
                ),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Unique suffix for parallel checkpoint tests in one process.
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Configuration that makes expected-value tests exact.
    fn exact_config() -> TabularQConfig {
        TabularQConfig {
            learning_rate: 1.0,
            gamma: 0.5,
            epsilon_start: 0.0,
            epsilon_end: 0.0,
            epsilon_decay_steps: 1,
            seed: 7,
        }
    }

    /// Build one transition concisely.
    const fn transition(
        state: usize,
        action: usize,
        reward: f64,
        next_state: usize,
        terminated: bool,
        truncated: bool,
    ) -> TabularTransition {
        TabularTransition {
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
        }
    }

    /// Create an isolated caller-supplied `best.mpk` path.
    fn temp_checkpoint_path() -> PathBuf {
        let suffix = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!(
                "bevy-gym-tabular-q-{}-{suffix}",
                std::process::id()
            ))
            .join("best.mpk")
    }

    /// Assert a scalar is within a tight deterministic-test tolerance.
    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, received {actual}"
        );
    }

    #[test]
    fn config_validation_rejects_inert_or_invalid_training() {
        let config = TabularQConfig {
            learning_rate: 0.0,
            ..TabularQConfig::default()
        };
        assert!(matches!(
            config.validate(2, 2),
            Err(TabularQError::InvalidConfig {
                field: "learning_rate",
                ..
            })
        ));

        let config = TabularQConfig {
            epsilon_start: 0.2,
            epsilon_end: 0.3,
            ..TabularQConfig::default()
        };
        assert!(matches!(
            config.validate(2, 2),
            Err(TabularQError::InvalidConfig {
                field: "epsilon",
                ..
            })
        ));
        assert!(TabularQConfig::default().validate(0, 2).is_err());
    }

    #[test]
    fn masked_selection_and_bootstrap_exclude_illegal_actions() {
        let config = TabularQConfig {
            epsilon_start: 1.0,
            epsilon_end: 1.0,
            ..exact_config()
        };
        let masks = vec![vec![true, false, true], vec![true, false, false]];
        let mut trainer = TabularQTrainer::new(2, 3, config, Some(masks)).expect("valid trainer");

        for _ in 0..128 {
            assert_ne!(trainer.select_action(0).expect("valid state"), 1);
        }

        let state_zero_illegal = checked_table_index(0, 1, 2, 3).expect("valid table index");
        let state_one_allowed = checked_table_index(1, 0, 2, 3).expect("valid table index");
        let state_one_illegal = checked_table_index(1, 1, 2, 3).expect("valid table index");
        *trainer
            .q_values
            .get_mut(state_zero_illegal)
            .expect("table slot") = 1_000.0;
        *trainer
            .q_values
            .get_mut(state_one_allowed)
            .expect("table slot") = 4.0;
        *trainer
            .q_values
            .get_mut(state_one_illegal)
            .expect("table slot") = 1_000.0;

        assert_eq!(trainer.greedy_action(0).expect("masked greedy action"), 0);
        let report = trainer
            .update(transition(0, 0, 1.0, 1, false, false))
            .expect("legal update");
        assert_close(report.target, 3.0);
        assert!(matches!(
            trainer.update(transition(0, 1, 0.0, 1, false, false)),
            Err(TabularQError::IllegalAction {
                state: 0,
                action: 1
            })
        ));
    }

    #[test]
    fn termination_stops_bootstrap_but_truncation_preserves_it() {
        let mut trainer = TabularQTrainer::new(2, 2, exact_config(), None).expect("valid trainer");
        trainer
            .update(transition(1, 0, 4.0, 1, true, false))
            .expect("seed next-state value");
        trainer
            .update(transition(1, 1, 2.0, 1, true, false))
            .expect("seed second next-state value");

        let terminal = trainer
            .update(transition(0, 0, 1.0, 1, true, false))
            .expect("terminal update");
        assert_close(terminal.target, 1.0);
        assert!(!terminal.bootstrapped);

        let truncated = trainer
            .update(transition(0, 1, 1.0, 1, false, true))
            .expect("truncated update");
        assert_close(truncated.target, 3.0);
        assert!(truncated.bootstrapped);
        assert!(truncated.transition_was_truncated);
    }

    #[test]
    fn trainer_learns_a_two_step_mdp_without_imitation() {
        let config = TabularQConfig {
            learning_rate: 0.4,
            gamma: 0.9,
            epsilon_start: 1.0,
            epsilon_end: 0.02,
            epsilon_decay_steps: 300,
            seed: 91,
        };
        let mut trainer = TabularQTrainer::new(2, 2, config, None).expect("valid trainer");
        let initial_fingerprint = trainer.table_fingerprint();

        for _episode in 0..1_000 {
            let first_action = trainer.select_action(0).expect("state zero action");
            if first_action == 0 {
                trainer
                    .update(transition(0, 0, 0.0, 0, true, false))
                    .expect("zero-return branch");
                continue;
            }
            trainer
                .update(transition(0, 1, 0.0, 1, false, false))
                .expect("advance to rewarding state");
            let second_action = trainer.select_action(1).expect("state one action");
            let reward = if second_action == 0 { 1.0 } else { 0.0 };
            trainer
                .update(transition(1, second_action, reward, 1, true, false))
                .expect("terminal reward update");
        }

        let policy = trainer.policy();
        assert_eq!(policy.greedy_action(0).expect("learned first action"), 1);
        assert_eq!(policy.greedy_action(1).expect("learned second action"), 0);
        assert!(policy.q_value(0, 1).expect("learned Q-value") > 0.5);
        assert_ne!(trainer.table_fingerprint(), initial_fingerprint);
        assert!(trainer.update_count() >= 1_000);
    }

    #[test]
    fn same_seed_produces_identical_actions_updates_and_evidence() {
        let config = TabularQConfig {
            learning_rate: 0.25,
            gamma: 0.8,
            epsilon_start: 0.75,
            epsilon_end: 0.1,
            epsilon_decay_steps: 200,
            seed: 4_242,
        };
        let mut first =
            TabularQTrainer::new(3, 3, config.clone(), None).expect("valid first trainer");
        let mut second = TabularQTrainer::new(3, 3, config, None).expect("valid second trainer");

        for step in 0..256 {
            let state = step % 3;
            let first_action = first.select_action(state).expect("first action");
            let second_action = second.select_action(state).expect("second action");
            assert_eq!(first_action, second_action);
            let reward = *[-0.25, 0.0, 0.25]
                .get(first_action)
                .expect("three-action reward lookup");
            let next_state = (state + 1) % 3;
            let terminated = step % 11 == 0;
            let truncated = !terminated && step % 7 == 0;
            let input = transition(
                state,
                first_action,
                reward,
                next_state,
                terminated,
                truncated,
            );
            assert_eq!(
                first.update(input).expect("first update"),
                second.update(input).expect("second update")
            );
        }

        assert_eq!(first.q_values(), second.q_values());
        assert_eq!(first.evidence(), second.evidence());
        assert_eq!(
            first.current_epsilon().to_bits(),
            second.current_epsilon().to_bits()
        );
    }

    #[test]
    #[expect(
        clippy::disallowed_methods,
        reason = "bounded synchronous filesystem setup keeps the std-only checkpoint round-trip test dependency-free"
    )]
    fn versioned_checkpoint_round_trip_is_resumable_and_collision_safe() {
        let path = temp_checkpoint_path();
        if let Some(root) = path.parent() {
            drop(fs::remove_dir_all(root));
        }
        let config = TabularQConfig {
            learning_rate: 0.5,
            gamma: 0.75,
            epsilon_start: 0.8,
            epsilon_end: 0.2,
            epsilon_decay_steps: 100,
            seed: 99,
        };
        let masks = vec![vec![true, false], vec![true, true]];
        let mut trainer = TabularQTrainer::new(2, 2, config, Some(masks)).expect("valid trainer");
        let _action = trainer.select_action(0).expect("advance RNG state");
        trainer
            .update(transition(0, 0, 2.0, 1, false, true))
            .expect("checkpointed update");

        assert_eq!(trainer.save(&path).expect("new checkpoint"), path);
        let bytes = fs::read(&path).expect("read checkpoint header");
        let version_bytes: [u8; 4] = bytes
            .get(CHECKPOINT_MAGIC.len()..CHECKPOINT_MAGIC.len() + 4)
            .expect("version bytes")
            .try_into()
            .expect("four version bytes");
        assert_eq!(
            u32::from_le_bytes(version_bytes),
            TABULAR_Q_CHECKPOINT_VERSION
        );

        let collision = trainer.save(&path).expect_err("existing path is rejected");
        assert!(matches!(
            collision,
            TabularQError::CheckpointIo {
                operation: TabularCheckpointOperation::Save,
                kind: io::ErrorKind::AlreadyExists,
                ..
            }
        ));

        let mut loaded = TabularQTrainer::load(&path).expect("load checkpoint");
        assert_eq!(loaded.config(), trainer.config());
        assert_eq!(loaded.q_values(), trainer.q_values());
        assert_eq!(loaded.evidence(), trainer.evidence());
        assert_eq!(loaded.action_masks, trainer.action_masks);
        assert_eq!(
            loaded.select_action(0).expect("resumed action"),
            trainer.select_action(0).expect("original resumed action")
        );

        if let Some(root) = path.parent() {
            drop(fs::remove_dir_all(root));
        }
    }
}
