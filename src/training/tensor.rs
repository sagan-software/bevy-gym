//! Tensorization boundary types and errors.

use std::error::Error;
use std::fmt;

/// Tensor element kind expected by a trainer boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    /// 32-bit float tensor.
    F32,

    /// 64-bit signed integer tensor.
    I64,
}

/// Observation shape and element type required by a trainer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationSpec {
    /// Observation dimensions excluding batch dimension.
    pub shape: Vec<usize>,

    /// Observation element kind.
    pub dtype: TensorDType,
}

impl ObservationSpec {
    /// Create a validated observation spec.
    ///
    /// # Errors
    ///
    /// Returns [`TensorizationError::EmptyObservationShape`] when shape is empty.
    pub fn new(
        shape: impl Into<Vec<usize>>,
        dtype: TensorDType,
    ) -> Result<Self, TensorizationError> {
        let shape = shape.into();
        if shape.is_empty() {
            return Err(TensorizationError::EmptyObservationShape);
        }
        if shape.contains(&0) {
            return Err(TensorizationError::ZeroObservationDimension);
        }

        Ok(Self { shape, dtype })
    }
}

/// Discrete action-space tensorization requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscreteActionSpec {
    /// Number of valid discrete actions.
    pub actions: usize,
}

impl DiscreteActionSpec {
    /// Create a validated discrete action spec.
    ///
    /// # Errors
    ///
    /// Returns [`TensorizationError::InvalidActionCount`] when there are no actions.
    pub const fn new(actions: usize) -> Result<Self, TensorizationError> {
        if actions == 0 {
            return Err(TensorizationError::InvalidActionCount);
        }

        Ok(Self { actions })
    }
}

/// Algorithm-facing action-space tensorization requirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionSpec {
    /// Discrete action space.
    Discrete(DiscreteActionSpec),
}

/// Errors raised before trainer tensor construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorizationError {
    /// The selected trainer requires an observation spec.
    MissingObservationSpec,

    /// The selected trainer requires an action spec.
    MissingActionSpec,

    /// Observation shape cannot be empty.
    EmptyObservationShape,

    /// Observation dimensions must all be non-zero.
    ZeroObservationDimension,

    /// Discrete action count must be at least one.
    InvalidActionCount,

    /// Observation or action shape did not match the trainer boundary spec.
    ShapeMismatch {
        /// Expected shape.
        expected: Vec<usize>,

        /// Actual shape.
        actual: Vec<usize>,
    },
}

impl fmt::Display for TensorizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingObservationSpec => formatter.write_str(
                "trainer boundary requires an observation spec; keep Env unchanged and provide tensorization metadata",
            ),
            Self::MissingActionSpec => formatter.write_str(
                "trainer boundary requires an action spec; keep Env unchanged and provide tensorization metadata",
            ),
            Self::EmptyObservationShape => formatter.write_str(
                "trainer boundary observation spec must include at least one dimension",
            ),
            Self::ZeroObservationDimension => formatter.write_str(
                "trainer boundary observation spec dimensions must all be non-zero",
            ),
            Self::InvalidActionCount => formatter.write_str(
                "trainer boundary discrete action spec must contain at least one action",
            ),
            Self::ShapeMismatch { expected, actual } => write!(
                formatter,
                "trainer boundary tensor shape mismatch: expected {expected:?}, actual {actual:?}"
            ),
        }
    }
}

impl Error for TensorizationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorization_errors_point_to_trainer_boundary() {
        let message = TensorizationError::MissingObservationSpec.to_string();

        assert!(message.contains("trainer boundary"));
        assert!(message.contains("keep Env unchanged"));
    }

    #[test]
    fn specs_validate_empty_shapes_and_action_counts() {
        assert_eq!(
            ObservationSpec::new([], TensorDType::F32),
            Err(TensorizationError::EmptyObservationShape)
        );
        assert_eq!(
            ObservationSpec::new([4, 0], TensorDType::F32),
            Err(TensorizationError::ZeroObservationDimension)
        );
        assert_eq!(
            DiscreteActionSpec::new(0),
            Err(TensorizationError::InvalidActionCount)
        );

        assert_eq!(
            ObservationSpec::new([4], TensorDType::F32)
                .expect("valid observation shape")
                .shape,
            vec![4]
        );
        assert_eq!(
            DiscreteActionSpec::new(2)
                .expect("valid action count")
                .actions,
            2
        );
    }
}
