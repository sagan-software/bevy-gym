//! Safe ownership and dimension checks around the `MuJoCo` 3.9 runtime.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::Path;

use mujoco_rs::prelude::{MjData, MjModel, MjtObj};

/// One owned `MuJoCo` model and its mutable simulation data.
#[derive(Debug)]
pub struct MujocoSimulation {
    /// Data owns the boxed model and keeps its native pointer valid.
    data: MjData<Box<MjModel>>,
}

impl MujocoSimulation {
    /// Load a `MuJoCo` XML model from disk.
    ///
    /// # Errors
    /// Returns [`MujocoSimulationError::LoadModel`] when `MuJoCo` rejects the XML.
    pub fn from_xml(path: impl AsRef<Path>) -> Result<Self, MujocoSimulationError> {
        let model = MjModel::from_xml(path)
            .map_err(|source| MujocoSimulationError::LoadModel(source.to_string()))?;
        Ok(Self {
            data: MjData::new(Box::new(model)),
        })
    }

    /// Load a self-contained `MuJoCo` XML model from memory.
    ///
    /// # Errors
    /// Returns [`MujocoSimulationError::LoadModel`] when `MuJoCo` rejects the XML.
    pub fn from_xml_string(xml: &str) -> Result<Self, MujocoSimulationError> {
        let model = MjModel::from_xml_string(xml)
            .map_err(|source| MujocoSimulationError::LoadModel(source.to_string()))?;
        Ok(Self {
            data: MjData::new(Box::new(model)),
        })
    }

    /// Return the generalized positions in `MuJoCo` order.
    #[must_use]
    pub fn qpos(&self) -> &[f64] {
        self.data.qpos()
    }

    /// Return the generalized velocities in `MuJoCo` order.
    #[must_use]
    pub fn qvel(&self) -> &[f64] {
        self.data.qvel()
    }

    /// Return generalized constraint forces in `MuJoCo` order.
    #[must_use]
    pub fn constraint_forces(&self) -> &[f64] {
        self.data.qfrc_constraint()
    }

    /// Return the current simulation time in seconds.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.data.time()
    }

    /// Return a named body's world-space position.
    ///
    /// # Errors
    /// Returns [`MujocoSimulationError::UnknownBody`] when the model has no such body.
    pub fn body_position(&self, name: &str) -> Result<[f64; 3], MujocoSimulationError> {
        // Translate native MuJoCo state without exposing unchecked wrapper internals to callers.
        let id = self
            .data
            .model()
            .name_to_id(MjtObj::mjOBJ_BODY, name)
            .ok_or_else(|| MujocoSimulationError::UnknownBody(name.to_owned()))?;
        self.data
            .xpos()
            .get(id)
            .copied()
            .ok_or_else(|| MujocoSimulationError::UnknownBody(name.to_owned()))
    }

    /// Return a named site's world-space position.
    ///
    /// # Errors
    /// Returns [`MujocoSimulationError::UnknownSite`] when the model has no such site.
    pub fn site_position(&self, name: &str) -> Result<[f64; 3], MujocoSimulationError> {
        // Translate native MuJoCo state without exposing unchecked wrapper internals to callers.
        let id = self
            .data
            .model()
            .name_to_id(MjtObj::mjOBJ_SITE, name)
            .ok_or_else(|| MujocoSimulationError::UnknownSite(name.to_owned()))?;
        self.data
            .site_xpos()
            .get(id)
            .copied()
            .ok_or_else(|| MujocoSimulationError::UnknownSite(name.to_owned()))
    }

    /// Reset all native state to the model defaults.
    pub fn reset(&mut self) {
        self.data.reset();
    }

    /// Replace generalized position and velocity, then recompute derived state.
    ///
    /// # Errors
    /// Returns a dimension error when either input does not match the model.
    pub fn set_state(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<(), MujocoSimulationError> {
        if qpos.len() != self.data.qpos().len() {
            return Err(MujocoSimulationError::PositionDimension {
                expected: self.data.qpos().len(),
                actual: qpos.len(),
            });
        }
        if qvel.len() != self.data.qvel().len() {
            return Err(MujocoSimulationError::VelocityDimension {
                expected: self.data.qvel().len(),
                actual: qvel.len(),
            });
        }
        // Write the complete native state before forwarding kinematics so all
        // derived body and site positions describe the new reset state.
        self.data.qpos_mut().copy_from_slice(qpos);
        self.data.qvel_mut().copy_from_slice(qvel);
        self.data.forward();
        Ok(())
    }

    /// Apply controls and advance the requested number of native frames.
    ///
    /// # Errors
    /// Returns a dimension error for invalid controls or zero frame skip.
    pub fn step(
        &mut self,
        controls: &[f64],
        frame_skip: usize,
    ) -> Result<(), MujocoSimulationError> {
        if frame_skip == 0 {
            return Err(MujocoSimulationError::ZeroFrameSkip);
        }
        if controls.len() != self.data.ctrl().len() {
            return Err(MujocoSimulationError::ControlDimension {
                expected: self.data.ctrl().len(),
                actual: controls.len(),
            });
        }
        // Hold one Gymnasium action constant across its configured frame skip.
        self.data.ctrl_mut().copy_from_slice(controls);
        for _ in 0..frame_skip {
            self.data.step();
        }
        Ok(())
    }
}

/// Failures at the safe `MuJoCo` simulation boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MujocoSimulationError {
    /// `MuJoCo` could not parse or compile the model.
    LoadModel(String),
    /// A generalized-position vector had the wrong width.
    PositionDimension {
        /// Model width.
        expected: usize,
        /// Supplied width.
        actual: usize,
    },
    /// A generalized-velocity vector had the wrong width.
    VelocityDimension {
        /// Model width.
        expected: usize,
        /// Supplied width.
        actual: usize,
    },
    /// An action vector had the wrong width.
    ControlDimension {
        /// Model width.
        expected: usize,
        /// Supplied width.
        actual: usize,
    },
    /// A simulation step requested no native frames.
    ZeroFrameSkip,
    /// The model does not define the named body.
    UnknownBody(String),
    /// The model does not define the named site.
    UnknownSite(String),
}

impl Display for MujocoSimulationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        // Translate native MuJoCo state without exposing unchecked wrapper internals to callers.
        match self {
            Self::LoadModel(message) => write!(formatter, "MuJoCo model load failed: {message}"),
            Self::PositionDimension { expected, actual } => write!(
                formatter,
                "MuJoCo qpos dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::VelocityDimension { expected, actual } => write!(
                formatter,
                "MuJoCo qvel dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::ControlDimension { expected, actual } => write!(
                formatter,
                "MuJoCo control dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::ZeroFrameSkip => formatter.write_str("MuJoCo frame skip must be at least one"),
            Self::UnknownBody(name) => write!(formatter, "MuJoCo body `{name}` does not exist"),
            Self::UnknownSite(name) => write!(formatter, "MuJoCo site `{name}` does not exist"),
        }
    }
}

impl Error for MujocoSimulationError {}
