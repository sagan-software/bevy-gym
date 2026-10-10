//! Frozen six-agent inference with one shared physical decision boundary.

use bevy_gym::robots::{
    DroidAction, DroneAction, RobotActions, RobotSlot, RobotSnapshot, RobotWorld,
};
use sha2::{digest::Output, Digest, Sha256};
use std::{error::Error, time::Duration};

use crate::{learning, standing::checkpoint};
use controllers::{Batch, Controllers};
use encoding::{element, Anchors};
use failure::Failure;

/// Frozen policies and separately owned per-robot memories.
mod controllers;
/// Own-body actor inputs and passive task anchors.
mod encoding;
/// Typed failures with the original causes.
mod failure;

/// Shared-world diagnostic; retained controllers do not establish competition qualification.
pub(crate) struct Session {
    /// Sole owner of shared physical integration and collision detection.
    world: RobotWorld,
    /// Last successfully completed frame; policy reads always use this snapshot.
    snapshot: RobotSnapshot,
    /// Fixed task anchors derived from the common reset state.
    anchors: Anchors,
    /// Frozen inference capability or its first permanent failure.
    inference: Inference,
    /// Requests from the last successfully integrated frame, for read-only inspection.
    last_action: Option<Applied>,
    /// Identity of the exact recorded hover bytes accepted by construction.
    drone_digest: Output<Sha256>,
    /// Validated standing checkpoint identity and claimed PPO counters.
    droid_record: checkpoint::Record,
}

/// Advancement requires both team policies and all six separately owned memories.
enum Inference {
    /// Frozen policies may stage bounded actuator requests.
    Ready(Controllers),
    /// The original failure remains visible; policies are dropped and cannot act.
    Failed(Failure),
}

/// Validated requests actually applied during one common physical frame.
struct Applied {
    /// Drone requests in stable slot order.
    drones: [DroneAction; 3],
    /// Droid requests in stable slot order.
    droids: [DroidAction; 3],
}

/// Retained RL hover reference; a different file cannot borrow this provenance.
const DRONE_REFERENCE: &[u8] = include_bytes!("../../../docs/progress/drone-hover.mpk");
/// Diagnostic clip limit; completion derives from physical elapsed time.
const HORIZON: Duration = Duration::from_secs(20);

impl Session {
    /// Validate both retained RL checkpoint identities before constructing physics.
    pub(crate) fn load(
        drone_bytes: Vec<u8>,
        droid_bytes: Vec<u8>,
        metadata: &[u8],
    ) -> Result<Self, Box<dyn Error>> {
        // Both byte identities and model architectures precede any physical construction.
        if drone_bytes.as_slice() != DRONE_REFERENCE {
            return Err(
                "shared-world drone weights must match the recorded PPO hover reference".into(),
            );
        }
        let drone_digest = Sha256::digest(&drone_bytes);
        let (droid_policy, droid_record) = checkpoint::from_bytes(droid_bytes, metadata)?;
        let drone_policy = learning::load_policy(drone_bytes)?;
        let controllers = Controllers::new(drone_policy, droid_policy);
        let world = RobotWorld::default();
        let snapshot = world.snapshot()?;
        let anchors = Anchors::new(&snapshot);
        Ok(Self {
            world,
            snapshot,
            anchors,
            inference: Inference::Ready(controllers),
            last_action: None,
            drone_digest,
            droid_record,
        })
    }

    /// Borrow the last complete physical snapshot.
    pub(crate) const fn snapshot(&self) -> &RobotSnapshot {
        &self.snapshot
    }

    /// Infer all six bounded requests before advancing the common world.
    pub(crate) fn step(&mut self) {
        if self.finished() {
            return;
        }
        let Inference::Ready(controllers) = &mut self.inference else {
            return;
        };
        // Inference is read-only until every request and next memory has been staged.
        let batch = match controllers.infer(&self.snapshot, &self.anchors) {
            Ok(batch) => batch,
            Err(cause) => {
                self.inference = Inference::Failed(cause);
                return;
            }
        };
        let Batch {
            drones,
            droids,
            drone_memory,
            droid_memory,
        } = batch;
        let actions = RobotActions::new(&self.snapshot, drones, droids);
        match self.world.advance(actions) {
            Ok(snapshot) => {
                // One successful common physical frame commits all six memories together.
                controllers.commit(drone_memory, droid_memory);
                self.snapshot = snapshot;
                self.last_action = Some(Applied { drones, droids });
            }
            Err(cause) => self.inference = Inference::Failed(Failure::World(cause)),
        }
    }

    /// Read one applied motor request without allowing manual control.
    pub(crate) fn last_drone_action(&self, slot: RobotSlot) -> Option<DroneAction> {
        self.last_action
            .as_ref()
            .map(|last| *element(&last.drones, slot))
    }

    /// Read one applied joint request without allowing manual control.
    pub(crate) fn last_droid_action(&self, slot: RobotSlot) -> Option<DroidAction> {
        self.last_action
            .as_ref()
            .map(|last| *element(&last.droids, slot))
    }

    /// Borrow the original failure after all controllers stop.
    pub(crate) fn error(&self) -> Option<&dyn Error> {
        match &self.inference {
            Inference::Ready(_) => None,
            Inference::Failed(cause) => Some(cause),
        }
    }

    /// Read completion from the successful physical clock without a duplicate step counter.
    pub(crate) fn finished(&self) -> bool {
        self.snapshot.elapsed() >= HORIZON
    }

    /// Reset all bodies and ready-agent memories; a failed policy remains unusable.
    pub(crate) fn reset(&mut self) {
        match self.world.reset() {
            Ok(snapshot) => {
                self.anchors = Anchors::new(&snapshot);
                self.snapshot = snapshot;
                self.last_action = None;
                if let Inference::Ready(controllers) = &mut self.inference {
                    controllers.reset();
                }
            }
            Err(cause) => self.inference = Inference::Failed(Failure::World(cause)),
        }
    }

    /// Borrow the actual hover identity; this does not qualify the shared task.
    pub(crate) const fn drone_digest(&self) -> &Output<Sha256> {
        &self.drone_digest
    }

    /// Borrow standing provenance without exposing a policy or mutable physical state.
    pub(crate) const fn droid_record(&self) -> &checkpoint::Record {
        &self.droid_record
    }
}

#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
