//! Two frozen team policies and six separately owned recurrent memories.

use super::{
    encoding::{element, Anchors},
    failure::Failure,
};
use crate::{learning, standing};
use bevy_gym::{
    robots::{DroidAction, DroneAction, RobotId, RobotSlot, RobotSnapshot},
    training::{RecurrentMemory, RecurrentPpoPolicy},
};

/// Weights are shared within each team; every robot owns distinct recurrent state.
pub(super) struct Controllers {
    /// Immutable hover actor and critic weights; inference uses only the actor.
    drone_policy: Box<RecurrentPpoPolicy>,
    /// Immutable standing actor and critic weights; inference uses only the actor.
    droid_policy: Box<RecurrentPpoPolicy>,
    /// Drone memories in stable slot order.
    drone_memory: [RecurrentMemory; 3],
    /// Droid memories in stable slot order.
    droid_memory: [RecurrentMemory; 3],
}

/// All six requests and next memories staged without mutating any current memory.
pub(super) struct Batch {
    /// Three already validated rotor requests.
    pub(super) drones: [DroneAction; 3],
    /// Three already validated joint requests.
    pub(super) droids: [DroidAction; 3],
    /// Separate next drone memories; committed only after shared physics succeeds.
    pub(super) drone_memory: [RecurrentMemory; 3],
    /// Separate next droid memories; committed only after shared physics succeeds.
    pub(super) droid_memory: [RecurrentMemory; 3],
}

impl Controllers {
    /// Load one policy per team and allocate each robot's memory independently.
    pub(super) fn new(drone: RecurrentPpoPolicy, droid: RecurrentPpoPolicy) -> Self {
        let drone_memory = RobotSlot::ALL.map(|_| drone.initial_memory());
        let droid_memory = RobotSlot::ALL.map(|_| droid.initial_memory());
        Self {
            drone_policy: Box::new(drone),
            droid_policy: Box::new(droid),
            drone_memory,
            droid_memory,
        }
    }

    /// Infer all requests from one immutable snapshot and previous per-robot memories.
    pub(super) fn infer(
        &self,
        snapshot: &RobotSnapshot,
        anchors: &Anchors,
    ) -> Result<Batch, Failure> {
        let [first, second, third] = RobotSlot::ALL.map(|slot| self.drone(snapshot, anchors, slot));
        let [(drone_first, drone_first_memory), (drone_second, drone_second_memory), (drone_third, drone_third_memory)] =
            [first?, second?, third?];
        let [first, second, third] = RobotSlot::ALL.map(|slot| self.droid(snapshot, anchors, slot));
        let [(droid_first, droid_first_memory), (droid_second, droid_second_memory), (droid_third, droid_third_memory)] =
            [first?, second?, third?];
        Ok(Batch {
            drones: [drone_first, drone_second, drone_third],
            droids: [droid_first, droid_second, droid_third],
            drone_memory: [drone_first_memory, drone_second_memory, drone_third_memory],
            droid_memory: [droid_first_memory, droid_second_memory, droid_third_memory],
        })
    }

    /// Commit all six staged memories after the corresponding physical frame succeeds.
    pub(super) fn commit(&mut self, drones: [RecurrentMemory; 3], droids: [RecurrentMemory; 3]) {
        self.drone_memory = drones;
        self.droid_memory = droids;
    }

    /// Rebuild each ready robot's memory at the same physical reset boundary.
    pub(super) fn reset(&mut self) {
        self.drone_memory = RobotSlot::ALL.map(|_| self.drone_policy.initial_memory());
        self.droid_memory = RobotSlot::ALL.map(|_| self.droid_policy.initial_memory());
    }

    /// Decode this drone's recurrent output without clamps or substitute motor values.
    fn drone(
        &self,
        snapshot: &RobotSnapshot,
        anchors: &Anchors,
        slot: RobotSlot,
    ) -> Result<(DroneAction, RecurrentMemory), Failure> {
        let inferred = self
            .drone_policy
            .mean_action(
                &anchors.drone(snapshot, slot),
                element(&self.drone_memory, slot),
            )
            .map_err(|cause| Failure::Inference {
                robot: RobotId::Drone(slot),
                cause,
            })?;
        let action = learning::decode_action(&inferred.action)
            .map_err(|cause| Failure::DroneAction { slot, cause })?;
        Ok((action, inferred.next_memory))
    }

    /// Decode this droid's recurrent output without clamps or substitute joint values.
    fn droid(
        &self,
        snapshot: &RobotSnapshot,
        anchors: &Anchors,
        slot: RobotSlot,
    ) -> Result<(DroidAction, RecurrentMemory), Failure> {
        let inferred = self
            .droid_policy
            .mean_action(
                &anchors.droid(snapshot, slot),
                element(&self.droid_memory, slot),
            )
            .map_err(|cause| Failure::Inference {
                robot: RobotId::Droid(slot),
                cause,
            })?;
        let action = standing::encoding::decode(&inferred.action)
            .map_err(|cause| Failure::DroidAction { slot, cause })?;
        Ok((action, inferred.next_memory))
    }
}

#[cfg(test)]
#[path = "controllers_tests.rs"]
mod tests;
