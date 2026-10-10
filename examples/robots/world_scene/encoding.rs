//! Own-body inputs with passive task anchors derived from the reset snapshot.

use crate::{learning, standing};
use bevy::math::Vec3;
use bevy_gym::robots::{DroidBody, RobotSlot, RobotSnapshot};

/// Fixed task positions; these are environment setup rather than navigation decisions.
pub(super) struct Anchors {
    /// Each drone balances above its initial world-space position, in metres.
    drones: [Vec3; 3],
    /// Each droid balances relative to its own original rig's translated origin.
    droids: [Vec3; 3],
}

impl Anchors {
    /// Derive task positions from the shared reset and the original standing profile.
    pub(super) fn new(snapshot: &RobotSnapshot) -> Self {
        let drones = RobotSlot::ALL.map(|slot| snapshot.drone(slot).position());
        let droids = RobotSlot::ALL.map(|slot| {
            let pelvis = snapshot.droid(slot).body(DroidBody::Pelvis).position();
            // The original goal height is 0.99 m; rest pelvis offset is +0.028 m on Z.
            Vec3::new(pelvis.x, 0.99, pelvis.z - 0.028)
        });
        Self { drones, droids }
    }

    /// Encode only this drone's body and passive hover target with the original scales.
    pub(super) fn drone(&self, snapshot: &RobotSnapshot, slot: RobotSlot) -> [f32; 12] {
        learning::encoding::encode_target(snapshot.drone(slot), *element(&self.drones, slot))
    }

    /// Retain all original standing features while expressing the task origin in this lane.
    pub(super) fn droid(
        &self,
        snapshot: &RobotSnapshot,
        slot: RobotSlot,
    ) -> [f32; standing::encoding::FEATURES] {
        let observation = snapshot.droid(slot);
        let mut features = standing::encoding::encode(&observation);
        let pelvis = observation.body(DroidBody::Pelvis);
        // Displacement is metres divided by the original 1 m scale, in pelvis coordinates.
        let displacement =
            pelvis.orientation().inverse() * (*element(&self.droids, slot) - pelvis.position());
        let (root_displacement, _) = features.split_at_mut(3);
        root_displacement.copy_from_slice(&displacement.to_array());
        features
    }
}

/// Borrow one closed slot without indexing, allocation or cloning its value.
pub(super) const fn element<T>(values: &[T; 3], slot: RobotSlot) -> &T {
    let [first, second, third] = values;
    match slot {
        RobotSlot::First => first,
        RobotSlot::Second => second,
        RobotSlot::Third => third,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::robots::{DroidAction, DroneAction, RobotActions, RobotWorld};

    /// Moving other robots changes hidden state without changing either focal actor input.
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn actor_inputs_exclude_other_robot_state() {
        let mut reference = RobotWorld::default();
        let mut changed = RobotWorld::default();
        let initial = reference.snapshot().expect("reference reset");
        let other = changed.snapshot().expect("changed reset");
        let anchors = Anchors::new(&initial);
        let zero_drone = DroneAction::try_from([0.0; 4]).expect("test rotor fixture");
        let full_drone = DroneAction::try_from([1.0; 4]).expect("test rotor fixture");
        let zero_droid = DroidAction::try_from([0.0; 26]).expect("test joint fixture");
        let full_droid = DroidAction::try_from([1.0; 26]).expect("test joint fixture");
        let reference = reference
            .advance(RobotActions::new(
                &initial,
                [zero_drone; 3],
                [zero_droid; 3],
            ))
            .expect("reference step");
        let changed = changed
            .advance(RobotActions::new(
                &other,
                [zero_drone, full_drone, full_drone],
                [zero_droid, full_droid, full_droid],
            ))
            .expect("changed step");
        assert_ne!(
            reference.drone(RobotSlot::Third),
            changed.drone(RobotSlot::Third)
        );
        assert_ne!(
            reference.droid(RobotSlot::Third),
            changed.droid(RobotSlot::Third)
        );
        assert_eq!(
            anchors
                .drone(&reference, RobotSlot::First)
                .map(f32::to_bits),
            anchors.drone(&changed, RobotSlot::First).map(f32::to_bits)
        );
        assert_eq!(
            anchors
                .droid(&reference, RobotSlot::First)
                .map(f32::to_bits),
            anchors.droid(&changed, RobotSlot::First).map(f32::to_bits)
        );
    }
}
