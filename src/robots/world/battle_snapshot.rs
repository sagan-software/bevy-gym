//! Read-only physical and mechanical combat measurements.
use super::{RobotId, RobotSnapshot};
use crate::robots::{RobotHealth, RobotWeaponPhase};

/// One common physical boundary with health and weapon measurements for all six robots.
#[derive(Debug, Clone)]
pub struct RobotBattleSnapshot {
    /// Authoritative common physical boundary.
    pub(super) physical: RobotSnapshot,
    /// Drone health in stable slot order.
    pub(super) drones: [RobotHealth; 3],
    /// Droid health in stable slot order.
    pub(super) droids: [RobotHealth; 3],
    /// Drone mechanical phases in stable slot order.
    pub(super) drone_weapons: [RobotWeaponPhase; 3],
    /// Droid mechanical phases in stable slot order.
    pub(super) droid_weapons: [RobotWeaponPhase; 3],
}
impl RobotBattleSnapshot {
    /// Borrow the measured physical boundary used to bind complete policy actions.
    #[must_use]
    pub const fn physical(&self) -> &RobotSnapshot {
        &self.physical
    }
    /// Read one robot's damage state without exposing mutable environment state.
    #[must_use]
    pub const fn health(&self, owner: RobotId) -> RobotHealth {
        let (health, slot) = match owner {
            RobotId::Drone(slot) => (&self.drones, slot),
            RobotId::Droid(slot) => (&self.droids, slot),
        };
        let [first, second, third] = health;
        match slot {
            super::RobotSlot::First => *first,
            super::RobotSlot::Second => *second,
            super::RobotSlot::Third => *third,
        }
    }
    /// Read one robot's mechanical weapon state.
    #[must_use]
    pub const fn weapon_phase(&self, owner: RobotId) -> RobotWeaponPhase {
        let (weapons, slot) = match owner {
            RobotId::Drone(slot) => (&self.drone_weapons, slot),
            RobotId::Droid(slot) => (&self.droid_weapons, slot),
        };
        let [first, second, third] = weapons;
        match slot {
            super::RobotSlot::First => *first,
            super::RobotSlot::Second => *second,
            super::RobotSlot::Third => *third,
        }
    }
}
