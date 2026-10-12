//! Unrestricted diagnostic or damage-constrained combat actuation.
use super::{RobotId, RobotSlot, RobotWorld, RobotWorldError};
use crate::robots::{DroidAction, DroneMotor, DroneMotorState, RobotHealth};

/// Mutually exclusive mechanics for one complete physical advancement.
#[derive(Debug)]
pub(super) enum Actuation<'a> {
    /// Preserve the standalone shared-world diagnostic's unrestricted actuators.
    Unrestricted,
    /// Damage constraints and physical deaths apply at every solver substep.
    Combat {
        /// Drone health in the fixed slot order.
        drones: &'a mut [RobotHealth; 3],
        /// Droid health in the fixed slot order.
        droids: &'a mut [RobotHealth; 3],
    },
}
impl Actuation<'_> {
    /// Derive motor availability without choosing a living robot's requested thrust.
    pub(super) fn motors(&self, slot: RobotSlot) -> [DroneMotorState; 4] {
        match self {
            Self::Unrestricted => [DroneMotorState::Working; 4],
            Self::Combat { drones, .. } => {
                let [first, second, third] = &**drones;
                let health = match slot {
                    RobotSlot::First => first,
                    RobotSlot::Second => second,
                    RobotSlot::Third => third,
                };
                DroneMotor::ALL.map(|motor| health.motor_state(motor).expect("drone roster health"))
            }
        }
    }

    /// Preserve living policy torques and clear every destroyed body's persistent torque.
    pub(super) fn droid_action(&self, slot: RobotSlot, requested: DroidAction) -> DroidAction {
        match self {
            Self::Unrestricted => requested,
            Self::Combat { droids, .. } => {
                let [first, second, third] = &**droids;
                let health = match slot {
                    RobotSlot::First => first,
                    RobotSlot::Second => second,
                    RobotSlot::Third => third,
                };
                if health.is_alive() {
                    requested
                } else {
                    DroidAction::default()
                }
            }
        }
    }

    /// Destroy health at the first completed substep that observes a fatal contact.
    pub(super) fn contacts(&mut self, world: &RobotWorld) -> Result<(), RobotWorldError> {
        if let Self::Combat { drones, droids } = self {
            for ((slot, drone), droid) in RobotSlot::ALL
                .into_iter()
                .zip(drones.iter_mut())
                .zip(droids.iter_mut())
            {
                for (owner, health) in
                    [(RobotId::Drone(slot), drone), (RobotId::Droid(slot), droid)]
                {
                    if health.is_alive() && world.fatal_contact(owner)? {
                        health.destroy();
                    }
                }
            }
        }
        Ok(())
    }
}
