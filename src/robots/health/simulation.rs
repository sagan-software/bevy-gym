//! Team-derived life and damage transitions.
use super::{
    hull::{DroidHull, DroneHull},
    RobotDamage, RobotHitCount, RobotRotorHealth, RobotRotorHitRejected,
};
use crate::robots::{DroneMotor, DroneMotorState, RobotTeam};
/// Hull and rotor damage without choosing an agent action.
/// Death is absorbing; reset constructs fresh health from the same team.
///
/// ```compile_fail
/// use bevy_gym::robots::{RobotHealth, RobotTeam};
/// let mut health = RobotHealth::from(RobotTeam::Drones);
/// health.state = ();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RobotHealth {
    /// Team, life and applicable rotor payload share one closed state.
    state: State,
}
/// Only drone variants can retain rotors; dead variants cannot retain hull life.
#[derive(Debug, Clone, Copy)]
enum State {
    /// One through six drone hits, with independently damageable rotors.
    Drone {
        /// Positive life bounded by the drone's six-hit hull.
        hull: DroneHull,
        /// Independent rotor integrity in the actuator order.
        rotors: [RobotRotorHealth; 4],
    },
    /// A destroyed drone retains its last rotor observations.
    DeadDrone([RobotRotorHealth; 4]),
    /// One through three droid hits.
    Droid(DroidHull),
    /// Destroyed droid with no remaining hull or rotor payload.
    DeadDroid,
}
impl From<RobotTeam> for RobotHealth {
    fn from(team: RobotTeam) -> Self {
        let state = match team {
            RobotTeam::Drones => State::Drone {
                hull: DroneHull::Six,
                rotors: [RobotRotorHealth::Intact; 4],
            },
            RobotTeam::Droids => State::Droid(DroidHull::Three),
        };
        Self { state }
    }
}
impl RobotHealth {
    /// Derive team identity, including after death.
    #[must_use]
    pub const fn team(self) -> RobotTeam {
        match self.state {
            State::Drone { .. } | State::DeadDrone(_) => RobotTeam::Drones,
            State::Droid(_) | State::DeadDroid => RobotTeam::Droids,
        }
    }
    /// Read whether the hull retains life, independent of rotor failures.
    #[must_use]
    pub const fn is_alive(self) -> bool {
        matches!(self.state, State::Drone { .. } | State::Droid(_))
    }
    /// Read the authoritative bounded hull count.
    #[must_use]
    pub fn body_hits_remaining(self) -> RobotHitCount {
        match self.state {
            State::Drone { hull, .. } => hull.into(),
            State::Droid(hull) => hull.into(),
            State::DeadDrone(_) | State::DeadDroid => RobotHitCount::ZERO,
        }
    }
    /// Read a named rotor, preserving its last state after hull destruction.
    ///
    /// # Errors
    /// Rejects both living and destroyed droids before reading rotor state.
    pub const fn rotor(self, motor: DroneMotor) -> Result<RobotRotorHealth, RobotRotorHitRejected> {
        match self.state {
            State::Drone { rotors, .. } | State::DeadDrone(rotors) => Ok(read_rotor(rotors, motor)),
            State::Droid(_) | State::DeadDroid => Err(RobotRotorHitRejected),
        }
    }
    /// Derive motor availability from both hull life and rotor integrity.
    ///
    /// # Errors
    /// Rejects both living and destroyed droids.
    pub const fn motor_state(
        self,
        motor: DroneMotor,
    ) -> Result<DroneMotorState, RobotRotorHitRejected> {
        // Hull death disables every motor even when its last rotor observation was intact.
        match self.state {
            State::Drone { rotors, .. } => match read_rotor(rotors, motor) {
                RobotRotorHealth::Intact | RobotRotorHealth::Damaged => {
                    Ok(DroneMotorState::Working)
                }
                RobotRotorHealth::Destroyed => Ok(DroneMotorState::Failed),
            },
            State::DeadDrone(_) => Ok(DroneMotorState::Failed),
            State::Droid(_) | State::DeadDroid => Err(RobotRotorHitRejected),
        }
    }
    /// Consume one body hit and emit hull destruction only once.
    pub const fn hit_body(&mut self) -> RobotDamage {
        // Closed team-specific hulls prevent a living zero or a droid with excess life.
        match self.state {
            State::Drone { hull, rotors } => match hull.next() {
                Some(hull) => {
                    self.state = State::Drone { hull, rotors };
                    RobotDamage::Hit
                }
                None => self.destroy(),
            },
            State::Droid(hull) => match hull.next() {
                Some(hull) => {
                    self.state = State::Droid(hull);
                    RobotDamage::Hit
                }
                None => self.destroy(),
            },
            State::DeadDrone(_) | State::DeadDroid => RobotDamage::Ignored,
        }
    }
    /// Consume one rotor hit; destruction leaves hull life unchanged.
    ///
    /// # Errors
    /// Rejects both living and destroyed droids without changing health.
    pub const fn hit_rotor(
        &mut self,
        motor: DroneMotor,
    ) -> Result<RobotDamage, RobotRotorHitRejected> {
        // Validate the hull kind before lifecycle denial; a dead droid still has no rotors.
        match &mut self.state {
            State::Drone { rotors, .. } => {
                let rotor = rotor_mut(rotors, motor);
                match rotor {
                    RobotRotorHealth::Intact => {
                        *rotor = RobotRotorHealth::Damaged;
                        Ok(RobotDamage::Hit)
                    }
                    RobotRotorHealth::Damaged => {
                        *rotor = RobotRotorHealth::Destroyed;
                        Ok(RobotDamage::RotorDestroyed(motor))
                    }
                    RobotRotorHealth::Destroyed => Ok(RobotDamage::Ignored),
                }
            }
            State::DeadDrone(_) => Ok(RobotDamage::Ignored),
            State::Droid(_) | State::DeadDroid => Err(RobotRotorHitRejected),
        }
    }
    /// Fatal physical contact disables the hull once and preserves team identity.
    pub const fn destroy(&mut self) -> RobotDamage {
        match self.state {
            State::Drone { rotors, .. } => {
                self.state = State::DeadDrone(rotors);
                RobotDamage::Destroyed
            }
            State::Droid(_) => {
                self.state = State::DeadDroid;
                RobotDamage::Destroyed
            }
            State::DeadDrone(_) | State::DeadDroid => RobotDamage::Ignored,
        }
    }
}

/// Read by the named physical storage order, without an unchecked index.
const fn read_rotor(rotors: [RobotRotorHealth; 4], motor: DroneMotor) -> RobotRotorHealth {
    let [front_left, front_right, rear_right, rear_left] = rotors;
    match motor {
        DroneMotor::FrontLeft => front_left,
        DroneMotor::FrontRight => front_right,
        DroneMotor::RearRight => rear_right,
        DroneMotor::RearLeft => rear_left,
    }
}

/// Borrow exactly one named rotor without exposing any mutable hull payload.
const fn rotor_mut(rotors: &mut [RobotRotorHealth; 4], motor: DroneMotor) -> &mut RobotRotorHealth {
    let [front_left, front_right, rear_right, rear_left] = rotors;
    match motor {
        DroneMotor::FrontLeft => front_left,
        DroneMotor::FrontRight => front_right,
        DroneMotor::RearRight => rear_right,
        DroneMotor::RearLeft => rear_left,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Physical array positions must agree with named access, not just with each other.
    #[test]
    fn storage_indices_follow_physical_motor_order() {
        use RobotRotorHealth::{Damaged, Intact};
        for (motor, expected) in [
            (DroneMotor::FrontLeft, [Damaged, Intact, Intact, Intact]),
            (DroneMotor::FrontRight, [Intact, Damaged, Intact, Intact]),
            (DroneMotor::RearRight, [Intact, Intact, Damaged, Intact]),
            (DroneMotor::RearLeft, [Intact, Intact, Intact, Damaged]),
        ] {
            let mut rotors = [Intact; 4];
            *rotor_mut(&mut rotors, motor) = Damaged;
            assert_eq!(rotors, expected);
            for (known, value) in DroneMotor::ALL.into_iter().zip(expected) {
                assert_eq!(read_rotor(rotors, known), value);
            }
        }
    }
}
