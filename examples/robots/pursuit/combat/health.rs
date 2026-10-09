//! Body and rotor damage with one-time destruction transitions.

use bevy_gym::robots::DroneMotor;
use std::num::NonZeroU8;

/// Remaining body life and independently damageable rotor weak points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DroneHealth {
    /// Six pistol hits destroy the body; absence means the drone is dead.
    body: Option<NonZeroU8>,
    /// Motor order follows `DroneMotor::ALL`.
    rotors: [RotorHealth; 4],
}

/// A rotor needs two pistol hits to lose its motor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RotorHealth {
    /// No pistol hit has reached this rotor.
    #[default]
    Intact,
    /// One more hit destroys this rotor.
    Damaged,
    /// Further hits cannot repeat the destruction event.
    Destroyed,
}

/// The transition caused by one collision or hit, consumed by effects and physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Damage {
    /// An already destroyed part or dead drone does not change.
    Ignored,
    /// Health decreased without destroying a part.
    Hit,
    /// Disable this motor and emit its destruction effect once.
    RotorDestroyed(DroneMotor),
    /// Stop the drone and emit its body destruction effect once.
    Destroyed,
}

impl Default for DroneHealth {
    fn default() -> Self {
        Self {
            body: NonZeroU8::new(6),
            rotors: [RotorHealth::Intact; 4],
        }
    }
}

impl DroneHealth {
    /// A body with remaining hits can still act, even with failed rotors.
    pub(crate) const fn is_alive(&self) -> bool {
        self.body.is_some()
    }

    /// Read the remaining body hits for a health indicator.
    pub(crate) fn body_hits_remaining(&self) -> u8 {
        self.body.map_or(0, NonZeroU8::get)
    }

    /// Read one named weak point without exposing mutable health.
    pub(crate) const fn rotor(&self, motor: DroneMotor) -> RotorHealth {
        match motor {
            DroneMotor::FrontLeft => self.rotors[0],
            DroneMotor::FrontRight => self.rotors[1],
            DroneMotor::RearRight => self.rotors[2],
            DroneMotor::RearLeft => self.rotors[3],
        }
    }

    /// Apply one body hit; zero remaining hits is the terminal transition.
    pub(crate) const fn hit_body(&mut self) -> Damage {
        let Some(remaining) = self.body else {
            return Damage::Ignored;
        };
        if remaining.get() == 1 {
            self.crash()
        } else {
            self.body = NonZeroU8::new(remaining.get() - 1);
            Damage::Hit
        }
    }

    /// Advance one rotor through its two-hit sequence while the body lives.
    pub(crate) const fn hit_rotor(&mut self, motor: DroneMotor) -> Damage {
        if !self.is_alive() {
            return Damage::Ignored;
        }
        let rotor = match motor {
            DroneMotor::FrontLeft => &mut self.rotors[0],
            DroneMotor::FrontRight => &mut self.rotors[1],
            DroneMotor::RearRight => &mut self.rotors[2],
            DroneMotor::RearLeft => &mut self.rotors[3],
        };
        match rotor {
            RotorHealth::Intact => {
                *rotor = RotorHealth::Damaged;
                Damage::Hit
            }
            RotorHealth::Damaged => {
                *rotor = RotorHealth::Destroyed;
                Damage::RotorDestroyed(motor)
            }
            RotorHealth::Destroyed => Damage::Ignored,
        }
    }

    /// Fatal physical contact uses the same terminal state as body damage.
    pub(crate) const fn crash(&mut self) -> Damage {
        if self.body.take().is_some() {
            Damage::Destroyed
        } else {
            Damage::Ignored
        }
    }
}
