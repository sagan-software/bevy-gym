//! Pickup ownership, bounded ammunition, and elapsed-time firing cooldown.

use bevy::math::Vec3;
use std::time::Duration;

/// Ownership of the arena's one pistol; reset restores an available pickup.
#[derive(Debug, Default)]
pub(crate) struct Pistol {
    /// Absence means unarmed and the pickup is still available.
    owned: Option<LoadedPistol>,
}

/// A pistol can only be constructed with the fixed magazine capacity.
#[derive(Debug)]
struct LoadedPistol {
    /// Remaining rounds, decremented only after all firing checks pass.
    magazine: Magazine,
    /// Time until firing is permitted again.
    cooldown: Duration,
}

/// A private bounded count: construction starts at twelve; firing only subtracts.
#[derive(Debug)]
struct Magazine(u8);

/// Pickup failure does not change ownership or refill a magazine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PickupError {
    /// Coordinates or the computed distance are not finite.
    InvalidPosition,
    /// Character centre is more than 1.5 metres from the pickup.
    TooFar,
    /// The arena's only pistol has already been taken.
    AlreadyOwned,
}

/// A rejected firing attempt never consumes ammunition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FireError {
    /// The actor has not collected the pistol.
    Unarmed,
    /// No rounds remain; reset is currently the only resupply.
    Empty,
    /// The preceding shot's 250-millisecond cooldown has not expired.
    CoolingDown,
}

impl Pistol {
    /// Fixed pickup centre in metres, beside the spawn and clear of cover.
    pub(crate) const LOCATION: Vec3 = Vec3::new(-1.0, 0.92, 8.0);

    /// Ownership also determines whether the pickup remains in the scene.
    pub(crate) const fn is_armed(&self) -> bool {
        self.owned.is_some()
    }

    /// Read ammunition for the viewer without allowing arbitrary refill.
    pub(crate) fn rounds(&self) -> u8 {
        self.owned.as_ref().map_or(0, |pistol| pistol.magazine.0)
    }

    /// Validate coordinates before checking ownership and inclusive pickup range.
    pub(crate) fn pick_up(&mut self, character: Vec3) -> Result<(), PickupError> {
        let distance = character.distance(Self::LOCATION);
        if !character.is_finite() || !distance.is_finite() {
            return Err(PickupError::InvalidPosition);
        }
        if self.owned.is_some() {
            return Err(PickupError::AlreadyOwned);
        }
        if distance > 1.5 {
            return Err(PickupError::TooFar);
        }
        self.owned = Some(LoadedPistol {
            magazine: Magazine(12),
            cooldown: Duration::ZERO,
        });
        Ok(())
    }

    /// Consume exactly one round after ownership, ammunition, and cooldown checks.
    pub(crate) fn fire(&mut self) -> Result<(), FireError> {
        let pistol = self.owned.as_mut().ok_or(FireError::Unarmed)?;
        if pistol.magazine.0 == 0 {
            return Err(FireError::Empty);
        }
        if !pistol.cooldown.is_zero() {
            return Err(FireError::CoolingDown);
        }
        pistol.magazine.0 -= 1;
        pistol.cooldown = Duration::from_millis(250);
        Ok(())
    }

    /// Recover the visual kick during the first 120 milliseconds of the firing cooldown.
    pub(crate) fn recoil(&self) -> f32 {
        self.owned.as_ref().map_or(0.0, |pistol| {
            pistol
                .cooldown
                .saturating_sub(Duration::from_millis(130))
                .as_secs_f32()
                / 0.12
        })
    }

    /// Advance simulation time; excess elapsed time cannot underflow the cooldown.
    pub(crate) const fn advance(&mut self, elapsed: Duration) {
        if let Some(pistol) = &mut self.owned {
            pistol.cooldown = pistol.cooldown.saturating_sub(elapsed);
        }
    }
}
