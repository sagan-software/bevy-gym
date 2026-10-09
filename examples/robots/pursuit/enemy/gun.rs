//! A current visible contact must survive the warning before the drone can fire.

use super::projectile::Projectile;
use crate::{arena::Arena, sight::Contact};
use bevy::math::{Dir3, Vec3};
use std::time::Duration;

/// Read-only weapon state; callers cannot replace the gun's phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// No current firing opportunity has started a warning.
    Idle,
    /// Continuous valid sight is required until this warning finishes.
    Charging {
        /// Remaining warning time.
        remaining: Duration,
    },
    /// The first shot has fired; later rounds still require current sight.
    Volley {
        /// Delay until the next round.
        remaining: Duration,
        /// Remaining rounds, never zero.
        rounds: Rounds,
    },
    /// Recovery runs even when the target leaves view.
    Cooling {
        /// Time before another warning can begin.
        remaining: Duration,
    },
    /// Death or controller failure prevents reactivation until reset.
    Disabled,
}

/// Only one or two rounds can remain after a volley starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rounds {
    /// Both follow-up rounds remain.
    Two,
    /// Only the final round remains.
    One,
}

/// The weapon retains timing only; aim is derived from the current observation.
pub(crate) struct Gun {
    /// Authoritative state of the firing sequence.
    phase: Phase,
}

impl Default for Gun {
    fn default() -> Self {
        Self { phase: Phase::Idle }
    }
}

impl Gun {
    /// Observe timing without permitting an early discharge.
    pub(crate) const fn phase(&self) -> Phase {
        self.phase
    }

    /// Advance exactly twenty milliseconds; return at most one newly fired projectile.
    pub(crate) fn advance(
        &mut self,
        arena: &Arena,
        origin: Vec3,
        contact: Contact,
    ) -> Option<Projectile> {
        let step = Duration::from_millis(20);
        match self.phase {
            Phase::Disabled => None,
            Phase::Cooling { remaining } => {
                // Recovery runs independently of target visibility.
                let remaining = remaining.saturating_sub(step);
                self.phase = if remaining.is_zero() {
                    Phase::Idle
                } else {
                    Phase::Cooling { remaining }
                };
                None
            }
            Phase::Idle => {
                if direction(arena, origin, contact).is_some() {
                    self.phase = Phase::Charging {
                        remaining: Duration::from_millis(800),
                    };
                }
                None
            }
            Phase::Charging { remaining } => {
                self.charge(remaining, direction(arena, origin, contact), origin)
            }
            Phase::Volley { remaining, rounds } => {
                let Some(direction) = direction(arena, origin, contact) else {
                    // Cancelling a volley pays the same recovery time as completing it.
                    self.phase = cooling();
                    return None;
                };
                let remaining = remaining.saturating_sub(step);
                if remaining.is_zero() {
                    self.phase = match rounds {
                        Rounds::Two => Phase::Volley {
                            remaining: Duration::from_millis(120),
                            rounds: Rounds::One,
                        },
                        Rounds::One => cooling(),
                    };
                    Some(Projectile::new(origin, direction))
                } else {
                    self.phase = Phase::Volley { remaining, rounds };
                    None
                }
            }
        }
    }

    /// A lost opportunity discards all warning progress; a completed warning starts the volley.
    const fn charge(
        &mut self,
        remaining: Duration,
        direction: Option<Dir3>,
        origin: Vec3,
    ) -> Option<Projectile> {
        let Some(direction) = direction else {
            self.phase = Phase::Idle;
            return None;
        };
        let remaining = remaining.saturating_sub(Duration::from_millis(20));
        if remaining.is_zero() {
            self.phase = Phase::Volley {
                remaining: Duration::from_millis(120),
                rounds: Rounds::Two,
            };
            Some(Projectile::new(origin, direction))
        } else {
            self.phase = Phase::Charging { remaining };
            None
        }
    }

    /// Disable future shots while leaving already fired projectiles to their owner.
    pub(crate) const fn disable(&mut self) {
        self.phase = Phase::Disabled;
    }
}

/// Recovery always lasts 1,200 milliseconds, including a cancelled volley.
const fn cooling() -> Phase {
    Phase::Cooling {
        remaining: Duration::from_millis(1200),
    }
}

/// Validate only a current measurement; remembered or unknown contact cannot authorize fire.
fn direction(arena: &Arena, origin: Vec3, contact: Contact) -> Option<Dir3> {
    let Contact::Visible(point) = contact else {
        return None;
    };
    if [origin, point]
        .into_iter()
        .any(|value| !value.is_finite() || value.abs().max_element() > 1000.0)
    {
        return None;
    }
    let offset = point - origin;
    // A normalized ray reports distances in metres. Limit muzzle range to twenty metres.
    if offset.length_squared() > 20.0 * 20.0 {
        return None;
    }
    let direction = Dir3::new(offset).ok()?;
    arena
        .obstruction(origin, point)
        .is_none()
        .then_some(direction)
}
