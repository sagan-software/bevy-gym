//! Event-driven hearing retains coarse bearings instead of hidden source coordinates.
//!
//! Local game rules: 8 m footsteps, 24 m gunshots, half range through an obstacle,
//! and two seconds of memory. This is not an acoustic propagation simulation.

use super::arena::Arena;
use bevy::math::Vec3;
use std::time::Duration;

/// Only actual movement and accepted shots may emit these events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Noise {
    /// One grounded walking stride.
    Footstep,
    /// One accepted pistol round, including a miss or wall impact.
    Gunshot,
}

impl Noise {
    /// Unobstructed hearing radius in metres, including the endpoint.
    const fn range(self) -> f32 {
        match self {
            Self::Footstep => 8.0,
            Self::Gunshot => 24.0,
        }
    }

    /// Short event names for the observer HUD and guide.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Footstep => "steps",
            Self::Gunshot => "shot",
        }
    }
}

/// World-relative horizontal sectors; north is negative Z and east is positive X.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bearing {
    /// Negative Z, within 22.5 degrees of north.
    North,
    /// Positive X and negative Z, between the cardinal sectors.
    NorthEast,
    /// Positive X, within 22.5 degrees of east.
    East,
    /// Positive X and positive Z, between the cardinal sectors.
    SouthEast,
    /// Positive Z, within 22.5 degrees of south.
    South,
    /// Negative X and positive Z, between the cardinal sectors.
    SouthWest,
    /// Negative X, within 22.5 degrees of west.
    West,
    /// Negative X and negative Z, between the cardinal sectors.
    NorthWest,
    /// Coincident horizontal coordinates provide no horizontal bearing.
    Unresolved,
}

impl Bearing {
    /// Quantize finite offsets; cardinal sectors own their exact boundaries.
    fn from_offset(offset: Vec3) -> Self {
        let x = offset.x;
        let z = offset.z;
        if x == 0.0 && z == 0.0 {
            return Self::Unresolved;
        }
        let diagonal = std::f32::consts::SQRT_2 - 1.0;
        if x.abs() <= diagonal * z.abs() {
            return if z < 0.0 { Self::North } else { Self::South };
        }
        if z.abs() <= diagonal * x.abs() {
            return if x < 0.0 { Self::West } else { Self::East };
        }
        match (x < 0.0, z < 0.0) {
            (false, true) => Self::NorthEast,
            (false, false) => Self::SouthEast,
            (true, false) => Self::SouthWest,
            (true, true) => Self::NorthWest,
        }
    }

    /// Text identifies the coarse sector without implying an exact sound position.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::North => "north",
            Self::NorthEast => "northeast",
            Self::East => "east",
            Self::SouthEast => "southeast",
            Self::South => "south",
            Self::SouthWest => "southwest",
            Self::West => "west",
            Self::NorthWest => "northwest",
            Self::Unresolved => "unresolved",
        }
    }
}

/// A finite-lived measurement with no exact emitter or listener coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Heard {
    /// Audible event class.
    noise: Noise,
    /// Quantized horizontal direction at the moment of the event.
    bearing: Bearing,
    /// Elapsed simulation time, strictly below two seconds.
    age: Duration,
}

impl Heard {
    /// Read the event class without refreshing its lifetime.
    pub(crate) const fn noise(self) -> Noise {
        self.noise
    }

    /// Read the original bearing; later listener or source motion cannot update it.
    pub(crate) const fn bearing(self) -> Bearing {
        self.bearing
    }

    /// Read elapsed simulation time since the event was heard.
    pub(crate) const fn age(self) -> Duration {
        self.age
    }
}

/// Retain the most recent audible event until it expires or the episode resets.
#[derive(Default)]
pub(crate) struct Hearing {
    /// Absence carries no stale direction or event class.
    latest: Option<Heard>,
}

impl Hearing {
    /// Read the filtered cue without looking up a hidden source.
    pub(crate) const fn latest(&self) -> Option<Heard> {
        self.latest
    }

    /// Invalid or inaudible events leave existing memory and its age unchanged.
    pub(crate) fn hear(&mut self, arena: &Arena, listener: Vec3, source: Vec3, noise: Noise) {
        // Validate ephemeral query coordinates before subtraction or squared distance.
        if [listener, source]
            .into_iter()
            .any(|point| !point.is_finite() || point.abs().max_element() > 1_000.0)
        {
            return;
        }
        let offset = source - listener;
        let mut range = noise.range();
        // One obstruction halves range; this rule does not model material or thickness.
        if arena.obstruction(listener, source).is_some() {
            range *= 0.5;
        }
        if offset.length_squared() <= range * range {
            self.latest = Some(Heard {
                noise,
                bearing: Bearing::from_offset(offset),
                age: Duration::ZERO,
            });
        }
    }

    /// Age only the recorded event; silence never refreshes it.
    pub(crate) fn advance(&mut self, elapsed: Duration) {
        if let Some(cue) = &mut self.latest {
            let age = cue.age().saturating_add(elapsed);
            if age < Duration::from_secs(2) {
                cue.age = age;
            } else {
                self.latest = None;
            }
        }
    }

    /// Clear every cue at reset, death, or sensor failure.
    pub(crate) const fn forget(&mut self) {
        self.latest = None;
    }
}
