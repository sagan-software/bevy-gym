//! Bounded geometric sight and last-seen memory for the pursuit game.
//!
//! The local camera profile uses a 90-degree cone and 20-metre range. Occlusion
//! uses the arena's Rapier 0.36 solid rays, including hits at the target endpoint:
//! <https://rapier.rs/docs/user_guides/rust/scene_queries/>.

use super::arena::Arena;
use bevy::math::{Dir3, Vec3};
use std::time::Duration;

/// Information available to an actor after geometry has filtered world state.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) enum Contact {
    /// No current or sufficiently recent sighting exists.
    #[default]
    Unknown,
    /// One exposed body point measured during the current sample, in world metres.
    Visible(Vec3),
    /// An earlier measured point; current target coordinates are unavailable.
    Remembered {
        /// Last measured world position in metres, unchanged while hidden.
        point: Vec3,
        /// Simulation time since the last successful sample, below three seconds.
        age: Duration,
    },
}

/// Retain one observation without retaining privileged character coordinates.
#[derive(Default, Clone)]
pub(crate) struct Sight {
    /// Legal states carry only their own measured position and memory age.
    contact: Contact,
}

impl Sight {
    /// Query up to three body points, then age memory by elapsed simulation time.
    pub(crate) fn sample(
        &mut self,
        arena: &Arena,
        origin: Vec3,
        forward: Dir3,
        character: Vec3,
        elapsed: Duration,
    ) -> Contact {
        if let Some(point) = visible_point(arena, origin, forward, character) {
            self.contact = Contact::Visible(point);
            return self.contact;
        }
        // Hidden movement cannot replace a measured point or infer target velocity.
        let (point, age) = match self.contact {
            Contact::Unknown => return Contact::Unknown,
            Contact::Visible(point) => (point, Duration::ZERO),
            Contact::Remembered { point, age } => (point, age),
        };
        let age = age.saturating_add(elapsed);
        self.contact = if age < Duration::from_secs(3) {
            Contact::Remembered { point, age }
        } else {
            Contact::Unknown
        };
        self.contact
    }

    /// Clear episode memory without changing the arena or character.
    pub(crate) const fn forget(&mut self) {
        self.contact = Contact::Unknown;
    }
}

/// Return the first exposed head, chest, or hip proxy; allocate no sample buffer.
fn visible_point(arena: &Arena, origin: Vec3, forward: Dir3, character: Vec3) -> Option<Vec3> {
    // Reject malformed centres before subtraction, squared distance, or ray queries.
    if !origin.is_finite()
        || !character.is_finite()
        || origin.abs().max_element() > 1_000.0
        || character.abs().max_element() > 1_000.0
    {
        return None;
    }
    [0.6, 0.2, -0.3]
        .into_iter()
        .map(|height| character + Vec3::Y * height)
        .find(|&point| {
            let offset = point - origin;
            let squared_metres = offset.length_squared();
            let forward_metres = offset.dot(*forward);
            // Range is 20 m inclusive. cos²(45°)=0.5 defines the cone boundary.
            // Coincident points have no view direction and cannot reveal a target.
            // Extend rays 1 mm past the sample to close rounding gaps on wall faces.
            squared_metres > 0.0
                && squared_metres <= 20.0 * 20.0
                && forward_metres > 0.0
                && forward_metres * forward_metres >= 0.5 * squared_metres
                && arena
                    .obstruction(origin, point + offset / squared_metres.sqrt() * 0.001)
                    .is_none()
        })
}
