//! Read-only snapshot of the complete articulated droid.

use super::{DroidBody, DroidBodyState};

/// Thirteen physical segments from one simulation boundary.
///
/// Relative joint rotations and velocities can be derived from linked segments.
/// This snapshot exposes no solver mutation or task reward information.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroidObservation {
    /// Segment snapshots in `DroidBody::ALL` order.
    pub(super) bodies: [DroidBodyState; 13],
}
impl DroidObservation {
    /// Read shared fall conditions; callers separately validate finiteness and lesson bounds.
    /// Pelvis height below 0.5 m or non-foot floor contact is fatal; foot support is permitted.
    pub(in crate::robots) fn fatal_contact(&self) -> bool {
        self.body(DroidBody::Pelvis).position().y < 0.5
            || DroidBody::ALL.into_iter().any(|identity| {
                self.body(identity).floor_contact()
                    && !matches!(identity, DroidBody::LeftFoot | DroidBody::RightFoot)
            })
    }

    /// Return one segment's physical state.
    ///
    /// # Panics
    /// Panics only if the internal array and closed identity enum disagree.
    /// Public construction preserves their matching lengths.
    #[must_use]
    pub fn body(&self, body: DroidBody) -> DroidBodyState {
        *self.bodies.get(body as usize).expect("closed body index")
    }
}
