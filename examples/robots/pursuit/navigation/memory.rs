//! Retain bounded observations without reading hidden target coordinates.

use super::INTERVAL;
use crate::sight::Contact;
use bevy::math::Vec3;
use std::time::Duration;

/// Investigate a last-seen point for at most ten seconds.
const RETENTION: Duration = Duration::from_secs(10);

/// Each state carries only information justified by earlier sightings.
#[derive(Default)]
pub(super) enum Memory {
    /// No investigation has started; a later remembered sighting may seed one.
    #[default]
    Fresh,
    /// Earlier memory expired; wait for a cleared sensor or a current sighting.
    Expired,
    /// Consecutive visible samples support a measured horizontal velocity.
    Visible {
        /// Latest finite world position in metres.
        point: Vec3,
        /// Horizontal metres per second, capped at the character's speed.
        velocity: Vec3,
    },
    /// Hidden movement cannot change the measured point or velocity.
    Remembered {
        /// Last finite measured world position in metres.
        point: Vec3,
        /// Velocity from the last pair of consecutive sightings.
        velocity: Vec3,
        /// Remaining investigation time, never refreshed by remembered inputs.
        remaining: Duration,
    },
}

impl Memory {
    /// Validate contact coordinates before differences or distance calculations.
    pub(super) fn visible(contact: Contact) -> Option<Vec3> {
        match contact {
            Contact::Visible(point) if valid(point) => Some(point),
            Contact::Unknown | Contact::Visible(_) | Contact::Remembered { .. } => None,
        }
    }

    /// Accept one sample; only a current sighting refreshes existing memory.
    pub(super) fn advance(&mut self, contact: Contact) {
        if let Some(point) = Self::visible(contact) {
            let velocity = match *self {
                Self::Visible {
                    point: previous, ..
                } => {
                    let delta = (point - previous) / INTERVAL.as_secs_f32();
                    Vec3::new(delta.x, 0.0, delta.z).clamp_length_max(4.0)
                }
                Self::Fresh | Self::Expired | Self::Remembered { .. } => Vec3::ZERO,
            };
            *self = Self::Visible { point, velocity };
            return;
        }
        // An initial remembered input already consumed part of its retention period.
        if matches!(self, Self::Fresh) {
            *self = Self::initial(contact);
            return;
        }
        // Expire from elapsed simulation time even if the caller repeats an old contact.
        *self = match *self {
            Self::Visible { point, velocity } => Self::Remembered {
                point,
                velocity,
                remaining: RETENTION.saturating_sub(INTERVAL),
            },
            Self::Remembered {
                point,
                velocity,
                remaining,
            } if remaining > INTERVAL => Self::Remembered {
                point,
                velocity,
                remaining: remaining.saturating_sub(INTERVAL),
            },
            Self::Fresh | Self::Expired | Self::Remembered { .. } => {
                // Cleared sensor memory ends the old contact sequence. A later short
                // sighting may arrive as Remembered between navigation decisions.
                if matches!(contact, Contact::Unknown) {
                    Self::Fresh
                } else {
                    Self::Expired
                }
            }
        };
    }

    /// Seed a new contact sequence only from a finite, unexpired sensor memory.
    fn initial(contact: Contact) -> Self {
        match contact {
            Contact::Remembered { point, age } if valid(point) && age < Duration::from_secs(3) => {
                Self::Remembered {
                    point,
                    velocity: Vec3::ZERO,
                    remaining: RETENTION.saturating_sub(age),
                }
            }
            Contact::Unknown | Contact::Visible(_) | Contact::Remembered { .. } => Self::Fresh,
        }
    }

    /// Predict for at most one second from measured motion while investigating.
    pub(super) fn centre(&self) -> Option<Vec3> {
        match *self {
            Self::Fresh | Self::Expired => None,
            Self::Visible { point, .. } => Some(point),
            Self::Remembered {
                point,
                velocity,
                remaining,
            } => {
                let seconds = RETENTION.saturating_sub(remaining).as_secs_f32().min(1.0);
                Some(point + velocity * seconds)
            }
        }
    }

    /// A visible target's half-second lead helps the pilot track lateral movement.
    pub(super) fn lead(&self) -> Vec3 {
        match *self {
            Self::Visible { velocity, .. } => velocity * 0.5,
            Self::Fresh | Self::Expired | Self::Remembered { .. } => Vec3::ZERO,
        }
    }
}

/// Match the sight sensor's bounded world-coordinate profile.
fn valid(point: Vec3) -> bool {
    point.is_finite() && point.abs().max_element() <= 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn a_repeated_remembered_contact_cannot_extend_investigation() {
        let mut memory = Memory::default();
        let contact = Contact::Remembered {
            point: Vec3::X,
            age: Duration::from_secs(1),
        };
        memory.advance(contact);
        for _ in 0..89 {
            memory.advance(contact);
            assert_eq!(memory.centre(), Some(Vec3::X));
        }
        memory.advance(contact);
        assert_eq!(memory.centre(), None);
        memory.advance(contact);
        assert_eq!(memory.centre(), None);
        assert_eq!(memory.lead(), Vec3::ZERO);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn cleared_sensor_allows_a_new_brief_sighting_after_expiry() {
        let mut memory = Memory::default();
        let old = Contact::Remembered {
            point: Vec3::X,
            age: Duration::from_secs(1),
        };
        memory.advance(old);
        for _ in 0..90 {
            memory.advance(old);
        }
        assert_eq!(memory.centre(), None);
        memory.advance(old);
        assert_eq!(memory.centre(), None);
        memory.advance(Contact::Unknown);
        memory.advance(Contact::Remembered {
            point: Vec3::Z,
            age: Duration::from_millis(20),
        });
        assert_eq!(memory.centre(), Some(Vec3::Z));
        assert_eq!(memory.lead(), Vec3::ZERO);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn hidden_frames_break_velocity_measurement_and_expire_at_ten_seconds() {
        let mut memory = Memory::default();
        memory.advance(Contact::Visible(Vec3::ZERO));
        assert_eq!(memory.centre(), Some(Vec3::ZERO));
        assert_eq!(memory.lead(), Vec3::ZERO);
        memory.advance(Contact::Visible(Vec3::new(1.0, 2.0, 0.0)));
        assert_eq!(memory.lead(), Vec3::X * 2.0);
        memory.advance(Contact::Unknown);
        assert!(memory
            .centre()
            .unwrap()
            .abs_diff_eq(Vec3::new(1.4, 2.0, 0.0), 1e-5));
        assert_eq!(memory.lead(), Vec3::ZERO);
        for _ in 0..10 {
            memory.advance(Contact::Unknown);
        }
        assert_eq!(memory.centre(), Some(Vec3::new(5.0, 2.0, 0.0)));
        for _ in 0..88 {
            memory.advance(Contact::Unknown);
        }
        assert!(memory.centre().is_some());
        memory.advance(Contact::Unknown);
        assert_eq!(memory.centre(), None);
        memory.advance(Contact::Visible(Vec3::X));
        assert_eq!(memory.lead(), Vec3::ZERO);
        memory.advance(Contact::Unknown);
        memory.advance(Contact::Visible(Vec3::new(9.0, 0.0, 0.0)));
        assert_eq!(memory.lead(), Vec3::ZERO);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn malformed_coordinates_and_expired_sensor_memory_cannot_seed_a_search() {
        for axis in 0..3 {
            for value in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1_000.1,
                -1_000.1,
            ] {
                let mut point = Vec3::ZERO;
                point[axis] = value;
                for contact in [
                    Contact::Visible(point),
                    Contact::Remembered {
                        point,
                        age: Duration::ZERO,
                    },
                ] {
                    let mut memory = Memory::default();
                    assert_eq!(Memory::visible(contact), None);
                    memory.advance(contact);
                    assert_eq!(memory.centre(), None);
                }
            }
        }
        for age in [Duration::from_secs(3), Duration::MAX] {
            let mut memory = Memory::default();
            memory.advance(Contact::Remembered {
                point: Vec3::X,
                age,
            });
            assert_eq!(memory.centre(), None);
        }
        for point in [Vec3::splat(-1_000.0), Vec3::splat(1_000.0)] {
            let mut memory = Memory::default();
            memory.advance(Contact::Visible(point));
            assert_eq!(memory.centre(), Some(point));
        }
    }
}
