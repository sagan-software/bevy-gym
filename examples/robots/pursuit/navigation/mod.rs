//! Plan a viewing position from static geometry and filtered sightings.
//!
//! This programmed controller chooses goals for the learned motor pilot. It cannot
//! access the character, scripted routes, or hidden target coordinates.

mod map;
mod memory;

use crate::{camera::EYE_OFFSET, flight_control::FlightGoal, sight::Contact};
use bevy::math::{Dir2, Rot2, Vec2, Vec3};
use bevy_gym::robots::DroneObservation;
use map::Map;
use memory::Memory;
use std::time::Duration;

/// Navigation advances once per five 20-millisecond physics actions.
pub(crate) const INTERVAL: Duration = Duration::from_millis(100);

/// Programmed navigation owns static geometry and bounded observation history.
pub(crate) struct Navigator {
    /// Query-only map without a moving character collider.
    map: Map,
    /// Validated sight history and bounded velocity estimates.
    memory: Memory,
    /// Last simulation time each potential target sample was checked.
    searched: Vec<Option<Duration>>,
    /// Elapsed navigation time, independent of render frames.
    elapsed: Duration,
    /// Scanning reference in world X/Z, initialized from the measured body heading.
    scan: Option<Dir2>,
    /// Previous commanded heading bounds the next reference turn.
    heading: Option<Dir2>,
}

impl Default for Navigator {
    fn default() -> Self {
        let map = Map::default();
        let searched = vec![None; map.targets.len()];
        Self {
            map,
            memory: Memory::default(),
            searched,
            elapsed: Duration::ZERO,
            scan: None,
            heading: None,
        }
    }
}

impl Navigator {
    /// Clear episode history while retaining the immutable collision graph.
    pub(crate) fn reset(&mut self) {
        self.memory = Memory::default();
        self.searched.fill(None);
        self.elapsed = Duration::ZERO;
        self.scan = None;
        self.heading = None;
    }

    /// Advance one 100-millisecond decision and return a validated flight goal.
    pub(crate) fn goal(&mut self, own: DroneObservation, contact: Contact) -> FlightGoal {
        self.elapsed = self.elapsed.saturating_add(INTERVAL);
        self.memory.advance(contact);
        let position = own.position();
        let forward = own.orientation() * Vec3::NEG_Z;
        let facing = horizontal(forward);
        let scan = *self.scan.get_or_insert(facing);
        // Use the same body-mounted eye as the rendered lens and actual sight sensor.
        self.mark_searched(position + own.orientation() * EYE_OFFSET, forward);
        let hold = Vec3::new(
            position.x.clamp(-9.5, 9.5),
            2.0,
            position.z.clamp(-9.5, 9.5),
        );
        let visible = Memory::visible(contact);
        let target = visible.or_else(|| self.search_target(position));
        let (position, heading) = if let Some(target) = target {
            let lead = self.memory.lead();
            (
                self.map.viewpoint(position, target + lead).unwrap_or(hold),
                horizontal(target - position),
            )
        } else {
            let next = Rot2::radians(0.1) * scan;
            self.scan = Some(next);
            (hold, next)
        };
        // Slew the reference by at most one radian per second, including reacquisition.
        let previous = self.heading.unwrap_or(facing);
        let turn = previous
            .perp_dot(*heading)
            .atan2(previous.dot(*heading))
            .clamp(-0.1, 0.1);
        let heading = Rot2::radians(turn) * previous;
        self.heading = Some(heading);
        FlightGoal::try_from((position, heading))
            .expect("Map cells and hold positions stay inside the flight box")
    }

    /// Record visible samples without claiming that the surrounding cell is empty.
    fn mark_searched(&mut self, eye: Vec3, forward: Vec3) {
        for (point, searched) in self.map.targets.iter().zip(&mut self.searched) {
            let delta = *point - eye;
            let distance_squared = delta.length_squared();
            let ahead = forward.dot(delta);
            if distance_squared > 0.0
                && distance_squared <= 400.0
                && ahead > 0.0
                && ahead * ahead >= 0.5 * distance_squared
                && self.map.visible(eye, *point)
            {
                *searched = Some(self.elapsed);
            }
        }
    }

    /// Inspect the nearest stale sample around the measured or predicted last sighting.
    fn search_target(&self, position: Vec3) -> Option<Vec3> {
        let centre = self.memory.centre();
        if self.elapsed < Duration::from_secs(8) && centre.is_none() {
            return None;
        }
        let centre = centre.unwrap_or(position);
        self.map
            .targets
            .iter()
            .zip(&self.searched)
            .filter(|(_, searched)| {
                searched
                    .is_none_or(|time| self.elapsed.saturating_sub(time) > Duration::from_secs(10))
            })
            .min_by(|(a, _), (b, _)| {
                a.distance_squared(centre)
                    .total_cmp(&b.distance_squared(centre))
            })
            .map(|(point, _)| *point)
    }
}

/// Project into the horizontal plane; a vertical or coincident direction faces +Z.
fn horizontal(direction: Vec3) -> Dir2 {
    Dir2::new(Vec2::new(direction.x, direction.z)).unwrap_or(Dir2::Y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::{robots::DroneHover, Env};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn heading_reference_starts_at_the_body_and_bounds_scan_and_reacquisition() {
        for seed in [0, 42, u64::MAX] {
            let own = DroneHover::disturbed().reset(Some(seed)).observation;
            let mut search = Navigator::default();
            let mut previous = horizontal(own.orientation() * Vec3::NEG_Z);
            for contact in [
                Contact::Unknown,
                Contact::Visible(Vec3::new(-9.0, 1.0, 9.0)),
                Contact::Visible(Vec3::new(9.0, 1.0, -9.0)),
                Contact::Unknown,
            ] {
                search.goal(own, contact);
                let current = search.heading.unwrap();
                let radians = previous.perp_dot(*current).atan2(previous.dot(*current));
                assert!(radians.abs() <= 0.100_001);
                previous = current;
            }
        }
        assert_eq!(horizontal(Vec3::ZERO), Dir2::Y);
        assert_eq!(horizontal(Vec3::Y), Dir2::Y);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn initial_scan_expires_and_fully_checked_samples_wait_until_stale() {
        let mut search = Navigator::default();
        let point = Vec3::new(0.0, 2.0, 0.0);
        search.elapsed = Duration::from_millis(7999);
        assert_eq!(search.search_target(point), None);
        search.elapsed = Duration::from_secs(8);
        assert!(search.search_target(point).is_some());
        search.searched.fill(Some(search.elapsed));
        search.elapsed = Duration::from_secs(18);
        assert_eq!(search.search_target(point), None);
        search.elapsed += Duration::from_millis(1);
        assert!(search.search_target(point).is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn search_rays_start_at_the_same_offset_eye_as_sight() {
        let own = DroneHover::disturbed().reset(Some(42)).observation;
        let mut search = Navigator::default();
        // A point just behind the eye is inside the body-centred cone, but outside eye sight.
        let forward = own.orientation() * Vec3::NEG_Z;
        let point = own.position() + forward * 0.1;
        search.map.targets = vec![point];
        search.searched = vec![None];
        search.goal(own, Contact::Unknown);
        assert_eq!(search.searched, vec![None]);
        search.reset();
        search.elapsed = INTERVAL;
        search.mark_searched(own.position(), forward);
        assert_eq!(search.searched, vec![Some(INTERVAL)]);
    }
}
