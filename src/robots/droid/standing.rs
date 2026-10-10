//! Standing episode rules over a private articulated physics world.

use super::{
    physics::{Articulation, PHYSICS_INTERVAL},
    DroidAction, DroidBody, DroidObservation,
};
use crate::{training::SplitMix64, Env, EpisodeStatus, Reset, Step};
use bevy::math::Vec3;
use rapier3d::prelude::{Rotation, Vector};
use std::{fmt, time::Duration};

/// Balance thirteen dynamic segments through twenty-six direct torque actuators.
///
/// Actions last 20 ms, with four 5 ms physics steps. Non-foot floor contact,
/// pelvis height below 0.5 m, or any segment outside x/z ±5 m and y [0, 3] m
/// terminates the episode. Non-finite body states also terminate. Terminal
/// observations freeze and earn zero until reset. Use [`crate::TimeLimit`] for a horizon.
///
/// Reward is torso uprightness times pelvis height fraction, divided by
/// `1 + horizontal_distance_squared / (1 m²) + pelvis_speed_squared / (1 m²/s²)`.
/// Uprightness is `(1 + torso_up.y) / 2` clamped to [0, 1]. Height fraction is
/// pelvis height / 0.99 m clamped to [0, 1]. Horizontal distance is from the origin.
/// These are environment measurements, never action selectors.
///
/// Reset samples common yaw ±pi radians and pitch/roll ±0.01 radians, then lifts
/// the bind pose by 0.03 m. All initial velocities are zero. Explicit seeds restart
/// the reset stream; omitted seeds continue it. The world is rebuilt on reset.
///
/// ```compile_fail
/// use bevy_gym::robots::DroidStanding;
/// let world = DroidStanding::default().articulation;
/// ```
pub struct DroidStanding {
    /// Private physical tree, rebuilt on reset.
    articulation: Articulation,
    /// Seeded initial-pose stream, unaffected by stepping.
    random: SplitMix64,
    /// Mutually exclusive episode lifecycle states.
    state: State,
}

/// Standing actions advance physics; fallen actions preserve the terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// The physical task still accepts actions.
    Standing,
    /// A physical termination condition has been observed.
    Fallen,
}

impl DroidStanding {
    /// Time held by one policy action, unless a terminal substep stops it early.
    pub const POLICY_INTERVAL: Duration = Duration::from_millis(20);

    /// Rebuild a randomized, mechanically connected initial pose.
    fn from_random(mut random: SplitMix64) -> Self {
        let yaw = random.f64_between(-std::f64::consts::PI, std::f64::consts::PI) as f32;
        let pitch = random.f64_between(-0.01, 0.01) as f32;
        let roll = random.f64_between(-0.01, 0.01) as f32;
        let rotation = Rotation::from_rotation_y(yaw)
            * Rotation::from_rotation_x(pitch)
            * Rotation::from_rotation_z(roll);
        Self {
            articulation: Articulation::new(rotation, Vector::Y * 0.03),
            random,
            state: State::Standing,
        }
    }

    /// Reject invalid physical states and standing-task failures at each substep.
    fn fallen(observation: &DroidObservation) -> bool {
        observation.body(DroidBody::Pelvis).position().y < 0.5
            || DroidBody::ALL.into_iter().any(|identity| {
                let body = observation.body(identity);
                let position = body.position();
                !position.is_finite()
                    || !body.orientation().is_finite()
                    || !body.linear_velocity().is_finite()
                    || !body.angular_velocity().is_finite()
                    || position.x.abs() > 5.0
                    || position.z.abs() > 5.0
                    || !(0.0..=3.0).contains(&position.y)
                    || (body.floor_contact()
                        && !matches!(identity, DroidBody::LeftFoot | DroidBody::RightFoot))
            })
    }

    /// Measure uprightness, standing height, horizontal drift and physical speed.
    fn reward(observation: &DroidObservation) -> f64 {
        let pelvis = observation.body(DroidBody::Pelvis);
        let position = pelvis.position();
        let upright = (observation.body(DroidBody::Torso).orientation() * Vec3::Y)
            .y
            .midpoint(1.0)
            .clamp(0.0, 1.0);
        let height = (position.y / 0.99).clamp(0.0, 1.0);
        // Both squared quantities are divided by their respective unit-square scales.
        let denominator = position
            .z
            .mul_add(position.z, position.x.mul_add(position.x, 1.0))
            + pelvis.linear_velocity().length_squared();
        f64::from(upright * height / denominator)
    }
}

impl Default for DroidStanding {
    fn default() -> Self {
        Self::from_random(SplitMix64::new(0))
    }
}
impl fmt::Debug for DroidStanding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DroidStanding")
            .field("observation", &self.articulation.observation())
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}
impl Env for DroidStanding {
    type Observation = DroidObservation;
    type Action = DroidAction;
    type Info = ();
    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        *self = Self::from_random(seed.map_or(self.random, SplitMix64::new));
        Reset {
            observation: self.articulation.observation(),
            info: (),
        }
    }
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        if self.state == State::Standing {
            for _ in 0..(Self::POLICY_INTERVAL.as_nanos() / PHYSICS_INTERVAL.as_nanos()) {
                self.articulation.apply(action);
                self.articulation.world.step();
                // Freeze the first failed substep; later commands cannot move a fallen body.
                if Self::fallen(&self.articulation.observation()) {
                    self.state = State::Fallen;
                    break;
                }
            }
        }
        let observation = self.articulation.observation();
        let (status, reward) = match self.state {
            State::Standing => (EpisodeStatus::Continuing, Self::reward(&observation)),
            State::Fallen => (EpisodeStatus::Terminated, 0.0),
        };
        Step {
            observation,
            reward,
            status,
            info: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Quat;

    /// Each spatial and non-finite boundary independently ends the task.
    #[test]
    fn physical_failure_boundaries() {
        let initial = DroidStanding::default().articulation.observation();
        assert!(!DroidStanding::fallen(&initial));
        for identity in DroidBody::ALL {
            for position in [
                Vec3::new(-5.0, 1.0, 0.0),
                Vec3::new(5.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, -5.0),
                Vec3::new(0.0, 1.0, 5.0),
                Vec3::new(0.0, 3.0, 0.0),
            ] {
                let mut observation = initial;
                body_mut(&mut observation, identity).position = position;
                assert!(!DroidStanding::fallen(&observation));
            }
            for position in [
                Vec3::new(-5.001, 1.0, 0.0),
                Vec3::new(5.001, 1.0, 0.0),
                Vec3::new(0.0, 1.0, -5.001),
                Vec3::new(0.0, 1.0, 5.001),
                Vec3::new(0.0, -0.001, 0.0),
                Vec3::new(0.0, 3.001, 0.0),
                Vec3::splat(f32::NAN),
                Vec3::splat(f32::INFINITY),
                Vec3::splat(f32::NEG_INFINITY),
            ] {
                let mut observation = initial;
                body_mut(&mut observation, identity).position = position;
                assert!(DroidStanding::fallen(&observation));
            }
        }
        for (height, fallen) in [(0.499, true), (0.5, false)] {
            let mut observation = initial;
            body_mut(&mut observation, DroidBody::Pelvis).position.y = height;
            assert_eq!(DroidStanding::fallen(&observation), fallen);
        }
        let mut observation = initial;
        body_mut(&mut observation, DroidBody::Head).position.y = 0.0;
        assert!(!DroidStanding::fallen(&observation));
    }

    /// Return one mutable segment only inside isolated boundary tests.
    fn body_mut(
        observation: &mut DroidObservation,
        identity: DroidBody,
    ) -> &mut super::super::DroidBodyState {
        observation
            .bodies
            .get_mut(identity as usize)
            .expect("segment")
    }

    /// Non-finite motion and forbidden floor contacts terminate independently.
    #[test]
    fn motion_and_contact_failure_boundaries() {
        let initial = DroidStanding::default().articulation.observation();
        for identity in DroidBody::ALL {
            for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut observation = initial;
                body_mut(&mut observation, identity).orientation = Quat::from_array([invalid; 4]);
                assert!(DroidStanding::fallen(&observation));
                let mut observation = initial;
                body_mut(&mut observation, identity).linear_velocity = Vec3::splat(invalid);
                assert!(DroidStanding::fallen(&observation));
                let mut observation = initial;
                body_mut(&mut observation, identity).angular_velocity = Vec3::splat(invalid);
                assert!(DroidStanding::fallen(&observation));
            }
            let mut observation = initial;
            body_mut(&mut observation, identity).floor_contact = true;
            assert_eq!(
                DroidStanding::fallen(&observation),
                !matches!(identity, DroidBody::LeftFoot | DroidBody::RightFoot)
            );
        }
    }

    /// Known physical states independently establish every reward term and clamp.
    #[test]
    fn reward_terms_have_explicit_units_and_bounds() {
        let mut observation = DroidStanding::default().articulation.observation();
        body_mut(&mut observation, DroidBody::Torso).orientation = Quat::IDENTITY;
        body_mut(&mut observation, DroidBody::Pelvis).position = Vec3::new(0.0, 0.99, 0.0);
        assert!((DroidStanding::reward(&observation) - 1.0).abs() < 1e-6);
        body_mut(&mut observation, DroidBody::Pelvis).position = Vec3::new(1.0, 0.495, 1.0);
        body_mut(&mut observation, DroidBody::Pelvis).linear_velocity = Vec3::X;
        assert!((DroidStanding::reward(&observation) - 0.125).abs() < 1e-6);
        body_mut(&mut observation, DroidBody::Pelvis).position = Vec3::new(0.0, 2.0, 0.0);
        body_mut(&mut observation, DroidBody::Pelvis).linear_velocity = Vec3::ZERO;
        assert!((DroidStanding::reward(&observation) - 1.0).abs() < 1e-6);
        body_mut(&mut observation, DroidBody::Pelvis).position.y = -1.0;
        assert!(DroidStanding::reward(&observation).abs() < 1e-6);
        body_mut(&mut observation, DroidBody::Pelvis).position.y = 0.99;
        body_mut(&mut observation, DroidBody::Torso).orientation =
            Quat::from_rotation_x(std::f32::consts::PI);
        assert!(DroidStanding::reward(&observation).abs() < 1e-6);
    }
}
