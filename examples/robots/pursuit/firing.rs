//! Player and future policy actions share pickup, firing, and feedback state.

use super::combat::health::Damage;
use super::{
    arena::Arena,
    combat::pistol::{FireError, PickupError, Pistol},
    shot::{
        aim::Aim,
        target::{Shot, Target},
    },
};
use bevy::math::{Dir3, Quat, Vec3};
use bevy_gym::robots::DroneMotor;
use std::{collections::VecDeque, time::Duration};

/// One interaction at the fixed action boundary; callers cannot choose a shot origin.
#[derive(Debug, Clone, Copy)]
pub(super) enum Action {
    /// Attempt to collect the arena's pistol at the current character position.
    PickUp,
    /// Fire along a validated direction from the character's chest.
    Fire(Dir3),
}

/// One-shot presentation events; health remains the authoritative source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Effect {
    /// Remove all transients before rendering the new episode.
    Reset,
    /// The first hit starts smoke at this live rotor.
    RotorDamaged(DroneMotor),
    /// A named rotor reached its destroyed state.
    Rotor(DroneMotor),
    /// The body reached its absorbing dead state.
    Destroyed,
}

/// Last meaningful interaction result displayed by the viewer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Feedback {
    /// Flight contact or a flight-region exit ended the drone's episode.
    Crashed,
    /// Initial state before the pistol is collected.
    Unarmed,
    /// Pickup succeeded and the magazine is full.
    Armed,
    /// Pickup failed without changing ownership.
    Pickup(PickupError),
    /// Firing failed without spending ammunition.
    Rejected(FireError),
    /// A projectile is travelling; damage has not been resolved yet.
    Launched,
    /// An accepted shot hit a target, wall, or nothing.
    Fired(Shot),
}

/// One short-lived trace, replaced by the next accepted shot.
#[derive(Clone, Copy)]
pub(super) struct Trace {
    /// Authoritative world-space shot origin.
    pub(super) from: Vec3,
    /// First impact or maximum-range endpoint.
    pub(super) to: Vec3,
    /// Remaining presentation time; independent of weapon cooldown.
    remaining: Duration,
}

/// One finite-speed shot with a derived current position and bounded travel distance.
struct Projectile {
    /// Validated launch point and constant travel direction.
    aim: Aim,
    /// Distance already swept in metres, never greater than the weapon range.
    travelled: f32,
}

/// A resolved projectile hit supplies the physical response at the same simulation boundary.
pub(super) struct Impact {
    /// First world-space contact point in metres.
    pub(super) point: Vec3,
    /// World-space travel direction used for the impulse.
    pub(super) direction: Dir3,
}

/// Damageable target, pickup ownership, and feedback shared by player actions.
pub(super) struct Combat {
    /// The sole pickup and its bounded magazine.
    pistol: Pistol,
    /// Damageable drone target with private shot hitboxes.
    target: Target,
    /// Last interaction result for UI copy.
    feedback: Feedback,
    /// Bounded visual trace; absent once its lifetime expires.
    trace: Option<Trace>,
    /// Current projectile; its 250 ms range lifetime cannot outlast the firing cooldown.
    projectile: Option<Projectile>,
    /// At most one reset, four first hits, four rotor failures, and one body death.
    effects: VecDeque<Effect>,
}

impl Default for Combat {
    fn default() -> Self {
        Self {
            pistol: Pistol::default(),
            target: Target::try_from((Vec3::new(0.0, 2.0, 1.0), Quat::IDENTITY))
                .expect("Authored finite target pose"),
            feedback: Feedback::Unarmed,
            trace: None,
            projectile: None,
            effects: VecDeque::from([Effect::Reset]),
        }
    }
}

impl Combat {
    /// Update collision and aiming geometry without repairing existing damage.
    pub(super) fn project_flight(
        &mut self,
        observation: bevy_gym::robots::DroneObservation,
    ) -> Result<(), super::shot::target::InvalidTarget> {
        self.target
            .move_to(observation.position(), observation.orientation())
    }

    /// Emit one death event when physical flight termination destroys a living body.
    pub(super) fn crash(&mut self) {
        if self.target.crash() == Damage::Destroyed {
            self.effects.push_back(Effect::Destroyed);
            self.feedback = Feedback::Crashed;
        }
    }

    /// Consume one presentation event without allowing presentation to change health.
    pub(super) fn pop_effect(&mut self) -> Option<Effect> {
        self.effects.pop_front()
    }

    /// A chest-origin ray cannot start beyond a wall penetrated by the displayed barrel.
    pub(super) fn origin(character: Vec3) -> Vec3 {
        character + Vec3::Y * 0.4
    }

    /// Read ownership for pickup and held-pistol visibility.
    pub(super) const fn is_armed(&self) -> bool {
        self.pistol.is_armed()
    }

    /// Read the bounded magazine for the HUD.
    pub(super) fn rounds(&self) -> u8 {
        self.pistol.rounds()
    }

    /// The bounded weapon cooldown also controls its visual recoil envelope.
    pub(super) fn recoil(&self) -> f32 {
        self.pistol.recoil()
    }

    /// Borrow the target for camera queries and rendering.
    pub(super) const fn target(&self) -> &Target {
        &self.target
    }

    /// Read the last meaningful interaction result.
    pub(super) const fn feedback(&self) -> Feedback {
        self.feedback
    }

    /// Read the active trace without allowing presentation to alter its lifetime.
    pub(super) const fn trace(&self) -> Option<Trace> {
        self.trace
    }

    /// Advance a finite projectile before the next shot may replace it.
    pub(super) fn advance(&mut self, elapsed: Duration, arena: &Arena) -> Option<Impact> {
        self.pistol.advance(elapsed);
        if let Some(trace) = &mut self.trace {
            trace.remaining = trace.remaining.saturating_sub(elapsed);
            if trace.remaining.is_zero() {
                self.trace = None;
            }
        }
        if elapsed.is_zero() {
            return None;
        }
        let projectile = self.projectile.as_mut()?;
        let from = projectile.aim.point(projectile.travelled);
        // Metres per second times seconds gives segment length in metres.
        let distance = (120.0 * elapsed.as_secs_f32()).min(Aim::RANGE - projectile.travelled);
        let aim = Aim::try_from((from, *projectile.aim.direction()))
            .expect("Owned projectile remains within the arena query bounds");
        let shot = self.target.sweep(arena, aim, distance);
        projectile.travelled += distance;
        let shot = shot.or_else(|| {
            (projectile.travelled >= Aim::RANGE).then_some(Shot::Miss {
                point: projectile.aim.point(Aim::RANGE),
            })
        });
        let to = shot.map_or_else(
            || aim.point(distance),
            |shot| match shot {
                Shot::Miss { point } | Shot::Wall { point } | Shot::Hit { point, .. } => point,
            },
        );
        self.trace = Some(Trace {
            from,
            to,
            remaining: Duration::from_millis(30),
        });
        let shot = shot?;
        let direction = projectile.aim.direction();
        self.projectile = None;
        self.feedback = Feedback::Fired(shot);
        self.damage_effect(shot);
        match shot {
            Shot::Hit { point, .. } => Some(Impact { point, direction }),
            Shot::Miss { .. } | Shot::Wall { .. } => None,
        }
    }

    /// Health transitions emit once when a projectile reaches its first solid surface.
    fn damage_effect(&mut self, shot: Shot) {
        if let Shot::Hit { damage, part, .. } = shot {
            if damage == Damage::Hit {
                if let super::shot::target::Part::Rotor(motor) = part {
                    self.effects.push_back(Effect::RotorDamaged(motor));
                }
            }
            match damage {
                Damage::RotorDestroyed(motor) => self.effects.push_back(Effect::Rotor(motor)),
                Damage::Destroyed => self.effects.push_back(Effect::Destroyed),
                Damage::Hit | Damage::Ignored => {}
            }
        }
    }

    /// Apply one interaction using the arena's authoritative character position.
    pub(super) fn act(&mut self, arena: &Arena, action: Action) {
        match action {
            Action::PickUp => {
                self.feedback = match self.pistol.pick_up(arena.position()) {
                    Ok(()) => Feedback::Armed,
                    Err(error) => Feedback::Pickup(error),
                };
            }
            Action::Fire(direction) => self.fire(arena, direction),
        }
    }

    /// Keep cooldown rejection quiet while retaining all other firing outcomes.
    fn fire(&mut self, arena: &Arena, direction: Dir3) {
        let chest = Self::origin(arena.position());
        let sight = Aim::try_from((chest, *direction)).expect("Finite chest aim");
        let point = self.target.aim_point(arena, sight);
        let pose = super::weapon_pose::clear(
            arena,
            super::weapon_pose::pose(arena.position(), direction, 0.0),
        );
        let from = super::weapon_pose::muzzle(&pose);
        let travel = Dir3::new(point - from).unwrap_or(direction);
        let aim = Aim::try_from((from, *travel)).expect("Finite muzzle aim");
        match self.pistol.fire() {
            Ok(()) => {
                self.projectile = Some(Projectile {
                    aim,
                    travelled: 0.0,
                });
                self.trace = Some(Trace {
                    from,
                    to: from,
                    remaining: Duration::from_millis(30),
                });
                self.feedback = Feedback::Launched;
            }
            Err(FireError::CoolingDown) => {}
            Err(error) => self.feedback = Feedback::Rejected(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Movement;

    #[test]
    fn cover_retracts_the_shared_muzzle_and_blocks_the_projectile() {
        let mut arena = Arena::default();
        let mut combat = Combat::default();
        combat.act(&arena, Action::PickUp);
        for _ in 0..22 {
            arena.step(Movement::Right);
        }
        for _ in 0..50 {
            arena.step(Movement::Forward);
        }
        combat.act(&arena, Action::Fire(Dir3::NEG_Z));
        let trace = combat.trace().expect("Accepted shot");
        assert!((6.0..6.01).contains(&trace.from.z));
        let visible = super::super::weapon_pose::clear(
            &arena,
            super::super::weapon_pose::pose(arena.position(), Dir3::NEG_Z, 0.0),
        );
        assert_eq!(trace.from, super::super::weapon_pose::muzzle(&visible));
        assert!(combat.advance(Duration::from_millis(20), &arena).is_none());
        assert!(matches!(
            combat.feedback(),
            Feedback::Fired(Shot::Wall { .. })
        ));
        assert_eq!(combat.target().health().body_hits_remaining(), 6);
        assert_eq!(combat.rounds(), 11);
    }

    #[test]
    fn projectile_segments_do_not_damage_before_arrival_and_cannot_replay_a_hit() {
        let arena = Arena::default();
        let mut combat = Combat::default();
        combat.act(&arena, Action::PickUp);
        let origin = Combat::origin(arena.position());
        let aim = Dir3::new(combat.target().position() - origin).expect("Body direction");
        combat.act(&arena, Action::Fire(aim));
        assert_eq!(combat.feedback(), Feedback::Launched);
        assert!(combat.advance(Duration::ZERO, &arena).is_none());
        assert_eq!(combat.target().health().body_hits_remaining(), 6);
        assert!(combat.advance(Duration::from_millis(20), &arena).is_none());
        let trace = combat.trace().expect("Travelling segment");
        let muzzle = super::super::weapon_pose::muzzle(&super::super::weapon_pose::pose(
            arena.position(),
            aim,
            0.0,
        ));
        assert!(trace.from.abs_diff_eq(muzzle, 0.00001));
        assert!((trace.to.distance(muzzle) - 2.4).abs() < 0.00001);
        assert_eq!(combat.target().health().body_hits_remaining(), 6);
        let hit = combat
            .advance(Duration::from_millis(80), &arena)
            .expect("Arrived hit");
        let travel = (trace.to - trace.from).normalize();
        assert!((*hit.direction).abs_diff_eq(travel, 0.00001));
        assert_eq!(combat.target().health().body_hits_remaining(), 5);
        assert!(combat.advance(Duration::from_secs(1), &arena).is_none());
        assert_eq!(combat.target().health().body_hits_remaining(), 5);
        assert!(combat.trace().is_none());
    }

    #[test]
    fn a_target_can_evade_a_launched_projectile_and_reset_clears_travel() {
        let arena = Arena::default();
        let mut combat = Combat::default();
        combat.act(&arena, Action::PickUp);
        let aim = Dir3::new(combat.target().position() - Combat::origin(arena.position()))
            .expect("Body direction");
        combat.act(&arena, Action::Fire(aim));
        combat
            .target
            .move_to(Vec3::new(3.0, 2.0, 1.0), Quat::IDENTITY)
            .expect("Moved target");
        assert!(combat.advance(Duration::from_secs(1), &arena).is_none());
        assert_eq!(combat.target().health().body_hits_remaining(), 6);
        assert!(matches!(
            combat.feedback(),
            Feedback::Fired(Shot::Wall { .. } | Shot::Miss { .. })
        ));
        combat.act(&arena, Action::Fire(Dir3::Y));
        combat = Combat::default();
        assert!(combat.advance(Duration::from_secs(1), &arena).is_none());
        assert_eq!(combat.feedback(), Feedback::Unarmed);
        assert!(combat.trace().is_none());
    }

    #[test]
    fn rejected_pickup_and_fire_preserve_ownership_and_ammunition() {
        let mut arena = Arena::default();
        let mut combat = Combat::default();
        combat.act(&arena, Action::Fire(Dir3::Y));
        assert_eq!(combat.feedback(), Feedback::Rejected(FireError::Unarmed));
        for _ in 0..30 {
            arena.step(Movement::Backward);
        }
        combat.act(&arena, Action::PickUp);
        assert_eq!(combat.feedback(), Feedback::Pickup(PickupError::TooFar));
        assert!(!combat.is_armed());
        let arena = Arena::default();
        combat.act(&arena, Action::PickUp);
        combat.act(&arena, Action::PickUp);
        assert_eq!(
            combat.feedback(),
            Feedback::Pickup(PickupError::AlreadyOwned)
        );
        assert_eq!(combat.rounds(), 12);
    }

    #[test]
    fn miss_wall_cooldown_and_empty_feedback_follow_accepted_shots() {
        let arena = Arena::default();
        let mut combat = Combat::default();
        combat.act(&arena, Action::PickUp);
        combat.act(&arena, Action::Fire(Dir3::Y));
        assert!(matches!(combat.feedback(), Feedback::Launched));
        let feedback = combat.feedback();
        combat.advance(Duration::from_millis(1), &arena);
        assert!(combat.trace().is_some());
        combat.act(&arena, Action::Fire(Dir3::NEG_Y));
        assert_eq!(combat.feedback(), feedback);
        assert_eq!(combat.rounds(), 11);
        combat.advance(Duration::from_secs(1), &arena);
        assert!(matches!(
            combat.feedback(),
            Feedback::Fired(Shot::Miss { .. })
        ));
        combat.advance(Duration::from_millis(31), &arena);
        assert!(combat.trace().is_none());
        for _ in 0..11 {
            combat.act(&arena, Action::Fire(Dir3::NEG_Y));
            combat.advance(Duration::from_millis(20), &arena);
            assert!(matches!(
                combat.feedback(),
                Feedback::Fired(Shot::Wall { .. })
            ));
            combat.advance(Duration::from_secs(1), &arena);
        }
        combat.act(&arena, Action::Fire(Dir3::Y));
        assert_eq!(combat.feedback(), Feedback::Rejected(FireError::Empty));
        assert_eq!(combat.rounds(), 0);
        assert!(combat.trace().is_none());
    }
    #[test]
    fn destruction_events_are_ordered_once_and_reset_replaces_pending_events() {
        let arena = Arena::default();
        let mut combat = Combat::default();
        assert_eq!(combat.pop_effect(), Some(Effect::Reset));
        assert_eq!(combat.pop_effect(), None);
        combat.act(&arena, Action::PickUp);
        let motor = DroneMotor::RearRight;
        let aim = Dir3::new(combat.target().rotor_centre(motor) - Combat::origin(arena.position()))
            .expect("Rotor direction");
        combat.act(&arena, Action::Fire(aim));
        combat.advance(Duration::from_millis(100), &arena);
        assert_eq!(combat.pop_effect(), Some(Effect::RotorDamaged(motor)));
        assert_eq!(combat.pop_effect(), None);
        combat.advance(Duration::from_secs(1), &arena);
        combat.act(&arena, Action::Fire(aim));
        combat.advance(Duration::from_millis(100), &arena);
        assert_eq!(combat.pop_effect(), Some(Effect::Rotor(motor)));
        assert_eq!(combat.pop_effect(), None);
        let aim = Dir3::new(combat.target().position() - Combat::origin(arena.position()))
            .expect("Body direction");
        for _ in 0..6 {
            combat.advance(Duration::from_secs(1), &arena);
            combat.act(&arena, Action::Fire(aim));
            combat.advance(Duration::from_millis(100), &arena);
        }
        assert_eq!(combat.pop_effect(), Some(Effect::Destroyed));
        combat.advance(Duration::from_secs(1), &arena);
        combat.act(&arena, Action::Fire(aim));
        combat.advance(Duration::from_millis(100), &arena);
        assert_eq!(combat.pop_effect(), None);
        combat = Combat::default();
        combat.act(&arena, Action::PickUp);
        let aim = Dir3::new(combat.target().rotor_centre(motor) - Combat::origin(arena.position()))
            .expect("Rotor direction");
        for _ in 0..2 {
            combat.act(&arena, Action::Fire(aim));
            combat.advance(Duration::from_millis(100), &arena);
            combat.advance(Duration::from_secs(1), &arena);
        }
        assert_eq!(combat.pop_effect(), Some(Effect::Reset));
        assert_eq!(combat.pop_effect(), Some(Effect::RotorDamaged(motor)));
        assert_eq!(combat.pop_effect(), Some(Effect::Rotor(motor)));
        assert_eq!(combat.pop_effect(), None);
    }
}
