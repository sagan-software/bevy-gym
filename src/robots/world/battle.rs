//! Shared physics, weapon timing and damage without autonomous decisions.
use super::{
    RobotActions, RobotBattleSnapshot, RobotId, RobotMuzzle, RobotSlot, RobotTeam, RobotWorld,
    RobotWorldError, RobotWorldFailure,
};
use crate::{
    robots::{DroneMotor, RobotHealth, RobotWeapon, RobotWeaponOutcome},
    training::CombatSignal,
};

/// Six physical robots whose trigger requests come exclusively from supplied policy actions.
///
/// ```compile_fail
/// use bevy_gym::robots::RobotBattle;
/// let mut battle = RobotBattle::default();
/// battle.world.reset();
/// ```
#[derive(Debug)]
pub struct RobotBattle {
    /// Private shared physics owner.
    world: RobotWorld,
    /// Drone damage state in stable slot order.
    drones: [RobotHealth; 3],
    /// Droid damage state in stable slot order.
    droids: [RobotHealth; 3],
    /// Drone weapon timing in stable slot order.
    drone_weapons: [RobotWeapon; 3],
    /// Droid weapon timing in stable slot order.
    droid_weapons: [RobotWeapon; 3],
    /// At most three live rounds per drone under the fixed charge and recovery profile.
    projectiles: [Option<super::projectile::Projectile>; 9],
}
impl Default for RobotBattle {
    fn default() -> Self {
        Self {
            world: RobotWorld::default(),
            drones: [RobotHealth::from(RobotTeam::Drones); 3],
            droids: [RobotHealth::from(RobotTeam::Droids); 3],
            drone_weapons: [RobotWeapon::from(RobotTeam::Drones); 3],
            droid_weapons: [RobotWeapon::from(RobotTeam::Droids); 3],
            projectiles: [const { None }; 9],
        }
    }
}
impl RobotBattle {
    /// Read one common physical and mechanical boundary.
    ///
    /// # Errors
    /// Returns the original shared-world error.
    ///
    /// # Panics
    /// Panics only if private drone health has the wrong team; construction preserves its team.
    pub fn snapshot(&self) -> Result<RobotBattleSnapshot, RobotWorldError> {
        let mut physical = self.world.snapshot()?;
        // Health observations report the same failed motors used by the physical actuators.
        for (body, health) in physical.drones.iter_mut().zip(self.drones) {
            body.motor_states = DroneMotor::ALL
                .map(|motor| health.motor_state(motor).expect("drone roster health"));
        }
        Ok(RobotBattleSnapshot {
            physical,
            drones: self.drones,
            droids: self.droids,
            drone_weapons: self.drone_weapons.map(|weapon| weapon.phase()),
            droid_weapons: self.droid_weapons.map(|weapon| weapon.phase()),
        })
    }
    /// Advance complete motor, torque and trigger requests from one source boundary.
    ///
    /// # Errors
    /// Returns source-frame, clock or physical-state failures before further advancement.
    ///
    /// # Panics
    /// Panics only if private weapon timing exceeds nine live drone rounds.
    /// The fixed charge, volley, recovery and lifetime profile preserves that bound.
    pub fn advance(
        &mut self,
        actions: RobotActions,
        drone_signals: [CombatSignal; 3],
        droid_signals: [CombatSignal; 3],
    ) -> Result<RobotBattleSnapshot, RobotWorldError> {
        // Reject the common source before consuming any trigger or mechanical time.
        self.world.check_actions(&actions)?;
        let physical = self.world.snapshot()?;
        let (drone_fired, droid_fired) = self.advance_triggers(drone_signals, droid_signals);
        self.fire_rounds(&physical, drone_fired, droid_fired)?;
        self.world
            .advance_with_health(actions, &mut self.drones, &mut self.droids)?;
        self.advance_rounds()?;
        self.disable_destroyed();
        self.snapshot()
    }

    /// Advance mechanical timing for all live requests before any collision damage.
    fn advance_triggers(
        &mut self,
        drone_signals: [CombatSignal; 3],
        droid_signals: [CombatSignal; 3],
    ) -> ([bool; 3], [bool; 3]) {
        // Collect all live discharges before applying damage, preserving simultaneous decisions.
        let mut drone_fired = [false; 3];
        let mut droid_fired = [false; 3];
        for (((weapon, signal), health), fired) in self
            .drone_weapons
            .iter_mut()
            .zip(drone_signals)
            .zip(self.drones)
            .zip(&mut drone_fired)
        {
            if !health.is_alive() {
                weapon.disable();
            }
            *fired = weapon.advance(signal) == RobotWeaponOutcome::Fired;
        }
        for (((weapon, signal), health), fired) in self
            .droid_weapons
            .iter_mut()
            .zip(droid_signals)
            .zip(self.droids)
            .zip(&mut droid_fired)
        {
            if !health.is_alive() {
                weapon.disable();
            }
            *fired = weapon.advance(signal) == RobotWeaponOutcome::Fired;
        }
        (drone_fired, droid_fired)
    }

    /// Derive physical muzzles for discharges already authorized by weapon timing.
    fn fire_rounds(
        &mut self,
        physical: &super::RobotSnapshot,
        drone_fired: [bool; 3],
        droid_fired: [bool; 3],
    ) -> Result<(), RobotWorldError> {
        for ((slot, drone), droid) in RobotSlot::ALL.into_iter().zip(drone_fired).zip(droid_fired) {
            if drone {
                let owner = RobotId::Drone(slot);
                let muzzle =
                    RobotMuzzle::try_from((physical, owner)).map_err(|_invalid_geometry| {
                        RobotWorldError::Rejected(RobotWorldFailure::NonFinite(owner))
                    })?;
                let vacant = self
                    .projectiles
                    .iter_mut()
                    .find(|round| round.is_none())
                    .expect("weapon timing bounds live drone rounds to nine");
                *vacant = Some(super::projectile::Projectile::from(&muzzle));
            }
            if droid {
                let owner = RobotId::Droid(slot);
                let muzzle =
                    RobotMuzzle::try_from((physical, owner)).map_err(|_invalid_geometry| {
                        RobotWorldError::Rejected(RobotWorldFailure::NonFinite(owner))
                    })?;
                if let Some(hit) = self.world.cast_muzzle(&muzzle)? {
                    self.hit(hit.target().owner());
                }
            }
        }
        Ok(())
    }

    /// Sweep retained projectiles and apply each consumed collision once.
    fn advance_rounds(&mut self) -> Result<(), RobotWorldError> {
        // Sweep every retained round against the resulting authoritative collision poses.
        let mut hits = [None; 9];
        for (round, hit) in self.projectiles.iter_mut().zip(&mut hits) {
            if let Some(projectile) = round.take() {
                match projectile.advance(&self.world)? {
                    super::projectile::Flight::Flying(projectile) => *round = Some(projectile),
                    super::projectile::Flight::Expired => {}
                    super::projectile::Flight::Hit(collision) => {
                        *hit = collision.target().owner();
                        *round = None;
                    }
                }
            }
        }
        for owner in hits {
            self.hit(owner);
        }
        Ok(())
    }

    /// Revoke destroyed robots' future trigger requests without removing passive bodies.
    fn disable_destroyed(&mut self) {
        // Fatal damage disables later requests while retained bodies remain in the solver.
        for (weapon, health) in self
            .drone_weapons
            .iter_mut()
            .zip(self.drones)
            .chain(self.droid_weapons.iter_mut().zip(self.droids))
        {
            if !health.is_alive() {
                weapon.disable();
            }
        }
    }

    /// Rebuild physics, weapons, health and projectiles with a fresh source-frame scope.
    ///
    /// # Errors
    /// Returns a physical-state error if the rebuilt world is invalid.
    pub fn reset(&mut self) -> Result<RobotBattleSnapshot, RobotWorldError> {
        *self = Self::default();
        self.snapshot()
    }

    /// Apply one physical body collision; floor hits consume rounds without damaging robots.
    const fn hit(&mut self, owner: Option<RobotId>) {
        let Some(owner) = owner else {
            return;
        };
        let health = match owner {
            RobotId::Drone(_) => &mut self.drones,
            RobotId::Droid(_) => &mut self.droids,
        };
        let [first, second, third] = health;
        let target = match owner.slot() {
            RobotSlot::First => first,
            RobotSlot::Second => second,
            RobotSlot::Third => third,
        };
        target.hit_body();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robots::{DroidAction, DroneAction, DroneMotorState};

    /// Destroyed bodies ignore high actuator requests while continuing passive physics.
    #[test]
    fn destroyed_bodies_cannot_exert_motor_or_joint_force() {
        let mut requested = RobotBattle::default();
        let mut released = RobotBattle::default();
        for battle in [&mut requested, &mut released] {
            let initial = battle.snapshot().unwrap();
            battle
                .advance(
                    fixture_actions(initial.physical(), 1.0, 1.0),
                    [CombatSignal::IdleReleased; 3],
                    [CombatSignal::IdleReleased; 3],
                )
                .unwrap();
            let [_, _, drone] = &mut battle.drones;
            drone.destroy();
            let [_, _, droid] = &mut battle.droids;
            droid.destroy();
        }
        let high = requested.snapshot().unwrap();
        let zero = released.snapshot().unwrap();
        let high_actions = fixture_actions(high.physical(), 1.0, 1.0);
        let zero_actions = fixture_actions(zero.physical(), 0.0, 0.0);
        let high = requested
            .advance(
                high_actions,
                [CombatSignal::PursuitRequested; 3],
                [CombatSignal::PursuitRequested; 3],
            )
            .unwrap();
        let zero = released
            .advance(
                zero_actions,
                [CombatSignal::PursuitRequested; 3],
                [CombatSignal::PursuitRequested; 3],
            )
            .unwrap();
        assert_eq!(
            high.physical().drone(RobotSlot::Third),
            zero.physical().drone(RobotSlot::Third)
        );
        assert_eq!(
            high.physical().droid(RobotSlot::Third),
            zero.physical().droid(RobotSlot::Third)
        );
        assert_disabled(&high);
    }

    /// Confirm mechanical shutdown for the fixture's third drone and droid.
    fn assert_disabled(snapshot: &RobotBattleSnapshot) {
        for motor in DroneMotor::ALL {
            assert_eq!(
                snapshot
                    .physical()
                    .drone(RobotSlot::Third)
                    .motor_state(motor),
                DroneMotorState::Failed
            );
        }
        for owner in [
            RobotId::Drone(RobotSlot::Third),
            RobotId::Droid(RobotSlot::Third),
        ] {
            assert_eq!(
                snapshot.weapon_phase(owner),
                crate::robots::RobotWeaponPhase::Disabled
            );
        }
    }

    /// Build isolated actuator fixtures without creating an autonomous controller.
    fn fixture_actions(
        snapshot: &super::super::RobotSnapshot,
        drone: f32,
        droid: f32,
    ) -> RobotActions {
        RobotActions::new(
            snapshot,
            [
                DroneAction::try_from([0.5; 4]).unwrap(),
                DroneAction::try_from([0.5; 4]).unwrap(),
                DroneAction::try_from([drone; 4]).unwrap(),
            ],
            [
                DroidAction::default(),
                DroidAction::default(),
                DroidAction::try_from([droid; 26]).unwrap(),
            ],
        )
    }

    /// A failed rotor contributes exactly the same force as a valid zero request.
    #[test]
    fn failed_rotor_suppresses_its_physical_thrust() {
        let mut damaged = RobotBattle::default();
        let mut zero = RobotBattle::default();
        let [_, _, health] = &mut damaged.drones;
        health.hit_rotor(DroneMotor::FrontLeft).unwrap();
        health.hit_rotor(DroneMotor::FrontLeft).unwrap();
        let snapshot = damaged.snapshot().unwrap();
        let high = fixture_actions(snapshot.physical(), 1.0, 0.0);
        let snapshot = zero.snapshot().unwrap();
        let mut released = fixture_actions(snapshot.physical(), 1.0, 0.0);
        let [_, _, drone] = &mut released.drones;
        *drone = DroneAction::try_from([0.0, 1.0, 1.0, 1.0]).unwrap();
        let high = damaged
            .advance(
                high,
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3],
            )
            .unwrap();
        let released = zero
            .advance(
                released,
                [CombatSignal::IdleReleased; 3],
                [CombatSignal::IdleReleased; 3],
            )
            .unwrap();
        let high = high.physical().drone(RobotSlot::Third);
        let released = released.physical().drone(RobotSlot::Third);
        assert_eq!(high.position(), released.position());
        assert_eq!(high.orientation(), released.orientation());
        assert_eq!(high.linear_velocity(), released.linear_velocity());
        assert_eq!(high.angular_velocity(), released.angular_velocity());
        assert_eq!(
            high.motor_state(DroneMotor::FrontLeft),
            DroneMotorState::Failed
        );
    }

    /// A pistol discharge hits the nearest physical body along its actual forearm axis.
    #[test]
    fn pistol_collision_applies_one_body_hit_without_target_selection() {
        let mut battle = RobotBattle::default();
        let snapshot = battle.snapshot().unwrap();
        let muzzle =
            RobotMuzzle::try_from((snapshot.physical(), RobotId::Droid(RobotSlot::Second)))
                .unwrap();
        battle
            .world
            .place_drone_fixture(RobotSlot::Second, muzzle.origin() + muzzle.direction());
        assert_eq!(
            battle.world.cast_muzzle(&muzzle).unwrap().unwrap().target(),
            super::super::RobotRayTarget::Drone(RobotSlot::Second)
        );
        let snapshot = battle.snapshot().unwrap();
        let actions = fixture_actions(snapshot.physical(), 0.5, 0.0);
        let after = battle
            .advance(
                actions,
                [CombatSignal::IdleReleased; 3],
                [
                    CombatSignal::IdleReleased,
                    CombatSignal::PursuitRequested,
                    CombatSignal::IdleReleased,
                ],
            )
            .unwrap();
        assert_eq!(
            after
                .health(RobotId::Drone(RobotSlot::Second))
                .body_hits_remaining()
                .get(),
            5
        );
        assert_eq!(
            after
                .health(RobotId::Drone(RobotSlot::First))
                .body_hits_remaining()
                .get(),
            6
        );
    }

    /// Source rejection occurs before any ammunition, charge or projectile state changes.
    #[test]
    fn rejected_actions_preserve_all_weapon_state() {
        let mut battle = RobotBattle::default();
        let foreign = RobotWorld::default().snapshot().unwrap();
        let actions = fixture_actions(&foreign, 0.5, 0.0);
        assert_eq!(
            battle
                .advance(
                    actions,
                    [CombatSignal::PursuitRequested; 3],
                    [CombatSignal::PursuitRequested; 3]
                )
                .unwrap_err(),
            RobotWorldError::Rejected(RobotWorldFailure::WrongFrame)
        );
        assert!(battle.projectiles.iter().all(Option::is_none));
        for weapon in battle.drone_weapons.into_iter().chain(battle.droid_weapons) {
            assert_eq!(weapon.phase(), crate::robots::RobotWeaponPhase::Ready);
        }
    }

    /// Floor collisions consume rounds without changing health; repeated fatal hits stay absorbed.
    #[test]
    fn body_damage_uses_collision_owner_and_absorbing_death() {
        let mut battle = RobotBattle::default();
        battle.hit(None);
        for slot in RobotSlot::ALL {
            let drone = RobotId::Drone(slot);
            let droid = RobotId::Droid(slot);
            for _ in 0..8 {
                battle.hit(Some(drone));
                battle.hit(Some(droid));
            }
            let snapshot = battle.snapshot().unwrap();
            assert!(!snapshot.health(drone).is_alive());
            assert!(!snapshot.health(droid).is_alive());
        }
    }
}
