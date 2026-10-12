//! Terminal shared combat lifecycle without autonomous decisions.
use super::{
    RobotActions, RobotBattle, RobotBattleSnapshot, RobotMatchError, RobotMatchResult,
    RobotMatchStatus, RobotWorldError,
};
use crate::training::CombatSignal;
use std::{num::NonZeroU32, time::Duration};
/// A match whose terminal state cannot retain a mutable combat environment.
///
/// ```compile_fail
/// use bevy_gym::robots::RobotMatch;
/// let mut game = RobotMatch::try_from(std::time::Duration::from_mins(2)).unwrap();
/// game.lifecycle.battle.advance();
/// ```
#[derive(Debug)]
pub struct RobotMatch {
    /// Validated complete duration, retained only for explicit reset.
    limit: MatchFrames,
    /// Own the capabilities permitted by exactly one current lifecycle state.
    lifecycle: Lifecycle,
}
/// Positive bounded count of complete 20 ms match frames.
#[derive(Debug, Clone, Copy)]
struct MatchFrames(NonZeroU32);
/// Terminal states own no mutable combat backend.
#[derive(Debug)]
enum Lifecycle {
    /// Continue policy requests through shared mechanics.
    Running {
        /// All six physical bodies, health, weapons and pending projectiles.
        battle: Box<RobotBattle>,
        /// Positive number of complete physical frames remaining.
        remaining: MatchFrames,
    },
    /// Retain only a read-only final snapshot and its mechanical result.
    Finished {
        /// Final shared physical boundary.
        snapshot: Box<RobotBattleSnapshot>,
        /// Settled elimination or elapsed duration.
        result: RobotMatchResult,
    },
    /// Preserve the original physical failure without a mutable backend.
    Failed(RobotWorldError),
}
impl TryFrom<Duration> for RobotMatch {
    type Error = RobotMatchError;
    fn try_from(duration: Duration) -> Result<Self, Self::Error> {
        // Convert nanoseconds to exact 20 ms frames before constructing retained state.
        let nanos = duration.as_nanos();
        if !nanos.is_multiple_of(20_000_000) {
            return Err(RobotMatchError::InvalidDuration);
        }
        let frames = u32::try_from(nanos / 20_000_000)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(RobotMatchError::InvalidDuration)?;
        let limit = MatchFrames(frames);
        Ok(Self {
            limit,
            lifecycle: Lifecycle::Running {
                battle: Box::default(),
                remaining: limit,
            },
        })
    }
}
impl RobotMatch {
    /// Derive status without exposing the mutable combat backend.
    #[must_use]
    pub const fn status(&self) -> RobotMatchStatus {
        match &self.lifecycle {
            Lifecycle::Running { .. } => RobotMatchStatus::Running,
            Lifecycle::Finished { result, .. } => RobotMatchStatus::Finished(*result),
            Lifecycle::Failed(error) => RobotMatchStatus::Failed(*error),
        }
    }
    /// Read physical and mechanical measurements in running or finished states.
    /// # Errors
    /// Returns the preserved physical cause when the match failed.
    pub fn snapshot(&self) -> Result<RobotBattleSnapshot, RobotMatchError> {
        match &self.lifecycle {
            Lifecycle::Running { battle, .. } => battle.snapshot().map_err(RobotMatchError::Failed),
            Lifecycle::Finished { snapshot, .. } => Ok((**snapshot).clone()),
            Lifecycle::Failed(error) => Err(RobotMatchError::Failed(*error)),
        }
    }
    /// Advance living policy requests until settled elimination or configured timeout.
    /// # Errors
    /// Rejects terminal requests without side effects and preserves physical failures.
    /// # Panics
    /// Panics only if private frame-count or projectile-storage invariants are violated.
    /// Validated durations and fixed weapon timing preserve these invariants.
    pub fn advance(
        &mut self,
        actions: RobotActions,
        drones: [CombatSignal; 3],
        droids: [CombatSignal; 3],
    ) -> Result<RobotBattleSnapshot, RobotMatchError> {
        // Deny terminal requests before source validation, ammunition or physical time.
        let (battle, remaining) = match &mut self.lifecycle {
            Lifecycle::Running { battle, remaining } => (battle, remaining),
            Lifecycle::Finished { result, .. } => return Err(RobotMatchError::Finished(*result)),
            Lifecycle::Failed(error) => return Err(RobotMatchError::Failed(*error)),
        };
        let snapshot = match battle.advance(actions, drones, droids) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                // Retire the mutable backend on the first failure, retaining its exact diagnostic.
                self.lifecycle = Lifecycle::Failed(error);
                return Err(RobotMatchError::Failed(error));
            }
        };
        // Settled elimination precedes timeout; in-flight rounds can still change surviving health.
        let elimination = settled_result(&snapshot, battle.rounds_pending());
        let result =
            elimination.or_else(|| (remaining.0.get() == 1).then_some(RobotMatchResult::Timeout));
        if let Some(result) = result {
            self.lifecycle = Lifecycle::Finished {
                snapshot: Box::new(snapshot.clone()),
                result,
            };
        } else {
            remaining.0 = NonZeroU32::new(remaining.0.get() - 1)
                .expect("nonterminal frame count exceeds one");
        }
        Ok(snapshot)
    }
    /// Recreate all physical mechanics and the validated complete duration.
    pub fn reset(&mut self) {
        self.lifecycle = Lifecycle::Running {
            battle: Box::default(),
            remaining: self.limit,
        };
    }
}

/// Derive settled elimination solely from measured health and pending mechanical rounds.
fn settled_result(
    snapshot: &RobotBattleSnapshot,
    rounds_pending: bool,
) -> Option<RobotMatchResult> {
    if rounds_pending {
        return None;
    }
    let drones_alive = super::RobotSlot::ALL
        .into_iter()
        .any(|slot| snapshot.health(super::RobotId::Drone(slot)).is_alive());
    let droids_alive = super::RobotSlot::ALL
        .into_iter()
        .any(|slot| snapshot.health(super::RobotId::Droid(slot)).is_alive());
    match (drones_alive, droids_alive) {
        (true, true) => None,
        (true, false) => Some(RobotMatchResult::DronesWin),
        (false, true) => Some(RobotMatchResult::DroidsWin),
        (false, false) => Some(RobotMatchResult::MutualElimination),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Every elimination outcome waits for pending rounds; one death leaves both teams living.
    #[test]
    fn elimination_requires_a_complete_team_and_settled_rounds() {
        let mut snapshot = RobotBattle::default().snapshot().unwrap();
        assert_eq!(settled_result(&snapshot, false), None);
        let [first, ..] = &mut snapshot.drones;
        first.destroy();
        assert_eq!(settled_result(&snapshot, false), None);
        for drone in &mut snapshot.drones {
            drone.destroy();
        }
        assert_eq!(settled_result(&snapshot, true), None);
        assert_eq!(
            settled_result(&snapshot, false),
            Some(RobotMatchResult::DroidsWin)
        );
        for droid in &mut snapshot.droids {
            droid.destroy();
        }
        assert_eq!(settled_result(&snapshot, true), None);
        assert_eq!(
            settled_result(&snapshot, false),
            Some(RobotMatchResult::MutualElimination)
        );
        snapshot.drones = [crate::robots::RobotHealth::from(super::super::RobotTeam::Drones); 3];
        assert_eq!(
            settled_result(&snapshot, false),
            Some(RobotMatchResult::DronesWin)
        );
    }
    /// The exact maximum duration remains valid, while the next complete frame is rejected.
    #[test]
    fn duration_upper_boundary_is_exact() {
        let maximum = Duration::from_millis(u64::from(u32::MAX) * 20);
        RobotMatch::try_from(maximum).unwrap();
        assert_eq!(
            RobotMatch::try_from(maximum + Duration::from_millis(20)).unwrap_err(),
            RobotMatchError::InvalidDuration
        );
    }
}
