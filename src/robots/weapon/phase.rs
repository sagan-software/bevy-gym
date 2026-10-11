//! Read-only weapon timing measurements.
use super::RobotVolleyRounds;
use std::time::Duration;
/// Derived mechanical state; constructing this measurement cannot change a weapon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobotWeaponPhase {
    /// A current requested trigger may begin charging or discharge.
    Ready,
    /// The drone requires a continuously requested trigger throughout charging.
    Charging {
        /// Positive remaining charge time.
        remaining: Duration,
    },
    /// A later round requires both elapsed delay and a current trigger request.
    Volley {
        /// Positive time before the next round.
        remaining: Duration,
        /// Closed count of later rounds.
        rounds: RobotVolleyRounds,
    },
    /// Recovery elapses independently of trigger requests.
    Cooling {
        /// Positive remaining cooldown time.
        remaining: Duration,
    },
    /// The droid magazine is exhausted; no automatic reload exists.
    Empty,
    /// Lifecycle shutdown prevents further discharges.
    Disabled,
}
