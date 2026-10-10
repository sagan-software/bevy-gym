//! Complete typed actuator requests bound to an observed physical frame.

use super::{RobotFrame, RobotSnapshot};
use crate::robots::{DroidAction, DroneAction};

/// Complete six-agent actuator frame; each array follows `RobotSlot::ALL`.
/// The world validates the source frame; this type never selects an agent action.
///
/// ```compile_fail
/// use bevy_gym::robots::{DroneAction, DroidAction, RobotActions, RobotWorld};
/// let snapshot = RobotWorld::default().snapshot().unwrap();
/// RobotActions::new(&snapshot, [DroneAction::try_from([0.5; 4]).unwrap(); 2],
///     [DroidAction::try_from([0.0; 26]).unwrap(); 3]);
/// ```
#[derive(Debug, Clone)]
pub struct RobotActions {
    /// The immutable physical boundary from which callers inferred these actions.
    pub(super) frame: RobotFrame,
    /// Three validated four-motor requests.
    pub(super) drones: [DroneAction; 3],
    /// Three validated twenty-six-torque requests.
    pub(super) droids: [DroidAction; 3],
}
impl RobotActions {
    /// Bind all six already validated actions to an observed snapshot.
    #[must_use]
    pub fn new(
        snapshot: &RobotSnapshot,
        drones: [DroneAction; 3],
        droids: [DroidAction; 3],
    ) -> Self {
        Self {
            frame: snapshot.frame.clone(),
            drones,
            droids,
        }
    }
}
