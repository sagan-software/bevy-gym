//! Read-only position, orientation, and velocity in the world frame.

use bevy::math::{Quat, Vec3};

use super::{DroneMotor, DroneMotorState};

/// Drone state in a right-handed frame with +X right, +Y up, and -Z forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneObservation {
    /// Body centre position in metres.
    pub(super) position: Vec3,
    /// Unit quaternion rotating body coordinates into the world frame.
    pub(super) orientation: Quat,
    /// World linear velocity in metres per second.
    pub(super) linear_velocity: Vec3,
    /// World angular velocity in radians per second.
    pub(super) angular_velocity: Vec3,
    /// Actuator health in the documented motor-command order.
    pub(super) motor_states: [DroneMotorState; 4],
}

impl DroneObservation {
    /// Return one motor's health without exposing writable environment state.
    ///
    /// ```compile_fail
    /// use bevy_gym::{Env, robots::{DroneHover, DroneMotorState}};
    /// let mut observation = DroneHover::default().reset(Some(0)).observation;
    /// observation.motor_states = [DroneMotorState::Failed; 4];
    /// ```
    #[must_use]
    pub const fn motor_state(self, motor: DroneMotor) -> DroneMotorState {
        let [front_left, front_right, rear_right, rear_left] = self.motor_states;
        match motor {
            DroneMotor::FrontLeft => front_left,
            DroneMotor::FrontRight => front_right,
            DroneMotor::RearRight => rear_right,
            DroneMotor::RearLeft => rear_left,
        }
    }

    /// Return the body centre position in metres.
    #[must_use]
    pub const fn position(self) -> Vec3 {
        self.position
    }

    /// Return the unit quaternion rotating body coordinates into world coordinates.
    #[must_use]
    pub const fn orientation(self) -> Quat {
        self.orientation
    }

    /// Return world linear velocity in metres per second.
    #[must_use]
    pub const fn linear_velocity(self) -> Vec3 {
        self.linear_velocity
    }

    /// Return world angular velocity in radians per second.
    #[must_use]
    pub const fn angular_velocity(self) -> Vec3 {
        self.angular_velocity
    }
}
