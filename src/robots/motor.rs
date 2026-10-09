//! Named actuator identities in the existing motor-command order.

/// One rotor, viewed from above with forward along body -Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneMotor {
    /// Negative X, negative Z.
    FrontLeft,
    /// Positive X, negative Z.
    FrontRight,
    /// Positive X, positive Z.
    RearRight,
    /// Negative X, positive Z.
    RearLeft,
}

impl DroneMotor {
    /// Every rotor in the order accepted by `DroneAction`.
    pub const ALL: [Self; 4] = [
        Self::FrontLeft,
        Self::FrontRight,
        Self::RearRight,
        Self::RearLeft,
    ];
}
