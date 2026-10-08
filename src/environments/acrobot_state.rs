//! Bounded Acrobot joint angles and angular velocities.

/// Angles within ±pi radians and velocities within ±4pi and ±9pi radians per second.
///
/// Diagnostic construction rejects states outside these default environment bounds.
/// This keeps every accepted state within the finite numerical integration profile.
/// Private fields prevent bypassing validation:
///
/// ```compile_fail
/// use bevy_gym::environments::AcrobotState;
/// let invalid = AcrobotState { values: [f64::INFINITY; 4] };
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcrobotState {
    /// Angles first, followed by angular velocities, in Gymnasium order.
    values: [f64; 4],
}
/// A nonfinite coordinate or a coordinate outside its default physical bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidAcrobotState;
impl TryFrom<[f64; 4]> for AcrobotState {
    type Error = InvalidAcrobotState;
    fn try_from(values: [f64; 4]) -> Result<Self, Self::Error> {
        let pi = std::f64::consts::PI;
        let limits = [pi, pi, 4.0 * pi, 9.0 * pi];
        // Inclusive ranges also reject NaN; no invalid coordinate can be retained.
        if values
            .iter()
            .zip(limits)
            .all(|(value, limit)| (-limit..=limit).contains(value))
        {
            Ok(Self { values })
        } else {
            Err(InvalidAcrobotState)
        }
    }
}
impl AcrobotState {
    /// Borrow the validated angles and angular velocities.
    #[must_use]
    pub const fn values(&self) -> &[f64; 4] {
        &self.values
    }
}
impl std::fmt::Display for InvalidAcrobotState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .write_str("Acrobot requires angles within ±pi and velocities within ±4pi and ±9pi")
    }
}
impl std::error::Error for InvalidAcrobotState {}
