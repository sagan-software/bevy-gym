//! Two-hit rotor integrity.
/// Rotor integrity remains readable after hull death.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RobotRotorHealth {
    /// Two hits remain before motor failure.
    #[default]
    Intact,
    /// One more hit causes motor failure.
    Damaged,
    /// Further hits have no effect.
    Destroyed,
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Standalone default integrity agrees with the fresh physical rotor state.
    #[test]
    fn default_rotor_is_intact() {
        assert_eq!(RobotRotorHealth::default(), RobotRotorHealth::Intact);
    }
}
