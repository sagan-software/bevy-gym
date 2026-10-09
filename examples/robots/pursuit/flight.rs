//! Follow sight-driven goals with learned motor control and arena collisions.

use super::{
    arena::{layout::Surface, Arena},
    combat::health::{DroneHealth, RotorHealth},
    flight_control::{FlightGoal, FlightPilot},
    navigation::{Navigator, INTERVAL},
    sight::Contact,
};
use bevy_gym::{
    robots::{DroneHover, DroneMotor, DroneObservation, DroneObstacle},
    Env, Step,
};
use std::time::Duration;

/// Flight owns its solver, programmed navigation, and either a loaded pilot or failure.
pub(super) struct Flight {
    /// Existing flight task with immutable arena collisions.
    environment: DroneHover,
    /// Failure stops inference and physics until reset retries the bundled checkpoint.
    pilot: Result<FlightPilot, String>,
    /// Static geometry and observation history contain no hidden character state.
    navigation: Navigator,
    /// A goal remains fixed across its five motor actions.
    command: Command,
}

/// Navigation runs at ten hertz while the motor pilot runs at fifty hertz.
#[derive(Clone, Copy)]
enum Command {
    /// The next motor action first consumes a filtered sighting.
    Due,
    /// Later motor actions retain the same goal until its interval ends.
    Holding {
        /// Validated position and heading for the motor pilot.
        goal: FlightGoal,
        /// Positive time remaining before another navigation decision.
        remaining: Duration,
    },
}

impl Flight {
    /// Copy the authored static layout once; the flight task already owns its floor.
    pub(super) fn new(arena: &Arena) -> Self {
        let obstacles = arena
            .blocks()
            .iter()
            .filter(|block| block.surface != Surface::Floor)
            .map(|block| {
                DroneObstacle::try_from((block.centre, block.half, block.rotation))
                    .expect("Authored arena boxes are valid")
            });
        let mut environment = DroneHover::with_obstacles(obstacles);
        environment.reset(Some(42));
        Self {
            environment,
            pilot: FlightPilot::bundled().map_err(|error| error.to_string()),
            navigation: Navigator::default(),
            command: Command::Due,
        }
    }

    /// Reset physics and observation history while retaining valid weights and static geometry.
    pub(super) fn reset(&mut self) {
        self.environment.reset(Some(42));
        self.navigation.reset();
        self.command = Command::Due;
        if self.pilot.is_err() {
            self.pilot = FlightPilot::bundled().map_err(|error| error.to_string());
        }
    }

    /// Report an inference failure without exposing a controller or mutable world.
    pub(super) fn error(&self) -> Option<&str> {
        self.pilot.as_ref().err().map(String::as_str)
    }

    /// Stop further control while preserving the last physics observation for diagnosis.
    pub(super) fn fail(&mut self, message: String) {
        self.pilot = Err(message);
    }

    /// Read the authoritative pose, motor states, and velocities.
    pub(super) fn observation(&self) -> DroneObservation {
        self.environment.observation()
    }

    /// Forward a hit to the same solver used by the learned motor policy.
    pub(super) fn impact(
        &mut self,
        impulse: bevy_gym::robots::DroneImpulse,
    ) -> Result<(), bevy_gym::robots::DroneImpulseRejected> {
        self.environment.apply_impulse(impulse)
    }

    /// Apply rotor health before navigation, inference, and twenty milliseconds of physics.
    pub(super) fn advance(
        &mut self,
        health: &DroneHealth,
        contact: Contact,
    ) -> Option<Step<DroneObservation>> {
        if !health.is_alive() || self.pilot.is_err() {
            return None;
        }
        for motor in DroneMotor::ALL {
            if health.rotor(motor) == RotorHealth::Destroyed {
                if let Err(error) = self.environment.fail_motor(motor) {
                    self.fail(error.to_string());
                    return None;
                }
            }
        }
        let goal = self.goal(contact);
        match self
            .pilot
            .as_ref()
            .ok()?
            .action(self.environment.observation(), goal)
        {
            Ok(action) => Some(self.environment.step(action)),
            Err(error) => {
                self.fail(error.to_string());
                None
            }
        }
    }

    /// Consume contact only when a new navigation interval begins.
    fn goal(&mut self, contact: Contact) -> FlightGoal {
        let (goal, remaining) = match self.command {
            Command::Due => (
                self.navigation
                    .goal(self.environment.observation(), contact),
                INTERVAL,
            ),
            Command::Holding { goal, remaining } => (goal, remaining),
        };
        let remaining = remaining.saturating_sub(Duration::from_millis(20));
        self.command = if remaining.is_zero() {
            Command::Due
        } else {
            Command::Holding { goal, remaining }
        };
        goal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec3;
    use bevy_gym::robots::DroneAction;

    #[test]
    fn controller_failure_freezes_physics_and_reset_retries() {
        let mut flight = Flight::new(&Arena::default());
        let initial = flight.observation();
        flight.fail("test controller failure".to_owned());
        assert_eq!(flight.error(), Some("test controller failure"));
        assert!(flight
            .advance(&DroneHealth::default(), Contact::Unknown)
            .is_none());
        assert_eq!(flight.observation(), initial);
        flight.reset();
        assert!(flight.error().is_none());
        assert!(flight
            .advance(&DroneHealth::default(), Contact::Unknown)
            .is_some());
    }

    #[test]
    fn navigation_holds_each_goal_for_exactly_five_motor_actions() {
        let mut flight = Flight::new(&Arena::default());
        let first = flight.goal(Contact::Unknown);
        let visible = Contact::Visible(Vec3::new(-5.0, 1.5, -3.0));
        for _ in 0..4 {
            assert_eq!(flight.goal(visible), first);
        }
        assert_ne!(flight.goal(visible), first);
        flight.reset();
        assert_eq!(flight.goal(Contact::Unknown), first);
    }

    #[test]
    fn terminal_motor_rejection_cannot_resume_a_private_stopped_world() {
        let mut flight = Flight::new(&Arena::default());
        let off = DroneAction::try_from([0.0; 4]).expect("Valid command");
        while !flight.environment.step(off).is_done() {}
        let terminal = flight.observation();
        let mut health = DroneHealth::default();
        health.hit_rotor(DroneMotor::FrontLeft);
        health.hit_rotor(DroneMotor::FrontLeft);
        assert!(flight.advance(&health, Contact::Unknown).is_none());
        assert!(flight.error().is_some());
        assert_eq!(flight.observation(), terminal);
    }
}
