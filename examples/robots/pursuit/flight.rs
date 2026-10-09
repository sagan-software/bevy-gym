//! Reuse qualified motor dynamics with arena geometry and the healthy hover policy.

use super::{
    arena::{layout::Surface, Arena},
    combat::health::{DroneHealth, RotorHealth},
    encoding, model,
};
use bevy_gym::{
    robots::{DroneAction, DroneHover, DroneMotor, DroneObservation, DroneObstacle},
    training::{RecurrentMemory, RecurrentPpoPolicy},
    Env, Step,
};

/// Flight owns its solver and either a loaded controller or a diagnostic failure.
pub(super) struct Flight {
    /// Existing hover task with immutable arena collisions.
    environment: DroneHover,
    /// Failure stops inference and physics until reset retries the bundled checkpoint.
    pilot: Result<Pilot, String>,
}

/// Frozen weights and the recurrent memory of this episode.
struct Pilot {
    /// Bundled twelve-feature healthy hover policy.
    policy: RecurrentPpoPolicy,
    /// Updated only after the policy emits a valid motor command.
    memory: RecurrentMemory,
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
            pilot: Pilot::bundled(),
        }
    }

    /// Reset deterministic physics and episode memory while retaining valid weights.
    pub(super) fn reset(&mut self) {
        self.environment.reset(Some(42));
        match &mut self.pilot {
            Ok(pilot) => pilot.memory = pilot.policy.initial_memory(),
            Err(_) => self.pilot = Pilot::bundled(),
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

    /// Apply rotor health before requesting the next twenty-millisecond flight action.
    pub(super) fn advance(&mut self, health: &DroneHealth) -> Option<Step<DroneObservation>> {
        if !health.is_alive() {
            return None;
        }
        let pilot = self.pilot.as_mut().ok()?;
        for motor in DroneMotor::ALL {
            if health.rotor(motor) == RotorHealth::Destroyed {
                if let Err(error) = self.environment.fail_motor(motor) {
                    self.fail(error.to_string());
                    return None;
                }
            }
        }
        let command = pilot.command(self.environment.observation());
        match command {
            Ok(action) => Some(self.environment.step(action)),
            Err(error) => {
                self.fail(error);
                None
            }
        }
    }
}

impl Pilot {
    /// Decode the same weights and architecture used by the hover viewer.
    fn bundled() -> Result<Self, String> {
        Self::load(include_bytes!("../../../assets/robots/recovery.mpk").to_vec())
    }

    /// Keep checkpoint decoding fallible so corrupted assets cannot start physics.
    fn load(bytes: Vec<u8>) -> Result<Self, String> {
        let policy = model::load_policy(bytes).map_err(|error| error.to_string())?;
        let memory = policy.initial_memory();
        Ok(Self { policy, memory })
    }

    /// Validate output before committing its next recurrent state.
    fn command(&mut self, observation: DroneObservation) -> Result<DroneAction, String> {
        let prediction = self
            .policy
            .mean_action(&encoding::encode(observation), &self.memory)
            .map_err(|error| error.to_string())?;
        let action =
            encoding::decode_action(&prediction.action).map_err(|error| error.to_string())?;
        self.memory = prediction.next_memory;
        Ok(action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_gym::training::{RecurrentPpoAgent, SeedConfig};

    #[test]
    fn failed_loading_and_inference_freeze_physics_and_reset_retries() {
        let mut flight = Flight::new(&Arena::default());
        let initial = flight.observation();
        flight.pilot = Pilot::load(Vec::new());
        assert!(flight.error().is_some());
        assert!(flight.advance(&DroneHealth::default()).is_none());
        assert_eq!(flight.observation(), initial);
        flight.reset();
        assert!(flight.error().is_none());
        for (inputs, outputs) in [(11, 4), (12, 3)] {
            let agent = RecurrentPpoAgent::new(
                inputs,
                inputs,
                1,
                &vec![0.0; outputs],
                &vec![1.0; outputs],
                model::learning_config(),
                SeedConfig::from_root(7),
            )
            .expect("Valid incompatible policy");
            let policy = agent.policy();
            let memory = policy.initial_memory();
            flight.pilot = Ok(Pilot { policy, memory });
            assert!(flight.advance(&DroneHealth::default()).is_none());
            assert!(flight.error().is_some());
            assert_eq!(flight.observation(), initial);
            flight.reset();
            assert!(flight.error().is_none());
        }
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
        assert!(flight.advance(&health).is_none());
        assert!(flight.error().is_some());
        assert_eq!(flight.observation(), terminal);
    }
}
