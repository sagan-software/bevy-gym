//! Apply curriculum failures at observation boundaries without changing drone physics.

use std::num::NonZeroU16;

use bevy_gym::robots::{DroneAction, DroneHover, DroneMotor, DroneObservation};
use bevy_gym::training::SplitMix64;
use bevy_gym::{Env, Reset, Step};

/// A calm flight with one fixed curriculum schedule retained across resets.
pub(crate) struct DamageTask {
    /// Authoritative physical state and actuator health.
    drone: DroneHover,
    /// Failure distribution selected for this lesson.
    lesson: Lesson,
    /// Schedule stream independent of the drone's physical reset stream.
    random: SplitMix64,
    /// Remaining intact actions; absence means no future failure is scheduled.
    pending: Option<PendingFailure>,
}

/// Legal lesson profiles; a fixed failure occurs immediately at reset.
#[derive(Clone, Copy)]
enum Lesson {
    /// Intact calm hover supplies the initial motor-balance lesson.
    Hover,
    /// The same corner fails at every reset.
    FrontLeft,
    /// Choose a corner and either two or five intact seconds on each reset.
    Scheduled,
}

/// A future failure with a positive count of intact commands remaining.
struct PendingFailure {
    /// Motor selected when this episode began.
    motor: DroneMotor,
    /// Number of intact 20 ms commands remaining before the failure.
    remaining: NonZeroU16,
}

impl DamageTask {
    /// Begin the intact hover lesson.
    pub(crate) fn hover() -> Self {
        Self::new(Lesson::Hover)
    }

    /// Begin the fixed-corner lesson with the front-left motor already failed.
    pub(crate) fn front_left() -> Self {
        Self::new(Lesson::FrontLeft)
    }

    /// Begin the randomized corner and failure-time lesson.
    pub(crate) fn scheduled() -> Self {
        Self::new(Lesson::Scheduled)
    }

    /// Establish a valid initial observation even before the caller's first reset.
    fn new(lesson: Lesson) -> Self {
        let mut task = Self {
            drone: DroneHover::default(),
            lesson,
            random: SplitMix64::new(0),
            pending: None,
        };
        task.reset(Some(0));
        task
    }
}

impl Env for DamageTask {
    type Observation = DroneObservation;
    type Action = DroneAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<DroneObservation> {
        self.drone.reset(seed);
        if let Some(seed) = seed {
            self.random = SplitMix64::new(seed ^ 0x4441_4d41_4745);
        }
        self.pending = None;
        match self.lesson {
            Lesson::Hover => {}
            Lesson::FrontLeft => {
                self.drone
                    .fail_motor(DroneMotor::FrontLeft)
                    .expect("reset flight is active");
            }
            Lesson::Scheduled => {
                let corner = self.random.unit_f64();
                let motor = if corner < 0.25 {
                    DroneMotor::FrontLeft
                } else if corner < 0.5 {
                    DroneMotor::FrontRight
                } else if corner < 0.75 {
                    DroneMotor::RearRight
                } else {
                    DroneMotor::RearLeft
                };
                let actions = if self.random.unit_f64() < 0.5 {
                    100
                } else {
                    250
                };
                self.pending = Some(PendingFailure {
                    motor,
                    remaining: NonZeroU16::new(actions).expect("positive schedule"),
                });
            }
        }
        Reset {
            observation: self.drone.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: DroneAction) -> Step<DroneObservation> {
        let mut result = self.drone.step(action);
        // Termination wins over a scheduled failure; ended physics stays absorbing.
        if result.is_done() {
            return result;
        }
        if let Some(pending) = self.pending.take() {
            if let Some(remaining) = NonZeroU16::new(pending.remaining.get() - 1) {
                self.pending = Some(PendingFailure {
                    remaining,
                    ..pending
                });
            } else {
                self.drone
                    .fail_motor(pending.motor)
                    .expect("continuing flight is active");
                // The next policy command must see health changed at this boundary.
                result.observation = self.drone.observation();
            }
        }
        result
    }
}
