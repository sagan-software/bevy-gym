//! Simulation-time survival score with explicit terminal outcomes.

use super::Game;
use bevy::prelude::*;
use std::time::Duration;

/// A finished run retains its final score until reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Run {
    /// A living player is still pursued by a functioning drone.
    Active(Duration),
    /// The result owns the final elapsed time; further ticks cannot change it.
    Finished {
        /// Simulated time survived, excluding asset loading and paused frames.
        elapsed: Duration,
        /// Event that ended the run.
        outcome: Outcome,
    },
}

/// The first terminal event determines the displayed result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// The drone disabled the player robot.
    RobotDisabled,
    /// Player fire or a crash disabled the drone.
    DroneDisabled,
    /// An inference failure stopped the opponent.
    FlightStopped,
}

impl Default for Run {
    fn default() -> Self {
        Self::Active(Duration::ZERO)
    }
}

impl Run {
    /// Count one fixed twenty-millisecond simulation action while the run remains active.
    pub(super) const fn tick(&mut self) {
        if let Self::Active(elapsed) = self {
            *elapsed = elapsed.saturating_add(Duration::from_millis(20));
        }
    }

    /// Freeze the first outcome without awarding time for later actions.
    pub(super) const fn finish(&mut self, outcome: Outcome) {
        if let Self::Active(elapsed) = *self {
            *self = Self::Finished { elapsed, outcome };
        }
    }

    /// Read either the running clock or the frozen final score.
    pub(super) const fn elapsed(self) -> Duration {
        match self {
            Self::Active(elapsed) | Self::Finished { elapsed, .. } => elapsed,
        }
    }
}

/// Survival clock beneath the scene title.
#[derive(Component)]
pub(super) struct Clock;

/// Keep elapsed time visible without adding another control panel.
pub(super) fn setup(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
    commands.spawn((
        Clock,
        Text::new("Survival 0.0 s"),
        TextFont {
            font: assets.load("fonts/MonaSans-VariableFont.ttf"),
            font_size: 18.0,
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            left: px(16),
            top: px(58),
            padding: UiRect::all(px(8)),
            ..default()
        },
        BackgroundColor(Color::srgb(0.09, 0.11, 0.12)),
    ));
}

/// Report the final result separately from weapon and perception feedback.
pub(super) fn project(game: Res<'_, Game>, mut clocks: Query<'_, '_, &mut Text, With<Clock>>) {
    let seconds = game.run.elapsed().as_secs_f32();
    let label = match game.run {
        Run::Active(_) => "Survival",
        Run::Finished {
            outcome: Outcome::RobotDisabled,
            ..
        } => "Disabled after",
        Run::Finished {
            outcome: Outcome::DroneDisabled,
            ..
        } => "Drone defeated in",
        Run::Finished {
            outcome: Outcome::FlightStopped,
            ..
        } => "Flight stopped at",
    };
    for mut text in &mut clocks {
        text.0 = format!("{label} {seconds:.1} s");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_names_each_outcome_and_formats_simulation_seconds() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .init_resource::<Game>()
            .add_systems(Update, project);
        let clock = app.world_mut().spawn((Clock, Text::new(""))).id();
        let elapsed = Duration::from_millis(1234);
        for (run, expected) in [
            (Run::Active(elapsed), "Survival 1.2 s"),
            (
                Run::Finished {
                    elapsed,
                    outcome: Outcome::RobotDisabled,
                },
                "Disabled after 1.2 s",
            ),
            (
                Run::Finished {
                    elapsed,
                    outcome: Outcome::DroneDisabled,
                },
                "Drone defeated in 1.2 s",
            ),
            (
                Run::Finished {
                    elapsed,
                    outcome: Outcome::FlightStopped,
                },
                "Flight stopped at 1.2 s",
            ),
        ] {
            app.world_mut().resource_mut::<Game>().run = run;
            app.update();
            assert_eq!(app.world().get::<Text>(clock).expect("Clock").0, expected);
        }
    }

    #[test]
    fn first_outcome_freezes_time_and_cannot_be_replaced() {
        for outcome in [
            Outcome::RobotDisabled,
            Outcome::DroneDisabled,
            Outcome::FlightStopped,
        ] {
            let mut run = Run::default();
            run.tick();
            run.finish(outcome);
            let ended = run;
            run.tick();
            run.finish(Outcome::RobotDisabled);
            assert_eq!(run, ended);
            assert_eq!(run.elapsed(), Duration::from_millis(20));
        }
    }
}
