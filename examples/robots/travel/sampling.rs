//! Training-only reset recipes over the unchanged travel environment.

use super::{environment::TravelTask, stage::Stage};
use bevy_gym::robots::{DroneAction, DroneTravelObservation};
use bevy_gym::training::{SeedConfig, SplitMix64};
use bevy_gym::{Env, Reset, Step};

/// Closed training recipes; frozen evaluators always use `TravelTask` directly.
#[derive(Clone, Copy)]
pub(crate) enum Recipe {
    /// Preserve original task sampling and artifact output.
    Original,
    /// Rehearse prerequisites without changing inference or evaluation distributions.
    Rehearsal,
}

impl Recipe {
    /// Supply noncapturing factories without retaining reset closures or allocating adapters.
    pub(crate) fn factory(self, stage: Stage) -> fn() -> TrainingTask {
        match (self, stage) {
            (Self::Original, Stage::Endurance) => {
                || TrainingTask::new(Stage::Endurance, Self::Original)
            }
            (Self::Original, Stage::Near) => || TrainingTask::new(Stage::Near, Self::Original),
            (Self::Original, Stage::Far) => || TrainingTask::new(Stage::Far, Self::Original),
            (Self::Original, Stage::Fast) => || TrainingTask::new(Stage::Fast, Self::Original),
            (Self::Rehearsal, Stage::Endurance) => {
                || TrainingTask::new(Stage::Endurance, Self::Rehearsal)
            }
            (Self::Rehearsal, Stage::Near) => || TrainingTask::new(Stage::Near, Self::Rehearsal),
            (Self::Rehearsal, Stage::Far) => || TrainingTask::new(Stage::Far, Self::Rehearsal),
            (Self::Rehearsal, Stage::Fast) => || TrainingTask::new(Stage::Fast, Self::Rehearsal),
        }
    }
}

/// Confine task mixtures to rollout collection; actor and evaluator contracts remain unchanged.
pub(crate) struct TrainingTask {
    /// Own the original task, reset root, episode counter and authoritative physical world.
    task: TravelTask,
    /// Choose one closed training recipe without duplicating reset state.
    recipe: Recipe,
}

impl TrainingTask {
    /// Construct episode zero; rehearsal performs one additional bounded reset.
    fn new(stage: Stage, recipe: Recipe) -> Self {
        let mut task = Self {
            task: TravelTask::factory(stage)(),
            recipe,
        };
        if matches!(recipe, Recipe::Rehearsal) {
            task.reset(Some(0));
        }
        task
    }
}

impl Env for TrainingTask {
    type Observation = DroneTravelObservation;
    type Action = DroneAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation> {
        match self.recipe {
            Recipe::Original => self.task.reset(seed),
            Recipe::Rehearsal => self.task.reset_episode(seed, select_rehearsal_stage),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        self.task.step(action)
    }
}

/// Derive a task from channel two without consuming body or destination draws.
fn select_rehearsal_stage(stage: Stage, seeds: SeedConfig, episode: u64) -> Stage {
    let mut random = SplitMix64::new(seeds.environment_episode(2, episode));
    rehearsal_stage(stage, random.unit_f64())
}

/// Assign half the unit interval to the current task and divide the rest among prerequisites.
///
/// The supplied draw comes only from the generator's finite `[0, 1)` grid. This constant-time
/// partition allocates no memory and never selects a future stage.
fn rehearsal_stage(stage: Stage, draw: f64) -> Stage {
    match stage {
        Stage::Endurance => Stage::Endurance,
        Stage::Near if draw < 0.5 => Stage::Endurance,
        Stage::Far if draw < 0.25 => Stage::Endurance,
        Stage::Far if draw < 0.5 => Stage::Near,
        Stage::Fast if draw < 1.0 / 6.0 => Stage::Endurance,
        Stage::Fast if draw < 1.0 / 3.0 => Stage::Near,
        Stage::Fast if draw < 0.5 => Stage::Far,
        Stage::Near | Stage::Far | Stage::Fast => stage,
    }
}

#[cfg(test)]
mod tests {
    use super::{rehearsal_stage, Stage};

    /// Exact partition boundaries retain the current half and every prerequisite interval.
    #[test]
    fn rehearsal_partition_has_closed_stage_boundaries() {
        for (stage, draw, expected) in [
            (Stage::Endurance, 0.0, Stage::Endurance),
            (Stage::Endurance, 0.999, Stage::Endurance),
            (Stage::Near, 0.0, Stage::Endurance),
            (Stage::Near, 0.499, Stage::Endurance),
            (Stage::Near, 0.5, Stage::Near),
            (Stage::Near, 0.999, Stage::Near),
            (Stage::Far, 0.249, Stage::Endurance),
            (Stage::Far, 0.25, Stage::Near),
            (Stage::Far, 0.499, Stage::Near),
            (Stage::Far, 0.5, Stage::Far),
            (Stage::Far, 0.999, Stage::Far),
            (Stage::Fast, 0.166, Stage::Endurance),
            (Stage::Fast, 1.0 / 6.0, Stage::Near),
            (Stage::Fast, 0.333, Stage::Near),
            (Stage::Fast, 1.0 / 3.0, Stage::Far),
            (Stage::Fast, 0.499, Stage::Far),
            (Stage::Fast, 0.5, Stage::Fast),
            (Stage::Fast, 0.999, Stage::Fast),
        ] {
            assert_eq!(rehearsal_stage(stage, draw), expected);
        }
    }
}
