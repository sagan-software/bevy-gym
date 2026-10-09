//! Bounded lesson transitions shared with the qualified native curriculum.

use serde::Serialize;
use serde_json::{json, Value};

use crate::learning::evaluation::EpisodeScore;
use crate::lesson::{self, Lesson};

/// Maximum successful updates in each lesson.
pub(crate) const LESSON_LIMIT: u16 = 600;

/// Closed outcomes emitted by both training modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ProgressStatus {
    /// More optimizer updates are permitted.
    Training,
    /// The selected mode completed its required work.
    Complete,
    /// A curriculum lesson consumed its budget without passing.
    Exhausted,
}

/// Curriculum stages retain only the boundary needed to derive lesson progress.
#[derive(Clone, Copy, Default)]
enum Stage {
    /// Calm hover begins at total update zero.
    #[default]
    Hover,
    /// Disturbed recovery begins after the recorded hover update.
    Recovery(u16),
    /// Both lessons passed; retain recovery's start for progress display.
    Complete(u16),
    /// The named lesson reached its budget without passing.
    Exhausted(Lesson, u16),
}

/// One curriculum and its latest independent evaluation, bounded to five scores.
#[derive(Default)]
pub(crate) struct Curriculum {
    /// Only active stages may consume another update.
    stage: Stage,
    /// Most recent evaluated lesson and its scores; promotion never relabels them.
    evaluation: Option<(Lesson, Vec<EpisodeScore>)>,
}

impl Curriculum {
    /// Current or final lesson, and the total update where it began.
    const fn position(&self) -> (Lesson, u16) {
        match self.stage {
            Stage::Hover => (Lesson::Hover, 0),
            Stage::Recovery(start) | Stage::Complete(start) => (Lesson::Recovery, start),
            Stage::Exhausted(lesson, start) => (lesson, start),
        }
    }

    /// A terminal curriculum keeps its checkpoint but cannot train again.
    pub(crate) const fn status(&self) -> ProgressStatus {
        match self.stage {
            Stage::Hover | Stage::Recovery(_) => ProgressStatus::Training,
            Stage::Complete(_) => ProgressStatus::Complete,
            Stage::Exhausted(_, _) => ProgressStatus::Exhausted,
        }
    }

    /// Select evaluation only after each twentieth update in the active lesson.
    pub(crate) fn due(&self, total: u16) -> Option<Lesson> {
        let (lesson, start) = self.position();
        (self.status() == ProgressStatus::Training && (total - start).is_multiple_of(20))
            .then_some(lesson)
    }

    /// Apply a frozen evaluation. Return a new lesson only when lanes must reset.
    pub(crate) fn finish(&mut self, total: u16, scores: Vec<EpisodeScore>) -> Option<Lesson> {
        if self.status() != ProgressStatus::Training {
            return None;
        }
        let (lesson, start) = self.position();
        let passed = lesson::passes(&scores);
        self.evaluation = Some((lesson, scores));
        if passed {
            match lesson {
                Lesson::Hover => {
                    self.stage = Stage::Recovery(total);
                    return Some(Lesson::Recovery);
                }
                Lesson::Recovery => self.stage = Stage::Complete(start),
            }
        } else if total - start == LESSON_LIMIT {
            self.stage = Stage::Exhausted(lesson, start);
        }
        None
    }

    /// Project derived progress and lossless decimal-string seeds to the host.
    pub(crate) fn progress(&self, total: u16) -> Value {
        let (lesson, start) = self.position();
        let evaluation = self.evaluation.as_ref().map(|(evaluated, scores)| {
            json!({"lesson": evaluated.name(), "passed": lesson::passes(scores),
                "episodes": scores.iter().map(super::protocol::episode_json).collect::<Vec<_>>()})
        });
        json!({"lesson": lesson.name(), "lesson_updates": total - start,
            "lesson_limit": LESSON_LIMIT, "evaluation": evaluation})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::SELECTION_SEEDS;

    /// Five independent scores at the inclusive native promotion thresholds.
    fn scores(passed: bool) -> Vec<EpisodeScore> {
        SELECTION_SEEDS
            .into_iter()
            .map(|seed| EpisodeScore {
                seed,
                steps: 500,
                reward: if passed { 400.0 } else { 399.0 },
                final_distance: 0.5,
                survived: true,
            })
            .collect()
    }

    #[test]
    fn promotion_resets_the_lesson_count_and_keeps_the_evaluated_lesson() {
        let mut course = Curriculum::default();
        assert_eq!(course.status(), ProgressStatus::Training);
        assert!(course.due(19).is_none());
        assert!(matches!(course.due(20), Some(Lesson::Hover)));
        assert!(course.finish(20, scores(false)).is_none());
        assert_eq!(course.progress(20)["evaluation"]["passed"], false);
        assert!(matches!(
            course.finish(40, scores(true)),
            Some(Lesson::Recovery)
        ));
        let progress = course.progress(40);
        assert_eq!(progress["lesson"], "recovery");
        assert_eq!(progress["lesson_updates"], 0);
        assert_eq!(progress["evaluation"]["lesson"], "hover");
        assert_eq!(
            progress["evaluation"]["episodes"][4]["seed"],
            "18446744073709551615"
        );
    }

    #[test]
    fn recovery_completion_is_terminal() {
        let mut course = Curriculum::default();
        course.finish(40, scores(true));
        assert!(course.due(59).is_none());
        assert!(matches!(course.due(60), Some(Lesson::Recovery)));
        assert!(course.finish(60, scores(true)).is_none());
        assert_eq!(course.status(), ProgressStatus::Complete);
        assert!(course.due(80).is_none());
        assert_eq!(course.progress(60)["lesson_updates"], 20);
        let before = course.progress(60);
        assert!(course.finish(60, scores(false)).is_none());
        assert_eq!(course.progress(60), before);
    }

    #[test]
    fn hover_budget_exhaustion_is_terminal() {
        let mut hover = Curriculum::default();
        assert!(hover.finish(600, scores(false)).is_none());
        assert_eq!(hover.status(), ProgressStatus::Exhausted);
        assert!(hover.due(600).is_none());
        assert_eq!(hover.progress(600)["lesson"], "hover");
        assert_eq!(hover.progress(600)["lesson_updates"], 600);
        let before = hover.progress(600);
        assert!(hover.finish(600, scores(true)).is_none());
        assert_eq!(hover.progress(600), before);
    }

    #[test]
    fn recovery_budget_exhausts_after_hover_passes_at_its_limit() {
        let mut recovery = Curriculum::default();
        assert!(matches!(
            recovery.finish(600, scores(true)),
            Some(Lesson::Recovery)
        ));
        assert_eq!(recovery.status(), ProgressStatus::Training);
        assert!(recovery.finish(1200, scores(false)).is_none());
        assert_eq!(recovery.status(), ProgressStatus::Exhausted);
        assert_eq!(recovery.progress(1200)["lesson"], "recovery");
        assert_eq!(recovery.progress(1200)["lesson_updates"], 600);
    }

    #[test]
    fn passing_at_both_limits_completes_the_course() {
        let mut complete = Curriculum::default();
        complete.finish(600, scores(true));
        complete.finish(1200, scores(true));
        assert_eq!(complete.status(), ProgressStatus::Complete);
    }
}
