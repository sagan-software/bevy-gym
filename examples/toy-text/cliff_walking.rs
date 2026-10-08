//! Cliff walking involves crossing a gridworld from start to goal while avoiding falling off a cliff.
//!
//! ## Description
//! The game starts with the player at location [3, 0] of the 4x12 grid world with the
//! goal located at [3, 11]. If the player reaches the goal the episode ends.
//!
//! A cliff runs along [3, 1..10]. If the player moves to a cliff location it
//! returns to the start location.
//!
//! The player makes moves until they reach the goal.
//!
//! Adapted from Example 6.6 (page 132) from Reinforcement Learning: An Introduction
//! by Sutton and Barto [<a href="#cliffwalk_ref">1</a>].
//!
//! The cliff can be chosen to be slippery (disabled by default) so the player may move perpendicular
//! to the intended direction sometimes (see <a href="#is_slippy">`is_slippery`</a>).
//!
//! With inspiration from:
//! [https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py](https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)
//!
//! ## Action Space
//! The action shape is `(1,)` in the range `{0, 3}` indicating
//! which direction to move the player.
//!
//! - 0: Move up
//! - 1: Move right
//! - 2: Move down
//! - 3: Move left
//!
//! ## Observation Space
//! There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at
//! the goal as the latter results in the end of the episode. What remains are all
//! the positions of the first 3 rows plus the bottom-left cell.
//!
//! The observation is a value representing the player's current position as
//! `current_row` * ncols + `current_col` (where both the row and col start at 0).
//!
//! For example, the starting position can be calculated as follows: 3 * 12 + 0 = 36.
//!
//! The observation is returned as an `int()`.
//!
//! ## Starting State
//! The episode starts with the player in state `[36]` (location [3, 0]).
//!
//! ## Reward
//! Each time step incurs -1 reward, unless the player stepped into the cliff,
//! which incurs -100 reward.
//!
//! ## Episode End
//! The episode terminates when the player enters state `[47]` (location [3, 11]).
//!
//! ## Information
//!
//! `step()` and `reset()` return a dict with the following keys:
//! - "p" - transition proability for the state.
//!
//! As cliff walking is not stochastic, the transition probability returned always 1.0.
//!
//! ## Arguments
//!
//! ```python
//! import gymnasium as gym
//! gym.make('CliffWalking-v1')
//! ```
//!
//! ## References
//! <a id="cliffwalk_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
//! An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)
//!
//! ## Version History
//! - v1: Add slippery version of cliffwalking
//! - v0: Initial version release

use bevy_gym::{Env, EpisodeStatus, Reset, Step};
use std::error::Error;

use std::path::Path;

use bevy_gym::training::{
    run_tabular_workflow, IndexedAction, TabularEvaluation as Evaluation, TabularExample,
};

/// Gymnasium's `CliffWalking` action order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CliffDirection {
    /// Move one row up.
    Up,
    /// Move one column right.
    Right,
    /// Move one row down.
    Down,
    /// Move one column left.
    Left,
}

impl CliffDirection {
    /// Convert a tabular action index into the closed action vocabulary.
    const fn from_index(index: usize) -> Self {
        match index % 4 {
            0 => Self::Up,
            1 => Self::Right,
            2 => Self::Down,
            _ => Self::Left,
        }
    }
}

/// Deterministic transition probability.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct CliffInfo {
    /// Probability of the selected transition.
    probability: f64,
}

/// Gymnasium CliffWalking-v1 with default non-slippery movement.
#[derive(Debug, Clone)]
struct CliffWalking {
    /// Current row-major 4x12 grid index.
    state: usize,
    /// Last action for the directional elf sprite.
    last_action: Option<CliffDirection>,
}

impl Default for CliffWalking {
    fn default() -> Self {
        Self {
            state: 36,
            last_action: None,
        }
    }
}

impl CliffWalking {
    /// Apply one grid-bounded move before cliff and goal handling.
    fn moved_state(&self, action: CliffDirection) -> usize {
        let row = self.state / 12;
        let column = self.state % 12;
        match action {
            CliffDirection::Up => row.saturating_sub(1) * 12 + column,
            CliffDirection::Right => row * 12 + column.saturating_add(1).min(11),
            CliffDirection::Down => row.saturating_add(1).min(3) * 12 + column,
            CliffDirection::Left => row * 12 + column.saturating_sub(1),
        }
    }
}

impl Env for CliffWalking {
    type Observation = usize;
    type Action = CliffDirection;
    type Info = CliffInfo;

    fn reset(&mut self, _seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        self.state = 36;
        self.last_action = None;
        Reset {
            observation: self.state,
            info: CliffInfo { probability: 1.0 },
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let moved = self.moved_state(action);
        let fell = (37..=46).contains(&moved);
        self.state = if fell { 36 } else { moved };
        self.last_action = Some(action);

        Step {
            observation: self.state,
            reward: if fell { -100.0 } else { -1.0 },
            status: if self.state == 47 {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            },
            info: CliffInfo { probability: 1.0 },
        }
    }
}

impl IndexedAction for CliffDirection {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }
}

impl TabularExample for CliffWalking {
    const ENV_NAME: &'static str = "cliff-walking";
    const GYMNASIUM_ID: &'static str = "CliffWalking-v1";
    const STATE_COUNT: usize = 48;
    const ACTION_COUNT: usize = 4;
    const MAX_STEPS: usize = 100;
    const DEFAULT_EPISODES: usize = 1_000;
    const DEFAULT_EVAL_INTERVAL: usize = 250;
    const DEFAULT_EVAL_EPISODES: usize = 100;
    const DEFAULT_LEARNING_RATE: f64 = 0.8;
    const GAMMA: f64 = 0.99;
    const SOLVED_SCORE: f64 = -13.0;
    const GIF_PATH: &'static str = "docs/images/cliff-walking.gif";

    fn selection_score(evaluation: &Evaluation) -> f64 {
        evaluation.mean_reward
    }

    fn is_success(final_observation: usize, _total_reward: f64, status: EpisodeStatus) -> bool {
        final_observation == 47 && status == EpisodeStatus::Terminated
    }

    fn watch(checkpoint: &Path) -> Result<(), Box<dyn Error>> {
        #[cfg(not(feature = "render"))]
        let _ = checkpoint;
        #[cfg(feature = "render")]
        return render::run_visual(checkpoint, None);
        #[cfg(not(feature = "render"))]
        return Err("watch mode requires the render feature".into());
    }

    fn gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        #[cfg(not(feature = "render"))]
        let _ = (checkpoint, output);
        #[cfg(feature = "render")]
        return render::render_gif(checkpoint, output);
        #[cfg(not(feature = "render"))]
        return Err("GIF mode requires the render feature".into());
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    run_tabular_workflow::<CliffWalking>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{CliffDirection, CliffWalking, Env, Error, Path};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::{TabularQPolicy, TabularQTrainer};

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Policy and deterministic environment used by the rendered demonstration.
    #[derive(Resource)]
    struct VisualCliff {
        /// Greedy validation-selected policy.
        policy: TabularQPolicy,
        /// Visible environment.
        env: CliffWalking,
        /// Current table state.
        state: usize,
    }

    /// Exact Gymnasium cliff and character sprites.
    #[derive(Resource)]
    struct CliffImages {
        /// Elf facing up.
        up: Handle<Image>,
        /// Elf facing right.
        right: Handle<Image>,
        /// Elf facing down.
        down: Handle<Image>,
        /// Elf facing left.
        left: Handle<Image>,
    }

    impl CliffImages {
        /// Select the sprite for the requested action.
        fn elf(&self, action: Option<CliffDirection>) -> Handle<Image> {
            match action.unwrap_or(CliffDirection::Down) {
                CliffDirection::Up => self.up.clone(),
                CliffDirection::Right => self.right.clone(),
                CliffDirection::Down => self.down.clone(),
                CliffDirection::Left => self.left.clone(),
            }
        }
    }

    /// Visible walker sprite.
    #[derive(Component)]
    struct CliffElf;

    /// Four-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let trainer = TabularQTrainer::load(checkpoint)?;
        let mut env = CliffWalking::default();
        let state = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualCliff {
            policy: trainer.policy(),
            env,
            state,
        })
        .insert_resource(VisualClock(Timer::from_seconds(0.25, TimerMode::Repeating)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "examples/toy-text/assets".into(),
                    ..default()
                })
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym CliffWalking-v1".into(),
                        resolution: WindowResolution::new(720, 240),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_cliff_scene);

        if let Some(directory) = capture_dir {
            app.insert_resource(GifCapture::new(directory))
                .add_systems(Update, capture_gif_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_elf).chain());
        }

        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    #[cfg(not(feature = "render"))]
    /// Report the explicit render requirement in headless builds.
    pub(super) fn run_visual(
        _checkpoint: &Path,
        _capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    /// Build Gymnasium's 12-by-4 cliff composition from first-party assets.
    fn setup_cliff_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("CliffWalking Camera")));
        let backgrounds = [
            assets.load("img/mountain_bg1.png"),
            assets.load("img/mountain_bg2.png"),
        ];
        let near_cliff = [
            assets.load("img/mountain_near-cliff1.png"),
            assets.load("img/mountain_near-cliff2.png"),
        ];
        let cliff = assets.load("img/mountain_cliff.png");
        let stool = assets.load("img/stool.png");
        let cookie = assets.load("img/cookie.png");
        let images = CliffImages {
            up: assets.load("img/elf_up.png"),
            right: assets.load("img/elf_right.png"),
            down: assets.load("img/elf_down.png"),
            left: assets.load("img/elf_left.png"),
        };

        for state in 0..48 {
            let row = state / 12;
            let column = state % 12;
            let position = cliff_cell_position(state);
            let checker = (row % 2) ^ (column % 2);
            commands.spawn((
                Sprite {
                    image: backgrounds
                        .get(checker)
                        .cloned()
                        .expect("fixed example index is valid"),
                    custom_size: Some(Vec2::splat(60.0)),
                    ..default()
                },
                Transform::from_xyz(position.x, position.y, 0.0),
            ));

            let overlay = if (37..=46).contains(&state) {
                Some(cliff.clone())
            } else if row == 2 && (1..=10).contains(&column) {
                Some(
                    near_cliff
                        .get(checker)
                        .cloned()
                        .expect("fixed example index is valid"),
                )
            } else if state == 36 {
                Some(stool.clone())
            } else if state == 47 {
                Some(cookie.clone())
            } else {
                None
            };
            if let Some(image) = overlay {
                commands.spawn((
                    Sprite {
                        image,
                        custom_size: Some(Vec2::splat(60.0)),
                        ..default()
                    },
                    Transform::from_xyz(position.x, position.y, 1.0),
                ));
            }
        }

        let initial = cliff_cell_position(36);
        commands.spawn((
            Sprite {
                image: images.elf(None),
                custom_size: Some(Vec2::splat(60.0)),
                ..default()
            },
            Transform::from_xyz(initial.x, initial.y + 6.0, 2.0),
            CliffElf,
            Name::new("CliffWalking Elf"),
        ));
        commands.insert_resource(images);
    }

    /// Advance policy playback at Gymnasium's four frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualCliff>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Advance one greedy action or reset after the goal.
    fn advance_visual(visual: &mut VisualCliff) {
        if visual.state == 47 {
            visual.state = visual.env.reset(None).observation;
            return;
        }
        let Ok(action_index) = visual.policy.greedy_action(visual.state) else {
            return;
        };
        let transition = visual.env.step(CliffDirection::from_index(action_index));
        visual.state = transition.observation;
    }

    /// Synchronize the elf sprite and grid position.
    fn sync_elf(
        visual: Res<'_, VisualCliff>,
        images: Res<'_, CliffImages>,
        mut elf: Single<'_, '_, (&mut Transform, &mut Sprite), With<CliffElf>>,
    ) {
        let position = cliff_cell_position(visual.state);
        elf.0.translation.x = position.x;
        elf.0.translation.y = position.y + 6.0;
        elf.1.image = images.elf(visual.env.last_action);
    }

    /// Capture 100 frames at 20 FPS while advancing the policy at 4 Hz.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualCliff>,
        images: Res<'_, CliffImages>,
        mut elf: Single<'_, '_, (&mut Transform, &mut Sprite), With<CliffElf>>,
        captures: Query<'_, '_, Entity, With<Capturing>>,
        mut exit: MessageWriter<'_, AppExit>,
    ) {
        if !captures.is_empty() {
            return;
        }
        if capture.is_warming_up() {
            return;
        }
        if capture.is_complete() {
            exit.write(AppExit::Success);
            return;
        }
        if capture.has_started() && capture.frame_index().is_multiple_of(5) {
            advance_visual(&mut visual);
        }
        let position = cliff_cell_position(visual.state);
        elf.0.translation.x = position.x;
        elf.0.translation.y = position.y + 6.0;
        elf.1.image = images.elf(visual.env.last_action);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Return the center of one 60-pixel Gymnasium grid cell.
    const fn cliff_cell_position(state: usize) -> Vec2 {
        let row = state / 12;
        let column = state % 12;
        Vec2::new(
            (column as f32).mul_add(60.0, -330.0),
            (row as f32).mul_add(-60.0, 90.0),
        )
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "cliff-walking", 720, 240, |frames| {
            run_visual(checkpoint, Some(frames))
        })
    }

    #[cfg(not(feature = "render"))]
    /// Report the explicit render requirement in headless builds.
    pub(super) fn render_gif(_checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        Err("GIF mode requires the default `render` feature".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cliff_transition_returns_to_start_with_minus_one_hundred_reward() {
        let mut cliff = CliffWalking::default();
        assert_eq!(cliff.reset(Some(5)).observation, 36);

        let step = cliff.step(CliffDirection::Right);

        assert_eq!(step.observation, 36);
        assert_eq!(step.reward, -100.0);
        assert_eq!(step.status, EpisodeStatus::Continuing);
    }

    #[test]
    fn shortest_safe_route_finishes_in_thirteen_steps_for_minus_thirteen() {
        let mut cliff = CliffWalking::default();
        cliff.reset(Some(7));
        let mut reward = 0.0;
        let mut final_step = cliff.step(CliffDirection::Up);
        reward += final_step.reward;
        for _ in 0..11 {
            final_step = cliff.step(CliffDirection::Right);
            reward += final_step.reward;
        }
        final_step = cliff.step(CliffDirection::Down);
        reward += final_step.reward;

        assert_eq!(final_step.observation, 47);
        assert_eq!(final_step.status, EpisodeStatus::Terminated);
        assert_eq!(reward, -13.0);
    }
}
