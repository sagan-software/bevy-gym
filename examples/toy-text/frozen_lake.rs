//!  Frozen lake involves crossing a frozen lake from start to goal without falling into any holes
//!  by walking over the frozen lake.
//!  The player may not always move in the intended direction due to the slippery nature of the frozen lake.
//!
//!  ## Description
//!  The game starts with the player at location `[0,0]` of the frozen lake grid world with the
//!  goal located at far extent of the world e.g. `[3,3]` for the 4x4 environment.
//!
//!  Holes in the ice are distributed in set locations when using a pre-determined map
//!  or in random locations when a random map is generated.
//!  Randomly generated worlds will always have a path to the goal.
//!
//!  The player makes moves until they reach the goal or fall in a hole.
//!
//!  The lake is slippery (unless disabled) so the player may move perpendicular
//!  to the intended direction sometimes (see `is_slippery` in Argument section).
//!
//!  Elf and stool from [https://franuka.itch.io/rpg-snow-tileset](https://franuka.itch.io/rpg-snow-tileset).
//!  All other assets by Mel Tillery [http://www.cyaneus.com/](http://www.cyaneus.com/).
//!
//!  ## Action Space
//!  The action shape is `(1,)` in the range `{0, 3}` indicating
//!  which direction to move the player.
//!
//!  - 0: Move left
//!  - 1: Move down
//!  - 2: Move right
//!  - 3: Move up
//!
//!  ## Observation Space
//!  The observation is a value representing the player's current position as
//!  `current_row * ncols + current_col` (where both the row and col start at 0).
//!  Therefore, the observation is returned as an integer.
//!
//!  For example, the goal position in the 4x4 map can be calculated as follows: 3 * 4 + 3 = 15.
//!  The number of possible observations is dependent on the size of the map.
//!
//!  ## Starting State
//!  The episode starts with the player in state `[0]` (location [0, 0]).
//!
//!  ## Rewards
//!
//!  Default reward schedule:
//!  - Reach goal: +1
//!  - Reach hole: 0
//!  - Reach frozen: 0
//!
//!  See `reward_schedule` for reward customization in the Argument section.
//!
//!  ## Episode End
//!  The episode ends if the following happens:
//!
//!  - Termination:
//!      1. The player moves into a hole.
//!      2. The player reaches the goal at `max(nrow) * max(ncol) - 1` (location `[max(nrow)-1, max(ncol)-1]`).
//!
//!  - Truncation (using the `time_limit` wrapper):
//!      1. The length of the episode is 100 for `FrozenLake4x4`, 200 for `FrozenLake8x8`.
//!
//!  ## Information
//!
//!  `step()` and `reset()` return a dict with the following keys:
//!  - `p`: transition probability for the state which will be impacted by the `is_slippery` parameter.
//!
//!  ## Arguments
//!
//!  `FrozenLake` has five parameters:
//!  ```python
//!  import gymnasium as gym
//!  gym.make(
//!      'FrozenLake-v1',
//!      desc=None,
//!      map_name="4x4",
//!      is_slippery=True,
//!      success_rate=1.0/3.0,
//!      reward_schedule=(1, 0, 0)
//!  )
//!  ```
//!
//!  * `desc=None`: Used to specify maps non-preloaded maps.
//!    If `desc=None` then `map_name` will be used. If both `desc` and `map_name` are
//!    `None` a random 8x8 map with 80% of locations frozen will be generated.
//!
//!      To Specify a custom map - `desc=["SFFF", "FHFH", "FFFH", "HFFG"]`
//!    The tile letters denote:
//!      - "S" for Start tile
//!      - "G" for Goal tile
//!      - "F" for frozen tile
//!      - "H" for a tile with a hole
//!
//!      A random generated map can be specified by calling the function `generate_random_map`.
//!      ```
//!      from gymnasium.envs.toy_text.frozen_lake import generate_random_map
//!
//!      gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
//!      ```
//!
//!  * `map_name="4x4"` - Helps load two predefined map names (`4x4` and `8x8`)
//!      ```
//!      "4x4":[
//!          "SFFF",
//!          "FHFH",
//!          "FFFH",
//!          "HFFG"
//!      ]
//!
//!      "8x8": [
//!          "SFFFFFFF",
//!          "FFFFFFFF",
//!          "FFFHFFFF",
//!          "FFFFFHFF",
//!          "FFFHFFFF",
//!          "FHHFFFHF",
//!          "FHFFHFHF",
//!          "FFFHFFFG",
//!      ]
//!      ```
//!
//! * `is_slippery=True`: If true the player will move in intended direction with probability specified by the
//!   `success_rate` else will move in either perpendicular direction with equal probability in both directions.
//!
//!      For example, if action is left, `is_slippery` is True, and `success_rate` is 1/3, then:
//!      - P(move left)=1/3
//!      - P(move up)=1/3
//!      - P(move down)=1/3
//!
//!      If action is up, `is_slippery` is True, and `success_rate` is 3/4, then:
//!      - P(move up)=3/4
//!      - P(move left)=1/8
//!      - P(move right)=1/8
//!
//! * `success_rate=1.0/3.0`: Used to specify the probability of moving in the intended direction when `is_slippery=True`
//!
//! * `reward_schedule=(1, 0, 0)`: Used to specify reward amounts for reaching certain tiles.
//!   The indices correspond to: Reach Goal, Reach Hole, Reach Frozen (includes Start), Respectively
//!
//!  ## Version History
//!  * v1: Bug fixes to rewards (v1.3, added reward customization)
//!  * v0: Initial version release

use shakmaty as _;
use tokio as _;

use bevy_gym::{Env, EpisodeStatus, Reset, Step};
use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::{
    run_tabular_workflow, IndexedAction, TabularEvaluation as Evaluation, TabularExample,
};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use serde_json as _;

/// Gymnasium's four discrete movement actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// Move one column left.
    Left,
    /// Move one row down.
    Down,
    /// Move one column right.
    Right,
    /// Move one row up.
    Up,
}

impl Direction {
    /// Convert a wrapped action index back into the closed action vocabulary.
    const fn from_index(index: usize) -> Self {
        match index % 4 {
            0 => Self::Left,
            1 => Self::Down,
            2 => Self::Right,
            _ => Self::Up,
        }
    }

    /// Return the Gymnasium action index.
    const fn as_index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Down => 1,
            Self::Right => 2,
            Self::Up => 3,
        }
    }
}

/// Transition probability returned by Gymnasium's `info["prob"]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct FrozenLakeInfo {
    /// Probability of the sampled transition.
    probability: f64,
}

/// Gymnasium FrozenLake-v1 on its canonical 4x4 map.
#[derive(Debug, Clone)]
struct FrozenLake {
    /// Current row-major cell index.
    state: usize,
    /// Last requested direction, used by the renderer.
    last_action: Option<Direction>,
    /// Whether actions can slip to either perpendicular direction.
    is_slippery: bool,
    /// Deterministic environment-owned transition generator.
    rng: SplitMix64,
}

impl FrozenLake {
    /// Construct the canonical stochastic environment.
    const fn slippery() -> Self {
        Self {
            state: 0,
            last_action: None,
            is_slippery: true,
            rng: SplitMix64::new(0),
        }
    }

    /// Construct the canonical deterministic diagnostic environment.
    #[cfg(test)]
    const fn deterministic() -> Self {
        Self {
            state: 0,
            last_action: None,
            is_slippery: false,
            rng: SplitMix64::new(0),
        }
    }

    /// Apply one bounded move on the four-by-four map.
    fn next_state(state: usize, action: Direction) -> usize {
        let row = state / 4;
        let column = state % 4;
        match action {
            Direction::Left => row * 4 + column.saturating_sub(1),
            Direction::Down => row.saturating_add(1).min(3) * 4 + column,
            Direction::Right => row * 4 + column.saturating_add(1).min(3),
            Direction::Up => row.saturating_sub(1) * 4 + column,
        }
    }

    /// Sample the requested direction or one of its perpendicular slips.
    fn sampled_action(&mut self, requested: Direction) -> Direction {
        if !self.is_slippery {
            return requested;
        }

        let offset = self.rng.index(3);
        Direction::from_index((requested.as_index() + offset + 3) % 4)
    }
}

impl Default for FrozenLake {
    fn default() -> Self {
        Self::slippery()
    }
}

impl Env for FrozenLake {
    type Observation = usize;
    type Action = Direction;
    type Info = FrozenLakeInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state = 0;
        self.last_action = None;
        Reset {
            observation: self.state,
            info: FrozenLakeInfo { probability: 1.0 },
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let sampled = self.sampled_action(action);
        self.state = Self::next_state(self.state, sampled);
        self.last_action = Some(action);
        let terminal = matches!(self.state, 5 | 7 | 11 | 12 | 15);

        Step {
            observation: self.state,
            reward: f64::from(self.state == 15),
            status: if terminal {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Continuing
            },
            info: FrozenLakeInfo {
                probability: if self.is_slippery { 1.0 / 3.0 } else { 1.0 },
            },
        }
    }
}

/// Small deterministic generator used only for environment transition sampling.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Generator state.
    state: u64,
}

impl SplitMix64 {
    /// Construct a generator from one root seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return one uniformly distributed index below `upper`.
    fn index(&mut self, upper: usize) -> usize {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let mixed = value ^ (value >> 31);
        usize::try_from(mixed % u64::try_from(upper).unwrap_or(1)).unwrap_or_default()
    }
}

impl IndexedAction for Direction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }
}

impl TabularExample for FrozenLake {
    const ENV_NAME: &'static str = "frozen-lake";
    const GYMNASIUM_ID: &'static str = "FrozenLake-v1";
    const STATE_COUNT: usize = 16;
    const ACTION_COUNT: usize = 4;
    const MAX_STEPS: usize = 100;
    const DEFAULT_EPISODES: usize = 50_000;
    const DEFAULT_EVAL_INTERVAL: usize = 5_000;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const DEFAULT_LEARNING_RATE: f64 = 0.1;
    const GAMMA: f64 = 0.95;
    const SOLVED_SCORE: f64 = 0.70;
    const GIF_PATH: &'static str = "docs/images/frozen-lake.gif";

    fn selection_score(evaluation: &Evaluation) -> f64 {
        evaluation.success_rate
    }

    fn is_success(_final_observation: usize, total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Terminated && total_reward > 0.0
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
    run_tabular_workflow::<FrozenLake>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{Direction, Env, Error, FrozenLake, Path};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::{TabularQPolicy, TabularQTrainer};
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Policy and environment state owned by the visual demonstration.
    #[derive(Resource)]
    struct VisualLake {
        /// Greedy validation-selected policy.
        policy: TabularQPolicy,
        /// Visible environment instance.
        env: FrozenLake,
        /// Current observation encoded for the table.
        state: usize,
        /// Reset seed advanced after each terminal state.
        seed: u64,
    }

    /// Exact Gymnasium sprite handles.
    #[derive(Resource)]
    struct LakeImages {
        /// Elf facing left.
        elf_left: Handle<Image>,
        /// Elf facing down.
        elf_down: Handle<Image>,
        /// Elf facing right.
        elf_right: Handle<Image>,
        /// Elf facing up.
        elf_up: Handle<Image>,
        /// Cracked-hole terminal sprite.
        cracked_hole: Handle<Image>,
    }

    impl LakeImages {
        /// Select the elf sprite for the most recently requested action.
        fn elf(&self, action: Option<Direction>) -> Handle<Image> {
            match action.unwrap_or(Direction::Down) {
                Direction::Left => self.elf_left.clone(),
                Direction::Down => self.elf_down.clone(),
                Direction::Right => self.elf_right.clone(),
                Direction::Up => self.elf_up.clone(),
            }
        }
    }

    /// Visible elf entity.
    #[derive(Component)]
    struct Elf;

    /// Four-Hz watch playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let trainer = TabularQTrainer::load(checkpoint)?;
        let mut env = FrozenLake::slippery();
        let state = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualLake {
            policy: trainer.policy(),
            env,
            state,
            seed: 12_346,
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
                        title: "bevy-gym FrozenLake-v1".into(),
                        resolution: WindowResolution::new(256, 256),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_lake_scene);

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

    /// Build the exact 256-by-256 Gymnasium tile composition from first-party assets.
    fn setup_lake_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("FrozenLake Camera")));
        let ice = assets.load("img/ice.png");
        let hole = assets.load("img/hole.png");
        let goal = assets.load("img/goal.png");
        let stool = assets.load("img/stool.png");
        let images = LakeImages {
            elf_left: assets.load("img/elf_left.png"),
            elf_down: assets.load("img/elf_down.png"),
            elf_right: assets.load("img/elf_right.png"),
            elf_up: assets.load("img/elf_up.png"),
            cracked_hole: assets.load("img/cracked_hole.png"),
        };

        for state in 0..16 {
            let position = cell_position(state);
            commands.spawn((
                Sprite {
                    image: ice.clone(),
                    custom_size: Some(Vec2::splat(64.0)),
                    ..default()
                },
                Transform::from_xyz(position.x, position.y, 0.0),
                Name::new(format!("FrozenLake Ice {state}")),
            ));

            let overlay = match state {
                0 => Some(stool.clone()),
                5 | 7 | 11 | 12 => Some(hole.clone()),
                15 => Some(goal.clone()),
                _ => None,
            };
            if let Some(image) = overlay {
                commands.spawn((
                    Sprite {
                        image,
                        custom_size: Some(Vec2::splat(64.0)),
                        ..default()
                    },
                    Transform::from_xyz(position.x, position.y, 1.0),
                    Name::new(format!("FrozenLake Overlay {state}")),
                ));
            }

            let border = Color::srgb_u8(180, 200, 230);
            commands.spawn((
                Sprite::from_color(border, Vec2::new(64.0, 1.0)),
                Transform::from_xyz(position.x, position.y + 31.5, 1.5),
            ));
            commands.spawn((
                Sprite::from_color(border, Vec2::new(1.0, 64.0)),
                Transform::from_xyz(position.x - 31.5, position.y, 1.5),
            ));
        }

        let initial = cell_position(0);
        commands.spawn((
            Sprite {
                image: images.elf(None),
                custom_size: Some(Vec2::splat(64.0)),
                ..default()
            },
            Transform::from_xyz(initial.x, initial.y, 2.0),
            Elf,
            Name::new("FrozenLake Elf"),
        ));
        commands.insert_resource(images);
    }

    /// Advance policy playback at Gymnasium's four frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualLake>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Advance one greedy action or reset after a terminal tile.
    fn advance_visual(visual: &mut VisualLake) {
        if matches!(visual.state, 5 | 7 | 11 | 12 | 15) {
            visual.state = visual.env.reset(Some(visual.seed)).observation;
            visual.seed = visual.seed.wrapping_add(1);
            return;
        }
        let Ok(action_index) = visual.policy.greedy_action(visual.state) else {
            return;
        };
        let transition = visual.env.step(Direction::from_index(action_index));
        visual.state = transition.observation;
    }

    /// Synchronize the elf sprite and cell position after a policy transition.
    fn sync_elf(
        visual: Res<'_, VisualLake>,
        images: Res<'_, LakeImages>,
        mut elf: Single<'_, '_, (&mut Transform, &mut Sprite), With<Elf>>,
    ) {
        let position = cell_position(visual.state);
        elf.0.translation.x = position.x;
        elf.0.translation.y = position.y;
        elf.1.image = if matches!(visual.state, 5 | 7 | 11 | 12) {
            images.cracked_hole.clone()
        } else {
            images.elf(visual.env.last_action)
        };
    }

    /// Capture 100 frames at 20 FPS while advancing the policy at 4 Hz.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualLake>,
        images: Res<'_, LakeImages>,
        mut elf: Single<'_, '_, (&mut Transform, &mut Sprite), With<Elf>>,
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

        let position = cell_position(visual.state);
        elf.0.translation.x = position.x;
        elf.0.translation.y = position.y;
        elf.1.image = if matches!(visual.state, 5 | 7 | 11 | 12) {
            images.cracked_hole.clone()
        } else {
            images.elf(visual.env.last_action)
        };
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Return the center of one Gymnasium cell in Bevy world coordinates.
    const fn cell_position(state: usize) -> Vec2 {
        let row = state / 4;
        let column = state % 4;
        Vec2::new(
            (column as f32).mul_add(64.0, -96.0),
            (row as f32).mul_add(-64.0, 96.0),
        )
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "frozen-lake", 256, 256, |frames| {
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
    fn deterministic_route_reaches_the_goal_with_gymnasium_rewards() {
        let mut lake = FrozenLake::deterministic();
        let reset = lake.reset(Some(7));
        assert_eq!(reset.observation, 0);

        let mut final_step = None;
        for action in [
            Direction::Right,
            Direction::Right,
            Direction::Down,
            Direction::Down,
            Direction::Down,
            Direction::Right,
        ] {
            let step = lake.step(action);
            if action == Direction::Right && step.observation == 15 {
                final_step = Some(step);
                break;
            }
            assert_eq!(step.reward, 0.0);
            assert_eq!(step.status, EpisodeStatus::Continuing);
        }
        let step = final_step.expect("known-safe 4x4 route reaches state 15");
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.status, EpisodeStatus::Terminated);
    }

    #[test]
    fn entering_a_hole_terminates_without_reward() {
        let mut lake = FrozenLake::deterministic();
        lake.reset(Some(11));

        let first = lake.step(Direction::Down);
        assert_eq!(first.observation, 4);
        let hole = lake.step(Direction::Right);

        assert_eq!(hole.observation, 5);
        assert_eq!(hole.reward, 0.0);
        assert_eq!(hole.status, EpisodeStatus::Terminated);
    }

    #[test]
    fn seeded_slippery_transitions_are_reproducible() {
        let mut first = FrozenLake::slippery();
        let mut second = FrozenLake::slippery();
        first.reset(Some(19));
        second.reset(Some(19));

        let first_states = (0..12)
            .map(|_| first.step(Direction::Right).observation)
            .collect::<Vec<_>>();
        let second_states = (0..12)
            .map(|_| second.step(Direction::Right).observation)
            .collect::<Vec<_>>();

        assert_eq!(first_states, second_states);
    }
}
