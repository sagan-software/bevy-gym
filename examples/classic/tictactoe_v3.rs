//!
//! # Tic Tac Toe
//!
//! ```{figure} classic_tictactoe.gif
//! :width: 140px
//! :name: tictactoe
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/tictactoe-v3")`         |
//! |--------------------|-----------------------------------------------|
//! | Actions            | Discrete                                      |
//! | Parallel API       | Yes                                           |
//! | Manual Control     | No                                            |
//! | Agents             | `agents= ['player_1', 'player_2']`            |
//! | Agents             | 2                                             |
//! | Action Shape       | (1)                                           |
//! | Action Values      | [0, 8]                                        |
//! | Observation Shape  | (3, 3, 2)                                     |
//! | Observation Values | [0,1]                                         |
//!
//!
//! Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//! The main observation is 2 planes of the 3x3 board. For player_1, the first plane represents the placement of Xs, and the second plane shows the placement of Os. The possible values for each cell are 0 or 1; in the first plane, 1 indicates that an X has been placed in that cell, and 0 indicates
//! that X is not in that cell. Similarly, in the second plane, 1 indicates that an O has been placed in that cell, while 0 indicates that an O has not been placed. For player_2, the observation is the same, but Xs and Os swap positions, so Os are encoded in plane 1 and Xs in plane 2. This allows for
//! self-play.
//!
//! #### Legal Actions Mask
//!
//! The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
//! whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
//!
//! ### Action Space
//!
//! Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:
//!
//!
//!  ```
//! 0 | 3 | 6
//! _________
//!
//! 1 | 4 | 7
//! _________
//!
//! 2 | 5 | 8
//!  ```
//!
//! ### Rewards
//!
//! | Winner | Loser |
//! | :----: | :---: |
//! | +1     | -1    |
//!
//! If the game ends in a draw, both players will receive a reward of 0.
//!
//! ### Version History
//!
//! * v3: Fixed bug in arbitrary calls to observe() (1.8.0)
//! * v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
//! * v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
//! * v0: Initial versions release (1.0.0)
//!

#![expect(
    clippy::doc_markdown,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]

use std::error::Error;
use std::fmt;
use std::path::Path;

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use serde as _;
use serde_json as _;
use shakmaty as _;
use tokio as _;

use bevy_gym::training::{
    run_tabular_workflow, IndexedAction, TabularEvaluation as Evaluation, TabularExample,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// Three marks stored by the source board.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Mark {
    /// Unoccupied square, encoded as source value 0.
    #[default]
    Empty,
    /// Player 1's cross, encoded as source value 1.
    Cross,
    /// Player 2's circle, encoded as source value 2.
    Circle,
}

impl Mark {
    /// Return the source board integer used by base-three state encoding.
    const fn digit(self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Cross => 1,
            Self::Circle => 2,
        }
    }
}

/// PettingZoo's two agents in turn order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    /// `player_1`, which places crosses and starts.
    One,
    /// `player_2`, which places circles.
    Two,
}

impl Player {
    /// Return the other agent.
    const fn opponent(self) -> Self {
        match self {
            Self::One => Self::Two,
            Self::Two => Self::One,
        }
    }

    /// Return the source board mark owned by this player.
    const fn mark(self) -> Mark {
        match self {
            Self::One => Mark::Cross,
            Self::Two => Mark::Circle,
        }
    }

    /// Return the stable player-array index.
    const fn index(self) -> usize {
        match self {
            Self::One => 0,
            Self::Two => 1,
        }
    }
}

/// One source action index in `0..9`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Square(u8);

impl Square {
    /// Convert one table column into the source discrete action space.
    const fn from_index(index: usize) -> Self {
        let value = match index % 9 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 3,
            4 => 4,
            5 => 5,
            6 => 6,
            7 => 7,
            _ => 8,
        };
        Self(value)
    }

    /// Return the source action index.
    fn index(self) -> usize {
        usize::from(self.0)
    }
}

impl IndexedAction for Square {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }
}

/// Natural terminal result of a legal game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Three marks in one winning line.
    Winner(Player),
    /// Full board without a winning line.
    Draw,
}

/// Recoverable raw-board move rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveError {
    /// An action targeted an occupied square.
    Occupied(Square),
    /// The caller supplied the agent that is not selected.
    WrongPlayer {
        /// Agent selected by the AEC turn sequence.
        expected: Player,
        /// Agent supplied by the caller.
        actual: Player,
    },
    /// The game already reached a natural terminal result.
    GameFinished(Outcome),
}

impl fmt::Display for MoveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Occupied(square) => write!(formatter, "square {} is occupied", square.index()),
            Self::WrongPlayer { expected, actual } => {
                write!(formatter, "expected {expected:?}, received {actual:?}")
            }
            Self::GameFinished(outcome) => write!(formatter, "game finished with {outcome:?}"),
        }
    }
}

impl Error for MoveError {}

/// PettingZoo observation dictionary for one agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TicTacToeObservation {
    /// Two 3-by-3 planes, with the observing agent first.
    planes: [[[u8; 2]; 3]; 3],
    /// Legal source actions for the selected agent.
    action_mask: [bool; 9],
}

/// Exact board and agent selector for the raw source environment.
#[derive(Debug, Clone)]
struct Board {
    /// Flat source order: `0,1,2` down the first column, then the next column.
    cells: [Mark; 9],
    /// Agent selected by the AEC turn sequence.
    next_player: Player,
    /// Natural terminal result, if any.
    outcome: Option<Outcome>,
}

impl Default for Board {
    fn default() -> Self {
        Self {
            cells: [Mark::Empty; 9],
            next_player: Player::One,
            outcome: None,
        }
    }
}

impl Board {
    /// Restore the source reset state.
    const fn reset(&mut self) {
        self.cells = [Mark::Empty; 9];
        self.next_player = Player::One;
        self.outcome = None;
    }

    /// Place one raw-environment mark after checking the AEC turn contract.
    fn play(&mut self, player: Player, square: Square) -> Result<Option<Outcome>, MoveError> {
        if let Some(outcome) = self.outcome {
            return Err(MoveError::GameFinished(outcome));
        }
        if player != self.next_player {
            return Err(MoveError::WrongPlayer {
                expected: self.next_player,
                actual: player,
            });
        }
        let cell = self
            .cells
            .get_mut(square.index())
            .ok_or(MoveError::Occupied(square))?;
        if *cell != Mark::Empty {
            return Err(MoveError::Occupied(square));
        }

        // The board stores source values while the agent selector advances
        // independently, exactly as the Python `Board` and `raw_env` do.
        *cell = player.mark();
        self.outcome = self.detect_outcome();
        self.next_player = player.opponent();
        Ok(self.outcome)
    }

    /// Return all currently empty source action indices.
    fn legal_mask(&self) -> [bool; 9] {
        self.cells.map(|mark| mark == Mark::Empty)
    }

    /// Build the source two-plane observation from one agent's perspective.
    fn observe(&self, player: Player) -> TicTacToeObservation {
        let mut planes = [[[0_u8; 2]; 3]; 3];
        for (mark_planes, mark) in planes.iter_mut().flatten().zip(self.cells) {
            let [own, opponent] = mark_planes;
            *own = u8::from(mark == player.mark());
            *opponent = u8::from(mark == player.opponent().mark());
        }
        TicTacToeObservation {
            planes,
            action_mask: if self.outcome.is_none() && player == self.next_player {
                self.legal_mask()
            } else {
                [false; 9]
            },
        }
    }

    /// Encode the exact flat board as one base-three table state.
    fn state_index(&self) -> usize {
        self.cells.iter().rev().fold(0, |state, mark| {
            state.saturating_mul(3).saturating_add(mark.digit())
        })
    }

    /// Check source winning lines before the full-board draw rule.
    fn detect_outcome(&self) -> Option<Outcome> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];
        for line in LINES {
            let [first, second, third] =
                line.map(|index| self.cells.get(index).copied().unwrap_or(Mark::Empty));
            if first != Mark::Empty && first == second && first == third {
                return Some(Outcome::Winner(if first == Mark::Cross {
                    Player::One
                } else {
                    Player::Two
                }));
            }
        }
        self.cells
            .iter()
            .all(|mark| *mark != Mark::Empty)
            .then_some(Outcome::Draw)
    }
}

/// Per-step multi-agent details retained by the Bevy Gym adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TicTacToeInfo {
    /// Observations in source player order.
    observations: [TicTacToeObservation; 2],
    /// Integer terminal rewards in source player order.
    rewards: [i8; 2],
}

impl Default for TicTacToeInfo {
    fn default() -> Self {
        let observation = TicTacToeObservation {
            planes: [[[0; 2]; 3]; 3],
            action_mask: [false; 9],
        };
        Self {
            observations: [observation; 2],
            rewards: [0; 2],
        }
    }
}

/// Deterministic random opponent generator.
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
        let upper_u64 = match u64::try_from(upper) {
            Ok(converted) if converted > 0 => converted,
            Ok(_) | Err(_) => return 0,
        };
        usize::try_from(mixed % upper_u64).unwrap_or_default()
    }
}

/// Official Tianshou view: player two learns against random player one.
#[derive(Debug, Clone)]
struct TicTacToeTrainingEnv {
    /// Complete source-compatible AEC board.
    board: Board,
    /// Seeded player-one action generator.
    rng: SplitMix64,
}

impl Default for TicTacToeTrainingEnv {
    fn default() -> Self {
        Self {
            board: Board::default(),
            rng: SplitMix64::new(0),
        }
    }
}

impl TicTacToeTrainingEnv {
    /// Return both source observations and the supplied reward pair.
    fn info(&self, rewards: [i8; 2]) -> TicTacToeInfo {
        TicTacToeInfo {
            observations: [
                self.board.observe(Player::One),
                self.board.observe(Player::Two),
            ],
            rewards,
        }
    }

    /// Convert one raw-board result into the wrapper's terminal reward pair.
    const fn rewards(outcome: Option<Outcome>) -> [i8; 2] {
        match outcome {
            Some(Outcome::Winner(Player::One)) => [1, -1],
            Some(Outcome::Winner(Player::Two)) => [-1, 1],
            Some(Outcome::Draw) | None => [0, 0],
        }
    }

    /// Apply one action and reproduce `TerminateIllegalWrapper` on rejection.
    fn wrapped_play(&mut self, player: Player, action: Square) -> ([i8; 2], EpisodeStatus) {
        match self.board.play(player, action) {
            Ok(outcome) => (
                Self::rewards(outcome),
                if outcome.is_some() {
                    EpisodeStatus::Terminated
                } else {
                    EpisodeStatus::Continuing
                },
            ),
            Err(_error) => {
                let mut rewards = [0; 2];
                if let Some(reward) = rewards.get_mut(player.index()) {
                    *reward = -1;
                }
                (rewards, EpisodeStatus::Terminated)
            }
        }
    }

    /// Apply one uniformly random legal player-one action.
    fn play_random_player_one(&mut self) -> ([i8; 2], EpisodeStatus) {
        let legal: Vec<_> = self
            .board
            .legal_mask()
            .into_iter()
            .enumerate()
            .filter_map(|(index, is_legal)| is_legal.then_some(index))
            .collect();
        let action = legal
            .get(self.rng.index(legal.len()))
            .copied()
            .unwrap_or_default();
        self.wrapped_play(Player::One, Square::from_index(action))
    }
}

impl Env for TicTacToeTrainingEnv {
    type Observation = usize;
    type Action = Square;
    type Info = TicTacToeInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        self.board.reset();
        self.rng = SplitMix64::new(seed.unwrap_or(0));
        let (_rewards, _status) = self.play_random_player_one();
        Reset {
            observation: self.board.state_index(),
            info: self.info([0; 2]),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        // One learner transition includes the next random player-one action,
        // matching PettingZoo's official Tianshou policy assignment.
        let (mut rewards, mut status) = self.wrapped_play(Player::Two, action);
        if status == EpisodeStatus::Continuing {
            (rewards, status) = self.play_random_player_one();
        }
        Step {
            observation: self.board.state_index(),
            reward: f64::from(rewards[1]),
            status,
            info: self.info(rewards),
        }
    }
}

impl TabularExample for TicTacToeTrainingEnv {
    const ENV_NAME: &'static str = "tictactoe-v3";
    const GYMNASIUM_ID: &'static str = "classic/tictactoe-v3";
    const STATE_COUNT: usize = 19_683;
    const ACTION_COUNT: usize = 9;
    const MAX_STEPS: usize = 4;
    const DEFAULT_EPISODES: usize = 100_000;
    const DEFAULT_EVAL_INTERVAL: usize = 5_000;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const DEFAULT_LEARNING_RATE: f64 = 0.2;
    const GAMMA: f64 = 0.99;
    const EPSILON_START: f64 = 0.5;
    const EPSILON_END: f64 = 0.02;
    const EPSILON_DECAY_STEPS_PER_EPISODE: usize = 5;
    const SOLVED_SCORE: f64 = 0.6;
    const GIF_PATH: &'static str = "docs/images/classic-tictactoe-v3.gif";

    fn action_masks() -> Option<Vec<Vec<bool>>> {
        Some(
            (0..Self::STATE_COUNT)
                .map(|mut state| {
                    let mut mask = [false; 9];
                    for is_legal in &mut mask {
                        *is_legal = state % 3 == 0;
                        state /= 3;
                    }
                    if mask.iter().all(|is_legal| !is_legal) {
                        mask[0] = true;
                    }
                    mask.to_vec()
                })
                .collect(),
        )
    }

    fn selection_score(evaluation: &Evaluation) -> f64 {
        evaluation.mean_reward
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
    run_tabular_workflow::<TicTacToeTrainingEnv>()
}

/// Source-matching Bevy renderer.
#[cfg(feature = "render")]
mod render {
    use super::{Board, Error, Mark, Path, Player, SplitMix64, Square};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::{TabularQPolicy, TabularQTrainer};
    use bevy_inspector_egui as _;

    /// Documentation GIF viewport and source render profile.
    const VIEWPORT: f32 = 600.0;
    /// Integer viewport edge used by window and encoder APIs.
    const VIEWPORT_PIXELS: u32 = 600;

    /// Source board image and white clear color.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Policy, random opponent, and AEC board used by the visible game.
    #[derive(Resource)]
    struct VisualGame {
        /// Validation-selected player-two policy.
        policy: TabularQPolicy,
        /// Visible source board.
        board: Board,
        /// Seeded player-one action generator.
        rng: SplitMix64,
    }

    /// Source mark image handles.
    #[derive(Resource)]
    struct MarkImages {
        /// Cross image.
        cross: Handle<Image>,
        /// Circle image.
        circle: Handle<Image>,
    }

    /// One source board square sprite.
    #[derive(Component)]
    struct MarkSprite {
        /// Source action index.
        square: usize,
    }

    /// One-Hz source playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run an interactive or finite source-matching scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let trainer = TabularQTrainer::load(checkpoint)?;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualGame {
                policy: trainer.policy(),
                board: Board::default(),
                rng: SplitMix64::new(12_345),
            })
            .insert_resource(VisualClock(Timer::from_seconds(1.0, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets/tictactoe".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Tic-Tac-Toe".into(),
                            resolution: WindowResolution::new(VIEWPORT_PIXELS, VIEWPORT_PIXELS),
                            present_mode: PresentMode::AutoVsync,
                            resizable: false,
                            ..default()
                        }),
                        ..default()
                    }),
            )
            .add_systems(Startup, setup_scene);

        if let Some(directory) = capture_dir {
            app.insert_resource(GifCapture::new(directory))
                .add_systems(Update, capture_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_marks).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    /// Spawn the source board and nine fixed mark slots.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("Tic-Tac-Toe Camera")));
        commands.spawn((
            Sprite {
                image: assets.load("img/board.png"),
                custom_size: Some(Vec2::splat(VIEWPORT)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));
        let images = MarkImages {
            cross: assets.load("img/cross.png"),
            circle: assets.load("img/circle.png"),
        };
        for square in 0..9 {
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                MarkSprite { square },
            ));
        }
        commands.insert_resource(images);
    }

    /// Advance one AEC move at the source renderer's one FPS.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut game: ResMut<'_, VisualGame>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut game);
        }
    }

    /// Apply one selected-agent action or reset one tick after termination.
    fn advance_visual(game: &mut VisualGame) {
        if game.board.outcome.is_some() {
            game.board.reset();
            return;
        }
        let player = game.board.next_player;
        let action = if player == Player::Two {
            match game.policy.greedy_action(game.board.state_index()) {
                Ok(index) => index,
                Err(_error) => return,
            }
        } else {
            let legal: Vec<usize> = game
                .board
                .legal_mask()
                .into_iter()
                .enumerate()
                .filter_map(|(index, is_legal)| is_legal.then_some(index))
                .collect();
            match legal.get(game.rng.index(legal.len())).copied() {
                Some(index) => index,
                None => return,
            }
        };
        if game.board.play(player, Square::from_index(action)).is_err() {
            game.board.reset();
        }
    }

    /// Synchronize every mark with the flat source board.
    fn sync_marks(
        game: Res<'_, VisualGame>,
        images: Res<'_, MarkImages>,
        mut marks: Query<'_, '_, (&MarkSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
    ) {
        if !game.is_changed() {
            return;
        }
        sync_mark_sprites(&game, &images, &mut marks);
    }

    /// Apply source mark images and placement formulas to fixed sprite slots.
    fn sync_mark_sprites(
        game: &VisualGame,
        images: &MarkImages,
        marks: &mut Query<'_, '_, (&MarkSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
    ) {
        let tile_size = VIEWPORT / 4.0;
        for (slot, mut sprite, mut transform, mut visibility) in marks.iter_mut() {
            let Some(mark) = game.board.cells.get(slot.square).copied() else {
                *visibility = Visibility::Hidden;
                continue;
            };
            if mark == Mark::Empty {
                *visibility = Visibility::Hidden;
                continue;
            }
            sprite.image = if mark == Mark::Cross {
                images.cross.clone()
            } else {
                images.circle.clone()
            };
            sprite.custom_size = Some(Vec2::splat(tile_size));
            let x = slot.square / 3;
            let y = slot.square % 3;
            let left = (VIEWPORT / 3.1).mul_add(x as f32, VIEWPORT / 17.0);
            let top = (VIEWPORT / 3.145).mul_add(y as f32, VIEWPORT / 19.0);
            transform.translation = Vec3::new(
                left + tile_size / 2.0 - VIEWPORT / 2.0,
                VIEWPORT / 2.0 - top - tile_size / 2.0,
                1.0,
            );
            *visibility = Visibility::Visible;
        }
    }

    /// Capture five seconds at 20 FPS while advancing at one FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut game: ResMut<'_, VisualGame>,
        images: Res<'_, MarkImages>,
        mut marks: Query<'_, '_, (&MarkSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
        captures: Query<'_, '_, Entity, With<Capturing>>,
        mut exit: MessageWriter<'_, AppExit>,
    ) {
        if !captures.is_empty() || capture.is_warming_up() {
            return;
        }
        if capture.is_complete() {
            exit.write(AppExit::Success);
            return;
        }
        if capture.has_started() && capture.frame_index().is_multiple_of(20) {
            advance_visual(&mut game);
        }
        sync_mark_sprites(&game, &images, &mut marks);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render and encode the source-sized five-second demonstration.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-tictactoe-v3",
            VIEWPORT_PIXELS,
            VIEWPORT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Board, Env, EpisodeStatus, Mark, MoveError, Outcome, Player, Square, TicTacToeTrainingEnv,
    };

    #[test]
    fn first_column_wins_for_player_one() -> Result<(), MoveError> {
        let mut board = Board::default();
        for (player, square) in [
            (Player::One, 0),
            (Player::Two, 3),
            (Player::One, 1),
            (Player::Two, 4),
            (Player::One, 2),
        ] {
            board.play(player, Square::from_index(square))?;
        }

        assert_eq!(board.outcome, Some(Outcome::Winner(Player::One)));
        Ok::<(), MoveError>(())
    }

    #[test]
    fn all_eight_source_lines_detect_a_win() -> Result<(), MoveError> {
        let lines = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];
        for line in lines {
            let mut board = Board::default();
            let fillers: Vec<usize> = (0..9).filter(|square| !line.contains(square)).collect();
            board.play(Player::One, Square::from_index(line[0]))?;
            board.play(Player::Two, Square::from_index(fillers[0]))?;
            board.play(Player::One, Square::from_index(line[1]))?;
            board.play(Player::Two, Square::from_index(fillers[1]))?;
            board.play(Player::One, Square::from_index(line[2]))?;
            assert_eq!(board.outcome, Some(Outcome::Winner(Player::One)));
        }
        Ok(())
    }

    #[test]
    fn full_nonwinning_board_is_a_draw() -> Result<(), MoveError> {
        let mut board = Board::default();
        for square in [0, 1, 2, 4, 3, 5, 7, 6, 8] {
            let player = board.next_player;
            board.play(player, Square::from_index(square))?;
        }

        assert_eq!(board.outcome, Some(Outcome::Draw));
        Ok(())
    }

    #[test]
    fn observations_swap_planes_and_mask_only_the_selected_agent() -> Result<(), MoveError> {
        let mut board = Board::default();
        board.play(Player::One, Square::from_index(4))?;

        let first = board.observe(Player::One);
        let second = board.observe(Player::Two);

        assert_eq!(first.planes[1][1], [1, 0]);
        assert_eq!(second.planes[1][1], [0, 1]);
        assert_eq!(first.action_mask, [false; 9]);
        assert!(!second.action_mask[4]);
        assert_eq!(second.action_mask.iter().filter(|legal| **legal).count(), 8);
        Ok(())
    }

    #[test]
    fn occupied_and_wrong_player_moves_are_typed_errors() -> Result<(), MoveError> {
        let mut board = Board::default();
        assert_eq!(
            board.play(Player::Two, Square::from_index(0)),
            Err(MoveError::WrongPlayer {
                expected: Player::One,
                actual: Player::Two,
            })
        );
        board.play(Player::One, Square::from_index(0))?;
        assert_eq!(
            board.play(Player::Two, Square::from_index(0)),
            Err(MoveError::Occupied(Square::from_index(0)))
        );
        Ok(())
    }

    #[test]
    fn base_three_state_preserves_source_square_order() -> Result<(), MoveError> {
        let mut board = Board::default();
        board.play(Player::One, Square::from_index(0))?;
        board.play(Player::Two, Square::from_index(1))?;

        assert_eq!(board.state_index(), 7);
        assert_eq!(board.cells[0], Mark::Cross);
        assert_eq!(board.cells[1], Mark::Circle);
        Ok(())
    }

    #[test]
    fn wrapper_penalizes_an_illegal_action_and_terminates() {
        let mut env = TicTacToeTrainingEnv::default();
        env.reset(Some(9));
        env.board.cells[0] = Mark::Cross;

        let step = env.step(Square::from_index(0));

        assert_eq!(step.reward, -1.0);
        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(step.info.rewards, [0, -1]);
    }

    #[test]
    fn training_assigns_random_player_one_and_learned_player_two() {
        let mut first = TicTacToeTrainingEnv::default();
        let first_reset = first.reset(Some(7));
        let mut replay = TicTacToeTrainingEnv::default();
        let replay_reset = replay.reset(Some(7));

        assert_eq!(first_reset, replay_reset);
        assert_eq!(first.board.next_player, Player::Two);
        assert_eq!(
            first
                .board
                .cells
                .iter()
                .filter(|mark| **mark == Mark::Cross)
                .count(),
            1
        );
    }
}
