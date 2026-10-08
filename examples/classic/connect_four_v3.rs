//!
//! # Connect Four
//!
//! ```{figure} classic_connect_four.gif
//! :width: 140px
//! :name: connect_four
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/connect_four-v3")`         |
//! |--------------------|--------------------------------------------------|
//! | Actions            | Discrete                                         |
//! | Parallel API       | Yes                                              |
//! | Manual Control     | No                                               |
//! | Agents             | `agents= ['player_0', 'player_1']`               |
//! | Agents             | 2                                                |
//! | Action Shape       | (1,)                                             |
//! | Action Values      | Discrete(7)                                      |
//! | Observation Shape  | (6, 7, 2)                                        |
//! | Observation Values | [0,1]                                            |
//!
//!
//! Connect Four is a 2-player turn based game, where players must connect four of their tokens vertically, horizontally or diagonally. The players drop their respective token in a column of a standing grid, where each token will fall until it reaches the bottom of the column or reaches an existing
//! token. Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens, or when all 7 columns have been filled.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//!
//! The main observation space is 2 planes of a 6x7 grid. Each plane represents a specific agent's tokens, and each location in the grid represents the placement of the corresponding agent's token. 1 indicates that the agent has a token placed in that cell, and 0 indicates they do not have a token in
//! that cell. A 0 means that either the cell is empty, or the other agent has a token in that cell.
//!
//!
//! #### Legal Actions Mask
//!
//! The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
//! whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
//!
//!
//! ### Action Space
//!
//! The action space is the set of integers from 0 to 6 (inclusive), where the action represents which column a token should be dropped in.
//!
//! ### Rewards
//!
//! If an agent successfully connects four of their tokens, they will be rewarded 1 point. At the same time, the opponent agent will be awarded -1 points. If the game ends in a draw, both players are rewarded 0.
//!
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

use bevy_gym::training::{
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction, DqnConfig,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// Width of PettingZoo's Connect Four board.
const COLUMNS: usize = 7;
/// Height of PettingZoo's Connect Four board.
const ROWS: usize = 6;
/// Flat board cell count.
const CELLS: usize = ROWS * COLUMNS;
/// Two binary source observation planes.
const OBSERVATION_DIM: usize = CELLS * 2;

/// Values stored in PettingZoo's flat row-major board.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Disc {
    /// Source value zero.
    #[default]
    Empty,
    /// `player_0`, source value one.
    Red,
    /// `player_1`, source value two.
    Black,
}

/// The two PettingZoo agents in AEC order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    /// The red starting player.
    Red,
    /// The black second player.
    Black,
}

impl Player {
    /// Return the other agent.
    const fn opponent(self) -> Self {
        match self {
            Self::Red => Self::Black,
            Self::Black => Self::Red,
        }
    }

    /// Return this player's source disc value.
    const fn disc(self) -> Disc {
        match self {
            Self::Red => Disc::Red,
            Self::Black => Disc::Black,
        }
    }

    /// Return this player's stable reward-array index.
    const fn index(self) -> usize {
        match self {
            Self::Red => 0,
            Self::Black => 1,
        }
    }
}

/// One PettingZoo column action in `0..7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Column(u8);

impl Column {
    /// Convert a DQN output column into the exact action space.
    const fn from_index(index: usize) -> Self {
        Self((index % COLUMNS) as u8)
    }

    /// Return the flat action index.
    fn index(self) -> usize {
        usize::from(self.0)
    }
}

impl DqnAction for Column {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.index()
    }
}

/// A natural terminal outcome after a legal move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// Four connected discs owned by one agent.
    Winner(Player),
    /// A full board without four connected discs.
    Draw,
}

/// Typed raw-environment move rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveError {
    /// The selected column contains six discs.
    FullColumn(Column),
    /// The caller supplied the unselected AEC agent.
    WrongPlayer {
        /// Agent selected by the AEC turn sequence.
        expected: Player,
        /// Agent supplied by the caller.
        actual: Player,
    },
    /// The board has already reached a natural terminal outcome.
    GameFinished(Outcome),
}

impl fmt::Display for MoveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullColumn(column) => write!(formatter, "column {} is full", column.index()),
            Self::WrongPlayer { expected, actual } => {
                write!(formatter, "expected {expected:?}, received {actual:?}")
            }
            Self::GameFinished(outcome) => write!(formatter, "game finished with {outcome:?}"),
        }
    }
}

impl Error for MoveError {}

/// Exact source observation dictionary for one agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConnectFourObservation {
    /// Row-major `6 x 7 x 2` planes with the observing agent first.
    planes: [[[u8; 2]; COLUMNS]; ROWS],
    /// Legal columns, populated only for the selected agent.
    action_mask: [bool; COLUMNS],
}

/// Exact board state and selected AEC agent.
#[derive(Debug, Clone)]
struct Board {
    /// Source row-major board, from top-left to bottom-right.
    cells: [Disc; CELLS],
    /// Agent selected by PettingZoo's alternating selector.
    next_player: Player,
    /// Natural terminal outcome, if the game ended.
    outcome: Option<Outcome>,
}

impl Default for Board {
    fn default() -> Self {
        Self {
            cells: [Disc::Empty; CELLS],
            next_player: Player::Red,
            outcome: None,
        }
    }
}

impl Board {
    /// Restore PettingZoo's empty-board reset state.
    const fn reset(&mut self) {
        self.cells = [Disc::Empty; CELLS];
        self.next_player = Player::Red;
        self.outcome = None;
    }

    /// Drop one disc using the source's bottom-to-top slot search.
    fn play(&mut self, player: Player, column: Column) -> Result<Option<Outcome>, MoveError> {
        if let Some(outcome) = self.outcome {
            return Err(MoveError::GameFinished(outcome));
        }
        if player != self.next_player {
            return Err(MoveError::WrongPlayer {
                expected: self.next_player,
                actual: player,
            });
        }
        let Some(index) = (0..ROWS)
            .rev()
            .map(|row| row * COLUMNS + column.index())
            .find(|index| self.cells.get(*index).copied() == Some(Disc::Empty))
        else {
            return Err(MoveError::FullColumn(column));
        };

        // PettingZoo mutates one flat slot before checking the shared board
        // helper for a win and then advancing its agent selector.
        let Some(cell) = self.cells.get_mut(index) else {
            return Err(MoveError::FullColumn(column));
        };
        *cell = player.disc();
        self.outcome = self.detect_outcome(player);
        self.next_player = player.opponent();
        Ok(self.outcome)
    }

    /// Return columns whose top source cell remains empty.
    fn legal_mask(&self) -> [bool; COLUMNS] {
        std::array::from_fn(|column| self.cells.get(column).copied() == Some(Disc::Empty))
    }

    /// Build the two source planes from one agent's perspective.
    fn observe(&self, player: Player) -> ConnectFourObservation {
        let mut planes = [[[0_u8; 2]; COLUMNS]; ROWS];
        for (disc_planes, disc) in planes.iter_mut().flatten().zip(self.cells) {
            let [own, opponent] = disc_planes;
            *own = u8::from(disc == player.disc());
            *opponent = u8::from(disc == player.opponent().disc());
        }
        ConnectFourObservation {
            planes,
            action_mask: if self.outcome.is_none() && player == self.next_player {
                self.legal_mask()
            } else {
                [false; COLUMNS]
            },
        }
    }

    /// Encode one selected player's exact planes as the DQN input vector.
    fn encoded_observation(&self, player: Player) -> Vec<f32> {
        self.observe(player)
            .planes
            .into_iter()
            .flatten()
            .flatten()
            .map(f32::from)
            .collect()
    }

    /// Check every four-cell segment containing the newly placed color.
    fn detect_outcome(&self, player: Player) -> Option<Outcome> {
        const DIRECTIONS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for row in 0..ROWS {
            for column in 0..COLUMNS {
                for (row_step, column_step) in DIRECTIONS {
                    let connected = (0..4).all(|offset| {
                        let candidate_row = row as isize + row_step * offset;
                        let candidate_column = column as isize + column_step * offset;
                        self.disc_at(candidate_row, candidate_column) == Some(player.disc())
                    });
                    if connected {
                        return Some(Outcome::Winner(player));
                    }
                }
            }
        }
        self.cells
            .iter()
            .all(|disc| *disc != Disc::Empty)
            .then_some(Outcome::Draw)
    }

    /// Return one bounded row-column board cell.
    fn disc_at(&self, row: isize, column: isize) -> Option<Disc> {
        let row = usize::try_from(row).ok()?;
        let column = usize::try_from(column).ok()?;
        if row >= ROWS || column >= COLUMNS {
            return None;
        }
        self.cells.get(row * COLUMNS + column).copied()
    }
}

/// Per-step source multi-agent details retained by the Gym adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConnectFourInfo {
    /// Observations in PettingZoo agent order.
    observations: [ConnectFourObservation; 2],
    /// Integer rewards in PettingZoo agent order.
    rewards: [i8; 2],
}

impl Default for ConnectFourInfo {
    fn default() -> Self {
        let observation = ConnectFourObservation {
            planes: [[[0; 2]; COLUMNS]; ROWS],
            action_mask: [false; COLUMNS],
        };
        Self {
            observations: [observation; 2],
            rewards: [0; 2],
        }
    }
}

/// Small deterministic random stream for the held-out opponent.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Current generator state.
    state: u64,
}

impl SplitMix64 {
    /// Construct one stream from an environment seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Select a uniform index below a nonzero bound.
    fn index(&mut self, upper: usize) -> usize {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let mixed = value ^ (value >> 31);
        match u64::try_from(upper) {
            Ok(bound) if bound > 0 => usize::try_from(mixed % bound).unwrap_or_default(),
            Ok(_) | Err(_) => 0,
        }
    }
}

/// Policy-control mode for training and the official held-out comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    /// One shared policy supplies every AEC action during training.
    SharedPolicy,
    /// The policy controls player one against random player zero.
    PolicyAsBlackVsRandomRed,
}

/// Source AEC game with shared-policy training and random-first evaluation.
#[derive(Debug, Clone)]
struct ConnectFourEnv {
    /// Complete source-compatible board.
    board: Board,
    /// Opponent action stream.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for ConnectFourEnv {
    fn default() -> Self {
        Self {
            board: Board::default(),
            rng: SplitMix64::new(0),
            control: ControlMode::SharedPolicy,
        }
    }
}

impl ConnectFourEnv {
    /// Return both source observations and one reward pair.
    fn info(&self, rewards: [i8; 2]) -> ConnectFourInfo {
        ConnectFourInfo {
            observations: [
                self.board.observe(Player::Red),
                self.board.observe(Player::Black),
            ],
            rewards,
        }
    }

    /// Convert a natural outcome to PettingZoo's zero-sum rewards.
    const fn rewards(outcome: Option<Outcome>) -> [i8; 2] {
        match outcome {
            Some(Outcome::Winner(Player::Red)) => [1, -1],
            Some(Outcome::Winner(Player::Black)) => [-1, 1],
            Some(Outcome::Draw) | None => [0, 0],
        }
    }

    /// Return the observation for the policy-controlled agent.
    fn observation(&self) -> Vec<f32> {
        let player = match self.control {
            ControlMode::SharedPolicy => self.board.next_player,
            ControlMode::PolicyAsBlackVsRandomRed => Player::Black,
        };
        self.board.encoded_observation(player)
    }

    /// Select one uniformly random legal player-zero action during evaluation.
    fn random_red_action(&mut self) -> Column {
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
        Column::from_index(action)
    }

    /// Reproduce `TerminateIllegalWrapper` for one rejected raw move.
    fn wrapped_play(&mut self, player: Player, column: Column) -> ([i8; 2], EpisodeStatus) {
        match self.board.play(player, column) {
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
}

impl Env for ConnectFourEnv {
    type Observation = Vec<f32>;
    type Action = Column;
    type Info = ConnectFourInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        self.board.reset();
        self.rng = SplitMix64::new(seed.unwrap_or(0));
        Reset {
            observation: self.observation(),
            info: self.info([0; 2]),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let actor = self.board.next_player;
        let (rewards, status) = self.wrapped_play(actor, action);
        let [red_reward, black_reward] = rewards;
        let reward = match self.control {
            ControlMode::SharedPolicy if actor == Player::Red => red_reward,
            ControlMode::SharedPolicy | ControlMode::PolicyAsBlackVsRandomRed => black_reward,
        };
        Step {
            observation: self.observation(),
            reward: f64::from(reward),
            status,
            info: self.info(rewards),
        }
    }
}

impl DiscreteDqnExample for ConnectFourEnv {
    const ENV_NAME: &'static str = "connect-four-v3";
    const GYMNASIUM_ID: &'static str = "classic/connect_four-v3";
    const OBSERVATION_DIM: usize = OBSERVATION_DIM;
    const ACTION_COUNT: usize = COLUMNS;
    const DEFAULT_TRAIN_STEPS: usize = 250_000;
    const DEFAULT_EVAL_INTERVAL: usize = 5_000;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const DEFAULT_NUM_ENVS: usize = 16;
    const ZERO_SUM_COMPARISON: bool = true;
    const BOOTSTRAP_MULTIPLIER: f32 = -1.0;
    const MIN_TRAIN_STEPS: usize = 20_480;
    const SOLVED_MEAN_REWARD: f64 = 0.80;
    const GIF_PATH: &'static str = "docs/images/classic-connect-four-v3.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 0.99,
            learning_rate,
            replay_capacity: 100_000,
            min_replay_size: 1_000,
            batch_size: 128,
            target_update_interval: 500,
            epsilon_start: 0.5,
            epsilon_end: 0.02,
            epsilon_decay_steps: 60_000,
            ..DqnConfig::default()
        }
    }

    fn action_mask(observation: &[f32]) -> Vec<bool> {
        // A top cell is occupied when either source plane contains one.
        observation
            .chunks_exact(2)
            .take(COLUMNS)
            .map(|planes| planes.iter().sum::<f32>() == 0.0)
            .collect()
    }

    fn prepare_evaluation(&mut self) {
        self.control = ControlMode::PolicyAsBlackVsRandomRed;
    }

    fn evaluation_action(
        &mut self,
        policy: &bevy_gym::training::DqnPolicy,
        observation: &[f32],
    ) -> Result<Self::Action, bevy_gym::training::DqnError> {
        if self.board.next_player == Player::Red {
            Ok(self.random_red_action())
        } else {
            Self::deployment_action(policy, observation)
        }
    }

    fn selection_score(evaluation: &DiscreteEvaluation) -> f64 {
        evaluation.comparison_win_rate
    }

    fn is_success(_final_observation: &[f32], total_reward: f64, status: EpisodeStatus) -> bool {
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
    run_discrete_workflow::<ConnectFourEnv>()
}

/// PettingZoo-image renderer driven by the trained player-zero policy.
#[cfg(feature = "render")]
mod render {
    use super::{
        Board, Column, ConnectFourEnv, DiscreteDqnExample, Error, Path, Player, SplitMix64,
        COLUMNS, OBSERVATION_DIM, ROWS,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;

    /// Documentation GIF width and Python renderer aspect ratio.
    const VIEWPORT_WIDTH: f32 = 600.0;
    /// Integer viewport width used by window and encoder APIs.
    const VIEWPORT_WIDTH_PIXELS: u32 = 600;
    /// `86 / 99` of the source width.
    const VIEWPORT_HEIGHT: f32 = 522.0;
    /// Integer viewport height used by window and encoder APIs.
    const VIEWPORT_HEIGHT_PIXELS: u32 = 522;
    /// Source tile formula `(width * 91 / 99) / 7`.
    const TILE_SIZE: f32 = (VIEWPORT_WIDTH * 91.0 / 99.0) / COLUMNS as f32;
    /// Source chip image formula `tile_size * 9 / 13`.
    const CHIP_SIZE: f32 = TILE_SIZE * 9.0 / 13.0;

    /// White background used by PettingZoo's human renderer.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Checkpoint policy, random opponent, and visible AEC board.
    #[derive(Resource)]
    struct VisualGame {
        /// Greedy player-zero DQN policy.
        policy: DqnPolicy,
        /// Exact visible board.
        board: Board,
        /// Seeded player-one action stream.
        rng: SplitMix64,
    }

    /// Source red and black chip images.
    #[derive(Resource)]
    struct DiscImages {
        /// `player_0` chip.
        red: Handle<Image>,
        /// `player_1` chip.
        black: Handle<Image>,
    }

    /// One fixed sprite slot for a row-major board cell.
    #[derive(Component)]
    struct DiscSprite {
        /// Source flat board index.
        index: usize,
    }

    /// Two-Hz AEC playback clock from the source renderer metadata.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run an interactive scene or a finite GIF capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = ConnectFourEnv::dqn_config(ConnectFourEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(checkpoint, OBSERVATION_DIM, COLUMNS, &config.hidden_sizes)?;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualGame {
                policy,
                board: Board::default(),
                rng: SplitMix64::new(12_345),
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.5, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets/connect_four".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Connect Four".into(),
                            resolution: WindowResolution::new(
                                VIEWPORT_WIDTH_PIXELS,
                                VIEWPORT_HEIGHT_PIXELS,
                            ),
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
            app.add_systems(Update, (advance_watch, sync_discs).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    /// Spawn the exact source board image and 42 reusable chip slots.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("Connect Four Camera")));
        commands.spawn((
            Sprite {
                image: assets.load("img/Connect4Board.png"),
                custom_size: Some(Vec2::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT)),
                ..default()
            },
            Transform::from_xyz(0.0, 0.0, 0.0),
        ));
        let images = DiscImages {
            red: assets.load("img/C4RedPiece.png"),
            black: assets.load("img/C4BlackPiece.png"),
        };
        for index in 0..ROWS * COLUMNS {
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                DiscSprite { index },
            ));
        }
        commands.insert_resource(images);
    }

    /// Advance one selected AEC agent at PettingZoo's two FPS.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut game: ResMut<'_, VisualGame>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut game);
        }
    }

    /// Apply one policy or random-opponent move, or reset after termination.
    fn advance_visual(game: &mut VisualGame) {
        if game.board.outcome.is_some() {
            game.board.reset();
            return;
        }
        let player = game.board.next_player;
        let action = if player == Player::Black {
            let observation = game.board.encoded_observation(Player::Black);
            match ConnectFourEnv::deployment_action(&game.policy, &observation) {
                Ok(column) => column.index(),
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
        if game.board.play(player, Column::from_index(action)).is_err() {
            game.board.reset();
        }
    }

    /// Synchronize every chip with its exact source row and column.
    fn sync_discs(
        game: Res<'_, VisualGame>,
        images: Res<'_, DiscImages>,
        mut discs: Query<'_, '_, (&DiscSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
    ) {
        if game.is_changed() {
            apply_disc_sprites(&game, &images, &mut discs);
        }
    }

    /// Apply the Python renderer's chip size and top-left placement formulas.
    fn apply_disc_sprites(
        game: &VisualGame,
        images: &DiscImages,
        discs: &mut Query<'_, '_, (&DiscSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
    ) {
        for (slot, mut sprite, mut transform, mut visibility) in discs.iter_mut() {
            let Some(disc) = game.board.cells.get(slot.index).copied() else {
                *visibility = Visibility::Hidden;
                continue;
            };
            let image = match disc {
                super::Disc::Empty => {
                    *visibility = Visibility::Hidden;
                    continue;
                }
                super::Disc::Red => images.red.clone(),
                super::Disc::Black => images.black.clone(),
            };
            sprite.image = image;
            sprite.custom_size = Some(Vec2::splat(CHIP_SIZE));
            let row = slot.index / COLUMNS;
            let column = slot.index % COLUMNS;
            let left = (column as f32).mul_add(TILE_SIZE, TILE_SIZE * 6.0 / 13.0);
            let top = (row as f32).mul_add(TILE_SIZE, TILE_SIZE * 6.0 / 13.0);
            transform.translation = Vec3::new(
                left + CHIP_SIZE / 2.0 - VIEWPORT_WIDTH / 2.0,
                VIEWPORT_HEIGHT / 2.0 - top - CHIP_SIZE / 2.0,
                1.0,
            );
            *visibility = Visibility::Visible;
        }
    }

    /// Capture five seconds at 20 FPS while the game advances at two FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut game: ResMut<'_, VisualGame>,
        images: Res<'_, DiscImages>,
        mut discs: Query<'_, '_, (&DiscSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
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
        if capture.has_started() && capture.frame_index().is_multiple_of(10) {
            advance_visual(&mut game);
        }
        apply_disc_sprites(&game, &images, &mut discs);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Render and encode the exact-size five-second source-style GIF.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-connect-four-v3",
            VIEWPORT_WIDTH_PIXELS,
            VIEWPORT_HEIGHT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Board, Column, ConnectFourEnv, Disc, DiscreteDqnExample, Env, EpisodeStatus, MoveError,
        Outcome, Player, COLUMNS, OBSERVATION_DIM,
    };
    use bevy_gym::training::{DqnAgent, SeedConfig};

    #[test]
    fn training_and_evaluation_use_only_the_dqn_policy() {
        assert_eq!(ConnectFourEnv::GUIDED_TRANSITIONS, 0);

        let observation = Board::default().encoded_observation(Player::Red);
        let mask = ConnectFourEnv::action_mask(&observation);
        for seed in 0..8 {
            let agent = DqnAgent::new(
                OBSERVATION_DIM,
                COLUMNS,
                ConnectFourEnv::dqn_config(1.0e-3),
                SeedConfig::from_root(seed),
            )
            .expect("the documented DQN configuration is valid");
            let policy = agent.policy();
            let expected = policy
                .greedy_action_masked(&observation, &mask)
                .expect("the initial observation and mask are valid");
            let actual = ConnectFourEnv::deployment_action(&policy, &observation)
                .expect("deployment accepts the initial observation")
                .index();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn training_preserves_aec_turns_and_evaluation_randomizes_player_zero() {
        let mut training = ConnectFourEnv::default();
        training.reset(Some(7));
        let step = training.step(Column::from_index(3));
        assert_eq!(step.status, EpisodeStatus::Continuing);
        assert_eq!(training.board.next_player, Player::Black);
        assert_eq!(
            step.observation,
            training.board.encoded_observation(Player::Black)
        );

        let mut evaluation = ConnectFourEnv::default();
        evaluation.prepare_evaluation();
        evaluation.reset(Some(7));
        assert_eq!(evaluation.board.next_player, Player::Red);
        let random_action = evaluation.random_red_action();
        let reset = evaluation.step(random_action);
        assert_eq!(evaluation.board.next_player, Player::Black);
        assert_eq!(
            evaluation
                .board
                .cells
                .iter()
                .filter(|disc| **disc == Disc::Red)
                .count(),
            1
        );
        assert_eq!(
            reset.observation,
            evaluation.board.encoded_observation(Player::Black)
        );
    }

    #[test]
    fn discs_fall_to_the_lowest_open_row() -> Result<(), MoveError> {
        let mut board = Board::default();
        board.play(Player::Red, Column::from_index(2))?;
        board.play(Player::Black, Column::from_index(2))?;

        assert_eq!(board.cells[5 * COLUMNS + 2], Disc::Red);
        assert_eq!(board.cells[4 * COLUMNS + 2], Disc::Black);
        Ok(())
    }

    #[test]
    fn horizontal_vertical_and_diagonal_fours_win() -> Result<(), MoveError> {
        for moves in [
            vec![0, 6, 1, 6, 2, 5, 3],
            vec![0, 1, 0, 1, 0, 1, 0],
            vec![0, 1, 1, 2, 5, 2, 2, 3, 5, 3, 6, 3, 3],
        ] {
            let mut board = Board::default();
            for column in moves {
                let player = board.next_player;
                board.play(player, Column::from_index(column))?;
            }
            assert_eq!(board.outcome, Some(Outcome::Winner(Player::Red)));
        }
        Ok(())
    }

    #[test]
    fn observations_swap_planes_and_mask_only_selected_player() -> Result<(), MoveError> {
        let mut board = Board::default();
        board.play(Player::Red, Column::from_index(3))?;

        let red = board.observe(Player::Red);
        let black = board.observe(Player::Black);

        assert_eq!(red.planes[5][3], [1, 0]);
        assert_eq!(black.planes[5][3], [0, 1]);
        assert_eq!(red.action_mask, [false; COLUMNS]);
        assert!(black.action_mask.into_iter().all(|legal| legal));
        Ok(())
    }

    #[test]
    fn full_column_and_wrong_player_are_typed_errors() -> Result<(), MoveError> {
        let mut board = Board::default();
        assert_eq!(
            board.play(Player::Black, Column::from_index(0)),
            Err(MoveError::WrongPlayer {
                expected: Player::Red,
                actual: Player::Black,
            })
        );
        for _ in 0..6 {
            let player = board.next_player;
            board.play(player, Column::from_index(0))?;
        }
        assert_eq!(
            board.play(Player::Red, Column::from_index(0)),
            Err(MoveError::FullColumn(Column::from_index(0)))
        );
        Ok(())
    }

    #[test]
    fn wrapper_penalizes_an_illegal_column_and_terminates() {
        let mut env = ConnectFourEnv::default();
        env.reset(Some(9));
        for row in 0..6 {
            env.board.cells[row * COLUMNS] = if row.is_multiple_of(2) {
                Disc::Red
            } else {
                Disc::Black
            };
        }

        let step = env.step(Column::from_index(0));

        assert_eq!(step.reward, -1.0);
        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(step.info.rewards, [-1, 0]);
    }
}
