//!
//! # Go
//!
//! ```{figure} classic_go.gif
//! :width: 140px
//! :name: go
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/go-v5")`         |
//! |--------------------|----------------------------------------|
//! | Actions            | Discrete                               |
//! | Parallel API       | Yes                                    |
//! | Manual Control     | No                                     |
//! | Agents             | `agents= ['black_0', 'white_0']`       |
//! | Agents             | 2                                      |
//! | Action Shape       | Discrete(362)                          |
//! | Action Values      | Discrete(362)                          |
//! | Observation Shape  | (19, 19, 3)                            |
//! | Observation Values | [0, 1]                                 |
//!
//!
//! Go is a board game with 2 players, black and white. The black player starts by placing a black stone at an empty board intersection. The white player follows by placing a stone of their own, aiming to either surround more territory than their opponent or capture the opponent's stones. The game
//! ends if both players sequentially decide to pass.
//!
//! Our implementation is a wrapper for [MiniGo](https://github.com/tensorflow/minigo).
//!
//! ### Arguments
//!
//! Go takes two optional arguments that define the board size (int) and komi compensation points (float). The default values for the board size and komi are 19 and 7.5, respectively.
//!
//! ```python
//! from pettingzoo import make
//!
//! make("aec", "classic/go-v5", board_size=19, komi=7.5)
//! ```
//!
//! `board_size`: The length of each size of the board.
//!
//! `komi`: The number of points given to white to compensate it for the disadvantage inherent to moving second. 7.5 is the standard value for Chinese tournament Go, but may not be perfectly balanced.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//!
//! The main observation shape is a function of the board size _N_ and has a shape of (N, N, 3). The first plane, (:, :, 0), represent the stones on the board for the current player while the second plane, (:, :, 1), encodes the stones of the opponent. The third plane, (:, :, 2), is all 1 if the
//! current player is `black_0` or all 0 if the player is `white_0`. The state of the board is represented with the top left corner as (0, 0). For example, a (9, 9) board is
//! ```
//!    0 1 2 3 4 5 6 7 8
//!  0 . . . . . . . . .  0
//!  1 . . . . . . . . .  1
//!  2 . . . . . . . . .  2
//!  3 . . . . . . . . .  3
//!  4 . . . . . . . . .  4
//!  5 . . . . . . . . .  5
//!  6 . . . . . . . . .  6
//!  7 . . . . . . . . .  7
//!  8 . . . . . . . . .  8
//!    0 1 2 3 4 5 6 7 8
//! ```
//!
//! |  Plane  | Description                                               |
//! |:-------:|-----------------------------------------------------------|
//! |    0    | Current Player's stones<br>_'`0`: no stone, `1`: stone_   |
//! |    1    | Opponent Player's stones<br>_'`0`: no stone, `1`: stone_  |
//! |    2    | Player<br>_'`0`: white, `1`: black_                       |
//!
//! While rendering, the board coordinate system is [GTP](http://www.lysator.liu.se/~gunnar/gtp/).
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
//! Similar to the observation space, the action space is dependent on the board size _N_.
//!
//! |                          Action ID                           | Description                                                  |
//! | :----------------------------------------------------------: | ------------------------------------------------------------ |
//! | $0 \ldots (N-1)$ | Place a stone on the 1st row of the board.<br>_`0`: (0,0), `1`: (0,1), ..., `N-1`: (0,N-1)_ |
//! | $N \ldots (2N- 1)$ | Place a stone on the 2nd row of the board.<br>_`N`: (1,0), `N+1`: (1,1), ..., `2N-1`: (1,N-1)_ |
//! |                             ...                              | ...                                                          |
//! | $(N^2-N) \ldots (N^2-1)$ | Place a stone on the Nth row of the board.<br>_`N^2-N`: (N-1,0), `N^2-N+1`: (N-1,1), ..., `N^2-1`: (N-1,N-1)_ |
//! | $N^2$ | Pass                                                         |
//!
//! For example, you would use action `4` to place a stone on the board at the (0,3) location or action `N^2` to pass. You can transform a non-pass action `a` back into its 2D (x,y) coordinate by computing `(a//N, a%N)`. The total action space is
//! $N^2+1$.
//!
//! ### Rewards
//!
//! | Winner | Loser |
//! | :----: | :---: |
//! | +1     | -1    |
//!
//! ### Version History
//!
//! * v5: Changed observation space to proper AlphaZero style frame stacking (1.11.0)
//! * v4: Fixed bug in how black and white pieces were saved in observation space (1.10.0)
//! * v3: Fixed bug in arbitrary calls to observe() (1.8.0)
//! * v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
//! * v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
//! * v0: Initial versions release (1.0.0)
//!

#![expect(
    clippy::doc_markdown,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]

use std::collections::{HashSet, VecDeque};
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
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction, DqnConfig,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// PettingZoo's default board edge length.
const BOARD_SIZE: usize = 19;
/// Number of board intersections.
const POINT_COUNT: usize = BOARD_SIZE * BOARD_SIZE;
/// All points plus pass.
const ACTION_COUNT: usize = POINT_COUNT + 1;
/// Default Chinese-rules compensation for white.
const KOMI: f64 = 7.5;

/// Empty, black, or white board point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Stone {
    /// No stone occupies this intersection.
    Empty,
    /// First-player stone.
    Black,
    /// Second-player stone.
    White,
}

impl Stone {
    /// Return the opposing stone color.
    const fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
            Self::Empty => Self::Empty,
        }
    }
}

/// Flat row-major point or the final pass action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GoAction(usize);

impl GoAction {
    /// Construct an action using PettingZoo's modulo table adapter convention.
    const fn from_index(index: usize) -> Self {
        Self(index % ACTION_COUNT)
    }

    /// Construct a board point from a top-left row and column.
    #[cfg(test)]
    const fn point(row: usize, column: usize) -> Self {
        Self(row * BOARD_SIZE + column)
    }

    /// Construct the pass action `N²`.
    #[cfg(test)]
    const fn pass() -> Self {
        Self(POINT_COUNT)
    }

    /// Return a board index or `None` for pass.
    const fn board_index(self) -> Option<usize> {
        if self.0 == POINT_COUNT {
            None
        } else {
            Some(self.0)
        }
    }
}

impl DqnAction for GoAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.0
    }
}

/// Exact 19 by 19 by 17 AlphaZero-style observation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GoObservation {
    /// Eight paired board-history planes followed by the player plane.
    values: Vec<u8>,
    /// Legal actions for the selected player.
    action_mask: Vec<bool>,
}

/// Typed raw-environment rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoError {
    /// The caller supplied the unselected player.
    WrongPlayer {
        /// Player selected by the position.
        expected: u8,
        /// Player supplied by the caller.
        actual: u8,
    },
    /// The move is occupied, suicidal, or forbidden by simple ko.
    IllegalMove {
        /// Source action index rejected by the rules.
        action: u16,
    },
    /// Two passes already ended the game.
    GameFinished,
}

impl fmt::Display for GoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPlayer { expected, actual } => {
                write!(
                    formatter,
                    "expected player {expected}, received player {actual}"
                )
            }
            Self::IllegalMove { action } => write!(formatter, "Go action {action} is illegal"),
            Self::GameFinished => formatter.write_str("the Go game has finished"),
        }
    }
}

impl Error for GoError {}

/// Minimal Go position matching the bundled MiniGo rules.
#[derive(Debug, Clone)]
struct GoGame {
    /// Row-major board.
    board: [Stone; POINT_COUNT],
    /// Selected color.
    to_play: Stone,
    /// Point forbidden by one-stone simple ko.
    ko: Option<usize>,
    /// Most recent actions, retained to detect two passes.
    recent: Vec<GoAction>,
    /// Eight paired planes in PettingZoo's newest-first layout.
    board_history: Vec<u8>,
    /// Chinese-rules white compensation.
    komi: f64,
    /// Terminal rewards in black, white order.
    rewards: Option<[f64; 2]>,
}

impl Default for GoGame {
    fn default() -> Self {
        Self::new(KOMI)
    }
}

impl GoGame {
    /// Construct an empty position with black selected.
    fn new(komi: f64) -> Self {
        Self {
            board: [Stone::Empty; POINT_COUNT],
            to_play: Stone::Black,
            ko: None,
            recent: Vec::new(),
            board_history: vec![0; POINT_COUNT * 16],
            komi,
            rewards: None,
        }
    }

    /// Reset all game and frame-stack state.
    fn reset(&mut self) {
        *self = Self::new(self.komi);
    }

    /// Return the selected player index.
    const fn current_player(&self) -> usize {
        match self.to_play {
            Stone::White => 1,
            Stone::Black | Stone::Empty => 0,
        }
    }

    /// Return orthogonal neighbors without allocation.
    fn neighbors(index: usize) -> impl Iterator<Item = usize> {
        let row = index / BOARD_SIZE;
        let column = index % BOARD_SIZE;
        [
            row.checked_sub(1).map(|value| value * BOARD_SIZE + column),
            (row + 1 < BOARD_SIZE).then_some((row + 1) * BOARD_SIZE + column),
            column.checked_sub(1).map(|value| row * BOARD_SIZE + value),
            (column + 1 < BOARD_SIZE).then_some(row * BOARD_SIZE + column + 1),
        ]
        .into_iter()
        .flatten()
    }

    /// Find a connected chain and its liberties.
    fn chain_and_liberties(&self, start: usize) -> (Vec<usize>, HashSet<usize>) {
        let Some(color) = self.board.get(start).copied() else {
            return (Vec::new(), HashSet::new());
        };
        let mut chain = Vec::new();
        let mut liberties = HashSet::new();
        let mut visited = [false; POINT_COUNT];
        let mut frontier = vec![start];
        if let Some(was_visited) = visited.get_mut(start) {
            *was_visited = true;
        }
        while let Some(point) = frontier.pop() {
            chain.push(point);
            for neighbor in Self::neighbors(point) {
                match self.board.get(neighbor).copied().unwrap_or(Stone::Empty) {
                    Stone::Empty => {
                        liberties.insert(neighbor);
                    }
                    candidate
                        if candidate == color
                            && !visited.get(neighbor).copied().unwrap_or(true) =>
                    {
                        if let Some(was_visited) = visited.get_mut(neighbor) {
                            *was_visited = true;
                        }
                        frontier.push(neighbor);
                    }
                    _ => {}
                }
            }
        }
        (chain, liberties)
    }

    /// Simulate one placement and return its board, captures, and ko point.
    fn placement_result(
        &self,
        index: usize,
    ) -> Option<([Stone; POINT_COUNT], Vec<usize>, Option<usize>)> {
        if self.board.get(index).copied() != Some(Stone::Empty) || self.ko == Some(index) {
            return None;
        }
        let mut candidate = self.clone();
        *candidate.board.get_mut(index)? = self.to_play;
        let mut captured = Vec::new();
        let mut checked = [false; POINT_COUNT];
        for neighbor in Self::neighbors(index) {
            if candidate.board.get(neighbor).copied() != Some(self.to_play.opponent())
                || checked.get(neighbor).copied().unwrap_or(true)
            {
                continue;
            }
            let (chain, liberties) = candidate.chain_and_liberties(neighbor);
            for point in &chain {
                if let Some(was_checked) = checked.get_mut(*point) {
                    *was_checked = true;
                }
            }
            if liberties.is_empty() {
                for point in chain {
                    if let Some(stone) = candidate.board.get_mut(point) {
                        *stone = Stone::Empty;
                        captured.push(point);
                    }
                }
            }
        }
        let (placed_chain, liberties) = candidate.chain_and_liberties(index);
        if liberties.is_empty() {
            return None;
        }
        let new_ko = if captured.len() == 1 && placed_chain.len() == 1 && liberties.len() == 1 {
            captured.first().copied()
        } else {
            None
        };
        Some((candidate.board, captured, new_ko))
    }

    /// Return the current `N²+1` legal-action mask.
    fn legal_mask(&self) -> Vec<bool> {
        (0..POINT_COUNT)
            .map(|index| self.placement_result(index).is_some())
            .chain(std::iter::once(true))
            .collect()
    }

    /// Build one agent's source-compatible 17-plane observation.
    fn observe(&self, player: usize) -> GoObservation {
        let mut values = Vec::with_capacity(POINT_COUNT * 17);
        values.extend_from_slice(&self.board_history);
        // PettingZoo v5's implementation emits zero for black and one for
        // white, despite the older prose table above stating the reverse.
        values.extend(std::iter::repeat_n(u8::from(player == 1), POINT_COUNT));
        GoObservation {
            values,
            action_mask: if self.rewards.is_none() && player == self.current_player() {
                self.legal_mask()
            } else {
                vec![false; ACTION_COUNT]
            },
        }
    }

    /// Apply one selected-player action and return terminal rewards when done.
    fn play(&mut self, player: usize, action: GoAction) -> Result<Option<[f64; 2]>, GoError> {
        if self.rewards.is_some() {
            return Err(GoError::GameFinished);
        }
        if player != self.current_player() {
            return Err(GoError::WrongPlayer {
                expected: self.current_player() as u8,
                actual: player as u8,
            });
        }
        if let Some(index) = action.board_index() {
            let Some((board, _captured, ko)) = self.placement_result(index) else {
                return Err(GoError::IllegalMove {
                    action: action.0 as u16,
                });
            };
            self.board = board;
            self.ko = ko;
        } else {
            self.ko = None;
        }
        self.recent.push(action);
        self.push_history(self.to_play);
        self.to_play = self.to_play.opponent();
        if self.recent.len() >= 2
            && self
                .recent
                .iter()
                .rev()
                .take(2)
                .all(|recent| recent.board_index().is_none())
        {
            self.rewards = Some(if self.score() > 0.0 {
                [1.0, -1.0]
            } else {
                [-1.0, 1.0]
            });
        }
        Ok(self.rewards)
    }

    /// Insert the acting player's current/opponent planes at the front.
    fn push_history(&mut self, acting: Stone) {
        self.board_history
            .copy_within(0..POINT_COUNT * 14, POINT_COUNT * 2);
        for (index, stone) in self.board.iter().copied().enumerate() {
            if let Some(value) = self.board_history.get_mut(index) {
                *value = u8::from(stone == acting);
            }
            if let Some(value) = self.board_history.get_mut(POINT_COUNT + index) {
                *value = u8::from(stone == acting.opponent());
            }
        }
    }

    /// Return Chinese area score from black's perspective.
    fn score(&self) -> f64 {
        let mut scored = self.board;
        let mut visited = [false; POINT_COUNT];
        for start in 0..POINT_COUNT {
            if scored.get(start).copied() != Some(Stone::Empty)
                || visited.get(start).copied().unwrap_or(true)
            {
                continue;
            }
            let mut region = Vec::new();
            let mut border = HashSet::new();
            let mut frontier = VecDeque::from([start]);
            if let Some(was_visited) = visited.get_mut(start) {
                *was_visited = true;
            }
            while let Some(point) = frontier.pop_front() {
                region.push(point);
                for neighbor in Self::neighbors(point) {
                    match scored.get(neighbor).copied().unwrap_or(Stone::Empty) {
                        Stone::Empty if !visited.get(neighbor).copied().unwrap_or(true) => {
                            if let Some(was_visited) = visited.get_mut(neighbor) {
                                *was_visited = true;
                            }
                            frontier.push_back(neighbor);
                        }
                        Stone::Empty => {}
                        color => {
                            border.insert(color);
                        }
                    }
                }
            }
            if border.len() == 1 {
                if let Some(color) = border.iter().next().copied() {
                    for point in region {
                        if let Some(stone) = scored.get_mut(point) {
                            *stone = color;
                        }
                    }
                }
            }
        }
        let black = scored
            .iter()
            .filter(|stone| **stone == Stone::Black)
            .count();
        let white = scored
            .iter()
            .filter(|stone| **stone == Stone::White)
            .count();
        black as f64 - white as f64 - self.komi
    }
}

/// Seeded random stream used for white's comparison policy.
#[derive(Debug, Clone, Copy, Default)]
struct SplitMix64(u64);

impl SplitMix64 {
    /// Return one uniformly distributed index below `upper`.
    fn index(&mut self, upper: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let Ok(bound) = u64::try_from(upper) else {
            return 0;
        };
        if bound == 0 {
            return 0;
        }
        usize::try_from((value ^ (value >> 31)) % bound).unwrap_or_default()
    }
}

/// Policy-control mode for self-play training and random-first evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    /// One shared policy supplies every AEC action during training.
    SharedPolicy,
    /// The policy controls white against random black.
    PolicyAsWhiteVsRandomBlack,
}

/// Source AEC game with shared-policy training and random-first evaluation.
#[derive(Debug, Clone)]
struct GoTrainingEnv {
    /// Complete raw Go position.
    game: GoGame,
    /// Random comparison action stream reset from the environment seed.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for GoTrainingEnv {
    fn default() -> Self {
        Self {
            game: GoGame::default(),
            rng: SplitMix64::default(),
            control: ControlMode::SharedPolicy,
        }
    }
}

/// Per-transition multi-agent evidence.
#[derive(Debug, Clone, PartialEq)]
struct GoInfo {
    /// Exact observations in black, white order.
    observations: [GoObservation; 2],
    /// Source rewards in black, white order.
    rewards: [f64; 2],
    /// Chinese area score from black's perspective.
    score: f64,
}

impl Default for GoInfo {
    fn default() -> Self {
        let observation = GoObservation {
            values: vec![0; POINT_COUNT * 17],
            action_mask: vec![false; ACTION_COUNT],
        };
        Self {
            observations: [observation.clone(), observation],
            rewards: [0.0; 2],
            score: -KOMI,
        }
    }
}

impl GoTrainingEnv {
    /// Build observations plus exact score and reward evidence.
    fn info(&self, rewards: [f64; 2]) -> GoInfo {
        GoInfo {
            observations: [self.game.observe(0), self.game.observe(1)],
            rewards,
            score: self.game.score(),
        }
    }

    /// Flatten the policy agent's exact observation and legal-action mask.
    fn observation(&self) -> Vec<f32> {
        let player = match self.control {
            ControlMode::SharedPolicy => self.game.current_player(),
            ControlMode::PolicyAsWhiteVsRandomBlack => 1,
        };
        let observation = self.game.observe(player);
        observation
            .values
            .into_iter()
            .map(f32::from)
            .chain(
                observation
                    .action_mask
                    .into_iter()
                    .map(u8::from)
                    .map(f32::from),
            )
            .collect()
    }

    /// Select one uniformly random legal move for the current player.
    fn random_action(&mut self) -> GoAction {
        let legal: Vec<_> = self
            .game
            .legal_mask()
            .into_iter()
            .enumerate()
            .filter_map(|(index, is_legal)| is_legal.then_some(index))
            .collect();
        let action = legal
            .get(self.rng.index(legal.len()))
            .copied()
            .unwrap_or(POINT_COUNT);
        GoAction::from_index(action)
    }
}

impl Env for GoTrainingEnv {
    type Observation = Vec<f32>;
    type Action = GoAction;
    type Info = GoInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        self.game.reset();
        self.rng = SplitMix64(seed.unwrap_or_default());
        Reset {
            observation: self.observation(),
            info: self.info([0.0; 2]),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let actor = self.game.current_player();
        let result = self.game.play(actor, action);
        let (rewards, status) = match result {
            Err(_error) if actor == 0 => ([-1.0, 0.0], EpisodeStatus::Terminated),
            Err(_error) => ([0.0, -1.0], EpisodeStatus::Terminated),
            Ok(Some(rewards)) => (rewards, EpisodeStatus::Terminated),
            Ok(None) => ([0.0; 2], EpisodeStatus::Continuing),
        };
        let [black_reward, white_reward] = rewards;
        Step {
            observation: self.observation(),
            reward: match self.control {
                ControlMode::SharedPolicy if actor == 0 => black_reward,
                ControlMode::SharedPolicy | ControlMode::PolicyAsWhiteVsRandomBlack => white_reward,
            },
            status,
            info: self.info(rewards),
        }
    }
}

impl DiscreteDqnExample for GoTrainingEnv {
    const ENV_NAME: &'static str = "go-v5";
    const GYMNASIUM_ID: &'static str = "classic/go-v5";
    const OBSERVATION_DIM: usize = POINT_COUNT * 17;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const DEFAULT_TRAIN_STEPS: usize = 2_000_000;
    const DEFAULT_EVAL_INTERVAL: usize = 50_000;
    const DEFAULT_EVAL_EPISODES: usize = 200;
    const DEFAULT_NUM_ENVS: usize = 8;
    const ZERO_SUM_COMPARISON: bool = true;
    const BOOTSTRAP_MULTIPLIER: f32 = -1.0;
    const MIN_TRAIN_STEPS: usize = 250_000;
    const SOLVED_MEAN_REWARD: f64 = 0.0;
    const GIF_PATH: &'static str = "docs/images/classic-go-v5.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 0.99,
            learning_rate,
            replay_capacity: 250_000,
            min_replay_size: 1_024,
            batch_size: 128,
            target_update_interval: 2_000,
            epsilon_decay_steps: 1_000_000,
            ..DqnConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation.iter().take(POINT_COUNT * 17).copied().collect()
    }

    fn action_mask(observation: &[f32]) -> Vec<bool> {
        observation
            .iter()
            .skip(POINT_COUNT * 17)
            .take(ACTION_COUNT)
            .map(|value| *value > 0.5)
            .collect()
    }

    fn prepare_evaluation(&mut self) {
        self.control = ControlMode::PolicyAsWhiteVsRandomBlack;
    }

    fn evaluation_action(
        &mut self,
        policy: &bevy_gym::training::DqnPolicy,
        observation: &[f32],
    ) -> Result<Self::Action, bevy_gym::training::DqnError> {
        if self.game.current_player() == 0 {
            Ok(self.random_action())
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
    run_discrete_workflow::<GoTrainingEnv>()
}

/// PettingZoo tile-and-stone renderer for the default board.
#[cfg(feature = "render")]
mod render {
    use super::{
        DiscreteDqnExample, Error, GoAction, GoGame, GoTrainingEnv, Path, SplitMix64, Stone,
        ACTION_COUNT, BOARD_SIZE, POINT_COUNT,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;
    use bevy_inspector_egui as _;

    /// Dimensions of the checked-in reference GIF.
    const SCREEN_SIZE: f32 = 1_026.0;
    /// Integer viewport edge used by window and encoder APIs.
    const SCREEN_SIZE_PIXELS: u32 = 1_026;
    /// Source board cell width and height.
    const TILE_SIZE: f32 = SCREEN_SIZE / BOARD_SIZE as f32;
    /// Source tile image size, including overlap.
    const TILE_IMAGE_SIZE: f32 = TILE_SIZE * 7.0 / 6.0;
    /// Source stone diameter.
    const STONE_SIZE: f32 = TILE_SIZE * 5.0 / 6.0;

    /// Configure the square Go viewport.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::BLACK));
        }
    }

    /// Loaded policy and raw visible position.
    #[derive(Resource)]
    struct VisualGame {
        /// Greedy player-zero DQN policy.
        policy: DqnPolicy,
        /// Exact visible Go game.
        game: GoGame,
        /// Seeded player-one action stream.
        rng: SplitMix64,
    }

    /// Stone sprite attached to one flat point.
    #[derive(Component)]
    struct StoneSprite(usize);

    /// Source two-Hz playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run interactive playback or a finite reference-sized capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = GoTrainingEnv::dqn_config(GoTrainingEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(
            checkpoint,
            POINT_COUNT * 17,
            ACTION_COUNT,
            &config.hidden_sizes,
        )?;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualGame {
                policy,
                game: GoGame::default(),
                rng: SplitMix64(12_345),
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.5, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets/go".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Go".into(),
                            resolution: WindowResolution::new(
                                SCREEN_SIZE_PIXELS,
                                SCREEN_SIZE_PIXELS,
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
            app.add_systems(Update, (advance_watch, sync_stones).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    /// Spawn all source board tiles and reusable stone slots.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("Go Camera")));
        for row in 0..BOARD_SIZE {
            for column in 0..BOARD_SIZE {
                let index = row * BOARD_SIZE + column;
                let tile = tile_index(row, column);
                commands.spawn((
                    Sprite {
                        image: assets.load(format!("img/GO_Tile{tile}.png")),
                        custom_size: Some(Vec2::splat(TILE_IMAGE_SIZE)),
                        ..default()
                    },
                    Transform::from_translation(point_position(row, column, 0.0)),
                    Name::new(format!("Go Tile {row},{column}")),
                ));
                commands.spawn((
                    Sprite::default(),
                    Transform::from_translation(point_position(row, column, 1.0)),
                    Visibility::Hidden,
                    StoneSprite(index),
                ));
            }
        }
    }

    /// Return PettingZoo's interior, edge, or corner tile ID.
    const fn tile_index(row: usize, column: usize) -> usize {
        match (row, column) {
            (0, 0) => 5,
            (0, column) if column == BOARD_SIZE - 1 => 8,
            (row, 0) if row == BOARD_SIZE - 1 => 6,
            (row, column) if row == BOARD_SIZE - 1 && column == BOARD_SIZE - 1 => 7,
            (0, _) => 1,
            (_, 0) => 2,
            (row, _) if row == BOARD_SIZE - 1 => 3,
            (_, column) if column == BOARD_SIZE - 1 => 4,
            _ => 0,
        }
    }

    /// Convert source top-left grid placement into centered Bevy coordinates.
    fn point_position(row: usize, column: usize, z: f32) -> Vec3 {
        Vec3::new(
            (row as f32).mul_add(TILE_SIZE, TILE_IMAGE_SIZE / 2.0) - SCREEN_SIZE / 2.0,
            (column as f32).mul_add(-TILE_SIZE, SCREEN_SIZE / 2.0) - TILE_IMAGE_SIZE / 2.0,
            z,
        )
    }

    /// Advance one raw AEC action at two FPS.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualGame>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Apply one random black move, one learned white move, or a reset.
    fn advance_visual(visual: &mut VisualGame) {
        if visual.game.rewards.is_some() {
            visual.game.reset();
            return;
        }
        let player = visual.game.current_player();
        let action = if player == 1 {
            let observation = GoTrainingEnv {
                game: visual.game.clone(),
                rng: visual.rng,
                control: super::ControlMode::PolicyAsWhiteVsRandomBlack,
            }
            .observation();
            match GoTrainingEnv::deployment_action(&visual.policy, &observation) {
                Ok(action) => action,
                Err(_error) => return,
            }
        } else {
            let legal: Vec<_> = visual
                .game
                .legal_mask()
                .into_iter()
                .enumerate()
                .filter_map(|(index, is_legal)| is_legal.then_some(index))
                .collect();
            match legal.get(visual.rng.index(legal.len())).copied() {
                Some(index) => GoAction::from_index(index),
                None => return,
            }
        };
        if visual.game.play(player, action).is_err() {
            visual.game.reset();
        }
    }

    /// Synchronize black and white stone images from the raw board.
    fn sync_stones(
        visual: Res<'_, VisualGame>,
        assets: Res<'_, AssetServer>,
        mut stones: Query<'_, '_, (&StoneSprite, &mut Sprite, &mut Visibility)>,
    ) {
        if visual.is_changed() {
            apply_stones(&visual, &assets, &mut stones);
        }
    }

    /// Update every reusable stone slot.
    fn apply_stones(
        visual: &VisualGame,
        assets: &AssetServer,
        stones: &mut Query<'_, '_, (&StoneSprite, &mut Sprite, &mut Visibility)>,
    ) {
        for (slot, mut sprite, mut visibility) in stones.iter_mut() {
            let image = match visual
                .game
                .board
                .get(slot.0)
                .copied()
                .unwrap_or(Stone::Empty)
            {
                Stone::Empty => {
                    *visibility = Visibility::Hidden;
                    continue;
                }
                Stone::Black => "img/GoBlackPiece.png",
                Stone::White => "img/GoWhitePiece.png",
            };
            sprite.image = assets.load(image);
            sprite.custom_size = Some(Vec2::splat(STONE_SIZE));
            *visibility = Visibility::Visible;
        }
    }

    /// Capture five seconds at 20 FPS while advancing at two FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualGame>,
        assets: Res<'_, AssetServer>,
        mut stones: Query<'_, '_, (&StoneSprite, &mut Sprite, &mut Visibility)>,
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
            advance_visual(&mut visual);
        }
        apply_stones(&visual, &assets, &mut stones);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Encode the source-sized five-second demonstration.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-go-v5",
            SCREEN_SIZE_PIXELS,
            SCREEN_SIZE_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty 19 by 19 board exposes every point and pass.
    #[test]
    fn reset_exposes_all_362_actions_to_black_only() {
        let game = GoGame::default();

        assert_eq!(
            game.legal_mask().iter().filter(|legal| **legal).count(),
            ACTION_COUNT
        );
        assert!(game.observe(0).action_mask.into_iter().all(|legal| legal));
        assert!(game.observe(1).action_mask.into_iter().all(|legal| !legal));
    }

    /// Surrounding a corner chain removes it from the board.
    #[test]
    fn placement_captures_a_chain_without_liberties() {
        let mut game = GoGame::default();
        assert_eq!(game.play(0, GoAction::point(1, 0)), Ok(None));
        assert_eq!(game.play(1, GoAction::point(0, 0)), Ok(None));
        assert_eq!(game.play(0, GoAction::point(0, 1)), Ok(None));

        assert_eq!(game.board[GoAction::point(0, 0).0], Stone::Empty);
    }

    /// A surrounded placement that captures nothing is illegal.
    #[test]
    fn suicide_is_rejected() {
        let mut game = GoGame::default();
        for point in [(8, 9), (10, 9), (9, 8), (9, 10)] {
            game.board[GoAction::point(point.0, point.1).0] = Stone::Black;
        }
        game.to_play = Stone::White;

        assert_eq!(
            game.play(1, GoAction::point(9, 9)),
            Err(GoError::IllegalMove {
                action: GoAction::point(9, 9).0 as u16,
            })
        );
    }

    /// A one-stone recapture is forbidden until ko clears.
    #[test]
    fn simple_ko_prevents_immediate_recapture() {
        let mut game = GoGame::default();
        for point in [(0, 2), (2, 2), (1, 3)] {
            game.board[GoAction::point(point.0, point.1).0] = Stone::Black;
        }
        for point in [(0, 1), (2, 1), (1, 0)] {
            game.board[GoAction::point(point.0, point.1).0] = Stone::White;
        }
        game.board[GoAction::point(1, 2).0] = Stone::White;

        assert_eq!(game.play(0, GoAction::point(1, 1)), Ok(None));
        assert_eq!(game.ko, Some(GoAction::point(1, 2).0));
        assert_eq!(
            game.play(1, GoAction::point(1, 2)),
            Err(GoError::IllegalMove {
                action: GoAction::point(1, 2).0 as u16,
            })
        );
    }

    /// Frame stacking records the acting perspective before changing turns.
    #[test]
    fn observation_matches_source_history_and_player_plane() {
        let mut game = GoGame::default();
        let center = GoAction::point(9, 9).0;
        assert_eq!(game.play(0, GoAction(center)), Ok(None));
        let white = game.observe(1);

        assert_eq!(white.values.len(), POINT_COUNT * 17);
        assert_eq!(white.values[center], 1);
        assert_eq!(white.values[POINT_COUNT + center], 0);
        assert!(white.values[POINT_COUNT * 16..]
            .iter()
            .all(|value| *value == 1));
    }

    /// Two passes score an empty default board for white by komi.
    #[test]
    fn consecutive_passes_end_and_score_the_game() {
        let mut game = GoGame::default();

        assert_eq!(game.play(0, GoAction::pass()), Ok(None));
        assert_eq!(game.play(1, GoAction::pass()), Ok(Some([-1.0, 1.0])));
        assert_eq!(game.score(), -KOMI);
    }

    /// Training preserves AEC turns and evaluation randomizes player zero.
    #[test]
    fn training_preserves_aec_turns_and_evaluation_randomizes_black() {
        let mut first = GoTrainingEnv::default();
        let reset = first.reset(Some(7));
        assert_eq!(reset.observation.len(), POINT_COUNT * 17 + ACTION_COUNT);
        assert!(GoTrainingEnv::action_mask(&reset.observation)
            .into_iter()
            .all(|legal| legal));
        let first_step = first.step(GoAction::point(9, 9));

        let occupied = first
            .game
            .board
            .iter()
            .filter(|stone| **stone != Stone::Empty)
            .count();
        assert_eq!(occupied, 1);
        assert_eq!(first.game.current_player(), 1);
        assert_eq!(first_step.observation, first.observation());

        let mut evaluation = GoTrainingEnv::default();
        evaluation.prepare_evaluation();
        evaluation.reset(Some(7));
        assert_eq!(evaluation.game.current_player(), 0);
        let random_action = evaluation.random_action();
        let reset = evaluation.step(random_action);
        assert_eq!(evaluation.game.current_player(), 1);
        assert_eq!(
            evaluation
                .game
                .board
                .iter()
                .filter(|stone| **stone == Stone::Black)
                .count(),
            1
        );
        assert_eq!(reset.observation, evaluation.observation());
    }
}
