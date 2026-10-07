//!
//! # Chess
//!
//! ```{figure} classic_chess.gif
//! :width: 140px
//! :name: chess
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/chess-v6")`  |
//! |--------------------|------------------------------------|
//! | Actions            | Discrete                           |
//! | Parallel API       | Yes                                |
//! | Manual Control     | No                                 |
//! | Agents             | `agents= ['player_0', 'player_1']` |
//! | Agents             | 2                                  |
//! | Action Shape       | Discrete(4672)                     |
//! | Action Values      | Discrete(4672)                     |
//! | Observation Shape  | (8,8,111)                          |
//! | Observation Values | [0,1]                              |
//!
//!
//! Chess is one of the oldest studied games in AI. Our implementation of the observation and action spaces for chess are what the AlphaZero method uses, with two small changes.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//! Like AlphaZero, the main observation space is an 8x8 image representing the board. It has 111 channels representing:
//!
//! * Channels 0 - 3: Castling rights:
//!   * Channel 0: All ones if white can castle queenside
//!   * Channel 1: All ones if white can castle kingside
//!   * Channel 2: All ones if black can castle queenside
//!   * Channel 3: All ones if black can castle kingside
//! * Channel 4: Is black or white
//! * Channel 5: A move clock counting up to the 50 move rule. Represented by a single channel where the *n* th element in the flattened channel is set if there has been *n* moves
//! * Channel 6: All ones to help neural networks find board edges in padded convolutions
//! * Channel 7 - 18: One channel for each piece type and player color combination. For example, there is a specific channel that represents black knights. An index of this channel is set to 1 if a black knight is in the corresponding spot on the game board, otherwise, it is set to 0.
//! Similar to LeelaChessZero, en passant possibilities are represented by displaying the vulnerable pawn on the 8th row instead of the 5th.
//! * Channel 19: represents whether a position has been seen before (whether a position is a 2-fold repetition)
//! * Channel 20 - 111 represents the previous 7 boards, with each board represented by 13 channels. The latest board occupies the first 13 channels, followed by the second latest board, and so on. These 13 channels correspond to channels 7 - 20.
//!
//!
//! Similar to AlphaZero, our observation space follows a stacking approach, where it accumulates the previous 8 board observations.
//!
//! Unlike AlphaZero, where the board orientation may vary, in our system, the `env.board_history` always maintains the orientation towards the white agent, with the white agent's king consistently positioned on the 1st row. In simpler terms, both players are observing the same board layout.
//!
//! Nevertheless, we have incorporated a convenient feature, the env.observe('player_1') function, specifically for the black agent's orientation. This facilitates the training of agents capable of playing proficiently as both black and white.
//!
//! #### Legal Actions Mask
//!
//! The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
//! whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
//!
//! ### Action Space
//!
//! From the AlphaZero chess paper:
//!
//! > [In AlphaChessZero, the] action space is a 8x8x73 dimensional array.
//! Each of the 8×8 positions identifies the square from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
//! moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
//! next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
//! underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
//! rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
//! queen.
//!
//! We instead flatten this into 8×8×73 = 4672 discrete action space.
//!
//! You can get back the original (x,y,c) coordinates from the integer action `a` with the following expression: `(a // (8*73), (a // 73) % 8, a % (8*73) % 73)`
//!
//! Example:
//!     >>> x = 6
//!     >>> y = 0
//!     >>> c = 12
//!     >>> a = x*(8*73) + y*73 + c
//!     >>> print(a // (8*73), a % (8*73) // 73, a % (8*73) % 73)
//!     6 0 12
//!
//! Note: the coordinates (6, 0, 12) correspond to column 6, row 0, plane 12. In chess notation, this would signify square G1:
//!
//! | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
//! | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
//! | A | B | C | D | E | F | G | H |
//!
//! ### Rewards
//!
//! | Winner | Loser | Draw |
//! | :----: | :---: | :---: |
//! | +1     | -1    | 0 |
//!
//! ### Version History
//!
//! * v6: Fixed wrong player starting first, check for insufficient material/50-turn rule/three fold repetition (1.23.2)
//! * v5: Changed python-chess version to version 1.7 (1.13.1)
//! * v4: Changed observation space to proper AlphaZero style frame stacking (1.11.0)
//! * v3: Fixed bug in arbitrary calls to observe() (1.8.0)
//! * v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
//! * v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
//! * v0: Initial versions release (1.0.0)
//!

#![expect(
    clippy::doc_markdown,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]
#![expect(
    clippy::doc_lazy_continuation,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]

use std::collections::VecDeque;
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
use tokio as _;

use bevy_gym::training::{
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction, DqnConfig,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};
#[cfg(test)]
use shakmaty::uci::UciMove;
use shakmaty::{
    CastlingSide, Chess, Color, EnPassantMode, KnownOutcome, Move, Position, Role, Square,
};

/// AlphaZero's flattened `8 x 8 x 73` action count.
const ACTION_COUNT: usize = 4_672;
/// PettingZoo's `8 x 8 x 111` observation width.
const OBSERVATION_SIZE: usize = 7_104;
/// One source board-history frame has 13 planes.
const FRAME_SIZE: usize = 8 * 8 * 13;

/// PettingZoo's two AEC agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Player {
    /// `player_0`, the white starting player.
    White,
    /// `player_1`, the black player.
    Black,
}

impl Player {
    /// Convert the chess-library color boundary into an agent.
    const fn from_color(color: Color) -> Self {
        match color {
            Color::White => Self::White,
            Color::Black => Self::Black,
        }
    }
}

/// One exact flattened AlphaZero action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ChessAction(u16);

impl ChessAction {
    /// Keep any table output inside the source action space.
    const fn from_index(index: usize) -> Self {
        Self((index % ACTION_COUNT) as u16)
    }

    /// Return the exact source action integer.
    const fn index(self) -> usize {
        self.0 as usize
    }
}

impl DqnAction for ChessAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.index()
    }
}

/// Exact PettingZoo observation dictionary for one agent.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChessObservation {
    /// Row-major `8 x 8 x 111` binary observation.
    planes: Vec<u8>,
    /// Legal source actions for the selected agent.
    action_mask: Vec<bool>,
}

/// Raw move rejection before the illegal-action wrapper is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MoveError {
    /// The caller supplied the unselected AEC agent.
    WrongPlayer {
        /// Agent selected by the chess position.
        expected: Player,
        /// Agent supplied by the caller.
        actual: Player,
    },
    /// The selected action does not encode a legal move in this position.
    IllegalAction(ChessAction),
    /// The position has already reached a terminal result.
    GameFinished(KnownOutcome),
}

impl fmt::Display for MoveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPlayer { expected, actual } => {
                write!(formatter, "expected {expected:?}, received {actual:?}")
            }
            Self::IllegalAction(action) => {
                write!(formatter, "action {} is illegal", action.index())
            }
            Self::GameFinished(outcome) => write!(formatter, "game finished with {outcome}"),
        }
    }
}

impl Error for MoveError {}

/// Full source-compatible chess position and eight-frame history.
#[derive(Debug, Clone)]
struct ChessGame {
    /// Legal standard-chess position.
    position: Chess,
    /// Newest-first white-oriented 13-plane history.
    history: VecDeque<[u8; FRAME_SIZE]>,
    /// Every reached position for two- and three-fold repetition checks.
    positions: Vec<Chess>,
    /// Claimed or forced terminal result.
    outcome: Option<KnownOutcome>,
}

impl Default for ChessGame {
    fn default() -> Self {
        let position = Chess::default();
        Self {
            positions: vec![position.clone()],
            position,
            history: VecDeque::with_capacity(8),
            outcome: None,
        }
    }
}

impl ChessGame {
    /// Restore PettingZoo's initial board and zero-filled history.
    fn reset(&mut self) {
        *self = Self::default();
    }

    /// Return the agent selected by the position's side to move.
    fn selected_player(&self) -> Player {
        Player::from_color(self.position.turn())
    }

    /// Return exact source action and legal move pairs for this position.
    fn legal_action_moves(&self) -> Vec<(ChessAction, Move)> {
        self.position
            .legal_moves()
            .into_iter()
            .map(|chess_move| (move_to_action(self.position.turn(), chess_move), chess_move))
            .collect()
    }

    /// Resolve one UCI move through the same legal-move boundary used by tests.
    #[cfg(test)]
    fn action_for_uci(&self, uci: &str) -> Result<ChessAction, Box<dyn Error>> {
        let uci_move: UciMove = uci.parse()?;
        let chess_move = uci_move.to_move(&self.position)?;
        Ok(move_to_action(self.position.turn(), chess_move))
    }

    /// Apply one exact action after checking selected-agent and legal-mask rules.
    fn play(
        &mut self,
        player: Player,
        action: ChessAction,
    ) -> Result<Option<KnownOutcome>, MoveError> {
        if let Some(outcome) = self.outcome {
            return Err(MoveError::GameFinished(outcome));
        }
        let expected = self.selected_player();
        if player != expected {
            return Err(MoveError::WrongPlayer {
                expected,
                actual: player,
            });
        }
        let Some((_encoded, chess_move)) = self
            .legal_action_moves()
            .into_iter()
            .find(|(encoded, _chess_move)| *encoded == action)
        else {
            return Err(MoveError::IllegalAction(action));
        };

        self.position = self
            .position
            .clone()
            .play(chess_move)
            .map_err(|_error| MoveError::IllegalAction(action))?;
        self.positions.push(self.position.clone());
        self.push_history_frame();
        self.outcome = self.detect_outcome();
        Ok(self.outcome)
    }

    /// Apply one known legal UCI move for a deterministic rules fixture.
    #[cfg(test)]
    fn play_uci(&mut self, uci: &str) -> Result<Option<KnownOutcome>, Box<dyn Error>> {
        let player = self.selected_player();
        let action = self.action_for_uci(uci)?;
        Ok(self.play(player, action)?)
    }

    /// Build one agent's exact observation and selected-agent action mask.
    fn observe(&self, player: Player) -> ChessObservation {
        let mut planes = vec![0_u8; OBSERVATION_SIZE];
        for row in 0..8 {
            for column in 0..8 {
                for channel in 0..7 {
                    if let Some(cell) = planes.get_mut((row * 8 + column) * 111 + channel) {
                        *cell = self.aux_value(player, row, column, channel);
                    }
                }
                for history_channel in 0..104 {
                    let frame_index = history_channel / 13;
                    let mut channel = history_channel % 13;
                    let source_row = if player == Player::Black {
                        7 - row
                    } else {
                        row
                    };
                    if player == Player::Black {
                        channel = match channel {
                            0..=5 => channel + 6,
                            6..=11 => channel - 6,
                            _ => channel,
                        };
                    }
                    if let Some(value) = self.history.get(frame_index).and_then(|frame| {
                        frame.get((source_row * 8 + column) * 13 + channel).copied()
                    }) {
                        if let Some(cell) =
                            planes.get_mut((row * 8 + column) * 111 + 7 + history_channel)
                        {
                            *cell = value;
                        }
                    }
                }
            }
        }
        let mut action_mask = vec![false; ACTION_COUNT];
        if self.outcome.is_none() && player == self.selected_player() {
            for (action, _chess_move) in self.legal_action_moves() {
                if let Some(is_legal) = action_mask.get_mut(action.index()) {
                    *is_legal = true;
                }
            }
        }
        ChessObservation {
            planes,
            action_mask,
        }
    }

    /// Return one of PettingZoo's seven current-position auxiliary values.
    fn aux_value(&self, player: Player, row: usize, column: usize, channel: usize) -> u8 {
        let (first, second) = if player == Player::White {
            (Color::White, Color::Black)
        } else {
            (Color::Black, Color::White)
        };
        match channel {
            0 => u8::from(self.position.castles().has(first, CastlingSide::KingSide)),
            1 => u8::from(self.position.castles().has(first, CastlingSide::QueenSide)),
            2 => u8::from(self.position.castles().has(second, CastlingSide::KingSide)),
            3 => u8::from(self.position.castles().has(second, CastlingSide::QueenSide)),
            4 => u8::from(player == Player::Black),
            5 => {
                let point = usize::try_from(self.position.halfmoves() / 2).unwrap_or(usize::MAX);
                let source_row = if player == Player::Black {
                    7 - row
                } else {
                    row
                };
                u8::from(source_row * 8 + column == point)
            }
            6 => 1,
            _ => 0,
        }
    }

    /// Push the current white-oriented piece and repetition planes.
    fn push_history_frame(&mut self) {
        let mut frame = [0_u8; FRAME_SIZE];
        for (square, piece) in self.position.board() {
            let row = 7 - square.rank().to_u32() as usize;
            let column = square.file().to_u32() as usize;
            let color_offset = if piece.color == Color::White { 0 } else { 6 };
            let channel = color_offset + role_index(piece.role);
            if let Some(cell) = frame.get_mut((row * 8 + column) * 13 + channel) {
                *cell = 1;
            }
        }
        self.apply_en_passant_plane(&mut frame);
        if self.repetition_count(&self.position) >= 2 {
            for cell in frame.chunks_exact_mut(13) {
                if let Some(repetition) = cell.last_mut() {
                    *repetition = 1;
                }
            }
        }
        self.history.push_front(frame);
        self.history.truncate(8);
    }

    /// Apply PettingZoo's Leela-style vulnerable-pawn back-rank encoding.
    fn apply_en_passant_plane(&self, frame: &mut [u8; FRAME_SIZE]) {
        let Some(target) = self.position.ep_square(EnPassantMode::Always) else {
            return;
        };
        let target_index = target.to_u32() as usize;
        let file = target.file().to_u32() as usize;
        let (pawn_square, destination_row, channel) = if target_index < 32 {
            (target_index + 8, 7, 0)
        } else {
            (target_index - 8, 0, 6)
        };
        let pawn_row = 7 - pawn_square / 8;
        let pawn_column = pawn_square % 8;
        if let Some(cell) = frame.get_mut((pawn_row * 8 + pawn_column) * 13 + channel) {
            *cell = 0;
        }
        if let Some(cell) = frame.get_mut((destination_row * 8 + file) * 13 + channel) {
            *cell = 1;
        }
    }

    /// Return how often a FIDE-equivalent position appears in game history.
    fn repetition_count(&self, position: &Chess) -> usize {
        self.positions
            .iter()
            .filter(|candidate| *candidate == position)
            .count()
    }

    /// Reproduce forced ends plus Python-chess's claimable draw checks.
    fn detect_outcome(&self) -> Option<KnownOutcome> {
        if let Some(outcome) = self.position.outcome().known() {
            return Some(outcome);
        }
        let current_threefold = self.repetition_count(&self.position) >= 3;
        let can_claim_threefold = current_threefold
            || self.position.legal_moves().into_iter().any(|chess_move| {
                self.position
                    .clone()
                    .play(chess_move)
                    .is_ok_and(|next| self.repetition_count(&next) >= 2)
            });
        let can_claim_fifty = self.position.halfmoves() >= 100
            || (self.position.halfmoves() >= 99
                && self
                    .position
                    .legal_moves()
                    .into_iter()
                    .any(|chess_move| !chess_move.is_zeroing()));
        (can_claim_threefold || can_claim_fifty).then_some(KnownOutcome::Draw)
    }
}

/// Convert one chess role into PettingZoo's plane order.
const fn role_index(role: Role) -> usize {
    match role {
        Role::Pawn => 0,
        Role::Knight => 1,
        Role::Bishop => 2,
        Role::Rook => 3,
        Role::Queen => 4,
        Role::King => 5,
    }
}

/// Encode a legal move in PettingZoo's exact AlphaZero plane mapping.
fn move_to_action(turn: Color, chess_move: Move) -> ChessAction {
    let oriented = if turn == Color::Black {
        chess_move.to_mirrored()
    } else {
        chess_move
    };
    let from = oriented
        .from()
        .expect("standard chess has no drop moves in legal_moves");
    let to = match oriented {
        Move::Castle { king, rook } => {
            if king < rook {
                Square::G1
            } else {
                Square::C1
            }
        }
        _ => oriented.to(),
    };
    let delta_file = to.file().to_u32() as i32 - from.file().to_u32() as i32;
    let delta_rank = to.rank().to_u32() as i32 - from.rank().to_u32() as i32;
    let plane = if matches!(
        oriented.promotion(),
        Some(Role::Knight | Role::Bishop | Role::Rook)
    ) {
        let promotion = oriented.promotion().map_or(1, role_index).saturating_sub(1);
        64 + 3 * usize::try_from(delta_file + 1).expect("promotion file delta is -1..=1")
            + promotion
    } else if delta_file.abs() + delta_rank.abs() == 3 && (1..=2).contains(&delta_file.abs()) {
        let mut counter = 0;
        let mut selected = 0;
        for file in -2_i32..=2 {
            for rank in -2_i32..=2 {
                if file.abs() + rank.abs() == 3 {
                    if (file, rank) == (delta_file, delta_rank) {
                        selected = counter;
                    }
                    counter += 1;
                }
            }
        }
        56 + selected
    } else {
        let step_file = delta_file.signum();
        let step_rank = delta_rank.signum();
        let magnitude = usize::try_from(delta_file.unsigned_abs().max(delta_rank.unsigned_abs()))
            .unwrap_or(1)
            .saturating_sub(1);
        let mut counter = 0;
        let mut direction = 0;
        for file in -1..=1 {
            for rank in -1..=1 {
                if file == 0 && rank == 0 {
                    continue;
                }
                if (file, rank) == (step_file, step_rank) {
                    direction = counter;
                }
                counter += 1;
            }
        }
        magnitude * 8 + direction
    };
    let source = from.file().to_u32() as usize * 8 + from.rank().to_u32() as usize;
    ChessAction((source * 73 + plane) as u16)
}

/// Multi-agent details returned by the random-opponent adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ChessInfo {
    /// Exact observations in player order.
    observations: [ChessObservation; 2],
    /// Source integer rewards in player order.
    rewards: [i8; 2],
}

impl Default for ChessInfo {
    fn default() -> Self {
        let observation = ChessObservation {
            planes: vec![0; OBSERVATION_SIZE],
            action_mask: vec![false; ACTION_COUNT],
        };
        Self {
            observations: [observation.clone(), observation],
            rewards: [0; 2],
        }
    }
}

/// Seeded random stream used for the comparison opponent.
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
    /// The policy controls black against random white.
    PolicyAsBlackVsRandomWhite,
}

/// Source AEC game with shared-policy training and random-first evaluation.
#[derive(Debug, Clone)]
struct ChessTrainingEnv {
    /// Complete raw chess game.
    game: ChessGame,
    /// Random comparison action stream reset from the environment seed.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for ChessTrainingEnv {
    fn default() -> Self {
        Self {
            game: ChessGame::default(),
            rng: SplitMix64::default(),
            control: ControlMode::SharedPolicy,
        }
    }
}

impl ChessTrainingEnv {
    /// Return source observations and one reward pair.
    fn info(&self, rewards: [i8; 2]) -> ChessInfo {
        ChessInfo {
            observations: [
                self.game.observe(Player::White),
                self.game.observe(Player::Black),
            ],
            rewards,
        }
    }

    /// Convert an exact chess result into PettingZoo's zero-sum rewards.
    const fn rewards(outcome: Option<KnownOutcome>) -> [i8; 2] {
        match outcome {
            Some(KnownOutcome::Decisive {
                winner: Color::White,
            }) => [1, -1],
            Some(KnownOutcome::Decisive {
                winner: Color::Black,
            }) => [-1, 1],
            Some(KnownOutcome::Draw) | None => [0, 0],
        }
    }

    /// Flatten the policy agent's exact observation and legal-action mask.
    fn observation(&self) -> Vec<f32> {
        let player = match self.control {
            ControlMode::SharedPolicy => self.game.selected_player(),
            ControlMode::PolicyAsBlackVsRandomWhite => Player::Black,
        };
        let observation = self.game.observe(player);
        observation
            .planes
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
    fn random_action(&mut self) -> Option<ChessAction> {
        let legal = self.game.legal_action_moves();
        legal
            .get(self.rng.index(legal.len()))
            .map(|(action, _chess_move)| *action)
    }
}

impl Env for ChessTrainingEnv {
    type Observation = Vec<f32>;
    type Action = ChessAction;
    type Info = ChessInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        self.game.reset();
        self.rng = SplitMix64(seed.unwrap_or_default());
        Reset {
            observation: self.observation(),
            info: self.info([0; 2]),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let actor = self.game.selected_player();
        let result = self.game.play(actor, action);
        let (rewards, status) = match result {
            Ok(outcome) => (
                Self::rewards(outcome),
                if outcome.is_some() {
                    EpisodeStatus::Terminated
                } else {
                    EpisodeStatus::Continuing
                },
            ),
            Err(_error) => ([-1, 0], EpisodeStatus::Terminated),
        };
        let [white_reward, black_reward] = rewards;
        Step {
            observation: self.observation(),
            reward: f64::from(match self.control {
                ControlMode::SharedPolicy if actor == Player::White => white_reward,
                ControlMode::SharedPolicy | ControlMode::PolicyAsBlackVsRandomWhite => black_reward,
            }),
            status,
            info: self.info(rewards),
        }
    }
}

impl DiscreteDqnExample for ChessTrainingEnv {
    const ENV_NAME: &'static str = "chess-v6";
    const GYMNASIUM_ID: &'static str = "classic/chess-v6";
    const OBSERVATION_DIM: usize = OBSERVATION_SIZE;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const DEFAULT_TRAIN_STEPS: usize = 1_000_000;
    const DEFAULT_EVAL_INTERVAL: usize = 25_000;
    const DEFAULT_EVAL_EPISODES: usize = 200;
    const DEFAULT_NUM_ENVS: usize = 8;
    const ZERO_SUM_COMPARISON: bool = true;
    const BOOTSTRAP_MULTIPLIER: f32 = -1.0;
    const MIN_TRAIN_STEPS: usize = 100_000;
    const SOLVED_MEAN_REWARD: f64 = 0.0;
    const GIF_PATH: &'static str = "docs/images/classic-chess-v6.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 0.99,
            learning_rate,
            replay_capacity: 200_000,
            min_replay_size: 1_024,
            batch_size: 128,
            target_update_interval: 1_000,
            epsilon_decay_steps: 500_000,
            ..DqnConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation.iter().take(OBSERVATION_SIZE).copied().collect()
    }

    fn action_mask(observation: &[f32]) -> Vec<bool> {
        observation
            .iter()
            .skip(OBSERVATION_SIZE)
            .take(ACTION_COUNT)
            .map(|value| *value > 0.5)
            .collect()
    }

    fn prepare_evaluation(&mut self) {
        self.control = ControlMode::PolicyAsBlackVsRandomWhite;
    }

    fn evaluation_action(
        &mut self,
        policy: &bevy_gym::training::DqnPolicy,
        observation: &[f32],
    ) -> Result<Self::Action, bevy_gym::training::DqnError> {
        if self.game.selected_player() == Player::White {
            Ok(self
                .random_action()
                .unwrap_or_else(|| ChessAction::from_index(0)))
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
    run_discrete_workflow::<ChessTrainingEnv>()
}

/// Source-image chess renderer with two-FPS AEC playback.
#[cfg(feature = "render")]
mod render {
    use super::{
        ChessGame, ChessTrainingEnv, DiscreteDqnExample, Error, Path, Player, Role, SplitMix64,
        ACTION_COUNT, OBSERVATION_SIZE,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;
    use bevy_inspector_egui as _;
    use shakmaty::{Color as ChessColor, Position, Square};

    /// Documentation GIF viewport and source square scale.
    const VIEWPORT: f32 = 600.0;
    /// Integer viewport edge used by window and encoder APIs.
    const VIEWPORT_PIXELS: u32 = 600;
    /// Eight equal source board cells.
    const CELL_SIZE: f32 = VIEWPORT / 8.0;

    /// White clear color behind the opaque board asset.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Checkpoint policy and visible source-compatible game.
    #[derive(Resource)]
    struct VisualGame {
        /// Greedy player-zero DQN policy.
        policy: DqnPolicy,
        /// Exact visible chess position.
        game: ChessGame,
        /// Seeded player-one action stream.
        rng: SplitMix64,
    }

    /// Handles for all twelve source piece images.
    #[derive(Resource)]
    struct PieceImages {
        /// White images in pawn-through-king role order.
        white: [Handle<Image>; 6],
        /// Black images in pawn-through-king role order.
        black: [Handle<Image>; 6],
    }

    /// One reusable piece sprite attached to a source square.
    #[derive(Component)]
    struct PieceSprite {
        /// Chess square represented by this slot.
        square: Square,
    }

    /// Source two-Hz playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run interactive playback or finite frame capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = ChessTrainingEnv::dqn_config(ChessTrainingEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(
            checkpoint,
            OBSERVATION_SIZE,
            ACTION_COUNT,
            &config.hidden_sizes,
        )?;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualGame {
                policy,
                game: ChessGame::default(),
                rng: SplitMix64(12_345),
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.5, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets/chess".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Chess".into(),
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
            app.add_systems(Update, (advance_watch, sync_pieces).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    /// Spawn the board image and one reusable sprite per square.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("Chess Camera")));
        commands.spawn((
            Sprite {
                image: assets.load("img/chessboard.png"),
                custom_size: Some(Vec2::splat(VIEWPORT)),
                ..default()
            },
            Transform::default(),
        ));
        let images = PieceImages {
            white: [
                assets.load("img/pawn_white.png"),
                assets.load("img/knight_white.png"),
                assets.load("img/bishop_white.png"),
                assets.load("img/rook_white.png"),
                assets.load("img/queen_white.png"),
                assets.load("img/king_white.png"),
            ],
            black: [
                assets.load("img/pawn_black.png"),
                assets.load("img/knight_black.png"),
                assets.load("img/bishop_black.png"),
                assets.load("img/rook_black.png"),
                assets.load("img/queen_black.png"),
                assets.load("img/king_black.png"),
            ],
        };
        for square in Square::ALL {
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                PieceSprite { square },
            ));
        }
        commands.insert_resource(images);
    }

    /// Advance one selected AEC agent at the source renderer rate.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualGame>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Apply one random white move, one learned black move, or a reset.
    fn advance_visual(visual: &mut VisualGame) {
        if visual.game.outcome.is_some() {
            visual.game.reset();
            return;
        }
        if visual.game.selected_player() == Player::Black {
            let observation = ChessTrainingEnv {
                game: visual.game.clone(),
                rng: visual.rng,
                control: super::ControlMode::PolicyAsBlackVsRandomWhite,
            }
            .observation();
            let Ok(action) = ChessTrainingEnv::deployment_action(&visual.policy, &observation)
            else {
                return;
            };
            if visual.game.play(Player::Black, action).is_err() {
                visual.game.reset();
            }
        } else {
            let legal = visual.game.legal_action_moves();
            let Some((action, _chess_move)) = legal.get(visual.rng.index(legal.len())) else {
                return;
            };
            if visual.game.play(Player::White, *action).is_err() {
                visual.game.reset();
            }
        }
    }

    /// Synchronize every fixed square slot after a visible move.
    fn sync_pieces(
        visual: Res<'_, VisualGame>,
        images: Res<'_, PieceImages>,
        mut pieces: Query<'_, '_, (&PieceSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
    ) {
        if visual.is_changed() {
            apply_piece_sprites(&visual, &images, &mut pieces);
        }
    }

    /// Apply the source image, size, and top-left board coordinates.
    fn apply_piece_sprites(
        visual: &VisualGame,
        images: &PieceImages,
        pieces: &mut Query<'_, '_, (&PieceSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
    ) {
        for (slot, mut sprite, mut transform, mut visibility) in pieces.iter_mut() {
            let Some(piece) = visual.game.position.board().piece_at(slot.square) else {
                *visibility = Visibility::Hidden;
                continue;
            };
            let role_index = match piece.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };
            let image = if piece.color == ChessColor::White {
                images.white.get(role_index)
            } else {
                images.black.get(role_index)
            };
            let Some(image) = image else {
                *visibility = Visibility::Hidden;
                continue;
            };
            sprite.image = image.clone();
            sprite.custom_size = Some(Vec2::splat(CELL_SIZE));
            let column = slot.square.file().to_u32() as f32;
            let row_from_top = 7.0 - slot.square.rank().to_u32() as f32;
            transform.translation = Vec3::new(
                column.mul_add(CELL_SIZE, CELL_SIZE / 2.0) - VIEWPORT / 2.0,
                VIEWPORT / 2.0 - row_from_top * CELL_SIZE - CELL_SIZE / 2.0,
                1.0,
            );
            *visibility = Visibility::Visible;
        }
    }

    /// Capture five seconds at 20 FPS while moves advance at two FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualGame>,
        images: Res<'_, PieceImages>,
        mut pieces: Query<'_, '_, (&PieceSprite, &mut Sprite, &mut Transform, &mut Visibility)>,
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
        apply_piece_sprites(&visual, &images, &mut pieces);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Render and encode the source-sized five-second chess GIF.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-chess-v6",
            VIEWPORT_PIXELS,
            VIEWPORT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::{
        ChessAction, ChessGame, ChessTrainingEnv, DiscreteDqnExample, Env, KnownOutcome, Player,
        ACTION_COUNT, OBSERVATION_SIZE,
    };
    use shakmaty::Color;

    #[test]
    fn initial_position_has_twenty_exact_legal_actions() {
        let game = ChessGame::default();
        let white = game.observe(Player::White);
        let black = game.observe(Player::Black);

        assert_eq!(white.planes.len(), OBSERVATION_SIZE);
        assert_eq!(white.action_mask.len(), ACTION_COUNT);
        assert_eq!(white.action_mask.iter().filter(|legal| **legal).count(), 20);
        assert!(black.action_mask.iter().all(|legal| !legal));
        assert!(white
            .planes
            .chunks_exact(111)
            .all(|cell| cell[7..].iter().all(|value| *value == 0)));
    }

    #[test]
    fn alpha_zero_mapping_matches_source_for_e2e4_and_black_mirror() -> Result<(), Box<dyn Error>> {
        let mut game = ChessGame::default();
        let white_action = game.action_for_uci("e2e4")?;
        assert_eq!(white_action, ChessAction::from_index(2_421));

        game.play(Player::White, white_action)?;
        let black_action = game.action_for_uci("e7e5")?;
        assert_eq!(black_action, ChessAction::from_index(2_421));
        Ok(())
    }

    #[test]
    fn en_passant_plane_moves_the_vulnerable_pawn_to_its_back_rank() -> Result<(), Box<dyn Error>> {
        let mut game = ChessGame::default();
        game.play_uci("e2e4")?;

        let observation = game.observe(Player::White);
        let e1_history_pawn = observation.planes[(7 * 8 + 4) * 111 + 7];
        let e4_history_pawn = observation.planes[(4 * 8 + 4) * 111 + 7];

        assert_eq!(e1_history_pawn, 1);
        assert_eq!(e4_history_pawn, 0);
        Ok(())
    }

    #[test]
    fn black_observation_flips_rows_and_swaps_history_piece_planes() -> Result<(), Box<dyn Error>> {
        let mut game = ChessGame::default();
        game.play_uci("e2e4")?;

        let observation = game.observe(Player::Black);

        assert_eq!(observation.planes[(6 * 8) * 111 + 7], 1);
        assert_eq!(observation.planes[(1 * 8) * 111 + 13], 1);
        Ok(())
    }

    #[test]
    fn wrong_player_and_illegal_actions_are_typed_errors() {
        let mut game = ChessGame::default();

        assert!(game
            .play(Player::Black, ChessAction::from_index(0))
            .is_err());
        assert!(game
            .play(Player::White, ChessAction::from_index(0))
            .is_err());
    }

    #[test]
    fn scholars_mate_is_a_white_checkmate() -> Result<(), Box<dyn Error>> {
        const WHITE_LINE: [&str; 4] = ["e2e4", "f1c4", "d1h5", "h5f7"];
        const BLACK_LINE: [&str; 3] = ["e7e5", "b8c6", "g8f6"];
        let mut game = ChessGame::default();
        for stage in 0..WHITE_LINE.len() {
            game.play_uci(WHITE_LINE[stage])?;
            if let Some(reply) = BLACK_LINE.get(stage) {
                game.play_uci(reply)?;
            }
        }

        assert_eq!(
            game.outcome,
            Some(KnownOutcome::Decisive {
                winner: Color::White,
            })
        );
        Ok(())
    }

    #[test]
    fn training_preserves_aec_turns_and_evaluation_randomizes_white() -> Result<(), Box<dyn Error>>
    {
        let mut first = ChessTrainingEnv::default();
        let mut second = ChessTrainingEnv::default();
        let first_reset = first.reset(Some(42));
        let second_reset = second.reset(Some(42));
        assert_eq!(first_reset.observation, second_reset.observation);
        assert_eq!(
            first_reset.observation.len(),
            OBSERVATION_SIZE + ACTION_COUNT
        );
        assert_eq!(
            first_reset.observation[OBSERVATION_SIZE..]
                .iter()
                .filter(|legal| **legal > 0.5)
                .count(),
            20
        );

        let action = first.game.action_for_uci("e2e4")?;
        let first_step = first.step(action);
        let second_step = second.step(action);

        assert_eq!(first_step.observation, second_step.observation);
        assert_eq!(first.game.positions, second.game.positions);
        assert_eq!(first.game.selected_player(), Player::Black);

        let mut evaluation = ChessTrainingEnv::default();
        evaluation.prepare_evaluation();
        evaluation.reset(Some(42));
        assert_eq!(evaluation.game.positions.len(), 1);
        let random_action = evaluation.random_action().expect("white has legal moves");
        let reset = evaluation.step(random_action);
        assert_eq!(evaluation.game.positions.len(), 2);
        assert_eq!(evaluation.game.selected_player(), Player::Black);
        assert_eq!(reset.observation, evaluation.observation());
        Ok(())
    }
}
