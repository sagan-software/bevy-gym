//!
//! # Hanabi
//!
//! ```{figure} classic_hanabi.gif
//! :width: 140px
//! :name: hanabi
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation             | `make("aec", "classic/hanabi-v5")`         |
//! |----------------------|--------------------------------------------|
//! | Actions              | Discrete                                   |
//! | Parallel API         | Yes                                        |
//! | Manual Control       | No                                         |
//! | Agents               | `agents= ['player_0', 'player_1']`         |
//! | Agents               | 2                                          |
//! | Action Shape         | Discrete(20)                               |
//! | Action Values        | Discrete(20)                               |
//! | Observation Shape    | (658,)                                     |
//! | Observation Values   | [0,1]                                      |
//!
//!
//! Hanabi is a 2-5 player cooperative game where players work together to form fireworks of different colors. A firework is a set of cards of the same color, ordered from 1 to 5. Cards in the game have both a color and number; each player can view the cards another player holds, but not their own.
//! Players cannot directly communicate with each other, but must instead remove an info token from play in order to give information. Players can tell other players which of the cards in their hand is a specific color, or a specific number. There are initially 8 info tokens, but players can discard
//! cards in their hand to return an info token into play. Players can also play a card from their hand: the card must either begin a new firework or be appended in order to an existing firework. However, 2 fireworks cannot have the same color, and a single firework cannot repeat numbers. If the
//! played card does not satisfy these conditions, a life token is placed. The game ends when either 3 life tokens have been placed, all 5 fireworks have been completed, or all cards have been drawn from the deck. Points are awarded based on the largest card value in each created firework.
//!
//! ### Environment arguments
//!
//! Hanabi takes in a number of arguments defining the size and complexity of the game. Default is a full 2 player hanabi game.
//!
//! ```python
//! from pettingzoo import make
//!
//! make("aec", "classic/hanabi-v5", colors=5, ranks=5, players=2, hand_size=5,
//! max_information_tokens=8, max_life_tokens=3, observation_type="minimal")
//! ```
//!
//! `colors`: Number of colors the cards can take (affects size of deck)
//!
//! `ranks`: Number of ranks the cards can take (affects size of deck)
//!
//! `hand_size`: Size of player's hands. Standard game is (4 if players >= 4 else 5)
//!
//! `max_information_tokens`: Maximum number of information tokens (more tokens makes the game easier by allowing more information to be revealed)
//!
//! `max_life_tokens`: Maximum number of life tokens (more tokens makes the game easier by allowing more information to be revealed)
//!
//! `observation_type`:
//!     "minimal": Minimal observation (what a human sees).
//!     "card_knowledge": includes per-card knowledge of past hints, as well as simple inferred knowledge of the form
//!         "this card is not red, because it was not revealed as red in a past".
//!     "seer" shows all cards, including the player's own cards, regardless of what hints have been given.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//! The main observation space of an agent is a 658 sized vector (for the default 2-player full game) representing the life and info tokens left, the currently constructed fireworks, the hands of all other agents, the current deck size and the discarded cards. Since v5 the observation is produced by
//! OpenSpiel's canonical Hanabi encoder (via Shimmy), so its layout — and the index ranges in the table below — depend on the chosen configuration (`colors`, `ranks`, `players`, `hand_size`, `max_information_tokens`, `max_life_tokens`). The table below is for the default configuration
//! (`colors=5, ranks=5, players=2, hand_size=5, max_information_tokens=8, max_life_tokens=3`).
//!
//! Each card is encoded with a `colors*ranks` bit one-hot vector (25 bits by default), where the index of the set bit is `color*ranks + rank`. The remaining deck size is represented with a unary encoding whose length is `max_deck_size - players*hand_size` (40 bits by default, since the
//! `players*hand_size` cards dealt at the start can never be in the deck — note this is *not* the full deck size of 50). The state of each colored firework is represented with a one-hot encoding of the highest rank played (all-zero if no card of that color has been played yet). The information
//! tokens remaining and the life tokens remaining are each represented with a unary encoding. The discard pile is represented with a thermometer encoding of the ranks of each discarded card, that is the least significant bit being set to 1 indicates the lowest rank card of that color has been
//! discarded.
//!
//! As players reveal info about their cards, the information revealed per card is also observed (the "card knowledge" section). Each card uses `colors*ranks + colors + ranks` bits (35 by default). The first `colors*ranks` bits represent whether or not that specific card could still be a specific
//! color/rank given the hints received. The next `colors` bits store whether the color of that card was explicitly revealed, so if the card was revealed to be red, then these bits would be 10000. Finally the last `ranks` bits are the revealed rank of the card. So if the card was revealed to be of
//! rank 1, then these bits would be 10000. These bits are tracked and observed for every card in every player's hand (this player first, then the others).
//!
//! |  Index  | Description                                                    |  Values  |
//! |:-------:|----------------------------------------------------------------|:--------:|
//! |   0-124 | Other player's hand: 5 cards × 25-bit one-hot                  |  [0, 1]  |
//! | 125-126 | Per-player flag set when that player's hand is missing a card  |  [0, 1]  |
//! | 127-166 | Unary Encoding of Remaining Deck Size (max_deck − players×hand)|  [0, 1]  |
//! | 167-171 | Vector of Red Firework                                         |  [0, 1]  |
//! | 172-176 | Vector of Yellow Firework                                      |  [0, 1]  |
//! | 177-181 | Vector of Green Firework                                       |  [0, 1]  |
//! | 182-186 | Vector of White Firework                                       |  [0, 1]  |
//! | 187-191 | Vector of Blue Firework                                        |  [0, 1]  |
//! | 192-199 | Unary Encoding of Remaining Info Tokens                        |  [0, 1]  |
//! | 200-202 | Unary Encoding of Remaining Life Tokens                        |  [0, 1]  |
//! | 203-252 | Thermometer Encoding of Discard Pile                           |  [0, 1]  |
//! | 253-254 | Last action: Acting Player (relative offset, one-hot)          |  [0, 1]  |
//! | 255-258 | Last action: Move Type (play / discard / reveal color / rank)  |  [0, 1]  |
//! | 259-260 | Last action: Target Player of a reveal (relative offset)       |  [0, 1]  |
//! | 261-265 | Last action: Color Revealed                                    |  [0, 1]  |
//! | 266-270 | Last action: Rank Revealed                                     |  [0, 1]  |
//! | 271-275 | Last action: Which Cards in the Hand were Revealed             |  [0, 1]  |
//! | 276-280 | Last action: Position of the Card that was played or discarded |  [0, 1]  |
//! | 281-305 | Last action: Vector Representing the Card that was played/discarded | [0, 1] |
//! | 306-306 | Last action: Whether the played card was added to a firework   |  [0, 1]  |
//! | 307-307 | Last action: Whether an info token was added                   |  [0, 1]  |
//! | 308-342 | Card Knowledge of This Player's 0th Card                       |  [0, 1]  |
//! | 343-377 | Card Knowledge of This Player's 1st Card                       |  [0, 1]  |
//! | 378-412 | Card Knowledge of This Player's 2nd Card                       |  [0, 1]  |
//! | 413-447 | Card Knowledge of This Player's 3rd Card                       |  [0, 1]  |
//! | 448-482 | Card Knowledge of This Player's 4th Card                       |  [0, 1]  |
//! | 483-517 | Card Knowledge of Other Player's 0th Card                      |  [0, 1]  |
//! | 518-552 | Card Knowledge of Other Player's 1st Card                      |  [0, 1]  |
//! | 553-587 | Card Knowledge of Other Player's 2nd Card                      |  [0, 1]  |
//! | 588-622 | Card Knowledge of Other Player's 3rd Card                      |  [0, 1]  |
//! | 623-657 | Card Knowledge of Other Player's 4th Card                      |  [0, 1]  |
//!
//!
//! #### Legal Actions Mask
//!
//! The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
//! whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
//!
//! ### Action Space
//!
//! The action space is a scalar value, which ranges from 0 to the max number of actions. The values represent all possible actions a player can make, legal or not. Each possible move in the environment is mapped to a UUID, which ranges from 0 to the max number of moves. By default the max number of
//! moves is 20. The first range of actions are to discard a card in the agent's hand. If there are k cards in the player's hand, then the first k action values are to discard one of those cards. The next k actions would be to play one of the cards in the player's hand. Finally, the remaining actions
//! are to reveal a color or rank in another players hand. The first set of reveal actions would be revealing all colors or values of cards for the next player in order, and this repeats for all the other players in the environment.
//!
//! | Action ID | Action                                                      |
//! |:---------:|-------------------------------------------------------------|
//! |     0     | Discard Card at position 0                                  |
//! |     1     | Discard Card at position 1                                  |
//! |     2     | Discard Card at position 2                                  |
//! |     3     | Discard Card at position 3                                  |
//! |     4     | Discard Card at position 4                                  |
//! |     5     | Play Card at position 0                                     |
//! |     6     | Play Card at position 1                                     |
//! |     7     | Play Card at position 2                                     |
//! |     8     | Play Card at position 3                                     |
//! |     9     | Play Card at position 4                                     |
//! |    10     | Reveal Red Cards for Player 1                               |
//! |    11     | Reveal Yellow Cards for Player 1                            |
//! |    12     | Reveal Green Cards for Player 1                             |
//! |    13     | Reveal White Cards for Player 1                             |
//! |    14     | Reveal Blue Cards for Player 1                              |
//! |    15     | Reveal Rank 1 Cards for Player 1                            |
//! |    16     | Reveal Rank 2 Cards for Player 1                            |
//! |    17     | Reveal Rank 3 Cards for Player 1                            |
//! |    18     | Reveal Rank 4 Cards for Player 1                            |
//! |    19     | Reveal Rank 5 Cards for Player 1                            |
//!
//! ### Rewards
//!
//! The reward of each step is calculated as the change in game score from the last step. The game score is calculated as the sum of values in each constructed firework. If the game is lost, the score is set to zero, so the final reward will be the negation of all reward received so far.
//!
//! For example, if fireworks were created as follows:
//!
//! Blue 1, Blue 2, Red 1, Green 1, Green 2, Green 3
//!
//! At the end of the game, the total score would be 2 + 1 + 3 = 6
//!
//! If an illegal action is taken, the game terminates and the one player that took the illegal action loses. Like an ordinary loss, their final reward will be the negation of all reward received so far. The reward of the other players will not be affected by the illegal action.
//!
//!
//! ### Rendering
//!
//! Hanabi supports `human` and `rgb_array` render modes. In both modes the board
//! is drawn with pygame: the fireworks piles, every player's hand (the player
//! whose turn it is is highlighted), the remaining info and life tokens, the deck
//! size, and the discard pile.
//!
//! ### Version History
//!
//! * v5: Switched environment to depend on OpenSpiel (using Shimmy) for future compatibility, and replaced the `ansi` text rendering with pygame (`human`/`rgb_array`) rendering (1.23.0)
//! * v4: Fixed bug in arbitrary calls to observe() (1.8.0)
//! * v3: Legal action mask in observation replaced illegal move list in infos (1.5.0)
//! * v2: Fixed default parameters (1.4.2)
//! * v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
//! * v0: Initial versions release (1.0.0)
//!

#![expect(
    clippy::doc_markdown,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::path::Path;

use bevy_gym::training::{
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction, DqnConfig,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// Full-game colors in OpenSpiel and renderer order.
const COLOR_COUNT: usize = 5;
/// Full-game ranks.
const RANK_COUNT: usize = 5;
/// Default two-player hand size.
const HAND_SIZE: usize = 5;
/// Discard, play, color-reveal, and rank-reveal actions.
const ACTION_COUNT: usize = 20;
/// OpenSpiel's canonical default observation length.
const OBSERVATION_SIZE: usize = 658;
/// Initial information-token supply.
const MAX_INFORMATION_TOKENS: u8 = 8;
/// Initial life-token supply.
const MAX_LIFE_TOKENS: u8 = 3;

/// One color and rank from the standard 50-card deck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Card {
    /// Red, yellow, green, white, or blue.
    color: u8,
    /// One through five, stored zero-based.
    rank: u8,
}

impl Card {
    /// Construct one checked default-profile card.
    const fn new(color: u8, rank: u8) -> Self {
        assert!(color < COLOR_COUNT as u8 && rank < RANK_COUNT as u8);
        Self { color, rank }
    }

    /// Return the canonical 25-bit card identity index.
    const fn identity(self) -> usize {
        self.color as usize * RANK_COUNT + self.rank as usize
    }
}

/// Knowledge retained for one hidden card.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CardKnowledge {
    /// Card identities still compatible with all hints.
    possible: [bool; COLOR_COUNT * RANK_COUNT],
    /// Explicitly revealed color.
    color: Option<u8>,
    /// Explicitly revealed rank.
    rank: Option<u8>,
}

impl Default for CardKnowledge {
    fn default() -> Self {
        Self {
            possible: [true; COLOR_COUNT * RANK_COUNT],
            color: None,
            rank: None,
        }
    }
}

impl CardKnowledge {
    /// Apply a positive or negative color hint to this position.
    fn reveal_color(&mut self, color: u8, matches: bool) {
        for (identity, possible) in self.possible.iter_mut().enumerate() {
            let candidate_color = identity / RANK_COUNT;
            if (candidate_color == color as usize) != matches {
                *possible = false;
            }
        }
        if matches {
            self.color = Some(color);
        }
    }

    /// Apply a positive or negative rank hint to this position.
    fn reveal_rank(&mut self, rank: u8, matches: bool) {
        for (identity, possible) in self.possible.iter_mut().enumerate() {
            let candidate_rank = identity % RANK_COUNT;
            if (candidate_rank == rank as usize) != matches {
                *possible = false;
            }
        }
        if matches {
            self.rank = Some(rank);
        }
    }
}

/// Default OpenSpiel Hanabi action mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HanabiAction {
    /// Discard one hand position.
    Discard(usize),
    /// Play one hand position.
    Play(usize),
    /// Reveal one color in the other hand.
    RevealColor(u8),
    /// Reveal one rank in the other hand.
    RevealRank(u8),
}

impl HanabiAction {
    /// Convert one action-space index.
    const fn from_index(index: usize) -> Self {
        match index % ACTION_COUNT {
            value @ 0..=4 => Self::Discard(value),
            value @ 5..=9 => Self::Play(value - 5),
            value @ 10..=14 => Self::RevealColor((value - 10) as u8),
            value => Self::RevealRank((value - 15) as u8),
        }
    }

    /// Return the source action-space index.
    const fn index(self) -> usize {
        match self {
            Self::Discard(position) => position,
            Self::Play(position) => HAND_SIZE + position,
            Self::RevealColor(color) => 10 + color as usize,
            Self::RevealRank(rank) => 15 + rank as usize,
        }
    }
}

impl DqnAction for HanabiAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.index()
    }
}

/// Information encoded in the 55-bit last-action section.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LastAction {
    /// Acting player.
    player: usize,
    /// Applied action.
    action: HanabiAction,
    /// Target player for a reveal.
    target: Option<usize>,
    /// Hand positions matched by a reveal.
    revealed: [bool; HAND_SIZE],
    /// Removed card for play or discard.
    card: Option<Card>,
    /// Whether a play advanced a firework.
    successful_play: bool,
    /// Whether the action restored an information token.
    information_added: bool,
}

/// Exact source observation dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HanabiObservation {
    /// Canonical OpenSpiel card-knowledge observation.
    values: [u8; OBSERVATION_SIZE],
    /// Legal actions for the selected player.
    action_mask: [bool; ACTION_COUNT],
}

/// Typed raw-environment rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HanabiError {
    /// The caller supplied the unselected player.
    WrongPlayer {
        /// Player selected by the cooperative turn sequence.
        expected: u8,
        /// Player supplied by the caller.
        actual: u8,
    },
    /// Action absent from the source legal mask.
    IllegalAction {
        /// Source action index rejected by the rules.
        action: u8,
    },
    /// The cooperative game has ended.
    GameFinished,
}

impl fmt::Display for HanabiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPlayer { expected, actual } => {
                write!(
                    formatter,
                    "expected player {expected}, received player {actual}"
                )
            }
            Self::IllegalAction { action } => {
                write!(formatter, "Hanabi action {action} is illegal")
            }
            Self::GameFinished => formatter.write_str("the Hanabi game has finished"),
        }
    }
}

impl Error for HanabiError {}

/// Set one observation or action-mask slot when the index is in bounds.
fn set_slot<T: Copy>(values: &mut [T], index: usize, value: T) {
    if let Some(slot) = values.get_mut(index) {
        *slot = value;
    }
}

/// Complete default two-player Hanabi state.
#[derive(Debug, Clone)]
struct HanabiGame {
    /// Remaining deck in draw order.
    deck: VecDeque<Card>,
    /// Player hands.
    hands: [Vec<Card>; 2],
    /// Knowledge indexed by card owner, then hand position.
    knowledge: [Vec<CardKnowledge>; 2],
    /// Highest successfully played rank count per color.
    fireworks: [u8; COLOR_COUNT],
    /// Cards removed without advancing a firework.
    discards: Vec<Card>,
    /// Available information tokens.
    information_tokens: u8,
    /// Remaining life tokens.
    life_tokens: u8,
    /// Selected player.
    current_player: usize,
    /// Final actions remaining after the last draw.
    final_turns: Option<u8>,
    /// Most recent source action.
    last_action: Option<LastAction>,
    /// Terminal marker.
    terminated: bool,
}

impl Default for HanabiGame {
    fn default() -> Self {
        let mut game = Self::empty();
        game.reset(0);
        game
    }
}

impl HanabiGame {
    /// Allocate an empty game before dealing.
    fn empty() -> Self {
        Self {
            deck: VecDeque::new(),
            hands: std::array::from_fn(|_| Vec::with_capacity(HAND_SIZE)),
            knowledge: std::array::from_fn(|_| Vec::with_capacity(HAND_SIZE)),
            fireworks: [0; COLOR_COUNT],
            discards: Vec::new(),
            information_tokens: MAX_INFORMATION_TOKENS,
            life_tokens: MAX_LIFE_TOKENS,
            current_player: 0,
            final_turns: None,
            last_action: None,
            terminated: false,
        }
    }

    /// Borrow one player's hand without unchecked indexing.
    const fn hand(&self, player: usize) -> &Vec<Card> {
        let [first, second] = &self.hands;
        if player == 0 {
            first
        } else {
            second
        }
    }

    /// Mutably borrow one player's hand without unchecked indexing.
    const fn hand_mut(&mut self, player: usize) -> &mut Vec<Card> {
        let [first, second] = &mut self.hands;
        if player == 0 {
            first
        } else {
            second
        }
    }

    /// Borrow one player's knowledge without unchecked indexing.
    const fn knowledge(&self, player: usize) -> &Vec<CardKnowledge> {
        let [first, second] = &self.knowledge;
        if player == 0 {
            first
        } else {
            second
        }
    }

    /// Mutably borrow one player's knowledge without unchecked indexing.
    const fn knowledge_mut(&mut self, player: usize) -> &mut Vec<CardKnowledge> {
        let [first, second] = &mut self.knowledge;
        if player == 0 {
            first
        } else {
            second
        }
    }

    /// Shuffle and deal one deterministic full game.
    fn reset(&mut self, seed: u64) {
        let mut deck = standard_deck();
        let mut rng = SplitMix64::new(seed);
        for upper in (1..deck.len()).rev() {
            let selected = rng.index(upper + 1);
            deck.swap(upper, selected);
        }
        self.reset_with_deck(deck);
    }

    /// Deal one caller-arranged deck in round-robin order.
    fn reset_with_deck(&mut self, deck: Vec<Card>) {
        *self = Self::empty();
        self.deck = deck.into();
        for deal in 0..HAND_SIZE * 2 {
            let player = deal % 2;
            self.draw_to(player);
        }
    }

    /// Draw one card and attach fully unknown knowledge.
    fn draw_to(&mut self, player: usize) {
        if let Some(card) = self.deck.pop_front() {
            self.hand_mut(player).push(card);
            self.knowledge_mut(player).push(CardKnowledge::default());
            if self.deck.is_empty() && self.final_turns.is_none() {
                self.final_turns = Some(2);
            }
        }
    }

    /// Return the source legal-action mask.
    fn legal_mask(&self) -> [bool; ACTION_COUNT] {
        let mut mask = [false; ACTION_COUNT];
        let hand_len = self.hand(self.current_player).len();
        for position in 0..hand_len {
            set_slot(
                &mut mask,
                position,
                self.information_tokens < MAX_INFORMATION_TOKENS,
            );
            set_slot(&mut mask, HAND_SIZE + position, true);
        }
        if self.information_tokens > 0 {
            let target = (self.current_player + 1) % 2;
            for card in self.hand(target) {
                set_slot(&mut mask, 10 + usize::from(card.color), true);
                set_slot(&mut mask, 15 + usize::from(card.rank), true);
            }
        }
        mask
    }

    /// Build one player's exact 658-bit canonical observation.
    fn observe(&self, player: usize) -> HanabiObservation {
        let mut values = [0_u8; OBSERVATION_SIZE];
        let other = (player + 1) % 2;
        for (position, card) in self.hand(other).iter().enumerate() {
            set_slot(&mut values, position * 25 + card.identity(), 1);
        }
        set_slot(
            &mut values,
            125,
            u8::from(self.hand(player).len() < HAND_SIZE),
        );
        set_slot(
            &mut values,
            126,
            u8::from(self.hand(other).len() < HAND_SIZE),
        );
        for index in 0..self.deck.len().min(40) {
            set_slot(&mut values, 127 + index, 1);
        }
        for (color, height) in self.fireworks.into_iter().enumerate() {
            if height > 0 {
                set_slot(&mut values, 167 + color * 5 + usize::from(height - 1), 1);
            }
        }
        for index in 0..usize::from(self.information_tokens) {
            set_slot(&mut values, 192 + index, 1);
        }
        for index in 0..usize::from(self.life_tokens) {
            set_slot(&mut values, 200 + index, 1);
        }
        self.encode_discards(&mut values);
        self.encode_last_action(player, &mut values);
        self.encode_knowledge(player, &mut values);
        HanabiObservation {
            values,
            action_mask: if !self.terminated && player == self.current_player {
                self.legal_mask()
            } else {
                [false; ACTION_COUNT]
            },
        }
    }

    /// Encode per-color, per-rank discarded-copy thermometers.
    fn encode_discards(&self, values: &mut [u8; OBSERVATION_SIZE]) {
        let copies = [3_usize, 2, 2, 2, 1];
        for color in 0..COLOR_COUNT {
            let mut offset = 203 + color * 10;
            for (rank, copy_count) in copies.into_iter().enumerate() {
                let discarded = self
                    .discards
                    .iter()
                    .filter(|card| card.color as usize == color && card.rank as usize == rank)
                    .count();
                for copy in 0..discarded {
                    set_slot(values, offset + copy, 1);
                }
                offset += copy_count;
            }
        }
    }

    /// Encode the previous move relative to the observing player.
    fn encode_last_action(&self, player: usize, values: &mut [u8; OBSERVATION_SIZE]) {
        let Some(last) = &self.last_action else {
            return;
        };
        set_slot(values, 253 + (last.player + 2 - player) % 2, 1);
        let move_type = match last.action {
            HanabiAction::Play(_) => 0,
            HanabiAction::Discard(_) => 1,
            HanabiAction::RevealColor(_) => 2,
            HanabiAction::RevealRank(_) => 3,
        };
        set_slot(values, 255 + move_type, 1);
        if let Some(target) = last.target {
            set_slot(values, 259 + (target + 2 - player) % 2, 1);
        }
        match last.action {
            HanabiAction::RevealColor(color) => {
                set_slot(values, 261 + usize::from(color), 1);
            }
            HanabiAction::RevealRank(rank) => {
                set_slot(values, 266 + usize::from(rank), 1);
            }
            HanabiAction::Play(position) | HanabiAction::Discard(position) => {
                set_slot(values, 276 + position, 1);
            }
        }
        for (position, revealed) in last.revealed.into_iter().enumerate() {
            set_slot(values, 271 + position, u8::from(revealed));
        }
        if let Some(card) = last.card {
            set_slot(values, 281 + card.identity(), 1);
        }
        set_slot(values, 306, u8::from(last.successful_play));
        set_slot(values, 307, u8::from(last.information_added));
    }

    /// Encode own knowledge first, then the other player's knowledge.
    fn encode_knowledge(&self, player: usize, values: &mut [u8; OBSERVATION_SIZE]) {
        for relative in 0..2 {
            let owner = (player + relative) % 2;
            for position in 0..HAND_SIZE {
                let base = 308 + (relative * HAND_SIZE + position) * 35;
                let Some(knowledge) = self.knowledge(owner).get(position) else {
                    continue;
                };
                for (identity, possible) in knowledge.possible.into_iter().enumerate() {
                    set_slot(values, base + identity, u8::from(possible));
                }
                if let Some(color) = knowledge.color {
                    set_slot(values, base + 25 + usize::from(color), 1);
                }
                if let Some(rank) = knowledge.rank {
                    set_slot(values, base + 30 + usize::from(rank), 1);
                }
            }
        }
    }

    /// Apply one legal selected-player action and return its cooperative reward.
    fn play(&mut self, player: usize, action: HanabiAction) -> Result<f64, HanabiError> {
        self.validate_action(player, action)?;

        let score_before = self.score();
        let final_countdown_was_active = self.final_turns.is_some();
        let mut last = LastAction {
            player,
            action,
            target: None,
            revealed: [false; HAND_SIZE],
            card: None,
            successful_play: false,
            information_added: false,
        };
        match action {
            HanabiAction::Discard(position) => self.apply_discard(player, position, &mut last)?,
            HanabiAction::Play(position) => self.apply_play(player, position, &mut last)?,
            HanabiAction::RevealColor(color) => self.apply_color_hint(player, color, &mut last),
            HanabiAction::RevealRank(rank) => self.apply_rank_hint(player, rank, &mut last),
        }

        self.last_action = Some(last);
        self.finish_turn(final_countdown_was_active);
        let score_after = self.score();
        if self.life_tokens == 0 {
            Ok(-f64::from(score_before))
        } else {
            Ok(f64::from(score_after - score_before))
        }
    }

    /// Check the lifecycle, active player, and source action mask before mutation.
    fn validate_action(&self, player: usize, action: HanabiAction) -> Result<(), HanabiError> {
        if self.terminated {
            return Err(HanabiError::GameFinished);
        }
        if player != self.current_player {
            return Err(HanabiError::WrongPlayer {
                expected: self.current_player as u8,
                actual: player as u8,
            });
        }
        if !self
            .legal_mask()
            .get(action.index())
            .copied()
            .unwrap_or(false)
        {
            return Err(HanabiError::IllegalAction {
                action: action.index() as u8,
            });
        }
        Ok(())
    }

    /// Apply one discard and replenish its hand from the deck.
    fn apply_discard(
        &mut self,
        player: usize,
        position: usize,
        last: &mut LastAction,
    ) -> Result<(), HanabiError> {
        let Some(card) = self.remove_card(player, position) else {
            return Err(HanabiError::IllegalAction {
                action: HanabiAction::Discard(position).index() as u8,
            });
        };
        self.discards.push(card);
        last.card = Some(card);
        if self.information_tokens < MAX_INFORMATION_TOKENS {
            self.information_tokens += 1;
            last.information_added = true;
        }
        self.draw_to(player);
        Ok(())
    }

    /// Apply one attempted play and update fireworks or fuse tokens.
    fn apply_play(
        &mut self,
        player: usize,
        position: usize,
        last: &mut LastAction,
    ) -> Result<(), HanabiError> {
        let Some(card) = self.remove_card(player, position) else {
            return Err(HanabiError::IllegalAction {
                action: HanabiAction::Play(position).index() as u8,
            });
        };
        last.card = Some(card);
        let firework = self.fireworks.get_mut(usize::from(card.color));
        if firework.as_deref() == Some(&card.rank) {
            if let Some(height) = firework {
                *height += 1;
            }
            last.successful_play = true;
            if card.rank == 4 && self.information_tokens < MAX_INFORMATION_TOKENS {
                self.information_tokens += 1;
                last.information_added = true;
            }
        } else {
            self.life_tokens -= 1;
            self.discards.push(card);
        }
        self.draw_to(player);
        Ok(())
    }

    /// Reveal every matching color position in the partner's hand.
    fn apply_color_hint(&mut self, player: usize, color: u8, last: &mut LastAction) {
        let target = (player + 1) % 2;
        self.information_tokens -= 1;
        last.target = Some(target);
        let matches: [bool; HAND_SIZE] = std::array::from_fn(|position| {
            self.hand(target)
                .get(position)
                .is_some_and(|card| card.color == color)
        });
        for (position, matches) in matches.into_iter().enumerate() {
            set_slot(&mut last.revealed, position, matches);
            if let Some(knowledge) = self.knowledge_mut(target).get_mut(position) {
                knowledge.reveal_color(color, matches);
            }
        }
    }

    /// Reveal every matching rank position in the partner's hand.
    fn apply_rank_hint(&mut self, player: usize, rank: u8, last: &mut LastAction) {
        let target = (player + 1) % 2;
        self.information_tokens -= 1;
        last.target = Some(target);
        let matches: [bool; HAND_SIZE] = std::array::from_fn(|position| {
            self.hand(target)
                .get(position)
                .is_some_and(|card| card.rank == rank)
        });
        for (position, matches) in matches.into_iter().enumerate() {
            set_slot(&mut last.revealed, position, matches);
            if let Some(knowledge) = self.knowledge_mut(target).get_mut(position) {
                knowledge.reveal_rank(rank, matches);
            }
        }
    }

    /// Advance the selected player, final countdown, and terminal rules.
    fn finish_turn(&mut self, final_countdown_was_active: bool) {
        self.current_player = (self.current_player + 1) % 2;
        if final_countdown_was_active {
            if let Some(turns) = self.final_turns.as_mut() {
                *turns = turns.saturating_sub(1);
            }
        }
        self.terminated = self.life_tokens == 0
            || self.score() == (COLOR_COUNT * RANK_COUNT) as u8
            || self.final_turns == Some(0);
    }

    /// Remove a hand card and its aligned knowledge entry.
    fn remove_card(&mut self, player: usize, position: usize) -> Option<Card> {
        if position >= self.hand(player).len() || position >= self.knowledge(player).len() {
            return None;
        }
        self.knowledge_mut(player).remove(position);
        Some(self.hand_mut(player).remove(position))
    }

    /// Return the visible cooperative score.
    fn score(&self) -> u8 {
        self.fireworks.iter().sum()
    }
}

/// Small deterministic shuffle stream.
#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    /// Start one stream.
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Select a uniform index below `upper`.
    fn index(&mut self, upper: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        usize::try_from((value ^ (value >> 31)) % upper as u64).unwrap_or_default()
    }
}

/// Build the standard rank multiplicities: three ones, two twos through fours, one five.
fn standard_deck() -> Vec<Card> {
    let copies = [3_u8, 2, 2, 2, 1];
    let mut deck = Vec::with_capacity(50);
    for color in 0..COLOR_COUNT as u8 {
        for (rank, count) in copies.into_iter().enumerate() {
            for _ in 0..count {
                deck.push(Card::new(color, rank as u8));
            }
        }
    }
    deck
}

/// Policy-control mode for self-play training and random-first evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ControlMode {
    /// One shared policy supplies every AEC action during training.
    SharedPolicy,
    /// The policy controls player one against random player zero.
    PolicyAsPlayerOneVsRandomPlayerZero,
}

/// Source AEC game with shared-policy training and random-first evaluation.
#[derive(Debug, Clone)]
struct HanabiTrainingEnv {
    /// Complete raw default game.
    game: HanabiGame,
    /// Independent stream used only for random comparison actions.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for HanabiTrainingEnv {
    fn default() -> Self {
        let mut game = HanabiGame::empty();
        game.reset(0);
        Self {
            game,
            rng: SplitMix64::new(0),
            control: ControlMode::SharedPolicy,
        }
    }
}

/// Per-transition cooperative evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HanabiInfo {
    /// Exact observations in player order.
    observations: [HanabiObservation; 2],
    /// Current shared score.
    score: u8,
    /// Remaining information tokens.
    information_tokens: u8,
    /// Remaining life tokens.
    life_tokens: u8,
}

impl Default for HanabiInfo {
    fn default() -> Self {
        let observation = HanabiObservation {
            values: [0; OBSERVATION_SIZE],
            action_mask: [false; ACTION_COUNT],
        };
        Self {
            observations: [observation.clone(), observation],
            score: 0,
            information_tokens: MAX_INFORMATION_TOKENS,
            life_tokens: MAX_LIFE_TOKENS,
        }
    }
}

impl HanabiTrainingEnv {
    /// Build current source observations and public evidence.
    fn info(&self) -> HanabiInfo {
        HanabiInfo {
            observations: [self.game.observe(0), self.game.observe(1)],
            score: self.game.score(),
            information_tokens: self.game.information_tokens,
            life_tokens: self.game.life_tokens,
        }
    }

    /// Pack the canonical observation and its exact legal-action mask.
    fn observation(&self) -> Vec<f32> {
        let player = match self.control {
            ControlMode::SharedPolicy => self.game.current_player,
            ControlMode::PolicyAsPlayerOneVsRandomPlayerZero => 1,
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

    /// Sample one uniformly random action from the source legal-action mask.
    fn random_legal_action(&mut self) -> Option<HanabiAction> {
        let mask = self.game.legal_mask();
        let legal_count = mask.into_iter().filter(|legal| *legal).count();
        if legal_count == 0 {
            return None;
        }
        let selected = self.rng.index(legal_count);
        mask.into_iter()
            .enumerate()
            .filter_map(|(index, legal)| legal.then_some(index))
            .nth(selected)
            .map(HanabiAction::from_index)
    }
}

impl Env for HanabiTrainingEnv {
    type Observation = Vec<f32>;
    type Action = HanabiAction;
    type Info = HanabiInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        let seed = seed.unwrap_or(0);
        self.game.reset(seed);
        self.rng = SplitMix64::new(seed ^ 0xa076_1d64_78bd_642f);
        Reset {
            observation: self.observation(),
            info: self.info(),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let actor = self.game.current_player;
        let result = self.game.play(actor, action);
        let (reward, status) = match result {
            Err(_error) => (-1.0, EpisodeStatus::Terminated),
            Ok(reward) if self.game.terminated => (reward, EpisodeStatus::Terminated),
            Ok(reward) => (reward, EpisodeStatus::Continuing),
        };
        Step {
            observation: self.observation(),
            reward,
            status,
            info: self.info(),
        }
    }
}

impl DiscreteDqnExample for HanabiTrainingEnv {
    const ENV_NAME: &'static str = "hanabi-v5";
    const GYMNASIUM_ID: &'static str = "classic/hanabi-v5";
    const OBSERVATION_DIM: usize = OBSERVATION_SIZE;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const DEFAULT_TRAIN_STEPS: usize = 1_000_000;
    const DEFAULT_EVAL_INTERVAL: usize = 25_000;
    const DEFAULT_EVAL_EPISODES: usize = 500;
    const DEFAULT_NUM_ENVS: usize = 8;
    const MIN_TRAIN_STEPS: usize = 100_000;
    const SOLVED_MEAN_REWARD: f64 = f64::INFINITY;
    const GIF_PATH: &'static str = "docs/images/classic-hanabi-v5.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 1.0,
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
        self.control = ControlMode::PolicyAsPlayerOneVsRandomPlayerZero;
    }

    fn evaluation_action(
        &mut self,
        policy: &bevy_gym::training::DqnPolicy,
        observation: &[f32],
    ) -> Result<Self::Action, bevy_gym::training::DqnError> {
        if self.game.current_player == 0 {
            Ok(self
                .random_legal_action()
                .unwrap_or_else(|| HanabiAction::from_index(0)))
        } else {
            Self::deployment_action(policy, observation)
        }
    }

    fn selection_score(evaluation: &DiscreteEvaluation) -> f64 {
        evaluation.mean_reward
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
    run_discrete_workflow::<HanabiTrainingEnv>()
}

/// Bevy reconstruction of PettingZoo's pygame Hanabi board.
#[cfg(feature = "render")]
mod render {
    use super::{
        Card, DiscreteDqnExample, Error, HanabiAction, HanabiGame, HanabiTrainingEnv, Path,
        SplitMix64, ACTION_COUNT, COLOR_COUNT, OBSERVATION_SIZE,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;

    /// Source renderer width.
    const WIDTH: f32 = 980.0;
    /// Integer viewport width used by window and encoder APIs.
    const WIDTH_PIXELS: u32 = 980;
    /// Source renderer height for two players.
    const HEIGHT: f32 = 884.0;
    /// Integer viewport height used by window and encoder APIs.
    const HEIGHT_PIXELS: u32 = 884;
    /// Source hand-card width at scale 1.6.
    const CARD_WIDTH: f32 = 102.0;
    /// Source hand-card height at scale 1.6.
    const CARD_HEIGHT: f32 = 140.0;
    /// Horizontal hand and firework gap.
    const CARD_GAP: f32 = 14.0;
    /// Common outer padding.
    const PADDING: f32 = 22.0;

    /// Configure the fixed reference viewport.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb_u8(10, 12, 24)));
        }
    }

    /// Loaded policy and visible cooperative game.
    #[derive(Resource)]
    struct VisualGame {
        /// Greedy player-zero DQN policy.
        policy: DqnPolicy,
        /// Exact visible Hanabi game.
        game: HanabiGame,
        /// Seeded player-one action stream.
        rng: SplitMix64,
    }

    /// One reusable card sprite.
    #[derive(Component)]
    struct CardVisual(CardLocation);

    /// Card role and index.
    #[derive(Debug, Clone, Copy)]
    enum CardLocation {
        /// Highest card of one firework.
        Firework(usize),
        /// One player's hand position.
        Hand {
            /// Player who owns the hand.
            player: usize,
            /// Position within that player's hand.
            position: usize,
        },
        /// Face-down remaining deck.
        Deck,
        /// One sorted discard position.
        Discard(usize),
    }

    /// Text that changes with the game state.
    #[derive(Component)]
    struct DynamicText(TextRole);

    /// Dynamic label role.
    #[derive(Debug, Clone, Copy)]
    enum TextRole {
        /// Shared score.
        Score,
        /// Information-token row.
        Information,
        /// Life-token row.
        Life,
        /// Remaining deck count.
        Deck,
        /// Player label and current-turn marker.
        Player(usize),
    }

    /// One player-row panel.
    #[derive(Component)]
    struct PlayerPanel {
        /// Owning player row.
        player: usize,
        /// Inner fill rather than outer border.
        inner: bool,
    }

    /// One-Hz half-second playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run interactive playback or a finite source-sized capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = HanabiTrainingEnv::dqn_config(HanabiTrainingEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(
            checkpoint,
            OBSERVATION_SIZE,
            ACTION_COUNT,
            &config.hidden_sizes,
        )?;
        let mut game = HanabiGame::empty();
        game.reset(0);
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualGame {
                policy,
                game,
                rng: SplitMix64::new(12_345),
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.5, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Hanabi".into(),
                            resolution: WindowResolution::new(WIDTH_PIXELS, HEIGHT_PIXELS),
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
            app.add_systems(Update, (advance_watch, sync_scene).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    /// Spawn the gradient, panels, headings, and reusable card slots.
    #[expect(
        clippy::too_many_lines,
        reason = "the declarative scene layout mirrors one fixed reference frame"
    )]
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        assets: Res<'_, AssetServer>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<ColorMaterial>>,
    ) {
        commands.spawn((Camera2d, Name::new("Hanabi Camera")));
        // Retain every commissioned card handle so mid-capture swaps never
        // expose an asynchronous-load gap between source-rate actions.
        for color in 0..COLOR_COUNT as u8 {
            for rank in 0..5 {
                commands.spawn((
                    Sprite {
                        image: assets.load(card_asset(Card::new(color, rank))),
                        ..default()
                    },
                    Visibility::Hidden,
                ));
            }
        }
        for row in 0..HEIGHT_PIXELS {
            let color = Color::srgb_u8(
                lerp_channel(18, 10, row),
                lerp_channel(22, 12, row),
                lerp_channel(40, 24, row),
            );
            commands.spawn((
                Sprite::from_color(color, Vec2::new(WIDTH, 1.0)),
                Transform::from_xyz(0.0, HEIGHT / 2.0 - row as f32 - 0.5, -2.0),
            ));
        }
        let font = assets.load("rlcard_envs/font/Minecraft.ttf");
        spawn_text(
            &mut commands,
            "Hanabi",
            22.0,
            20.0,
            30.0,
            Color::srgb_u8(232, 236, 245),
            &font,
        );
        spawn_text(
            &mut commands,
            "Fireworks",
            22.0,
            96.0,
            20.0,
            Color::srgb_u8(150, 160, 185),
            &font,
        );
        spawn_text(
            &mut commands,
            "Discards",
            22.0,
            680.0,
            20.0,
            Color::srgb_u8(150, 160, 185),
            &font,
        );
        spawn_text(
            &mut commands,
            "none",
            22.0,
            716.0,
            15.0,
            Color::srgb_u8(150, 160, 185),
            &font,
        );
        spawn_text(
            &mut commands,
            "Info",
            660.0,
            20.0,
            15.0,
            Color::srgb_u8(150, 160, 185),
            &font,
        );
        spawn_text(
            &mut commands,
            "Life",
            660.0,
            54.0,
            15.0,
            Color::srgb_u8(150, 160, 185),
            &font,
        );
        for (row_y, count, color) in [
            (30.0, 8, Color::srgb_u8(90, 170, 255)),
            (64.0, 3, Color::srgb_u8(232, 72, 88)),
        ] {
            let material = materials.add(color);
            for index in 0..count {
                commands.spawn((
                    Mesh2d(meshes.add(Circle::new(9.0))),
                    MeshMaterial2d(material.clone()),
                    Transform::from_xyz(
                        (index as f32).mul_add(24.0, 704.0) - WIDTH / 2.0,
                        HEIGHT / 2.0 - row_y,
                        3.0,
                    ),
                ));
            }
        }
        spawn_dynamic_text(&mut commands, TextRole::Score, 22.0, 58.0, 20.0, &font);
        spawn_dynamic_text(
            &mut commands,
            TextRole::Information,
            900.0,
            20.0,
            16.0,
            &font,
        );
        spawn_dynamic_text(&mut commands, TextRole::Life, 752.0, 54.0, 16.0, &font);
        spawn_dynamic_text(&mut commands, TextRole::Deck, 936.0, 94.0, 15.0, &font);

        let total_width = (COLOR_COUNT as f32).mul_add(CARD_WIDTH, 4.0 * CARD_GAP);
        let firework_start = (WIDTH - total_width) / 2.0;
        let colors = [
            Color::srgb_u8(214, 48, 49),
            Color::srgb_u8(240, 196, 25),
            Color::srgb_u8(46, 204, 90),
            Color::srgb_u8(232, 232, 238),
            Color::srgb_u8(52, 120, 232),
        ];
        for (color, slot_color) in colors.into_iter().enumerate() {
            let x = (color as f32).mul_add(CARD_WIDTH + CARD_GAP, firework_start);
            commands.spawn((
                Sprite::from_color(
                    Color::srgba(0.0, 0.0, 0.0, 0.25),
                    Vec2::new(CARD_WIDTH, CARD_HEIGHT),
                ),
                Transform::from_translation(top_left(x, 124.0, CARD_WIDTH, CARD_HEIGHT, 0.0)),
            ));
            spawn_text(
                &mut commands,
                ["R", "Y", "G", "W", "B"].get(color).copied().unwrap_or("R"),
                x + CARD_WIDTH / 2.0 - 10.0,
                174.0,
                30.0,
                slot_color,
                &font,
            );
            commands.spawn((
                Sprite::default(),
                Transform::from_translation(top_left(x, 124.0, CARD_WIDTH, CARD_HEIGHT, 1.0)),
                Visibility::Hidden,
                CardVisual(CardLocation::Firework(color)),
            ));
        }
        for player in 0..2 {
            let y = (player as f32).mul_add(192.0, 296.0);
            commands.spawn((
                Sprite::from_color(Color::srgb_u8(30, 36, 60), Vec2::new(958.0, 180.0)),
                Transform::from_translation(top_left(11.0, y, 958.0, 180.0, -0.5)),
                PlayerPanel {
                    player,
                    inner: false,
                },
            ));
            commands.spawn((
                Sprite::from_color(Color::srgb_u8(30, 36, 60), Vec2::new(952.0, 174.0)),
                Transform::from_translation(top_left(14.0, y + 3.0, 952.0, 174.0, -0.4)),
                PlayerPanel {
                    player,
                    inner: true,
                },
            ));
            spawn_dynamic_text(
                &mut commands,
                TextRole::Player(player),
                22.0,
                y + 10.0,
                20.0,
                &font,
            );
            let hand_start = WIDTH - PADDING - total_width;
            for position in 0..5 {
                let x = (position as f32).mul_add(CARD_WIDTH + CARD_GAP, hand_start);
                commands.spawn((
                    Sprite::default(),
                    Transform::from_translation(top_left(x, y + 8.0, CARD_WIDTH, CARD_HEIGHT, 1.0)),
                    Visibility::Hidden,
                    CardVisual(CardLocation::Hand { player, position }),
                ));
            }
        }
        commands.spawn((
            Sprite::default(),
            Transform::from_translation(top_left(914.0, 20.0, 44.0, 61.0, 1.0)),
            Visibility::Visible,
            CardVisual(CardLocation::Deck),
        ));
        for position in 0..50 {
            let column = position % 18;
            let row = position / 18;
            commands.spawn((
                Sprite::default(),
                Transform::from_translation(top_left(
                    (column as f32).mul_add(50.0, 22.0),
                    (row as f32).mul_add(67.0, 708.0),
                    44.0,
                    61.0,
                    1.0,
                )),
                Visibility::Hidden,
                CardVisual(CardLocation::Discard(position)),
            ));
        }
    }

    /// Interpolate one gradient color channel.
    fn lerp_channel(start: u8, end: u8, row: u32) -> u8 {
        let final_row = HEIGHT_PIXELS - 1;
        let weighted = u32::from(start) * (final_row - row) + u32::from(end) * row;
        u8::try_from(weighted / final_row).unwrap_or(end)
    }

    /// Spawn one source-positioned static label.
    fn spawn_text(
        commands: &mut Commands<'_, '_>,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        color: Color,
        font: &Handle<Font>,
    ) {
        commands.spawn((
            Text2d::new(text),
            TextFont {
                font: font.clone(),
                font_size: size,
                ..default()
            },
            TextColor(color),
            Anchor::TOP_LEFT,
            Transform::from_xyz(x - WIDTH / 2.0, HEIGHT / 2.0 - y, 3.0),
        ));
    }

    /// Spawn one source-positioned changing label.
    fn spawn_dynamic_text(
        commands: &mut Commands<'_, '_>,
        role: TextRole,
        x: f32,
        y: f32,
        size: f32,
        font: &Handle<Font>,
    ) {
        commands.spawn((
            Text2d::new(""),
            TextFont {
                font: font.clone(),
                font_size: size,
                ..default()
            },
            TextColor(Color::srgb_u8(232, 236, 245)),
            Anchor::TOP_LEFT,
            Transform::from_xyz(x - WIDTH / 2.0, HEIGHT / 2.0 - y, 3.0),
            DynamicText(role),
        ));
    }

    /// Convert a source top-left rectangle into centered Bevy coordinates.
    fn top_left(x: f32, y: f32, width: f32, height: f32, z: f32) -> Vec3 {
        Vec3::new(
            x + width / 2.0 - WIDTH / 2.0,
            HEIGHT / 2.0 - y - height / 2.0,
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

    /// Apply one random player-zero action, one learned player-one action, or a reset.
    fn advance_visual(visual: &mut VisualGame) {
        if visual.game.terminated {
            visual.game.reset(visual.rng.0);
            return;
        }
        let player = visual.game.current_player;
        let action = if player == 1 {
            let observation = HanabiTrainingEnv {
                game: visual.game.clone(),
                rng: visual.rng,
                control: super::ControlMode::PolicyAsPlayerOneVsRandomPlayerZero,
            }
            .observation();
            match HanabiTrainingEnv::deployment_action(&visual.policy, &observation) {
                Ok(action) => action,
                Err(_error) => return,
            }
        } else {
            let mask = visual.game.legal_mask();
            let legal: Vec<_> = mask
                .into_iter()
                .enumerate()
                .filter_map(|(index, is_legal)| is_legal.then_some(index))
                .collect();
            match legal.get(visual.rng.index(legal.len())).copied() {
                Some(index) => HanabiAction::from_index(index),
                None => return,
            }
        };
        if visual.game.play(player, action).is_err() {
            visual.game.reset(visual.rng.0);
        }
    }

    /// Synchronize cards, labels, and current-player panels.
    fn sync_scene(
        visual: Res<'_, VisualGame>,
        assets: Res<'_, AssetServer>,
        mut cards: Query<'_, '_, (&CardVisual, &mut Sprite, &mut Visibility), Without<PlayerPanel>>,
        mut texts: Query<'_, '_, (&DynamicText, &mut Text2d, &mut TextColor), Without<PlayerPanel>>,
        mut panels: Query<
            '_,
            '_,
            (&PlayerPanel, &mut Sprite),
            (Without<CardVisual>, Without<DynamicText>),
        >,
    ) {
        if visual.is_changed() {
            apply_scene(&visual, &assets, &mut cards, &mut texts, &mut panels);
        }
    }

    /// Apply all dynamic renderer state.
    fn apply_scene(
        visual: &VisualGame,
        assets: &AssetServer,
        cards: &mut Query<
            '_,
            '_,
            (&CardVisual, &mut Sprite, &mut Visibility),
            Without<PlayerPanel>,
        >,
        texts: &mut Query<
            '_,
            '_,
            (&DynamicText, &mut Text2d, &mut TextColor),
            Without<PlayerPanel>,
        >,
        panels: &mut Query<
            '_,
            '_,
            (&PlayerPanel, &mut Sprite),
            (Without<CardVisual>, Without<DynamicText>),
        >,
    ) {
        let mut discards = visual.game.discards.clone();
        discards.sort_unstable();
        for (slot, mut sprite, mut visibility) in cards.iter_mut() {
            let (card, size) = match slot.0 {
                CardLocation::Firework(color) => (
                    visual
                        .game
                        .fireworks
                        .get(color)
                        .copied()
                        .unwrap_or_default()
                        .checked_sub(1)
                        .map(|rank| Card::new(color as u8, rank)),
                    Vec2::new(CARD_WIDTH, CARD_HEIGHT),
                ),
                CardLocation::Hand { player, position } => {
                    if player == visual.game.current_player
                        && position < visual.game.hand(player).len()
                    {
                        sprite.image = assets.load("hanabi/img/back.png");
                        sprite.custom_size = Some(Vec2::new(CARD_WIDTH, CARD_HEIGHT));
                        *visibility = Visibility::Visible;
                        continue;
                    }
                    (
                        visual.game.hand(player).get(position).copied(),
                        Vec2::new(CARD_WIDTH, CARD_HEIGHT),
                    )
                }
                CardLocation::Deck => {
                    sprite.image = assets.load("hanabi/img/back.png");
                    sprite.custom_size = Some(Vec2::new(44.0, 61.0));
                    *visibility = Visibility::Visible;
                    continue;
                }
                CardLocation::Discard(position) => {
                    (discards.get(position).copied(), Vec2::new(44.0, 61.0))
                }
            };
            let Some(card) = card else {
                *visibility = Visibility::Hidden;
                continue;
            };
            sprite.image = assets.load(card_asset(card));
            sprite.custom_size = Some(size);
            *visibility = Visibility::Visible;
        }
        for (role, mut text, mut color) in texts.iter_mut() {
            let (content, tint) = match role.0 {
                TextRole::Score => (
                    format!("Score {} / 25", visual.game.score()),
                    Color::srgb_u8(255, 214, 92),
                ),
                TextRole::Information => (
                    visual.game.information_tokens.to_string(),
                    Color::srgb_u8(90, 170, 255),
                ),
                TextRole::Life => (
                    visual.game.life_tokens.to_string(),
                    Color::srgb_u8(232, 72, 88),
                ),
                TextRole::Deck => (
                    format!("x{}", visual.game.deck.len()),
                    Color::srgb_u8(150, 160, 185),
                ),
                TextRole::Player(player) => (
                    if player == visual.game.current_player {
                        format!("Player {player}  (current turn)")
                    } else {
                        format!("Player {player}")
                    },
                    if player == visual.game.current_player {
                        Color::srgb_u8(255, 214, 92)
                    } else {
                        Color::srgb_u8(232, 236, 245)
                    },
                ),
            };
            **text = content;
            color.0 = tint;
        }
        for (panel, mut sprite) in panels.iter_mut() {
            sprite.color = if panel.inner {
                if panel.player == visual.game.current_player {
                    Color::srgb_u8(42, 50, 80)
                } else {
                    Color::srgb_u8(30, 36, 60)
                }
            } else if panel.player == visual.game.current_player {
                Color::srgb_u8(255, 214, 92)
            } else {
                Color::srgb_u8(30, 36, 60)
            };
        }
    }

    /// Resolve one card to the commissioned sprite path.
    fn card_asset(card: Card) -> String {
        let color = ["red", "yellow", "green", "white", "blue"]
            .get(usize::from(card.color))
            .copied()
            .unwrap_or("red");
        format!("hanabi/img/{color}_{}.png", card.rank + 1)
    }

    /// Capture five seconds at 20 FPS while advancing at two FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualGame>,
        assets: Res<'_, AssetServer>,
        mut cards: Query<'_, '_, (&CardVisual, &mut Sprite, &mut Visibility), Without<PlayerPanel>>,
        mut texts: Query<'_, '_, (&DynamicText, &mut Text2d, &mut TextColor), Without<PlayerPanel>>,
        mut panels: Query<
            '_,
            '_,
            (&PlayerPanel, &mut Sprite),
            (Without<CardVisual>, Without<DynamicText>),
        >,
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
        apply_scene(&visual, &assets, &mut cards, &mut texts, &mut panels);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Encode the exact reference-sized demonstration.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-hanabi-v5",
            WIDTH_PIXELS,
            HEIGHT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The standard deck uses three ones, two twos through fours, and one five.
    #[test]
    fn standard_deck_has_exact_rank_multiplicities() {
        let deck = standard_deck();

        assert_eq!(deck.len(), 50);
        for color in 0..COLOR_COUNT as u8 {
            let counts: Vec<usize> = (0..RANK_COUNT as u8)
                .map(|rank| {
                    deck.iter()
                        .filter(|card| card.color == color && card.rank == rank)
                        .count()
                })
                .collect();
            assert_eq!(counts, vec![3, 2, 2, 2, 1]);
        }
    }

    /// Full information forbids discards but permits every hand play and matching clue.
    #[test]
    fn initial_legal_mask_matches_token_and_hand_rules() {
        let game = HanabiGame::default();
        let mask = game.legal_mask();

        assert!(mask[..5].iter().all(|legal| !legal));
        assert!(mask[5..10].iter().all(|legal| *legal));
        for card in &game.hands[1] {
            assert!(mask[10 + card.color as usize]);
            assert!(mask[15 + card.rank as usize]);
        }
    }

    /// A clue spends one token and eliminates incompatible card identities.
    #[test]
    fn color_hint_updates_positive_and_negative_knowledge() {
        let mut game = HanabiGame::default();
        let color = game.hands[1][0].color;

        assert_eq!(game.play(0, HanabiAction::RevealColor(color)), Ok(0.0));

        assert_eq!(game.information_tokens, 7);
        for (position, card) in game.hands[1].iter().enumerate() {
            let knowledge = &game.knowledge[1][position];
            assert_eq!(knowledge.color, (card.color == color).then_some(color));
            for identity in 0..25 {
                assert_eq!(
                    knowledge.possible[identity],
                    (identity / RANK_COUNT == color as usize) == (card.color == color)
                );
            }
        }
    }

    /// Completing a firework scores one and restores a missing information token.
    #[test]
    fn rank_five_completion_restores_information() {
        let mut game = HanabiGame::default();
        game.fireworks[0] = 4;
        game.information_tokens = 7;
        game.hands[0][0] = Card::new(0, 4);

        assert_eq!(game.play(0, HanabiAction::Play(0)), Ok(1.0));
        assert_eq!(game.fireworks[0], 5);
        assert_eq!(game.information_tokens, 8);
    }

    /// Losing the last life zeroes the accumulated score through a negative delta.
    #[test]
    fn final_misplay_negates_existing_score() {
        let mut game = HanabiGame::default();
        game.fireworks[0] = 2;
        game.life_tokens = 1;
        game.hands[0][0] = Card::new(1, 4);

        assert_eq!(game.play(0, HanabiAction::Play(0)), Ok(-2.0));
        assert_eq!(game.life_tokens, 0);
        assert!(game.terminated);
        assert!(game.discards.contains(&Card::new(1, 4)));
    }

    /// The default encoder fills each documented section at its canonical offset.
    #[test]
    fn observation_has_exact_open_spiel_layout() {
        let game = HanabiGame::default();
        let observation = game.observe(0);
        let first_other_card = game.hands[1][0];

        assert_eq!(observation.values.len(), OBSERVATION_SIZE);
        assert_eq!(observation.values[first_other_card.identity()], 1);
        assert!(observation.values[127..167].iter().all(|value| *value == 1));
        assert!(observation.values[192..200].iter().all(|value| *value == 1));
        assert!(observation.values[200..203].iter().all(|value| *value == 1));
        assert!(observation.values[308..333].iter().all(|value| *value == 1));
    }

    /// Emptying the deck grants one final action to each player.
    #[test]
    fn deck_exhaustion_starts_two_action_countdown() {
        let deck: Vec<Card> = standard_deck().into_iter().take(11).collect();
        let mut game = HanabiGame::empty();
        game.reset_with_deck(deck);
        game.hands[0][0] = Card::new(0, 0);

        assert_eq!(game.play(0, HanabiAction::Play(0)), Ok(1.0));
        assert_eq!(game.final_turns, Some(2));
        assert!(!game.terminated);
        let first = game.hands[1][0];
        game.fireworks[first.color as usize] = first.rank;
        assert_eq!(game.play(1, HanabiAction::Play(0)), Ok(1.0));
        assert_eq!(game.final_turns, Some(1));
        let second = game.hands[0][0];
        game.fireworks[second.color as usize] = second.rank;
        assert_eq!(game.play(0, HanabiAction::Play(0)), Ok(1.0));
        assert_eq!(game.final_turns, Some(0));
        assert!(game.terminated);
    }

    /// Training preserves AEC turns and evaluation randomizes player zero.
    #[test]
    fn training_preserves_aec_turns_and_evaluation_randomizes_player_zero() {
        let mut first = HanabiTrainingEnv::default();
        let first_reset = first.reset(Some(7));
        assert_eq!(
            first_reset.observation.len(),
            OBSERVATION_SIZE + ACTION_COUNT
        );
        assert_eq!(
            HanabiTrainingEnv::action_mask(&first_reset.observation),
            first.game.legal_mask()
        );
        let action = first
            .game
            .legal_mask()
            .into_iter()
            .position(|legal| legal)
            .map(HanabiAction::from_index)
            .expect("a fresh game has legal actions");
        let first_step = first.step(action);

        let mut replay = HanabiTrainingEnv::default();
        replay.reset(Some(7));
        let replay_step = replay.step(action);
        assert_eq!(first_step, replay_step);
        assert_eq!(first.game.current_player, 1);

        let mut different = HanabiTrainingEnv::default();
        let different_reset = different.reset(Some(9_001));
        assert_ne!(first_reset.observation, different_reset.observation);

        let mut evaluation = HanabiTrainingEnv::default();
        evaluation.prepare_evaluation();
        evaluation.reset(Some(7));
        assert_eq!(evaluation.game.current_player, 0);
        let random_action = evaluation
            .random_legal_action()
            .expect("player zero has legal actions");
        let evaluation_reset = evaluation.step(random_action);
        assert_eq!(evaluation.game.current_player, 1);
        assert!(evaluation.game.last_action.is_some());
        assert_eq!(evaluation_reset.observation, evaluation.observation());
    }
}
