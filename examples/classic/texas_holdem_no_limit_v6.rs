//!
//! # Texas Hold'em No Limit
//!
//! ```{figure} classic_texas_holdem_no_limit.gif
//! :width: 140px
//! :name: texas_holdem_no_limit
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/texas_holdem_no_limit-v6")`         |
//! |--------------------|-----------------------------------------------------------|
//! | Actions            | Discrete                                                  |
//! | Parallel API       | Yes                                                       |
//! | Manual Control     | No                                                        |
//! | Agents             | `agents= ['player_0', 'player_1']`                        |
//! | Agents             | 2                                                         |
//! | Action Shape       | Discrete(5)                                               |
//! | Action Values      | Discrete(5)                                               |
//! | Observation Shape  | (54,)                                                     |
//! | Observation Values | [0, 100]                                                  |
//!
//!
//! Texas Hold'em No Limit is a variation of Texas Hold'em where there is no limit on the amount of each raise or the number of raises.
//!
//! Our implementation wraps [RLCard](http://rlcard.org/games.html#no-limit-texas-hold-em) and you can refer to its documentation for additional details. Please cite their work if you use this game in research.
//!
//! ### Arguments
//!
//! ```python
//! from pettingzoo import make
//!
//! make("aec", "classic/texas_holdem_no_limit-v6", num_players=2)
//! ```
//!
//! `num_players`: Sets the number of players in the game. Minimum is 2.
//!
//!
//! Texas Hold'em is a poker game involving 2 players and a regular 52 cards deck. At the beginning, both players get two cards. After betting, three community cards are shown and another round follows. At any time, a player could fold and the game will end. The winner will receive +1 as a reward and
//! the loser will get -1. This is an implementation of the standard limited version of Texas Hold'm, sometimes referred to as 'Limit Texas Hold'em'.
//!
//! Our implementation wraps [RLCard](http://rlcard.org/games.html#limit-texas-hold-em) and you can refer to its documentation for additional details. Please cite their work if you use this game in research.
//!
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//! The main observation space is similar to Texas Hold'em. The first 52 entries represent the union of the current player's hand and the community cards.
//!
//! |  Index  | Description                                  |  Values  |
//! |:-------:|----------------------------------------------|:--------:|
//! |  0 - 12 | Spades<br>_`0`: A, `1`: 2, ..., `12`: K_     |  [0, 1]  |
//! | 13 - 25 | Hearts<br>_`13`: A, `14`: 2, ..., `25`: K_   |  [0, 1]  |
//! | 26 - 38 | Diamonds<br>_`26`: A, `27`: 2, ..., `38`: K_ |  [0, 1]  |
//! | 39 - 51 | Clubs<br>_`39`: A, `40`: 2, ..., `51`: K_    |  [0, 1]  |
//! |    52   | Number of Chips of current acting player     | [0, 100] |
//! |    53   | Max Number of Chips of all players           | [0, 100] |
//!
//! #### Legal Actions Mask
//!
//! The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
//! whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
//!
//! ### Action Space
//!
//! | Action ID   |     Action         |
//! | ----------- | :----------------- |
//! | 0           | Fold               |
//! | 1           | Check & Call       |
//! | 2           | Raise Half Pot     |
//! | 3           | Raise Full Pot     |
//! | 4           | All In             |
//!
//! ### Rewards
//!
//! | Winner          | Loser           |
//! | :-------------: | :-------------: |
//! | +raised chips/2 | -raised chips/2 |
//!
//! ### Version History
//!
//! * v6: Upgrade to RLCard 1.0.5, fixes to the action space as ACPC (1.12.0)
//! * v5: Upgrade to RLCard 1.0.4, fixes to rewards with greater than 2 players (1.11.1)
//! * v4: Upgrade to RLCard 1.0.3 (1.11.0)
//! * v3: Fixed bug in arbitrary calls to observe() (1.8.0)
//! * v2: Bumped RLCard version, bug fixes, legal action mask in observation replaced illegal move list in infos (1.5.0)
//! * v1: Bumped RLCard version, fixed observation space, adopted new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
//! * v0: Initial versions release (1.0.0)
//!

#![expect(
    clippy::doc_markdown,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]

use std::cmp::Ordering;
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
use shakmaty as _;
use tokio as _;

use bevy_gym::training::{
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction, DqnConfig,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// RLCard's fixed starting stack.
const STARTING_CHIPS: u16 = 100;
/// Fold, check/call, half-pot, full-pot, and all-in.
const ACTION_COUNT: usize = 5;

/// One card in RLCard's suit-major observation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Card(u8);

impl Card {
    /// Construct one checked source card index.
    const fn new(index: u8) -> Self {
        assert!(index < 52);
        Self(index)
    }

    /// Observation index in `0..52`.
    const fn index(self) -> usize {
        self.0 as usize
    }

    /// Poker rank from two through ace.
    const fn rank(self) -> u8 {
        match self.0 % 13 {
            0 => 14,
            value => value + 1,
        }
    }

    /// Suit index in spades, hearts, diamonds, clubs order.
    const fn suit(self) -> u8 {
        self.0 / 13
    }
}

/// The five ACPC-compatible RLCard actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PokerAction {
    /// Concede the current pot.
    Fold,
    /// Check when matched or call the current wager.
    CheckCall,
    /// Raise by integer half of the current pot.
    RaiseHalfPot,
    /// Raise by the current pot.
    RaiseFullPot,
    /// Commit every remaining chip.
    AllIn,
}

impl PokerAction {
    /// Convert a table column into the source enum.
    const fn from_index(index: usize) -> Self {
        match index % ACTION_COUNT {
            0 => Self::Fold,
            1 => Self::CheckCall,
            2 => Self::RaiseHalfPot,
            3 => Self::RaiseFullPot,
            _ => Self::AllIn,
        }
    }

    /// Return the source action integer.
    const fn index(self) -> usize {
        match self {
            Self::Fold => 0,
            Self::CheckCall => 1,
            Self::RaiseHalfPot => 2,
            Self::RaiseFullPot => 3,
            Self::AllIn => 4,
        }
    }
}

impl DqnAction for PokerAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.index()
    }
}

/// RLCard player status relevant to heads-up no-limit Hold'em.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayerStatus {
    /// Still able to act.
    Alive,
    /// Conceded the pot.
    Folded,
    /// Committed the complete stack.
    AllIn,
}

/// Chips, cards, and status for one player.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PokerPlayer {
    /// Two private cards.
    hand: Vec<Card>,
    /// Chips committed across all rounds.
    in_chips: u16,
    /// Uncommitted stack.
    remaining: u16,
    /// Current participation state.
    status: PlayerStatus,
}

impl Default for PokerPlayer {
    fn default() -> Self {
        Self {
            hand: Vec::with_capacity(2),
            in_chips: 0,
            remaining: STARTING_CHIPS,
            status: PlayerStatus::Alive,
        }
    }
}

impl PokerPlayer {
    /// Commit at most the remaining stack.
    fn bet(&mut self, chips: u16) {
        let quantity = chips.min(self.remaining);
        self.in_chips += quantity;
        self.remaining -= quantity;
    }
}

/// One RLCard betting round.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BettingRound {
    /// Chips committed in this round by player.
    raised: [u16; 2],
    /// Number of actions since the last raise.
    not_raise: u8,
    /// Folded or all-in players accumulated across rounds.
    not_playing: u8,
}

impl BettingRound {
    /// Start the blind round.
    const fn with_blinds(raised: [u16; 2]) -> Self {
        Self {
            raised,
            not_raise: 0,
            not_playing: 0,
        }
    }

    /// Reset round-local wagers while retaining bypassed players.
    const fn start_next(&mut self) {
        self.raised = [0; 2];
        self.not_raise = 0;
    }

    /// Return whether all players agreed or bypassed action.
    const fn is_over(&self) -> bool {
        self.not_raise + self.not_playing >= 2
    }

    /// Return one player's round-local committed chips.
    const fn raised_for(&self, player: usize) -> u16 {
        let [first, second] = self.raised;
        if player == 0 {
            first
        } else {
            second
        }
    }

    /// Replace one player's round-local committed chips.
    const fn set_raised(&mut self, player: usize, chips: u16) {
        let [first, second] = &mut self.raised;
        if player == 0 {
            *first = chips;
        } else {
            *second = chips;
        }
    }

    /// Add chips to one player's round-local commitment.
    const fn add_raised(&mut self, player: usize, chips: u16) {
        let next = self.raised_for(player).saturating_add(chips);
        self.set_raised(player, next);
    }
}

/// Exact 54-value RLCard observation and selected-agent action mask.
#[derive(Debug, Clone, PartialEq)]
struct NoLimitObservation {
    /// Card union followed by current and maximum committed chips.
    values: [f32; 54],
    /// Legal ACPC actions for the selected player.
    action_mask: [bool; ACTION_COUNT],
}

/// Recoverable raw action rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PokerError {
    /// The caller supplied the unselected player.
    WrongPlayer {
        /// Player selected by the betting round.
        expected: u8,
        /// Player supplied by the caller.
        actual: u8,
    },
    /// The action is absent from the exact legal mask.
    IllegalAction(PokerAction),
    /// The hand already ended.
    HandFinished,
}

impl fmt::Display for PokerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPlayer { expected, actual } => {
                write!(
                    formatter,
                    "expected player {expected}, received player {actual}"
                )
            }
            Self::IllegalAction(action) => write!(formatter, "action {action:?} is illegal"),
            Self::HandFinished => formatter.write_str("the poker hand has finished"),
        }
    }
}

impl Error for PokerError {}

/// Remove one action from a fixed legal-action mask.
fn forbid(mask: &mut [bool; ACTION_COUNT], action: PokerAction) {
    if let Some(is_legal) = mask.get_mut(action.index()) {
        *is_legal = false;
    }
}

/// Seeded, source-compatible heads-up no-limit Hold'em state machine.
#[derive(Debug, Clone)]
struct NoLimitGame {
    /// Undealt cards in deal order.
    deck: VecDeque<Card>,
    /// Both source players.
    players: [PokerPlayer; 2],
    /// Revealed flop, turn, and river.
    public_cards: Vec<Card>,
    /// Dealer button player index.
    dealer: usize,
    /// Selected player index.
    current_player: usize,
    /// Current betting round.
    round: BettingRound,
    /// Completed betting rounds in `0..=4`.
    round_counter: u8,
    /// Terminal chip payoffs, if the hand ended.
    payoffs: Option<[i16; 2]>,
}

impl Default for NoLimitGame {
    fn default() -> Self {
        let mut game = Self::empty();
        game.reset(0);
        game
    }
}

impl NoLimitGame {
    /// Allocate an uninitialized game before arranging a deck.
    fn empty() -> Self {
        Self {
            deck: VecDeque::new(),
            players: std::array::from_fn(|_| PokerPlayer::default()),
            public_cards: Vec::with_capacity(5),
            dealer: 0,
            current_player: 0,
            round: BettingRound::with_blinds([0; 2]),
            round_counter: 0,
            payoffs: None,
        }
    }

    /// Borrow one of the two players without unchecked indexing.
    const fn player(&self, index: usize) -> &PokerPlayer {
        let [first, second] = &self.players;
        if index == 0 {
            first
        } else {
            second
        }
    }

    /// Mutably borrow one of the two players without unchecked indexing.
    const fn player_mut(&mut self, index: usize) -> &mut PokerPlayer {
        let [first, second] = &mut self.players;
        if index == 0 {
            first
        } else {
            second
        }
    }

    /// Shuffle and deal one regular RLCard hand from a deterministic seed.
    fn reset(&mut self, seed: u64) {
        let mut cards: Vec<Card> = (0..52).map(Card::new).collect();
        let mut rng = SplitMix64::new(seed);
        for upper in (1..cards.len()).rev() {
            let selected = rng.index(upper + 1);
            cards.swap(upper, selected);
        }
        let dealer = rng.index(2);
        self.reset_with_deck(cards, dealer);
    }

    /// Deal a caller-arranged deck for deterministic tutorial scenarios.
    fn reset_with_deck(&mut self, cards: Vec<Card>, dealer: usize) {
        self.deck = cards.into();
        self.players = std::array::from_fn(|_| PokerPlayer::default());
        self.public_cards.clear();
        self.dealer = dealer % 2;
        self.round_counter = 0;
        self.payoffs = None;
        for deal in 0..4 {
            let player = deal % 2;
            let card = self.deal_card();
            self.player_mut(player).hand.push(card);
        }
        let small_blind = (self.dealer + 1) % 2;
        let big_blind = self.dealer;
        self.player_mut(small_blind).bet(1);
        self.player_mut(big_blind).bet(2);
        self.current_player = small_blind;
        let [first, second] = &self.players;
        self.round = BettingRound::with_blinds([first.in_chips, second.in_chips]);
    }

    /// Remove the next card in RLCard deal order.
    fn deal_card(&mut self) -> Card {
        self.deck
            .pop_front()
            .expect("a 52-card deck covers one Hold'em hand")
    }

    /// Return the exact legal actions from RLCard 1.0.5's round rules.
    fn legal_mask(&self) -> [bool; ACTION_COUNT] {
        let mut legal = [true; ACTION_COUNT];
        let player = self.player(self.current_player);
        let maximum = self.round.raised.into_iter().max().unwrap_or_default();
        let current_raise = self.round.raised_for(self.current_player);
        let difference = maximum - current_raise;
        if difference > 0 && difference >= player.remaining {
            forbid(&mut legal, PokerAction::RaiseHalfPot);
            forbid(&mut legal, PokerAction::RaiseFullPot);
            forbid(&mut legal, PokerAction::AllIn);
        } else {
            let pot = self.pot();
            if pot > player.remaining {
                forbid(&mut legal, PokerAction::RaiseFullPot);
            }
            if pot / 2 > player.remaining || pot / 2 + current_raise <= maximum {
                forbid(&mut legal, PokerAction::RaiseHalfPot);
            }
        }
        legal
    }

    /// Build the 54-value observation for any player.
    fn observe(&self, player: usize) -> NoLimitObservation {
        let mut values = [0.0; 54];
        for card in self
            .player(player)
            .hand
            .iter()
            .chain(self.public_cards.iter())
        {
            if let Some(value) = values.get_mut(card.index()) {
                *value = 1.0;
            }
        }
        if let Some(value) = values.get_mut(52) {
            *value = f32::from(self.player(player).in_chips);
        }
        if let Some(value) = values.get_mut(53) {
            *value = f32::from(
                self.players
                    .iter()
                    .map(|candidate| candidate.in_chips)
                    .max()
                    .unwrap_or_default(),
            );
        }
        NoLimitObservation {
            values,
            action_mask: if self.payoffs.is_none() && player == self.current_player {
                self.legal_mask()
            } else {
                [false; ACTION_COUNT]
            },
        }
    }

    /// Apply one selected-player action and advance streets exactly once.
    fn play(&mut self, player: usize, action: PokerAction) -> Result<Option<[i16; 2]>, PokerError> {
        if self.payoffs.is_some() {
            return Err(PokerError::HandFinished);
        }
        if player != self.current_player {
            return Err(PokerError::WrongPlayer {
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
            return Err(PokerError::IllegalAction(action));
        }

        self.apply_action(action);
        if self.active_count() == 1 {
            self.payoffs = Some(self.calculate_payoffs());
            return Ok(self.payoffs);
        }
        if self.round.is_over() {
            self.advance_streets();
        }
        if self.round_counter >= 4 {
            self.payoffs = Some(self.calculate_payoffs());
        }
        Ok(self.payoffs)
    }

    /// Mutate wagers, status counters, and the circular player pointer.
    fn apply_action(&mut self, action: PokerAction) {
        let player_index = self.current_player;
        let maximum = self.round.raised.into_iter().max().unwrap_or_default();
        match action {
            PokerAction::CheckCall => {
                let difference = maximum - self.round.raised_for(player_index);
                self.round.set_raised(player_index, maximum);
                self.player_mut(player_index).bet(difference);
                self.round.not_raise += 1;
            }
            PokerAction::AllIn => {
                let quantity = self.player(player_index).remaining;
                self.round.add_raised(player_index, quantity);
                self.player_mut(player_index).bet(quantity);
                self.round.not_raise = 1;
            }
            PokerAction::RaiseFullPot => {
                let pot = self.pot();
                self.round.add_raised(player_index, pot);
                self.player_mut(player_index).bet(pot);
                self.round.not_raise = 1;
            }
            PokerAction::RaiseHalfPot => {
                let quantity = self.pot() / 2;
                self.round.add_raised(player_index, quantity);
                self.player_mut(player_index).bet(quantity);
                self.round.not_raise = 1;
            }
            PokerAction::Fold => self.player_mut(player_index).status = PlayerStatus::Folded,
        }
        if self.player(player_index).remaining == 0
            && self.player(player_index).status != PlayerStatus::Folded
        {
            self.player_mut(player_index).status = PlayerStatus::AllIn;
        }
        self.current_player = (self.current_player + 1) % 2;
        if self.player(player_index).status != PlayerStatus::Alive {
            self.round.not_playing += 1;
            if self.player(player_index).status == PlayerStatus::AllIn {
                self.round.not_raise = self.round.not_raise.saturating_sub(1);
            }
        }
        while self.player(self.current_player).status == PlayerStatus::Folded {
            self.current_player = (self.current_player + 1) % 2;
        }
    }

    /// Reveal flop, turn, and river, including RLCard's all-in fast-forward.
    fn advance_streets(&mut self) {
        let bypass = self
            .players
            .iter()
            .filter(|player| player.status != PlayerStatus::Alive)
            .count();
        self.current_player = (self.dealer + 1) % 2;
        if bypass < 2 {
            while self.player(self.current_player).status != PlayerStatus::Alive {
                self.current_player = (self.current_player + 1) % 2;
            }
        }
        if self.round_counter == 0 {
            for _ in 0..3 {
                let card = self.deal_card();
                self.public_cards.push(card);
            }
            if bypass == 2 {
                self.round_counter += 1;
            }
        }
        if self.round_counter == 1 {
            let card = self.deal_card();
            self.public_cards.push(card);
            if bypass == 2 {
                self.round_counter += 1;
            }
        }
        if self.round_counter == 2 {
            let card = self.deal_card();
            self.public_cards.push(card);
            if bypass == 2 {
                self.round_counter += 1;
            }
        }
        self.round_counter += 1;
        self.round.start_next();
    }

    /// Total committed chips used for pot-sized raises.
    fn pot(&self) -> u16 {
        self.players.iter().map(|player| player.in_chips).sum()
    }

    /// Number of non-folded players.
    fn active_count(&self) -> usize {
        self.players
            .iter()
            .filter(|player| player.status != PlayerStatus::Folded)
            .count()
    }

    /// Allocate the heads-up main pot and return net chip payoffs.
    fn calculate_payoffs(&self) -> [i16; 2] {
        let [first, second] = &self.players;
        let contributions = [first.in_chips, second.in_chips];
        let [first_contribution, second_contribution] = contributions;
        let matched = first_contribution.min(second_contribution);
        let mut allocated = [first_contribution - matched, second_contribution - matched];
        let winners = if first.status == PlayerStatus::Folded {
            [false, true]
        } else if second.status == PlayerStatus::Folded {
            [true, false]
        } else {
            compare_hands(&first.hand, &second.hand, &self.public_cards)
        };
        match (&mut allocated, winners) {
            ([first_allocated, _], [true, false]) => *first_allocated += matched * 2,
            ([_, second_allocated], [false, true]) => *second_allocated += matched * 2,
            (_, [true, true]) => {
                let [first_allocated, second_allocated] = &mut allocated;
                *first_allocated += matched;
                *second_allocated += matched;
            }
            (_, [false, false]) => {}
        }
        let [first_allocated, second_allocated] = allocated;
        [
            first_allocated as i16 - first_contribution as i16,
            second_allocated as i16 - second_contribution as i16,
        ]
    }
}

/// Deterministic random stream used for shuffle and dealer selection.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Current state.
    state: u64,
}

impl SplitMix64 {
    /// Start one stream.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return one index below a nonzero bound.
    fn index(&mut self, upper: usize) -> usize {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        usize::try_from((value ^ (value >> 31)) % upper as u64).unwrap_or_default()
    }
}

/// Comparable five-card poker value, from high card through straight flush.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct HandValue([u8; 6]);

/// Compare two seven-card Hold'em holdings.
fn compare_hands(first: &[Card], second: &[Card], public: &[Card]) -> [bool; 2] {
    let first_value = best_hand(first, public);
    let second_value = best_hand(second, public);
    match first_value.cmp(&second_value) {
        Ordering::Greater => [true, false],
        Ordering::Less => [false, true],
        Ordering::Equal => [true, true],
    }
}

/// Return the best five-card value from two private and five public cards.
fn best_hand(private: &[Card], public: &[Card]) -> HandValue {
    let cards: Vec<Card> = private.iter().chain(public.iter()).copied().collect();
    let mut best = HandValue([0; 6]);
    for a in 0..cards.len() - 4 {
        for b in a + 1..cards.len() - 3 {
            for c in b + 1..cards.len() - 2 {
                for d in c + 1..cards.len() - 1 {
                    for e in d + 1..cards.len() {
                        if let (Some(&a), Some(&b), Some(&c), Some(&d), Some(&e)) = (
                            cards.get(a),
                            cards.get(b),
                            cards.get(c),
                            cards.get(d),
                            cards.get(e),
                        ) {
                            best = best.max(five_card_value([a, b, c, d, e]));
                        }
                    }
                }
            }
        }
    }
    best
}

/// Rank one five-card hand with lexicographic kickers.
fn five_card_value(cards: [Card; 5]) -> HandValue {
    let mut ranks: Vec<u8> = cards.iter().map(|card| card.rank()).collect();
    ranks.sort_unstable_by(|left, right| right.cmp(left));
    let first_suit = cards.first().map_or(0, |card| card.suit());
    let flush = cards.iter().all(|card| card.suit() == first_suit);
    let mut unique = ranks.clone();
    unique.dedup();
    let straight_high = if unique == [14, 5, 4, 3, 2] {
        Some(5)
    } else if let [high, _, _, _, low] = unique.as_slice() {
        (*high - *low == 4).then_some(*high)
    } else {
        None
    };
    let mut frequencies = [0_u8; 15];
    for rank in &ranks {
        if let Some(count) = frequencies.get_mut(usize::from(*rank)) {
            *count = count.saturating_add(1);
        }
    }
    let mut counts: Vec<(u8, u8)> = unique
        .iter()
        .map(|rank| {
            (
                *rank,
                frequencies.get(usize::from(*rank)).copied().unwrap_or(0),
            )
        })
        .collect();
    counts.sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| right.0.cmp(&left.0)));
    if let Some(high) = straight_high.filter(|_| flush) {
        return HandValue([8, high, 0, 0, 0, 0]);
    }
    let first = counts.first().copied().unwrap_or_default();
    let second = counts.get(1).copied().unwrap_or_default();
    let third = counts.get(2).copied().unwrap_or_default();
    let fourth = counts.get(3).copied().unwrap_or_default();
    let ranks = match ranks.as_slice() {
        [first, second, third, fourth, fifth] => [*first, *second, *third, *fourth, *fifth],
        _ => [0; 5],
    };
    let [first_rank, second_rank, third_rank, fourth_rank, fifth_rank] = ranks;
    if first.1 == 4 {
        return HandValue([7, first.0, second.0, 0, 0, 0]);
    }
    if first.1 == 3 && second.1 == 2 {
        return HandValue([6, first.0, second.0, 0, 0, 0]);
    }
    if flush {
        return HandValue([
            5,
            first_rank,
            second_rank,
            third_rank,
            fourth_rank,
            fifth_rank,
        ]);
    }
    if let Some(high) = straight_high {
        return HandValue([4, high, 0, 0, 0, 0]);
    }
    if first.1 == 3 {
        return HandValue([3, first.0, second.0, third.0, 0, 0]);
    }
    if first.1 == 2 && second.1 == 2 {
        return HandValue([2, first.0, second.0, third.0, 0, 0]);
    }
    if first.1 == 2 {
        return HandValue([1, first.0, second.0, third.0, fourth.0, 0]);
    }
    HandValue([
        0,
        first_rank,
        second_rank,
        third_rank,
        fourth_rank,
        fifth_rank,
    ])
}

/// Fixed aces-over-kings deck used only by raw rules tests.
#[cfg(test)]
fn deterministic_test_deck() -> Vec<Card> {
    let opening = [0, 12, 13, 25, 27, 45, 34, 50, 37];
    let mut used = [false; 52];
    for index in opening {
        if let Some(is_used) = used.get_mut(index) {
            *is_used = true;
        }
    }
    opening
        .into_iter()
        .chain((0..52).filter(|index| !used.get(*index).copied().unwrap_or(false)))
        .map(|index| Card::new(index as u8))
        .collect()
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
struct NoLimitTrainingEnv {
    /// Complete raw game.
    game: NoLimitGame,
    /// Opponent action stream reset from the environment seed.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for NoLimitTrainingEnv {
    fn default() -> Self {
        let mut game = NoLimitGame::empty();
        game.reset(0);
        Self {
            game,
            rng: SplitMix64::new(0),
            control: ControlMode::SharedPolicy,
        }
    }
}

/// Multi-agent source details returned by the adapter.
#[derive(Debug, Clone, PartialEq)]
struct NoLimitInfo {
    /// Observations in player order.
    observations: [NoLimitObservation; 2],
    /// Exact net chip rewards.
    rewards: [i16; 2],
}

impl Default for NoLimitInfo {
    fn default() -> Self {
        let observation = NoLimitObservation {
            values: [0.0; 54],
            action_mask: [false; ACTION_COUNT],
        };
        Self {
            observations: [observation.clone(), observation],
            rewards: [0; 2],
        }
    }
}

impl NoLimitTrainingEnv {
    /// Build current source observations and reward evidence.
    fn info(&self, rewards: [i16; 2]) -> NoLimitInfo {
        NoLimitInfo {
            observations: [self.game.observe(0), self.game.observe(1)],
            rewards,
        }
    }

    /// Flatten the learner observation and append its exact action mask.
    fn observation(&self) -> Vec<f32> {
        let player = match self.control {
            ControlMode::SharedPolicy => self.game.current_player,
            ControlMode::PolicyAsPlayerOneVsRandomPlayerZero => 1,
        };
        let observation = self.game.observe(player);
        observation
            .values
            .into_iter()
            .chain(
                observation
                    .action_mask
                    .into_iter()
                    .map(u8::from)
                    .map(f32::from),
            )
            .collect()
    }

    /// Select one uniformly random legal action for player zero.
    fn random_player_zero_action(&mut self) -> PokerAction {
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
            .unwrap_or_default();
        PokerAction::from_index(action)
    }
}

impl Env for NoLimitTrainingEnv {
    type Observation = Vec<f32>;
    type Action = PokerAction;
    type Info = NoLimitInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        let seed = seed.unwrap_or_default();
        self.game.reset(seed);
        self.rng = SplitMix64::new(seed ^ 0xa5a5_a5a5_a5a5_a5a5);
        Reset {
            observation: self.observation(),
            info: self.info([0; 2]),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let actor = self.game.current_player;
        let result = self.game.play(actor, action);
        let (rewards, status) = match result {
            Err(_error) => ([-1, 0], EpisodeStatus::Terminated),
            Ok(Some(payoffs)) => (payoffs, EpisodeStatus::Terminated),
            Ok(None) => ([0; 2], EpisodeStatus::Continuing),
        };
        let [player_zero_reward, player_one_reward] = rewards;
        Step {
            observation: self.observation(),
            reward: f64::from(match self.control {
                ControlMode::SharedPolicy if actor == 0 => player_zero_reward,
                ControlMode::SharedPolicy | ControlMode::PolicyAsPlayerOneVsRandomPlayerZero => {
                    player_one_reward
                }
            }),
            status,
            info: self.info(rewards),
        }
    }
}

impl DiscreteDqnExample for NoLimitTrainingEnv {
    const ENV_NAME: &'static str = "texas-holdem-no-limit-v6";
    const GYMNASIUM_ID: &'static str = "classic/texas_holdem_no_limit-v6";
    const OBSERVATION_DIM: usize = 54;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const ZERO_SUM_COMPARISON: bool = true;
    const BOOTSTRAP_MULTIPLIER: f32 = -1.0;
    const MIN_TRAIN_STEPS: usize = 100_000;
    const SOLVED_MEAN_REWARD: f64 = 0.0;
    const GIF_PATH: &'static str = "docs/images/classic-texas-holdem-no-limit-v6.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 1.0,
            learning_rate,
            replay_capacity: 100_000,
            min_replay_size: 1_024,
            batch_size: 128,
            target_update_interval: 1_000,
            epsilon_decay_steps: 250_000,
            ..DqnConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation.iter().take(54).copied().collect()
    }

    fn action_mask(observation: &[f32]) -> Vec<bool> {
        observation
            .iter()
            .skip(54)
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
            Ok(self.random_player_zero_action())
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
    run_discrete_workflow::<NoLimitTrainingEnv>()
}

/// RLCard-style card-table renderer for shuffled seeded hands.
#[cfg(feature = "render")]
mod render {
    use super::{
        Card, DiscreteDqnExample, Error, NoLimitGame, NoLimitTrainingEnv, Path, PokerAction,
        SplitMix64, ACTION_COUNT,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;
    use bevy_inspector_egui as _;

    /// Source documentation GIF dimensions for two players.
    const WIDTH: f32 = 650.0;
    /// Integer viewport width used by window and encoder APIs.
    const WIDTH_PIXELS: u32 = 650;
    /// Source renderer default height.
    const HEIGHT: f32 = 1_000.0;
    /// Integer viewport height used by window and encoder APIs.
    const HEIGHT_PIXELS: u32 = 1_000;
    /// `screen_height * 2 / 10`.
    const TILE_SIZE: f32 = 200.0;
    /// Source card width `tile_size * 142 / 197`.
    const CARD_WIDTH: f32 = TILE_SIZE * 142.0 / 197.0;

    /// Green RLCard table background.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb_u8(7, 99, 36)));
        }
    }

    /// Checkpoint policy and visible raw AEC hand.
    #[derive(Resource)]
    struct VisualGame {
        /// Greedy player-zero DQN policy.
        policy: DqnPolicy,
        /// Exact visible game.
        game: NoLimitGame,
        /// Seeded player-one action stream.
        rng: SplitMix64,
        /// Seed for the next shuffled hand.
        next_seed: u64,
    }

    /// One of four private-card or five community-card slots.
    #[derive(Component)]
    struct CardSprite {
        /// Player hand index or community group.
        group: CardGroup,
        /// Index within that group.
        index: usize,
    }

    /// Visual card ownership group.
    #[derive(Debug, Clone, Copy)]
    enum CardGroup {
        /// Player zero's private hand.
        PlayerZero,
        /// Player one's private hand.
        PlayerOne,
        /// Shared community cards.
        Public,
    }

    /// One chip stack sprite slot.
    #[derive(Component)]
    struct ChipSprite {
        /// Owning player.
        player: usize,
        /// Stack index below 100.
        index: usize,
    }

    /// Dynamic committed-chip text.
    #[derive(Component)]
    struct ChipLabel(usize);

    /// One-Hz RLCard playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run interactive playback or a finite source-sized capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = NoLimitTrainingEnv::dqn_config(NoLimitTrainingEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(checkpoint, 54, ACTION_COUNT, &config.hidden_sizes)?;
        let mut game = NoLimitGame::empty();
        game.reset(0);
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualGame {
                policy,
                game,
                rng: SplitMix64::new(12_345),
                next_seed: 1,
            })
            .insert_resource(VisualClock(Timer::from_seconds(1.0, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets/rlcard_envs".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Texas Hold'em No Limit".into(),
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

    /// Spawn source labels plus reusable card and chip slots.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("No-Limit Hold'em Camera")));
        let font = assets.load("font/Minecraft.ttf");
        for player in 0..2 {
            let source_y = if player == 0 { 30.0 } else { 980.0 };
            commands.spawn((
                Text2d::new(format!("Player {}", player + 1)),
                TextFont {
                    font: font.clone(),
                    font_size: 36.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Anchor::CENTER,
                Transform::from_xyz(source_x(325.0), source_y_to_world(source_y), 2.0),
            ));
            commands.spawn((
                Text2d::new("0"),
                TextFont {
                    font: font.clone(),
                    font_size: 24.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Anchor::CENTER,
                Transform::default(),
                ChipLabel(player),
            ));
            for index in 0..2 {
                commands.spawn((
                    Sprite::default(),
                    Transform::default(),
                    Visibility::Hidden,
                    CardSprite {
                        group: if player == 0 {
                            CardGroup::PlayerZero
                        } else {
                            CardGroup::PlayerOne
                        },
                        index,
                    },
                ));
            }
            for index in 0..100 {
                commands.spawn((
                    Sprite::default(),
                    Transform::default(),
                    Visibility::Hidden,
                    ChipSprite { player, index },
                ));
            }
        }
        for index in 0..5 {
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                CardSprite {
                    group: CardGroup::Public,
                    index,
                },
            ));
        }
    }

    /// Advance one selected raw AEC player at RLCard's one FPS.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualGame>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Apply one random player-zero action, one learned player-one action, or reset.
    fn advance_visual(visual: &mut VisualGame) {
        if visual.game.payoffs.is_some() {
            visual.game.reset(visual.next_seed);
            visual.next_seed = visual.next_seed.wrapping_add(1);
            return;
        }
        let player = visual.game.current_player;
        let action = if player == 1 {
            let observation = NoLimitTrainingEnv {
                game: visual.game.clone(),
                rng: visual.rng,
                control: super::ControlMode::PolicyAsPlayerOneVsRandomPlayerZero,
            }
            .observation();
            match NoLimitTrainingEnv::deployment_action(&visual.policy, &observation) {
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
                Some(index) => PokerAction::from_index(index),
                None => return,
            }
        };
        if visual.game.play(player, action).is_err() {
            visual.game.reset(visual.next_seed);
            visual.next_seed = visual.next_seed.wrapping_add(1);
        }
    }

    /// Synchronize cards, chip stacks, and committed-chip labels.
    fn sync_scene(
        visual: Res<'_, VisualGame>,
        assets: Res<'_, AssetServer>,
        mut cards: Query<
            '_,
            '_,
            (&CardSprite, &mut Sprite, &mut Transform, &mut Visibility),
            (Without<ChipSprite>, Without<ChipLabel>),
        >,
        mut chips: Query<
            '_,
            '_,
            (&ChipSprite, &mut Sprite, &mut Transform, &mut Visibility),
            (Without<CardSprite>, Without<ChipLabel>),
        >,
        mut labels: Query<
            '_,
            '_,
            (&ChipLabel, &mut Text2d, &mut Transform),
            (Without<CardSprite>, Without<ChipSprite>),
        >,
    ) {
        if visual.is_changed() {
            apply_scene(&visual, &assets, &mut cards, &mut chips, &mut labels);
        }
    }

    /// Apply PettingZoo's exact two-player card and chip placement formulas.
    fn apply_scene(
        visual: &VisualGame,
        assets: &AssetServer,
        cards: &mut Query<
            '_,
            '_,
            (&CardSprite, &mut Sprite, &mut Transform, &mut Visibility),
            (Without<ChipSprite>, Without<ChipLabel>),
        >,
        chips: &mut Query<
            '_,
            '_,
            (&ChipSprite, &mut Sprite, &mut Transform, &mut Visibility),
            (Without<CardSprite>, Without<ChipLabel>),
        >,
        labels: &mut Query<
            '_,
            '_,
            (&ChipLabel, &mut Text2d, &mut Transform),
            (Without<CardSprite>, Without<ChipSprite>),
        >,
    ) {
        for (slot, mut sprite, mut transform, mut visibility) in cards.iter_mut() {
            let (source, x, y) = match slot.group {
                CardGroup::PlayerZero => (
                    visual.game.players[0].hand.get(slot.index).copied(),
                    hand_x(slot.index),
                    50.0,
                ),
                CardGroup::PlayerOne => (
                    visual.game.players[1].hand.get(slot.index).copied(),
                    hand_x(slot.index),
                    750.0,
                ),
                CardGroup::Public => {
                    let card = visual.game.public_cards.get(slot.index).copied();
                    let (x, y) = public_position(slot.index, visual.game.public_cards.len());
                    (card, x, y)
                }
            };
            let Some(card) = source else {
                *visibility = Visibility::Hidden;
                continue;
            };
            sprite.image = assets.load(format!("img/{}.png", card_code(card)));
            sprite.custom_size = Some(Vec2::new(CARD_WIDTH, TILE_SIZE));
            transform.translation = top_left_to_world(x, y, CARD_WIDTH, TILE_SIZE, 1.0);
            *visibility = Visibility::Visible;
        }
        for (slot, mut sprite, mut transform, mut visibility) in chips.iter_mut() {
            let total = usize::from(visual.game.player(slot.player).in_chips);
            if slot.index >= total {
                *visibility = Visibility::Hidden;
                continue;
            }
            sprite.image = assets.load("img/ChipWhite.png");
            sprite.custom_size = Some(Vec2::new(100.0, TILE_SIZE * 16.0 / 45.0));
            let base_y = if slot.player == 0 { 150.0 } else { 850.0 };
            let y = base_y - slot.index as f32 * TILE_SIZE / 15.0;
            transform.translation =
                top_left_to_world(495.0, y, 100.0, TILE_SIZE * 16.0 / 45.0, 1.0);
            *visibility = Visibility::Visible;
        }
        for (label, mut text, mut transform) in labels.iter_mut() {
            let chips = visual.game.player(label.0).in_chips;
            **text = chips.to_string();
            let count = usize::from(chips);
            let base_y = if label.0 == 0 { 150.0 } else { 850.0 };
            let y = base_y - (count + 1) as f32 * TILE_SIZE / 15.0;
            transform.translation = Vec3::new(source_x(545.0), source_y_to_world(y), 2.0);
        }
    }

    /// Source x-position for a two-card private hand.
    const fn hand_x(index: usize) -> f32 {
        if index == 0 {
            171.0
        } else {
            335.0
        }
    }

    /// Convert one source card index into its two-character asset stem.
    fn card_code(card: Card) -> String {
        const SUITS: [char; 4] = ['S', 'H', 'D', 'C'];
        const RANKS: [char; 13] = [
            'A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K',
        ];
        let suit = SUITS.get(usize::from(card.suit())).copied().unwrap_or('S');
        let rank = RANKS.get(usize::from(card.0 % 13)).copied().unwrap_or('A');
        format!("{suit}{rank}")
    }

    /// Source x/y placement for three- and five-card community layouts.
    fn public_position(index: usize, card_count: usize) -> (f32, f32) {
        if card_count <= 3 {
            (
                [89.0, 253.0, 417.0].get(index).copied().unwrap_or(0.0),
                400.0,
            )
        } else if index <= 2 {
            (
                [89.0, 253.0, 417.0].get(index).copied().unwrap_or(0.0),
                290.0,
            )
        } else {
            (
                [171.0, 335.0]
                    .get(index.saturating_sub(3))
                    .copied()
                    .unwrap_or(0.0),
                510.0,
            )
        }
    }

    /// Convert a source top-left rectangle into Bevy world coordinates.
    fn top_left_to_world(x: f32, y: f32, width: f32, height: f32, z: f32) -> Vec3 {
        Vec3::new(
            x + width / 2.0 - WIDTH / 2.0,
            HEIGHT / 2.0 - y - height / 2.0,
            z,
        )
    }

    /// Convert one source x coordinate into centered Bevy space.
    const fn source_x(x: f32) -> f32 {
        x - WIDTH / 2.0
    }

    /// Convert one source y coordinate into centered Bevy space.
    const fn source_y_to_world(y: f32) -> f32 {
        HEIGHT / 2.0 - y
    }

    /// Capture five seconds at 20 FPS while actions advance at one FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualGame>,
        assets: Res<'_, AssetServer>,
        mut cards: Query<
            '_,
            '_,
            (&CardSprite, &mut Sprite, &mut Transform, &mut Visibility),
            (Without<ChipSprite>, Without<ChipLabel>),
        >,
        mut chips: Query<
            '_,
            '_,
            (&ChipSprite, &mut Sprite, &mut Transform, &mut Visibility),
            (Without<CardSprite>, Without<ChipLabel>),
        >,
        mut labels: Query<
            '_,
            '_,
            (&ChipLabel, &mut Text2d, &mut Transform),
            (Without<CardSprite>, Without<ChipSprite>),
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
        if capture.has_started() && capture.frame_index().is_multiple_of(20) {
            advance_visual(&mut visual);
        }
        apply_scene(&visual, &assets, &mut cards, &mut chips, &mut labels);
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Encode the source-sized five-second demonstration.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-texas-holdem-no-limit-v6",
            WIDTH_PIXELS,
            HEIGHT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        best_hand, deterministic_test_deck, Card, DiscreteDqnExample, Env, NoLimitGame,
        NoLimitTrainingEnv, PlayerStatus, PokerAction, ACTION_COUNT,
    };

    #[test]
    fn heads_up_blinds_and_initial_legal_actions_match_rlcard() {
        let mut game = NoLimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 1);

        assert_eq!(game.current_player, 0);
        assert_eq!(game.players[0].in_chips, 1);
        assert_eq!(game.players[1].in_chips, 2);
        assert_eq!(game.legal_mask(), [true, true, false, true, true]);
    }

    #[test]
    fn all_in_restricts_the_caller_to_fold_or_check_call() -> Result<(), super::PokerError> {
        let mut game = NoLimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 1);

        game.play(0, PokerAction::AllIn)?;

        assert_eq!(game.players[0].status, PlayerStatus::AllIn);
        assert_eq!(game.legal_mask(), [true, true, false, false, false]);
        Ok(())
    }

    #[test]
    fn aces_over_kings_all_in_returns_exact_stack_payoff() -> Result<(), super::PokerError> {
        let mut game = NoLimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 1);
        game.play(0, PokerAction::AllIn)?;

        let result = game.play(1, PokerAction::CheckCall)?;

        assert_eq!(result, Some([100, -100]));
        assert_eq!(game.public_cards.len(), 5);
        Ok(())
    }

    #[test]
    fn observation_contains_private_public_cards_and_committed_chips(
    ) -> Result<(), super::PokerError> {
        let mut game = NoLimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 1);
        let initial = game.observe(0);
        assert_eq!(initial.values[Card::new(0).index()], 1.0);
        assert_eq!(initial.values[Card::new(13).index()], 1.0);
        assert_eq!(initial.values[52], 1.0);
        assert_eq!(initial.values[53], 2.0);

        game.play(0, PokerAction::AllIn)?;
        game.play(1, PokerAction::CheckCall)?;
        let terminal = game.observe(0);
        assert_eq!(
            terminal
                .values
                .iter()
                .take(52)
                .filter(|value| **value == 1.0)
                .count(),
            7
        );
        assert_eq!(terminal.values[52], 100.0);
        assert_eq!(terminal.action_mask, [false; 5]);
        Ok(())
    }

    #[test]
    fn hand_evaluator_orders_straight_flush_above_four_of_a_kind() {
        let straight_flush = [Card::new(9), Card::new(10)];
        let straight_board = [
            Card::new(5),
            Card::new(6),
            Card::new(7),
            Card::new(8),
            Card::new(26),
        ];
        let quads = [Card::new(0), Card::new(13)];
        let quads_board = [
            Card::new(26),
            Card::new(39),
            Card::new(1),
            Card::new(15),
            Card::new(29),
        ];

        assert!(best_hand(&straight_flush, &straight_board) > best_hand(&quads, &quads_board));
    }

    #[test]
    fn training_adapter_uses_shuffled_hands_and_complete_legal_masks() {
        let mut first = NoLimitTrainingEnv::default();
        let mut second = NoLimitTrainingEnv::default();
        let mut different = NoLimitTrainingEnv::default();

        let first_reset = first.reset(Some(42));
        let second_reset = second.reset(Some(42));
        let different_reset = different.reset(Some(9_001));

        assert_eq!(first_reset.observation, second_reset.observation);
        assert_ne!(first_reset.observation, different_reset.observation);
        assert_eq!(first_reset.observation.len(), 54 + ACTION_COUNT);
        let legal = &first_reset.observation[54..];
        assert!(legal.iter().filter(|value| **value > 0.5).count() >= 2);
        assert_eq!(first_reset.observation, first.observation());

        let mut evaluation = NoLimitTrainingEnv::default();
        evaluation.prepare_evaluation();
        for seed in 0..16 {
            evaluation.reset(Some(seed));
            if evaluation.game.current_player == 0 {
                break;
            }
        }
        assert_eq!(evaluation.game.current_player, 0);
        let action = evaluation.random_player_zero_action();
        assert!(evaluation.game.legal_mask()[action.index()]);
        let step = evaluation.step(action);
        assert_eq!(step.reward, f64::from(step.info.rewards[1]));
    }
}
