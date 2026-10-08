//!
//! # Texas Hold'em
//!
//! ```{figure} classic_texas_holdem.gif
//! :width: 140px
//! :name: texas_holdem
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/texas_holdem-v4")`         |
//! |--------------------|--------------------------------------------------|
//! | Actions            | Discrete                                         |
//! | Parallel API       | Yes                                              |
//! | Manual Control     | No                                               |
//! | Agents             | `agents= ['player_0', 'player_1']`               |
//! | Agents             | 2                                                |
//! | Action Shape       | Discrete(4)                                      |
//! | Action Values      | Discrete(4)                                      |
//! | Observation Shape  | (72,)                                            |
//! | Observation Values | [0, 1]                                           |
//!
//!
//! ## Arguments
//!
//! ```python
//! from pettingzoo import make
//!
//! make("aec", "classic/texas_holdem-v4", num_players=2)
//! ```
//!
//! `num_players`: Sets the number of players in the game. Minimum is 2.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//! The main observation space is a vector of 72 boolean integers. The first 52 entries depict the current player's hand plus any community cards as follows
//!
//! |  Index  | Description                                                 |
//! |:-------:|-------------------------------------------------------------|
//! |  0 - 12 | Spades<br>_`0`: A, `1`: 2, ..., `12`: K_                    |
//! | 13 - 25 | Hearts<br>_`13`: A, `14`: 2, ..., `25`: K_                  |
//! | 26 - 38 | Diamonds<br>_`26`: A, `27`: 2, ..., `38`: K_                |
//! | 39 - 51 | Clubs<br>_`39`: A, `40`: 2, ..., `51`: K_                   |
//! | 52 - 56 | Chips raised in Round 1<br>_`52`: 0, `53`: 1, ..., `56`: 4_ |
//! | 57 - 61 | Chips raised in Round 2<br>_`57`: 0, `58`: 1, ..., `61`: 4_ |
//! | 62 - 66 | Chips raised in Round 3<br>_`62`: 0, `63`: 1, ..., `66`: 4_ |
//! | 67 - 71 | Chips raised in Round 4<br>_`67`: 0, `68`: 1, ..., `71`: 4_ |
//!
//! #### Legal Actions Mask
//!
//! The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
//! whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.
//!
//! ### Action Space
//!
//! | Action ID | Action |
//! |:---------:|--------|
//! |     0     | Call   |
//! |     1     | Raise  |
//! |     2     | Fold   |
//! |     3     | Check  |
//!
//! ### Rewards
//!
//! | Winner          | Loser           |
//! | :-------------: | :-------------: |
//! | +raised chips/2 | -raised chips/2 |
//!
//! ### Version History
//!
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

use bevy_gym::training::{
    run_discrete_workflow, DiscreteDqnExample, DiscreteEvaluation, DqnAction, DqnConfig,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

/// Call, raise, fold, and check.
const ACTION_COUNT: usize = 4;

/// One card in RLCard's suit-major 52-bit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Card(u8);

impl Card {
    /// Construct a checked card index.
    const fn new(index: u8) -> Self {
        assert!(index < 52);
        Self(index)
    }

    /// Source observation index.
    const fn index(self) -> usize {
        self.0 as usize
    }

    /// Poker rank from two through ace.
    const fn rank(self) -> u8 {
        match self.0 % 13 {
            0 => 14,
            rank => rank + 1,
        }
    }

    /// Suit index.
    const fn suit(self) -> u8 {
        self.0 / 13
    }
}

/// RLCard's four limit betting actions and IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LimitAction {
    /// Match the current round wager.
    Call,
    /// Match and add the fixed raise amount.
    Raise,
    /// Concede the pot.
    Fold,
    /// Continue without adding chips when matched.
    Check,
}

impl LimitAction {
    /// Convert a table column into the source action enum.
    const fn from_index(index: usize) -> Self {
        match index % ACTION_COUNT {
            0 => Self::Call,
            1 => Self::Raise,
            2 => Self::Fold,
            _ => Self::Check,
        }
    }

    /// Return the source action integer.
    const fn index(self) -> usize {
        match self {
            Self::Call => 0,
            Self::Raise => 1,
            Self::Fold => 2,
            Self::Check => 3,
        }
    }
}

impl DqnAction for LimitAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.index()
    }
}

/// One source player.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LimitPlayer {
    /// Two private cards.
    hand: Vec<Card>,
    /// Chips committed in this hand.
    in_chips: u16,
    /// Whether the player has folded.
    folded: bool,
}

impl Default for LimitPlayer {
    fn default() -> Self {
        Self {
            hand: Vec::with_capacity(2),
            in_chips: 0,
            folded: false,
        }
    }
}

/// RLCard's per-street fixed-limit counters.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LimitRound {
    /// Round-local committed chips.
    raised: [u16; 2],
    /// Fixed raise size, two preflop/flop and four turn/river.
    raise_amount: u16,
    /// Raises made in this round.
    have_raised: u8,
    /// Actions since the last raise.
    not_raise: u8,
}

impl LimitRound {
    /// Create the blind round.
    const fn with_blinds(raised: [u16; 2]) -> Self {
        Self {
            raised,
            raise_amount: 2,
            have_raised: 0,
            not_raise: 0,
        }
    }

    /// Reset one new street.
    const fn start_next(&mut self, raise_amount: u16) {
        self.raised = [0; 2];
        self.raise_amount = raise_amount;
        self.have_raised = 0;
        self.not_raise = 0;
    }

    /// Return whether both players responded since the last raise.
    const fn is_over(&self) -> bool {
        self.not_raise >= 2
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
}

/// Exact 72-bit source observation dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LimitObservation {
    /// Card union and four one-hot raise-count groups.
    values: [u8; 72],
    /// Legal actions for the selected player.
    action_mask: [bool; ACTION_COUNT],
}

/// Typed raw-environment action rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LimitError {
    /// The caller supplied the unselected player.
    WrongPlayer {
        /// Player selected by the betting round.
        expected: u8,
        /// Player supplied by the caller.
        actual: u8,
    },
    /// Action absent from the current legal mask.
    IllegalAction(LimitAction),
    /// The hand has already ended.
    HandFinished,
}

impl fmt::Display for LimitError {
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

impl Error for LimitError {}

/// Source-compatible two-player limit Hold'em state machine.
#[derive(Debug, Clone)]
struct LimitGame {
    /// Undealt cards in deal order.
    deck: VecDeque<Card>,
    /// Both players.
    players: [LimitPlayer; 2],
    /// Revealed cards.
    public_cards: Vec<Card>,
    /// Selected player.
    current_player: usize,
    /// Current betting round.
    round: LimitRound,
    /// Completed street count.
    round_counter: u8,
    /// Raise count retained for each source observation group.
    raise_history: [u8; 4],
    /// Terminal big-blind-scaled rewards.
    rewards: Option<[i16; 2]>,
}

impl Default for LimitGame {
    fn default() -> Self {
        let mut game = Self::empty();
        game.reset(0);
        game
    }
}

impl LimitGame {
    /// Allocate an empty game before dealing.
    fn empty() -> Self {
        Self {
            deck: VecDeque::new(),
            players: std::array::from_fn(|_| LimitPlayer::default()),
            public_cards: Vec::with_capacity(5),
            current_player: 0,
            round: LimitRound::with_blinds([0; 2]),
            round_counter: 0,
            raise_history: [0; 4],
            rewards: None,
        }
    }

    /// Borrow one of the two players without unchecked indexing.
    const fn player(&self, index: usize) -> &LimitPlayer {
        let [first, second] = &self.players;
        if index == 0 {
            first
        } else {
            second
        }
    }

    /// Mutably borrow one of the two players without unchecked indexing.
    const fn player_mut(&mut self, index: usize) -> &mut LimitPlayer {
        let [first, second] = &mut self.players;
        if index == 0 {
            first
        } else {
            second
        }
    }

    /// Shuffle and deal one deterministic seeded hand.
    fn reset(&mut self, seed: u64) {
        let mut cards: Vec<Card> = (0..52).map(Card::new).collect();
        let mut rng = SplitMix64::new(seed);
        for upper in (1..cards.len()).rev() {
            let selected = rng.index(upper + 1);
            cards.swap(upper, selected);
        }
        self.reset_with_deck(cards, rng.index(2));
    }

    /// Deal one caller-arranged tutorial hand and blind assignment.
    fn reset_with_deck(&mut self, cards: Vec<Card>, small_blind: usize) {
        self.deck = cards.into();
        self.players = std::array::from_fn(|_| LimitPlayer::default());
        self.public_cards.clear();
        self.round_counter = 0;
        self.raise_history = [0; 4];
        self.rewards = None;
        for deal in 0..4 {
            let card = self.deal_card();
            self.player_mut(deal % 2).hand.push(card);
        }
        let big_blind = (small_blind + 1) % 2;
        self.player_mut(small_blind).in_chips = 1;
        self.player_mut(big_blind).in_chips = 2;
        self.current_player = small_blind;
        let [first, second] = &self.players;
        self.round = LimitRound::with_blinds([first.in_chips, second.in_chips]);
    }

    /// Deal the next source card.
    fn deal_card(&mut self) -> Card {
        self.deck
            .pop_front()
            .expect("one full deck covers a Hold'em hand")
    }

    /// Return RLCard's exact legal action mask.
    fn legal_mask(&self) -> [bool; ACTION_COUNT] {
        let maximum = self.round.raised.into_iter().max().unwrap_or_default();
        let matched = self.round.raised_for(self.current_player) == maximum;
        [!matched, self.round.have_raised < 4, true, matched]
    }

    /// Build one player's 72-bit source observation.
    fn observe(&self, player: usize) -> LimitObservation {
        let mut values = [0_u8; 72];
        for card in self
            .player(player)
            .hand
            .iter()
            .chain(self.public_cards.iter())
        {
            if let Some(value) = values.get_mut(card.index()) {
                *value = 1;
            }
        }
        for (round, raises) in self.raise_history.into_iter().enumerate() {
            if let Some(value) = values.get_mut(52 + round * 5 + usize::from(raises)) {
                *value = 1;
            }
        }
        LimitObservation {
            values,
            action_mask: if self.rewards.is_none() && player == self.current_player {
                self.legal_mask()
            } else {
                [false; ACTION_COUNT]
            },
        }
    }

    /// Apply one legal selected-player action.
    fn play(&mut self, player: usize, action: LimitAction) -> Result<Option<[i16; 2]>, LimitError> {
        if self.rewards.is_some() {
            return Err(LimitError::HandFinished);
        }
        if player != self.current_player {
            return Err(LimitError::WrongPlayer {
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
            return Err(LimitError::IllegalAction(action));
        }

        let maximum = self.round.raised.into_iter().max().unwrap_or_default();
        match action {
            LimitAction::Call => {
                let difference = maximum - self.round.raised_for(player);
                self.round.set_raised(player, maximum);
                self.player_mut(player).in_chips += difference;
                self.round.not_raise += 1;
            }
            LimitAction::Raise => {
                let difference = maximum - self.round.raised_for(player) + self.round.raise_amount;
                self.round
                    .set_raised(player, maximum + self.round.raise_amount);
                self.player_mut(player).in_chips += difference;
                self.round.have_raised += 1;
                self.round.not_raise = 1;
            }
            LimitAction::Fold => self.player_mut(player).folded = true,
            LimitAction::Check => self.round.not_raise += 1,
        }
        self.current_player = (self.current_player + 1) % 2;
        while self.player(self.current_player).folded {
            self.current_player = (self.current_player + 1) % 2;
        }
        if let Some(raises) = self.raise_history.get_mut(usize::from(self.round_counter)) {
            *raises = self.round.have_raised;
        }
        if self
            .players
            .iter()
            .filter(|candidate| !candidate.folded)
            .count()
            == 1
        {
            self.rewards = Some(self.calculate_rewards());
            return Ok(self.rewards);
        }
        if self.round.is_over() {
            self.advance_street();
        }
        if self.round_counter >= 4 {
            self.rewards = Some(self.calculate_rewards());
        }
        Ok(self.rewards)
    }

    /// Reveal the next community cards and reset fixed-limit counters.
    fn advance_street(&mut self) {
        let deal_count = if self.round_counter == 0 { 3 } else { 1 };
        if self.round_counter <= 2 {
            for _ in 0..deal_count {
                let card = self.deal_card();
                self.public_cards.push(card);
            }
        }
        self.round_counter += 1;
        let raise_amount = if self.round_counter >= 2 { 4 } else { 2 };
        self.round.start_next(raise_amount);
    }

    /// Allocate the heads-up pot and scale net chips by the two-chip blind.
    fn calculate_rewards(&self) -> [i16; 2] {
        let [first, second] = &self.players;
        let contributions = [first.in_chips, second.in_chips];
        let [first_contribution, second_contribution] = contributions;
        let matched = first_contribution.min(second_contribution);
        let mut allocated = [first_contribution - matched, second_contribution - matched];
        let winners = if first.folded {
            [false, true]
        } else if second.folded {
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
            (first_allocated as i16 - first_contribution as i16) / 2,
            (second_allocated as i16 - second_contribution as i16) / 2,
        ]
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

/// Comparable five-card poker value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct HandValue([u8; 6]);

/// Compare two complete Hold'em hands.
fn compare_hands(first: &[Card], second: &[Card], public: &[Card]) -> [bool; 2] {
    match best_hand(first, public).cmp(&best_hand(second, public)) {
        Ordering::Greater => [true, false],
        Ordering::Less => [false, true],
        Ordering::Equal => [true, true],
    }
}

/// Choose the strongest five-card subset.
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

/// Rank one five-card subset with exact kickers.
fn five_card_value(cards: [Card; 5]) -> HandValue {
    let mut ranks: Vec<u8> = cards.iter().map(|card| card.rank()).collect();
    ranks.sort_unstable_by(|left, right| right.cmp(left));
    let first_suit = cards.first().map_or(0, |card| card.suit());
    let flush = cards.iter().all(|card| card.suit() == first_suit);
    let mut unique = ranks.clone();
    unique.dedup();
    let straight = if unique == [14, 5, 4, 3, 2] {
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
    let first = counts.first().copied().unwrap_or_default();
    let second = counts.get(1).copied().unwrap_or_default();
    let third = counts.get(2).copied().unwrap_or_default();
    let fourth = counts.get(3).copied().unwrap_or_default();
    let ranks = match ranks.as_slice() {
        [first, second, third, fourth, fifth] => [*first, *second, *third, *fourth, *fifth],
        _ => [0; 5],
    };
    let [first_rank, second_rank, third_rank, fourth_rank, fifth_rank] = ranks;
    let straight_high = straight.unwrap_or_default();
    if straight.is_some() && flush {
        HandValue([8, straight_high, 0, 0, 0, 0])
    } else if first.1 == 4 {
        HandValue([7, first.0, second.0, 0, 0, 0])
    } else if first.1 == 3 && second.1 == 2 {
        HandValue([6, first.0, second.0, 0, 0, 0])
    } else if flush {
        HandValue([
            5,
            first_rank,
            second_rank,
            third_rank,
            fourth_rank,
            fifth_rank,
        ])
    } else if straight.is_some() {
        HandValue([4, straight_high, 0, 0, 0, 0])
    } else if first.1 == 3 {
        HandValue([3, first.0, second.0, third.0, 0, 0])
    } else if first.1 == 2 && second.1 == 2 {
        HandValue([2, first.0, second.0, third.0, 0, 0])
    } else if first.1 == 2 {
        HandValue([1, first.0, second.0, third.0, fourth.0, 0])
    } else {
        HandValue([
            0,
            first_rank,
            second_rank,
            third_rank,
            fourth_rank,
            fifth_rank,
        ])
    }
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
struct LimitTrainingEnv {
    /// Complete raw game.
    game: LimitGame,
    /// Opponent action stream reset from the environment seed.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for LimitTrainingEnv {
    fn default() -> Self {
        let mut game = LimitGame::empty();
        game.reset(0);
        Self {
            game,
            rng: SplitMix64::new(0),
            control: ControlMode::SharedPolicy,
        }
    }
}

/// Per-transition multi-agent evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LimitInfo {
    /// Exact observations in player order.
    observations: [LimitObservation; 2],
    /// Source rewards in player order.
    rewards: [i16; 2],
}

impl Default for LimitInfo {
    fn default() -> Self {
        let observation = LimitObservation {
            values: [0; 72],
            action_mask: [false; ACTION_COUNT],
        };
        Self {
            observations: [observation.clone(), observation],
            rewards: [0; 2],
        }
    }
}

impl LimitTrainingEnv {
    /// Return current observations and one reward pair.
    fn info(&self, rewards: [i16; 2]) -> LimitInfo {
        LimitInfo {
            observations: [self.game.observe(0), self.game.observe(1)],
            rewards,
        }
    }

    /// Flatten the source observation and append its legal-action mask.
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

    /// Select one uniformly random legal action for player zero.
    fn random_player_zero_action(&mut self) -> LimitAction {
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
        LimitAction::from_index(action)
    }
}

impl Env for LimitTrainingEnv {
    type Observation = Vec<f32>;
    type Action = LimitAction;
    type Info = LimitInfo;

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
            Ok(Some(rewards)) => (rewards, EpisodeStatus::Terminated),
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

impl DiscreteDqnExample for LimitTrainingEnv {
    const ENV_NAME: &'static str = "texas-holdem-v4";
    const GYMNASIUM_ID: &'static str = "classic/texas_holdem-v4";
    const OBSERVATION_DIM: usize = 72;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const ZERO_SUM_COMPARISON: bool = true;
    const BOOTSTRAP_MULTIPLIER: f32 = -1.0;
    const MIN_TRAIN_STEPS: usize = 100_000;
    const SOLVED_MEAN_REWARD: f64 = 0.0;
    const GIF_PATH: &'static str = "docs/images/classic-texas-holdem-v4.gif";

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
        observation.iter().take(72).copied().collect()
    }

    fn action_mask(observation: &[f32]) -> Vec<bool> {
        observation
            .iter()
            .skip(72)
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
    run_discrete_workflow::<LimitTrainingEnv>()
}

/// RLCard-style card-table renderer for shuffled fixed-limit hands.
#[cfg(feature = "render")]
mod render {
    use super::{
        Card, DiscreteDqnExample, Error, LimitAction, LimitGame, LimitTrainingEnv, Path,
        SplitMix64, ACTION_COUNT,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;

    /// Source documentation GIF width for two players.
    const WIDTH: f32 = 650.0;
    /// Integer viewport width used by window and encoder APIs.
    const WIDTH_PIXELS: u32 = 650;
    /// Source renderer default height.
    const HEIGHT: f32 = 1_000.0;
    /// Integer viewport height used by window and encoder APIs.
    const HEIGHT_PIXELS: u32 = 1_000;
    /// `screen_height * 2 / 10` in the source renderer.
    const TILE_SIZE: f32 = 200.0;
    /// Source card width `tile_size * 142 / 197`.
    const CARD_WIDTH: f32 = TILE_SIZE * 142.0 / 197.0;

    /// Configure the source green table background.
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
        game: LimitGame,
        /// Seeded player-one action stream.
        rng: SplitMix64,
        /// Seed for the next shuffled hand.
        next_seed: u64,
    }

    /// One private-card or community-card sprite slot.
    #[derive(Component)]
    struct CardSprite {
        /// Player hand or shared board.
        group: CardGroup,
        /// Index inside the group.
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

    /// One white committed-chip sprite.
    #[derive(Component)]
    struct ChipSprite {
        /// Owning player.
        player: usize,
        /// Index within the stack.
        index: usize,
    }

    /// Dynamic committed-chip count.
    #[derive(Component)]
    struct ChipLabel(usize);

    /// Source one-Hz playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run interactive playback or a finite source-sized capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = LimitTrainingEnv::dqn_config(LimitTrainingEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(checkpoint, 72, ACTION_COUNT, &config.hidden_sizes)?;
        let mut game = LimitGame::empty();
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
                            title: "Texas Hold'em".into(),
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
        commands.spawn((Camera2d, Name::new("Texas Hold'em Camera")));
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
            for index in 0..20 {
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

    /// Advance one selected raw AEC player at one FPS.
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
        if visual.game.rewards.is_some() {
            visual.game.reset(visual.next_seed);
            visual.next_seed = visual.next_seed.wrapping_add(1);
            return;
        }
        let player = visual.game.current_player;
        let action = if player == 1 {
            let observation = LimitTrainingEnv {
                game: visual.game.clone(),
                rng: visual.rng,
                control: super::ControlMode::PolicyAsPlayerOneVsRandomPlayerZero,
            }
            .observation();
            match LimitTrainingEnv::deployment_action(&visual.policy, &observation) {
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
                Some(index) => LimitAction::from_index(index),
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

    /// Apply PettingZoo's two-player card and chip placement formulas.
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
            "classic-texas-holdem-v4",
            WIDTH_PIXELS,
            HEIGHT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The small blind acts first and may call, raise, or fold.
    #[test]
    fn reset_posts_blinds_and_exposes_source_legal_actions() {
        let mut game = LimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);

        assert_eq!(game.players[0].in_chips, 1);
        assert_eq!(game.players[1].in_chips, 2);
        assert_eq!(game.current_player, 0);
        assert_eq!(game.legal_mask(), [true, true, true, false]);
    }

    /// A preflop raise matches two chips and adds the fixed two-chip raise.
    #[test]
    fn raise_uses_fixed_limit_amount_and_rotates_player() {
        let mut game = LimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);

        assert_eq!(game.play(0, LimitAction::Raise), Ok(None));

        assert_eq!(game.players[0].in_chips, 4);
        assert_eq!(game.round.raised, [4, 2]);
        assert_eq!(game.current_player, 1);
        assert_eq!(game.legal_mask(), [true, true, true, false]);
    }

    /// Fold settlement returns the unmatched raise before scaling by the blind.
    #[test]
    fn opponent_fold_produces_exact_big_blind_scaled_rewards() {
        let mut game = LimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);
        assert_eq!(game.play(0, LimitAction::Raise), Ok(None));

        assert_eq!(game.play(1, LimitAction::Fold), Ok(Some([1, -1])));
        assert_eq!(game.rewards, Some([1, -1]));
    }

    /// The 72-bit observation combines visible cards and one-hot raise counts.
    #[test]
    fn observation_matches_source_card_and_raise_planes() {
        let mut game = LimitGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);
        let initial = game.observe(0);

        assert_eq!(initial.values[0], 1);
        assert_eq!(initial.values[13], 1);
        assert_eq!(initial.values[52], 1);
        assert_eq!(
            initial.values.iter().copied().map(u16::from).sum::<u16>(),
            6
        );

        assert_eq!(game.play(0, LimitAction::Raise), Ok(None));
        let after_raise = game.observe(1);
        assert_eq!(after_raise.values[52], 0);
        assert_eq!(after_raise.values[53], 1);
    }

    /// The local evaluator orders a straight above three of a kind.
    #[test]
    fn showdown_evaluator_orders_complete_hands() {
        let public = [
            Card::new(1),
            Card::new(15),
            Card::new(29),
            Card::new(43),
            Card::new(8),
        ];
        let straight = [Card::new(4), Card::new(18)];
        let trips = [Card::new(27), Card::new(40)];

        assert_eq!(compare_hands(&straight, &trips, &public), [true, false]);
    }

    /// The training adapter preserves shuffled deals and complete source masks.
    #[test]
    fn training_adapter_uses_shuffled_hands_and_complete_legal_masks() {
        let mut first = LimitTrainingEnv::default();
        let mut second = LimitTrainingEnv::default();
        let mut different = LimitTrainingEnv::default();

        let first_reset = first.reset(Some(42));
        let second_reset = second.reset(Some(42));
        let different_reset = different.reset(Some(9_001));

        assert_eq!(first_reset.observation, second_reset.observation);
        assert_ne!(first_reset.observation, different_reset.observation);
        assert_eq!(first_reset.observation.len(), 72 + ACTION_COUNT);
        assert!(first_reset.observation[72..]
            .iter()
            .any(|value| *value > 0.5));
        assert_eq!(first_reset.observation, first.observation());

        let mut evaluation = LimitTrainingEnv::default();
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
