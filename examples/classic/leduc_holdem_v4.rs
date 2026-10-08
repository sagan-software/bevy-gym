//!
//! # Leduc Hold'em
//!
//! ```{figure} classic_leduc_holdem.gif
//! :width: 140px
//! :name: leduc_holdem
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/leduc_holdem-v4")`         |
//! |--------------------|--------------------------------------------------|
//! | Actions            |                                                  |
//! | Parallel API       | Yes                                              |
//! | Manual Control     | No                                               |
//! | Agents             | `agents= ['player_0', 'player_1']`               |
//! | Agents             | 2                                                |
//! | Action Shape       | Discrete(4)                                      |
//! | Action Values      | Discrete(4)                                      |
//! | Observation Shape  | (36,)                                            |
//! | Observation Values | [0, 1]                                           |
//!
//!
//! Leduc Hold'em is a variation of Limit Texas Hold'em with fixed number of 2 players, 2 rounds and a deck of six cards (Jack, Queen, and King in 2 suits). At the beginning of the game, each player receives one card and, after betting, one public card is revealed.
//! Another round follows. At the end, the player with the best hand wins and receives a reward (+1) and the loser receives -1. At any time, any player can fold.
//!
//! Our implementation wraps [RLCard](http://rlcard.org/games.html#leduc-hold-em) and you can refer to its documentation for additional details. Please cite their work if you use this game in research.
//!
//! ### Observation Space
//!
//! The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.
//!
//! As described by [RLCard](https://github.com/datamllab/rlcard/blob/master/docs/games#leduc-holdem), the first 3 entries of the main observation space correspond to the player's hand (J, Q, and K) and the next 3 represent the public cards. Indexes 6 to 19 and 20 to 33 encode the number of chips by
//! the current player and the opponent, respectively.
//!
//! |  Index  | Description                                                                  |
//! |:-------:|------------------------------------------------------------------------------|
//! |  0 - 2  | Current Player's Hand<br>_`0`: J, `1`: Q, `2`: K_                            |
//! |  3 - 5  | Community Cards<br>_`3`: J, `4`: Q, `5`: K_                                  |
//! |  6 - 20 | Current Player's Chips<br>_`6`: 0 chips, `7`: 1 chip, ..., `20`: 14 chips_   |
//! | 21 - 35 | Opponent's Chips<br>_`21`: 0 chips, `22`: 1 chip, ..., `35`: 14 chips_       |
//!
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
//! |      Winner       |       Loser       |
//! | :---------------: | :---------------: |
//! | +raised chips / 2 | -raised chips / 2 |
//!
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

/// One of the six source cards: two suits for each rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeducCard {
    /// Rank index: Jack, Queen, or King.
    rank: u8,
    /// Suit index used only to distinguish duplicate ranks.
    suit: u8,
}

impl LeducCard {
    /// Construct one checked rank and suit pair.
    const fn new(rank: u8, suit: u8) -> Self {
        assert!(rank < 3 && suit < 2);
        Self { rank, suit }
    }
}

/// RLCard's four fixed-limit action IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeducAction {
    /// Match the current wager.
    Call,
    /// Match and add the street's fixed amount.
    Raise,
    /// Concede the pot.
    Fold,
    /// Continue without chips when already matched.
    Check,
}

impl LeducAction {
    /// Convert a table column into an action.
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

impl DqnAction for LeducAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }

    fn as_index(self) -> usize {
        self.index()
    }
}

/// One source Leduc player.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeducPlayer {
    /// Single private card.
    hand: LeducCard,
    /// Total chips committed in this game.
    in_chips: u8,
    /// Whether this player folded.
    folded: bool,
}

/// One fixed-limit betting round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LeducRound {
    /// Street-local chip commitments.
    raised: [u8; 2],
    /// Two chips before the public card and four after it.
    raise_amount: u8,
    /// Number of raises made on this street.
    have_raised: u8,
    /// Consecutive non-raise responses.
    not_raise: u8,
}

impl LeducRound {
    /// Initialize the first round from the one- and two-chip blinds.
    const fn with_blinds() -> Self {
        Self {
            raised: [1, 2],
            raise_amount: 2,
            have_raised: 0,
            not_raise: 0,
        }
    }

    /// Start the post-public-card round.
    const fn start_second(&mut self) {
        self.raised = [0; 2];
        self.raise_amount = 4;
        self.have_raised = 0;
        self.not_raise = 0;
    }

    /// Return one player's round-local committed chips.
    const fn raised_for(self, player: usize) -> u8 {
        let [first, second] = self.raised;
        if player == 0 {
            first
        } else {
            second
        }
    }

    /// Replace one player's round-local committed chips.
    const fn set_raised(&mut self, player: usize, chips: u8) {
        let [first, second] = &mut self.raised;
        if player == 0 {
            *first = chips;
        } else {
            *second = chips;
        }
    }
}

/// Exact 36-bit source observation dictionary.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LeducObservation {
    /// Private rank, public rank, and perspective chip counts.
    values: [u8; 36],
    /// Legal actions for the selected player.
    action_mask: [bool; ACTION_COUNT],
}

/// Typed raw-environment rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeducError {
    /// The caller supplied the unselected player.
    WrongPlayer {
        /// Player selected by the betting round.
        expected: u8,
        /// Player supplied by the caller.
        actual: u8,
    },
    /// The action is absent from the current legal mask.
    IllegalAction(LeducAction),
    /// The game has already ended.
    GameFinished,
}

impl fmt::Display for LeducError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPlayer { expected, actual } => {
                write!(
                    formatter,
                    "expected player {expected}, received player {actual}"
                )
            }
            Self::IllegalAction(action) => write!(formatter, "action {action:?} is illegal"),
            Self::GameFinished => formatter.write_str("the Leduc game has finished"),
        }
    }
}

impl Error for LeducError {}

/// Source-compatible two-player Leduc state machine.
#[derive(Debug, Clone)]
struct LeducGame {
    /// Remaining cards in deal order.
    deck: VecDeque<LeducCard>,
    /// Two private hands and chip totals.
    players: [LeducPlayer; 2],
    /// Public card revealed after the first betting round.
    public_card: Option<LeducCard>,
    /// Selected player.
    current_player: usize,
    /// Current fixed-limit round.
    round: LeducRound,
    /// Zero before the public card and one after it.
    round_counter: u8,
    /// Terminal big-blind-scaled payoff.
    rewards: Option<[f64; 2]>,
}

impl Default for LeducGame {
    fn default() -> Self {
        let mut game = Self::empty();
        game.reset(0);
        game
    }
}

impl LeducGame {
    /// Allocate a placeholder before the first deal.
    const fn empty() -> Self {
        Self {
            deck: VecDeque::new(),
            players: [
                LeducPlayer {
                    hand: LeducCard::new(0, 0),
                    in_chips: 1,
                    folded: false,
                },
                LeducPlayer {
                    hand: LeducCard::new(0, 1),
                    in_chips: 2,
                    folded: false,
                },
            ],
            public_card: None,
            current_player: 0,
            round: LeducRound::with_blinds(),
            round_counter: 0,
            rewards: None,
        }
    }

    /// Borrow one of the two players without unchecked indexing.
    const fn player(&self, index: usize) -> &LeducPlayer {
        let [first, second] = &self.players;
        if index == 0 {
            first
        } else {
            second
        }
    }

    /// Mutably borrow one of the two players without unchecked indexing.
    const fn player_mut(&mut self, index: usize) -> &mut LeducPlayer {
        let [first, second] = &mut self.players;
        if index == 0 {
            first
        } else {
            second
        }
    }

    /// Shuffle and deal one deterministic seeded game.
    fn reset(&mut self, seed: u64) {
        let mut deck = source_deck();
        let mut rng = SplitMix64::new(seed);
        for upper in (1..deck.len()).rev() {
            let selected = rng.index(upper + 1);
            deck.swap(upper, selected);
        }
        self.reset_with_deck(deck, rng.index(2));
    }

    /// Deal one caller-arranged game with the chosen small blind.
    fn reset_with_deck(&mut self, cards: Vec<LeducCard>, small_blind: usize) {
        self.deck = cards.into();
        let first = self.deal_card();
        let second = self.deal_card();
        let big_blind = (small_blind + 1) % 2;
        self.players = [
            LeducPlayer {
                hand: first,
                in_chips: u8::from(small_blind == 0) + 2 * u8::from(big_blind == 0),
                folded: false,
            },
            LeducPlayer {
                hand: second,
                in_chips: u8::from(small_blind == 1) + 2 * u8::from(big_blind == 1),
                folded: false,
            },
        ];
        self.public_card = None;
        self.current_player = small_blind;
        let [first_player, second_player] = &self.players;
        self.round = LeducRound {
            raised: [first_player.in_chips, second_player.in_chips],
            ..LeducRound::with_blinds()
        };
        self.round_counter = 0;
        self.rewards = None;
    }

    /// Remove the next card from a six-card deck.
    fn deal_card(&mut self) -> LeducCard {
        self.deck
            .pop_front()
            .expect("three dealt cards fit in the six-card deck")
    }

    /// Return RLCard's exact legal action mask.
    fn legal_mask(&self) -> [bool; ACTION_COUNT] {
        let maximum = self.round.raised.into_iter().max().unwrap_or_default();
        let matched = self.round.raised_for(self.current_player) == maximum;
        [!matched, self.round.have_raised < 2, true, matched]
    }

    /// Build one player's exact 36-bit observation.
    fn observe(&self, player: usize) -> LeducObservation {
        let mut values = [0_u8; 36];
        let own = self.player(player);
        if let Some(value) = values.get_mut(usize::from(own.hand.rank)) {
            *value = 1;
        }
        if let Some(card) = self.public_card {
            if let Some(value) = values.get_mut(3 + usize::from(card.rank)) {
                *value = 1;
            }
        }
        let own_chips = usize::from(own.in_chips);
        let other_chips = usize::from(self.player((player + 1) % 2).in_chips);
        if let Some(value) = values.get_mut(6 + own_chips) {
            *value = 1;
        }
        if let Some(value) = values.get_mut(21 + other_chips) {
            *value = 1;
        }
        LeducObservation {
            values,
            action_mask: if self.rewards.is_none() && player == self.current_player {
                self.legal_mask()
            } else {
                [false; ACTION_COUNT]
            },
        }
    }

    /// Apply one legal selected-player action.
    fn play(&mut self, player: usize, action: LeducAction) -> Result<Option<[f64; 2]>, LeducError> {
        if self.rewards.is_some() {
            return Err(LeducError::GameFinished);
        }
        if player != self.current_player {
            return Err(LeducError::WrongPlayer {
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
            return Err(LeducError::IllegalAction(action));
        }

        let maximum = self.round.raised.into_iter().max().unwrap_or_default();
        match action {
            LeducAction::Call => {
                let difference = maximum - self.round.raised_for(player);
                self.round.set_raised(player, maximum);
                self.player_mut(player).in_chips += difference;
                self.round.not_raise += 1;
            }
            LeducAction::Raise => {
                let raise_amount = self.round.raise_amount;
                let difference = maximum - self.round.raised_for(player) + raise_amount;
                self.round.set_raised(player, maximum + raise_amount);
                self.player_mut(player).in_chips += difference;
                self.round.have_raised += 1;
                self.round.not_raise = 1;
            }
            LeducAction::Fold => self.player_mut(player).folded = true,
            LeducAction::Check => self.round.not_raise += 1,
        }
        self.current_player = (self.current_player + 1) % 2;
        while self.player(self.current_player).folded {
            self.current_player = (self.current_player + 1) % 2;
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
        if self.round.not_raise >= 2 {
            if self.round_counter == 0 {
                self.public_card = Some(self.deal_card());
                self.round_counter = 1;
                self.round.start_second();
            } else {
                self.round_counter = 2;
                self.rewards = Some(self.calculate_rewards());
            }
        }
        Ok(self.rewards)
    }

    /// Allocate the complete pot and divide net chips by the big blind.
    fn calculate_rewards(&self) -> [f64; 2] {
        let [first, second] = &self.players;
        let chips = [first.in_chips, second.in_chips];
        let [first_chips, second_chips] = chips;
        let total = f64::from(first_chips + second_chips);
        let [first_wins, second_wins] = self.winners();
        let winner_count = f64::from(u8::from(first_wins) + u8::from(second_wins));
        let allocation = |wins| if wins { total / winner_count } else { 0.0 };
        [
            (allocation(first_wins) - f64::from(first_chips)) / 2.0,
            (allocation(second_wins) - f64::from(second_chips)) / 2.0,
        ]
    }

    /// Select folds, paired public ranks, or the highest private rank.
    fn winners(&self) -> [bool; 2] {
        let [first, second] = &self.players;
        if first.folded {
            return [false, true];
        }
        if second.folded {
            return [true, false];
        }
        let Some(public_rank) = self.public_card.map(|card| card.rank) else {
            return [true, true];
        };
        let paired = [
            first.hand.rank == public_rank,
            second.hand.rank == public_rank,
        ];
        if paired.into_iter().any(|matches| matches) {
            paired
        } else {
            match first.hand.rank.cmp(&second.hand.rank) {
                std::cmp::Ordering::Greater => [true, false],
                std::cmp::Ordering::Less => [false, true],
                std::cmp::Ordering::Equal => [true, true],
            }
        }
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

    /// Select one index below a nonzero bound.
    fn index(&mut self, upper: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        usize::try_from((value ^ (value >> 31)) % upper as u64).unwrap_or_default()
    }
}

/// Build RLCard's two-suit Jack-through-King deck.
fn source_deck() -> Vec<LeducCard> {
    (0..2)
        .flat_map(|suit| (0..3).map(move |rank| LeducCard::new(rank, suit)))
        .collect()
}

/// Fixed paired-King deck used only by raw rules tests.
#[cfg(test)]
fn deterministic_test_deck() -> Vec<LeducCard> {
    vec![
        LeducCard::new(2, 0),
        LeducCard::new(1, 0),
        LeducCard::new(2, 1),
        LeducCard::new(0, 0),
        LeducCard::new(0, 1),
        LeducCard::new(1, 1),
    ]
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
struct LeducTrainingEnv {
    /// Complete raw Leduc game.
    game: LeducGame,
    /// Opponent action stream reset from the environment seed.
    rng: SplitMix64,
    /// Current policy-control protocol.
    control: ControlMode,
}

impl Default for LeducTrainingEnv {
    fn default() -> Self {
        let mut game = LeducGame::empty();
        game.reset(0);
        Self {
            game,
            rng: SplitMix64::new(0),
            control: ControlMode::SharedPolicy,
        }
    }
}

/// Per-transition multi-agent evidence.
#[derive(Debug, Clone, PartialEq)]
struct LeducInfo {
    /// Exact observations in player order.
    observations: [LeducObservation; 2],
    /// Source rewards in player order.
    rewards: [f64; 2],
}

impl Default for LeducInfo {
    fn default() -> Self {
        let observation = LeducObservation {
            values: [0; 36],
            action_mask: [false; ACTION_COUNT],
        };
        Self {
            observations: [observation.clone(), observation],
            rewards: [0.0; 2],
        }
    }
}

impl LeducTrainingEnv {
    /// Build current observations and payoff evidence.
    fn info(&self, rewards: [f64; 2]) -> LeducInfo {
        LeducInfo {
            observations: [self.game.observe(0), self.game.observe(1)],
            rewards,
        }
    }

    /// Flatten the source observation and append its exact action mask.
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
    fn random_player_zero_action(&mut self) -> LeducAction {
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
        LeducAction::from_index(action)
    }
}

impl Env for LeducTrainingEnv {
    type Observation = Vec<f32>;
    type Action = LeducAction;
    type Info = LeducInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        let seed = seed.unwrap_or_default();
        self.game.reset(seed);
        self.rng = SplitMix64::new(seed ^ 0xa5a5_a5a5_a5a5_a5a5);
        Reset {
            observation: self.observation(),
            info: self.info([0.0; 2]),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let actor = self.game.current_player;
        let result = self.game.play(actor, action);
        let (rewards, status) = match result {
            Err(_error) => ([-1.0, 0.0], EpisodeStatus::Terminated),
            Ok(Some(rewards)) => (rewards, EpisodeStatus::Terminated),
            Ok(None) => ([0.0; 2], EpisodeStatus::Continuing),
        };
        let [player_zero_reward, player_one_reward] = rewards;
        Step {
            observation: self.observation(),
            reward: match self.control {
                ControlMode::SharedPolicy if actor == 0 => player_zero_reward,
                ControlMode::SharedPolicy | ControlMode::PolicyAsPlayerOneVsRandomPlayerZero => {
                    player_one_reward
                }
            },
            status,
            info: self.info(rewards),
        }
    }
}

impl DiscreteDqnExample for LeducTrainingEnv {
    const ENV_NAME: &'static str = "leduc-holdem-v4";
    const GYMNASIUM_ID: &'static str = "classic/leduc_holdem-v4";
    const OBSERVATION_DIM: usize = 36;
    const ACTION_COUNT: usize = ACTION_COUNT;
    const DEFAULT_TRAIN_STEPS: usize = 250_000;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const ZERO_SUM_COMPARISON: bool = true;
    const BOOTSTRAP_MULTIPLIER: f32 = -1.0;
    const MIN_TRAIN_STEPS: usize = 50_000;
    const SOLVED_MEAN_REWARD: f64 = 1.0;
    const GIF_PATH: &'static str = "docs/images/classic-leduc-holdem-v4.gif";

    fn dqn_config(learning_rate: f64) -> DqnConfig {
        DqnConfig {
            hidden_sizes: vec![128, 128],
            gamma: 1.0,
            learning_rate,
            replay_capacity: 100_000,
            min_replay_size: 1_024,
            batch_size: 128,
            target_update_interval: 1_000,
            epsilon_decay_steps: 125_000,
            ..DqnConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        observation.iter().take(36).copied().collect()
    }

    fn action_mask(observation: &[f32]) -> Vec<bool> {
        observation
            .iter()
            .skip(36)
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
    run_discrete_workflow::<LeducTrainingEnv>()
}

/// Source-sized Leduc card-table renderer.
#[cfg(feature = "render")]
mod render {
    use super::{
        DiscreteDqnExample, Error, LeducAction, LeducCard, LeducGame, LeducTrainingEnv, Path,
        SplitMix64, ACTION_COUNT,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::DqnPolicy;

    /// Source renderer width for two players.
    const WIDTH: f32 = 550.0;
    /// Integer viewport width used by window and encoder APIs.
    const WIDTH_PIXELS: u32 = 550;
    /// Source renderer height.
    const HEIGHT: f32 = 1_000.0;
    /// Integer viewport height used by window and encoder APIs.
    const HEIGHT_PIXELS: u32 = 1_000;
    /// Source card height.
    const TILE_SIZE: f32 = 200.0;
    /// Source card width.
    const CARD_WIDTH: f32 = TILE_SIZE * 142.0 / 197.0;

    /// Configure PettingZoo's green table.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb_u8(7, 99, 36)));
        }
    }

    /// Loaded checkpoint and visible raw game.
    #[derive(Resource)]
    struct VisualGame {
        /// Greedy player-zero DQN policy.
        policy: DqnPolicy,
        /// Exact source game.
        game: LeducGame,
        /// Seeded player-one action stream.
        rng: SplitMix64,
        /// Seed for the next shuffled hand.
        next_seed: u64,
    }

    /// Private or public card sprite.
    #[derive(Component)]
    struct CardSprite(CardGroup);

    /// Card ownership determines source placement.
    #[derive(Debug, Clone, Copy)]
    enum CardGroup {
        /// Player zero's card.
        PlayerZero,
        /// Player one's card.
        PlayerOne,
        /// Shared public card.
        Public,
    }

    /// One white chip in a player's stack.
    #[derive(Component)]
    struct ChipSprite {
        /// Owning player.
        player: usize,
        /// Stack index below fourteen.
        index: usize,
    }

    /// Dynamic committed-chip text.
    #[derive(Component)]
    struct ChipLabel(usize);

    /// Source one-Hz playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run interactive playback or a finite capture.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = LeducTrainingEnv::dqn_config(LeducTrainingEnv::DEFAULT_LEARNING_RATE);
        let policy = DqnPolicy::load(checkpoint, 36, ACTION_COUNT, &config.hidden_sizes)?;
        let mut game = LeducGame::empty();
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
                            title: "Leduc Hold'em".into(),
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

    /// Spawn fixed labels and reusable sprite slots.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("Leduc Hold'em Camera")));
        let font = assets.load("font/Minecraft.ttf");
        for player in 0..2 {
            let label_y = if player == 0 { 30.0 } else { 980.0 };
            commands.spawn((
                Text2d::new(format!("Player {}", player + 1)),
                TextFont {
                    font: font.clone(),
                    font_size: 36.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Anchor::CENTER,
                Transform::from_xyz(source_x(195.0), source_y(label_y), 2.0),
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
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                CardSprite(if player == 0 {
                    CardGroup::PlayerZero
                } else {
                    CardGroup::PlayerOne
                }),
            ));
            for index in 0..14 {
                commands.spawn((
                    Sprite::default(),
                    Transform::default(),
                    Visibility::Hidden,
                    ChipSprite { player, index },
                ));
            }
        }
        commands.spawn((
            Sprite::default(),
            Transform::default(),
            Visibility::Hidden,
            CardSprite(CardGroup::Public),
        ));
    }

    /// Advance one raw AEC action each second.
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
            let observation = LeducTrainingEnv {
                game: visual.game.clone(),
                rng: visual.rng,
                control: super::ControlMode::PolicyAsPlayerOneVsRandomPlayerZero,
            }
            .observation();
            match LeducTrainingEnv::deployment_action(&visual.policy, &observation) {
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
                Some(index) => LeducAction::from_index(index),
                None => return,
            }
        };
        if visual.game.play(player, action).is_err() {
            visual.game.reset(visual.next_seed);
            visual.next_seed = visual.next_seed.wrapping_add(1);
        }
    }

    /// Synchronize all visible source entities.
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

    /// Apply the exact two-player source placement formulas.
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
            let (card, x, y) = match slot.0 {
                CardGroup::PlayerZero => (Some(visual.game.player(0).hand), 121.0, 50.0),
                CardGroup::PlayerOne => (Some(visual.game.player(1).hand), 121.0, 750.0),
                CardGroup::Public => (visual.game.public_card, 221.0, 400.0),
            };
            let Some(card) = card else {
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
                top_left_to_world(325.0, y, 100.0, TILE_SIZE * 16.0 / 45.0, 1.0);
            *visibility = Visibility::Visible;
        }
        for (label, mut text, mut transform) in labels.iter_mut() {
            let total = visual.game.player(label.0).in_chips;
            **text = total.to_string();
            let base_y = if label.0 == 0 { 150.0 } else { 850.0 };
            let y = base_y - (usize::from(total) + 1) as f32 * TILE_SIZE / 15.0;
            transform.translation = Vec3::new(source_x(375.0), source_y(y), 2.0);
        }
    }

    /// Convert one source card into its image stem.
    fn card_code(card: LeducCard) -> String {
        let suit = if card.suit == 0 { 'S' } else { 'H' };
        let rank = ['J', 'Q', 'K']
            .get(usize::from(card.rank))
            .copied()
            .unwrap_or('J');
        format!("{suit}{rank}")
    }

    /// Convert a source top-left rectangle into Bevy world coordinates.
    fn top_left_to_world(x: f32, y: f32, width: f32, height: f32, z: f32) -> Vec3 {
        Vec3::new(
            x + width / 2.0 - WIDTH / 2.0,
            HEIGHT / 2.0 - y - height / 2.0,
            z,
        )
    }

    /// Convert a source x coordinate into centered Bevy space.
    const fn source_x(x: f32) -> f32 {
        x - WIDTH / 2.0
    }

    /// Convert a source y coordinate into centered Bevy space.
    const fn source_y(y: f32) -> f32 {
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

    /// Encode the source-sized demonstration.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-leduc-holdem-v4",
            WIDTH_PIXELS,
            HEIGHT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The small blind acts first with the same four-action mask as limit poker.
    #[test]
    fn reset_posts_blinds_and_exposes_legal_actions() {
        let mut game = LeducGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);

        assert_eq!([game.players[0].in_chips, game.players[1].in_chips], [1, 2]);
        assert_eq!(game.current_player, 0);
        assert_eq!(game.legal_mask(), [true, true, true, false]);
    }

    /// Two raises cap the street and a call reveals exactly one public card.
    #[test]
    fn first_round_caps_raises_and_reveals_public_card() {
        let mut game = LeducGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);

        assert_eq!(game.play(0, LeducAction::Raise), Ok(None));
        assert_eq!(game.play(1, LeducAction::Raise), Ok(None));
        assert_eq!(game.legal_mask(), [true, false, true, false]);
        assert_eq!(game.play(0, LeducAction::Call), Ok(None));

        assert_eq!(game.public_card, Some(LeducCard::new(2, 1)));
        assert_eq!(game.round_counter, 1);
        assert_eq!(game.round.raise_amount, 4);
    }

    /// Pairing the public King wins the fully raised 28-chip pot.
    #[test]
    fn maximum_showdown_reward_is_seven_big_blinds() {
        let mut game = LeducGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);
        assert_eq!(game.play(0, LeducAction::Raise), Ok(None));
        assert_eq!(game.play(1, LeducAction::Raise), Ok(None));
        assert_eq!(game.play(0, LeducAction::Call), Ok(None));
        assert_eq!(game.play(1, LeducAction::Raise), Ok(None));
        assert_eq!(game.play(0, LeducAction::Raise), Ok(None));

        assert_eq!(game.play(1, LeducAction::Call), Ok(Some([7.0, -7.0])));
        assert_eq!(
            [game.players[0].in_chips, game.players[1].in_chips],
            [14, 14]
        );
    }

    /// Observation chips use the selected player's perspective.
    #[test]
    fn observation_matches_source_rank_and_chip_planes() {
        let mut game = LeducGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);
        let player_zero = game.observe(0);
        let player_one = game.observe(1);

        assert_eq!(player_zero.values[2], 1);
        assert_eq!(player_zero.values[7], 1);
        assert_eq!(player_zero.values[23], 1);
        assert_eq!(player_one.values[1], 1);
        assert_eq!(player_one.values[8], 1);
        assert_eq!(player_one.values[22], 1);
        assert_eq!(player_one.action_mask, [false; ACTION_COUNT]);
    }

    /// A fold pays the surviving player the matched half-pot in blind units.
    #[test]
    fn fold_settlement_uses_source_chip_payoff() {
        let mut game = LeducGame::empty();
        game.reset_with_deck(deterministic_test_deck(), 0);

        assert_eq!(game.play(0, LeducAction::Fold), Ok(Some([-0.5, 0.5])));
    }

    /// The training adapter preserves shuffled deals and complete source masks.
    #[test]
    fn training_adapter_uses_shuffled_hands_and_complete_legal_masks() {
        let mut first = LeducTrainingEnv::default();
        let mut second = LeducTrainingEnv::default();

        let first_reset = first.reset(Some(42));
        let second_reset = second.reset(Some(42));

        assert_eq!(first_reset.observation, second_reset.observation);
        assert_eq!(first_reset.observation.len(), 36 + ACTION_COUNT);
        assert!(first_reset.observation[36..]
            .iter()
            .any(|value| *value > 0.5));
        assert_eq!(first_reset.observation, first.observation());

        let mut observed_hands = Vec::new();
        for seed in 0..16 {
            first.reset(Some(seed));
            let hand = first.game.players[0].hand;
            if !observed_hands.contains(&hand) {
                observed_hands.push(hand);
            }
        }
        assert!(observed_hands.len() > 1);

        let mut evaluation = LeducTrainingEnv::default();
        evaluation.prepare_evaluation();
        evaluation.reset(Some(42));
        assert_eq!(evaluation.observation(), {
            let observation = evaluation.game.observe(1);
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
                .collect::<Vec<_>>()
        });
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
        assert_eq!(step.reward, step.info.rewards[1]);
    }
}
