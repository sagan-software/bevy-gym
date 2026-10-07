//! Blackjack is a card game where the goal is to beat the dealer by obtaining cards
//! that sum to closer to 21 (without going over 21) than the dealers cards.
//!
//! ## Description
//! The game starts with the dealer having one face up and one face down card,
//! while the player has two face up cards. All cards are drawn from an infinite deck
//! (i.e. with replacement).
//!
//! The card values are:
//! - Face cards (Jack, Queen, King) have a point value of 10.
//! - Aces can either count as 11 (called a 'usable ace') or 1.
//! - Numerical cards (2-10) have a value equal to their number.
//!
//! The player has the sum of cards held. The player can request
//! additional cards (hit) until they decide to stop (stick) or exceed 21 (bust,
//! immediate loss).
//!
//! After the player sticks, the dealer reveals their facedown card, and draws cards
//! until their sum is 17 or greater. If the dealer goes bust, the player wins.
//!
//! If neither the player nor the dealer busts, the outcome (win, lose, draw) is
//! decided by whose sum is closer to 21.
//!
//! This environment corresponds to the version of the blackjack problem
//! described in Example 5.1 in Reinforcement Learning: An Introduction
//! by Sutton and Barto [<a href="#blackjack_ref">1</a>].
//!
//! ## Action Space
//! The action shape is `(1,)` in the range `{0, 1}` indicating
//! whether to stick or hit.
//!
//! - 0: Stick
//! - 1: Hit
//!
//! ## Observation Space
//! The observation consists of a 3-tuple containing: the player's current sum,
//! the value of the dealer's one showing card (1-10 where 1 is ace),
//! and whether the player holds a usable ace (0 or 1).
//!
//! The observation is returned as `(int(), int(), int())`.
//!
//! ## Starting State
//! The starting state is initialised with the following values.
//!
//! | Observation               | Values         |
//! |---------------------------|----------------|
//! | Player current sum        |  4, 5, ..., 21 |
//! | Dealer showing card value |  1, 2, ..., 10 |
//! | Usable Ace                |  0, 1          |
//!
//! ## Rewards
//! - win game: +1
//! - lose game: -1
//! - draw game: 0
//! - win game with natural blackjack:
//!   +1.5 (if <a href="#nat">natural</a> is True)
//!   +1 (if <a href="#nat">natural</a> is False)
//!
//! ## Episode End
//! The episode ends if the following happens:
//!
//! - Termination:
//! 1. The player hits and the sum of hand exceeds 21.
//! 2. The player sticks.
//!
//! An ace will always be counted as usable (11) unless it busts the player.
//!
//! ## Information
//!
//! No additional information is returned.
//!
//! ## Arguments
//!
//! ```python
//! import gymnasium as gym
//! gym.make('Blackjack-v1', natural=False, sab=True)
//! ```
//!
//! <a id="nat"></a>`natural=False`: Whether to give an additional reward for
//! starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).
//!
//! <a id="sab"></a>`sab=True`: Whether to follow the exact rules outlined in the book by
//! Sutton and Barto. If `sab` is `True`, the keyword argument `natural` will be ignored.
//! If the player achieves a natural blackjack and the dealer does not, the player
//! will win (i.e. get a reward of +1). The reverse rule does not apply.
//! If both the player and the dealer get a natural, it will be a draw (i.e. reward 0).
//!
//! ## References
//! <a id="blackjack_ref"></a>[1] R. Sutton and A. Barto, “Reinforcement Learning:
//! An Introduction” 2020. [Online]. Available: [http://www.incompleteideas.net/book/RLbook2020.pdf](http://www.incompleteideas.net/book/RLbook2020.pdf)
//!
//! ## Version History
//! * v1: Fix the natural handling in Blackjack
//! * v0: Initial version release

use shakmaty as _;
use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::{Env, EpisodeStatus, Reset, Step};

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

/// Player decision in Gymnasium action order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlackjackAction {
    /// End the player's turn and resolve the dealer.
    Stick,
    /// Draw one additional player card.
    Hit,
}

impl BlackjackAction {
    /// Convert a table column into the closed action vocabulary.
    const fn from_index(index: usize) -> Self {
        if index.is_multiple_of(2) {
            Self::Stick
        } else {
            Self::Hit
        }
    }
}

impl IndexedAction for BlackjackAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }
}

/// Four suits used to choose the visible dealer card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Suit {
    /// Clubs.
    Clubs,
    /// Diamonds.
    Diamonds,
    /// Hearts.
    Hearts,
    /// Spades.
    Spades,
}

/// Visible rank used by Gymnasium's card sprite names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rank {
    /// Ace.
    Ace,
    /// Numeric rank two through nine.
    Number(u8),
    /// Jack.
    Jack,
    /// Queen.
    Queen,
    /// King.
    King,
}

/// One visible dealer card face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CardFace {
    /// Card suit.
    suit: Suit,
    /// Card rank.
    rank: Rank,
}

impl CardFace {
    /// Return the exact Gymnasium sprite path for this visible face.
    #[cfg(feature = "render")]
    fn asset_path(self) -> String {
        let suit = match self.suit {
            Suit::Clubs => "C",
            Suit::Diamonds => "D",
            Suit::Hearts => "H",
            Suit::Spades => "S",
        };
        let rank = match self.rank {
            Rank::Ace => "A".to_owned(),
            Rank::Number(value) => value.to_string(),
            Rank::Jack => "J".to_owned(),
            Rank::Queen => "Q".to_owned(),
            Rank::King => "K".to_owned(),
        };
        format!("img/{suit}{rank}.png")
    }
}

/// Gymnasium Blackjack-v1 with registered defaults `natural=false` and `sab=true`.
#[derive(Debug, Clone)]
struct Blackjack {
    /// Player card values.
    player: Vec<u8>,
    /// Dealer card values.
    dealer: Vec<u8>,
    /// Visible dealer top-card art choice.
    dealer_face: CardFace,
    /// Deterministic environment-owned card generator.
    rng: SplitMix64,
}

impl Default for Blackjack {
    fn default() -> Self {
        Self {
            player: Vec::new(),
            dealer: Vec::new(),
            dealer_face: CardFace {
                suit: Suit::Clubs,
                rank: Rank::Ace,
            },
            rng: SplitMix64::new(0),
        }
    }
}

impl Blackjack {
    /// Draw one value from Gymnasium's infinite 13-card deck.
    fn draw_card(&mut self) -> u8 {
        const DECK: [u8; 13] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];
        DECK.get(self.rng.index(DECK.len())).copied().unwrap_or(10)
    }

    /// Encode `(player_sum, dealer_card, usable_ace)` into 704 table states.
    fn observation(&self) -> usize {
        let player_sum = usize::from(hand_sum(&self.player).min(31));
        let dealer_card = usize::from(self.dealer.first().copied().unwrap_or(0).min(10));
        let usable_ace = usize::from(has_usable_ace(&self.player));
        (player_sum * 11 + dealer_card) * 2 + usable_ace
    }

    /// Choose visual suit and face rank without changing game card values.
    fn choose_dealer_face(&mut self) -> CardFace {
        let value = self.dealer.first().copied().unwrap_or(1);
        let suit = match self.rng.index(4) {
            0 => Suit::Clubs,
            1 => Suit::Diamonds,
            2 => Suit::Hearts,
            _ => Suit::Spades,
        };
        let rank = match value {
            1 => Rank::Ace,
            2..=9 => Rank::Number(value),
            _ => match self.rng.index(3) {
                0 => Rank::Jack,
                1 => Rank::Queen,
                _ => Rank::King,
            },
        };
        CardFace { suit, rank }
    }
}

impl Env for Blackjack {
    type Observation = usize;
    type Action = BlackjackAction;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.dealer = vec![self.draw_card(), self.draw_card()];
        self.player = vec![self.draw_card(), self.draw_card()];
        self.dealer_face = self.choose_dealer_face();
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let (reward, status) = match action {
            BlackjackAction::Hit => {
                let card = self.draw_card();
                self.player.push(card);
                if is_bust(&self.player) {
                    (-1.0, EpisodeStatus::Terminated)
                } else {
                    (0.0, EpisodeStatus::Continuing)
                }
            }
            BlackjackAction::Stick => {
                while hand_sum(&self.dealer) < 17 {
                    let card = self.draw_card();
                    self.dealer.push(card);
                }
                let player = score(&self.player);
                let dealer = score(&self.dealer);
                let comparison = f64::from(player > dealer) - f64::from(player < dealer);
                let reward = if is_natural(&self.player) && !is_natural(&self.dealer) {
                    1.0
                } else {
                    comparison
                };
                (reward, EpisodeStatus::Terminated)
            }
        };
        Step {
            observation: self.observation(),
            reward,
            status,
            info: (),
        }
    }
}

/// Return whether an ace can count as eleven without busting.
fn has_usable_ace(hand: &[u8]) -> bool {
    hand.contains(&1) && hand.iter().map(|card| u16::from(*card)).sum::<u16>() + 10 <= 21
}

/// Return the effective hand sum with one usable ace counted as eleven.
fn hand_sum(hand: &[u8]) -> u8 {
    let raw = hand.iter().copied().sum::<u8>();
    if has_usable_ace(hand) {
        raw + 10
    } else {
        raw
    }
}

/// Return whether the effective hand exceeds 21.
fn is_bust(hand: &[u8]) -> bool {
    hand_sum(hand) > 21
}

/// Return whether the two-card hand is a natural blackjack.
fn is_natural(hand: &[u8]) -> bool {
    hand.len() == 2 && hand.contains(&1) && hand.contains(&10)
}

/// Return zero for a bust or the effective hand sum.
fn score(hand: &[u8]) -> u8 {
    if is_bust(hand) {
        0
    } else {
        hand_sum(hand)
    }
}

/// Small deterministic generator used for card and art sampling.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Generator state.
    state: u64,
}

impl SplitMix64 {
    /// Construct a stream from one root seed.
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

impl TabularExample for Blackjack {
    const ENV_NAME: &'static str = "blackjack";
    const GYMNASIUM_ID: &'static str = "Blackjack-v1";
    const STATE_COUNT: usize = 704;
    const ACTION_COUNT: usize = 2;
    const MAX_STEPS: usize = 32;
    const DEFAULT_EPISODES: usize = 100_000;
    const DEFAULT_EVAL_INTERVAL: usize = 10_000;
    const DEFAULT_EVAL_EPISODES: usize = 10_000;
    const DEFAULT_LEARNING_RATE: f64 = 0.01;
    const GAMMA: f64 = 1.0;
    const EPSILON_START: f64 = 0.1;
    const EPSILON_END: f64 = 0.1;
    const EPSILON_DECAY_STEPS_PER_EPISODE: usize = 2;
    const SOLVED_SCORE: f64 = -0.05;
    const GIF_PATH: &'static str = "docs/images/blackjack.gif";

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
    run_tabular_workflow::<Blackjack>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        hand_sum, has_usable_ace, Blackjack, BlackjackAction, CardFace, Env, EpisodeStatus, Error,
        Path, Rank, Suit,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
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
            app.insert_resource(ClearColor(Color::srgb_u8(7, 99, 36)));
        }
    }

    /// Policy, hand, and terminal state used by the rendered demonstration.
    #[derive(Resource)]
    struct VisualBlackjack {
        /// Greedy validation-selected policy.
        policy: TabularQPolicy,
        /// Visible game.
        game: Blackjack,
        /// Current encoded table state.
        state: usize,
        /// Whether the current hand has ended.
        terminal: bool,
        /// Reset seed advanced between hands.
        seed: u64,
    }

    /// Four-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Visible dealer face card.
    #[derive(Component)]
    struct DealerCard;

    /// Dealer label updated with the showing value.
    #[derive(Component)]
    struct DealerLabel;

    /// Player total text.
    #[derive(Component)]
    struct PlayerTotal;

    /// Optional usable-ace label.
    #[derive(Component)]
    struct UsableAceLabel;

    /// Strong handles for every card face Gymnasium can show.
    #[derive(Resource)]
    struct CardImages {
        /// Complete visible `(face, image)` set.
        faces: Vec<(CardFace, Handle<Image>)>,
    }

    impl CardImages {
        /// Load every possible visible face before policy playback begins.
        fn load(assets: &AssetServer) -> Self {
            let suits = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades];
            let ranks = [
                Rank::Ace,
                Rank::Number(2),
                Rank::Number(3),
                Rank::Number(4),
                Rank::Number(5),
                Rank::Number(6),
                Rank::Number(7),
                Rank::Number(8),
                Rank::Number(9),
                Rank::Jack,
                Rank::Queen,
                Rank::King,
            ];
            let mut faces = Vec::with_capacity(suits.len() * ranks.len());
            for suit in suits {
                for rank in ranks {
                    let face = CardFace { suit, rank };
                    faces.push((face, assets.load(face.asset_path())));
                }
            }
            Self { faces }
        }

        /// Return the preloaded image for one face.
        fn get(&self, face: CardFace) -> Handle<Image> {
            self.faces
                .iter()
                .find(|(candidate, _)| *candidate == face)
                .map(|(_, image)| image.clone())
                .unwrap_or_default()
        }
    }

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let trainer = TabularQTrainer::load(checkpoint)?;
        let mut game = Blackjack::default();
        let state = game.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualBlackjack {
            policy: trainer.policy(),
            game,
            state,
            terminal: false,
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
                        title: "bevy-gym Blackjack-v1".into(),
                        resolution: WindowResolution::new(600, 500),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_blackjack_scene);

        if let Some(directory) = capture_dir {
            app.insert_resource(GifCapture::new(directory))
                .add_systems(Update, capture_gif_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_blackjack_scene).chain());
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

    /// Build Gymnasium's card table with exact first-party card art and font.
    fn setup_blackjack_scene(
        mut commands: Commands<'_, '_>,
        assets: Res<'_, AssetServer>,
        visual: Res<'_, VisualBlackjack>,
    ) {
        commands.spawn((Camera2d, Name::new("Blackjack Camera")));
        let font = assets.load("font/Minecraft.ttf");
        let white = TextColor(Color::WHITE);
        let card_images = CardImages::load(&assets);
        let dealer_face = card_images.get(visual.game.dealer_face);

        commands.spawn((
            Text2d::new(format!(
                "Dealer: {}",
                visual.game.dealer.first().copied().unwrap_or(0)
            )),
            TextFont {
                font: font.clone(),
                font_size: 33.0,
                ..default()
            },
            white,
            Anchor::TOP_LEFT,
            Transform::from_xyz(-275.0, 225.0, 2.0),
            DealerLabel,
        ));
        commands.spawn((
            Sprite {
                image: dealer_face,
                custom_size: Some(Vec2::new(119.0, 166.0)),
                ..default()
            },
            Transform::from_xyz(-71.5, 84.0, 1.0),
            DealerCard,
        ));
        commands.spawn((
            Sprite {
                image: assets.load("img/Card.png"),
                custom_size: Some(Vec2::new(119.0, 166.0)),
                ..default()
            },
            Transform::from_xyz(71.5, 84.0, 1.0),
        ));
        commands.spawn((
            Text2d::new("Player"),
            TextFont {
                font: font.clone(),
                font_size: 33.0,
                ..default()
            },
            white,
            Anchor::TOP_LEFT,
            Transform::from_xyz(-275.0, -36.0, 2.0),
        ));
        commands.spawn((
            Text2d::new(hand_sum(&visual.game.player).to_string()),
            TextFont {
                font: font.clone(),
                font_size: 83.0,
                ..default()
            },
            white,
            Anchor::TOP_CENTER,
            Transform::from_xyz(0.0, -94.0, 2.0),
            PlayerTotal,
        ));
        commands.spawn((
            Text2d::new(if has_usable_ace(&visual.game.player) {
                "usable ace"
            } else {
                ""
            }),
            TextFont {
                font,
                font_size: 33.0,
                ..default()
            },
            white,
            Anchor::TOP_CENTER,
            Transform::from_xyz(0.0, -176.0, 2.0),
            UsableAceLabel,
        ));
        commands.insert_resource(card_images);
    }

    /// Advance one policy decision at Gymnasium's four frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualBlackjack>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Advance one policy decision or deal a new hand after termination.
    fn advance_visual(visual: &mut VisualBlackjack) {
        if visual.terminal {
            visual.state = visual.game.reset(Some(visual.seed)).observation;
            visual.seed = visual.seed.wrapping_add(1);
            visual.terminal = false;
            return;
        }
        let Ok(action_index) = visual.policy.greedy_action(visual.state) else {
            return;
        };
        let transition = visual.game.step(BlackjackAction::from_index(action_index));
        visual.state = transition.observation;
        visual.terminal = transition.status == EpisodeStatus::Terminated;
    }

    /// Synchronize card art and labels with the visible hand.
    fn sync_blackjack_scene(
        visual: Res<'_, VisualBlackjack>,
        cards: Res<'_, CardImages>,
        mut dealer_card: Single<'_, '_, &mut Sprite, With<DealerCard>>,
        mut dealer_label: Single<'_, '_, &mut Text2d, (With<DealerLabel>, Without<PlayerTotal>)>,
        mut player_total: Single<
            '_,
            '_,
            &mut Text2d,
            (
                With<PlayerTotal>,
                Without<DealerLabel>,
                Without<UsableAceLabel>,
            ),
        >,
        mut usable_ace: Single<
            '_,
            '_,
            &mut Text2d,
            (
                With<UsableAceLabel>,
                Without<DealerLabel>,
                Without<PlayerTotal>,
            ),
        >,
    ) {
        dealer_card.image = cards.get(visual.game.dealer_face);
        ***dealer_label = format!(
            "Dealer: {}",
            visual.game.dealer.first().copied().unwrap_or(0)
        );
        ***player_total = hand_sum(&visual.game.player).to_string();
        ***usable_ace = if has_usable_ace(&visual.game.player) {
            "usable ace".into()
        } else {
            String::new()
        };
    }

    /// Capture 100 frames at 20 FPS while advancing the policy at 4 Hz.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualBlackjack>,
        cards: Res<'_, CardImages>,
        mut dealer_card: Single<'_, '_, &mut Sprite, With<DealerCard>>,
        mut dealer_label: Single<'_, '_, &mut Text2d, (With<DealerLabel>, Without<PlayerTotal>)>,
        mut player_total: Single<
            '_,
            '_,
            &mut Text2d,
            (
                With<PlayerTotal>,
                Without<DealerLabel>,
                Without<UsableAceLabel>,
            ),
        >,
        mut usable_ace: Single<
            '_,
            '_,
            &mut Text2d,
            (
                With<UsableAceLabel>,
                Without<DealerLabel>,
                Without<PlayerTotal>,
            ),
        >,
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
        dealer_card.image = cards.get(visual.game.dealer_face);
        ***dealer_label = format!(
            "Dealer: {}",
            visual.game.dealer.first().copied().unwrap_or(0)
        );
        ***player_total = hand_sum(&visual.game.player).to_string();
        ***usable_ace = if has_usable_ace(&visual.game.player) {
            "usable ace".into()
        } else {
            String::new()
        };
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "blackjack", 600, 500, |frames| {
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
    fn usable_ace_counts_as_eleven_until_it_would_bust() {
        assert_eq!(hand_sum(&[1, 7]), 18);
        assert!(has_usable_ace(&[1, 7]));
        assert_eq!(hand_sum(&[1, 7, 8]), 16);
        assert!(!has_usable_ace(&[1, 7, 8]));
    }

    #[test]
    fn sticking_on_twenty_beats_a_dealer_seventeen() {
        let mut game = Blackjack {
            player: vec![10, 10],
            dealer: vec![10, 7],
            ..Blackjack::default()
        };

        let step = game.step(BlackjackAction::Stick);

        assert_eq!(step.reward, 1.0);
        assert_eq!(step.status, EpisodeStatus::Terminated);
    }

    #[test]
    fn sab_natural_beats_a_non_natural_twenty_one() {
        let mut game = Blackjack {
            player: vec![1, 10],
            dealer: vec![7, 7, 7],
            ..Blackjack::default()
        };

        let step = game.step(BlackjackAction::Stick);

        assert_eq!(step.reward, 1.0);
        assert_eq!(step.status, EpisodeStatus::Terminated);
    }

    #[test]
    fn seeded_resets_and_hits_are_reproducible() {
        let mut first = Blackjack::default();
        let mut second = Blackjack::default();
        assert_eq!(first.reset(Some(29)), second.reset(Some(29)));

        let first_hit = first.step(BlackjackAction::Hit);
        let second_hit = second.step(BlackjackAction::Hit);

        assert_eq!(first_hit, second_hit);
    }
}
