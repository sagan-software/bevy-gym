//!
//! # Rock Paper Scissors
//!
//! ```{figure} classic_rps.gif
//! :width: 140px
//! :name: rps
//! ```
//!
//! This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.
//!
//! | Creation           | `make("aec", "classic/rps-v2")`         |
//! |--------------------|-----------------------------------------|
//! | Actions            | Discrete                                |
//! | Parallel API       | Yes                                     |
//! | Manual Control     | No                                      |
//! | Agents             | `agents= ['player_0', 'player_1']`      |
//! | Agents             | 2                                       |
//! | Action Shape       | Discrete(3)                             |
//! | Action Values      | Discrete(3)                             |
//! | Observation Shape  | Discrete(4)                             |
//! | Observation Values | Discrete(4)                             |
//!
//!
//! Rock, Paper, Scissors is a 2-player hand game where each player chooses either rock, paper or scissors and reveals their choices simultaneously. If both players make the same choice, then it is a draw. However, if their choices are different, the winner is determined as follows: rock beats
//! scissors, scissors beat paper, and paper beats rock.
//!
//! The game can be expanded to have extra actions by adding new action pairs. Adding the new actions in pairs allows for a more balanced game. This means that the final game will have an odd number of actions and each action wins over exactly half of the other actions while being defeated by the
//! other half. The most common expansion of this game is [Rock, Paper, Scissors, Lizard, Spock](http://www.samkass.com/theories/RPSSL.html), in which only one extra action pair is added.
//!
//! ### Arguments
//!
//! ```python
//! from pettingzoo import make
//!
//! make("aec", "classic/rps-v2", num_actions=3, max_cycles=15)
//! ```
//!
//! `num_actions`:  number of actions applicable in the game. The default value is 3 for the game of Rock, Paper, Scissors. This argument must be an integer greater than 3 and with odd parity. If the value given is 5, the game is expanded to Rock, Paper, Scissors, Lizard, Spock.
//!
//! `max_cycles`:  after max_cycles steps all agents will return done.
//!
//! ### Observation Space
//!
//! #### Rock, Paper, Scissors
//!
//! If 3 actions are required, the game played is the standard Rock, Paper, Scissors. The observation is the last opponent action and its space is a scalar value with 4 possible values. Since both players reveal their choices at the same time, the observation is None until both players have acted.
//! Therefore, 3 represents no action taken yet. Rock is represented with 0, paper with 1 and scissors with 2.
//!
//! | Value  |  Observation |
//! | :----: | :---------:  |
//! | 0      | Rock         |
//! | 1      | Paper        |
//! | 2      | Scissors     |
//! | 3      | None         |
//!
//! #### Expanded Game
//!
//! If the number of actions required in the game is greater than 3, the observation is still the last opponent action and its space is a scalar with 1 + n possible values, where n is the number of actions. The observation will as well be None until both players have acted and the largest possible
//! scalar value for the space, 1 + n, represents no action taken yet. The additional actions are encoded in increasing order starting from the 0 Rock action. If 5 actions are required the game is expanded to Rock, Paper, Scissors, Lizard, Spock. The following table shows an example of an observation
//! space with 7 possible actions.
//!
//! | Value  |  Observation |
//! | :----: | :---------:  |
//! | 0      | Rock         |
//! | 1      | Paper        |
//! | 2      | Scissors     |
//! | 3      | Lizard       |
//! | 4      | Spock        |
//! | 5      | Action_6     |
//! | 6      | Action_7     |
//! | 7      | None         |
//!
//! ### Action Space
//!
//! #### Rock, Paper, Scissors
//!
//! The action space is a scalar value with 3 possible values. The values are encoded as follows: Rock is 0, paper is 1 and scissors is 2.
//!
//! | Value  |  Action |
//! | :----: | :---------:  |
//! | 0      | Rock         |
//! | 1      | Paper        |
//! | 2      | Scissors     |
//!
//! #### Expanded Game
//!
//! The action space is a scalar value with n possible values, where n is the number of additional action pairs. The values for 7 possible actions are encoded as in the following table.
//!
//! | Value  |  Action |
//! | :----: | :---------:  |
//! | 0      | Rock         |
//! | 1      | Paper        |
//! | 2      | Scissors     |
//! | 3      | Lizard       |
//! | 4      | Spock        |
//! | 5      | Action_6     |
//! | 6      | Action_7     |
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
//! * v2: Merge RPS and rock paper lizard scissors spock environments, add num_actions and max_cycles arguments (1.9.0)
//! * v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
//! * v0: Initial versions release (1.0.0)
//!

#![expect(
    clippy::doc_markdown,
    reason = "the module documentation is copied verbatim from PettingZoo"
)]

use std::collections::HashMap;
use std::error::Error;
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

/// PettingZoo's named action prefix in exact numeric order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Move {
    /// Action 0.
    Rock,
    /// Action 1.
    Paper,
    /// Action 2.
    Scissors,
}

impl Move {
    /// Return the stable PettingZoo action index.
    const fn index(self) -> usize {
        match self {
            Self::Rock => 0,
            Self::Paper => 1,
            Self::Scissors => 2,
        }
    }

    /// Convert a table column into one default-game action.
    const fn from_index(index: usize) -> Self {
        match index % 3 {
            0 => Self::Rock,
            1 => Self::Paper,
            _ => Self::Scissors,
        }
    }
}

impl IndexedAction for Move {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }
}

/// One simultaneous PettingZoo parallel-API transition.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RoundTransition {
    /// Each player observes the opponent's current action.
    observations: [usize; 2],
    /// Zero-sum rewards in player order.
    rewards: [f64; 2],
    /// The episode truncates after `max_cycles` rounds.
    status: EpisodeStatus,
}

/// Exact simultaneous game state for PettingZoo `rps_v2`.
#[derive(Debug, Clone, PartialEq)]
struct RpsGame {
    /// Odd action count accepted by the source environment.
    action_count: usize,
    /// Maximum simultaneous rounds before truncation.
    max_cycles: usize,
    /// Actions displayed in the source renderer's current row.
    current: [Option<usize>; 2],
    /// Five previous pairs displayed newest first.
    history: [Option<usize>; 10],
    /// Completed simultaneous rounds.
    cycles: usize,
    /// Episode returns in stable player order.
    returns: [f64; 2],
}

impl Default for RpsGame {
    fn default() -> Self {
        match Self::new(3, 15) {
            Ok(game) => game,
            Err(_message) => Self {
                action_count: 3,
                max_cycles: 15,
                current: [None; 2],
                history: [None; 10],
                cycles: 0,
                returns: [0.0; 2],
            },
        }
    }
}

impl RpsGame {
    /// Construct one source-compatible action and cycle profile.
    const fn new(action_count: usize, max_cycles: usize) -> Result<Self, &'static str> {
        if action_count < 3 {
            return Err("action count must be at least three");
        }
        if action_count.is_multiple_of(2) {
            return Err("action count must be odd");
        }
        if max_cycles == 0 {
            return Err("maximum cycles must be greater than zero");
        }
        Ok(Self {
            action_count,
            max_cycles,
            current: [None; 2],
            history: [None; 10],
            cycles: 0,
            returns: [0.0; 2],
        })
    }

    /// Restore the source reset state and return each player's `None` observation.
    const fn reset(&mut self) -> [usize; 2] {
        self.current = [None; 2];
        self.history = [None; 10];
        self.cycles = 0;
        self.returns = [0.0; 2];
        [self.action_count; 2]
    }

    /// Apply PettingZoo's parity rule to one simultaneous action pair.
    fn play_actions(&mut self, actions: [usize; 2]) -> Result<RoundTransition, &'static str> {
        if actions.iter().any(|action| *action >= self.action_count) {
            return Err("action must belong to the configured discrete space");
        }
        Ok(self.apply_actions(actions))
    }

    /// Advance one already-validated simultaneous action pair.
    fn apply_actions(&mut self, actions: [usize; 2]) -> RoundTransition {
        // The Python renderer shifts the old current pair into history only
        // when the next pair arrives, so the visible rows keep exact timing.
        if self.current[0].is_some() {
            self.history.copy_within(0..8, 2);
            self.history[0] = self.current[0];
            self.history[1] = self.current[1];
        }
        self.current = [Some(actions[0]), Some(actions[1])];

        let rewards = round_rewards(actions);
        self.cycles += 1;
        self.returns[0] += rewards[0];
        self.returns[1] += rewards[1];
        RoundTransition {
            observations: [actions[1], actions[0]],
            rewards,
            status: if self.cycles >= self.max_cycles {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
        }
    }
}

/// Reproduce the source environment's generalized odd-action payoff rule.
const fn round_rewards(actions: [usize; 2]) -> [f64; 2] {
    let [first, second] = actions;
    if first == second {
        return [0.0, 0.0];
    }
    let first_wins = if (first + second).is_multiple_of(2) {
        first < second
    } else {
        first > second
    };
    if first_wins {
        [1.0, -1.0]
    } else {
        [-1.0, 1.0]
    }
}

/// Multi-agent details retained by the single-learner Bevy Gym adapter.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RpsInfo {
    /// Observations in PettingZoo player order.
    observations: [usize; 2],
    /// Rewards in PettingZoo player order.
    rewards: [f64; 2],
}

/// Seeded uniform opponent used by the official random-policy comparison.
#[derive(Debug, Clone, Copy, Default)]
struct SplitMix64(u64);

impl SplitMix64 {
    /// Return one uniformly distributed index below `upper`.
    fn index(&mut self, upper: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let mixed = value ^ (value >> 31);
        let Ok(bound) = u64::try_from(upper) else {
            return 0;
        };
        if bound == 0 {
            return 0;
        }
        usize::try_from(mixed % bound).unwrap_or_default()
    }
}

/// Player-zero learning view against a seeded uniform-random opponent.
#[derive(Debug, Clone, Default)]
struct RpsTrainingEnv {
    /// Complete two-player source-compatible game.
    game: RpsGame,
    /// Opponent action stream reset from the environment seed.
    rng: SplitMix64,
}

impl Env for RpsTrainingEnv {
    type Observation = usize;
    type Action = Move;
    type Info = RpsInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        let observations = self.game.reset();
        self.rng = SplitMix64(seed.unwrap_or_default());
        Reset {
            observation: observations[0],
            info: RpsInfo {
                observations,
                rewards: [0.0; 2],
            },
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        // Sample the comparison opponent from the same seeded discrete action
        // space that PettingZoo exposes to a random agent.
        let opponent = self.rng.index(self.game.action_count);
        let round = match self.game.play_actions([action.index(), opponent]) {
            Ok(round) => round,
            Err(_message) => RoundTransition {
                observations: [self.game.action_count; 2],
                rewards: [-1.0, 0.0],
                status: EpisodeStatus::Terminated,
            },
        };
        Step {
            observation: round.observations[0],
            reward: round.rewards[0],
            status: round.status,
            info: RpsInfo {
                observations: round.observations,
                rewards: round.rewards,
            },
        }
    }

    fn episode_extras(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("player_0_return".to_owned(), self.game.returns[0]),
            ("player_1_return".to_owned(), self.game.returns[1]),
        ])
    }
}

impl TabularExample for RpsTrainingEnv {
    const ENV_NAME: &'static str = "rps-v2";
    const GYMNASIUM_ID: &'static str = "classic/rps-v2";
    const STATE_COUNT: usize = 4;
    const ACTION_COUNT: usize = 3;
    const MAX_STEPS: usize = 15;
    const DEFAULT_EPISODES: usize = 500;
    const DEFAULT_EVAL_INTERVAL: usize = 25;
    const DEFAULT_EVAL_EPISODES: usize = 100;
    const DEFAULT_LEARNING_RATE: f64 = 0.2;
    const GAMMA: f64 = 0.0;
    const EPSILON_START: f64 = 0.3;
    const EPSILON_END: f64 = 0.01;
    const EPSILON_DECAY_STEPS_PER_EPISODE: usize = 15;
    const SOLVED_SCORE: f64 = f64::INFINITY;
    const GIF_PATH: &'static str = "docs/images/classic-rps-v2.gif";

    fn selection_score(evaluation: &Evaluation) -> f64 {
        evaluation.mean_reward
    }

    fn is_success(_final_observation: usize, total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Truncated && total_reward > 0.0
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
    run_tabular_workflow::<RpsTrainingEnv>()
}

/// Source-matching Bevy renderer.
#[cfg(feature = "render")]
mod render {
    use super::{Env, Error, Move, Path, RpsTrainingEnv};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::sprite::Anchor;
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::{TabularQPolicy, TabularQTrainer};
    use bevy_inspector_egui as _;

    /// PettingZoo's default RPS viewport width.
    const WIDTH: f32 = 285.0;
    /// Integer viewport width used by window and encoder APIs.
    const WIDTH_PIXELS: u32 = 285;
    /// PettingZoo's default RPS viewport height.
    const HEIGHT: f32 = 800.0;
    /// Integer viewport height used by window and encoder APIs.
    const HEIGHT_PIXELS: u32 = 800;

    /// Adds the exact white source background.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Greedy policy and source-compatible visible state.
    #[derive(Resource)]
    struct VisualRps {
        /// Validation-selected table policy.
        policy: TabularQPolicy,
        /// Visible game and seeded random-opponent adapter.
        env: RpsTrainingEnv,
        /// Current player-0 observation.
        observation: usize,
    }

    /// Source hand images loaded once and reused by every slot.
    #[derive(Resource)]
    struct RpsImages {
        /// Rock hand.
        rock: Handle<Image>,
        /// Paper hand.
        paper: Handle<Image>,
        /// Scissors hand.
        scissors: Handle<Image>,
        /// Spock hand.
        spock: Handle<Image>,
        /// Lizard hand.
        lizard: Handle<Image>,
    }

    /// One source current-action slot.
    #[derive(Component)]
    struct CurrentMove {
        /// Stable player index.
        player: usize,
    }

    /// One source history slot.
    #[derive(Component)]
    struct HistoryMove {
        /// Stable history index, newest pair first.
        slot: usize,
    }

    /// Two-Hz source playback timer.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run an interactive or finite source-matching scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let trainer = TabularQTrainer::load(checkpoint)?;
        let mut env = RpsTrainingEnv::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin)
            .insert_resource(VisualRps {
                policy: trainer.policy(),
                env,
                observation,
            })
            .insert_resource(VisualClock(Timer::from_seconds(0.5, TimerMode::Repeating)))
            .add_plugins(
                DefaultPlugins
                    .set(AssetPlugin {
                        file_path: "examples/classic/assets/rps".into(),
                        ..default()
                    })
                    .set(ImagePlugin::default_nearest())
                    .set(WindowPlugin {
                        primary_window: Some(Window {
                            title: "Rock Paper Scissors".into(),
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

    /// Spawn the two labels and the exact two-by-six hand layout.
    fn setup_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("RPS Camera")));
        let images = RpsImages {
            rock: assets.load("img/Rock.png"),
            paper: assets.load("img/Paper.png"),
            scissors: assets.load("img/Scissors.png"),
            spock: assets.load("img/Spock.png"),
            lizard: assets.load("img/Lizard.png"),
        };
        let font = assets.load("font/Minecraft.ttf");

        for player in 0..2 {
            let x = if player == 0 { 64.125 } else { 220.875 };
            commands.spawn((
                Text2d::new(format!("Agent {}", player + 1)),
                TextFont {
                    font: font.clone(),
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::BLACK),
                Anchor::CENTER,
                Transform::from_xyz(source_x(x), source_y(20.0), 2.0),
            ));
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                CurrentMove { player },
            ));
        }

        for slot in 0..10 {
            commands.spawn((
                Sprite::default(),
                Transform::default(),
                Visibility::Hidden,
                HistoryMove { slot },
            ));
        }
        commands.insert_resource(images);
    }

    /// Advance the learned-versus-random comparison at PettingZoo's two FPS.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualRps>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Apply one policy action or reset after the fifteenth cycle.
    fn advance_visual(visual: &mut VisualRps) {
        if visual.env.game.cycles >= visual.env.game.max_cycles {
            visual.observation = visual.env.reset(None).observation;
            return;
        }
        let Ok(action) = visual.policy.greedy_action(visual.observation) else {
            return;
        };
        let transition = visual.env.step(Move::from_index(action));
        visual.observation = transition.observation;
    }

    /// Synchronize current and historical sprites after each visible change.
    fn sync_scene(
        visual: Res<'_, VisualRps>,
        images: Res<'_, RpsImages>,
        mut current: Query<'_, '_, (&CurrentMove, &mut Sprite, &mut Transform, &mut Visibility)>,
        mut history: Query<
            '_,
            '_,
            (&HistoryMove, &mut Sprite, &mut Transform, &mut Visibility),
            Without<CurrentMove>,
        >,
    ) {
        if !visual.is_changed() {
            return;
        }
        sync_sprites(&visual, &images, &mut current, &mut history);
    }

    /// Apply visible game state to both fixed sprite queries.
    fn sync_sprites(
        visual: &VisualRps,
        images: &RpsImages,
        current: &mut Query<'_, '_, (&CurrentMove, &mut Sprite, &mut Transform, &mut Visibility)>,
        history: &mut Query<
            '_,
            '_,
            (&HistoryMove, &mut Sprite, &mut Transform, &mut Visibility),
            Without<CurrentMove>,
        >,
    ) {
        for (slot, mut sprite, mut transform, mut visibility) in current.iter_mut() {
            let action = visual.env.game.current.get(slot.player).copied().flatten();
            set_move_sprite(
                action,
                slot.player,
                None,
                images,
                &mut sprite,
                &mut transform,
                &mut visibility,
            );
        }
        for (slot, mut sprite, mut transform, mut visibility) in history.iter_mut() {
            let action = visual.env.game.history.get(slot.slot).copied().flatten();
            set_move_sprite(
                action,
                slot.slot % 2,
                Some(slot.slot / 2),
                images,
                &mut sprite,
                &mut transform,
                &mut visibility,
            );
        }
    }

    /// Apply the source image, scaled size, and top-left placement for one slot.
    fn set_move_sprite(
        action: Option<usize>,
        player: usize,
        history_row: Option<usize>,
        images: &RpsImages,
        sprite: &mut Sprite,
        transform: &mut Transform,
        visibility: &mut Visibility,
    ) {
        let Some(action) = action else {
            *visibility = Visibility::Hidden;
            return;
        };
        let (image, aspect) = move_image(images, action);
        let base = if history_row.is_some() {
            HEIGHT / 9.0
        } else {
            HEIGHT / 7.0
        };
        let size = Vec2::new(base * aspect.x, base * aspect.y);
        let top = history_row.map_or(HEIGHT / 12.0, |row| {
            (HEIGHT / 7.0).mul_add(row as f32, HEIGHT * 7.0 / 24.0)
        });
        let left = if history_row.is_some() {
            if player == 0 {
                WIDTH / 2.0 - HEIGHT / 9.0 - HEIGHT * 7.0 / 126.0
            } else {
                WIDTH / 2.0 + HEIGHT * 7.0 / 126.0
            }
        } else if player == 0 {
            WIDTH / 2.0 - HEIGHT / 7.0 - HEIGHT / 42.0
        } else {
            WIDTH / 2.0 + HEIGHT / 42.0
        };
        sprite.image = image;
        sprite.custom_size = Some(size);
        transform.translation = Vec3::new(
            source_x(left + size.x / 2.0),
            source_y(top + size.y / 2.0),
            1.0,
        );
        *visibility = Visibility::Visible;
    }

    /// Return the source image and width/height multipliers for one action.
    fn move_image(images: &RpsImages, action: usize) -> (Handle<Image>, Vec2) {
        match action {
            0 => (images.rock.clone(), Vec2::new(1.0, 10.0 / 13.0)),
            1 => (images.paper.clone(), Vec2::new(1.0, 14.0 / 12.0)),
            2 => (images.scissors.clone(), Vec2::new(1.0, 14.0 / 13.0)),
            3 => (images.spock.clone(), Vec2::ONE),
            _ => (images.lizard.clone(), Vec2::new(9.0 / 18.0, 1.0)),
        }
    }

    /// Convert a source x coordinate into centered Bevy world space.
    const fn source_x(value: f32) -> f32 {
        value - WIDTH / 2.0
    }

    /// Convert a source y coordinate into Bevy's upward-positive world space.
    const fn source_y(value: f32) -> f32 {
        HEIGHT / 2.0 - value
    }

    /// Capture five seconds at 20 FPS while advancing the game at two FPS.
    fn capture_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualRps>,
        images: Res<'_, RpsImages>,
        mut current: Query<'_, '_, (&CurrentMove, &mut Sprite, &mut Transform, &mut Visibility)>,
        mut history: Query<
            '_,
            '_,
            (&HistoryMove, &mut Sprite, &mut Transform, &mut Visibility),
            Without<CurrentMove>,
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
        sync_sprites(&visual, &images, &mut current, &mut history);
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Render and encode the standard five-second demonstration.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(
            output,
            "classic-rps-v2",
            WIDTH_PIXELS,
            HEIGHT_PIXELS,
            |frames| run_visual(checkpoint, Some(frames)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{round_rewards, Env, EpisodeStatus, Move, RpsGame, RpsTrainingEnv};

    #[test]
    fn paper_beats_rock_with_zero_sum_rewards() -> Result<(), &'static str> {
        let mut game = RpsGame::default();

        let transition = game.play_actions([Move::Paper.index(), Move::Rock.index()])?;

        assert_eq!(transition.rewards, [1.0, -1.0]);
        Ok(())
    }

    #[test]
    fn default_payoff_matrix_matches_the_python_parity_rule() {
        let expected = [
            [[0.0, 0.0], [-1.0, 1.0], [1.0, -1.0]],
            [[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]],
            [[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]],
        ];

        for (first, row) in expected.iter().enumerate() {
            for (second, rewards) in row.iter().enumerate() {
                assert_eq!(round_rewards([first, second]), *rewards);
            }
        }
    }

    #[test]
    fn reset_and_observations_match_the_parallel_api() -> Result<(), &'static str> {
        let mut game = RpsGame::default();
        assert_eq!(game.reset(), [3, 3]);

        let transition = game.play_actions([Move::Scissors.index(), Move::Paper.index()])?;

        assert_eq!(
            transition.observations,
            [Move::Paper.index(), Move::Scissors.index()]
        );
        assert_eq!(game.current, [Some(2), Some(1)]);
        assert_eq!(game.history, [None; 10]);
        Ok(())
    }

    #[test]
    fn history_keeps_the_five_previous_action_pairs() {
        let mut game = RpsGame::default();
        for cycle in 0..7 {
            assert_eq!(
                game.play_actions([cycle % 3, (cycle + 1) % 3]).map(|_| ()),
                Ok(())
            );
        }

        assert_eq!(game.current, [Some(0), Some(1)]);
        assert_eq!(
            game.history,
            [
                Some(2),
                Some(0),
                Some(1),
                Some(2),
                Some(0),
                Some(1),
                Some(2),
                Some(0),
                Some(1),
                Some(2),
            ]
        );
    }

    #[test]
    fn source_profiles_reject_invalid_configuration() {
        assert_eq!(
            RpsGame::new(2, 15),
            Err("action count must be at least three")
        );
        assert_eq!(RpsGame::new(4, 15), Err("action count must be odd"));
        assert_eq!(
            RpsGame::new(5, 0),
            Err("maximum cycles must be greater than zero")
        );
        assert_eq!(RpsGame::new(5, 15).map(|_| ()), Ok(()));
    }

    #[test]
    fn default_episode_truncates_after_exactly_fifteen_cycles() -> Result<(), &'static str> {
        let mut game = RpsGame::default();
        for _ in 0..14 {
            assert_eq!(
                game.play_actions([Move::Paper.index(), Move::Rock.index()])?
                    .status,
                EpisodeStatus::Continuing
            );
        }

        let final_round = game.play_actions([Move::Paper.index(), Move::Rock.index()])?;

        assert_eq!(final_round.status, EpisodeStatus::Truncated);
        assert_eq!(game.returns, [15.0, -15.0]);
        Ok(())
    }

    #[test]
    fn training_adapter_uses_a_seeded_random_opponent() {
        let mut first = RpsTrainingEnv::default();
        let mut second = RpsTrainingEnv::default();
        first.reset(Some(42));
        second.reset(Some(42));

        let first_rewards: Vec<_> = (0..15).map(|_| first.step(Move::Rock).reward).collect();
        let second_rewards: Vec<_> = (0..15).map(|_| second.step(Move::Rock).reward).collect();

        assert_eq!(first_rewards, second_rewards);
        assert!(first_rewards.iter().any(|reward| *reward == -1.0));
        assert!(first_rewards.iter().any(|reward| *reward == 0.0));
        assert!(first_rewards.iter().any(|reward| *reward == 1.0));
    }
}
