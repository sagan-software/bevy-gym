//! The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them
//! off at one of four locations.
//!
//! ## Description
//! There are four designated pick-up and drop-off locations (Red, Green, Yellow and Blue) in the
//! 5x5 grid world. The taxi starts off at a random square and the passenger at one of the
//! designated locations.
//!
//! The goal is move the taxi to the passenger's location, pick up the passenger,
//! move to the passenger's desired destination, and
//! drop off the passenger. Once the passenger is dropped off, the episode ends.
//!
//! The player receives positive rewards for successfully dropping-off the passenger at the correct
//! location. Negative rewards for incorrect attempts to pick-up/drop-off passenger and
//! for each step where another reward is not received.
//!
//! Map:
//!
//!     +---------+
//!     |R: | : :G|
//!     | : | : : |
//!     | : : : : |
//!     | | : | : |
//!     |Y| : |B: |
//!     +---------+
//!
//! From "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
//! by Tom Dietterich [<a href="#taxi_ref">1</a>].
//!
//! ## Action Space
//! The action shape is `(1,)` in the range `{0, 5}` indicating
//! which direction to move the taxi or to pickup/drop off passengers.
//!
//! - 0: Move south (down)
//! - 1: Move north (up)
//! - 2: Move east (right)
//! - 3: Move west (left)
//! - 4: Pickup passenger
//! - 5: Drop off passenger
//!
//! ## Observation Space
//! There are 500 discrete states since there are 25 taxi positions, 5 possible
//! locations of the passenger (including the case when the passenger is in the
//! taxi), and 4 destination locations.
//!
//! Destination on the ansi rendered map are represented with the first letter of the color.
//!
//! Passenger locations:
//! - 0: Red
//! - 1: Green
//! - 2: Yellow
//! - 3: Blue
//! - 4: In taxi
//!
//! Destinations:
//! - 0: Red
//! - 1: Green
//! - 2: Yellow
//! - 3: Blue
//!
//! An observation is returned as an `int()` that encodes the corresponding state, calculated by
//! `((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination`
//!
//! Note that there are 400 states that can actually be reached during an
//! episode. The missing states correspond to situations in which the passenger
//! is at the same location as their destination, as this typically signals the
//! end of an episode. Four additional states can be observed right after a
//! successful episodes, when both the passenger and the taxi are at the destination.
//! This gives a total of 404 reachable discrete states.
//!
//! ## Starting State
//! The initial state is sampled uniformly from the possible states
//! where the passenger is neither at their destination nor inside the taxi.
//! There are 300 possible initial states: 25 taxi positions, 4 passenger locations (excluding inside the taxi)
//! and 3 destinations (excluding the passenger's current location).
//!
//! ## Rewards
//! - -1 per step unless another reward is triggered.
//! - +20 delivering passenger.
//! - -10  executing "pickup" and "drop-off" actions illegally.
//!
//! An action that results a noop, like moving into a wall, will incur the time step
//! penalty. Noops can be avoided by sampling the `action_mask` returned in `info`.
//!
//! ## Episode End
//! The episode ends if the following happens:
//!
//! - Termination:
//!     1. The taxi drops off the passenger.
//!
//! - Truncation (when using the `time_limit` wrapper):
//!     1. The length of the episode is 200.
//!
//! ## Information
//!
//! `step()` and `reset()` return a dict with the following keys:
//! - "prob": transition probability for the state.
//! - "`action_mask"`: if actions will cause a transition to a new state. This was added in v0.25.0
//!
//! For some cases, taking an action will have no effect on the state of the episode.
//! In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the actions specifying
//! if the action will change the state.
//!
//! To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
//! Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.
//!
//! ## Arguments
//!
//! ```python
//! import gymnasium as gym
//! gym.make('Taxi-v4')
//! ```
//!
//! <a id="is_rainy"></a>`is_rainy=False`: If True the cab will move in the intended direction with probability
//! 80%, controlled by `rainy_probability`, else in a lateral direction with equal probability.
//! Pickup and dropoff actions remain deterministic (probability 1.0).
//!
//! <a id="rainy_probability"></a>`rainy_probability=0.8`: When `is_rainy=True`, the probability of
//! moving in the intended direction. Each lateral direction is given `(1 - rainy_probability) / 2`.
//!
//! <a id="fickle_passenger"></a>`fickle_passenger=False`: If True the passenger has a chance 30%,
//! controlled by `fickle_probability`, of changing destinations when the cab has moved one square away from the
//! passenger's source location. Passenger fickleness only happens on the first pickup and successful movement.
//! If the passenger is dropped off at the source location and picked up again, it isn't triggered again.
//!
//! <a id="fickle_probability"></a>`fickle_probability=0.3`: When `fickle_passenger=True`, the probability
//! that the passenger changes destination on the first move after pickup.
//!
//! ## References
//! <a id="taxi_ref"></a>[1] T. G. Dietterich, “Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition,”
//! Journal of Artificial Intelligence Research, vol. 13, pp. 227–303, Nov. 2000, doi: 10.1613/jair.639.
//!
//! ## Version History
//! * v4: In v1.3.0, fix `is_rainy=True` and `fickle_passenger=True` implementations
//!     - Add `rainy_probability` and `fickle_probability` arguments to tune the stochastic behaviour
//! * v3: Map Correction + Cleaner Domain Description,
//!     - In v0.25.0 action masking added to the reset and step information
//!     - In v1.2.0 added `is_rainy` and `fickle_passenger` arguments to align with Dietterich, 2000 Section 7.1
//! * v2: Disallow Taxi start location = goal location, Update Taxi observations in the rollout, Update Taxi reward threshold.
//! * v1: Remove (3,2) from locs, add passidx<4 check
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

/// Taxi-v4 action in Gymnasium's table order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaxiAction {
    /// Move one row south when no boundary blocks the taxi.
    South,
    /// Move one row north when no boundary blocks the taxi.
    North,
    /// Move one column east when no median blocks the taxi.
    East,
    /// Move one column west when no median blocks the taxi.
    West,
    /// Put a co-located waiting passenger in the taxi.
    Pickup,
    /// Put the passenger at a named location or finish at the destination.
    Dropoff,
}

impl TaxiAction {
    /// Convert one table column into the closed action vocabulary.
    const fn from_index(index: usize) -> Self {
        match index % 6 {
            0 => Self::South,
            1 => Self::North,
            2 => Self::East,
            3 => Self::West,
            4 => Self::Pickup,
            _ => Self::Dropoff,
        }
    }

    /// Convert this action into Gymnasium's sprite orientation index.
    #[cfg(feature = "render")]
    const fn orientation(self) -> Option<usize> {
        match self {
            Self::South => Some(0),
            Self::North => Some(1),
            Self::East => Some(2),
            Self::West => Some(3),
            Self::Pickup | Self::Dropoff => None,
        }
    }
}

impl IndexedAction for TaxiAction {
    fn from_index(index: usize) -> Self {
        Self::from_index(index)
    }
}

/// One decoded Taxi-v4 table state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TaxiState {
    /// Taxi row in the five-by-five road grid.
    row: usize,
    /// Taxi column in the five-by-five road grid.
    column: usize,
    /// Passenger location, where four means inside the taxi.
    passenger: usize,
    /// Destination location index.
    destination: usize,
}

impl TaxiState {
    /// Encode this state with Gymnasium's exact row-major formula.
    const fn encode(self) -> usize {
        ((self.row * 5 + self.column) * 5 + self.passenger) * 4 + self.destination
    }

    /// Decode one of the 500 table states.
    const fn decode(mut encoded: usize) -> Self {
        let destination = encoded % 4;
        encoded /= 4;
        let passenger = encoded % 5;
        encoded /= 5;
        let column = encoded % 5;
        encoded /= 5;
        Self {
            row: encoded,
            column,
            passenger,
            destination,
        }
    }
}

/// Gymnasium Taxi-v4 with the default dry and non-fickle dynamics.
#[derive(Debug, Clone)]
struct Taxi {
    /// Current decoded state.
    state: TaxiState,
    /// Last action used by the matching renderer.
    last_action: Option<TaxiAction>,
    /// Deterministic environment-owned reset generator.
    rng: SplitMix64,
}

impl Default for Taxi {
    fn default() -> Self {
        Self {
            state: TaxiState {
                row: 0,
                column: 0,
                passenger: 0,
                destination: 1,
            },
            last_action: None,
            rng: SplitMix64::new(0),
        }
    }
}

impl Taxi {
    /// Return the four named locations in Gymnasium order: red, green, yellow, blue.
    const fn locations() -> [(usize, usize); 4] {
        [(0, 0), (0, 4), (4, 0), (4, 3)]
    }

    /// Return whether the map permits an east move from this cell.
    fn can_move_east(row: usize, column: usize) -> bool {
        const MAP: [&[u8; 11]; 7] = [
            b"+---------+",
            b"|R: | : :G|",
            b"| : | : : |",
            b"| : : : : |",
            b"| | : | : |",
            b"|Y| : |B: |",
            b"+---------+",
        ];
        column < 4
            && MAP
                .get(row + 1)
                .copied()
                .expect("fixed map row is valid")
                .get(2 * column + 2)
                .copied()
                .expect("fixed map column is valid")
                == b':'
    }

    /// Return the six fixed action-mask values for one state.
    fn action_mask(state: TaxiState) -> Vec<bool> {
        let mut mask = vec![
            state.row < 4,
            state.row > 0,
            Self::can_move_east(state.row, state.column),
            state.column > 0 && Self::can_move_east(state.row, state.column - 1),
            false,
            false,
        ];
        let taxi = (state.row, state.column);
        if state.passenger < 4
            && taxi
                == Self::locations()
                    .get(state.passenger)
                    .copied()
                    .expect("passenger location index is valid")
        {
            *mask.get_mut(4).expect("fixed example index is valid") = true;
        }
        if state.passenger == 4
            && (taxi
                == Self::locations()
                    .get(state.destination)
                    .copied()
                    .expect("destination location index is valid")
                || Self::locations().contains(&taxi))
        {
            *mask.get_mut(5).expect("fixed example index is valid") = true;
        }
        mask
    }

    /// Sample uniformly from Gymnasium's 300 legal initial states.
    fn sample_initial_state(&mut self) -> TaxiState {
        let sample = self.rng.index(300);
        let taxi = sample / 12;
        let passenger = (sample % 12) / 3;
        let destination_offset = sample % 3;
        let destination = if destination_offset >= passenger {
            destination_offset + 1
        } else {
            destination_offset
        };
        TaxiState {
            row: taxi / 5,
            column: taxi % 5,
            passenger,
            destination,
        }
    }
}

impl Env for Taxi {
    type Action = TaxiAction;
    type Observation = usize;
    type Info = TaxiInfo;

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.state = self.sample_initial_state();
        self.last_action = None;
        Reset {
            observation: self.state.encode(),
            info: TaxiInfo {
                probability: 1.0,
                action_mask: Self::action_mask(self.state),
            },
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let mut reward = -1.0;
        let mut status = EpisodeStatus::Continuing;
        match action {
            TaxiAction::South => self.state.row = (self.state.row + 1).min(4),
            TaxiAction::North => self.state.row = self.state.row.saturating_sub(1),
            TaxiAction::East if Self::can_move_east(self.state.row, self.state.column) => {
                self.state.column += 1;
            }
            TaxiAction::West
                if self.state.column > 0
                    && Self::can_move_east(self.state.row, self.state.column - 1) =>
            {
                self.state.column -= 1;
            }
            TaxiAction::Pickup
                if self.state.passenger < 4
                    && (self.state.row, self.state.column)
                        == Self::locations()
                            .get(self.state.passenger)
                            .copied()
                            .expect("passenger location index is valid") =>
            {
                self.state.passenger = 4;
            }
            TaxiAction::Dropoff
                if self.state.passenger == 4
                    && (self.state.row, self.state.column)
                        == Self::locations()
                            .get(self.state.destination)
                            .copied()
                            .expect("destination location index is valid") =>
            {
                self.state.passenger = self.state.destination;
                reward = 20.0;
                status = EpisodeStatus::Terminated;
            }
            TaxiAction::Dropoff if self.state.passenger == 4 => {
                if let Some(location) = Self::locations()
                    .iter()
                    .position(|location| *location == (self.state.row, self.state.column))
                {
                    self.state.passenger = location;
                } else {
                    reward = -10.0;
                }
            }
            TaxiAction::Pickup | TaxiAction::Dropoff => reward = -10.0,
            TaxiAction::East | TaxiAction::West => {}
        }
        self.last_action = Some(action);
        Step {
            observation: self.state.encode(),
            reward,
            status,
            info: TaxiInfo {
                probability: 1.0,
                action_mask: Self::action_mask(self.state),
            },
        }
    }
}

/// Taxi-v4 transition metadata.
#[derive(Debug, Clone, Default, PartialEq)]
struct TaxiInfo {
    /// Deterministic transition probability.
    probability: f64,
    /// Fixed legality mask for the resulting state.
    action_mask: Vec<bool>,
}

impl TabularExample for Taxi {
    const ENV_NAME: &'static str = "taxi";
    const GYMNASIUM_ID: &'static str = "Taxi-v4";
    const STATE_COUNT: usize = 500;
    const ACTION_COUNT: usize = 6;
    const MAX_STEPS: usize = 200;
    const DEFAULT_EPISODES: usize = 5_000;
    const DEFAULT_EVAL_INTERVAL: usize = 500;
    const DEFAULT_EVAL_EPISODES: usize = 1_000;
    const DEFAULT_LEARNING_RATE: f64 = 0.2;
    const GAMMA: f64 = 0.95;
    const EPSILON_START: f64 = 0.1;
    const EPSILON_END: f64 = 0.1;
    const EPSILON_DECAY_STEPS_PER_EPISODE: usize = 1;
    const SOLVED_SCORE: f64 = 8.0;
    const GIF_PATH: &'static str = "docs/images/taxi.gif";

    fn action_masks() -> Option<Vec<Vec<bool>>> {
        Some(
            (0..Self::STATE_COUNT)
                .map(|state| Self::action_mask(TaxiState::decode(state)))
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
    run_tabular_workflow::<Taxi>()
}

/// Small deterministic generator used only for initial-state sampling.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    /// Current generator state.
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from one seed.
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Return a bounded index.
    fn index(&mut self, upper: usize) -> usize {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        usize::try_from(value % u64::try_from(upper).unwrap_or(1)).unwrap_or(0)
    }
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{Env, EpisodeStatus, Error, Path, Taxi, TaxiAction, TaxiState};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
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
            app.insert_resource(ClearColor(Color::WHITE));
        }
    }

    /// Policy playback state shown in the Taxi scene.
    #[derive(Resource)]
    struct VisualTaxi {
        /// Greedy checkpoint policy.
        policy: TabularQPolicy,
        /// Live environment state.
        env: Taxi,
        /// Current encoded observation.
        state: usize,
        /// Current cab orientation index.
        orientation: usize,
    }

    /// Exact Gymnasium Taxi-v4 sprite handles.
    #[derive(Resource)]
    struct TaxiImages {
        /// Cab sprites in south, north, east, west order.
        cabs: [Handle<Image>; 4],
    }

    /// Visible taxi sprite.
    #[derive(Component)]
    struct TaxiCab;

    /// Visible passenger sprite.
    #[derive(Component)]
    struct TaxiPassenger;

    /// Visible destination hotel sprite.
    #[derive(Component)]
    struct TaxiHotel;

    /// Four-Hz interactive playback clock.
    #[derive(Resource)]
    struct VisualClock(Timer);

    /// Run the Gymnasium-matching interactive or finite-capture scene.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let trainer = TabularQTrainer::load(checkpoint)?;
        let mut env = Taxi::default();
        let state = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualTaxi {
            policy: trainer.policy(),
            env,
            state,
            orientation: 0,
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
                        title: "bevy-gym Taxi-v4".into(),
                        resolution: WindowResolution::new(550, 350),
                        present_mode: PresentMode::AutoVsync,
                        resizable: false,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_systems(Startup, setup_taxi_scene);

        if let Some(directory) = capture_dir {
            app.insert_resource(GifCapture::new(directory))
                .add_systems(Update, capture_gif_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_taxi_scene).chain());
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

    /// Build Gymnasium's 11-by-7 Taxi-v4 composition from first-party assets.
    fn setup_taxi_scene(mut commands: Commands<'_, '_>, assets: Res<'_, AssetServer>) {
        commands.spawn((Camera2d, Name::new("Taxi Camera")));
        spawn_taxi_map(&mut commands, &assets);

        let colors = [
            Color::srgba_u8(255, 0, 0, 128),
            Color::srgba_u8(0, 255, 0, 128),
            Color::srgba_u8(255, 255, 0, 128),
            Color::srgba_u8(0, 0, 255, 128),
        ];
        for (location, color) in Taxi::locations().into_iter().zip(colors) {
            let mut position = road_position(location.0, location.1);
            position.y -= 10.0;
            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(50.0)),
                    ..default()
                },
                Transform::from_xyz(position.x, position.y, 2.0),
            ));
        }

        spawn_taxi_actors(&mut commands, &assets);
    }

    /// Spawn the fixed Gymnasium map tiles and median segments.
    fn spawn_taxi_map(commands: &mut Commands<'_, '_>, assets: &AssetServer) {
        const MAP: [&str; 7] = [
            "+---------+",
            "|R: | : :G|",
            "| : | : : |",
            "| : : : : |",
            "| | : | : |",
            "|Y| : |B: |",
            "+---------+",
        ];

        let background = assets.load("img/taxi_background.png");
        let median_left = assets.load("img/gridworld_median_left.png");
        let median_horiz = assets.load("img/gridworld_median_horiz.png");
        let median_right = assets.load("img/gridworld_median_right.png");
        let median_top = assets.load("img/gridworld_median_top.png");
        let median_vert = assets.load("img/gridworld_median_vert.png");
        let median_bottom = assets.load("img/gridworld_median_bottom.png");
        for (row, line) in MAP.iter().enumerate() {
            let bytes = line.as_bytes();
            for (column, byte) in bytes.iter().copied().enumerate() {
                let position = map_cell_position(row, column);
                commands.spawn((
                    Sprite {
                        image: background.clone(),
                        custom_size: Some(Vec2::splat(50.0)),
                        ..default()
                    },
                    Transform::from_xyz(position.x, position.y, 0.0),
                ));
                let image = match byte {
                    b'|' if row == 0
                        || MAP
                            .get(row - 1)
                            .copied()
                            .expect("fixed example index is valid")
                            .as_bytes()
                            .get(column)
                            .copied()
                            .expect("fixed map column is valid")
                            != b'|' =>
                    {
                        Some(median_top.clone())
                    }
                    b'|' if row + 1 == MAP.len()
                        || MAP
                            .get(row + 1)
                            .copied()
                            .expect("fixed example index is valid")
                            .as_bytes()
                            .get(column)
                            .copied()
                            .expect("fixed map column is valid")
                            != b'|' =>
                    {
                        Some(median_bottom.clone())
                    }
                    b'|' => Some(median_vert.clone()),
                    b'-' if column == 0
                        || bytes
                            .get(column - 1)
                            .copied()
                            .expect("fixed example index is valid")
                            != b'-' =>
                    {
                        Some(median_left.clone())
                    }
                    b'-' if column + 1 == bytes.len()
                        || bytes
                            .get(column + 1)
                            .copied()
                            .expect("fixed example index is valid")
                            != b'-' =>
                    {
                        Some(median_right.clone())
                    }
                    b'-' => Some(median_horiz.clone()),
                    _ => None,
                };
                if let Some(image) = image {
                    commands.spawn((
                        Sprite {
                            image,
                            custom_size: Some(Vec2::splat(50.0)),
                            ..default()
                        },
                        Transform::from_xyz(position.x, position.y, 1.0),
                    ));
                }
            }
        }
    }

    /// Spawn the cab, passenger, hotel, and directional cab resources.
    fn spawn_taxi_actors(commands: &mut Commands<'_, '_>, assets: &AssetServer) {
        let cabs = [
            assets.load("img/cab_front.png"),
            assets.load("img/cab_rear.png"),
            assets.load("img/cab_right.png"),
            assets.load("img/cab_left.png"),
        ];
        let initial = road_position(0, 0);
        commands.spawn((
            Sprite {
                image: cabs[0].clone(),
                custom_size: Some(Vec2::splat(50.0)),
                ..default()
            },
            Transform::from_xyz(initial.x, initial.y, 4.0),
            TaxiCab,
        ));
        commands.spawn((
            Sprite {
                image: assets.load("img/passenger.png"),
                custom_size: Some(Vec2::splat(50.0)),
                ..default()
            },
            Transform::from_xyz(initial.x, initial.y, 3.0),
            Visibility::Visible,
            TaxiPassenger,
        ));
        commands.spawn((
            Sprite {
                image: assets.load("img/hotel.png"),
                color: Color::srgba(1.0, 1.0, 1.0, 170.0 / 255.0),
                custom_size: Some(Vec2::splat(50.0)),
                ..default()
            },
            Transform::from_xyz(initial.x, initial.y + 25.0, 3.0),
            TaxiHotel,
        ));
        commands.insert_resource(TaxiImages { cabs });
    }

    /// Advance policy playback at Gymnasium's four frames per second.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualTaxi>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual);
        }
    }

    /// Advance one greedy action or reset after a completed delivery.
    fn advance_visual(visual: &mut VisualTaxi) {
        let Ok(action_index) = visual.policy.greedy_action(visual.state) else {
            return;
        };
        let action = TaxiAction::from_index(action_index);
        let transition = visual.env.step(action);
        if let Some(orientation) = action.orientation() {
            visual.orientation = orientation;
        }
        visual.state = transition.observation;
        if transition.status == EpisodeStatus::Terminated {
            visual.state = visual.env.reset(None).observation;
        }
    }

    /// Synchronize all dynamic sprites with the decoded environment state.
    fn sync_taxi_scene(
        visual: Res<'_, VisualTaxi>,
        images: Res<'_, TaxiImages>,
        mut cab: Single<'_, '_, (&mut Transform, &mut Sprite), With<TaxiCab>>,
        mut passenger: Single<
            '_,
            '_,
            (&mut Transform, &mut Visibility),
            (With<TaxiPassenger>, Without<TaxiCab>),
        >,
        mut hotel: Single<
            '_,
            '_,
            (&mut Transform, &mut Visibility),
            (With<TaxiHotel>, Without<TaxiCab>, Without<TaxiPassenger>),
        >,
    ) {
        let state = TaxiState::decode(visual.state);
        let taxi_position = road_position(state.row, state.column);
        cab.0.translation = taxi_position.extend(4.0);
        cab.1.image = images
            .cabs
            .get(visual.orientation)
            .cloned()
            .expect("fixed example index is valid");

        if state.passenger < 4 {
            let location = Taxi::locations()
                .get(state.passenger)
                .copied()
                .expect("passenger location index is valid");
            passenger.0.translation = road_position(location.0, location.1).extend(3.0);
            *passenger.1 = Visibility::Visible;
        } else {
            *passenger.1 = Visibility::Hidden;
        }

        let destination = Taxi::locations()
            .get(state.destination)
            .copied()
            .expect("destination location index is valid");
        let hotel_position = road_position(destination.0, destination.1) + Vec2::Y * 25.0;
        hotel.0.translation =
            hotel_position.extend(if destination.0 <= state.row { 3.0 } else { 5.0 });
        *hotel.1 = Visibility::Visible;
    }

    /// Capture 100 frames at 20 FPS while advancing the policy at 4 Hz.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualTaxi>,
        images: Res<'_, TaxiImages>,
        mut cab: Single<'_, '_, (&mut Transform, &mut Sprite), With<TaxiCab>>,
        mut passenger: Single<
            '_,
            '_,
            (&mut Transform, &mut Visibility),
            (With<TaxiPassenger>, Without<TaxiCab>),
        >,
        mut hotel: Single<
            '_,
            '_,
            (&mut Transform, &mut Visibility),
            (With<TaxiHotel>, Without<TaxiCab>, Without<TaxiPassenger>),
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
        let state = TaxiState::decode(visual.state);
        let taxi_position = road_position(state.row, state.column);
        cab.0.translation = taxi_position.extend(4.0);
        cab.1.image = images
            .cabs
            .get(visual.orientation)
            .cloned()
            .expect("fixed example index is valid");
        if state.passenger < 4 {
            let location = Taxi::locations()
                .get(state.passenger)
                .copied()
                .expect("passenger location index is valid");
            passenger.0.translation = road_position(location.0, location.1).extend(3.0);
            *passenger.1 = Visibility::Visible;
        } else {
            *passenger.1 = Visibility::Hidden;
        }
        let destination = Taxi::locations()
            .get(state.destination)
            .copied()
            .expect("destination location index is valid");
        hotel.0.translation = (road_position(destination.0, destination.1) + Vec2::Y * 25.0)
            .extend(if destination.0 <= state.row { 3.0 } else { 5.0 });
        *hotel.1 = Visibility::Visible;
        let path = capture.next_path();
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }

    /// Return one 50-pixel map-character cell center.
    const fn map_cell_position(row: usize, column: usize) -> Vec2 {
        Vec2::new(
            (column as f32).mul_add(50.0, -250.0),
            (row as f32).mul_add(-50.0, 150.0),
        )
    }

    /// Return one logical Taxi road-cell center in the 11-by-7 render map.
    const fn road_position(row: usize, column: usize) -> Vec2 {
        Vec2::new(
            (column as f32).mul_add(100.0, -200.0),
            (row as f32).mul_add(-50.0, 100.0),
        )
    }

    /// Render 100 frames and encode five seconds at 20 FPS.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "taxi", 550, 350, |frames| {
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
    fn state_encoding_round_trips_all_five_hundred_states() {
        for encoded in 0..500 {
            assert_eq!(TaxiState::decode(encoded).encode(), encoded);
        }
    }

    #[test]
    fn co_located_pickup_puts_the_passenger_in_the_taxi() {
        let mut taxi = Taxi::default();
        taxi.state = TaxiState {
            row: 0,
            column: 0,
            passenger: 0,
            destination: 1,
        };

        let step = taxi.step(TaxiAction::Pickup);

        assert_eq!(TaxiState::decode(step.observation).passenger, 4);
        assert_eq!(step.reward, -1.0);
        assert_eq!(step.status, EpisodeStatus::Continuing);
    }

    #[test]
    fn correct_dropoff_finishes_with_twenty_reward() {
        let mut taxi = Taxi::default();
        taxi.state = TaxiState {
            row: 4,
            column: 3,
            passenger: 4,
            destination: 3,
        };

        let step = taxi.step(TaxiAction::Dropoff);

        assert_eq!(step.reward, 20.0);
        assert_eq!(step.status, EpisodeStatus::Terminated);
        assert_eq!(TaxiState::decode(step.observation).passenger, 3);
    }

    #[test]
    fn illegal_pickup_has_minus_ten_reward() {
        let mut taxi = Taxi::default();
        taxi.state = TaxiState {
            row: 2,
            column: 2,
            passenger: 0,
            destination: 1,
        };

        let step = taxi.step(TaxiAction::Pickup);

        assert_eq!(step.reward, -10.0);
        assert_eq!(step.status, EpisodeStatus::Continuing);
    }
}
