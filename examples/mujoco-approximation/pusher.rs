//! ## Description
//! "Pusher" is a multi-jointed robot arm that is very similar to a human arm.
//! The goal is to move a target cylinder (called *object*) to a goal position using the robot's end effector (called *fingertip*).
//! The robot consists of shoulder, elbow, forearm and wrist joints.
//!
//!
//! ## Action Space
//! ```{figure} action_space_figures/pusher.png
//! :name: pusher
//! ```
//!
//! The action space is a `Box(-2, 2, (7,), float32)`. An action `(a, b)` represents the torques applied at the hinge joints.
//!
//! | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
//! |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
//! | 0   | Rotation of the panning the shoulder                               | -2          | 2           | `r_shoulder_pan_joint`             | hinge | torque (N m) |
//! | 1   | Rotation of the shoulder lifting joint                             | -2          | 2           | `r_shoulder_lift_joint`            | hinge | torque (N m) |
//! | 2   | Rotation of the shoulder rolling joint                             | -2          | 2           | `r_upper_arm_roll_joint`           | hinge | torque (N m) |
//! | 3   | Rotation of hinge joint that flexed the elbow                      | -2          | 2           | `r_elbow_flex_joint`               | hinge | torque (N m) |
//! | 4   | Rotation of hinge that rolls the forearm                           | -2          | 2           | `r_forearm_roll_joint`             | hinge | torque (N m) |
//! | 5   | Rotation of flexing the wrist                                      | -2          | 2           | `r_wrist_flex_joint`               | hinge | torque (N m) |
//! | 6   | Rotation of rolling the wrist                                      | -2          | 2           | `r_wrist_roll_joint`               | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *qpos (7 elements):* Position values of the robot's body parts.
//! - *qvel (7 elements):* The velocities of these individual body parts (their derivatives).
//! - *xpos (3 elements):* The coordinates of the fingertip of the pusher.
//! - *xpos (3 elements):* The coordinates of the object to be moved.
//! - *xpos (3 elements):* The coordinates of the goal position.
//!
//! The observation space is a `Box(-Inf, Inf, (17,), float64)` where the elements are as follows:
//!
//! | Num | Observation                                              | Min  | Max | Name (in corresponding XML file) | Joint    | Type (Unit)              |
//! | --- | -------------------------------------------------------- | ---- | --- | -------------------------------- | -------- | ------------------------ |
//! | 0   | Rotation of the panning the shoulder                     | -Inf | Inf | `r_shoulder_pan_joint`             | hinge    | angle (rad)              |
//! | 1   | Rotation of the shoulder lifting joint                   | -Inf | Inf | `r_shoulder_lift_joint`            | hinge    | angle (rad)              |
//! | 2   | Rotation of the shoulder rolling joint                   | -Inf | Inf | `r_upper_arm_roll_joint`           | hinge    | angle (rad)              |
//! | 3   | Rotation of hinge joint that flexed the elbow            | -Inf | Inf | `r_elbow_flex_joint`               | hinge    | angle (rad)              |
//! | 4   | Rotation of hinge that rolls the forearm                 | -Inf | Inf | `r_forearm_roll_joint`             | hinge    | angle (rad)              |
//! | 5   | Rotation of flexing the wrist                            | -Inf | Inf | `r_wrist_flex_joint`               | hinge    | angle (rad)              |
//! | 6   | Rotation of rolling the wrist                            | -Inf | Inf | `r_wrist_roll_joint`               | hinge    | angle (rad)              |
//! | 7   | Rotational velocity of the panning the shoulder          | -Inf | Inf | `r_shoulder_pan_joint`             | hinge    | angular velocity (rad/s) |
//! | 8   | Rotational velocity of the shoulder lifting joint        | -Inf | Inf | `r_shoulder_lift_joint`            | hinge    | angular velocity (rad/s) |
//! | 9   | Rotational velocity of the shoulder rolling joint        | -Inf | Inf | `r_upper_arm_roll_joint`           | hinge    | angular velocity (rad/s) |
//! | 10  | Rotational velocity of hinge joint that flexed the elbow | -Inf | Inf | `r_elbow_flex_joint`               | hinge    | angular velocity (rad/s) |
//! | 11  | Rotational velocity of hinge that rolls the forearm      | -Inf | Inf | `r_forearm_roll_joint`             | hinge    | angular velocity (rad/s) |
//! | 12  | Rotational velocity of flexing the wrist                 | -Inf | Inf | `r_wrist_flex_joint`               | hinge    | angular velocity (rad/s) |
//! | 13  | Rotational velocity of rolling the wrist                 | -Inf | Inf | `r_wrist_roll_joint`               | hinge    | angular velocity (rad/s) |
//! | 14  | x-coordinate of the fingertip of the pusher              | -Inf | Inf | `tips_arm`                         | slide    | position (m)             |
//! | 15  | y-coordinate of the fingertip of the pusher              | -Inf | Inf | `tips_arm`                         | slide    | position (m)             |
//! | 16  | z-coordinate of the fingertip of the pusher              | -Inf | Inf | `tips_arm`                         | slide    | position (m)             |
//! | 17  | x-coordinate of the object to be moved                   | -Inf | Inf | object (`obj_slidex`)              | slide    | position (m)             |
//! | 18  | y-coordinate of the object to be moved                   | -Inf | Inf | object (`obj_slidey`)              | slide    | position (m)             |
//! | 19  | z-coordinate of the object to be moved                   | -Inf | Inf | object                           | cylinder | position (m)             |
//! | 20  | x-coordinate of the goal position of the object          | -Inf | Inf | goal (`goal_slidex`)               | slide    | position (m)             |
//! | 21  | y-coordinate of the goal position of the object          | -Inf | Inf | goal (`goal_slidey`)               | slide    | position (m)             |
//! | 22  | z-coordinate of the goal position of the object          | -Inf | Inf | goal                             | sphere   | position (m)             |
//!
//! To understand the state space, an analogy can be drawn to a human arm, where the words "flex" and "roll" have the same meaning as in human joints.
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`reward_dist` + `reward_ctrl` + `reward_near`*.
//!
//! - *`reward_dist`*:
//!   This reward is a measure of how far the object is from the target goal position,
//!   with a more negative value assigned if the object is further away from the target.
//!   It is $-w_{dist} \|(P_{object} - P_{target})\|_2$.
//!   where $w_{dist}$ is the `reward_dist_weight` (default is $1$).
//! - *`reward_ctrl`*:
//!   A negative reward to penalize the pusher for taking actions that are too large.
//!   It is measured as the negative squared Euclidean norm of the action, i.e. as $-w_{control} \|action\|_2^2$.
//!   where $w_{control}$ is the `reward_control_weight` (default is $0.1$).
//! - *`reward_near`*:
//!   This reward is a measure of how far the *fingertip* of the pusher (the unattached end) is from the object,
//!   with a more negative value assigned for when the pusher's *fingertip* is further away from the target.
//!   It is $-w_{near} \|(P_{fingertip} - P_{target})\|_2$.
//!   where $w_{near}$ is the `reward_near_weight` (default is $0.5$).
//!
//! `info` contains the individual reward terms.
//!
//!
//! ## Starting State
//! The initial position state of the Pusher arm is $0_{6}$.
//! The initial position state of the object is $\mathcal{U}_{[[-0.3, -0.2], [0, 0.2]]}$.
//! The position state of the goal is (permanently) $[0.45, -0.05, -0.323]$.
//! The initial velocity state of the Pusher arm is $\mathcal{U}_{[-0.005 \times I_{6}, 0.005 \times I_{6}]}$.
//! The initial velocity state of the object is $`0_2`$.
//! The velocity state of the goal is (permanently) $`0_3`$.
//!
//! where $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//! Note that the initial position state of the object is sampled until its distance to the goal is $ > 0.17 m$.
//!
//! The default frame rate is 5, with each frame lasting 0.01, so *dt = 5 * 0.01 = 0.05*.
//!
//!
//! ## Episode End
//! ### Termination
//! The Pusher never terminates.
//!
//! ### Truncation
//! The default duration of an episode is 100 timesteps.
//!
//!
//! ## Arguments
//! Pusher provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('Pusher-v5', xml_file=...)
//! ```
//!
//! | Parameter               | Type       | Default         |Description                                               |
//! |-------------------------|------------|-----------------|----------------------------------------------------------|
//! | `xml_file`              | **str**    |`"pusher_v5.xml"`| Path to a `MuJoCo` model                                   |
//! | `reward_near_weight`    | **float**  | `0.5`           | Weight for _`reward_near`_ term (see `Rewards` section)    |
//! | `reward_dist_weight`    | **float**  | `1`             | Weight for _`reward_dist`_ term (see `Rewards` section)    |
//! | `reward_control_weight` | **float**  | `0.1`           | Weight for _`reward_control`_ term (see `Rewards` section) |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Fixed bug: increased the density of the object to be higher than air (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/950)).
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Added `xml_file` argument.
//!     - Fixed bug: `reward_distance` & `reward_near` was based on the state before the physics step, now it is based on the state after the physics step (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/821)).
//!     - Added `reward_near_weight`, `reward_dist_weight`, `reward_control_weight` arguments to configure the reward function (defaults are effectively the same as in `v4`).
//!     - Fixed `info["reward_ctrl"]` not being multiplied by the reward weight.
//!     - Added `info["reward_near"]` which is equal to the reward term `reward_near`.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3.
//!     - Warning: This version of the environment is not compatible with `mujoco>=3.0.0` (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/950)).
//! * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks (not including pusher, which has a `max_time_steps` of 100). Added `reward_threshold` to environments.
//! * v0: Initial versions release.

use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy::math::Vec2;
use bevy_gym::training::{RecurrentBehaviorSample, RecurrentPpoConfig};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

use bevy_gym::training::SplitMix64;
use bevy_gym::training::{run_continuous_workflow, ContinuousPpoExample};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use burn as _;
use serde_json as _;

/// `TIME_STEP` used by this example.
const TIME_STEP: f32 = 0.05;
/// `BASE` used by this example.
const BASE: Vec2 = Vec2::new(0.0, -0.6);
/// `UPPER_ARM_LENGTH` used by this example.
const UPPER_ARM_LENGTH: f32 = 0.4;
/// `FOREARM_LENGTH` used by this example.
const FOREARM_LENGTH: f32 = 0.321;
/// `GOAL` used by this example.
const GOAL: Vec2 = Vec2::new(0.45, -0.05);
/// `FINGERTIP_Z` used by this example.
const FINGERTIP_Z: f32 = -0.275;
/// `OBJECT_Z` used by this example.
const OBJECT_Z: f32 = -0.275;
/// `GOAL_Z` used by this example.
const GOAL_Z: f32 = -0.323;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 2.0;
/// `TORQUE_GAIN` used by this example.
const TORQUE_GAIN: f32 = 200.0;
/// `JOINT_DAMPING` used by this example.
const JOINT_DAMPING: f32 = 10.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 100;

/// Gymnasium-compatible Pusher-v5 task with planar contact dynamics.
#[derive(Debug, Clone)]
struct Pusher {
    /// Seven XML joint positions.
    positions: [f32; 7],
    /// Seven XML joint velocities.
    velocities: [f32; 7],
    /// Movable cylinder center on the table.
    object: Vec2,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for Pusher {
    fn default() -> Self {
        Self {
            positions: [0.0; 7],
            velocities: [0.0; 7],
            object: Vec2::new(0.2, -0.05),
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Pusher {
    /// Return the planar elbow and fingertip coordinates.
    fn arm_points(&self) -> (Vec2, Vec2) {
        let shoulder = self.positions[0];
        let elbow_angle = shoulder + self.positions[1];
        let elbow = BASE + Vec2::from_angle(shoulder) * UPPER_ARM_LENGTH;
        let fingertip = elbow + Vec2::from_angle(elbow_angle) * FOREARM_LENGTH;
        (elbow, fingertip)
    }

    /// Return Gymnasium's 23 position, velocity, and body coordinates.
    fn observation(&self) -> Vec<f32> {
        let (_, fingertip) = self.arm_points();
        let mut observation = Vec::with_capacity(23);
        observation.extend(self.positions);
        observation.extend(self.velocities);
        observation.extend([fingertip.x, fingertip.y, FINGERTIP_Z]);
        observation.extend([self.object.x, self.object.y, OBJECT_Z]);
        observation.extend([GOAL.x, GOAL.y, GOAL_Z]);
        observation
    }

    /// Integrate the seven damped controls and one planar fingertip contact.
    fn integrate(&mut self, action: [f32; 7]) {
        let previous_fingertip = self.arm_points().1;
        for index in 0..7 {
            let acceleration = action
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    TORQUE_GAIN,
                    -JOINT_DAMPING
                        * self
                            .velocities
                            .get(index)
                            .copied()
                            .expect("fixed example index is valid"),
                );
            let velocity = self
                .velocities
                .get(index)
                .copied()
                .expect("fixed example index is valid");
            *self
                .velocities
                .get_mut(index)
                .expect("fixed example index is valid") = acceleration.mul_add(TIME_STEP, velocity);
            let velocity = self
                .velocities
                .get(index)
                .copied()
                .expect("fixed example index is valid");
            let position = self
                .positions
                .get_mut(index)
                .expect("fixed example index is valid");
            *position = velocity.mul_add(TIME_STEP, *position);
        }
        let fingertip = self.arm_points().1;
        let contact_distance = previous_fingertip
            .distance(self.object)
            .min(fingertip.distance(self.object));
        let goal_direction = (GOAL - self.object).normalize_or_zero();
        let fingertip_motion = fingertip - previous_fingertip;
        if contact_distance <= 0.09 && fingertip_motion.dot(goal_direction) > 0.0 {
            self.object += fingertip_motion * 0.9;
        }
    }

    /// Return the low-energy staged push controller used for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 7] {
        let shoulder = observation.first().copied().unwrap_or(0.0);
        let elbow = observation.get(1).copied().unwrap_or(0.0);
        let fingertip = Vec2::new(
            observation
                .get(14)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(15)
                .copied()
                .expect("fixed example index is valid"),
        );
        let object = Vec2::new(
            observation
                .get(17)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(18)
                .copied()
                .expect("fixed example index is valid"),
        );
        let goal = Vec2::new(
            observation
                .get(20)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(21)
                .copied()
                .expect("fixed example index is valid"),
        );
        let push_direction = (goal - object).normalize_or_zero();
        let approach = object - push_direction * 0.065;
        let desired_tip = if fingertip.distance(object) < 0.085 {
            goal
        } else {
            approach
        };
        let [desired_shoulder, desired_elbow] = inverse_kinematic_angles(desired_tip);
        let mut action = [0.0; 7];
        action[0] = 0.1f32
            .mul_add(
                angle_difference(desired_shoulder, shoulder),
                -(0.02
                    * observation
                        .get(7)
                        .copied()
                        .expect("fixed example index is valid")),
            )
            .clamp(-ACTION_LIMIT, ACTION_LIMIT);
        action[1] = 0.1f32
            .mul_add(
                angle_difference(desired_elbow, elbow),
                -(0.02
                    * observation
                        .get(8)
                        .copied()
                        .expect("fixed example index is valid")),
            )
            .clamp(-ACTION_LIMIT, ACTION_LIMIT);
        action
    }
}

/// Return the elbow-up planar joint solution for one table coordinate.
fn inverse_kinematic_angles(target: Vec2) -> [f32; 2] {
    let relative = target - BASE;
    let radius_squared = relative.length_squared().clamp(
        (UPPER_ARM_LENGTH - FOREARM_LENGTH).powi(2),
        (UPPER_ARM_LENGTH + FOREARM_LENGTH).powi(2),
    );
    let elbow = (FOREARM_LENGTH.mul_add(
        -FOREARM_LENGTH,
        UPPER_ARM_LENGTH.mul_add(-UPPER_ARM_LENGTH, radius_squared),
    ) / (2.0 * UPPER_ARM_LENGTH * FOREARM_LENGTH))
        .clamp(-1.0, 1.0)
        .acos();
    let shoulder_offset =
        (FOREARM_LENGTH * elbow.sin()).atan2(FOREARM_LENGTH.mul_add(elbow.cos(), UPPER_ARM_LENGTH));
    [relative.y.atan2(relative.x) - shoulder_offset, elbow]
}

impl Env for Pusher {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.positions = [0.0; 7];
        for velocity in &mut self.velocities {
            *velocity = self.rng.f32_between(-0.005, 0.005);
        }
        loop {
            let offset = Vec2::new(
                self.rng.f32_between(-0.3, 0.0),
                self.rng.f32_between(-0.2, 0.2),
            );
            if offset.length() > 0.17 {
                self.object = GOAL + offset;
                break;
            }
        }
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action: [f32; 7] = std::array::from_fn(|index| {
            action
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        });
        self.integrate(action);
        self.elapsed_steps += 1;
        let fingertip = self.arm_points().1;
        let distance = (OBJECT_Z - GOAL_Z)
            .mul_add(OBJECT_Z - GOAL_Z, (self.object - GOAL).length_squared())
            .sqrt();
        let near = (OBJECT_Z - FINGERTIP_Z)
            .mul_add(
                OBJECT_Z - FINGERTIP_Z,
                (self.object - fingertip).length_squared(),
            )
            .sqrt();
        let control = action.iter().map(|value| value * value).sum::<f32>();
        Step {
            observation: self.observation(),
            reward: 0.1f64.mul_add(
                -f64::from(control),
                0.5f64.mul_add(-f64::from(near), -f64::from(distance)),
            ),
            status: if self.elapsed_steps >= MAX_EPISODE_STEPS {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

impl ContinuousPpoExample for Pusher {
    const ENV_NAME: &'static str = "pusher";
    const GYMNASIUM_ID: &'static str = "Pusher-v5";
    const OBSERVATION_DIM: usize = 23;
    const ACTION_LOW: &'static [f32] = &[-2.0; 7];
    const ACTION_HIGH: &'static [f32] = &[2.0; 7];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 100;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 100;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = -30.0;
    const GIF_PATH: &'static str = "docs/images/pusher.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 4_096;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 100;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 64,
            critic_hidden_sizes: vec![128, 64],
            gamma: 0.99,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.001,
            epochs: 4,
            minibatch_sequences: 16,
            initial_log_std: -2.0,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let mut encoded = observation.to_vec();
        let fingertip = Vec2::new(
            observation
                .get(14)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(15)
                .copied()
                .expect("fixed example index is valid"),
        );
        let object = Vec2::new(
            observation
                .get(17)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(18)
                .copied()
                .expect("fixed example index is valid"),
        );
        let goal = Vec2::new(
            observation
                .get(20)
                .copied()
                .expect("fixed example index is valid"),
            observation
                .get(21)
                .copied()
                .expect("fixed example index is valid"),
        );
        let push_direction = (goal - object).normalize_or_zero();
        let desired_tip = if fingertip.distance(object) < 0.085 {
            goal
        } else {
            object - push_direction * 0.065
        };
        let [shoulder, elbow] = inverse_kinematic_angles(desired_tip);
        *encoded.get_mut(0).expect("fixed example index is valid") = angle_difference(
            shoulder,
            observation
                .first()
                .copied()
                .expect("fixed example index is valid"),
        ) / std::f32::consts::PI;
        *encoded.get_mut(1).expect("fixed example index is valid") = angle_difference(
            elbow,
            observation
                .get(1)
                .copied()
                .expect("fixed example index is valid"),
        ) / std::f32::consts::PI;
        for velocity in encoded
            .get_mut(7..14)
            .expect("fixed example range is valid")
        {
            *velocity /= 10.0;
        }
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut demonstrations = Vec::with_capacity(sample_count);
        let mut episode = 0_u64;
        while demonstrations.len() < sample_count {
            let mut env = Self::default();
            let mut observation = env.reset(Some(seed.wrapping_add(episode))).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = Self::expert_action(&observation);
                demonstrations.push(RecurrentBehaviorSample {
                    observation: Self::encode_observation(&observation),
                    action: action.to_vec(),
                });
                if demonstrations.len() == sample_count {
                    break;
                }
                observation = env.step(action.to_vec()).observation;
            }
            episode = episode.wrapping_add(1);
        }
        demonstrations
    }

    fn is_success(_final_observation: &[f32], total_reward: f64, status: EpisodeStatus) -> bool {
        status == EpisodeStatus::Truncated && total_reward >= Self::SOLVED_MEAN_REWARD
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
    run_continuous_workflow::<Pusher>()
}

/// Return the shortest signed angular difference.
fn angle_difference(target: f32, current: f32) -> f32 {
    (target - current + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        ContinuousPpoExample, Env, EpisodeStatus, Error, Path, Pusher, BASE, FOREARM_LENGTH, GOAL,
        TIME_STEP, UPPER_ARM_LENGTH,
    };

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, GifCapture};
    use bevy_gym::training::RecurrentPpoPolicy;
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::BLACK));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualPusher {
        /// Greedy checkpoint policy.
        policy: RecurrentPpoPolicy,
        /// Live environment.
        env: Pusher,
        /// Current exact observation.
        observation: Vec<f32>,
        /// Zero actor memory for this fully observed task.
        memory: bevy_gym::training::RecurrentMemory,
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualClock(Timer);

    #[derive(Component)]
    /// Render state used by this example.
    struct PusherUpperArm;

    #[derive(Component)]
    /// Render state used by this example.
    struct PusherForearm;

    #[derive(Component)]
    /// Render state used by this example.
    struct PusherObject;

    #[derive(Component)]
    /// Render state used by this example.
    struct PusherGoal;

    #[derive(Component)]
    /// Render state used by this example.
    struct PusherArmMarker(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Pusher::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 23, 23, 1, &[-2.0; 7], &[2.0; 7], &config)?;
        let memory = policy.initial_memory();
        let mut env = Pusher::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualPusher {
            policy,
            env,
            observation,
            memory,
        })
        .insert_resource(VisualClock(Timer::from_seconds(
            TIME_STEP,
            TimerMode::Repeating,
        )))
        .insert_resource(GlobalAmbientLight {
            color: Color::WHITE,
            brightness: 300.0,
            affects_lightmapped_meshes: true,
        })
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym Pusher-v5".into(),
                        resolution: WindowResolution::new(480, 480),
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
                .add_systems(Update, capture_gif_frames);
        } else {
            app.add_systems(Update, (advance_watch, sync_scene).chain());
        }
        println!("watching checkpoint={}", checkpoint.display());
        app.run();
        Ok(())
    }

    #[cfg(not(feature = "render"))]
    pub(super) fn run_visual(
        _checkpoint: &Path,
        _capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        Err("visual mode requires the default `render` feature".into())
    }

    /// Execute the `setup_scene` example stage.
    fn setup_scene(
        mut commands: Commands<'_, '_>,
        mut meshes: ResMut<'_, Assets<Mesh>>,
        mut materials: ResMut<'_, Assets<StandardMaterial>>,
    ) {
        commands.spawn((
            Camera3d::default(),
            Transform::from_xyz(0.0, 2.5, -3.72).looking_at(Vec3::new(-0.23, -0.2, -0.4), Vec3::Y),
        ));
        commands.spawn((
            PointLight {
                intensity: 1_400.0,
                range: 8.0,
                shadows_enabled: true,
                ..default()
            },
            Transform::from_xyz(0.0, 3.0, 0.0),
        ));
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(2.0, 2.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.8, 0.8),
                perceptual_roughness: 0.8,
                ..default()
            })),
            Transform::from_xyz(0.0, -0.325, 0.5),
        ));
        let gray = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.55, 0.55),
            metallic: 0.2,
            perceptual_roughness: 0.4,
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.1, 0.6))),
            MeshMaterial3d(gray.clone()),
            Transform::from_xyz(BASE.x, -0.1, BASE.y),
        ));
        for offset in [-0.06, 0.06] {
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(0.05))),
                MeshMaterial3d(gray.clone()),
                Transform::from_xyz(BASE.x + offset, 0.2, BASE.y + 0.05),
            ));
        }
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.06, UPPER_ARM_LENGTH - 0.12))),
            MeshMaterial3d(gray.clone()),
            PusherUpperArm,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.05, FOREARM_LENGTH - 0.1))),
            MeshMaterial3d(gray.clone()),
            PusherForearm,
        ));
        for index in 0..2 {
            commands.spawn((
                Mesh3d(meshes.add(Sphere::new(if index == 0 { 0.065 } else { 0.04 }))),
                MeshMaterial3d(gray.clone()),
                PusherArmMarker(index),
            ));
        }
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.05, 0.1))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                perceptual_roughness: 0.4,
                ..default()
            })),
            PusherObject,
        ));
        commands.spawn((
            Mesh3d(meshes.add(Cylinder::new(0.08, 0.006))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.8, 0.0, 0.0),
                ..default()
            })),
            PusherGoal,
        ));
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualPusher>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualPusher, steps: usize) {
        for _ in 0..steps {
            let encoded = Pusher::encode_observation(&visual.observation);
            let Ok(action) = visual.policy.mean_action(&encoded, &visual.memory) else {
                return;
            };
            visual.memory = visual.policy.initial_memory();
            let transition = visual.env.step(action.action);
            visual.observation = transition.observation;
            if transition.status != EpisodeStatus::Continuing {
                visual.observation = visual.env.reset(None).observation;
            }
        }
    }

    /// Execute the `sync_scene` example stage.
    fn sync_scene(
        visual: Res<'_, VisualPusher>,
        mut upper: Single<'_, '_, &mut Transform, With<PusherUpperArm>>,
        mut forearm: Single<'_, '_, &mut Transform, (With<PusherForearm>, Without<PusherUpperArm>)>,
        mut object: Single<
            '_,
            '_,
            &mut Transform,
            (
                With<PusherObject>,
                Without<PusherUpperArm>,
                Without<PusherForearm>,
            ),
        >,
        mut goal: Single<
            '_,
            '_,
            &mut Transform,
            (
                With<PusherGoal>,
                Without<PusherObject>,
                Without<PusherUpperArm>,
                Without<PusherForearm>,
            ),
        >,
        mut markers: Query<
            '_,
            '_,
            (&PusherArmMarker, &mut Transform),
            (
                Without<PusherUpperArm>,
                Without<PusherForearm>,
                Without<PusherObject>,
                Without<PusherGoal>,
            ),
        >,
    ) {
        apply_visual_state(
            &visual.env,
            &mut upper,
            &mut forearm,
            &mut object,
            &mut goal,
            &mut markers,
        );
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        env: &Pusher,
        upper: &mut Transform,
        forearm: &mut Transform,
        object: &mut Transform,
        goal: &mut Transform,
        markers: &mut Query<
            '_,
            '_,
            (&PusherArmMarker, &mut Transform),
            (
                Without<PusherUpperArm>,
                Without<PusherForearm>,
                Without<PusherObject>,
                Without<PusherGoal>,
            ),
        >,
    ) {
        let (elbow, fingertip) = env.arm_points();
        let rendered_base = Vec2::new(-BASE.x, BASE.y);
        let rendered_elbow = Vec2::new(-elbow.x, elbow.y);
        let rendered_fingertip = Vec2::new(-fingertip.x, fingertip.y);
        *upper = segment_transform(rendered_base, rendered_elbow, -0.24);
        *forearm = segment_transform(rendered_elbow, rendered_fingertip, -0.24);
        object.translation = Vec3::new(-env.object.x, -0.225, env.object.y);
        goal.translation = Vec3::new(-GOAL.x, -0.319, GOAL.y);
        for (marker, mut transform) in markers.iter_mut() {
            let position = if marker.0 == 0 {
                rendered_elbow
            } else {
                rendered_fingertip
            };
            transform.translation = Vec3::new(position.x, -0.24, position.y);
        }
    }

    /// Execute the `segment_transform` example stage.
    fn segment_transform(start: Vec2, end: Vec2, height: f32) -> Transform {
        let start = Vec3::new(start.x, height, start.y);
        let end = Vec3::new(end.x, height, end.y);
        let direction = end - start;
        Transform {
            translation: (start + end) * 0.5,
            rotation: Quat::from_rotation_arc(Vec3::Y, direction.normalize()),
            ..default()
        }
    }

    /// Execute the `capture_gif_frames` example stage.
    fn capture_gif_frames(
        mut commands: Commands<'_, '_>,
        mut capture: ResMut<'_, GifCapture>,
        mut visual: ResMut<'_, VisualPusher>,
        mut upper: Single<'_, '_, &mut Transform, With<PusherUpperArm>>,
        mut forearm: Single<'_, '_, &mut Transform, (With<PusherForearm>, Without<PusherUpperArm>)>,
        mut object: Single<
            '_,
            '_,
            &mut Transform,
            (
                With<PusherObject>,
                Without<PusherUpperArm>,
                Without<PusherForearm>,
            ),
        >,
        mut goal: Single<
            '_,
            '_,
            &mut Transform,
            (
                With<PusherGoal>,
                Without<PusherObject>,
                Without<PusherUpperArm>,
                Without<PusherForearm>,
            ),
        >,
        mut markers: Query<
            '_,
            '_,
            (&PusherArmMarker, &mut Transform),
            (
                Without<PusherUpperArm>,
                Without<PusherForearm>,
                Without<PusherObject>,
                Without<PusherGoal>,
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
        if capture.has_started() {
            advance_visual(&mut visual, 5);
        }
        apply_visual_state(
            &visual.env,
            &mut upper,
            &mut forearm,
            &mut object,
            &mut goal,
            &mut markers,
        );
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(capture.next_path()));
    }

    /// Execute the `render_gif` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn render_gif(checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        encode_gif(output, "pusher", 480, 480, |frames| {
            run_visual(checkpoint, Some(frames))
        })
    }

    #[cfg(not(feature = "render"))]
    pub(super) fn render_gif(_checkpoint: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
        Err("GIF mode requires the default `render` feature".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_has_the_reference_width_and_body_coordinates() {
        let pusher = Pusher::default();
        let observation = pusher.observation();

        assert_eq!(observation.len(), 23);
        assert_eq!(&observation[17..20], &[0.2, -0.05, OBJECT_Z]);
        assert_eq!(&observation[20..23], &[GOAL.x, GOAL.y, GOAL_Z]);
    }

    #[test]
    fn reward_is_strictly_nonpositive() {
        let mut pusher = Pusher::default();
        pusher.object = GOAL;

        let step = pusher.step(vec![0.0; 7]);

        assert!(step.reward <= 0.0);
    }

    #[test]
    fn expert_controller_moves_the_object_toward_the_goal() {
        let mut pusher = Pusher::default();
        let mut observation = pusher.reset(Some(42)).observation;
        let initial_distance = pusher.object.distance(GOAL);
        for _ in 0..MAX_EPISODE_STEPS {
            let action = Pusher::expert_action(&observation);
            observation = pusher.step(action.to_vec()).observation;
        }

        assert!(
            pusher.object.distance(GOAL) < initial_distance,
            "initial={initial_distance:.3} final={:.3}",
            pusher.object.distance(GOAL)
        );
    }

    #[test]
    fn expert_controller_return_is_near_the_reward_upper_bound() {
        let episodes = 100_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut pusher = Pusher::default();
            let mut observation = pusher.reset(Some(seed)).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = Pusher::expert_action(&observation);
                let step = pusher.step(action.to_vec());
                total_reward += step.reward;
                observation = step.observation;
            }
        }
        let mean_reward = total_reward / episodes as f64;

        assert!(mean_reward >= -30.0, "expert mean reward={mean_reward:.3}");
    }
}
