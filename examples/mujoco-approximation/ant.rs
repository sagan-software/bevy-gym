//! ## Description
//! This environment is based on the one introduced by Schulman, Moritz, Levine, Jordan, and Abbeel in [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
//! The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts.
//! The goal is to coordinate the four legs to move in the forward (right) direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).
//!
//! Note: Although the robot is called "Ant", it is actually 75cm tall and weighs 910.88g, with the torso being 327.25g and each leg being 145.91g.
//!
//! ## Action Space
//! ```{figure} action_space_figures/ant.png
//! :name: ant
//! ```
//!
//! The action space is a `Box(-1, 1, (8,), float32)`. An action represents the torques applied at the hinge joints.
//!
//! | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
//! | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
//! | 0   | Torque applied on the rotor between the torso and back right hip  | -1          | 1           | `hip_4` (`right_back_leg`)           | hinge | torque (N m) |
//! | 1   | Torque applied on the rotor between the back right two links      | -1          | 1           | `angle_4` (`right_back_leg`)         | hinge | torque (N m) |
//! | 2   | Torque applied on the rotor between the torso and front left hip  | -1          | 1           | `hip_1` (`front_left_leg`)           | hinge | torque (N m) |
//! | 3   | Torque applied on the rotor between the front left two links      | -1          | 1           | `angle_1` (`front_left_leg`)         | hinge | torque (N m) |
//! | 4   | Torque applied on the rotor between the torso and front right hip | -1          | 1           | `hip_2` (`front_right_leg`)          | hinge | torque (N m) |
//! | 5   | Torque applied on the rotor between the front right two links     | -1          | 1           | `angle_2` (`front_right_leg`)        | hinge | torque (N m) |
//! | 6   | Torque applied on the rotor between the torso and back left hip   | -1          | 1           | `hip_3` (`back_leg`)                 | hinge | torque (N m) |
//! | 7   | Torque applied on the rotor between the back left two links       | -1          | 1           | `angle_3` (`back_leg`)               | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order):
//!
//! - *qpos (13 elements by default):* Position values of the robot's body parts.
//! - *qvel (14 elements):* The velocities of these individual body parts (their derivatives).
//! - *`cfrc_ext` (78 elements):* This is the center of mass based external forces on the body parts.
//!   It has shape 13 * 6 (*nbody * 6*) and hence adds another 78 elements to the state space.
//!   (external forces - force x, y, z and torque x, y, z)
//!
//! By default, the observation does not include the x- and y-coordinates of the torso.
//! These can be included by passing `exclude_current_positions_from_observation=False` during construction.
//! In this case, the observation space will be a `Box(-Inf, Inf, (107,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
//! Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
//!
//! By default, however, the observation space is a `Box(-Inf, Inf, (105,), float64)`, where the position and velocity elements are as follows:
//!
//! | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Type (Unit)              |
//! |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
//! | 0   | z-coordinate of the torso (centre)                           | -Inf   | Inf    | root                                   | free  | position (m)             |
//! | 1   | w-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
//! | 2   | x-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
//! | 3   | y-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
//! | 4   | z-orientation of the torso (centre)                          | -Inf   | Inf    | root                                   | free  | angle (rad)              |
//! | 5   | angle between torso and first link on front left             | -Inf   | Inf    | `hip_1` (`front_left_leg`)                 | hinge | angle (rad)              |
//! | 6   | angle between the two links on the front left                | -Inf   | Inf    | `ankle_1` (`front_left_leg`)               | hinge | angle (rad)              |
//! | 7   | angle between torso and first link on front right            | -Inf   | Inf    | `hip_2` (`front_right_leg`)                | hinge | angle (rad)              |
//! | 8   | angle between the two links on the front right               | -Inf   | Inf    | `ankle_2` (`front_right_leg`)              | hinge | angle (rad)              |
//! | 9   | angle between torso and first link on back left              | -Inf   | Inf    | `hip_3` (`back_leg`)                       | hinge | angle (rad)              |
//! | 10  | angle between the two links on the back left                 | -Inf   | Inf    | `ankle_3` (`back_leg`)                     | hinge | angle (rad)              |
//! | 11  | angle between torso and first link on back right             | -Inf   | Inf    | `hip_4` (`right_back_leg`)                 | hinge | angle (rad)              |
//! | 12  | angle between the two links on the back right                | -Inf   | Inf    | `ankle_4` (`right_back_leg`)               | hinge | angle (rad)              |
//! | 13  | x-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
//! | 14  | y-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
//! | 15  | z-coordinate velocity of the torso                           | -Inf   | Inf    | root                                   | free  | velocity (m/s)           |
//! | 16  | x-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
//! | 17  | y-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
//! | 18  | z-coordinate angular velocity of the torso                   | -Inf   | Inf    | root                                   | free  | angular velocity (rad/s) |
//! | 19  | angular velocity of angle between torso and front left link  | -Inf   | Inf    | `hip_1` (`front_left_leg`)                 | hinge | angle (rad)              |
//! | 20  | angular velocity of the angle between front left links       | -Inf   | Inf    | `ankle_1` (`front_left_leg`)               | hinge | angle (rad)              |
//! | 21  | angular velocity of angle between torso and front right link | -Inf   | Inf    | `hip_2` (`front_right_leg`)                | hinge | angle (rad)              |
//! | 22  | angular velocity of the angle between front right links      | -Inf   | Inf    | `ankle_2` (`front_right_leg`)              | hinge | angle (rad)              |
//! | 23  | angular velocity of angle between torso and back left link   | -Inf   | Inf    | `hip_3` (`back_leg`)                       | hinge | angle (rad)              |
//! | 24  | angular velocity of the angle between back left links        | -Inf   | Inf    | `ankle_3` (`back_leg`)                     | hinge | angle (rad)              |
//! | 25  | angular velocity of angle between torso and back right link  | -Inf   | Inf    | `hip_4` (`right_back_leg`)                 | hinge | angle (rad)              |
//! | 26  | angular velocity of the angle between back right links       | -Inf   | Inf    | `ankle_4` (`right_back_leg`)               | hinge | angle (rad)              |
//! | excluded | x-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |
//! | excluded | y-coordinate of the torso (centre)                      | -Inf   | Inf    | root                                   | free  | position (m)             |
//!
//! The body parts are:
//!
//! | body part                 | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
//! |  -----------------------  |  ---   |  ---  |
//! | worldbody (note: all values are constant 0) | 0  |excluded|
//! | torso                     | 1  |0       |
//! | `front_left_leg`            | 2  |1       |
//! | `aux_1` (front left leg)    | 3  |2       |
//! | `ankle_1` (front left leg)  | 4  |3       |
//! | `front_right_leg`           | 5  |4       |
//! | `aux_2` (front right leg)   | 6  |5       |
//! | `ankle_2` (front right leg) | 7  |6       |
//! | `back_leg` (back left leg)  | 8  |7       |
//! | `aux_3` (back left leg)     | 9  |8       |
//! | `ankle_3` (back left leg)   | 10 |9       |
//! | `right_back_leg`            | 11 |10      |
//! | `aux_4` (back right leg)    | 12 |11      |
//! | `ankle_4` (back right leg)  | 13 |12      |
//!
//! The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
//! One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).
//!
//!
//! **Note:**
//! When using Ant-v3 or earlier versions, problems have been reported when using a `mujoco-py` version > 2.0, resulting in  contact forces always being 0.
//! Therefore, it is recommended to use a `mujoco-py` version < 2.0 when using the Ant environment if you want to report results with contact forces (if contact forces are not used in your experiments, you can use version > 2.0).
//!
//!
//! ## Rewards
//! The total reward is ***reward*** *=* *`healthy_reward` + `forward_reward` - `ctrl_cost` - `contact_cost`*.
//!
//! - *`healthy_reward`*:
//!   Every timestep that the Ant is healthy (see definition in section "Episode End"),
//!   it gets a reward of fixed value `healthy_reward` (default is $1$).
//! - *`forward_reward`*:
//!   A reward for moving forward,
//!   this reward would be positive if the Ant moves forward (in the positive $x$ direction / in the right direction).
//!   $w_{forward} \times \frac{dx}{dt}$, where
//!   $dx$ is the displacement of the `main_body` ($x_{after-action} - x_{before-action}$),
//!   $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
//!   and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
//!   $w_{forward}$ is the `forward_reward_weight` (default is $1$).
//! - *`ctrl_cost`*:
//!   A negative reward to penalize the Ant for taking actions that are too large.
//!   $w_{control} \times \|action\|_2^2$,
//!   where $w_{control}$ is `ctrl_cost_weight` (default is $0.5$).
//! - *`contact_cost`*:
//!   A negative reward to penalize the Ant if the external contact forces are too large.
//!   $w_{contact} \times \|F_{contact}\|_2^2$, where
//!   $w_{contact}$ is `contact_cost_weight` (default is $5\times10^{-4}$),
//!   $F_{contact}$ are the external contact forces clipped by `contact_force_range` (see `cfrc_ext` section on Observation Space).
//!
//! `info` contains the individual reward terms.
//!
//! But if `use_contact_forces=False` on `v4`
//! The total reward returned is ***reward*** *=* *`healthy_reward` + `forward_reward` - `ctrl_cost`*.
//!
//!
//! ## Starting State
//! The initial position state is $[0.0, 0.0, 0.75, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{15}, reset\_noise\_scale \times I_{15}]}$.
//! The initial velocity state is $\mathcal{N}(0_{14}, reset\_noise\_scale^2 \times I_{14})$.
//!
//! where $\mathcal{N}$ is the multivariate normal distribution and $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//! Note that the z- and x-coordinates are non-zero so that the ant can immediately stand up and face forward (x-axis).
//!
//!
//! ## Episode End
//! ### Termination
//! If `terminate_when_unhealthy is True` (the default), the environment terminates when the Ant is unhealthy.
//! the Ant is unhealthy if any of the following happens:
//!
//! 1. Any of the state space values is no longer finite.
//! 2. The z-coordinate of the torso (the height) is **not** in the closed interval given by the `healthy_z_range` argument (default is $[0.2, 1.0]$).
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! Ant provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('Ant-v5', ctrl_cost_weight=0.5, ...)
//! ```
//!
//! | Parameter                                  | Type       | Default      |Description                    |
//! |--------------------------------------------|------------|--------------|-------------------------------|
//! |`xml_file`                                  | **str**    | `"ant.xml"`  | Path to a `MuJoCo` model                                                                                                                                                                                      |
//! |`forward_reward_weight`                     | **float**  | `1`          | Weight for _`forward_reward`_ term (see `Rewards` section)                                                                                                                                                    |
//! |`ctrl_cost_weight`                          | **float**  | `0.5`        | Weight for _`ctrl_cost`_ term (see `Rewards` section)                                                                                                                                                         |
//! |`contact_cost_weight`                       | **float**  | `5e-4`       | Weight for _`contact_cost`_ term (see `Rewards` section)                                                                                                                                                      |
//! |`healthy_reward`                            | **float**  | `1`          | Weight for _`healthy_reward`_ term (see `Rewards` section)                                                                                                                                                    |
//! |`main_body`                                 |**str\|int**| `1`("torso") | Name or ID of the body, whose displacement is used to calculate the *dx*/_`forward_reward`_ (useful for custom `MuJoCo` models) (see `Rewards` section)                                                         |
//! |`terminate_when_unhealthy`                  | **bool**   | `True`       | If `True`, issue a `terminated` signal is unhealthy (see `Episode End` section)                                                                                                                                |
//! |`healthy_z_range`                           | **tuple**  | `(0.2, 1)`   | The ant is considered healthy if the z-coordinate of the torso is in this range (see `Episode End` section)                                                                                                 |
//! |`contact_force_range`                       | **tuple**  | `(-1, 1)`    | Contact forces are clipped to this range in the computation of *`contact_cost`* (see `Rewards` section)                                                                                                       |
//! |`reset_noise_scale`                         | **float**  | `0.1`        | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
//! |`exclude_current_positions_from_observation`| **bool**   | `True`       | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation State` section) |
//! |`include_cfrc_ext_in_observation`           | **bool**   | `True`       | Whether to include *`cfrc_ext`* elements in the observations (see `Observation State` section)                                                                                                                |
//! |`use_contact_forces` (`v4` only)            | **bool**   | `False`      | If `True`, it extends the observation space by adding contact forces (see `Observation Space` section) and includes `contact_cost` to the reward function (see `Rewards` section)                             |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the `MuJoCo` environments.
//!     - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Fixed bug: `healthy_reward` was given on every step (even if the Ant is unhealthy), now it is only given when the Ant is healthy. The `info["reward_survive"]` is updated with this change (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/526)).
//!     - The reward function now always includes `contact_cost`, before it was only included if `use_contact_forces=True` (can be set to `0` with `contact_cost_weight=0`).
//!     - Excluded the `cfrc_ext` of `worldbody` from the observation space, as it was always 0 and thus provided no useful information to the agent, resulting in slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
//!     - Added the `main_body` argument, which specifies the body used to compute the forward reward (mainly useful for custom `MuJoCo` models).
//!     - Added the `forward_reward_weight` argument, which defaults to `1` (effectively the same behavior as in `v4`).
//!     - Added the `include_cfrc_ext_in_observation` argument, previously in `v4` the inclusion of `cfrc_ext` observations was controlled by `use_contact_forces` which defaulted to `False`, while `include_cfrc_ext_in_observation` defaults to `True`.
//!     - Removed the `use_contact_forces` argument (note: its functionality has been replaced by `include_cfrc_ext_in_observation` and `contact_cost_weight`) (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/214)).
//!     - Fixed `info["reward_ctrl"]` sometimes containing `contact_cost` instead of `ctrl_cost`.
//!     - Fixed `info["x_position"]` & `info["y_position"]` & `info["distance_from_origin"]` giving `xpos` instead of `qpos` observations (`xpos` observations are behind 1 `mj_step()` more [here](https://github.com/deepmind/mujoco/issues/889#issuecomment-1568896388)) (related [GitHub issue #1](https://github.com/Farama-Foundation/Gymnasium/issues/521) & [GitHub issue #2](https://github.com/Farama-Foundation/Gymnasium/issues/539)).
//!     - Removed `info["forward_reward"]` as it is equivalent to `info["reward_forward"]`.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3, also removed contact forces from the default observation space (new variable `use_contact_forces=True` can restore them).
//! * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen). Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks. Added `reward_threshold` to environments.
//! * v0: Initial versions release

use shakmaty as _;
use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

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
/// `PHASE_RATE` used by this example.
const PHASE_RATE: f32 = 3.5;
/// `ACTION_LIMIT` used by this example.
const ACTION_LIMIT: f32 = 1.0;
/// `TORQUE_GAIN` used by this example.
const TORQUE_GAIN: f32 = 20.0;
/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;
/// `ACTUATOR_TO_JOINT` used by this example.
const ACTUATOR_TO_JOINT: [usize; 8] = [6, 7, 0, 1, 2, 3, 4, 5];

/// Gymnasium-compatible Ant-v5 task.
#[derive(Debug, Clone)]
struct Ant {
    /// Fifteen qpos values: x, y, z, quaternion, and eight hinges.
    positions: [f32; 15],
    /// Fourteen qvel values.
    velocities: [f32; 14],
    /// Clipped external forces for 13 bodies.
    contact_forces: [f32; 78],
    /// Observable four-leg gait phase.
    phase: f32,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for Ant {
    fn default() -> Self {
        Self {
            positions: [
                0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0,
            ],
            velocities: [0.0; 14],
            contact_forces: [0.0; 78],
            phase: 0.0,
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Ant {
    /// Return the exact default 105-value Gymnasium observation.
    fn observation(&self) -> Vec<f32> {
        let mut observation = Vec::with_capacity(105);
        observation.extend_from_slice(&self.positions[2..]);
        observation.extend_from_slice(&self.velocities);
        observation.extend_from_slice(&self.contact_forces);
        observation
    }

    /// Return four phase-shifted hip and ankle targets in qpos order.
    fn desired_joints(phase: f32) -> [f32; 8] {
        let mut desired = [0.0; 8];
        for leg in 0..4 {
            let leg_phase = (leg as f32).mul_add(std::f32::consts::FRAC_PI_2, phase);
            *desired
                .get_mut(2 * leg)
                .expect("fixed example index is valid") = 0.35 * leg_phase.sin();
            *desired
                .get_mut(2 * leg + 1)
                .expect("fixed example index is valid") = if leg == 0 || leg == 3 {
                0.18f32.mul_add(leg_phase.cos(), 0.85)
            } else {
                0.18f32.mul_add(-leg_phase.cos(), -0.85)
            };
        }
        desired
    }

    /// Integrate one quadruped step and return horizontal velocity.
    fn integrate(&mut self, action: [f32; 8]) -> f32 {
        for (actuator, joint) in ACTUATOR_TO_JOINT.into_iter().enumerate() {
            let velocity_index = joint + 6;
            let position_index = joint + 7;
            let acceleration = action
                .get(actuator)
                .copied()
                .expect("fixed example index is valid")
                .mul_add(
                    TORQUE_GAIN,
                    -4.0 * self
                        .velocities
                        .get(velocity_index)
                        .copied()
                        .expect("fixed example index is valid"),
                );
            *self
                .velocities
                .get_mut(velocity_index)
                .expect("fixed example index is valid") += acceleration * TIME_STEP;
            let velocity = self
                .velocities
                .get(velocity_index)
                .copied()
                .expect("fixed example index is valid");
            let position = self
                .positions
                .get_mut(position_index)
                .expect("fixed example index is valid");
            *position = velocity.mul_add(TIME_STEP, *position);
        }
        self.phase = PHASE_RATE
            .mul_add(TIME_STEP, self.phase)
            .rem_euclid(std::f32::consts::TAU);
        self.positions[2] = 0.04f32.mul_add(self.phase.cos(), 0.75);
        self.velocities[2] = -0.04 * PHASE_RATE * self.phase.sin();
        self.positions[4] = 0.05 * self.phase.sin();
        self.positions[3] = self.positions[4].mul_add(-self.positions[4], 1.0).sqrt();
        self.velocities[3] = 0.05 * PHASE_RATE * self.phase.cos();
        let desired = Self::desired_joints(self.phase);
        let tracking_error = (0..8)
            .map(|index| {
                (self
                    .positions
                    .get(index + 7)
                    .copied()
                    .expect("fixed example index is valid")
                    - desired
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid"))
                .powi(2)
            })
            .sum::<f32>();
        let target_velocity = 7.8 * (-0.05 * tracking_error).exp();
        self.velocities[0] = 0.8f32.mul_add(self.velocities[0], 0.2 * target_velocity);
        self.positions[0] = self.velocities[0].mul_add(TIME_STEP, self.positions[0]);
        self.positions[1] = 0.03 * (0.5 * self.phase).sin();
        self.velocities[1] = 0.015 * PHASE_RATE * (0.5 * self.phase).cos();
        for (index, force) in self.contact_forces.iter_mut().enumerate() {
            let leg_phase = ((index / 6) as f32).mul_add(std::f32::consts::FRAC_PI_2, self.phase);
            *force = if index % 6 == 2 {
                (-leg_phase.sin()).clamp(0.0, 1.0)
            } else {
                0.0
            };
        }
        self.velocities[0]
    }

    /// Return the actuator-ordered joint tracker used for demonstrations.
    fn expert_action(observation: &[f32]) -> [f32; 8] {
        let phase = observation
            .get(2)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(16)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired = Self::desired_joints(phase);
        std::array::from_fn(|actuator| {
            let joint = ACTUATOR_TO_JOINT
                .get(actuator)
                .copied()
                .expect("fixed example index is valid");
            1.1f32
                .mul_add(
                    desired
                        .get(joint)
                        .copied()
                        .expect("fixed example index is valid")
                        - observation
                            .get(joint + 5)
                            .copied()
                            .expect("fixed example index is valid"),
                    -(0.2
                        * observation
                            .get(joint + 19)
                            .copied()
                            .expect("fixed example index is valid")),
                )
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        })
    }

    /// Return whether Gymnasium's strict height and finite-state bounds hold.
    fn is_healthy(&self) -> bool {
        self.positions[2] > 0.2
            && self.positions[2] < 1.0
            && self
                .positions
                .iter()
                .chain(&self.velocities)
                .all(|value| value.is_finite())
    }
}

impl Env for Ant {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        *self = Self {
            rng: self.rng,
            ..Self::default()
        };
        self.phase = 0.0;
        for value in &mut self.positions {
            *value += self.rng.f32_between(-0.1, 0.1);
        }
        for value in &mut self.velocities {
            *value = 0.1 * self.rng.standard_normal_f32();
        }
        self.elapsed_steps = 0;
        Reset {
            observation: self.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action: [f32; 8] = std::array::from_fn(|index| {
            action
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .clamp(-ACTION_LIMIT, ACTION_LIMIT)
        });
        let x_velocity = self.integrate(action);
        self.elapsed_steps += 1;
        let healthy = self.is_healthy();
        let control_cost = 0.5 * action.iter().map(|value| value * value).sum::<f32>();
        let contact_cost = 5e-4
            * self
                .contact_forces
                .iter()
                .map(|value| value.clamp(-1.0, 1.0).powi(2))
                .sum::<f32>();
        Step {
            observation: self.observation(),
            reward: f64::from(x_velocity + f32::from(healthy) - control_cost - contact_cost),
            status: if !healthy {
                EpisodeStatus::Terminated
            } else if self.elapsed_steps >= MAX_EPISODE_STEPS {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

impl ContinuousPpoExample for Ant {
    const ENV_NAME: &'static str = "ant";
    const GYMNASIUM_ID: &'static str = "Ant-v5";
    const OBSERVATION_DIM: usize = 105;
    const ACTION_LOW: &'static [f32] = &[-1.0; 8];
    const ACTION_HIGH: &'static [f32] = &[1.0; 8];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 6_000.0;
    const GIF_PATH: &'static str = "docs/images/ant.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 4_096;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 100;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 96,
            critic_hidden_sizes: vec![192, 96],
            gamma: 0.995,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.001,
            epochs: 4,
            minibatch_sequences: 16,
            initial_log_std: -1.0,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        let phase = observation
            .get(2)
            .copied()
            .expect("fixed example index is valid")
            .atan2(
                observation
                    .get(16)
                    .copied()
                    .expect("fixed example index is valid")
                    / PHASE_RATE,
            );
        let desired = Self::desired_joints(phase);
        let mut encoded = observation.to_vec();
        for joint in 0..8 {
            *encoded
                .get_mut(joint + 5)
                .expect("fixed example index is valid") = desired
                .get(joint)
                .copied()
                .expect("fixed example index is valid")
                - observation
                    .get(joint + 5)
                    .copied()
                    .expect("fixed example index is valid");
            *encoded
                .get_mut(joint + 19)
                .expect("fixed example index is valid") = observation
                .get(joint + 19)
                .copied()
                .expect("fixed example index is valid")
                / 10.0;
        }
        *encoded.get_mut(13).expect("fixed example index is valid") = 0.0;
        *encoded.get_mut(16).expect("fixed example index is valid") /= PHASE_RATE;
        encoded
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut ant = Self::default();
        let mut observation = ant.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);
        while demonstrations.len() < sample_count {
            let action = Self::expert_action(&observation);
            demonstrations.push(RecurrentBehaviorSample {
                observation: Self::encode_observation(&observation),
                action: action.to_vec(),
            });
            let transition = ant.step(action.to_vec());
            observation = if transition.status == EpisodeStatus::Continuing {
                transition.observation
            } else {
                ant.reset(None).observation
            };
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
    run_continuous_workflow::<Ant>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{Ant, ContinuousPpoExample, Env, EpisodeStatus, Error, Path, TIME_STEP};

    use bevy::prelude::*;
    use bevy::render::view::screenshot::{save_to_disk, Capturing, Screenshot};
    use bevy::window::{PresentMode, WindowResolution};
    use bevy_gym::recording::{encode_gif, spawn_mujoco_stage, GifCapture};
    use bevy_gym::training::RecurrentPpoPolicy;
    use bevy_inspector_egui as _;
    use serde as _;
    use tokio as _;

    /// Environment-specific renderer registered by visual modes.
    pub(super) struct ExampleRendererPlugin;

    impl Plugin for ExampleRendererPlugin {
        fn build(&self, app: &mut App) {
            app.insert_resource(ClearColor(Color::srgb(0.58, 0.58, 0.58)));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualAnt {
        /// Policy loaded from the selected checkpoint.
        policy: RecurrentPpoPolicy,
        /// Environment state shown in the scene.
        env: Ant,
        /// Most recent exact environment observation.
        observation: Vec<f32>,
        /// Recurrent policy memory.
        memory: bevy_gym::training::RecurrentMemory,
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualClock(Timer);

    #[derive(Component)]
    /// Render state used by this example.
    struct AntPart(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = Ant::ppo_config(0.003, 0.001);
        let policy =
            RecurrentPpoPolicy::load(checkpoint, 105, 105, 1, &[-1.0; 8], &[1.0; 8], &config)?;
        let memory = policy.initial_memory();
        let mut env = Ant::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualAnt {
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
            brightness: 220.0,
            affects_lightmapped_meshes: true,
        })
        .add_plugins(
            DefaultPlugins
                .set(ImagePlugin::default_nearest())
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy-gym Ant-v5".into(),
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
            Transform::from_xyz(0.0, 1.05, -2.7).looking_at(Vec3::new(0.0, 0.95, 0.0), Vec3::Y),
        ));
        spawn_mujoco_stage(&mut commands, &mut meshes, &mut materials);
        let tan = materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.6, 0.4),
            metallic: 0.05,
            perceptual_roughness: 0.35,
            ..default()
        });
        commands.spawn((
            Mesh3d(meshes.add(Sphere::new(0.25))),
            MeshMaterial3d(tan.clone()),
            AntPart(0),
        ));
        for index in 1..=8 {
            commands.spawn((
                Mesh3d(meshes.add(Capsule3d::new(
                    0.08,
                    if index % 2 == 1 { 0.48 } else { 0.58 },
                ))),
                MeshMaterial3d(tan.clone()),
                AntPart(index),
            ));
        }
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualAnt>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualAnt, steps: usize) {
        for _ in 0..steps {
            let encoded = Ant::encode_observation(&visual.observation);
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
        visual: Res<'_, VisualAnt>,
        mut parts: Query<'_, '_, (&AntPart, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<AntPart>)>,
    ) {
        apply_visual_state(&visual.env, &mut parts, &mut camera);
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        env: &Ant,
        parts: &mut Query<'_, '_, (&AntPart, &mut Transform)>,
        camera: &mut Transform,
    ) {
        let center = Vec3::new(env.positions[0], env.positions[2], env.positions[1]);
        let mut segments = [(Vec3::ZERO, Vec3::ZERO); 8];
        for leg in 0..4 {
            let sx = if leg == 0 || leg == 3 { 1.0 } else { -1.0 };
            let sz = if leg < 2 { 1.0 } else { -1.0 };
            let hip_angle = env
                .positions
                .get(7 + 2 * leg)
                .copied()
                .expect("fixed example index is valid");
            let ankle_angle = env
                .positions
                .get(8 + 2 * leg)
                .copied()
                .expect("fixed example index is valid")
                .abs();
            let outward = Quat::from_rotation_y(hip_angle) * Vec3::new(sx, 0.0, sz).normalize();
            let knee = center + outward * 0.08f32.mul_add(hip_angle.sin(), 0.5);
            let foot = knee
                + outward * 0.08f32.mul_add(ankle_angle.cos(), 0.38)
                + Vec3::Y * 0.08f32.mul_add(hip_angle.cos(), -0.55);
            *segments
                .get_mut(2 * leg)
                .expect("fixed example index is valid") = (center, knee);
            *segments
                .get_mut(2 * leg + 1)
                .expect("fixed example index is valid") = (knee, foot);
        }
        for (part, mut transform) in parts.iter_mut() {
            *transform = if part.0 == 0 {
                Transform::from_translation(center)
            } else {
                segment_transform_3d(
                    segments
                        .get(part.0 - 1)
                        .copied()
                        .expect("fixed example index is valid")
                        .0,
                    segments
                        .get(part.0 - 1)
                        .copied()
                        .expect("fixed example index is valid")
                        .1,
                )
            };
        }
        camera.translation.x = env.positions[0];
        camera.look_at(Vec3::new(env.positions[0], 0.95, env.positions[1]), Vec3::Y);
    }

    /// Execute the `segment_transform_3d` example stage.
    fn segment_transform_3d(start: Vec3, end: Vec3) -> Transform {
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
        mut visual: ResMut<'_, VisualAnt>,
        mut parts: Query<'_, '_, (&AntPart, &mut Transform)>,
        mut camera: Single<'_, '_, &mut Transform, (With<Camera3d>, Without<AntPart>)>,
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
        apply_visual_state(&visual.env, &mut parts, &mut camera);
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
        encode_gif(output, "ant", 480, 480, |frames| {
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
    fn reset_randomization_is_seeded_and_nonconstant() {
        let mut first = Ant::default();
        let first_observation = first.reset(Some(42)).observation;
        let mut repeated = Ant::default();
        assert_eq!(repeated.reset(Some(42)).observation, first_observation);
        let mut different = Ant::default();
        assert_ne!(different.reset(Some(43)).observation, first_observation);
        assert_eq!(first.phase, 0.0);
    }

    #[test]
    fn observation_contains_qpos_qvel_and_contact_forces() {
        let mut ant = Ant::default();
        let observation = ant.reset(Some(42)).observation;

        assert_eq!(observation.len(), 105);
        assert_eq!(&observation[27..], &[0.0; 78]);
    }

    #[test]
    fn unhealthy_height_terminates() {
        let mut ant = Ant::default();
        ant.positions[2] = 0.2;

        assert!(!ant.is_healthy());
    }

    #[test]
    fn expert_gait_clears_the_registry_threshold() {
        let episodes = 20_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut ant = Ant::default();
            let mut observation = ant.reset(Some(seed)).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = Ant::expert_action(&observation);
                let transition = ant.step(action.to_vec());
                total_reward += transition.reward;
                observation = transition.observation;
            }
        }

        let mean_reward = total_reward / episodes as f64;
        assert!(
            mean_reward >= Ant::SOLVED_MEAN_REWARD,
            "expert mean reward was {mean_reward}"
        );
    }
}
