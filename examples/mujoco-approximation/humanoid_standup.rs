//! ## Description
//! This environment is based on the environment introduced by Tassa, Erez and Todorov in [Synthesis and stabilization of complex behaviors through online trajectory optimization](https://ieeexplore.ieee.org/document/6386025).
//! The 3D bipedal robot is designed to simulate a human.
//! It has a torso (abdomen) with a pair of legs and arms, and a pair of tendons connecting the hips to the knees.
//! The legs each consist of three body parts (thigh, shin, foot), and the arms consist of two body parts (upper arm, forearm).
//! The environment starts with the humanoid laying on the ground, and then the goal of the environment is to make the humanoid stand up and then keep it standing by applying torques to the various hinges.
//!
//!
//! ## Action Space
//! ```{figure} action_space_figures/humanoid.png
//! :name: humanoid
//! ```
//!
//! The action space is a `Box(-0.4, 0.4, (17,), float32)`. An action represents the torques applied at the hinge joints.
//!
//! | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
//! | --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
//! | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -0.4        | 0.4         | `abdomen_y`                        | hinge | torque (N m) |
//! | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -0.4        | 0.4         | `abdomen_z`                        | hinge | torque (N m) |
//! | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -0.4        | 0.4         | `abdomen_x`                        | hinge | torque (N m) |
//! | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -0.4        | 0.4         | `right_hip_x` (`right_thigh`)        | hinge | torque (N m) |
//! | 4   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -0.4        | 0.4         | `right_hip_z` (`right_thigh`)        | hinge | torque (N m) |
//! | 5   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -0.4        | 0.4         | `right_hip_y` (`right_thigh`)        | hinge | torque (N m) |
//! | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -0.4        | 0.4         | `right_knee`                       | hinge | torque (N m) |
//! | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -0.4        | 0.4         | `left_hip_x` (`left_thigh`)          | hinge | torque (N m) |
//! | 8   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -0.4        | 0.4         | `left_hip_z` (`left_thigh`)          | hinge | torque (N m) |
//! | 9   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -0.4        | 0.4         | `left_hip_y` (`left_thigh`)          | hinge | torque (N m) |
//! | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -0.4        | 0.4         | `left_knee`                        | hinge | torque (N m) |
//! | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -0.4        | 0.4         | `right_shoulder1`                  | hinge | torque (N m) |
//! | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -0.4        | 0.4         | `right_shoulder2`                  | hinge | torque (N m) |
//! | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -0.4        | 0.4         | `right_elbow`                      | hinge | torque (N m) |
//! | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -0.4        | 0.4         | `left_shoulder1`                   | hinge | torque (N m) |
//! | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -0.4        | 0.4         | `left_shoulder2`                   | hinge | torque (N m) |
//! | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -0.4        | 0.4         | `left_elbow`                       | hinge | torque (N m) |
//!
//!
//! ## Observation Space
//! The observation space consists of the following parts (in order)
//!
//! - *qpos (22 elements by default):* The position values of the robot's body parts.
//! - *qvel (23 elements):* The velocities of these individual body parts (their derivatives).
//! - *cinert (130 elements):* Mass and inertia of the rigid body parts relative to the center of mass,
//!   (this is an intermediate result of the transition).
//!   It has shape 13*10 (*nbody * 10*).
//!   (cinert - inertia matrix and body mass offset and body mass)
//! - *cvel (78 elements):* Center of mass based velocity.
//!   It has shape 13 * 6 (*nbody * 6*).
//!   (com velocity - velocity x, y, z and angular velocity x, y, z)
//! - *`qfrc_actuator` (17 elements):* Constraint force generated as the actuator force at each joint.
//!   This has shape `(17,)`  *(nv * 1)*.
//! - *`cfrc_ext` (78 elements):* This is the center of mass based external force on the body parts.
//!   It has shape 13 * 6 (*nbody * 6*) and thus adds another 78 elements to the observation space.
//!   (external forces - force x, y, z and torque x, y, z)
//!
//! where *nbody* is the number of bodies in the robot,
//! and *nv* is the number of degrees of freedom (*= dim(qvel)*).
//!
//! By default, the observation does not include the x- and y-coordinates of the torso.
//! These can be included by passing `exclude_current_positions_from_observation=False` during construction.
//! In this case, the observation space will be a `Box(-Inf, Inf, (350,), float64)`, where the first two observations are the x- and y-coordinates of the torso.
//! Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.
//!
//! By default, however, the observation space is a `Box(-Inf, Inf, (348,), float64)`, where the position and velocity elements are as follows:
//!
//! | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
//! | --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
//! | 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)               |
//! | 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
//! | 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
//! | 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
//! | 4   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)                |
//! | 5   | z-angle of the abdomen (in `lower_waist`)                                                                         | -Inf | Inf | `abdomen_z`                        | hinge | angle (rad)                |
//! | 6   | y-angle of the abdomen (in `lower_waist`)                                                                         | -Inf | Inf | `abdomen_y`                        | hinge | angle (rad)                |
//! | 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | `abdomen_x`                        | hinge | angle (rad)                |
//! | 8   | x-coordinate of angle between pelvis and right hip (in `right_thigh`)                                             | -Inf | Inf | `right_hip_x`                      | hinge | angle (rad)                |
//! | 9   | z-coordinate of angle between pelvis and right hip (in `right_thigh`)                                             | -Inf | Inf | `right_hip_z`                      | hinge | angle (rad)                |
//! | 10  | y-coordinate of angle between pelvis and right hip (in `right_thigh`)                                             | -Inf | Inf | `right_hip_y`                      | hinge | angle (rad)                |
//! | 11  | angle between right hip and the right shin (in `right_knee`)                                                      | -Inf | Inf | `right_knee`                       | hinge | angle (rad)                |
//! | 12  | x-coordinate of angle between pelvis and left hip (in `left_thigh`)                                               | -Inf | Inf | `left_hip_x`                       | hinge | angle (rad)                |
//! | 13  | z-coordinate of angle between pelvis and left hip (in `left_thigh`)                                               | -Inf | Inf | `left_hip_z`                       | hinge | angle (rad)                |
//! | 14  | y-coordinate of angle between pelvis and left hip (in `left_thigh`)                                               | -Inf | Inf | `left_hip_y`                       | hinge | angle (rad)                |
//! | 15  | angle between left hip and the left shin (in `left_knee`)                                                         | -Inf | Inf | `left_knee`                        | hinge | angle (rad)                |
//! | 16  | coordinate-1 (multi-axis) angle between torso and right arm (in `right_upper_arm`)                                | -Inf | Inf | `right_shoulder1`                  | hinge | angle (rad)                |
//! | 17  | coordinate-2 (multi-axis) angle between torso and right arm (in `right_upper_arm`)                                | -Inf | Inf | `right_shoulder2`                  | hinge | angle (rad)                |
//! | 18  | angle between right upper arm and `right_lower_arm`                                                               | -Inf | Inf | `right_elbow`                      | hinge | angle (rad)                |
//! | 19  | coordinate-1 (multi-axis) angle between torso and left arm (in `left_upper_arm`)                                  | -Inf | Inf | `left_shoulder1`                   | hinge | angle (rad)                |
//! | 20  | coordinate-2 (multi-axis) angle between torso and left arm (in `left_upper_arm`)                                  | -Inf | Inf | `left_shoulder2`                   | hinge | angle (rad)                |
//! | 21  | angle between left upper arm and `left_lower_arm`                                                                 | -Inf | Inf | `left_elbow`                       | hinge | angle (rad)                |
//! | 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
//! | 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
//! | 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)             |
//! | 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
//! | 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
//! | 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s)   |
//! | 28  | z-coordinate of angular velocity of the abdomen (in `lower_waist`)                                                | -Inf | Inf | `abdomen_z`                        | hinge | angular velocity (rad/s)   |
//! | 29  | y-coordinate of angular velocity of the abdomen (in `lower_waist`)                                                | -Inf | Inf | `abdomen_y`                        | hinge | angular velocity (rad/s)   |
//! | 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | `abdomen_x`                        | hinge | angular velocity (rad/s)   |
//! | 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in `right_thigh`)                 | -Inf | Inf | `right_hip_x`                      | hinge | angular velocity (rad/s)   |
//! | 32  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in `right_thigh`)                 | -Inf | Inf | `right_hip_z`                      | hinge | angular velocity (rad/s)   |
//! | 33  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in `right_thigh`)                 | -Inf | Inf | `right_hip_y`                      | hinge | angular velocity (rad/s)   |
//! | 34  | angular velocity of the angle between right hip and the right shin (in `right_knee`)                              | -Inf | Inf | `right_knee`                       | hinge | angular velocity (rad/s)   |
//! | 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in `left_thigh`)                   | -Inf | Inf | `left_hip_x`                       | hinge | angular velocity (rad/s)   |
//! | 36  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in `left_thigh`)                   | -Inf | Inf | `left_hip_z`                       | hinge | angular velocity (rad/s)   |
//! | 37  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in `left_thigh`)                   | -Inf | Inf | `left_hip_y`                       | hinge | angular velocity (rad/s)   |
//! | 38  | angular velocity of the angle between left hip and the left shin (in `left_knee`)                                 | -Inf | Inf | `left_knee`                        | hinge | angular velocity (rad/s)   |
//! | 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in `right_upper_arm`) | -Inf | Inf | `right_shoulder1`                  | hinge | angular velocity (rad/s)   |
//! | 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in `right_upper_arm`) | -Inf | Inf | `right_shoulder2`                  | hinge | angular velocity (rad/s)   |
//! | 41  | angular velocity of the angle between right upper arm and `right_lower_arm`                                       | -Inf | Inf | `right_elbow`                      | hinge | angular velocity (rad/s)   |
//! | 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in `left_upper_arm`)   | -Inf | Inf | `left_shoulder1`                   | hinge | angular velocity (rad/s)   |
//! | 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in `left_upper_arm`)   | -Inf | Inf | `left_shoulder2`                   | hinge | angular velocity (rad/s)   |
//! | 44  | angular velocity of the angle between left upper arm and `left_lower_arm`                                         | -Inf | Inf | `left_elbow`                       | hinge | angular velocity (rad/s)   |
//! | excluded | x-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |
//! | excluded | y-coordinate of the torso (centre)                                                                         | -Inf | Inf | root                             | free  | position (m)               |
//!
//! The body parts are:
//!
//! | body part       | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
//! |  -------------  |  ---   |  ---  |
//! | worldbody (note: all values are constant 0) | 0  |excluded|
//! | torso           |1  | 0      |
//! | lwaist          |2  | 1      |
//! | pelvis          |3  | 2      |
//! | `right_thigh`     |4  | 3      |
//! | `right_sin`       |5  | 4      |
//! | `right_foot`      |6  | 5      |
//! | `left_thigh`      |7  | 6      |
//! | `left_sin`        |8  | 7      |
//! | `left_foot`       |9  | 8      |
//! | `right_upper_arm` |10 | 9      |
//! | `right_lower_arm` |11 | 10     |
//! | `left_upper_arm`  |12 | 11     |
//! | `left_lower_arm`  |13 | 12     |
//!
//! The joints are:
//!
//! | joint           | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
//! |  -------------  |  ---   |  ---  |
//! | root (note: all values are constant 0) | 0  |excluded|
//! | root (note: all values are constant 0) | 1  |excluded|
//! | root (note: all values are constant 0) | 2  |excluded|
//! | root (note: all values are constant 0) | 3  |excluded|
//! | root (note: all values are constant 0) | 4  |excluded|
//! | root (note: all values are constant 0) | 5  |excluded|
//! | `abdomen_z`       | 6  | 0      |
//! | `abdomen_y`       | 7  | 1      |
//! | `abdomen_x`       | 8  | 2      |
//! | `right_hip_x`     | 9  | 3      |
//! | `right_hip_z`     | 10 | 4      |
//! | `right_hip_y`     | 11 | 5      |
//! | `right_knee`      | 12 | 6      |
//! | `left_hip_x`      | 13 | 7      |
//! | `left_hiz_z`      | 14 | 8      |
//! | `left_hip_y`      | 15 | 9      |
//! | `left_knee`       | 16 | 10     |
//! | `right_shoulder1` | 17 | 11     |
//! | `right_shoulder2` | 18 | 12     |
//! | `right_elbow`     | 19 | 13     |
//! | `left_shoulder1`  | 20 | 14     |
//! | `left_shoulder2`  | 21 | 15     |
//! | `left_elfbow`     | 22 | 16     |
//!
//! The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
//! One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).
//!
//! **Note:**
//! When using HumanoidStandup-v3 or earlier versions, problems have been reported when using a `mujoco-py` version > 2.0, resulting in  contact forces always being 0.
//! Therefore, it is recommended to use a `mujoco-py` version < 2.0 when using the `HumanoidStandup` environment if you want to report results with contact forces (if contact forces are not used in your experiments, you can use version > 2.0).
//!
//!
//! ## Rewards
//! The total reward is: ***reward*** *=* *`uph_cost` + 1 - `quad_ctrl_cost` - `quad_impact_cost`*.
//!
//! - *`uph_cost`*:
//!   A reward for moving up (trying to stand up).
//!   This is not a relative reward, measuring how far up the robot has moved since the last timestep,
//!   but an absolute reward measuring how far up the Humanoid has moved up in total.
//!   It is measured as $w_{uph} \times \frac{z_{after\_action} - 0}{dt}$,
//!   where $z_{after\_action}$ is the z coordinate of the torso after taking an action,
//!   and $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is $5$),
//!   and `frametime`, which is $0.01$ - so the default is $dt = 5 \times 0.01 = 0.05$,
//!   and $w_{uph}$ is `uph_cost_weight` (default is $1$).
//! - *`quad_ctrl_cost`*:
//!   A negative reward to penalize the Humanoid for taking actions that are too large.
//!   $w_{quad\_control} \times \|action\|_2^2$,
//!   where $w_{quad\_control}$ is `ctrl_cost_weight` (default is $0.1$).
//! - *`impact_cost`*:
//!   A negative reward to penalize the Humanoid if the external contact forces are too large.
//!   $w_{impact} \times clamp(impact\_cost\_range, \|F_{contact}\|_2^2)$, where
//!   $w_{impact}$ is `impact_cost_weight` (default is $5\times10^{-7}$),
//!   $F_{contact}$ are the external contact forces (see `cfrc_ext` section on Observation Space).
//!
//! `info` contains the individual reward terms.
//!
//!
//! ## Starting State
//! The initial position state is $[0.0, 0.0, 1.4, 1.0, 0.0, ... 0.0] + \mathcal{U}_{[-reset\_noise\_scale \times I_{24}, reset\_noise\_scale \times I_{24}]}$.
//! The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{23}, reset\_noise\_scale \times I_{23}]}$.
//!
//! where $\mathcal{U}$ is the multivariate uniform continuous distribution.
//!
//! Note that the z- and x-coordinates are non-zero so that the humanoid immediately lies down and faces forward (x-axis).
//!
//!
//! ## Episode End
//! ### Termination
//! The Humanoid never terminates.
//!
//! ### Truncation
//! The default duration of an episode is 1000 timesteps.
//!
//!
//! ## Arguments
//! `HumanoidStandup` provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
//! These parameters can be applied during `gymnasium.make` in the following way:
//!
//! ```python
//! import gymnasium as gym
//! env = gym.make('HumanoidStandup-v5', impact_cost_weight=0.5e-6, ....)
//! ```
//!
//! | Parameter                                    | Type      | Default               | Description                                                                                                                                                                                                 |
//! | -------------------------------------------- | --------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `xml_file`                                   | **str**   |`"humanoidstandup.xml"`| Path to a `MuJoCo` model                                                                                                                                                                                      |
//! | `uph_cost_weight`                            | **float** | `1`                   | Weight for _`uph_cost`_ term (see `Rewards` section)                                                                                                                                                          |
//! | `ctrl_cost_weight`                           | **float** | `0.1`                 | Weight for _`quad_ctrl_cost`_ term (see `Rewards` section)                                                                                                                                                    |
//! | `impact_cost_weight`                         | **float** | `0.5e-6`              | Weight for _`impact_cost`_ term (see `Rewards` section)                                                                                                                                                       |
//! | `impact_cost_range`                          | **float** | `(-np.inf, 10.0)`     | Clamps the _`impact_cost`_ (see `Rewards` section)                                                                                                                                                            |
//! | `reset_noise_scale`                          | **float** | `1e-2`                | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
//! | `exclude_current_positions_from_observation` | **bool**  | `True`                | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation Space` section) |
//! | `include_cinert_in_observation`              | **bool**  | `True`                | Whether to include *cinert* elements in the observations (see `Observation Space` section)                                                                                                                  |
//! | `include_cvel_in_observation`                | **bool**  | `True`                | Whether to include *cvel* elements in the observations (see `Observation Space` section)                                                                                                                    |
//! | `include_qfrc_actuator_in_observation`       | **bool**  | `True`                | Whether to include *`qfrc_actuator`* elements in the observations (see `Observation Space` section)                                                                                                           |
//! | `include_cfrc_ext_in_observation`            | **bool**  | `True`                | Whether to include *`cfrc_ext`* elements in the observations (see `Observation Space` section)                                                                                                                |
//!
//! ## Version History
//! * v5:
//!     - Minimum `mujoco` version is now 2.3.3.
//!     - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
//!     - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
//!     - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the `MuJoCo` environments.
//!     - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
//!     - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
//!     - Excluded the `cinert` & `cvel` & `cfrc_ext` of `worldbody` and `root`/`freejoint` `qfrc_actuator` from the observation space, as it was always 0, and thus provided no useful information to the agent, resulting in slightly faster training (related [GitHub issue](https://github.com/Farama-Foundation/Gymnasium/issues/204)).
//!     - Restored the `xml_file` argument (was removed in `v4`).
//!     - Added `xml_file` argument.
//!     - Added `uph_cost_weight`, `ctrl_cost_weight`, `impact_cost_weight`, `impact_cost_range` arguments to configure the reward function (defaults are effectively the same as in `v4`).
//!     - Added `reset_noise_scale` argument to set the range of initial states.
//!     - Added `include_cinert_in_observation`, `include_cvel_in_observation`, `include_qfrc_actuator_in_observation`, `include_cfrc_ext_in_observation` arguments to allow for the exclusion of observation elements from the observation space.
//!     - Added `info["tendon_length"]` and `info["tendon_velocity"]` containing observations of the Humanoid's 2 tendons connecting the hips to the knees.
//!     - Added `info["x_position"]` & `info["y_position"]` which contain the observations excluded when `exclude_current_positions_from_observation == True`.
//!     - Added `info["z_distance_from_origin"]` which is the vertical distance of the "torso" body from its initial position.
//! * v4: All `MuJoCo` environments now use the `MuJoCo` bindings in mujoco >= 2.1.3.
//! * v3: This environment does not have a v3 release. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v2: All continuous control environments now use mujoco-py >= 1.50. Moved to the [gymnasium-robotics repo](https://github.com/Farama-Foundation/gymnasium-robotics).
//! * v1: `max_time_steps` raised to 1000 for robot based tasks. Added `reward_threshold` to environments.
//! * v0: Initial versions release.

use shakmaty as _;
use tokio as _;

use std::error::Error;

use clap as _;
#[cfg(feature = "mujoco")]
use mujoco_rs as _;
use std::path::Path;

use bevy_gym::training::{
    HumanoidModel, HumanoidMotion, RecurrentBehaviorSample, RecurrentPpoConfig, SplitMix64,
    HUMANOID_ACTIONS, HUMANOID_ACTION_LIMIT, HUMANOID_OBSERVATIONS,
};
use bevy_gym::{Env, EpisodeStatus, Reset, Step};

#[cfg(feature = "ecosystem-inference")]
use avian2d as _;
use bevy as _;
#[cfg(feature = "bevy-mcp")]
use bevy_brp_extras as _;
use bevy_gym::training::{run_continuous_workflow, ContinuousPpoExample};
use burn as _;
use serde_json as _;

/// `MAX_EPISODE_STEPS` used by this example.
const MAX_EPISODE_STEPS: usize = 1_000;
/// `MODEL_TIME_STEP` used by this example.
const MODEL_TIME_STEP: f32 = 0.003;

/// Gymnasium-compatible HumanoidStandup-v5 task.
#[derive(Debug, Clone)]
struct HumanoidStandup {
    /// Shared exact observation blocks and articulated state.
    model: HumanoidModel,
    /// Current episode transition count.
    elapsed_steps: usize,
    /// Deterministic reset stream.
    rng: SplitMix64,
}

impl Default for HumanoidStandup {
    fn default() -> Self {
        Self {
            model: HumanoidModel::default(),
            elapsed_steps: 0,
            rng: SplitMix64::new(0),
        }
    }
}

impl Env for HumanoidStandup {
    type Action = Vec<f32>;
    type Observation = Vec<f32>;
    type Info = ();

    fn reset(&mut self, seed: Option<u64>) -> Reset<Self::Observation, Self::Info> {
        if let Some(seed) = seed {
            self.rng = SplitMix64::new(seed);
        }
        self.model.reset(&mut self.rng, HumanoidMotion::Stand);
        self.elapsed_steps = 0;
        Reset {
            observation: self.model.observation(),
            info: (),
        }
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation, Self::Info> {
        let action: [f32; HUMANOID_ACTIONS] = std::array::from_fn(|index| {
            action
                .get(index)
                .copied()
                .unwrap_or(0.0)
                .clamp(-HUMANOID_ACTION_LIMIT, HUMANOID_ACTION_LIMIT)
        });
        let _horizontal_velocity = self.model.integrate(action, HumanoidMotion::Stand);
        self.elapsed_steps += 1;
        let upward_reward = self.model.positions[2] / MODEL_TIME_STEP;
        let control_cost = 0.1 * action.iter().map(|value| value * value).sum::<f32>();
        let impact_cost = (5e-7 * self.model.contact_force_squared()).min(10.0);
        Step {
            observation: self.model.observation(),
            reward: f64::from(upward_reward + 1.0 - control_cost - impact_cost),
            status: if self.elapsed_steps >= MAX_EPISODE_STEPS {
                EpisodeStatus::Truncated
            } else {
                EpisodeStatus::Continuing
            },
            info: (),
        }
    }
}

impl ContinuousPpoExample for HumanoidStandup {
    const ENV_NAME: &'static str = "humanoid-standup";
    const GYMNASIUM_ID: &'static str = "HumanoidStandup-v5";
    const OBSERVATION_DIM: usize = HUMANOID_OBSERVATIONS;
    const ACTION_LOW: &'static [f32] = &[-0.4; HUMANOID_ACTIONS];
    const ACTION_HIGH: &'static [f32] = &[0.4; HUMANOID_ACTIONS];
    const DEFAULT_TRAIN_STEPS: usize = 500_000;
    const ROLLOUT_STEPS_PER_ENV: usize = 128;
    const DEFAULT_NUM_ENVS: usize = 32;
    const DEFAULT_EVAL_INTERVAL: usize = 20_000;
    const DEFAULT_EVAL_EPISODES: usize = 20;
    const DEFAULT_ACTOR_LEARNING_RATE: f64 = 0.003;
    const DEFAULT_CRITIC_LEARNING_RATE: f64 = 0.001;
    const SOLVED_MEAN_REWARD: f64 = 350_000.0;
    const GIF_PATH: &'static str = "docs/images/humanoid-standup.gif";
    const BEHAVIOR_PRETRAINING_SAMPLES: usize = 4_096;
    const BEHAVIOR_PRETRAINING_EPOCHS: usize = 50;
    const RETAIN_RECURRENT_MEMORY: bool = false;

    fn ppo_config(actor_learning_rate: f64, critic_learning_rate: f64) -> RecurrentPpoConfig {
        RecurrentPpoConfig {
            actor_hidden_size: 128,
            critic_hidden_sizes: vec![256, 128],
            gamma: 0.995,
            gae_lambda: 0.95,
            actor_learning_rate,
            critic_learning_rate,
            entropy_coefficient: 0.001,
            epochs: 4,
            minibatch_sequences: 16,
            initial_log_std: -1.5,
            ..RecurrentPpoConfig::default()
        }
    }

    fn encode_observation(observation: &[f32]) -> Vec<f32> {
        HumanoidModel::encode_observation(observation, HumanoidMotion::Stand)
    }

    fn behavior_demonstrations(sample_count: usize, seed: u64) -> Vec<RecurrentBehaviorSample> {
        let mut humanoid = Self::default();
        let mut observation = humanoid.reset(Some(seed)).observation;
        let mut demonstrations = Vec::with_capacity(sample_count);
        while demonstrations.len() < sample_count {
            let action = HumanoidModel::expert_action(&observation, HumanoidMotion::Stand);
            demonstrations.push(RecurrentBehaviorSample {
                observation: Self::encode_observation(&observation),
                action: action.to_vec(),
            });
            let transition = humanoid.step(action.to_vec());
            observation = if transition.status == EpisodeStatus::Continuing {
                transition.observation
            } else {
                humanoid.reset(None).observation
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
    run_continuous_workflow::<HumanoidStandup>()
}

/// Render-only scene implementation.
#[cfg(feature = "render")]
mod render {
    use super::{
        ContinuousPpoExample, Env, EpisodeStatus, Error, HumanoidModel, HumanoidStandup, Path,
        HUMANOID_ACTIONS, HUMANOID_OBSERVATIONS,
    };

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
            app.insert_resource(ClearColor(Color::srgb(0.18, 0.25, 0.31)));
        }
    }

    #[derive(Resource)]
    /// Render state used by this example.
    struct VisualHumanoid {
        /// Policy loaded from the selected checkpoint.
        policy: RecurrentPpoPolicy,
        /// Environment state shown in the scene.
        env: HumanoidStandup,
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
    struct HumanoidPart(usize);

    /// Execute the `run_visual` example stage.
    ///
    /// # Errors
    ///
    /// Returns an error when setup, rendering, or encoding fails.
    pub(super) fn run_visual(
        checkpoint: &Path,
        capture_dir: Option<&Path>,
    ) -> Result<(), Box<dyn Error>> {
        let config = HumanoidStandup::ppo_config(0.003, 0.001);
        let policy = RecurrentPpoPolicy::load(
            checkpoint,
            HUMANOID_OBSERVATIONS,
            HUMANOID_OBSERVATIONS,
            1,
            &[-0.4; HUMANOID_ACTIONS],
            &[0.4; HUMANOID_ACTIONS],
            &config,
        )?;
        let memory = policy.initial_memory();
        let mut env = HumanoidStandup::default();
        let observation = env.reset(Some(12_345)).observation;
        let mut app = App::new();
        app.add_plugins(ExampleRendererPlugin);
        app.insert_resource(VisualHumanoid {
            policy,
            env,
            observation,
            memory,
        })
        .insert_resource(VisualClock(Timer::from_seconds(0.05, TimerMode::Repeating)))
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
                        title: "bevy-gym HumanoidStandup-v5".into(),
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
            Transform::from_xyz(0.0, 1.35, -3.5).looking_at(Vec3::new(0.0, 0.75, 0.0), Vec3::Y),
        ));
        spawn_mujoco_stage(&mut commands, &mut meshes, &mut materials);
        let tan = materials.add(StandardMaterial {
            base_color: Color::srgb(0.8, 0.6, 0.4),
            metallic: 0.05,
            perceptual_roughness: 0.35,
            ..default()
        });
        for index in 0..14 {
            let mesh = if index == 0 || index >= 12 {
                meshes.add(Sphere::new(if index == 0 { 0.09 } else { 0.06 }))
            } else {
                meshes.add(Capsule3d::new(
                    if index <= 2 { 0.07 } else { 0.045 },
                    if index <= 2 { 0.3 } else { 0.28 },
                ))
            };
            commands.spawn((
                Mesh3d(mesh),
                MeshMaterial3d(tan.clone()),
                HumanoidPart(index),
            ));
        }
    }

    /// Execute the `advance_watch` example stage.
    fn advance_watch(
        time: Res<'_, Time>,
        mut clock: ResMut<'_, VisualClock>,
        mut visual: ResMut<'_, VisualHumanoid>,
    ) {
        if clock.0.tick(time.delta()).just_finished() {
            advance_visual(&mut visual, 1);
        }
    }

    /// Execute the `advance_visual` example stage.
    fn advance_visual(visual: &mut VisualHumanoid, steps: usize) {
        for _ in 0..steps {
            let encoded = HumanoidStandup::encode_observation(&visual.observation);
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
        visual: Res<'_, VisualHumanoid>,
        mut parts: Query<'_, '_, (&HumanoidPart, &mut Transform)>,
    ) {
        apply_visual_state(&visual.env.model, &mut parts);
    }

    /// Execute the `apply_visual_state` example stage.
    fn apply_visual_state(
        model: &HumanoidModel,
        parts: &mut Query<'_, '_, (&HumanoidPart, &mut Transform)>,
    ) {
        let progress = ((model.positions[2] - 0.35) / 1.05).clamp(0.0, 1.0);
        let standing = humanoid_points(model.positions[0]);
        let lying = standing.map(|point| {
            let lateral = point.x - model.positions[0];
            Vec3::new(
                0.65f32.mul_add(lateral, point.y - 0.75 + model.positions[0]),
                0.25f32.mul_add(lateral.abs(), 0.1),
                0.2f32.mul_add(lateral, point.z),
            )
        });
        let points = std::array::from_fn::<_, 16, _>(|index| {
            lying
                .get(index)
                .copied()
                .expect("fixed example index is valid")
                .lerp(
                    standing
                        .get(index)
                        .copied()
                        .expect("fixed example index is valid"),
                    progress,
                )
        });
        let segments = [
            (points[1], points[0]),
            (points[2], points[1]),
            (points[3], points[2]),
            (points[3], points[4]),
            (points[4], points[5]),
            (points[3], points[6]),
            (points[6], points[7]),
            (points[1], points[8]),
            (points[8], points[9]),
            (points[1], points[10]),
            (points[10], points[11]),
        ];
        for (part, mut transform) in parts.iter_mut() {
            *transform = match part.0 {
                0 => Transform::from_translation(points[0]),
                12 => Transform::from_translation(points[5]),
                13 => Transform::from_translation(points[7]),
                index => segment_transform_3d(
                    segments
                        .get(index - 1)
                        .copied()
                        .expect("fixed example index is valid")
                        .0,
                    segments
                        .get(index - 1)
                        .copied()
                        .expect("fixed example index is valid")
                        .1,
                ),
            };
        }
    }

    /// Execute the `humanoid_points` example stage.
    fn humanoid_points(x: f32) -> [Vec3; 16] {
        [
            Vec3::new(x, 1.58, 0.0),
            Vec3::new(x, 1.42, 0.0),
            Vec3::new(x, 1.12, 0.0),
            Vec3::new(x, 0.88, 0.0),
            Vec3::new(x - 0.12, 0.5, 0.0),
            Vec3::new(x - 0.1, 0.1, 0.03),
            Vec3::new(x + 0.12, 0.5, 0.0),
            Vec3::new(x + 0.1, 0.1, -0.03),
            Vec3::new(x - 0.3, 1.18, 0.0),
            Vec3::new(x - 0.5, 0.98, 0.0),
            Vec3::new(x + 0.3, 1.18, 0.0),
            Vec3::new(x + 0.5, 0.98, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ]
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
        mut visual: ResMut<'_, VisualHumanoid>,
        mut parts: Query<'_, '_, (&HumanoidPart, &mut Transform)>,
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
            advance_visual(&mut visual, 10);
        }
        apply_visual_state(&visual.env.model, &mut parts);
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
        encode_gif(output, "humanoid-standup", 480, 480, |frames| {
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
        let mut first = HumanoidStandup::default();
        let first_observation = first.reset(Some(42)).observation;
        let mut repeated = HumanoidStandup::default();
        assert_eq!(repeated.reset(Some(42)).observation, first_observation);
        let mut different = HumanoidStandup::default();
        assert_ne!(different.reset(Some(43)).observation, first_observation);
        assert_eq!(first.model.phase, 0.0);
    }

    #[test]
    fn observation_contains_all_default_humanoid_blocks() {
        let mut humanoid = HumanoidStandup::default();
        let observation = humanoid.reset(Some(42)).observation;

        assert_eq!(observation.len(), HUMANOID_OBSERVATIONS);
        assert_eq!(observation[0], humanoid.model.positions[2]);
    }

    #[test]
    fn reward_uses_model_timestep_not_frame_skip_time() {
        let mut humanoid = HumanoidStandup::default();
        drop(humanoid.reset(Some(42)));
        humanoid.model.positions[2] = 0.9;
        let transition = humanoid.step(vec![0.0; HUMANOID_ACTIONS]);

        assert!(transition.reward > 250.0);
    }

    #[test]
    fn expert_pose_reaches_the_local_target() {
        let episodes = 10_u64;
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut humanoid = HumanoidStandup::default();
            let mut observation = humanoid.reset(Some(seed)).observation;
            for _ in 0..MAX_EPISODE_STEPS {
                let action = HumanoidModel::expert_action(&observation, HumanoidMotion::Stand);
                let transition = humanoid.step(action.to_vec());
                total_reward += transition.reward;
                observation = transition.observation;
            }
        }

        let mean_reward = total_reward / episodes as f64;
        assert!(
            mean_reward >= HumanoidStandup::SOLVED_MEAN_REWARD,
            "expert mean reward was {mean_reward}"
        );
    }

    #[test]
    fn random_action_baseline_stays_below_the_local_target() {
        let episodes = 10_u64;
        let mut action_rng = SplitMix64::new(77);
        let mut total_reward = 0.0;
        for seed in 0..episodes {
            let mut humanoid = HumanoidStandup::default();
            drop(humanoid.reset(Some(seed)));
            for _ in 0..MAX_EPISODE_STEPS {
                let action = (0..HUMANOID_ACTIONS)
                    .map(|_| action_rng.f32_between(-0.4, 0.4))
                    .collect();
                total_reward += humanoid.step(action).reward;
            }
        }

        let mean_reward = total_reward / episodes as f64;
        assert!(
            mean_reward < 125_000.0,
            "random-action mean reward was {mean_reward}"
        );
    }
}
