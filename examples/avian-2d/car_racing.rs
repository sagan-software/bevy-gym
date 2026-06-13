//! ## Description
//! The easiest control task to learn from pixels - a top-down
//! racing environment. The generated track is random every episode.
//!
//! Some indicators are shown at the bottom of the window along with the
//! state RGB buffer. From left to right: true speed, four ABS sensors,
//! steering wheel position, and gyroscope.
//! To play yourself (it's rather fast for humans), type:
//! ```shell
//! python gymnasium/envs/box2d/car_racing.py
//! ```
//! Remember: it's a powerful rear-wheel drive car - don't press the accelerator
//! and turn at the same time.
//!
//! ## Action Space
//! If continuous there are 3 actions :
//! - 0: steering, -1 is full left, +1 is full right
//! - 1: gas
//! - 2: braking
//!
//! If discrete there are 5 actions:
//! - 0: do nothing
//! - 1: steer right
//! - 2: steer left
//! - 3: gas
//! - 4: brake
//!
//! ## Observation Space
//!
//! A top-down 96x96 RGB image of the car and race track.
//!
//! ## Rewards
//! The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles
//!  visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
//!
//! ## Starting State
//! The car starts at rest in the center of the road.
//!
//! ## Episode Termination
//! The episode finishes when all the tiles are visited. The car can also go outside the playfield -
//!  that is, far off the track, in which case it will receive -100 reward and die.
//!
//! ## Arguments
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
//! >>> env
//! <TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>
//!
//! ```
//!
//! * `lap_complete_percent=0.95` dictates the percentage of tiles that must be visited by
//!  the agent before a lap is considered complete.
//!
//! * `domain_randomize=False` enables the domain randomized variant of the environment.
//!  In this scenario, the background and track colours are different on every reset.
//!
//! * `continuous=True` specifies if the agent has continuous (true) or discrete (false) actions.
//!  See action space section for a description of each.
//!
//! ## Reset Arguments
//!
//! Passing the option `options["randomize"] = True` will change the current colour of the environment on demand.
//! Correspondingly, passing the option `options["randomize"] = False` will not change the current colour of the environment.
//! `domain_randomize` must be `True` on init for this argument to work.
//!
//! ```python
//! >>> import gymnasium as gym
//! >>> env = gym.make("CarRacing-v3", domain_randomize=True)
//!
//! # normal reset, this changes the colour scheme by default
//! >>> obs, _ = env.reset()
//!
//! # reset with colour scheme change
//! >>> randomize_obs, _ = env.reset(options={"randomize": True})
//!
//! # reset with no colour scheme change
//! >>> non_random_obs, _ = env.reset(options={"randomize": False})
//!
//! ```
//!
//! ## Version History
//! - v2: Change truncation to termination when finishing the lap (1.0.0)
//! - v1: Change track completion logic and add domain randomization (0.24.0)
//! - v0: Original version
//!
//! ## References
//! - Chris Campbell (2014), http://www.iforce2d.net/b2dtut/top-down-car.
//!
//! ## Credits
//! Created by Oleg Klimov
