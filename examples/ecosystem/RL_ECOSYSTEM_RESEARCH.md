# Ecosystem RL perception and navigation research

Research date: 2026-07-29

## Conclusion

The example already has memory. `src/training/recurrent_ppo.rs` implements a
Burn LSTM actor, and `examples/ecosystem/shared/training.rs` trains it with
recurrent PPO. The default actor memory has 64 hidden units. Rollouts retain
per-agent memory, train on contiguous 32-step chunks, and reset memory at
episode boundaries. Adding another LSTM would add complexity without addressing
the more immediate navigation problem.

The ray representation contains enough information in principle to steer
toward a visible resource. Each fixed ray slot supplies presence, normalized
distance, and a one-hot semantic kind. The slot index supplies an implicit
bearing. The stronger concern is that the observation omits signed motion and
contact state. It contains speed magnitude, but not whether the body is moving
forward or drifting sideways under inertia. It also duplicates the health
fraction in two scalar slots.

The existing learning result proves resource acquisition for one rotated but
otherwise fixed tutorial route. It does not prove that the policy responds to
food or water moving left, right, nearer, or farther away. The visual behavior
can therefore look unresponsive even when the held-out score is high.

## Patterns in established environments

- MPE2 supplies self velocity and agent-relative landmark positions. Its
  partially observable environments retain fixed-width nearby-entity slots.
  This makes direction and target displacement explicit instead of requiring a
  network to infer them from scalar speed and ray changes. See
  [MPE2 Simple](https://mpe2.farama.org/environments/simple/) and
  [MPE2 Simple Tag](https://mpe2.farama.org/environments/simple_tag/).

- PettingZoo Waterworld uses evenly spaced range sensors. It separates
  obstacle, boundary, food, poison, and pursuer distances by semantic channel,
  optionally includes sensed velocity, and adds explicit food and poison
  contact flags. Its reward combines a capture reward, a small food-contact
  reward, and an action-magnitude cost. See the official
  [Waterworld environment documentation](https://pettingzoo.farama.org/environments/sisl/waterworld/).

- PettingZoo Pursuit gives each agent a centered egocentric grid with separate
  wall, ally, and opponent channels. It combines a large capture reward with a
  small touch reward. This preserves spatial order while shortening credit
  assignment for initial contact. See the official
  [Pursuit environment documentation](https://pettingzoo.farama.org/environments/sisl/pursuit/).

- Neural MMO supplies a stable local tile crop, nearby entities, and internal
  health, food, and water state. Its published recurrent and convolutional
  policies learned robust foraging, but the environment uses much longer
  horizons than this 20-second demo. The official baseline also separates
  policy, recurrence, and reward wrappers and supports curriculum training.
  See the
  [Neural MMO ICML paper](https://neuralmmo.github.io/_build/html/_downloads/e88d1fc5952a922a6ef822bf4fbe4ebd/icml2020_paper.pdf),
  [Neural MMO environment API](https://neuralmmo.github.io/_build/html/rst/api.html),
  and [Neural MMO baselines](https://github.com/NeuralMMO/baselines).

- Animal-AI treats the world as partially observable. Its configurable fanned
  raycasts report object identity and distance. It can combine them with
  health, velocity, and position. The authors report that ray observations
  learn faster and cost less than camera observations. See the
  [Animal-AI Environment paper](https://link.springer.com/article/10.3758/s13428-025-02616-3).

- Unity ML-Agents recommends normalized observations, one-hot categorical
  values, and agent-relative coordinates. Its ray sensor exposes detectable
  types, ray count, angular extent, ray length, cast radius, and observation
  stacking. It recommends stacking or an RNN only when the policy must compare
  or remember observations over time. See
  [ML-Agents environment design](https://unity-technologies.github.io/ml-agents/Learning-Environment-Design-Agents/)
  and
  [ML-Agents recurrent training configuration](https://unity-technologies.github.io/ml-agents/Training-Configuration-File/).

- Stable-Baselines3 Contrib implements PPO with LSTM actor-critic policies and
  requires the recurrent state and episode-start signal during inference.
  This matches the current example's episode-scoped memory model. See the
  official [Recurrent PPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
  and
  [recurrent policy source](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/policies.py).

## Recommended changes for this example

1. Keep the LSTM and prove the observation first.

   Add deterministic bearing sweeps with food and water at left, center, and
   right bearings. Assert the semantic channel, ray slot, and monotonic distance
   value. Add a policy-response report that records forward throttle, turn
   direction, and gaze for the same sweep.

2. Add signed body-relative velocity.

   Encode longitudinal and lateral velocity divided by maximum speed. Replace
   the duplicated health value and absolute-world heading before increasing the
   tensor width. Retain angular velocity and gaze yaw. This follows MPE2 and
   Waterworld, and it lets the actor distinguish approach, retreat, and lateral
   drift.

3. Add compact egocentric resource summaries.

   Preserve rays for partial perception and visualization. Also expose the
   nearest visible food and well as
   `[present, proximity, sin(bearing), cos(bearing)]`. Derive these values only
   from currently visible ray samples. This keeps the policy local while making
   the steering relation explicit, as MPE makes landmark displacement explicit.

4. Add event observations.

   Add one-step `touched_food`, `ate`, `touched_well`, and `drank` flags. The
   Waterworld contact flags show the same pattern. These flags help the policy
   associate a sampled action and contact with a delayed hydration or satiation
   change.

5. Train navigation before active gaze.

   Use a short curriculum: centered visible target, randomized target within
   the frontal cone, randomized left/right bearing and distance, then movable
   gaze, then occlusion and obstacles. Keep the final survival reward and food
   and drink bonuses. A small contact reward is reasonable during acquisition
   lessons because Waterworld and Pursuit use one.

6. Treat distance shaping as a temporary training aid.

   If contact rewards remain too sparse, add a curriculum-only potential based
   on progress toward the current resource. Use
   `gamma * potential(next) - potential(current)` and define terminal potential
   explicitly. Potential-based shaping preserves the optimal policy only under
   its stated conditions; do not claim that guarantee for an arbitrary distance
   bonus. See
   [Ng, Harada, and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

7. Test whether memory helps.

   Compare the current recurrent policy with a feed-forward policy using a
   small observation stack. A visible stationary target should not require
   long-term memory. Occlusion, movable gaze, and remembering a resource after
   turning away are the cases where recurrence should improve results. Also
   compare 32-step and 64-step recurrent chunks before changing the 64-unit
   hidden state.

## Evidence required before calling navigation learned

- Randomize food and water bearings, distances, and order independently.
- Report first-contact rate, acquisition rate, path length, and time to contact.
- Bucket deterministic actions by visible target kind and bearing.
- Verify that left and right targets produce opposite turn responses.
- Relocate or swap resources during playback and show the policy changes course.
- Blank or shuffle resource ray channels and verify that acquisition falls.
- Keep fixed-seed evaluation, then repeat across at least three training seeds.

These checks distinguish visual navigation from an open-loop route, favorable
spawn geometry, or recurrent memorization.
