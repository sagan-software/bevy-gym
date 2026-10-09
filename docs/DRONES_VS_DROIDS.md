# Drones vs. Droids

Updated: 2026-10-09. Status: audited and planned; implementation and training pending.
This request supersedes the two-versus-two player-first shooter milestone.
The target is a spectator-first competitive match with three drones and three droids.
All living agents must choose their actions through reinforcement-trained policies.
The broader [examples roadmap](EXAMPLE_ROADMAP.md) remains required.

## Audit of the current pursuit example

Audited source checkpoint: `9c67238bab9f0c2d555074268a80899c18c14329`.
The current example does not meet the policy-only requirement.

- Motor commands come from `FlightPilot::action` in
  `examples/robots/flight_control/pilot.rs`. It loads `assets/robots/tracking.mpk`,
  an imitation-trained network, and discards recurrent history after every action.
  The checkpoint provenance is recorded in `assets/robots/README.md` and
  [the flight research record](DRONE_PURSUIT_GAME.md). A PPO policy type does not
  establish that the loaded weights were trained through reinforcement learning.
- Navigation comes from `Navigator::goal` in
  `examples/robots/pursuit/navigation/mod.rs`. Programmed map queries, target
  extrapolation, search-sample selection, scan rotation, and heading slew choose goals.
  `Flight::goal` in `pursuit/flight.rs` holds each goal for five motor steps.
- Firing comes from `Gun::advance` in `pursuit/enemy/gun.rs`. Visible contact starts
  an 800 ms charge, followed by an automatic three-round volley and 1,200 ms recovery.
  The gun also chooses aim from the observed target point.
- The lens state comes from `Detection::read` in `pursuit/perception.rs`.
  It maps health and geometric sight directly to colours. It is not a learned state.
  The requested four alert states are not fully implemented.
- Living droids use `Arena::step_relative` in `pursuit/arena.rs`: fixed-speed
  kinematic capsule movement. `pursuit/robot/animation.rs` plays authored clips.
  `pursuit/ragdoll.rs` hands the dead skeleton to physics. None of these proves
  learned living locomotion.
- Sight uses a geometric cone and solid rays, not rendered image observations.
  Hearing uses filtered events. These sensors can supply observations, but their
  outputs must not automatically command pursuit or firing.

Older navigation and PPO experiments remain research evidence. They do not establish
that the currently loaded pursuit controller is reinforcement-trained.
The shipped example remains a labelled hybrid baseline until its replacement qualifies.

## Control boundary

Policies choose navigation, aim, trigger requests, and behavioural alert state.
The drone motor policy chooses individual thrust commands. The droid motor policy
chooses bounded joint actuation; living movement cannot be produced by translating
its root or playing locomotion clips into authoritative bones.
A learned high-level policy may command a learned low-level policy. Every learned
layer requires its own checkpoint provenance and held-out evaluation.

Physics, contact, sensor measurements, health, damage, projectile travel, actuator
limits, ammunition, spin-up, cooldown, death, and match termination remain programmed
environment rules. For example, a trigger request cannot bypass cooldown, but visible
contact must never generate a trigger request. Occlusion blocks a projectile through
collision; it must not become an automatic aim or fire decision.

The policy selects one of idle, suspicious, pursuit, and search. These values drive
green, yellow, red, and orange lights and transition sounds respectively. Rendering
must not substitute a sight-derived state. Evaluate whether those learned signals
actually correspond to useful behaviour; colours alone do not prove perception.
Death and controller failure are lifecycle states, not learned behaviours.

A missing, corrupt, incompatible, or non-finite policy result stops the match with
an explicit failure. Never substitute a scripted controller or silently continue a
match with invented actions. No keyboard or spectator-camera input may affect agents.
Training curricula and rewards may configure tasks; they cannot supply agent actions.

## Batched competitive learning

Use one parameter-shared drone actor and one parameter-shared droid actor. Keep
independent recurrent memory for each of the six agents in each arena. Both teams
collect experience during the same matches. Multiple independent arenas contribute
to each rollout batch; one match with six agents is not six independent environments.

At each decision boundary, freeze the observation snapshot, infer every living
agent's action, then advance the shared world. Actor iteration order must not give
later agents information about earlier agents' unexecuted actions. Retain temporal
order within each agent trajectory. Reset memory at death and episode boundaries;
do not join unrelated agents or matches into one recurrent sequence.

Use centralized critics only during training and local sensor observations during
inference. The existing `RecurrentPpoPolicy` and `RecurrentPpoSequence` provide
parts of this separation. They do not yet supply the six-agent environment or its
rollout collector. The existing `EcosystemRuntime::step` also provides a local
reference for rejecting missing, duplicate, and stale multi-agent actions; its 2D
ecosystem is not the requested articulated 3D arena. The current actor is Gaussian; a joint continuous/categorical
policy needs correct sampling, log probabilities, entropy, and checkpoint metadata
for trigger and alert-state decisions before PPO training can be considered valid.

Freeze policy versions throughout each rollout. Update both team learners from the
collected batch, then start a new rollout with the new versions. Retain previous
opponents for evaluation and subsequent self-play experiments. Do not infer progress
from self-play win rate alone: stronger equally matched teams may remain near 50%.

[MAPPO](https://arxiv.org/abs/2103.01955v4) and its
[official implementation](https://github.com/marlbenchmark/on-policy) provide the
cooperative training reference. Competitive opponent selection is an additional
local design to qualify, not a guarantee from the cooperative benchmark results.

## Separate physical locomotion training

Interpret the requested crouching, prone movement, crawling, running, sprinting,
and jumping as droid skills. Retain a separate drone flight environment as well.

Build an articulated droid with validated mass, inertia, joint limits, ground contact,
and bounded actuators. Use the same body and actuator model in training and matches.
Observe joint positions and velocities, body orientation and velocity, contacts,
local terrain, and the high-level command. Do not expose hidden enemy state.

Train standing and recovery first, then commanded travel and turning, then variable
speed, crouch, prone transitions, crawling, and jumping. Test transitions as well as
individual skills. Include uneven ground, low cover, perturbations, and weapon load.
Reward task completion and physical stability while measuring energy, falls, foot
sliding, and joint-limit violations. Freeze evaluation scenarios before selecting
checkpoints. Training success is an empirical gate, not a promised outcome.

[AMP](https://arxiv.org/abs/2104.02180v2) is a research reference for using motion
clips as a learned style reward during physical RL. It is not currently implemented.
If motion data is used, record its licence and the training method. Playback or
behaviour cloning alone does not satisfy the requested reinforcement learning gate.

## Implementation invariants and checks

- Identify agents through closed team and three-slot types; validate external IDs.
  Derive team from identity. Do not store a second independently mutable team value.
- Keep match states running, finished with a result, and failed with a diagnostic
  mutually exclusive. Separate elimination, timeout, and controller failure.
- Validate finite actions and observations, tensor shapes, joint limits, policy
  versions, and environment compatibility before applying actions.
- Separate malformed actions from valid requests denied by weapon or lifecycle rules.
- Reset all six bodies, projectiles, effects, health, recurrent memories, and clocks.
  One death must not terminate a match while both teams still have living agents.
- Prove sensor isolation by moving a fully hidden enemy without changing actor input.
  Prove spectator input and training-only critic state cannot reach actor inference.
- Prove a visible enemy cannot cause firing when the policy requests no shot.
  Prove policy-selected alert state survives contradictory sensor input unchanged.
- Trace every actuator and trigger to the owning policy output and checkpoint hash.
  Audit all public construction and fallback paths before calling control policy-only.
- Test collision and damage among all six agents in one world, including simultaneous
  deaths and stable tie handling. Do not implement three isolated duels.
- Test batched gradient contribution from all living agents, per-agent memory
  isolation, team separation, death masks, truncation bootstrapping, and frozen
  rollout versions. Verify categorical and continuous log probabilities directly.

## Delivery checkpoints

1. Commit this audit and revised contract. Preserve the working hybrid demonstration
   and its recordings without presenting them as policy-only acceptance evidence.
2. Implement and test the shared six-agent world and policy action boundary.
   Use explicit test policies for deterministic tests; do not ship them as trained AI.
3. Implement mixed action distributions and the batched two-team rollout collector.
   Verify actual PPO updates and checkpoint round trips on a small diagnostic task.
4. Train and qualify separate RL drone and articulated droid locomotion policies.
   Publish learning curves, resource use, failed trials, checkpoints, and videos.
5. Train competitive 3v3 tactics using the qualified locomotion policies. Evaluate
   unseen seeds, cover layouts, damage, and frozen historical opponents for both teams.
6. Deliver browser spectator inference: start match, reset, follow any agent, free
   camera, pause, and speed controls. Show teams, health, learned state, and checkpoint
   identity. Record complete matches, contact sheets, and console results.
7. Re-audit all action sources, document remaining limits, and publish the qualified
   checkpoints and example. Continue the broader robot and reference-example roadmap.

Each checkpoint requires a scoped commit and non-force push to main after its gates.
Rust changes require focused failing tests first, exact repository/native/WASM lint
and test gates, personal lint, and measured changed-code coverage with explicit gaps.
Browser changes require rendered desktop/narrow inspection and video evidence.
Training records must include source/checkpoint hashes, seeds, environment version,
reward terms, update/sample counts, both teams' metrics, and held-out outcomes.
No 3v3 training run or trained droid locomotion checkpoint exists from this revision.
