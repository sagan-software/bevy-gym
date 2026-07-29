---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - Cargo.toml
  - docs/examples/cartpole.md
  - examples/classic-control/cart_pole.rs
  - src/core.rs
  - src/training/collector.rs
  - src/training/ppo.rs
  - src/training/checkpoint.rs
  - src/training/proof.rs
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Ecosystem multi-agent reinforcement learning, recurrent memory, curriculum transfer, Avian2D perception and physics, and learning evidence'
research_goals: 'Define a safe and testable training design for four incremental ecosystem examples using Bevy 0.18, Burn 0.21, Avian2D, recurrent PPO, multi-agent competition, predator-prey learning, and checkpoint-backed video evidence.'
user_name: 'Sagan'
date: '2026-07-26'
web_research_enabled: true
source_verification: true
source_policy: 'Primary sources and first-party documentation only'
---

# Ecosystem multi-agent, memory, curriculum, physics, and evidence research

Date: 2026-07-26
Local anchor: `bevy-gym` 0.3.3, Bevy 0.18, Burn 0.21
Intended examples: `examples/ecosystem/`

## Decision summary

The four examples should use one recurrent PPO family, one fixed observation schema, and one fixed
continuous action schema from the first lesson onward. Each later lesson changes the world and the
set of active species, not the actor tensor shapes. This is the simplest reliable way to transfer
Burn checkpoints without inventing weight surgery.

Use parameter sharing among bunnies and separately among foxes. Do not combine both species into one
shared policy by default. Their objectives and food semantics conflict. Each living agent contributes
its own temporally ordered samples to its species update. Use centralized training with decentralized
execution: the actor receives only proprioception, Avian ray observations, and its own recurrent
state; the critic may receive masked global state during training. At evaluation, only the actor and
its per-agent recurrent state exist.

Use an LSTM before the actor and critic heads. Burn 0.21 already exposes `Lstm`, `LstmState`, and
batch-first sequence input. Its forward call accepts an optional initial state and returns a final
state, which matches one state per living agent. The module is described as stateless because the
caller owns that state, not because it lacks temporal memory
([Burn `Lstm`](https://docs.rs/burn/0.21.0/burn/nn/struct.Lstm.html),
[Burn `LstmState`](https://docs.rs/burn/0.21.0/burn/nn/struct.LstmState.html)).

Use Avian2D 0.6.1. Avian's compatibility table assigns versions 0.5 through 0.6 to Bevy 0.18
([Avian version table](https://docs.rs/crate/avian2d/0.6.1)). Agents must be dynamic rigid bodies with
solid colliders so collisions can block and push them. Rocks, trees, and the well base should be
static bodies. Food, the well interaction area, and thorn damage areas should use sensor colliders and
explicit collision events.

Learning claims require held-out evaluation, not training reward or a video. Save the random initial
policy, every evaluation checkpoint, the validation-selected best policy, and a test-only result. Run
multiple training seeds. Report confidence intervals for survival improvement and direct behavioral
metrics such as eating, drinking, cause of death, predator encounters, and obstacle damage. A video
is a qualitative audit attached to the same immutable checkpoint and evaluation records.

## Local baseline and required extensions

The existing repository already supplies useful single-agent pieces:

- `Env`, `Reset`, `Step`, and the terminated-versus-truncated distinction in `src/core.rs`.
- A synchronous Bevy collector in `src/training/collector.rs`.
- Bounded continuous PPO in `src/training/ppo.rs`.
- Named MessagePack checkpoint paths and run artifacts in `src/training/checkpoint.rs` and
  `src/training/config.rs`.
- Validation, test, and demo seed separation plus checkpoint evidence types in
  `src/training/rng.rs` and `src/training/proof.rs`.
- Periodic checkpoint evaluation and FFmpeg video assembly in
  `examples/classic-control/cart_pole.rs`.

The current PPO is feed-forward and its actor and critic both consume the same local observation.
The current `Env` returns one observation, reward, and status. The ecosystem suite therefore needs
two deliberate additions before the multi-agent lessons:

1. A recurrent PPO path that collects intact sequences and owns hidden state per environment and
   agent.
2. A simultaneous multi-agent environment boundary with per-agent observations, actions, rewards,
   and statuses, plus optional training-only global state.

The PettingZoo Parallel API is a useful contract reference because it steps every live agent at once,
keeps a fixed `possible_agents` set, keeps a changing `agents` set, and returns observation, reward,
termination, truncation, and info mappings keyed by agent
([PettingZoo Parallel API](https://pettingzoo.farama.org/main/api/parallel/)). The local Rust API need
not copy Python dictionaries, but it should preserve those semantics. For efficient batching, use
stable typed agent IDs, fixed-capacity arrays, and an alive mask.

One ecosystem instance, containing all of its interacting agents, is one parallel environment. Do
not treat each agent as an independent Bevy environment. Several independent ecosystem instances can
still be collected in parallel for decorrelated rollouts.

## Environment and policy contract

### Stable action space

Use the same bounded continuous action in all four lessons:

```text
locomotion = [forward_acceleration, turn_rate], each in [-1, 1]
```

Eating and drinking should occur automatically while a living agent overlaps a compatible resource
interaction sensor. This keeps the action space small and stable. Facing remains meaningful because
the ray fan rotates with the agent. A no-op remains possible with `[0, 0]`.

The action is applied as acceleration or velocity control to a dynamic rigid body. Directly writing a
dynamic body's transform behaves like teleportation and can put it inside colliders; Avian recommends
velocity or forces for physical movement
([Avian rigid-body movement](https://docs.rs/avian2d/0.6.1/avian2d/dynamics/rigid_body/enum.RigidBody.html),
[Avian forces](https://docs.rs/avian2d/0.6.1/avian2d/dynamics/rigid_body/forces/index.html)).

### Stable local observation

Define the final superset shape before lesson one. Inactive channels are zero and accompanied by
presence masks. A practical initial schema is:

```text
proprioception
  hunger, thirst, hit_points, normalized_age
  linear_speed, angular_speed
  sin(facing), cos(facing)
  species_one_hot[bunny, fox]

vision for each ray, nearest hit only
  normalized_distance
  type_one_hot[food, well, bunny, fox, solid_obstacle, thorn, boundary]
  hit_present

curriculum context
  lesson_one_hot[solo, competition, predator_prey, obstacles]
```

The actor must not receive coordinates, resource counts, other agents' needs, map seed, or privileged
collision geometry. The curriculum indicator is legitimate task context, not a map oracle. All
features should be normalized to stable ranges.

Use a nonuniform forward ray fan. For example, thirteen rays at `0`, `+/-5`, `+/-10`, `+/-20`,
`+/-35`, `+/-55`, and `+/-80` degrees provide dense central vision and sparse peripheral vision. The
exact count is a performance parameter and should be benchmarked, but its order and angles are part
of the checkpoint schema once training begins.

### Training-only centralized state

The critic may consume a fixed, padded global state:

- Each possible agent's species, alive mask, pose, velocity, hunger, thirst, and hit points.
- The well pose and normalized water level.
- Fixed-capacity food slots with presence masks and poses.
- Fixed-capacity obstacle slots with type masks and poses.
- Time remaining and lesson ID.

This is centralized training with decentralized execution. The original multi-agent actor-critic
work identifies non-stationarity and growing policy-gradient variance as central multi-agent
difficulties and uses other-agent information during training
([MADDPG paper](https://arxiv.org/abs/1706.02275)). MAPPO demonstrates that an on-policy PPO baseline
with centralized value information can be competitive and sample-efficient in cooperative
multi-agent tasks
([MAPPO paper](https://arxiv.org/abs/2103.01955)).

The critic input must also use a final superset shape from lesson one. If that is too large for the
first implementation, preserve and transfer only the actor, then initialize a new lesson-specific
critic. Never claim whole-checkpoint curriculum transfer when only the actor transfers.

## Simultaneous multi-agent stepping

The world step should have this observable order:

1. Read every living agent's action from the same pre-step state.
2. Apply all locomotion commands without resolving agent interactions in iteration order.
3. Advance Avian by exactly one fixed physics step.
4. Resolve collision-derived eating, drinking, hunting, and thorn damage from that step.
5. Advance hunger, thirst, well refill, food spawning, and hit-point drains by fixed `dt`.
6. Mark newly dead agents and calculate each agent's reward and terminal status.
7. Run perception from the resulting world and emit next observations and training-only state.

This avoids first-agent advantage. The PettingZoo parallel model likewise applies one action per live
agent in one environment step and returns per-agent results together
([Parallel API](https://pettingzoo.farama.org/main/api/parallel/)). Avian's default physics schedule
is `FixedPostUpdate`, and it exposes schedule sets so application systems can be explicitly ordered
around the physics step
([Avian physics scheduling](https://docs.rs/avian2d/0.6.1/avian2d/schedule/struct.PhysicsSchedulePlugin.html)).

An individual death terminates that agent's trajectory and clears its LSTM state. The ecosystem
episode continues for the remaining agents until all controlled agents are dead or the time limit is
reached. Do not respawn an individual during the same episode unless respawning becomes an explicit
future environment rule. At the time limit, mark living trajectories truncated so value learning may
bootstrap; death is a natural termination and may not bootstrap. Gymnasium's documentation explains
why terminal observations and reset observations must not be conflated in rollout collection
([Gymnasium vector autoreset](https://gymnasium.farama.org/main/api/vector/)).

## Multi-agent training design

### Homogeneous competition

For the multi-bunny lesson, use one shared bunny actor and one shared bunny critic. Every living
bunny's sequence contributes samples to that shared update. Parameter sharing reduces the number of
trainable policies and is a strong homogeneous-agent baseline. Agent indication can permit agents to
specialize, but it can also break intended symmetry
([parameter-sharing paper](https://arxiv.org/abs/2005.13625)).

Start without a unique agent-ID input. Bunnies have identical abilities and objectives, and arbitrary
IDs can let them overfit fixed spawn-order roles. Randomize spawn ordering. Add a tested agent
indication only if symmetric sharing demonstrably cannot learn required role differentiation.

Individual survival rewards should remain individual. Do not replace them with a sum of all bunny
lifetimes: a team sum changes the task from "survive longest" to potentially sacrificing one bunny
for another. The centralized critic can still reduce variance without changing the actor's reward.

Normalize losses by the number of valid agent-time samples, not only environment steps. Otherwise,
changing the live population changes effective gradient scale. Store an alive mask, a natural
termination mask, and a truncation mask for every sequence element. Never manufacture zero-valued
"dead agent" transitions as ordinary training data.

### Predator-prey competition

Use one shared policy for all bunnies and a separate shared policy for all foxes. The same actor
architecture and action schema allow code reuse, but the parameters and optimizers should be
separate. Selective parameter-sharing research finds that sharing is most useful among agents with
compatible abilities and goals and can be harmful when applied indiscriminately
([selective sharing paper](https://arxiv.org/abs/2102.07475)).

Simultaneously changing bunny and fox policies makes each species' learning target non-stationary.
Mitigate this in the evidence protocol:

- Keep periodic immutable bunny and fox checkpoints.
- During training, sample some opponent episodes from recent frozen checkpoint snapshots rather than
  always using only the newest opponent.
- Evaluate each candidate against a fixed opponent panel: random, scripted-reactive, early,
  middle, and best-so-far checkpoints.
- Report a cross-play matrix, not only same-generation self-play.

Policy ensembles improved robustness in the original mixed cooperative-competitive actor-critic
experiments
([MADDPG paper](https://arxiv.org/abs/1706.02275)). Neural fictitious self-play provides stronger
evidence that training against an average of prior behavior can avoid divergence found in ordinary
self-play, although its poker-specific algorithm should not be copied literally here
([NFSP paper](https://arxiv.org/abs/1603.01121)). A small frozen-opponent panel is the proportional
version for these examples.

"All agents learn together" should therefore mean all valid current-agent trajectories update their
species policy in a shared training iteration. It should not mean all species share parameters or
that every opponent must update every episode.

## Recurrent perception and memory in Burn

### Why recurrence is appropriate

Ray observations make the environment partially observable. A food item or predator can leave the
field of view while remaining relevant. Deep recurrent Q-learning showed that an LSTM can integrate
observations over time and perform under partial observability without receiving full state
([DRQN paper](https://arxiv.org/abs/1507.06527)). This supports recurrent memory, but it does not prove
that an LSTM will learn a human-readable spatial map. The claim should be limited to learned temporal
state that can retain task-relevant evidence.

Use LSTM first, not a hand-built coordinate memory plus LSTM. Handing the policy exact remembered
world coordinates would bypass the intended perception problem. If learned recurrence later proves
insufficient, a bounded, noisy, egocentric memory feature can be evaluated as a separate lesson or
ablation.

### Burn 0.21 implementation shape

Burn's batch-first `Lstm::forward` accepts `[batch, sequence, input]` and an optional
`LstmState` whose cell and hidden tensors are `[batch, hidden]`; it returns sequence outputs and the
final cell and hidden state
([Burn `Lstm`](https://docs.rs/burn/0.21.0/burn/nn/struct.Lstm.html)). This maps cleanly to:

```text
normalized observation
  -> observation encoder MLP
  -> LSTM
  -> actor mean/log-standard-deviation head
  -> bounded continuous action

centralized state
  -> critic encoder MLP
  -> optional separate LSTM or feed-forward critic
  -> scalar value
```

Start with a recurrent actor and a feed-forward centralized critic. The actor requires memory for
decentralized behavior; the global critic already sees current state. Add critic recurrence only if
an ablation shows it improves validation performance.

Burn also provides a GRU in 0.21
([Burn neural-network module index](https://docs.rs/burn/0.21.0/burn/nn/)). A GRU is a valid later
ablation with less recurrent state, but Burn's LSTM API returns an explicit final state and directly
matches the requested memory design. Choosing both initially would multiply verification work.

### Rollout and hidden-state rules

The recurrent trainer must preserve these invariants:

- One independent LSTM state per ecosystem instance, species, and live agent ID.
- Zero state at episode reset, agent spawn, and immediately after natural termination.
- No hidden-state reuse when an agent ID is assigned to a different spawned body.
- Rollout minibatches are contiguous sequences, never randomly shuffled individual transitions.
- Sequence padding has a loss mask. A padded value may not influence policy, value, entropy, or
  normalization statistics.
- Natural termination prevents value bootstrap. Truncation permits value bootstrap from the final
  observation but still resets runtime hidden state before the next episode.
- Truncated backpropagation carries the numeric final hidden state into the next collection chunk but
  detaches its previous computation graph.
- Advantage normalization uses only valid, living samples.

For PPO, collect a fixed unroll length per independent ecosystem, then split each agent's data at
episode boundaries. The first hidden state of every unroll is part of the training sample. Recompute
log probabilities and values over the same ordered unroll during each PPO epoch. Randomize the order
of whole sequences, not timesteps within a sequence.

### Checkpoint and transfer rules

Burn module records contain learned module parameters. `LstmState` is a forward input/output owned by
the caller and is not a persistent model parameter; Burn's generated `LstmRecord` contains gate
records and configuration metadata, not live cell or hidden tensors
([Burn `LstmRecord`](https://docs.rs/burn/0.21.0/burn/nn/struct.LstmRecord.html)). Therefore:

- `best.mpk` contains actor/critic model weights, including LSTM weights.
- Runtime per-agent hidden states are not saved in `best.mpk` and begin at zero for a new evaluation
  episode.
- A resumable training checkpoint should additionally store optimizer state, global step, curriculum
  lesson, scheduler state, RNG streams, and opponent-pool manifest. That is distinct from the
  inference-oriented best policy.
- Checkpoint metadata must record observation schema version, ray angles, feature order, action
  schema, hidden width, species policy ID, and source lesson.

Burn's default named MessagePack recorder saves and loads model records, but load compatibility still
depends on constructing the matching module architecture
([Burn saving and loading](https://burn.dev/books/burn/saving-and-loading.html),
[`NamedMpkFileRecorder`](https://docs.rs/burn/0.21.0/burn/record/struct.NamedMpkFileRecorder.html)).
Keeping the final observation and action sizes from lesson one makes direct loading an enforced
contract instead of a best-effort conversion.

## Avian2D perception, collision, and interaction design

### Version and features

Use `avian2d = "0.6.1"` with the minimal 2D, `f32`, collider, and parallel features needed by the
headless examples. Enable rendering/debug features only behind the repository's render feature.
Avian's published compatibility table maps Bevy 0.18 to Avian 0.5 through 0.6
([Avian crate page](https://docs.rs/crate/avian2d/0.6.1)).

### Bodies and contacts

Use these body types:

- Bunny and fox: `RigidBody::Dynamic`, circle or capsule `Collider`, zero gravity, controlled
  velocity/forces, linear and angular damping, capped speed.
- Tree and rock: `RigidBody::Static` with solid colliders.
- Well base: static solid collider so agents can block one another around it.
- Well drink radius: child sensor collider with collision events.
- Food: sensor collider with collision events; despawn atomically after one valid consumer wins the
  simultaneous interaction tie-break.
- Thorn bush: sensor collider if it should damage without blocking, or a solid collider plus damage
  contact if it should also impede movement.

Avian defines dynamic bodies as affected by forces, velocity, and collisions. Kinematic bodies affect
dynamic bodies but are not themselves affected by collisions, so they do not satisfy mutual
agent-agent pushing
([Avian `RigidBody`](https://docs.rs/avian2d/0.6.1/avian2d/dynamics/rigid_body/enum.RigidBody.html)).

Avian collision events are emitted only for colliders with `CollisionEventsEnabled`. Sensor
colliders participate in detection without ordinary contact response
([Avian collision events](https://docs.rs/avian2d/0.6.1/avian2d/collision/collision_events/)).
Add explicit collision layers for agents, resources, hazards, obstacles, and boundaries. Test the
layer membership/filter matrix because relying on defaults makes ray filtering and contact intent
unclear.

Resolve contested consumption deterministically from the contact set, using stable agent IDs after
the physics step. Do not let Bevy query iteration order decide who eats or drinks. The well should
deduct a bounded drink amount per winner, never become negative, and refill by `rate * dt` up to
capacity.

### Ray perception

Avian supports per-frame ECS `RayCaster` components and direct `SpatialQuery` methods. A ray caster's
origin and direction are local, so it follows its entity or parent. `RayHits` can be sorted by
distance, and query filters control which colliders participate
([Avian spatial queries](https://docs.rs/avian2d/0.6.1/avian2d/spatial_query/),
[`RayCaster`](https://docs.rs/avian2d/0.6.1/avian2d/spatial_query/struct.RayCaster.html)).

Attach one `RayCaster` per eye ray as an agent child, with `ignore_self`, finite maximum distance,
and one closest hit. Query solid obstacles and semantic targets together so a tree occludes food,
water, and agents behind it. Map the hit entity's typed semantic component to the fixed one-hot ray
feature. A missed ray emits zero distance channels plus `hit_present = 0`; it must not reuse the
previous hit.

Use direct `SpatialQuery::cast_ray` instead if component-based ray updates cannot be scheduled after
the same physics step that produces the observation. The direct query offers explicit call timing
([Avian `SpatialQuery`](https://docs.rs/avian2d/0.6.1/avian2d/spatial_query/struct.SpatialQuery.html)).
Whichever route is selected, add a one-step test proving that the observation corresponds to the
post-action world rather than the previous physics frame.

### Reset and map generation

Use one root seed to derive disjoint deterministic streams for:

- Initial agent positions and facing.
- Static well position.
- Static obstacle layout.
- Food spawn times and positions.
- Evaluation opponent selection.

At reset, place the well once and keep it fixed until the next reset. Use rejection sampling plus
Avian shape-intersection queries to prevent initial overlap among agents, resources, obstacles, and
boundaries. Cap attempts and fail with a diagnostic rather than looping forever. Food spawning uses
the same overlap checks and a fixed maximum live-food count.

The headless and rendered modes must run the same fixed-step simulation logic. Rendering may
interpolate presentation transforms, but training state, needs, collisions, spawning, and reward must
advance only on fixed simulation steps.

## Curriculum and transfer design

Curriculum learning orders easier examples before harder ones so learned structure can support later
learning. The original curriculum-learning experiments frame this as gradually increasing difficulty
([Bengio et al.](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf)). Teacher-student
curriculum work selects tasks with high learning progress and revisits tasks whose performance is
falling, directly motivating explicit retention checks
([TSCL paper](https://arxiv.org/abs/1707.00183)).

### Lesson sequence

1. Solo survival: one bunny, food, one refillable well, no obstacles.
2. Resource competition: multiple bunnies sharing the lesson-one bunny policy.
3. Predator-prey: lesson-two bunnies plus a separately trained fox policy.
4. Obstacle ecosystem: lesson-three species plus trees, rocks, and thorns.

### Transfer matrix

| From | To | Transfer | Reinitialize |
| --- | --- | --- | --- |
| Solo bunny | Multi-bunny | Bunny actor, LSTM, actor optimizer optionally | Central critic if its global input was not fixed from lesson one |
| Multi-bunny | Predator-prey bunny | Bunny actor, LSTM, compatible critic | Opponent-pool state for the new lesson |
| Multi-bunny bunny | New fox | Observation encoder and LSTM as initialization only | Fox actor/value heads, optimizer, best-score history |
| Predator-prey | Obstacles | Both species actors, LSTMs, compatible critics | Lesson-specific best-score history |

The fox warm start should be recorded as initialization, not as evidence that fox behavior was
already learned. If copying only part of a Burn module is too invasive for the first implementation,
initialize the fox independently and preserve the bunny transfer. Correct transfer evidence matters
more than maximizing reuse.

Do not automatically advance after a fixed number of steps. Advance after a held-out validation gate
passes for several consecutive evaluations. After advancement, draw 10 to 20 percent of training
episodes from earlier lessons and keep a validation matrix over every learned lesson. This addresses
catastrophic forgetting without introducing a more complex continual-learning algorithm.

### Expanding worlds without expanding tensors

Changing a neural network's input or output width prevents direct whole-record loading. Avoid this by
reserving the final ray type channels, species bit, lesson bit, global critic slots, and masks in
lesson one. Absent foxes, obstacles, and thorns simply produce zero-presence channels. The official
MPE2 Simple Spread curriculum likewise zero-pads absent neighbors so the observation shape remains
fixed while difficulty changes
([MPE2 Simple Spread](https://mpe2.farama.org/main/environments/simple_spread/)).

If a future lesson truly needs a new action, add it through a new schema version and an explicit
migration experiment. Padding and masking heterogeneous action/observation spaces is established
multi-agent practice, but an invalid padded action still needs an action mask or deterministic no-op
semantics
([PettingZoo SuperSuit wrappers](https://pettingzoo.farama.org/main/api/wrappers/supersuit_wrappers/)).

## Reward design and reward-hacking controls

### Primary objective

Give each living agent `reward = dt` on every valid living transition and no survival reward after
death. With a fixed step, undiscounted episode return equals lifetime in seconds. Keep natural death
terminal and the external horizon truncated.

Do not add permanent rewards for moving toward visible food, eating, drinking, fleeing, hunting, or
avoiding obstacles. Those are mechanisms for survival, not the requested objective. Dense proxy
rewards create shortcuts: circling a food item, repeatedly entering a well sensor, farming collision
events, or avoiding all exploration can outscore actual survival. Reward hacking is the general
failure mode where an agent exploits the specified reward without satisfying the intended outcome
([Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)).

If solo survival cannot learn from the survival signal, first fix environment observability, time
constants, spawn distributions, and exploration. Only then test temporary potential-based shaping as
an explicitly flagged curriculum intervention. Do not let shaped scores select the final best policy.

### Environment controls against accidental exploits

- Hunger and thirst decrease on simulation time, not render frames or actions taken.
- Starvation and dehydration drains cannot be avoided by standing still, colliding, or leaving map
  bounds.
- Food consumption and well withdrawal are edge-triggered or rate-limited and conserve resource
  quantity.
- Food cannot spawn inside an agent, collider, or unreachable region.
- Boundary escape cannot freeze needs or continue survival reward.
- Thorn damage is time- or contact-impulse-based with an explicit cap, not multiplied by duplicate
  collision events.
- A dead body cannot eat, drink, push, block indefinitely, see, act, or earn reward.
- Simultaneous resource contacts have a deterministic tie-break independent of ECS iteration order.
- Physics NaN, out-of-bounds state, and invalid action terminate with a diagnostic and no favorable
  reward.

### Behavioral diagnostics

Log objective return and independent mechanism metrics. At minimum:

- Lifetime seconds and time-limit survival rate.
- Cause of death: starvation, dehydration, fox, thorn, invalid physics.
- Time since last meal and drink.
- Food eaten, water consumed, and failed resource interactions.
- Well empty duration and refill/withdrawal totals.
- Bunny and fox population over time.
- Fox kills and time between hunts.
- Agent-agent and agent-obstacle collision counts and impulses.
- Thorn contacts, damage, and thorn exposure while a fox is near versus absent.
- Distance traveled, stationary fraction, and boundary contacts.
- Actor learning rate, critic learning rate, policy loss, value loss, entropy, approximate KL,
  clipping fraction, explained variance, and gradient norm.

This distinguishes "learning rate" as the optimizer setting from the learning curve. Both should
appear in artifacts; only current optimizer rates belong in the HUD.

## Evaluation that can support a learning claim

Deep RL results vary substantially with seeds and implementation details
([Deep RL That Matters](https://arxiv.org/abs/1709.06560)). Point estimates from a few runs can be
misleading; interval estimates and robust aggregate measures are preferred
([statistical-precipice paper](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)).

### Required comparisons

For each lesson, evaluate:

1. Random action policy.
2. Saved step-zero initialized policy.
3. Best validation-selected learned policy.
4. No-memory ablation that zeros recurrent state every step.
5. For multi-agent lessons, the same policy against fixed opponent/population baselines.
6. For transferred lessons, transferred initialization versus a from-scratch run under the same
   environment-step budget.

The no-memory ablation is essential. A video of an LSTM policy returning to food does not establish
that memory caused the behavior. Compare survival and delayed-target retrieval with recurrent state
preserved versus reset.

### Seed and checkpoint protocol

- Use at least five independent training seeds for a final learning claim.
- Use fixed validation map seeds only for periodic checkpoint selection.
- Use a disjoint fixed test suite once after selection.
- Store exact ordered reset seeds, training seed, map generator version, physics timestep, model
  schema, and checkpoint SHA-256.
- Evaluate deterministic actor means unless stochastic evaluation is an explicit secondary suite.
- Load checkpoints in a fresh process and start every agent's hidden state at zero.
- Compare policies under identical seed and opponent panels.
- Report every run, median, interquartile mean where aggregation is useful, and 95 percent bootstrap
  confidence intervals for learned-minus-initial lifetime.

The repository already has separate validation, test, and demo seed streams and checkpoint reload
evidence. Reuse those contracts rather than creating example-only evidence rules.

### Lesson-specific evidence gates

Exact numeric thresholds should be calibrated with random and scripted baselines before they become
acceptance criteria. The evidence structure should be fixed now:

- Solo: learned-minus-initial survival confidence interval is positive; eating and drinking both
  occur on held-out maps; time-limit survival and cause-specific deaths are reported.
- Competition: shared-policy bunnies outperform initialized bunnies at aggregate survival;
  resource consumption is distributed across multiple agent IDs; performance does not depend on
  spawn-order ID.
- Predator-prey: bunny improvement holds against a frozen fox panel and fox improvement holds
  against a frozen bunny panel; report the full cross-play matrix so one species cannot appear better
  only because its current opponent regressed.
- Obstacles: both species retain predator-prey performance within a declared tolerance; solid
  obstacle contacts fall versus initialization; thorn exposure falls when no fox is nearby without
  eliminating escape-through-thorns behavior under threat.

Videos can reveal bad strategies and physics defects, but they are selected trajectories and cannot
replace these gates.

## Checkpoint video and README evidence

Bevy 0.18 provides an official `EasyScreenRecordPlugin` and programmatic `RecordScreen` messages.
Its official release notes also state that screen recording is not supported on Windows
([Bevy 0.18 video recording](https://bevy.org/news/bevy-0-18/#easy-screenshot-and-video-recording)).
Bevy also provides screenshot capture to disk for frame-level evidence
([Bevy screenshot example](https://bevy.org/examples/window/screenshot/)).

For this Linux-targeted repository, either the official screen recorder or the existing CartPole
PPM-to-FFmpeg pattern can produce MP4. The existing pattern is better for exact checkpoint stitching,
while Bevy capture is better for guaranteeing the video matches the rendered example. The ecosystem
implementation should render Bevy frames and use a deterministic stitch manifest rather than create
a second hand-drawn renderer.

Use the requested sequence exactly:

1. Ten seconds from `best.mpk`, labeled as a best-policy preview.
2. Five seconds from every selected periodic `checkpoints/step-*.mpk`, ordered by global step.
3. Twenty seconds from `best.mpk`, labeled as the final best-policy segment.

When there are too many periodic checkpoints, select them uniformly by global step and preserve the
first and last. Write `demo/video-manifest.json` containing checkpoint path, SHA-256, global step,
evaluation metrics, map seed, simulated seconds, playback speed, frame range, and HUD values for each
segment. Use the fixed demo seed stream, not a hand-picked favorable seed.

The HUD should show:

- Example and curriculum lesson.
- Checkpoint global step and segment index.
- Validation mean or median survival and learned delta from step zero.
- Current simulated time, playback multiplier, living bunny/fox counts.
- Current food count and well fill percentage.
- Actor and critic learning rates recorded at that checkpoint.
- Policy loss, value loss, entropy, and approximate KL from the checkpoint's training record.
- Demo seed and a visible `EVAL` marker so the video is not mistaken for live training.

The example README should link each video to its immutable run summary and test result. Do not treat
the MP4 alone as proof of learning. Burn's named MessagePack recorder is appropriate for the model
artifact and is already the repository convention
([Burn named MessagePack recorder](https://docs.rs/burn/0.21.0/burn/record/struct.NamedMpkFileRecorder.html)).

## Failure modes and required tests

### Environment tests

- Reset with the same seed reproduces well, agents, obstacles, and food spawn schedule.
- Different seeds alter each randomized category.
- Well capacity stays within `[0, capacity]` and refills at the configured fixed-time rate.
- Hunger, thirst, and hit-point drains are frame-rate independent.
- Eating and drinking conserve resource quantity and cannot trigger twice from one contested contact.
- Natural death and time-limit truncation produce different bootstrap masks.
- All living agents act from the same pre-step state.
- Agent collision results are invariant to entity spawn/query order within physics tolerance.
- Food and well behind a solid obstacle are occluded from the ray observation.
- A ray's semantic channel and normalized distance match the nearest hit.
- Dead agents have no collider response, action request, observation, recurrent state, or reward.

### Physics tests

- Two dynamic agent colliders block and push each other under opposed locomotion commands.
- A dynamic agent cannot pass through a static rock/tree at maximum configured speed.
- A thorn sensor damages without generating duplicate per-frame event multipliers.
- Food and well sensor contacts do not add unintended physical impulses.
- Spawn validation rejects overlaps and terminates after a bounded attempt count.
- Headless and rendered builds produce the same state trace for a short fixed seeded rollout, within
  declared floating-point tolerance.

### Recurrent trainer tests

- Hidden states are distinct for two agent IDs and two ecosystem instances.
- Hidden state resets on death and environment reset.
- Hidden state carries across rollout chunks but its old graph is detached.
- Sequence padding contributes exactly zero to every loss and statistic.
- Shuffling whole sequences preserves time order.
- Terminal values do not bootstrap; truncated values do.
- Saved and freshly reloaded policy outputs match for a fixed observation sequence.
- A checkpoint from each lesson loads into every later lesson with the declared transfer scope.

### Learning-system tests

- Step-zero evaluation is written before the first optimizer update.
- Every periodic model checkpoint has a matching evaluation record and hash.
- Best selection uses validation only; final test results cannot replace `best.mpk`.
- Every valid living agent contributes once to the shared species batch.
- Changing the number of living agents does not change loss normalization semantics.
- Actor inference cannot access centralized critic state through any alternate API.
- Opponent-panel evaluation uses immutable recorded checkpoint hashes.
- Video segments load checkpoint bytes from disk and match the manifest hash.

## Recommended implementation order

1. Lock the final action, local observation, centralized-state, and checkpoint metadata schemas.
2. Implement deterministic solo mechanics and Avian ray/collision tests without training.
3. Add recurrent PPO and prove hidden-state bookkeeping on a small partially observable fixture.
4. Train solo survival and establish multi-seed initial-versus-best evidence.
5. Add the simultaneous multi-agent boundary and shared-bunny MAPPO-style update.
6. Train competition and verify spawn-order symmetry and resource contention.
7. Add separate fox policy, frozen opponent panels, and cross-play evaluation.
8. Add static obstacles and thorn sensors, then verify retention across earlier lessons.
9. Generate checkpoint-backed videos and README links only after each lesson's evidence gate passes.

Do not begin with four large standalone example files. Put shared simulation, observation, rendering,
trainer, evaluation, and video code in `examples/ecosystem/common/` or a private example-support
module, then keep each runnable example focused on its lesson configuration and documentation.

## Source list

All external sources below are original research papers, official project documentation, or official
source/API documentation.

### Multi-agent learning

- Lowe et al., [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).
- Yu et al., [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955).
- Terry et al., [Revisiting Parameter Sharing in Multi-Agent Deep Reinforcement Learning](https://arxiv.org/abs/2005.13625).
- Christianos et al., [Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing](https://arxiv.org/abs/2102.07475).
- Heinrich and Silver, [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121).
- Farama Foundation, [PettingZoo Parallel API](https://pettingzoo.farama.org/main/api/parallel/).

### Recurrence, curriculum, and evaluation

- Hausknecht and Stone, [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527).
- Bengio et al., [Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf).
- Matiisen et al., [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183).
- Henderson et al., [Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560).
- Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
- Amodei et al., [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565).

### Burn, Avian, and Bevy

- Burn 0.21, [`Lstm`](https://docs.rs/burn/0.21.0/burn/nn/struct.Lstm.html),
  [`LstmState`](https://docs.rs/burn/0.21.0/burn/nn/struct.LstmState.html), and
  [saving/loading](https://burn.dev/books/burn/saving-and-loading.html).
- Avian2D 0.6.1, [crate/version documentation](https://docs.rs/crate/avian2d/0.6.1),
  [spatial queries](https://docs.rs/avian2d/0.6.1/avian2d/spatial_query/), and
  [rigid bodies](https://docs.rs/avian2d/0.6.1/avian2d/dynamics/rigid_body/enum.RigidBody.html).
- Bevy 0.18, [video recording release notes](https://bevy.org/news/bevy-0-18/#easy-screenshot-and-video-recording)
  and [official screenshot example](https://bevy.org/examples/window/screenshot/).
