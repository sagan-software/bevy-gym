# Ecosystem curriculum implementation plan

Status: active

This plan defines the implementation and evidence required for the four
`examples/ecosystem` environments. An environment is not complete because it
runs or renders. It is complete only after a freshly loaded policy beats its
fixed random-policy baseline on held-out seeds and the result is recorded.

The research behind the choices below is in [RESEARCH.md](RESEARCH.md).

## Desired outcome

Build one incremental top-down 2D curriculum:

1. A bunny learns to find food and a refillable well before starvation or
   dehydration kills it.
2. A shared bunny policy learns while several physical bunnies compete for the
   same resources.
3. Bunnies retain that skill while foxes learn to hunt them and both species
   must drink.
4. Both policies transfer into procedural maps containing solid trees and
   rocks plus traversable, damaging thorn bushes.

Every stage uses Avian2D colliders and the same agent observation and action
shapes. Each later stage loads the previous qualifying checkpoint instead of
silently starting over.

## Invariant scratchpad

### Closed states and modes

- `CurriculumStage`: `Survival`, `Competition`, `PredatorPrey`, `Obstacles`.
- `Species`: `Bunny`, `Fox`.
- `AgentLife`: `Alive`, `Dead(DeathCause)`.
- `DeathCause`: `Starvation`, `Dehydration`, `Thorns`, `Predation`.
- `PerceptKind`: `Food`, `Well`, `Bunny`, `Fox`, `SolidObstacle`, `Thorns`.
- `RunMode`: `Train`, `Eval`, `Watch`, `Video`.
- `EvalSuite`: `Validation`, `Test`, `Demo`.
- `LocomotionAction`: bounded forward acceleration and turn rate.

### Boundary validation and domain values

- Agent counts, map size, horizons, spawn intervals, capacities, and checkpoint
  intervals must be positive and bounded before a world is constructed.
- CLI stage is fixed by the selected Cargo example. It is not a free-form
  string passed into simulation logic.
- Seeds use the existing `SeedConfig` streams. Training, validation, test, and
  demo seeds remain disjoint.
- `AgentId` and policy roles are distinct types. An agent cannot index the
  wrong species policy by accident.
- Checkpoint initialization verifies observation width, action count, and
  hidden-layer widths before loading weights.

### Derived values

- Alive agent lists derive from lifecycle state.
- Health status, starvation, and dehydration derive from the current needs.
- Episode reward derives only from simulated seconds alive.
- Recurrent runtime state derives only from the ordered local observation
  history and resets at lifecycle boundaries.
- Map bounds, HUD counts, and evaluation aggregates derive from world state.
- Curriculum predecessor paths derive from the stage graph.

### Failure boundaries

- Malformed CLI/config/checkpoint input is a typed setup error.
- A valid checkpoint with an incompatible architecture is a compatibility
  error.
- A policy that fails a declared evidence gate is a completed experiment, not
  a successful curriculum stage.
- Missing `ffmpeg`, render support, or a checkpoint is a demo-generation error;
  it does not invalidate already recorded headless evaluation.

## Architecture decisions

### Training algorithm

Add a Burn recurrent PPO trainer with a decentralized actor and centralized,
training-only critic. The local actor encodes one observation, advances an
LSTM, and emits a bounded continuous action: forward acceleration and turn
rate. The critic receives the fixed padded global state and is absent from the
evaluation actor API.

Stage 1 and stage 2 use one shared bunny actor/critic. Stage 3 and stage 4 use
one shared bunny actor/critic and one shared fox actor/critic. Joint actions are
selected from the same pre-step world state and applied simultaneously.
Learning happens only after the joint environment step, preventing entity
iteration order from granting one agent a reaction advantage.

The procedural map expands with the curriculum: half-extents 10, 14, 20, and
25 from survival through obstacles. Survival starts with 12 food items and a
20-step spawn interval. These early densities make both resource loops
discoverable under exploration before later lessons add scarcity, agents,
predation, and obstacles. Evaluation uses the same declared settings; actor
input never receives coordinates or resource-placement hints.

Every living same-species agent contributes its contiguous sequence to that
species update. Losses normalize by valid agent-time samples, not ecosystem
steps. Sequence padding and dead slots are masked out of policy, value,
entropy, advantage normalization, and metrics.

### Perception and memory

Policies never receive absolute resource, opponent, predator, or obstacle
coordinates. Each agent has thirteen Avian2D raycasts at 0, +/-5, +/-10,
+/-20, +/-35, +/-55, and +/-80 degrees. A ray returns normalized distance,
hit presence, and a one-hot perceived kind. This makes forward vision denser
than peripheral vision while allowing solid bodies to occlude targets.

Each environment instance, species, and living agent ID owns an independent
Burn LSTM cell and hidden state. State starts at zero, carries across rollout
chunks, and resets immediately on death, truncation rollover, or environment
reset. Training uses contiguous sequences, initial states, padding masks, and
truncated backpropagation through time. Runtime hidden state is not stored in
`best.mpk`; fresh evaluation always begins at zero.

The LSTM is the memory system. Do not add exact remembered world coordinates,
which would bypass the intended partial-observation problem. Evaluation must
include a no-memory ablation that zeros recurrent state every step.

### Physics

- Use `avian2d` 0.6.1, compatible with Bevy 0.18.
- Run a fixed 10 Hz simulation with zero gravity.
- Agents are dynamic circular rigid bodies with rotation locked, bounded speed,
  damping, friction, and sleeping disabled.
- Agents collide with map boundaries, solid obstacles, and each other.
- Food, wells, and thorns have queryable sensor colliders.
- Trees use circles. Rocks and walls use solid boxes or convex shapes.
- Thorn overlap damages health and hit points without blocking motion.
- Contact tests must prove that two driven agents cannot pass through each
  other and that one can displace the other.

### Physiology and reward

Each living agent has hunger reserve, thirst reserve, health, and hit points.
Food replenishes bunny hunger. Eating a bunny replenishes fox hunger. Drinking
consumes well water and replenishes thirst. The well has finite capacity and a
continuous refill rate. Food spawns at deterministic pseudo-random positions
on a periodic schedule up to a cap.

Low hunger or thirst reduces both health and hit points. A zero need by itself
does not bypass those explicit damage paths. Zero hit points terminates that
agent with the applicable cause. Fox contact terminates a bunny as predation.

Reward is exactly simulated seconds alive. Consumption, approaching a target,
dealing damage, and exploration do not add reward. These events are metrics,
which prevents the implementation from claiming survival learning when it has
only learned a shaping reward.

### Stable curriculum contract

The local observation width, global-state width, action bounds, ray layout,
normalization, LSTM width, and network widths remain identical across all four
stages. Unavailable entities use zero features and explicit presence masks.
The local observation includes species and lesson one-hot channels.

Transfer graph:

```text
  survival bunny actor/LSTM
    -> competition bunny actor/LSTM
        -> predator-prey bunny actor/LSTM -> obstacles bunny actor/LSTM
                             new fox actor -> obstacles fox actor/LSTM
```

The predecessor checkpoint is recorded in config and provenance. A stage may
run from scratch for diagnostics, but it cannot qualify as curriculum evidence
without loading the declared predecessor.

## Source layout

```text
examples/
  README.md
  ecosystem/
    README.md
    PLAN.md
    RESEARCH.md
    survival.rs
    competition.rs
    predator_prey.rs
    obstacles.rs
    shared/
      mod.rs
      cli.rs
      domain.rs
      perception.rs
      simulation.rs
      training.rs
      rendering.rs
    media/
      survival-training.mp4
      competition-training.mp4
      predator-prey-training.mp4
      obstacles-training.mp4
      *.png
```

The four target files are documented entry points. Shared mechanics live in
`shared` so later examples literally build on the earlier implementation. The
reusable recurrent PPO model, sequence rollout, update, and checkpoint logic
live under `src/training` rather than being copied into the examples.
Generated training runs remain under ignored `runs/`. Curated videos and poster
frames are documentation assets only after their manifests and evaluation
evidence have been checked.

## Commands and artifact contract

Each target supports the same modes. Replace `<example>` with
`ecosystem-survival`, `ecosystem-competition`, `ecosystem-predator-prey`, or
`ecosystem-obstacles`.

```sh
cargo run --example <example>
cargo run --example <example> -- demo
cargo run --no-default-features --release --example <example> -- train
cargo run --release --example <example> -- eval --checkpoint <run-or-mpk>
cargo run --release --example <example> -- watch --checkpoint <run-or-mpk>
cargo run --release --example <example> -- video --checkpoint <run-dir>
```

The no-argument command trains visually by default. `demo` accepts safe
runtime overrides for iteration count, rollout count, horizon, evaluation
count, seed, playback speed, and run ID. The explicit `train` command is the
headless path.

Each run uses the existing `runs/<environment>-ppo/<run-id>/` structure and
writes config, seeds, metrics, evaluation rows, latest and periodic
checkpoints, `best.mpk`, a summary, and demo files.

Required training metrics:

- actor and critic learning rates;
- valid rollout samples, optimizer updates, policy loss, and value loss;
- entropy, approximate KL, clip fraction, explained variance, and gradient norm;
- per-species mean, median, minimum, and maximum lifetime;
- consumption and drinking counts;
- starvation, dehydration, thorn, and predation deaths;
- well empty fraction and mean food availability;
- per-species active count;
- collision and push contacts;
- validation checkpoint score and best-checkpoint selection.

Learning rates are constant for the initial recurrent PPO experiments and must
still be written into config, every checkpoint metric row, the terminal report,
and the HUD. A later scheduler must log its effective values, not only its
initial values.

## Evaluation protocol

Before training each stage, evaluate the random initialized policy on the same
fixed validation seeds used for periodic comparison. Select `best.mpk` only on
validation results. After training, reload `best.mpk` in a fresh process and
evaluate on disjoint test seeds. Demo seeds are separate from both.

Use at least 100 agent episodes for final single-agent test evidence and at
least 50 ecosystem episodes for multi-agent stages. Report bootstrap 95%
confidence intervals for mean lifetime and raw death counts. Run at least three
training seeds before calling a stage robust. One seed can establish an
implementation slice but cannot establish release qualification.

Predeclared stage gates:

| Stage | Primary held-out gate | Mechanism gate |
| --- | --- | --- |
| Survival | Best policy mean bunny lifetime is at least 30% above random and its 95% lower bound exceeds the random mean. | At least 70% of test episodes contain both eating and drinking before death or horizon. |
| Competition | Curriculum policy mean lifetime is at least 15% above the stage's random baseline and at least 80% of its solo predecessor on matched physiology settings. | At least two agents consume resources, collisions occur, and no stable agent-index advantage exceeds 20% after rotating spawn assignments. |
| Predator-prey | Each learned species exceeds its own random-policy lifetime by at least 15%. | Bunnies drink and eat food; foxes drink and eat bunnies; results remain non-degenerate with neither species winning every test episode. |
| Obstacles | Each transferred species retains at least 85% of its predator-prey test lifetime and exceeds a from-scratch control at the same step budget. | Solid-obstacle penetrations are zero and thorn damage per minute falls by at least 20% from random without reducing predator escape events to zero. |

If a gate fails, retain the run record, diagnose environment dynamics before
changing optimization, write the hypothesis, and rerun the same gate. Do not
move a threshold after seeing the result.

## Video and HUD contract

Each stage gets one top-down 1280x720 H.264/yuv420p training video. Its segment
order is exact:

1. 10 seconds from `best.mpk` on the fixed demo seed.
2. 5 seconds from every chronological periodic checkpoint at accelerated
   simulation time.
3. 20 seconds from `best.mpk` on the same demo seed at normal simulation time.

The middle sequence labels checkpoint step, validation lifetime, and playback
speed. The HUD also shows stage, species counts, simulation time, learning
actor/critic learning rates, entropy, mean lifetime, hunger/thirst summaries,
well capacity, food count, and checkpoint identity. The opening and ending best-policy segments
must be labelled so viewers do not mistake them for chronological training.

The real-time Bevy view uses a straight-down orthographic camera. Arrow keys or
WASD pan. Mouse wheel and `-`/`=` zoom. Programmer art uses stable colors and
basic shapes. Perception rays and compact LSTM-state diagnostics can be toggled
for debugging without revealing global coordinates to the actor.

`examples/README.md` embeds each curated video and links its run summary. The
suite README documents the exact reproduction commands and the limits of the
evidence.

## Incremental implementation checklist

### Slice 0: research and contracts

- [x] Fix stage, policy, observation, reward, and evidence invariants.
- [x] Complete primary-source research notes.
- [x] Register the four Cargo example targets and compatible Avian dependency.
- [x] Add suite and examples index documentation.

### Slice 1: shared deterministic simulation

- [x] Add validated stage configuration and deterministic RNG streams.
- [x] Add Avian world lifecycle, boundaries, agent bodies, and resource sensors.
- [x] Add physiology, well depletion/refill, periodic food spawning, and death.
- [x] Add post-physics ray perception with fixed superset feature channels.
- [ ] Test reset determinism, local observation shape/range, resource dynamics,
      starvation/dehydration, collision blocking/pushing, and memory expiry.

### Slice 2: recurrent PPO and survival training proof

- [x] Implement headless smoke, train, resume, and eval CLI modes.
- [x] Add the recurrent local actor, centralized critic, contiguous rollout,
      masked GAE, clipped PPO update, and actor-only evaluation API.
- [x] Prove hidden-state isolation, carry, reset, masking, and checkpoint reload.
- [x] Write metrics, fixed-seed eval, periodic checkpoints, and fresh reload.
- [x] Run smoke, baseline, and tuned training experiments.
- [x] Pass the survival held-out gates and record evidence.

### Slice 3: multi-agent competition

- [x] Add joint actions and individual lifecycle transitions.
- [x] Pool all valid bunny sequences into one parameter-sharing learner.
- [x] Implement direct predecessor checkpoint loading and lineage arguments.
- [ ] Test simultaneous stepping, fair spawn rotation, collision contacts, and
      valid rollout contribution from every agent.
- [ ] Pass competition held-out gates and record evidence.

### Slice 4: predator and prey

- [x] Add fox physiology, local perception, memory, and predation contact.
- [x] Train separate shared bunny and fox policies from the same joint steps.
- [ ] Load the competition bunny checkpoint and record its lineage.
- [x] Test species-specific food rules and terminal transitions.
- [ ] Pass both species' held-out gates and record evidence.

### Slice 5: obstacles and damage tradeoffs

- [x] Generate deterministic non-overlapping trees, rocks, and thorns.
- [x] Keep trees/rocks solid and thorns traversable.
- [x] Implement loading both predator-prey policies for continued training.
- [ ] Test collision penetration, thorn damage, and map reachability.
- [ ] Pass retention, from-scratch-control, and thorn-avoidance gates.

### Slice 6: visualization and durable evidence

- [x] Make every example start visual training without command arguments.
- [x] Add an Inspector-egui learning/physiology HUD, graph, and controls.
- [x] Keep training on a worker thread so the Bevy window stays responsive.
- [x] Require `train`, `headless`, or `--no-default-features` for headless use.
- [ ] Add optional perception-ray and collision debug overlays.
- [x] Add exact checkpoint video sequencing, HUD labels, checkpoint hashes,
      H.264/yuv420p encoding, and typed manifests.
- [ ] Produce four qualifying videos and poster frames.
- [ ] Add video manifests, suite README embeds, and reproduction commands.
- [ ] Inspect every video and poster, not only their metadata.

### Slice 7: final repository gates

- [x] `cargo fmt --all -- --check`
- [x] `cargo test --all-targets`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Run the unchanged personal-lint command recorded by `--dry-run`.
- [ ] Inspect every repository-local lint warning and rerun to zero candidate
      diagnostics.
- [ ] Run training artifact schema and fixture validators.
- [ ] Verify no generated cache or raw run directory is staged.

## Evidence log

Add one row only after a fresh-process held-out evaluation.

| Stage | Training seed | Initialization | Steps | Random mean lifetime | Best test mean lifetime | 95% CI | Gate | Run summary | Video |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| Survival | | random model | | | | | pending | | |
| Competition | | survival best | | | | | pending | | |
| Predator-prey bunny | | competition best | | | | | pending | | |
| Predator-prey fox | | random model | | | | | pending | | |
| Obstacles bunny | | predator-prey best | | | | | pending | | |
| Obstacles fox | | predator-prey best | | | | | pending | | |
