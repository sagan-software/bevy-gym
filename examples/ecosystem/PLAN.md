# Ecosystem continual curriculum implementation plan

Status: ready for implementation

This plan replaces the four-stage reward-bonus curriculum with an eight-stage
healthy-survival curriculum. Completion requires held-out quantitative evidence,
retention evidence across earlier stages, freshly loaded checkpoints, eight
30-second stage videos, and one 280-second aggregate video.

The controlling research is in
[CONTINUAL-CURRICULUM-RESEARCH.md](CONTINUAL-CURRICULUM-RESEARCH.md). Existing
implementation research remains in [RESEARCH.md](RESEARCH.md) and
[RL_ECOSYSTEM_RESEARCH.md](RL_ECOSYSTEM_RESEARCH.md).

## Desired outcome

Build one reproducible curriculum with these Cargo examples:

1. `ecosystem-forage`: one bunny learns to perceive, approach, and eat food.
2. `ecosystem-sprint`: the bunny reaches ephemeral food quickly and accurately.
3. `ecosystem-gorge`: the bunny crosses a visible bridge to alternating food.
4. `ecosystem-survival`: the bunny regulates hunger and thirst with food and water.
5. `ecosystem-shelter`: the bunny retains foraging while using shelter during exposure.
6. `ecosystem-competition`: a shared bunny policy survives fair resource scarcity.
7. `ecosystem-predator-prey`: transferred bunnies and a separate fox policy co-adapt.
8. `ecosystem-obstacles`: both species retain prior skills in the full ecosystem.

Add `ecosystem-curriculum` as the suite runner. One command trains the stages in
order, records predecessor lineage, evaluates retention, and writes a suite
manifest. A separate command renders every stage video and the aggregate video.

The word `proof` in this plan means repeatable empirical evidence under declared
seeds and gates. It does not mean a mathematical proof that the policies work in
every possible world.

## Acceptance checklist

- [ ] Eight stage examples use one stable observation and action contract.
- [ ] Later stages load the required qualifying predecessor actor and LSTM.
- [ ] PPO collections contain fresh current-policy rollouts from earlier stages.
- [ ] Reward implements one declared healthy-survival objective across all stages.
- [ ] Raw distance, movement, collision, pushing, and attack rewards are absent.
- [ ] Every stage passes its held-out behavior and retention gates for three seeds.
- [ ] Every selected checkpoint reloads in a fresh process and reproduces its result.
- [ ] Every stage has a manifest-bound 30-second progression video.
- [ ] The aggregate video has the exact 280-second structure defined below.
- [ ] Agents, status, perception, hitboxes, hurtboxes, and physics are visually auditable.
- [ ] Focused tests, full tests, strict Clippy, coverage, and personal lints close.
- [ ] `HANDOFF.md`, `EVIDENCE.md`, and `README.md` match the final artifacts.

## Invariant scratchpad

### Closed states and modes

- `CurriculumStage`: `Forage`, `Sprint`, `Gorge`, `Survival`, `Shelter`,
  `Competition`, `PredatorPrey`, `Obstacles`.
- `Species`: `Bunny`, `Fox`.
- `AgentLife`: `Alive`, `Dead(DeathCause)`.
- `HealthPresentation`: `Healthy`, `Hurt`, `Critical`, `Dead`.
- `DeathCause`: `Starvation`, `Dehydration`, `Exposure`, `Predation`, `Thorns`,
  `CombinedDamage`, `InvalidPhysics`.
- `InteractionIntent`: `Idle`, `Active`.
- `InteractionKind`: `Eat`, `Drink`, `Shelter`, `Attack`.
- `ColliderRole`: `SolidBody`, `Hitbox(InteractionKind)`, `Hurtbox(Species)`,
  `Resource`, `Shelter`, `Hazard`, `Boundary`.
- `RunMode`: `Demo`, `Train`, `Eval`, `Watch`, `Video`, `Smoke`.
- `SuiteMode`: `Train`, `Eval`, `Video`.
- `CheckpointRole`: `StepZero`, `Early`, `Middle`, `Best`.
- `EvalSuite`: `Validation`, `Test`, `Retention`, `Demo`.

### Legal stage transitions

```text
Forage
  -> Sprint
      -> Gorge
          -> Survival
              -> Shelter
                  -> Competition
                      -> PredatorPrey
                          -> Obstacles
```

Each transition is fallible. It requires a compatible predecessor checkpoint,
its immutable experiment profile, and a passing predecessor evidence record.
Diagnostic from-scratch runs are legal but cannot qualify as curriculum transfer.

### Validated boundary values

- Stage, checkpoint role, species, death cause, collider role, and run mode use
  closed enums.
- Agent count, horizon, map size, resource capacity, refill rate, checkpoint
  interval, video duration, FPS, and curriculum mixture validate before use.
- Seeds use disjoint training, validation, test, retention, and demo streams.
- Checkpoint loading validates profile version, stage, observation width, action
  width, ray order, network widths, reward version, and predecessor digest.
- Video manifests validate every checkpoint path, SHA-256 digest, frame count,
  stage, role, seed, playback rate, and experiment profile before capture.

### Derived values

- Healthy-survival reward derives from the post-transition physiology state.
- Resource feedback derives from actual reduction in normalized drive.
- Health presentation derives from life state and HP fraction.
- Alive lists derive from lifecycle state.
- Retention status derives from frozen stage gate results.
- Curriculum promotion derives from three consecutive validation passes.
- Video duration derives from segment frame counts and FPS.
- Aggregate ordering derives from the suite stage order.

### Failure boundaries

- Malformed CLI, stage config, suite manifest, or checkpoint input is a setup error.
- Incompatible checkpoint shape, reward version, or lineage is a compatibility error.
- A failed quantitative or retention gate is a completed experiment, not success.
- Missing `ffmpeg`, render support, frames, or checkpoints is a video error.
- A video cannot replace quantitative evidence.
- A mechanics change invalidates every result collected under the previous profile.

## Module and file map

Preserve the current brownfield shared implementation. Add focused modules for
new durable responsibilities instead of expanding the existing catch-all files.

```text
examples/ecosystem/
  forage.rs                      # stage entry point
  survival.rs                   # existing stage entry point
  shelter.rs                    # stage entry point
  competition.rs                # existing stage entry point
  predator_prey.rs              # existing stage entry point
  obstacles.rs                  # existing full-ecosystem entry point
  curriculum.rs                 # suite train/eval/video entry point
  shared/
    mod.rs                       # private module wiring and narrow re-exports
    curriculum.rs               # stage graph, mixtures, promotion, suite manifest
    reward.rs                   # normalized drive and healthy-survival reward
    interaction.rs              # typed hitbox/hurtbox contact resolution
    qualification.rs            # typed gates and retention evaluation
    curriculum_video.rs         # aggregate capture and concat manifest
    domain.rs                   # remaining established simulation values
    simulation.rs               # Avian world lifecycle and joint step
    training.rs                 # recurrent PPO collection and stage training
    rendering.rs                # programmer-art world and status presentation
    video.rs                    # one-stage checkpoint progression capture
    demo.rs                     # interactive visual training
    rng.rs                      # deterministic stage seed streams
  media/
    forage-training.mp4
    survival-training.mp4
    shelter-training.mp4
    competition-training.mp4
    predator-prey-training.mp4
    obstacles-training.mp4
    ecosystem-curriculum.mp4
    *.manifest.json
```

`Cargo.toml` only receives the three target registrations needed for
`ecosystem-forage`, `ecosystem-shelter`, and `ecosystem-curriculum`. Do not edit
other example targets while the concurrent Gymnasium agent is working.

## Stable policy contract

### Action

Use one four-value bounded continuous action in every stage:

```text
[forward_acceleration, body_turn_rate, gaze_yaw, interaction_intent]
```

The first three values remain in `[-1, 1]`. `interaction_intent > 0` activates
the species-appropriate mouth or attack hitbox subject to its cooldown. Forage
teaches this action before it is needed for drinking or predation.

### Perception

Replace duplicate left-eye rays with one shared ray origin centered between the
two rendered eyes. Reserve 24 ordered rays across a 120-degree gaze-relative fan.
Use denser central angles and sparser peripheral angles. Each ray returns nearest
normalized hit distance and one one-hot semantic kind. The semantic channels
also establish hit presence. Solid bodies occlude resources and agents.

The actor receives no world coordinates, target identity, path distance, hidden
resource state, species flag, curriculum stage, or other agent physiology. It
receives typed proprioception, physiology, interaction cooldown, and ray channels.
Food and well have no separate direction or proximity summaries. The critic may
receive padded global state during training only.

The observation width, ray order, action width, LSTM width, and network widths
are identical in all eight stages. This change increments the checkpoint profile
and deliberately invalidates current ecosystem checkpoints.

### Memory

Each living agent owns an independent LSTM state. State starts at zero and resets
on death, truncation, or environment reset. Rollout chunks remain contiguous.
Padding and dead slots do not contribute to losses or diagnostics. Fresh process
evaluation starts with zero recurrent state.

## Avian2D physics and interactions

- Use Avian2D dynamic solid bodies for bunnies and foxes.
- Use Avian2D static solid bodies for walls, trees, rocks, and the well base.
- Keep agent-agent solid collision and friction so blocking and pushing emerge.
- Attach a child hurtbox sensor to each living agent.
- Attach a forward child hitbox sensor for eating, drinking, and fox attacks.
- Activate interaction hitboxes only while `InteractionIntent::Active` and off cooldown.
- Use resource, well, shelter, and thorn sensors for semantic overlap events.
- Resolve contacts after the fixed physics step from Avian contact data.
- Apply every joint action from the same pre-step state.
- Resolve simultaneous claims with equal shares or rotating fair priority.
- Disable hitboxes and hurtboxes immediately when an agent dies.
- Keep presentation geometry aligned with every collider shape and offset.

Mechanics tests must prove solid blocking, displacement through pushing, sensor
non-blocking, mouth-range eating, well-range drinking, attack-range predation,
cooldown enforcement, fair simultaneous claims, and dead-agent noninteraction.

## Physiology and reward

Define normalized drive after each transition:

```text
D(s) = w_food     * food_deficit(s)^2
     + w_water    * water_deficit(s)^2
     + w_hp       * injury_fraction(s)^2
     + w_exposure * exposure_deficit(s)^2

sum(active weights) = 1
0 <= D(s) <= 1
```

Use one stable task reward:

```text
r_task(s, s') = alive(s') * (dt / reference_lifetime) * (1 - rho * D(s'))
0 <= rho <= 1
```

Optional direct resource feedback uses only absorbed physiological benefit:

```text
r_resource = eta * max(0, D(before_consumption) - D(after_consumption))
```

The same formulas apply in every stage. An inactive need has zero weight and the
remaining active weights renormalize. Food feeds bunnies. Prey feeds foxes.
Shelter reduces exposure. No reward uses distance, raw motion, collision,
pushing, damage dealt, exploration, target contact without absorption, or a
separate kill bonus.

Before implementation, add counterfactual return tests for:

- full healthy survival without unnecessary consumption;
- urgent resource use followed by death;
- stable homeostasis versus repeated consumption;
- cautious survival versus risky hoarding;
- shelter recovery versus remaining exposed;
- fox feeding versus a kill that provides no absorbed food.

The intended healthy-survival behavior must have the greatest return in every
applicable comparison. Keep total return scale comparable between stages.

## Stage definitions and gates

All gates use unseen test seeds after validation selects `best.mpk`. Single-agent
stages use at least 100 test episodes. Multi-agent stages use at least 50 complete
ecosystem episodes. Three independent training seeds must pass before a stage is
called robust.

### 1. Forage

- One bunny, one food need, one or two visible food items, no water or exposure.
- Begin with short central placement. Expand bearing and distance after promotion.
- Randomize left, right, near, and far placement on held-out seeds.
- Gate: at least 90% of test episodes contain one useful eating event.
- Gate: left and right targets produce opposite median turn directions.
- Gate: food-channel ablation reduces eating success by at least 50 percentage points.
- Gate: no reward is earned for eating at a full setpoint.

### 2. Survival

- Transfer the forage bunny actor and LSTM.
- Activate thirst, finite well water, food replenishment, and longer horizons.
- Randomize resource order, bearing, and separation independently.
- Gate: at least 80% of test episodes contain both eating and drinking.
- Gate: mean healthy lifetime improves at least 30% over step zero.
- Retention gate: forage success remains at least 90% of its predecessor result.

### 3. Shelter

- Transfer the survival bunny actor and LSTM.
- Add exposure that rises during weather and falls inside shelter.
- Keep food and water active. Randomize shelter position and weather onset.
- Gate: at least 80% of weather episodes contain shelter use before critical exposure.
- Gate: exposure deaths fall at least 50% relative to step zero.
- Retention gate: eating-and-drinking success remains at least 90% of survival.

### 4. Competition

- Transfer the shelter bunny actor and LSTM into one shared four-bunny policy.
- Begin with enough resources for all. Reduce per-agent supply after promotion.
- Keep physical blocking, pushing, finite water, shelter, and fair claims.
- Gate: mean healthy lifetime improves at least 15% over the stage's step zero.
- Gate: every identity consumes food and water across the test suite.
- Gate: no stable identity lifetime advantage exceeds 20% of population mean.
- Gate: agent-agent contact and displacement events are both nonzero.
- Retention gate: solo shelter performance remains at least 85% of predecessor.

### 5. Predator-prey

- Transfer the competition bunny actor and LSTM.
- Pretrain a new fox against slow frozen or scripted prey until predation is discoverable.
- Alternate current-species update windows after fox acquisition qualifies.
- Sample opponents from current, early, middle, and best historical policies.
- Gate: both species improve healthy lifetime at least 15% over their step zero.
- Gate: bunnies eat, drink, and use shelter. Foxes eat prey, drink, and use shelter.
- Gate: predation occurs in 10% through 90% of test ecosystems.
- Gate: the cross-play matrix has no single historical opponent blind spot below
  70% of the current same-generation score.
- Retention gate: competition and solo behavior stay above their declared floors.

### 6. Obstacles and full ecosystem

- Transfer both predator-prey actors and LSTMs.
- Add solid trees and rocks, traversable damaging thorns, occlusion, weather,
  procedural resource placement, scarcity, shelter, pushing, and predation.
- Gate: solid-obstacle penetration count is zero.
- Gate: thorn damage per minute falls at least 20% from step zero.
- Gate: both species improve healthy lifetime over equal-budget from-scratch controls.
- Retention gate: each species retains at least 85% of predator-prey performance.
- Retention gate: every earlier frozen stage suite remains above its floor.

If a gate fails, keep the run and hypothesis. Change one environment or training
factor, increment the experiment profile when mechanics change, and rerun the
same frozen gate. Do not lower a threshold after observing failure.

## Continual training and transfer

PPO remains on-policy. Each iteration collects new trajectories with the current
policy. Do not replay stale PPO transitions.

Start each new stage with this rollout distribution:

```text
70% current stage
20% direct predecessor
10% uniformly sampled earlier stages
```

Treat this mixture as a starting experiment profile. Increase earlier-stage
sampling when a retention gate declines. Promote difficulty after three
consecutive validation passes. Retain easier and harder variants in the active
distribution after promotion.

At each stage transition:

1. Freeze the qualifying predecessor checkpoint and evidence.
2. Load the actor encoder and LSTM.
3. Reset optimizer state after a material distribution change.
4. Reset the critic when reward channels or global-state meaning change.
5. Warm the new critic briefly with the actor frozen only if measured value error
   destabilizes the transferred actor.
6. Select checkpoints by current-stage score subject to every retention floor.

Add regularization or modular policies only after fresh-rollout rehearsal fails
to retain prior stages under a measured ablation.

## Programmer-art presentation

Use only Bevy meshes, sprites, colors, and text. Do not add external art assets.

- Bunny: light oval body, two ears, two visible eyes, and a short mouth marker.
- Fox: orange body, triangular ears, two visible eyes, and a forward attack marker.
- Healthy: full species color with complete or near-complete bars.
- Hurt: red outline and one brief damage pulse.
- Critical: persistent slow red pulse below 25% HP.
- Dead: gray body, crossed eyes, empty HP bar, and disabled hitbox marker.
- Status bars: HP red, food brown, water blue, exposure purple.
- Interaction: a short forward arc while the mouth or attack hitbox is active.
- Food: green circle. Well: blue circle and visible capacity ring.
- Shelter: tan outlined area with a roof marker. Thorns: purple circles.
- Trees and rocks: solid green and gray shapes matching their Avian colliders.
- Perception: toggleable rays from the shared point between the eyes.

The HUD shows stage, checkpoint role, seed, playback rate, simulation time,
species counts, deaths, resource counts, weather, reward, lifetime, retention
status, and PPO diagnostics. Video overlays identify every stage and checkpoint.

## Video evidence contract

Render H.264/yuv420p at 1280x720 and 30 FPS. Every captured segment uses a fixed
demo seed, a freshly loaded immutable checkpoint, zero LSTM state, and a visible
label. The manifest records the exact checkpoint digest, experiment profile,
seed, playback rate, frame range, and evaluation metrics.

Each stage progression video is exactly 30 seconds:

```text
0-5s    step-zero or worst checkpoint
5-10s   early checkpoint near one-third of training
10-15s  middle checkpoint near two-thirds of training
15-30s  validation-selected best checkpoint
```

If a required checkpoint role does not exist, video generation fails. It does
not silently duplicate another role. Playback defaults to 1x. Any accelerated
segment must display its rate and record it in the manifest.

The aggregate video begins with eight best-checkpoint previews and then concatenates
the complete stage progression videos:

```text
0-5s      forage best
5-10s     sprint best
10-15s    gorge best
15-20s    survival best
20-25s    shelter best
25-30s    competition best
30-35s    predator-prey best
35-40s    obstacles best
40-70s    forage progression
70-100s   sprint progression
100-130s  gorge progression
130-160s  survival progression
160-190s  shelter progression
190-220s  competition progression
220-250s  predator-prey progression
250-280s  obstacles progression
```

The aggregate manifest references all eight stage manifests and verifies their
SHA-256 digests. `ffprobe` must report 280 seconds within one
frame of tolerance, H.264 video, yuv420p, 1280x720, and 30 FPS.

## Implementation sequence

### Slice 0: plan, baseline, and red tests

- [ ] Record current ecosystem-only diff and concurrent-work boundary.
- [ ] Run the current focused ecosystem test baseline.
- [ ] Add failing public-seam tests for eight stages, stable tensor shape, reward
  counterexamples, hitbox/hurtbox contacts, retention selection, and video order.
- [ ] Record genuine failures caused by missing new behavior.

### Slice 1: eight-stage domain and healthy reward

- [ ] Add the stage graph, new entry points, stable four-value action, eight-stage
  observation profile, normalized drive, and reward tests.
- [ ] Remove the current food/drink bonus scale and terminal survival lump.
- [ ] Search for forbidden distance and auxiliary behavior rewards.
- [ ] Run forage and survival focused tests.

### Slice 2: Avian interactions and perception

- [ ] Implement shared-origin rays and occlusion tests.
- [ ] Add solid bodies, child hitboxes, hurtboxes, interaction cooldowns, and
  collision-layer tests.
- [ ] Prove pushing, fair simultaneous claims, and dead-agent disablement.
- [ ] Run every render-disabled ecosystem example test target.

### Slice 3: shelter and continual collection

- [ ] Add exposure, weather, shelter recovery, rendering, and tests.
- [ ] Add current-policy stage mixtures and typed retention evaluation.
- [ ] Add suite manifests and predecessor digest validation.
- [ ] Prove actor transfer, critic reset policy, and earlier-stage rollout contribution.

### Slice 4: competition and predator-prey

- [ ] Requalify fair scarcity under the healthy reward.
- [ ] Add fox acquisition against frozen prey, alternating updates, opponent pools,
  and cross-play evidence.
- [ ] Prove species-specific interactions and non-degenerate outcomes.

### Slice 5: full ecosystem

- [ ] Transfer both species into obstacles, hazards, weather, and procedural maps.
- [ ] Add from-scratch equal-budget controls and all-stage retention evaluation.
- [ ] Close every stage gate for three training seeds.

### Slice 6: video and documentation

- [ ] Extend stage video capture to the exact four checkpoint roles and 30 seconds.
- [ ] Add the aggregate preview and concat pipeline with typed manifests.
- [ ] Render and inspect every stage video and the aggregate video.
- [ ] Update `README.md`, `EVIDENCE.md`, and repo-root `HANDOFF.md` in a final
  prose-only edit after code and tests are green.

## Required verification

Run cheap focused gates after each slice. After the final edit, run every command
again with the exact scope shown.

```sh
cargo fmt --all -- --check
cargo test --example ecosystem-forage --all-features
cargo test --example ecosystem-survival --all-features
cargo test --example ecosystem-shelter --all-features
cargo test --example ecosystem-competition --all-features
cargo test --example ecosystem-predator-prey --all-features
cargo test --example ecosystem-obstacles --all-features
cargo test --example ecosystem-curriculum --all-features
cargo test
cargo clippy --all-targets --all-features -- -D warnings
personal-lints --repo .
git diff --check
```

Run the Rust skill audits against changed ecosystem Rust with zero unexplained
findings. Run the repository's configured coverage workflow if one exists. If no
coverage command exists, use `cargo llvm-cov` when installed and inspect every
reachable production branch added by this change.

Run direct reward and artifact checks:

```sh
rg -n 'distance.*reward|reward.*distance|movement.*reward|collision.*reward|push.*reward|kill.*reward' examples/ecosystem
ffprobe -v error -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate -show_entries format=duration -of json examples/ecosystem/media/ecosystem-curriculum.mp4
sha256sum examples/ecosystem/media/*.mp4 examples/ecosystem/media/*.manifest.json
```

Inspect the first, middle, and last frame of every stage segment. Tests and
manifests do not prove that status bars, collision geometry, labels, and behavior
are visually legible.

## Concurrency boundary

Another agent owns the broader Gymnasium example refinement. Before every patch,
inspect `git status --short -- examples/ecosystem Cargo.toml`. Write only the
ecosystem files named in this plan and the three required Cargo target blocks.
Do not format, stage, revert, or repair unrelated example files. If a concurrent
edit touches the same ecosystem lines, stop that slice, preserve both versions,
and reconcile from the current file instead of overwriting it.
