# Ecosystem curriculum handoff

Date: 2026-07-29
Repository: `/home/sagan/Sync/playground/bevy-gym`
Branch: `master`
Baseline commit: `e71f01c Add reusable training core and artifact contracts`
Persistent goal: `019f9f8f-6b82-7a22-a137-e0edfe02414d`

## Read first

Read the repository `AGENTS.md` and these files completely:

- `examples/ecosystem/PLAN.md`
- `examples/ecosystem/RESEARCH.md`
- `examples/ecosystem/EVIDENCE.md`
- `examples/ecosystem/README.md`
- `examples/README.md`

Preserve the dirty worktree. It contains the implementation described here.
Do not discard or reset any existing change.

`runs/`, `target/`, temporary frames, screenshots, and raw diagnostic videos
are run or build artifacts. Do not commit them.

## Objective

Finish and prove the four-stage ecosystem curriculum:

1. Single-agent hunger, thirst, food, finite well-water, health, hit points,
   and survival.
2. Shared-policy multi-agent resource competition with physical
   blocking and pushing.
3. Separately learned bunny and fox policies in a predator-prey environment.
4. Transferred predator-prey policies with trees, rocks, and damaging thorns.

Every stage also needs recurrent visual memory, fixed held-out learning
evidence, a useful HUD, checkpoint videos, and durable documentation.

## Current status

The implementation is substantial but incomplete.

- Every ecosystem example now starts visual training when run without
  arguments. Burn trains on a worker thread while Bevy renders the current
  policy.
- The Inspector-egui HUD provides world and training controls, exact reward,
  PPO metrics, fixed-seed survival time, and live learning curves. It starts
  with agent 0's exact 36 perception sectors visible and supports agent
  filtering.
- The HUD exposes reward weights, starting health, need drain, damage, and
  episode timeout. It also controls 1 through 36 evenly spaced perception rays
  without changing the fixed actor tensor. Training applies changes between PPO
  iterations. The visible world changes only after `Apply + restart visible
  world`.
- Each durable training and evaluation metric records the applied experiment
  profile. Learning-graph markers separate profile changes.
- Headless training requires `train`, `headless`, or
  `--no-default-features`.
- Recurrent PPO now validates an explicit initial state-independent log
  standard deviation. The generic default is `-1.0`; ecosystem training uses
  the verified `-0.5` exploration scale.
- The current survival actor has one-run held-out learning evidence. Its older
  three-seed checkpoints use an incompatible actor record shape.
- Recurrent PPO now weights optimization and diagnostics by valid agent
  timesteps; the mixed 1-step/32-step regression passes.
- Competition and predator-prey have useful diagnostic runs, but their prior
  qualification claims were collected before the P0 corrections below.
- Obstacles have mechanics tests but no learning qualification.
- No curated videos or poster frames are ready to commit.

The recurrent PPO timestep-weighting defect and environment-side stable-ID
resource bias are both fixed and tested. Every multi-agent stage must still be
retrained under the corrected mechanics.

## Implemented code

Four Cargo examples exist:

- `ecosystem-survival`
- `ecosystem-competition`
- `ecosystem-predator-prey`
- `ecosystem-obstacles`

They share a deterministic Avian2D simulation in
`examples/ecosystem/shared/simulation.rs` with:

- dynamic solid bunny and fox bodies;
- static boundaries, well base, trees, and rocks;
- sensor food, well radius, and traversable thorns;
- hunger, thirst, health, hit points, starvation, dehydration, thorn damage,
  and predation;
- finite well capacity, time-based refill, and periodic random food spawning;
- simultaneous joint actions and independent lifecycle termination;
- post-physics semantic raycasts with dense forward and sparse peripheral
  vision;
- fixed local observation and global critic tensors across all stages;
- spawn-slot rotation and per-step contested-resource rotation;
- authoritative Avian `ContactGraph` checks for contact, pushing, and
  predation.

The reusable Burn trainer in `src/training/recurrent_ppo.rs` provides:

- independent LSTM state per live agent;
- centralized critics and decentralized actor inference;
- contiguous recurrent chunks, masked GAE, clipped PPO, and tanh-corrected
  Gaussian actions;
- shared bunny parameters and separate bunny and fox learners;
- MessagePack checkpoints and fresh-process reload;
- deterministic minibatch shuffling with a rollout-specific RNG;
- maximin selection for synchronized predator checkpoints.
- state-independent learned exploration scales initialized away from their
  default clamps.

The default visual command for any stage is:

```sh
cargo run --example ecosystem-survival
```

Use this command for explicit headless training:

```sh
cargo run --no-default-features --release --example ecosystem-survival -- train
```

Render-gated watch and video modes exist. The viewer is top-down and supports
basic shape art, HUD text, pause, reset, pan, and zoom. Video output is
H.264/yuv420p at 1280x720 with a typed SHA-256 manifest.

## P0 correctness corrections

### Recurrent PPO valid-timestep weighting

`RecurrentPpoAgent::update` now multiplies each per-sequence actor and critic
mean by its valid length, divides each minibatch objective by total valid
timesteps, and weights actor loss, critic loss, entropy, and approximate-KL
diagnostics by the same valid-timestep counts across epochs.

`mixed_length_sequences_weight_update_metrics_by_valid_timesteps` compares
separate 1-step and 32-step updates with their worked 1:32 mixed mean and fails
under equal sequence weighting.

### Stable-ID resource priority

Food collection and predation previously selected the minimum stable ID.
Well depletion previously processed agents in ascending ID order.

The environment-side fix is implemented in
`examples/ecosystem/shared/simulation.rs`:

- simultaneous drinkers receive equal water shares;
- contested food priority rotates by simulation step;
- contested fox predation priority rotates by simulation step;
- `contested_resource_priority_rotates_across_agent_identities` covers a
  complete bunny and fox rotation.

The focused tests pass, but all previous multi-agent learning was collected
under the old mechanics and is diagnostic only.

### Video playback timing

All video segments currently advance one simulation step per rendered frame.
At 30 fps this makes every segment about 3x simulation speed. Add an explicit
playback rate or fractional simulation-step accumulator. The 10-second opening
and 20-second ending must show the best checkpoint at 1x. Accelerated
checkpoint segments must be labelled, and their rates must be in the manifest.

## Learning evidence

### Current sparse-food survival evidence

The no-argument current-architecture profile now learns within six updates.
The exact headless command completed 59,971 steps and improved fixed-seed mean
survival from 56.181 to 78.768 seconds. A fresh-process 100-episode evaluation
on seed 10001 improved from 54.367 seconds at step zero to 79.427 seconds for
the selected checkpoint. The learned 95% interval is `[75.163, 83.593]`, and
eat-and-drink success improved from 15% to 56%.

The root cause was an actor refactor that moved exploration into a separate
state-independent parameter but initialized it at the generic `-1.0` default.
The resulting 0.37 action standard deviation was too small for early resource
discovery. Ecosystem training now initializes and caps log standard deviation
at `-0.5`, restoring about 0.61 standard deviation. The selected checkpoint is:

```text
runs/ecosystem-survival-ppo/final-default-headless-20260729a/best.mpk
```

This current run proves learning but does not meet the 70% mechanism or
three-seed curriculum gates. All checkpoints described below predate the
current actor record shape and cannot be loaded by current code.

The sparse-food failure was a long-horizon credit-assignment defect. At the
0.1-second simulation step, `gamma=0.999` and GAE lambda `0.98` retained an
effective trace horizon of only 4.77 seconds, while resource acquisition affects
death tens of seconds later. The ecosystem default is now GAE lambda `1.0`
(100-second effective trace horizon), with validated `--gamma` and
`--gae-lambda` CLI overrides.

Three independent from-scratch runs improved substantially on 100 identical
held-out episodes (`seed=10001`):

| Training seed | Step-zero mean | Best mean | Best 95% CI | Gain | Ate and drank |
| --- | ---: | ---: | --- | ---: | ---: |
| 157 | 53.823 s | 109.875 s | [106.179, 113.340] | 104.1% | 89% |
| 163 | 56.484 s | 86.275 s | [82.364, 90.238] | 52.7% | 62% |
| 269 | 48.671 s | 61.825 s | [59.636, 64.022] | 27.0% | 24% |

All learned confidence intervals are disjoint from and above their step-zero
intervals. Seed 269 remains below the 70% food-and-water mechanism gate, so use
it as lifetime-learning robustness evidence rather than as a predecessor.

The historical selected survival curriculum checkpoint was:

```text
runs/ecosystem-survival-ppo/sparse-full-credit-157/best.mpk
```

Do not use this checkpoint with the current actor implementation.

Its training graph and visually inspected deterministic preview are in the same
ignored run directory as `training-progress.png` and `behavior-preview.mp4`.
The preview shows real map translation to the well and onward rather than
stationary tight-circle spinning.

### Historical survival evidence (pre sparse-food)

The seed-137/151/157 results below were valid under the former survival
configuration, but the rendered seed-157 policy exposed a plentiful-food
spinning exploit. Survival now starts with two food items, never exceeds two,
delays replenishment to the fixed 20-step interval while one item remains, and
immediately restores one item only when necessary to preserve the one-food
floor. These results no longer qualify the current survival environment.

Each result uses 100 unseen episodes:

| Seed | Random mean | Best mean | Best 95% CI | Gain | Ate and drank |
| --- | ---: | ---: | --- | ---: | ---: |
| 137 | 74.886 s | 100.992 s | [97.732, 104.423] | 34.9% | 92% |
| 151 | 57.282 s | 108.118 s | [104.215, 111.679] | 88.7% | 94% |
| 157 | 70.278 s | 117.138 s | [115.596, 118.496] | 66.7% | 99% |

The historical local checkpoint is:

```text
runs/ecosystem-survival-ppo/moderate-exploration-157/best.mpk
```

Do not use it as a current curriculum source. The first controlled sparse-food
seed-157 rerun also remains diagnostic: it completed 48,920 steps without
beating step zero. Fifty held-out episodes measured 53.932 seconds for its
selected step-zero checkpoint and 52.176 seconds for its final checkpoint;
ate-and-drank rates were 12% and 8%.

### Superseded competition diagnostic

The old selected run reported 99.627 seconds mean lifetime over 50 unseen
ecosystems versus 51.932 seconds for random. It also reported 92% ate-and-drank
success, all four IDs consuming resources, 13.6% maximum ID advantage over the
population mean, and 44,540 Avian contact observations.

```text
runs/ecosystem-competition-ppo/selected-regime-251/best.mpk
```

Do not cite this as qualified learning. Retrain from survival seed 157 after
the PPO fix. Establish corrected random baselines and require three independent
qualifying lineages before making a robustness claim.

### Corrected competition diagnostic

The first retraining lineage after both P0 corrections used seed 263, direct
transfer from survival seed 157, twelve four-ecosystem updates, and
`log_std_max=-0.5`:

```text
runs/ecosystem-competition-ppo/corrected-curriculum-263
```

On 50 fixed held-out ecosystems (`seed=10001`), the corrected random,
transferred, and selected policies measured:

```text
random:     51.641 s, 95% CI [50.734, 52.640], ate+drank 2.5%
transfer:   84.094 s, 95% CI [80.765, 87.578], ate+drank 69.0%
curriculum: 80.765 s, 95% CI [77.738, 83.849], ate+drank 64.0%
```

The selected policy remains 56.4% above random, all four identities consume
resources, its maximum identity advantage is 4.5%, and 38,900 contact
observations prove physical competition. It nevertheless regresses 4.0% from
its transferred initialization and retains only 68.9% of the qualified solo
lifetime, so this lineage fails the retention and curriculum-learning gates.
It is diagnostic only.

A separate seed-269 lineage with `log_std_max=-1.0` never exceeded its step-zero
validation score. Its selected checkpoint is therefore exactly the transferred
policy and measures 84.094 seconds held out (71.8% solo retention), not learned
competition improvement. A seed-271 low-variance continuation from the seed-263
checkpoint selected 78.693 seconds held out, 2.6% below its input checkpoint and
67.2% of solo lifetime. Both runs are retained as failed diagnostics:

```text
runs/ecosystem-competition-ppo/corrected-low-variance-269
runs/ecosystem-competition-ppo/corrected-continuation-271
```

The existing optimization regime is therefore not a qualifying corrected
competition recipe. Do not continue either failed checkpoint into
predator-prey; diagnose validation variance and the literal solo-retention gap
without changing the predeclared gates.

### Superseded predator-prey diagnostics

The old complete lineage reported:

```text
random bunnies: 48.111 s
learned bunnies: 68.956 s (+43.3%)
random foxes: 48.084 s
learned foxes: 52.323 s (+8.8%)
bunny ate and drank: 57.3%
fox prey and water: 8.0%
predation events: 48
```

An interrupted maximin continuation later selected 64.056 seconds for bunnies
and 58.275 seconds for foxes over eight validation ecosystems.

```text
runs/ecosystem-predator-prey-ppo/curriculum-311
runs/ecosystem-predator-prey-ppo/maximin-continuation-313
```

These runs inherit the old competition policy and both multi-agent defects.
Do not continue them for final evidence. Start predator-prey from the newly
qualified competition checkpoint.

The final held-out result must improve both species at least 15% over corrected
random means. Bunnies must eat and drink. Foxes must hunt and drink. Held-out
episodes must include both predation and successful evasion.

### Obstacles

Mechanics exist, but learning qualification has not started. Compare
transferred predator-prey policies with a from-scratch control at the same step
budget. Both species must retain at least 85% of predator-prey lifetime. Solid
penetrations must be zero. Thorn damage per minute must fall at least 20% from
random while predator escape events remain nonzero.

## Latest verification

The valid-timestep and fairness-focused gates pass:

```text
cargo test recurrent_ppo --lib
8 passed; 0 failed

cargo test --example ecosystem-competition
23 passed; 0 failed

cargo test --example ecosystem-survival
36 passed; 0 failed

cargo test
53 passed; 0 failed; doc tests passed

cargo test --all-targets
53 library tests, 6 CartPole tests, and 36 tests per ecosystem target passed

cargo fmt --all -- --check
passed

TMPDIR="$PWD/target/tmp" nix develop --command \
  cargo clippy --all-targets --all-features -- -D warnings
passed

git diff --check
passed
```

This shared test binary covers deterministic reset, bounded observations,
well depletion and refill, food capacity, starvation and dehydration removal,
thorn damage, spawn and contested-resource rotation, equal water allocation,
Avian pushing and contact metrics, predation, procedural well reachability,
solid-obstacle penetration, GAE masks, and shared-policy rollout inclusion.

The unchanged `personal-lints --repo .` gate does not pass. Its strict Clippy
phase reports pre-existing CartPole lifetime annotations and four pre-existing
ecosystem functions above its private 100-line threshold: `collect_contacts`,
`global_state`, `run_training_inner`, and `evaluate_policies`. Its Dylint
toolchain cannot compile `wayland-sys` because its pkg-config path lacks
`wayland-client`. Full output is in
`~/.cache/rust-personal-lints/logs/run-1783761-1785309525680`. Do not represent
the personal-lint gate as green.

## Remaining work in order

Before multi-agent retraining, fix the two verified conservation/value-state
issues: cap well withdrawal by absorbable thirst demand, and restrict/repack
food critic slots so configured two-item survival states do not rotate through
all 24 padded slots. Separately improve variance-head/clamp diagnostics before
treating combined actor loss as policy movement.

1. Diagnose the corrected competition validation/retention regression and find
   a qualifying regime from the selected sparse-food survival checkpoint.
2. Establish corrected random and held-out competition results across three
   qualifying training seeds.
3. Train and qualify predator-prey from the new competition checkpoint.
4. Train and qualify obstacles with transferred and matched scratch controls.
5. Add memory-expiry tests and a no-memory ablation.
6. Add valid sample count, optimizer update count, raw policy loss, clip
   fraction, explained variance, and gradient norm.
7. Add lifetime median/min/max, per-species consumption and drinking,
   well-empty fraction, mean food availability, active counts, push contacts,
   penetration, thorn damage per minute, selection score, and evaluation
   learning rate.
8. Add HUD species counts, entropy, validation mean, capacity, and loaded
   checkpoint metadata.
9. Add a collision debug overlay. The perception-ray overlay is complete.
10. Fix video playback timing, labels, and manifest fields.
11. Produce and visually inspect all four curated videos and poster frames.
12. Update `README.md`, `PLAN.md`, `EVIDENCE.md`, and this handoff only
    when their complete gates are satisfied.
13. Remove candidate-local lint suppressions where practical. Preserve typed
    errors and their sources.

## Required final gates

```sh
cargo fmt --all -- --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
personal-lints --repo .
git diff --check
git status --short
```

Also run render-target tests and inspect generated visual artifacts. A manifest
alone does not prove the rendered video is correct.

The personal lint tool is installed at:

```text
/home/sagan/.local/bin/personal-lints
```

## Worktree boundary

Tracked changes include `Cargo.toml`, `Cargo.lock`, training exports and
backends, and compatibility changes in existing DQN/PPO code.

New untracked source and documentation includes all of
`examples/ecosystem`, `examples/README.md`, `HANDOFF.md`, and
`src/training/recurrent_ppo.rs`.

Before the eventual commit, inspect every staged path. Commit only code,
documentation, intentional snapshots and manifests, and curated media. Exclude
all build and run artifacts.

## Shareable continuation prompt

```text
Continue the ecosystem curriculum implementation in
/home/sagan/Sync/playground/bevy-gym.

Read AGENTS.md and HANDOFF.md completely before acting. Preserve the dirty
worktree and every existing implementation change. Do not commit runs, target
output, temporary frames, screenshots, or raw diagnostic media.

Resume from the exact state in HANDOFF.md. Valid-timestep recurrent PPO
weighting and the environment-side stable-ID correction are implemented and
their focused tests pass. Preserve equal water shares and rotating contested
food and predation priority.

With both P0 corrections and current-actor survival learning verified, qualify
two more independent survival seeds. Then retrain competition from
`runs/ecosystem-survival-ppo/final-default-headless-20260729a/best.mpk`.
Treat all pre-correction competition and predator runs as diagnostic only.
Establish corrected random baselines and fixed held-out results across three
competition seeds. Rebuild predator-prey from the new competition checkpoint.
Use synchronized bunny/fox checkpoints and maximin selection. Do not advance a
learning claim until every predeclared resource, fairness, predation, evasion, and non-degeneracy gate passes.

Then qualify obstacles with transferred and matched from-scratch controls.
Finish missing metrics, memory ablation, HUD fields, debug overlays, and video
timing. Create and visually inspect all four required checkpoint videos and
poster frames. Update README, PLAN, EVIDENCE, and HANDOFF honestly. Run every
exact repository gate listed in HANDOFF.md. Keep failed runs as ignored local
evidence. Never move a threshold after seeing results.

Commit only code, documentation, intentional snapshots and manifests, and
curated media. Exclude all build and run artifacts.
```
