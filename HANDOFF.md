# Ecosystem curriculum handoff

Date: 2026-08-02
Repository: `/home/sagan/Sync/playground/bevy-gym`
Branch: `master`
Baseline commit: `e71f01c Add reusable training core and artifact contracts`
Persistent goal: `019fc083-1907-7b71-8b78-6c3e91c22c79`

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

## WASM demo status

The static ecosystem inference demo is implemented under `web/`. It exposes
fragment routes for forage, survival, shelter, competition, predator-prey, and
obstacles. Each route loads the curated manifest default, runs deterministic
Burn Flex inference, and renders the library-owned Avian ecosystem runtime
through Bevy WebGL2. The manifest includes a typed profile, architecture,
training seed, source run, source commit, Burn version, selection evidence,
qualification, content hash, byte length, and configuration sidecar for every
policy. All current defaults are honestly labelled
`best-compatible-available`.

The interface supports pause, restart, 0.25x through 4x playback, bunny policy
uploads, paired bunny and fox uploads on predator routes, and reset to curated
defaults. Uploaded `.mpk` and optional sidecar files remain browser-local. The
browser rejects files over 8 MiB, validates supplied sidecars against the
selected stage, and decodes a complete replacement before activation. A bad
upload keeps the last working policy active.

The verified live-reloading HTTPS server is:

```text
https://nixos.tail87afcc.ts.net:8443/#/ecosystem/survival
```

It runs the loopback Trunk server through
`nix run .#web-serve-tailscale-https`, then uses Tailscale Serve on port 8443.
Trunk rebuilds changed files without a server restart and uses secure WebSocket
reloads through the HTTPS proxy. A tailnet browser verified all six routes,
advancing inference metrics, curated defaults, and canvases on 2026-08-02.

Port 443 remains diverted to the existing Traefik listener. Tailscale Serve is
configured on port 8443 with:

```sh
sudo tailscale serve --bg --https=8443 8080
```

The Tailscale configuration persists independently of the Trunk process. Restart
the live-reloading backend with `nix run .#web-serve-tailscale-https` if needed.

The pure release contains a 32,880,630-byte WASM file. Its gzip size is
8,472,059 bytes. The eight checkpoint records total 4,532,856 bytes. A
single-policy route transfers 566,607 checkpoint bytes. A paired route
transfers 1,133,214 checkpoint bytes. One Chromium survival run measured 56.5
ms checkpoint decode, 742.9 ms to the first active rendered frame, and 19.305
ms mean active frame time. These measurements describe this machine and are
not performance guarantees.

The following focused demo gates pass against the current files:

```text
cargo test -p bevy-gym-web
cargo clippy -p bevy-gym-web --all-targets --all-features -- -D warnings
nix develop -c cargo check --manifest-path web/Cargo.toml --target wasm32-unknown-unknown --bin bevy-gym-web
nix run .#web-check
```

The browser suite covers all six routes for at least five seconds, visible
canvas changes, performance attributes, raw upload and reset, paired upload
with matching sidecars, malformed-upload recovery, pause, restart, speed, and
browser errors. It writes inspected survival and predator-prey screenshots to
`test-results/wasm-demo-survival.png` and
`test-results/wasm-demo-predator-prey.png`.

The repository-wide strict Clippy gate still fails on unrelated dirty-worktree
code. Its remaining failures are the global unused `shakmaty` dependency in
more than twenty example and test crates, plus the ecosystem qualification
function's `similar_names` and `too_many_lines` diagnostics. The personal lint
gate also remains red because both phases cannot find `wayland-client` through
their isolated pkg-config paths. Its current log is
`~/.cache/rust-personal-lints/logs/run-2760077-1785686423040`. `nix flake
check` reaches a treefmt failure caused by 51 existing formatting differences
across the dirty worktree. Do not rewrite those unrelated files without
approval.

## Objective

Finish and prove the eight-stage ecosystem curriculum:

1. Single-agent food perception and approach.
2. Ephemeral-food sprinting with a speed-sensitive reward.
3. Alternating-bank food reached through a lethal-gorge bridge.
4. Single-agent hunger, thirst, food, finite well-water, health, hit points,
   and survival.
5. Single-agent weather exposure and shelter use.
6. Shared-policy multi-agent resource competition with physical
   blocking and pushing.
7. Separately learned bunny and fox policies in a predator-prey environment.
8. Transferred predator-prey policies with trees, rocks, and damaging thorns.

Every stage also needs recurrent visual memory, fixed held-out learning
evidence, a useful HUD, checkpoint videos, and durable documentation.

## Current status

The implementation is substantial but incomplete.

- Sprint food expires after five simulated seconds and gives a larger reward
  for earlier contact. Gorge food retains that deadline and alternates banks.
  The gorge kills agents outside the visible rail-guided bridge corridor.
- Survival playback now defaults to 1x and interpolates between fixed physics
  states. The visible world pauses for one wall-clock second after each episode.
- Survival collects nine independent environments in parallel for each PPO
  update. Every environment supplies recurrent sequences and returns to that
  update. The visual demo replays the same nine trajectories in a labelled 3x3
  grid with pan and zoom. All nine lanes use survival maps and mechanics.
  Startup shows a labelled looping nine-environment preview while offline
  survival initialization runs, then replaces it with contributing batches.
  Completed traces keep looping while the worker collects, evaluates, or runs
  PPO, so optimizer work does not freeze the renderer.
- The training graph retains every episode return. Fixed-seed evaluation means
  remain a separate series.
- Survival episodes regenerate five solid obstacles. Agents cross the shallow
  well at half speed.
- Food, water, and prey interactions require an active forward mouth hitbox.
  Solid impacts apply speed-scaled damage from Avian contact data.
- Agents turn at up to 8 radians per second. Ground traction removes lateral
  sliding. Each agent HUD shows episode return and fading signed reward deltas.
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

A five-second nine-environment collection test took 0.36 seconds with parallel
collection and 1.03 seconds with a temporary sequential strategy on this
machine. The final code uses parallel collection. A one-update visual smoke run
recorded 450 joint steps and nine optimizer updates. It showed all nine
environments in a uniform 3x3 grid, and manual pan and zoom worked.

The no-argument survival command now uses 16 PPO updates, nine 120-second
episodes per update, 16 validation episodes every fourth update and after the
final update, and 1x playback. The accepted run completed 172,800 training steps
in 16 minutes 57 seconds:

```text
runs/ecosystem-survival-ppo/final-homeostasis-profile-20260801/best.mpk
```

All 16 selected validation episodes survived 120 seconds. Their mean terminal
health and satiation were 100%, mean hydration was 72.5%, and 43.75% ended with
all three stats at maximum. A disjoint 100-episode 120-second evaluation had
100% survival, 100% mean terminal health, 95.6% satiation, 72.8% hydration, and
31% exact maximum terminal stats.

A disjoint 100-episode 300-second stress test had 85% survival. It recorded 15
dehydration deaths. The 300-second continuation in
`runs/ecosystem-survival-ppo/final-long-horizon-20260801` reached 100% on its 16
selection episodes but only 80% on 100 disjoint episodes, so it is rejected.
Current evidence does not prove indefinite survival.

The recurrent PPO timestep-weighting defect and environment-side stable-ID
resource bias are both fixed and tested. Every multi-agent stage must still be
retrained under the corrected mechanics.

## Implemented code

Eight stage Cargo examples exist:

- `ecosystem-forage`
- `ecosystem-sprint`
- `ecosystem-gorge`
- `ecosystem-survival`
- `ecosystem-shelter`
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
