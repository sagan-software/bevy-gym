# Plan: complete Gymnasium Classic Control and Toy Text examples

- Status: planning only
- Research date: 2026-07-11
- Implementation baseline: Gymnasium 1.3.x; the repository currently pins official Gymnasium
  commit `7a1191388aa4aa973d3a5e4b039899cd99cc991f` under `ref/gymnasium`.

## Outcome

Ship all nine environments shown on Gymnasium's current Classic Control and Toy Text catalog pages
as real, registered Cargo examples. Every example must:

1. implement the current official environment semantics;
2. train a policy in first-party Rust code through the `bevy-gym` training surface;
3. write reproducible checkpoints, metrics, and provenance;
4. prove learning by reloading a checkpoint in a fresh evaluation process;
5. render with the same recognizable composition, palette, dimensions, and sprites as the official
   Gymnasium environment;
6. produce a shareable training-progress timelapse from chronological policy checkpoints; and
7. have tests, documentation, and automation that prevent a documentation-only stub or an
   unsubstantiated demo from being called complete.

The result is nine example binaries and nine demo/proof bundles, one for each documented environment
page.

## Scope decisions

### Included environment pages

Classic Control:

- `Acrobot-v1` -> Cargo target `acrobot`
- `CartPole-v1` -> Cargo target `cartpole`
- `MountainCar-v0` -> Cargo target `mountain_car`
- `MountainCarContinuous-v0` -> Cargo target `mountain_car_continuous`
- `Pendulum-v1` -> Cargo target `pendulum`

Toy Text:

- `Blackjack-v1` -> Cargo target `blackjack`
- `CliffWalking-v1` -> Cargo target `cliff_walking`
- `FrozenLake-v1` -> Cargo target `frozen_lake`
- `Taxi-v4` -> Cargo target `taxi`

### Required configurations, not extra demo binaries

- `FrozenLake8x8-v1` is implemented and proven through the `frozen_lake` target.
- `CliffWalkingSlippery-v1` is implemented and proven through the `cliff_walking` target.
- `Blackjack-v1` defaults to the registered `sab=true, natural=false` rules; supported rule options
  receive conformance tests.
- Taxi's rainy and fickle-passenger options receive conformance tests, while the default `Taxi-v4`
  configuration is the release-demo configuration.

### Excluded from this plan

- Legacy `CartPole-v0`; supporting its 200-step time limit may be cheap once a generic time-limit
  wrapper exists, but it does not get another video or block the nine-page milestone.
- Box2D, MuJoCo, Phys2D/JAX, Atari, robotics, and third-party environments.
- Python, Stable-Baselines3, or another external trainer as the shipping implementation. Python
  Gymnasium is an oracle for fixtures and comparison only.
- Rendering during the hot training loop. Training remains headless; rendering replays saved
  checkpoints afterward.
- Changing official dynamics, rewards, termination rules, or observation/action spaces to make an
  algorithm pass.

## Definition of done

The milestone is complete only when all of the following are true:

- `cargo metadata` lists the nine named example targets.
- Every environment passes reset/step, reward, termination/truncation, space, stochastic-transition,
  and relevant-configuration parity tests derived from the pinned Gymnasium source.
- Every target supports the same `train`, `eval`, `video`, and `watch` command shape.
- Qualifying runs contain real optimizer or Q-table updates, have imitation/warmup disabled, and
  show that the learned policy differs from the step-0 policy.
- A fixed validation suite selects `best.mpk`; a disjoint, fixed test suite evaluates that exact
  file after fresh-process reload.
- Five independent training seeds are run per release configuration; at least four pass and the
  median final result passes the declared gate.
- Every proof receipt includes environment/version, algorithm, source commit, Gymnasium oracle
  commit, exact configuration and command, all seed partitions, elapsed time, update count, step or
  episode count, checkpoint SHA-256, evaluation statistics, threshold, and stop reason.
- `watch` and `video` use the same environment renderer; the current CartPole dual-renderer drift is
  removed.
- Every official scene viewport passes visual-regression and manual side-by-side review.
- Every environment has a chronological 30-second H.264/yuv420p timelapse, poster frame, and segment
  manifest that links frames to real checkpoint hashes and evaluation rows.
- Final demo artifacts live outside ignored `runs/` data and can be opened or shared without this
  developer's local working tree.
- PR checks pass, the full proof matrix passes in nightly/release automation, and README/docs link
  to all nine proof bundles.

## Current-state audit

### Summary

| Area                     | Current state                                                                                                                                                                                                       | Consequence                                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Catalog coverage         | All five Classic Control and four Toy Text filenames exist.                                                                                                                                                         | The intended file layout is already present.                                                        |
| Cargo discovery          | Only `cartpole` is registered in `Cargo.toml`.                                                                                                                                                                      | The other eight files are not runnable Cargo examples.                                              |
| Eight non-CartPole files | Each contains only copied `//!` environment documentation and no executable Rust.                                                                                                                                   | Treat them as unimplemented, not partially complete.                                                |
| CartPole                 | Real environment, Burn DQN, train/eval/checkpoints, Bevy watch mode, screenshots, and timelapse exist in one 1,986-line file.                                                                                       | It is the reference vertical slice, but reusable pieces must be extracted.                          |
| Training library         | Burn backend, run paths, seed types, tensor validation, metrics, and checkpoint utilities exist.                                                                                                                    | Useful foundation is already in place.                                                              |
| DQN/PPO library APIs     | `src/training/dqn.rs` and `ppo.rs` are configuration/report scaffolds only.                                                                                                                                         | The functional DQN is trapped in CartPole; PPO is not implemented.                                  |
| Continuous actions       | Core spaces support bounded continuous actions, but trainer tensorization exposes only discrete action specs.                                                                                                       | Pendulum and continuous Mountain Car are blocked on a real continuous-action trainer path.          |
| Bevy-native collection   | The runner can step environment entities in parallel and emits typed transitions. CartPole training directly steps one local environment instead.                                                                   | Training must be connected to the Bevy runner to prove the product's parallel training path.        |
| Official reference       | `ref/gymnasium` already contains current source, docs videos, procedural render code, and all bundled sprites.                                                                                                      | No rediscovery or visual guessing is needed. Runtime code must not depend on the submodule.         |
| Local assets             | There is no first-party assets directory.                                                                                                                                                                           | Cleared assets must be promoted into a shippable location with attribution.                         |
| CartPole proof           | A local ignored run reports mean reward `34.17 -> 500.0`, no warmup, and threshold crossing at 80,000 steps. Direct same-seed fresh eval on this checkout measured step-0 `40.1` and best `500.0` over 20 episodes. | Learning is genuinely demonstrated locally, but the proof is not yet durable or fully reproducible. |
| CartPole video           | A valid 30-second, 600x400, 50 FPS H.264 timelapse exists.                                                                                                                                                          | The checkpoint-replay concept is proven. It is ignored and visually drifts from Gymnasium.          |
| Tests                    | `cargo test --all-targets` passes 23 tests.                                                                                                                                                                         | The current library is green, but none of the eight stubs or environment parity surfaces is tested. |
| GitHub CI                | It runs check/clippy with nonexistent root feature `x11`; the live command fails immediately.                                                                                                                       | CI must be repaired before it can be treated as evidence.                                           |
| Nix checks               | They check the library and CartPole render compilation, not all targets, training, reload, or video.                                                                                                                | Add layered smoke and proof checks.                                                                 |

### Already useful and to preserve

- The small `Env`, `Reset`, `Step`, `Transition`, and `EpisodeStatus` contract.
- Correct distinction between `Terminated` and `Truncated` for bootstrapping.
- `BevyGymPlugin`, parallel environment entities, typed action request/response messages,
  `TransitionEvent`, and `EpisodeEndEvent`.
- Burn `Flex` inference and `Autodiff<Flex>` training backends.
- Stable `runs/<env>-<algorithm>/<run-id>/` naming and existing config, seed, metrics, eval,
  summary, latest-checkpoint, and best-checkpoint concepts.
- CartPole's chronological `step-*.mpk` snapshots and ffmpeg frame-pipe proof of concept.
- Headless-by-default features with rendering gated behind `render`/`bevy_remote`.
- The family layout under `examples/classic-control` and `examples/toy-text`.

### CartPole gaps that must not be copied

- Periodic validation seeds currently change with training step, so the curve does not compare every
  checkpoint on the same episode set.
- The video also changes its rollout seed per checkpoint, confounding policy progress with rollout
  variation.
- A model seed is recorded but is not applied to model initialization.
- Best-checkpoint selection and success reporting use the same small evaluation set; there is no
  disjoint final test suite.
- A reused run ID can overwrite some files while appending to others.
- Training directly calls `CartPole::step` rather than collecting transitions through Bevy.
- The software video renderer and Bevy watch renderer are separate implementations.
- Current visuals use a black pole, wheels, and boundary markers; official CartPole uses a tan pole,
  lavender axle, black rectangular cart/track, and no wheels or boundary markers.
- `/runs/` is ignored, so the strongest checkpoint, metrics, screenshots, and MP4 are not durable
  deliverables.

## Official target and acceptance matrix

Official thresholds below come from Gymnasium's registry. Where Gymnasium has no threshold, the plan
defines a project gate up front so it cannot be moved after seeing results.

### Classic Control

| Environment              | Horizon and proof gate                                                                                          | Trainer                            | Official visual target                                                                       | Current local state                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Acrobot-v1               | 500 steps; greedy held-out mean return >= -100 over 100 episodes; >=95% target reach                            | Burn DQN                           | 500x500 at 15 Hz; white background, cyan links, yellow joints, black target line; procedural | Docs only                                                  |
| CartPole-v1              | 500 steps; held-out mean >=475 over 100 episodes; >=90% hit 500-step ceiling                                    | Burn DQN                           | 600x400 at 50 Hz; black track/cart, tan pole, lavender axle, white background; procedural    | Functional, locally solved, visual/proof hardening remains |
| MountainCar-v0           | 200 steps; held-out mean >=-110 over 100 episodes; >=95% reach position 0.5                                     | Burn DQN                           | 600x400 at 30 Hz; black sinusoidal hill/car, gray wheels, yellow flag; procedural            | Docs only                                                  |
| MountainCarContinuous-v0 | 999 steps; held-out mean >=90 over 100 episodes; >=95% reach position 0.45                                      | Bounded continuous-action Burn PPO | Same 600x400 Mountain Car scene at 30 Hz                                                     | Docs only; blocked on PPO                                  |
| Pendulum-v1              | 200 steps; project gate mean >=-200 over 100 episodes, plus >=70% of steps 50-199 within 15 degrees and 1 rad/s | Bounded continuous-action Burn PPO | 500x500 at 30 Hz; red rod, black pivot, white background, torque arrow                       | Docs only; blocked on PPO                                  |

### Toy Text

| Environment     | Horizon and proof gate                                                                                                                                                   | Trainer                                                                                       | Official visual target                                                                              | Current local state |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------- |
| Blackjack-v1    | No official limit/threshold; project gate over 100,000 held-out games: lower 95% confidence bound on mean return >=-0.08 and improvement over random >=0.25              | First-party tabular Q-learning; Monte Carlo control only if predeclared Q-learning lane fails | 600x500 at 4 Hz; green felt, dealer/player labels, card faces/back, player sum, usable-ace label    | Docs only           |
| CliffWalking-v1 | Default: 200-step safety cap, exact -13 return, 100% goal completion, zero cliff entries. Slippery variant: lower Wilson 95% success bound >=0.90 across 10,000 episodes | Tabular Q-learning                                                                            | 720x240 at 4 Hz; 4x12 pixel-art mountain/cliff grid, stool, cookie, directional elf                 | Docs only           |
| FrozenLake-v1   | 4x4: 100 steps and lower Wilson 95% success bound >=0.70 over 10,000 episodes. 8x8: 200 steps and bound >=0.85                                                           | Tabular Q-learning                                                                            | 256x256 4x4 or 512x512 8x8 at 4 Hz; ice, holes, cracked hole, goal, stool, directional elf          | Docs only           |
| Taxi-v4         | 200 steps; enumerate all 300 valid initial states, mean >=8, 100% delivery, zero illegal pickup/drop-off actions                                                         | Action-masked tabular Q-learning                                                              | 550x350 at 4 Hz; pixel-art roads/medians, R/G/Y/B markers, passenger, hotel, directional yellow cab | Docs only           |

For stochastic gates, store the sample count, confidence method, interval, and raw aggregate counts
in the proof receipt. A single attractive rollout is never sufficient proof.

## Visual and asset policy

### Source of truth

- Use the pinned `ref/gymnasium` source to port drawing geometry, palettes, native dimensions,
  sprite placement, action orientation, and render cadence.
- Generate fixed reset, mid-episode, and terminal-state reference captures from that exact commit.
- Keep the official scene viewport unchanged. Put training labels and metrics in surrounding
  1280x720 letterbox/panel space so overlays do not alter the fidelity comparison.
- Use nearest-neighbor sampling and integer layout for pixel art.

### Asset reuse

Classic Control is procedural except for Pendulum's `clockwise.png`. Toy Text includes real PNG and
font assets in `ref/gymnasium/gymnasium/envs/toy_text/{img,font}`.

Before copying any bytes, create `THIRD_PARTY_ASSETS.md` with:

- upstream commit and path;
- SHA-256;
- original creator/source URL;
- license or written permission;
- required attribution;
- whether raw source redistribution, compiled crate inclusion, rendered-video use, and modification
  are each allowed; and
- the decision: reuse unchanged, reuse with modification, or clean replacement.

Known caution areas:

- `clockwise.png` has no contrary attribution in the MIT-licensed upstream history and is the
  cleanest reuse candidate.
- Frozen Lake/Cliff Walking elf and stool art and Taxi's passenger art are attributed to Franuka.
  Franuka permits project use/editing but prohibits redistribution "as is"; raw public-crate
  inclusion must be cleared.
- Other Frozen Lake and Taxi art is attributed to Mel Tillery without a currently usable license
  source.
- Cliff Walking leaves some asset authorship unresolved upstream.
- Blackjack's card art is credited to stock art and `Minecraft.ttf` has no bundled license.

"Reuse when available" therefore means legally reusable as part of this source distribution, not
merely present in a submodule. If clearance is missing, create a clean replacement that matches the
official silhouette, palette, scale, and composition without copying the bytes, and document why.
Runtime examples must load promoted first-party assets, never `ref/gymnasium` paths.

### Visual acceptance

- Procedural scenes: native viewport and palette match; semantic anchors are within 2 pixels; SSIM
  is >=0.95 after a one-pixel blur to neutralize antialiasing differences.
- Cleared sprite scenes: source hashes match the asset ledger, anchors are within 1 pixel, and SSIM
  is >=0.98 outside the external metrics panel.
- Pixel art has no smoothing or half-pixel placement.
- Manual review covers reset, movement in every action direction, success, failure, truncation,
  sprite layering, clipping, terminal art, and text legibility.

## Shared implementation shape

Avoid copying CartPole's entire target eight times. Keep environment-specific semantics and
rendering near each example, while moving trainer and artifact behavior into reusable code.

### Library work

Planned modules or equivalent boundaries:

- `src/training/dqn.rs`: move the real replay buffer, model, optimizer, target updates, exploration,
  checkpointing, train loop, and eval loop out of CartPole; make observation encoding and discrete
  action mapping environment adapters.
- `src/training/ppo.rs`: implement Gaussian bounded-action actor, critic, rollout storage, GAE,
  clipped objective, entropy/value losses, deterministic eval, and checkpoint reload.
- `src/training/tabular_q.rs`: first-class discrete-state/action Q-learning with epsilon scheduling,
  action masks, stochastic transitions, checkpointing, and the same metrics/proof contract.
- `src/training/tensor.rs`: add bounded continuous-action specifications and validation.
- `src/training/evaluation.rs`: fixed validation/test/demo suites, aggregate statistics, confidence
  intervals, and threshold evaluation.
- `src/training/proof.rs`: provenance, checkpoint hashing, fresh-reload receipts, collision-safe run
  lifecycle, and artifact-schema validation.
- `src/training/timelapse.rs`: checkpoint/eval joining, chronological sampling, segment manifests,
  and ffmpeg invocation; environment drawing remains in render adapters.
- `src/wrappers/time_limit.rs`: reusable wrapper that produces `Truncated` without confusing natural
  termination.
- Runner integration that feeds `TransitionEvent<E>` batches into trainers and derives deterministic
  per-environment reset streams. A qualifying training example must use more than one Bevy
  environment entity unless the algorithm explicitly requires serial episode updates.

The library must continue to compile and train headlessly with no render feature.

### Example support work

Use a small shared support module, not a new CLI framework dependency, for:

- common argument parsing and help;
- `train`, `eval`, `video`, and `watch` dispatch;
- `smoke`, `proof`, and `demo` presets;
- run-path resolution;
- Bevy watch/offscreen setup;
- a common external metrics panel; and
- video verification and poster extraction.

Each existing environment file remains the discoverable target entry point and owns its environment
configuration, state/action types, encoder/codec, renderer, and targeted parity tests.

### Common command contract

Every target must support the following shape:

```text
cargo run --release --example <target> -- train --preset proof --seed <seed> --run-id <id>
cargo run --release --example <target> -- eval --checkpoint <run>/best.mpk --suite test
cargo run --release --features render --example <target> -- video --run <run> --output <path>
cargo run --release --features render --example <target> -- watch --checkpoint <run>/best.mpk
```

Environment-specific configuration is explicit after these shared flags. A pre-existing run
directory is rejected unless the caller explicitly selects a validated resume mode or destructive
overwrite.

### Run and proof layout

Extend, rather than discard, the existing contract:

```text
runs/<env>-<algorithm>/<run-id>/
  config.json
  seeds.json
  provenance.json
  metrics.jsonl
  eval.jsonl
  summary.json
  proof.json
  best.mpk
  checkpoints/
    latest.mpk
    step-000000.mpk
    step-......mpk
  demo/
    training-timelapse.mp4
    poster.png
    manifest.json
```

`best.mpk` stays uniform across neural and tabular policies. If a tabular Burn module is
impractical, introduce a typed checkpoint abstraction while preserving the user-facing `best.mpk`
path and hash contract.

## Proof protocol

### Seed partitions

Derive and record independent streams for:

- model initialization;
- environment construction and resets per Bevy environment ID;
- action/exploration sampling;
- replay/rollout sampling;
- fixed validation selection;
- disjoint final test evaluation; and
- fixed demo rollouts.

Actually apply every recorded seed. All checkpoints use the same validation suite, and every
timelapse checkpoint uses the same demo states/seeds. This makes changes attributable to learning,
not lucky episodes.

### A qualifying run

- Starts with a saved/evaluated step-0 policy.
- Uses official observations, actions, rewards, dynamics, and time limits.
- Uses no reward shaping, heuristic fallback, behavior cloning, or warmup imitation.
- Records nonzero optimizer/Q updates and parameter/table hashes that differ from step 0.
- Saves fixed milestones at 0%, 5%, 20%, 50%, 80%, and 100% of budget plus the first threshold
  crossing.
- Selects best only on the validation suite.
- Loads the selected bytes in a new process and evaluates only on the test suite.
- Refuses to mark success if the proof receipt, hashes, metrics, or checkpoint are missing or
  internally inconsistent.

### Multi-seed release rule

- Run five training seeds per required release configuration.
- Require at least four individual passes.
- Require the median final held-out result to pass.
- Publish every seed's summary, not only the representative demo run.
- Select the median passing seed for the timelapse unless a different selection rule was declared
  before training.

PR smoke runs establish that updates and reloads work; they do not claim an environment is solved.
Only the full multi-seed protocol can produce a release proof badge.

## Timelapse protocol

Each environment gets one 30-second H.264/yuv420p MP4 on a 1280x720 canvas:

- First 20 seconds: chronological step-0 and periodic checkpoint policies.
- Final 10 seconds: the reloaded `best.mpk` policy at the official real-time cadence.
- The official viewport keeps its native aspect and, for pixel art, integer scaling.
- Simulation/state-update cadence matches Gymnasium: Acrobot 15 Hz, CartPole 50 Hz, Mountain Cars
  and Pendulum 30 Hz, and Toy Text 4 Hz.
- The external panel identifies environment/version, algorithm, root seed, trained steps/episodes,
  checkpoint short hash, fixed-suite score/confidence interval, and pass/fail gate.
- The segment manifest records frame ranges, checkpoint SHA-256, metrics row, demo seeds/states, and
  playback speed.
- `ffmpeg` and `ffprobe` become declared development/release tools and are added to the Nix shell.

Environment-specific storytelling:

- Acrobot: truncation/failure becomes repeated target-line crossing.
- CartPole: early falling becomes sustained 500-step balance.
- Mountain Car: stalled motion becomes learned back-and-forth momentum and flag reach.
- Continuous Mountain Car: ineffective force becomes smooth energy-building and flag reach.
- Pendulum: uncontrolled swing becomes swing-up and sustained upright dwell.
- Blackjack: combine the official card view with fixed diagnostic hands and learned
  usable/non-usable ace policy grids; do not use one lucky random hand as proof.
- Cliff Walking: cliff-heavy wandering becomes the exact 13-step safe route; show slippery behavior
  in a short inset or montage.
- Frozen Lake: use a fixed multi-seed montage and display aggregate success confidence because one
  slippery success is misleading.
- Taxi: hold passenger/destination scenarios fixed across checkpoints so wandering becomes legal
  pickup and efficient delivery.

Validate each output with `ffprobe`, expected duration/frame count/dimensions/codec, checkpoint hash
links, and manual inspection of first/middle/final frames.

The default durable promotion path is `docs/assets/demos/<target>/` containing MP4, poster, proof,
and manifest. Keep each MP4 under 2 MiB and the suite under 20 MiB when practical. If those limits
cannot be met without visibly harming the demo, publish MP4s as project release assets and keep the
poster, proof, manifest, SHA-256, and stable release URL in the tracked docs directory. Never leave
the only shareable copy under ignored `runs/`.

## Delivery phases

### Phase 0: freeze the oracle and acceptance contracts

1. Record the target matrix, official registry thresholds, horizons, default kwargs, render
   dimensions/cadence, and project-defined gates in machine-readable fixtures.
2. Keep the existing pinned Gymnasium commit as the oracle for the first implementation pass. Before
   updating the submodule, diff all nine registry/source/render surfaces and consciously regenerate
   fixtures.
3. Generate deterministic dynamics fixtures from explicit starting states and action sequences.
4. Generate exact Toy Text transition-table fixtures.
5. Test stochastic resets/transitions statistically unless NumPy-identical RNG is deliberately
   adopted; do not pretend merely repeatable Rust RNG values are NumPy-identical.
6. Create the third-party asset ledger and resolve reuse/replacement before renderer work starts.
7. Define JSON schemas for config, seeds, metrics, eval, proof, and video manifests.

Exit: no environment can be implemented or tuned against an ambiguous version, threshold, visual,
or asset license.

### Phase 1: harden CartPole as the reusable vertical slice

1. Split CartPole environment, generic DQN, CLI, proof, renderer, and video responsibilities.
2. Move functional DQN into `src/training/dqn.rs` and drive it from Bevy transition batches.
3. Fix model seeding, fixed validation seeds, disjoint test seeds, run-directory collision behavior,
   and proof hashing.
4. Add a reusable time-limit wrapper and verify CartPole truncates at 500 while natural failure
   still terminates.
5. Replace dual renderers with one official-fidelity Bevy scene used by watch and offscreen video.
6. Match the official tan pole/lavender axle/cart/track composition and move demo metrics outside
   the 600x400 scene viewport.
7. Re-run five seeds, fresh-reload evaluation, visual regression, and timelapse promotion.
8. Preserve the existing ignored run as historical evidence only; do not promote it as the final
   release proof.

Exit: CartPole passes the complete definition of done and provides the template used by all
remaining targets.

### Phase 2: implement the shared Toy Text lane

1. Implement `TabularQTrainer` with action masks, stochastic transitions, fixed-suite eval,
   checkpointing, and proof/video milestones.
2. Implement Frozen Lake first to validate stochastic tabular training, sprite/tile rendering,
   confidence intervals, and 4x4/8x8 configuration reuse.
3. Implement Cliff Walking next; validate deterministic exact optimum and the named slippery
   transition table/configuration.
4. Implement Taxi-v4 with exact encode/decode, all 300 initial states, walls, action masks, legal
   and illegal pickup/drop-off rewards, rainy movement, and fickle-passenger semantics.
5. Implement Blackjack with infinite-deck sampling, usable ace, dealer policy, natural/SAB behavior,
   diagnostic-hand eval, and policy-grid visualization.
6. Clear or replace every Toy Text asset before promoting it into runtime assets.
7. Run proof and video gates for all four default catalog configurations plus required variant
   gates.

Exit: four registered Toy Text targets, four proof bundles/videos, and passing 4x4/8x8/slippery/rule
variant tests.

### Phase 3: finish discrete Classic Control

1. Implement MountainCar-v0 dynamics, reset bounds, collision behavior, time limit, DQN adapter, and
   official procedural renderer.
2. Implement Acrobot-v1 book dynamics, RK4 integration, velocity bounds, action/noise behavior,
   terminal height, time limit, DQN adapter, and official procedural renderer.
3. Use state-injection fixtures for numerical dynamics comparison and statistical tests for random
   initialization/noise.
4. Tune only trainer hyperparameters under fixed proof suites. If DQN fails, improve exploration,
   replay, target updates, or the model; never reshape official rewards.
5. Produce and promote both proof/video bundles.

Exit: all three discrete Classic Control DQN examples pass the official registry thresholds through
the shared trainer.

### Phase 4: implement continuous-action PPO and finish Classic Control

1. Add bounded continuous action encoding and a generic Burn PPO actor-critic implementation.
2. Unit-test tanh/action scaling, log probabilities, GAE, clipped loss, truncation bootstrapping,
   deterministic inference, and save/load equivalence.
3. Implement Pendulum-v1 dynamics/configuration and the official rod/pivot/torque-arrow renderer.
4. Implement MountainCarContinuous-v0 dynamics/reward/time limit and reuse the Mountain Car
   renderer.
5. Run five-seed proof gates and promote both videos.
6. If PPO cannot pass at least four of five seeds within the predeclared budget, stop and make an
   explicit trainer decision (for example, a separate SAC work item). Do not silently raise budgets,
   cherry-pick a seed, or alter environment semantics.

Exit: both continuous examples pass their gates from freshly loaded Burn checkpoints.

### Phase 5: complete automation, gallery, and release evidence

1. Register all nine targets in `Cargo.toml` and make `cargo check/test --all-targets` meaningful.
2. Repair GitHub CI's nonexistent `x11` feature and use the actual headless/render feature matrix.
3. Add PR gates for formatting, all-target compile/test/clippy, parity fixtures, checkpoint
   round-trip, short real-update training, render snapshots, and a tiny video/ffprobe smoke.
4. Add nightly/release jobs for the full five-seed solve matrix and nine videos; retain raw proof
   bundles as CI/release artifacts.
5. Add `ffmpeg`/`ffprobe` and required software-render libraries to the Nix environment and checks.
6. Update README's example table and add one doc page per environment with train/eval/video/watch
   commands, algorithm, threshold, expected runtime, proof link, poster, and demo link.
7. Add a nine-environment gallery and a machine-readable index of proof/video hashes.
8. Inspect every promoted video and poster for clipping, missing sprites/text, wrong orientation,
   blurred pixel art, stale scores, or mismatched hashes.

Exit: a fresh contributor or release job can reproduce, verify, open, and share all nine demos from
documented commands.

## Planned verification layers

### Per change / PR

```text
cargo fmt --check
cargo check --locked --all-targets
cargo test --locked --all-targets
cargo clippy --locked --all-targets --all-features -- -D warnings
nix flake check
```

Add target-specific parity and smoke commands to CI as they come online. A smoke preset must execute
at least one genuine update and checkpoint reload, but it is explicitly labeled non-solution proof.

### Nightly / release

- Run the five-seed proof matrix for every required default and named variant.
- Validate every proof schema and checkpoint hash.
- Regenerate videos only from qualifying proof runs.
- Run visual comparisons and video structural checks.
- Fail the matrix if any environment loses its pass ratio, median gate, fresh-load result, visual
  threshold, or durable demo artifact.

## Risks and stop rules

| Risk                                                                 | Mitigation / stop rule                                                                                            |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Eight stubs invite copy/paste implementations                        | Complete CartPole shared extraction before the second functional target.                                          |
| Sparse-reward Classic Control training is unstable                   | Hold semantics and suites fixed; tune trainer only; publish all seeds.                                            |
| PPO fails continuous Mountain Car reliably                           | Time-box the declared PPO budget and open an explicit algorithm decision instead of weakening proof.              |
| CartPole success is mistaken for generic trainer completion          | Require MountainCar and Acrobot to use the extracted DQN before calling DQN generic.                              |
| Videos look good but do not prove learning                           | Require checkpoint hashes, fixed demo states, proof manifests, and fresh held-out eval.                           |
| Best checkpoint is selected by luck                                  | Fixed validation suite, disjoint test suite, five seeds, 4/5 pass rule.                                           |
| Seed files claim reproducibility without applying seeds              | Assert seed application and reproduce a same-seed run/checkpoint hash in tests where backend determinism permits. |
| Bevy watch and offline video drift                                   | One renderer/state-to-scene mapping, used by both paths.                                                          |
| Official sprites cannot legally be redistributed                     | Block byte-copying until the asset ledger clears it; create documented clean replacements otherwise.              |
| Runtime depends on `ref/gymnasium`                                   | Tests/oracle may use the pinned submodule; published binaries/assets may not.                                     |
| Full training makes PR CI too slow                                   | Keep solution proof nightly/release; PR smoke proves execution, updates, schemas, and reload only.                |
| Generated videos bloat Git                                           | Enforce size budget; fall back to release assets while keeping tracked hashes/posters/proofs/URLs.                |
| Existing ignored artifacts are accidentally treated as release truth | Treat them as evidence for planning only and regenerate under the hardened protocol.                              |

Do not mark an environment done because it compiles, opens a window, emits reward logs, or has a
single good video. Stop only when its semantics, learning, checkpoint reload, visual parity, video,
and durable proof bundle all pass.

## Primary research sources

- [Current Gymnasium Classic Control catalog](https://gymnasium.farama.org/main/environments/classic_control/)
- [Current Gymnasium Toy Text catalog](https://gymnasium.farama.org/main/environments/toy_text/)
- [Gymnasium environment registrations and thresholds](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/__init__.py)
- [Gymnasium v1.3.0 release](https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.3.0)
- [Audited Gymnasium main commit from 2026-07-09](https://github.com/Farama-Foundation/Gymnasium/commit/24c3a0dfaf83223b63b66df062a2d681bdb54c68)
- [Classic Control rendering source](https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control)
- [Toy Text rendering source and sprites](https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/toy_text)
- [Official recording guidance](https://gymnasium.farama.org/main/introduction/record_agent/)
- [Official Blackjack Q-learning tutorial](https://gymnasium.farama.org/main/tutorials/training_agents/blackjack_tutorial/)
- [Official Frozen Lake Q-learning tutorial](https://gymnasium.farama.org/main/tutorials/training_agents/frozenlake_q_learning/)
- [Official Taxi action-masking tutorial](https://gymnasium.farama.org/main/tutorials/training_agents/action_masking_taxi/)
- [Gymnasium MIT license](https://github.com/Farama-Foundation/Gymnasium/blob/main/LICENSE)
- [Franuka RPG Snow Tileset usage terms](https://franuka.itch.io/rpg-snow-tileset)

Use the local pinned source for reproducible implementation details and the live links above to
audit drift before beginning implementation.
