---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - ../specs/spec-bevy-gym-api-redesign/SPEC.md
  - ../specs/spec-bevy-gym-api-redesign/api-contract.md
  - ../specs/spec-bevy-gym-api-redesign/architecture-and-migration.md
  - ../specs/spec-bevy-gym-api-redesign/training-engine.md
  - prd-burn-backed-bevy-gym-training-core-2026-06-13.md
  - epics-burn-backed-bevy-gym-training-core-2026-06-13.md
  - sprint-change-proposal-2026-06-13.md
  - ../implementation-artifacts/spec-bevy-gym-api-redesign.md
workflowType: 'architecture'
project_name: 'bevy-gym'
user_name: 'Sagan'
date: '2026-06-13'
lastStep: 8
status: 'complete'
completedAt: '2026-06-13'
---

# Architecture Decision Document: Burn-Backed Bevy Gym Training Core

This document is the formal architecture artifact for the approved course
correction. It supersedes any environment-runner-only interpretation of
`bevy-gym` while preserving the small environment contract as the trainer input
boundary.

## Project Context Analysis

### Requirements Overview

**Functional requirements**

The approved product is a Bevy-native reinforcement-learning training crate
powered by Burn:

- Keep `Env`, `Reset`, `Step`, and `EpisodeStatus` as the repo-owned
  environment contract.
- Keep the Bevy runner responsible for spawning, stepping, resetting, and
  emitting typed transitions from many environment entities.
- Add a required Burn-backed trainer layer with model definitions, optimizer
  setup, DQN replay storage, PPO rollout storage, metrics, checkpoint
  save/load, and deterministic eval.
- Accept the first slice only when CartPole DQN trains, deterministic eval
  reward improves over the initial or random baseline, `best.mpk` is saved, and
  eval loads `best.mpk` from disk successfully.
- Queue PPO as the second slice and use multiple Bevy environment entities for
  on-policy rollout collection.

**Non-functional requirements**

- Headless training must not depend on rendering.
- Burn is required and first-class; `rl-traits` must not return as the public
  contract and `ember-rl` must not become the product core.
- Training and eval seed streams must be explicit and deterministic.
- Large replay buffers, rollout tensors, model state, optimizer state, metrics,
  and checkpoints must live in trainer/run storage, not ordinary ECS
  components.
- Tests and smoke gates must validate behavior without relying on a fragile
  single peak reward.

**Scale and complexity**

- Primary domain: Rust library crate for reinforcement learning, Bevy ECS
  rollout orchestration, and Burn-backed training.
- Complexity level: high for a library MVP because correctness spans public
  API shape, Bevy scheduling, deterministic RL training, serialization, and
  reproducible evaluation.
- Estimated architectural components: core environment API, Bevy runner,
  trainer API, DQN trainer, PPO trainer, storage, metrics, checkpointing, eval,
  examples, docs, and validation gates.

### Technical Constraints And Dependencies

- Current source already contains the small core/runner surface in `src/core.rs`,
  `src/components.rs`, `src/events.rs`, `src/plugin.rs`, `src/systems/step.rs`,
  and `src/systems/reset.rs`.
- Current `Cargo.toml` depends on Bevy only. Implementation must add Burn as a
  required dependency and must not restore `rl-traits` or make `ember-rl` core.
- Current CartPole is a deterministic heuristic runner example. It becomes a
  baseline or support example; the first accepted product path is CartPole DQN
  train/eval.
- Historical Burn-agnostic research is superseded for product scope. The useful
  part that remains is the small `Env` boundary; Burn trainer code belongs
  above it.

### Cross-Cutting Concerns Identified

- Determinism: seeds must cover environment resets, action sampling, model
  initialization, replay sampling, and eval episodes.
- Episode semantics: `Terminated` and `Truncated` must remain distinct through
  storage, target computation, and advantage computation.
- Artifact stability: run directory layout and `best.mpk` must be stable enough
  for docs, tests, and review automation.
- Agent consistency: module boundaries must prevent future agents from putting
  model, optimizer, replay, rollout, or checkpoint internals into `Env`.

## Starter Template Evaluation

### Primary Technology Domain

This is an existing Rust library crate, not a greenfield web, mobile, or API
application. No external starter template should be introduced.

### Foundation Decision

Use the current crate as the foundation:

- Keep Bevy `0.18` as currently declared in `Cargo.toml`.
- Keep the existing `src/core.rs` and Bevy runner modules as the environment and
  rollout foundation.
- Add the trainer layer in `src/training/*`.
- Add new examples and docs around CartPole DQN train/eval.

### Current External Verification

Burn `0.21.0` docs describe Burn as a deep learning framework with generic
backends, training/inference support, and backend composability:

- Burn docs: https://docs.rs/burn/latest/burn/
- Burn feature flags: https://docs.rs/crate/burn/latest/features
- Burn backend feature list: https://burn.dev/docs/src/burn/lib.rs.html
- Flex backend docs: https://docs.rs/burn-flex/latest/burn_flex/
- Recorder docs: https://docs.rs/burn-core/latest/burn_core/record/index.html
- Burn saving/loading book: https://github.com/tracel-ai/burn/blob/main/burn-book/src/saving-and-loading.md

The architecture therefore uses Burn 0.21 with the pure-Rust CPU `Flex` backend,
the `autodiff` backend decorator for training, and named MessagePack recorders
for `*.mpk` policy artifacts.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical decisions**

1. Product layering: `Env` stays small; the trainer is a required layer above
   runner transitions.
2. Burn dependency: Burn is required, with a CPU/headless default backend.
3. Checkpoint contract: `best.mpk` is the required best-policy artifact and
   deterministic eval must load it from disk.
4. First slice: DQN on CartPole is the acceptance path.
5. Second slice: PPO uses multiple Bevy env entities for rollout collection
   after DQN acceptance.

**Important decisions**

1. Run directories are stable and inspectable under `runs/<env>-<algorithm>/<run-id>/`.
2. Metrics are append-only JSONL.
3. CI uses a deterministic smoke preset; longer useful-policy validation uses a
   separate manual/release preset.
4. The first train/eval command surface is an example binary plus library API,
   avoiding a new CLI dependency unless explicitly approved.

**Deferred decisions**

- Full training resume from replay/optimizer state is deferred unless it falls
  out naturally from the first checkpoint implementation. DQN acceptance only
  requires model checkpoint save/load for deterministic eval from `best.mpk`.
- GPU backends, WGPU/CUDA acceleration, Python bridges, and `burn-rl` adoption
  are deferred. They must not hide the crate-owned Burn trainer behind another
  framework.

### Product Layers

```text
Env reset/step
  -> Bevy runner entities and typed transitions
  -> trainer storage
  -> Burn model optimization
  -> metrics/checkpoints
  -> deterministic eval from checkpoint
```

**Layer 1: core environment vocabulary**

Lives in `src/core.rs`. Owns:

- `Env`
- `Reset`
- `Step`
- `EpisodeStatus`
- `Transition`
- optional spaces/checkers needed at trainer boundaries

This layer must not own model definitions, optimizers, replay buffers, rollout
buffers, metrics, checkpoint recorders, or eval loops.

**Layer 2: Bevy runner**

Lives in `src/components.rs`, `src/events.rs`, `src/plugin.rs`, and
`src/systems/*`. Owns:

- `BevyGymPlugin<E>`
- environment entities and env ids
- typed action requests/responses
- transition and episode-end messages
- reset policy and duplicate action response behavior

The runner is required infrastructure, but not product-complete by itself.

**Layer 3: Burn training engine**

Lives in `src/training/*`. Owns:

- Burn backend type aliases
- model definitions
- optimizer setup
- replay and rollout storage
- training loops
- metrics
- checkpoint save/load
- deterministic eval

**Layer 4: examples and docs**

Lives in `examples/*`, `docs/*`, `README.md`, and crate docs. CartPole DQN
train/eval is the first accepted path. Heuristic/manual examples can remain as
baselines or runner demonstrations, but docs must not present them as the
product endpoint.

### Burn Backend, Autodiff, And Recorder

**Decision:** default to Burn `0.21` with `Flex` CPU backend and `Autodiff` for
training.

Implementation target:

```toml
[dependencies.burn]
version = "0.21"
default-features = false
features = ["std", "autodiff", "flex"]
```

Default backend aliases should be centralized in `src/training/backend.rs`:

```rust
pub type InferenceBackend = burn::backend::Flex;
pub type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
```

Burn 0.21's `Autodiff` type has a second checkpoint-strategy parameter with a
default of `NoCheckpointing`, so the one-parameter alias above is the default
architecture.

Rationale:

- `Flex` is pure Rust, CPU-only, thread-safe, and documented by Burn as the
  preferred portable CPU backend for new projects.
- `Autodiff` is Burn's backend decorator for gradient computation.
- This default avoids GPU, LibTorch, CUDA, WGPU, BLAS, and C runtime assumptions
  in CI and new-user examples.

**Recorder decision:** use `NamedMpkFileRecorder<FullPrecisionSettings>` for
policy checkpoints.

The user-facing artifact is always `best.mpk`. If Burn's `save_file` API wants
an extensionless base path for automatic extension handling, wrap that behavior
in `training::checkpoint` so callers still pass and inspect `best.mpk`.

### Run Directory And Checkpoint Architecture

Each training run writes:

```text
runs/<env>-<algorithm>/<run-id>/
  config.json
  seeds.json
  metrics.jsonl
  eval.jsonl
  summary.json
  checkpoints/
    latest.mpk
    step-000000.mpk
    step-010000.mpk
  best.mpk
```

Rules:

- `<env>-<algorithm>` is lowercase kebab case, for example `cartpole-dqn`.
- `<run-id>` defaults to a timestamp plus short seed suffix. Tests may pass a
  fixed run id.
- `config.json` records algorithm config, environment count, training budget,
  eval cadence, and threshold preset.
- `seeds.json` records root seed plus derived seed streams for env reset,
  action exploration, replay sampling, model initialization, and eval.
- `metrics.jsonl` records train-step metrics and episode summaries.
- `eval.jsonl` records deterministic eval results and checkpoint selection.
- `summary.json` records final status, best eval reward, best checkpoint path,
  and acceptance outcome.
- `checkpoints/latest.mpk` is overwritten with the latest policy checkpoint.
- `checkpoints/step-*.mpk` is periodic and may be disabled for CI smoke.
- `best.mpk` is overwritten only when deterministic eval improves the best
  selection metric.

Checkpoint save/load acceptance for DQN means:

- Save model policy weights to `best.mpk`.
- Load `best.mpk` from disk into a freshly initialized model.
- Run deterministic eval from the loaded model.
- Report the checkpoint path and eval reward.

Full training resume from optimizer and replay state is a later capability
unless explicitly added to BG-TRAIN-003. Do not block DQN acceptance on replay
serialization.

### Train And Eval API Shape

Expose both library API and example commands. Do not add `clap` or another CLI
dependency for the first slice unless approved; parse a small fixed argument
surface in the example or keep arguments minimal.

Library API target:

```rust
use bevy_gym::training::dqn::{train_cartpole_dqn, DqnConfig};
use bevy_gym::training::eval::{eval_checkpoint, EvalConfig};

let report = train_cartpole_dqn(DqnConfig::smoke())?;
let eval = eval_checkpoint(report.best_checkpoint, EvalConfig::deterministic())?;
```

Example command target:

```sh
cargo run --example cartpole_dqn --release -- train --preset smoke --seed 7 --run-id ci-smoke
cargo run --example cartpole_dqn --release -- eval --checkpoint runs/cartpole-dqn/ci-smoke/best.mpk --seed 1007 --episodes 5
```

Longer validation:

```sh
cargo run --example cartpole_dqn --release -- train --preset useful --seed 7
cargo run --example cartpole_dqn --release -- eval --checkpoint <run-dir>/best.mpk --seed 1007 --episodes 20
```

The command names are part of the architecture. Exact flags may expand, but the
subcommands `train` and `eval`, `--checkpoint`, `--seed`, and `--episodes` must
remain stable once implemented.

### CI Smoke Versus Manual Validation

**CI smoke preset**

Purpose: prove the full path is wired and deterministic without requiring a
solved CartPole policy.

Required checks:

- fixed root seed and eval seeds;
- deterministic random or initial-policy baseline measured in the same run;
- DQN train loop runs for a bounded small budget;
- deterministic eval reward improves over baseline by at least `+10.0` mean
  reward or another documented threshold calibrated before merge;
- `best.mpk`, `config.json`, `seeds.json`, `metrics.jsonl`, `eval.jsonl`, and
  `summary.json` exist;
- a separate eval command loads `best.mpk` from disk and reports the same reward
  within deterministic tolerance.

If the first implementation proves `+10.0` is flaky on CI hardware, the
threshold may be changed in the implementation artifact before dev starts, but
the gate must still prove positive improvement over a same-run baseline plus
checkpoint creation and checkpoint eval.

**Manual/release useful-policy preset**

Purpose: prove the crate trains a genuinely useful CartPole policy out of the
box.

Required checks:

- longer training budget than CI;
- deterministic eval over at least 20 episodes;
- mean eval reward target starts at `>= 195.0`;
- `best.mpk` loads in a fresh process;
- metrics show eval improvement over time.

This useful-policy preset is not a CI blocker unless the project later decides
to spend the runtime budget.

### DQN First Slice

DQN owns:

- MLP Q-network model definition;
- deterministic model initialization;
- optimizer setup;
- epsilon-greedy exploration schedule;
- replay buffer;
- target network update policy;
- Huber or MSE TD loss;
- transition tensorization;
- deterministic eval with exploration disabled.

CartPole-specific DQN support may live behind a small example adapter, but the
DQN trainer and storage modules must be generic enough to avoid becoming
single-example code.

### PPO Second Slice

PPO starts only after DQN, metrics, checkpoint, and deterministic eval are
accepted.

PPO owns:

- actor-critic Burn model;
- rollout storage across multiple Bevy env entities;
- log probabilities, values, rewards, statuses, and env ids;
- advantage/return computation that preserves terminated versus truncated
  bootstrap semantics;
- minibatch PPO updates;
- checkpoint and eval using the same run directory contract.

## Implementation Patterns And Consistency Rules

### Critical Conflict Points

Potential AI-agent conflicts to prevent:

- Adding model or optimizer fields to `Env`.
- Putting replay/rollout tensors into ECS components.
- Choosing a different Burn backend per module.
- Saving checkpoints under ad hoc names instead of `best.mpk`.
- Treating CartPole heuristic behavior as acceptance.
- Starting PPO before DQN acceptance.
- Reintroducing `rl-traits` or `ember-rl` as public contract dependencies.

### Naming Patterns

- Rust modules: snake case.
- Public trainer types: `DqnConfig`, `DqnTrainer`, `DqnReport`,
  `EvalConfig`, `EvalReport`, `CheckpointError`.
- Algorithm module names: `training::dqn`, `training::ppo`.
- Storage modules: `training::storage::replay` and
  `training::storage::rollout`.
- Metrics fields: slash-separated names in JSONL, for example
  `train/loss`, `train/epsilon`, `eval/mean_reward`, `checkpoint/best`.
- Run directories: lowercase kebab case, for example `runs/cartpole-dqn/<run-id>/`.
- Checkpoint files: `best.mpk`, `latest.mpk`, and `step-000000.mpk`.

### Structure Patterns

- Keep core environment types in `src/core.rs`.
- Keep Bevy runner concerns in the existing runner modules.
- Put all Burn references in `src/training/*` except for top-level re-exports
  from `src/lib.rs`.
- Keep CartPole environment code usable by both runner-only and trainer
  examples. If needed, move shared CartPole logic to an example support module,
  not into `src/core.rs`.
- Tests for trainer internals should be colocated with modules or in
  `tests/training_*` integration tests once public API stabilizes.

### Format Patterns

Use JSON/JSONL for human-inspectable run metadata:

```json
{"global_step":1000,"train/loss":0.031,"train/epsilon":0.42}
{"global_step":1000,"eval/mean_reward":48.2,"checkpoint/best":true}
```

Use Burn named MessagePack for policy model records:

```text
best.mpk
checkpoints/latest.mpk
checkpoints/step-010000.mpk
```

Do not invent additional binary formats in the first slice.

### Communication Patterns

- Runner-to-trainer communication consumes `TransitionEvent<E>` and
  `EpisodeEndEvent`.
- Trainer-to-runner action selection uses typed `ActionResponse<E>`.
- Manual policies continue to use `ActionRequest<E>` and `ActionResponse<E>`.
- DQN and PPO trainer loops may own Bevy `App` construction helpers, but the
  single-environment contract remains independent.

### Error Handling Patterns

- Trainer APIs return `Result<_, TrainingError>` or algorithm-specific errors
  that convert into `TrainingError`.
- Checkpoint errors include the path and whether the operation was save or load.
- Missing tensorization/spec errors must name the trainer-boundary requirement
  and must not suggest editing `Env`.
- Determinism mismatches in tests should print seed, run id, checkpoint path,
  expected reward, and observed reward.

### Good Examples

- `Env` returns `Step { observation, reward, status, info }`; DQN storage later
  samples this transition and tensorizes it.
- `training::checkpoint::save_best(...)` writes `best.mpk` and returns the
  exact path that deterministic eval will load.
- PPO rollout storage records `EpisodeStatus::Truncated` separately from
  `EpisodeStatus::Terminated` before computing advantages.

### Anti-Patterns

- Adding `fn model()` or `fn optimizer()` to `Env`.
- Saving `best.bin`, `model.pt`, or `checkpoint.json` as the policy artifact
  instead of `best.mpk`.
- Using rendering output as proof of training correctness.
- Starting with `burn-rl`, `ember-rl`, or Python SB3 as the implemented trainer
  core.
- Storing large tensor batches in Bevy components.

## Project Structure And Boundaries

### Complete Project Directory Structure

Target structure after the DQN slice:

```text
bevy-gym/
  Cargo.toml
  README.md
  src/
    lib.rs
    core.rs
    components.rs
    events.rs
    plugin.rs
    systems/
      mod.rs
      reset.rs
      step.rs
    training/
      mod.rs
      backend.rs
      config.rs
      rng.rs
      tensor.rs
      models/
        mod.rs
        dqn.rs
        ppo.rs
      optim.rs
      storage/
        mod.rs
        replay.rs
        rollout.rs
      metrics.rs
      checkpoint.rs
      eval.rs
      dqn.rs
      ppo.rs
  examples/
    classic-control/
      cart_pole.rs
      cart_pole_dqn.rs
  docs/
    examples/
      cartpole.md
    plugins/
      bevy_gym_plugin.md
```

PPO may add a PPO-specific example after DQN acceptance.

### Architectural Boundaries

**Core boundary**

`src/core.rs` is algorithm-agnostic. It may contain optional spaces/checkers and
transition vocabulary, but no Burn model, optimizer, storage, metrics, or
checkpoint implementation.

**Runner boundary**

The runner owns Bevy ECS scheduling and message flow. It emits transitions and
episode summaries. It does not know about DQN loss, PPO advantages, replay
sampling, checkpoint formats, or eval thresholds.

**Trainer boundary**

The trainer owns Burn and all algorithm state. It may build a Bevy app or
install systems to collect transitions, but trainer storage remains outside ECS
component state.

**Example boundary**

Examples show train/eval commands and baseline policies. They can adapt CartPole
to DQN tensorization, but the DQN implementation itself belongs in
`src/training`.

### Requirements To Structure Mapping

| Requirement or story | Structure |
| --- | --- |
| FR-1, BG-TRAIN-001 small Env contract | `src/core.rs`, `src/lib.rs` exports |
| FR-2 terminated/truncated semantics | `src/core.rs`, `src/systems/*`, `src/training/storage/*`, `src/training/dqn.rs`, `src/training/ppo.rs` |
| FR-3 parallel env stepping | `src/plugin.rs`, `src/systems/step.rs`, tests/examples |
| FR-4 typed action flow | `src/events.rs`, `src/systems/step.rs`, docs |
| FR-5 required Burn dependency | `Cargo.toml`, `src/training/mod.rs`, `src/training/backend.rs` |
| FR-6 DQN model/optimizer/replay | `src/training/models/dqn.rs`, `src/training/optim.rs`, `src/training/storage/replay.rs`, `src/training/dqn.rs` |
| FR-7 metrics/checkpoints | `src/training/metrics.rs`, `src/training/checkpoint.rs` |
| FR-8 deterministic eval | `src/training/eval.rs`, `examples/classic-control/cart_pole_dqn.rs` |
| FR-9 CartPole DQN acceptance | `examples/classic-control/cart_pole_dqn.rs`, `docs/examples/cartpole.md` |
| FR-10 PPO rollout storage | `src/training/storage/rollout.rs`, `src/training/ppo.rs` |
| FR-11 PPO actor-critic eval | `src/training/models/ppo.rs`, `src/training/ppo.rs`, future example |

### Data Flow

```text
ActionRequest<E>
  -> policy/trainer writes ActionResponse<E>
  -> step_system calls Env::step
  -> TransitionEvent<E>
  -> DQN replay or PPO rollout storage
  -> Burn optimizer update
  -> deterministic eval
  -> metrics/checkpoint writer
```

### Development Workflow Integration

Implementation stories should proceed in this order:

1. Add required Burn dependency and `src/training` skeleton.
2. Add backend, config, rng, tensorization, metrics, and checkpoint modules.
3. Add DQN model/optimizer/replay.
4. Connect CartPole DQN train loop to Bevy transitions.
5. Add deterministic eval from `best.mpk`.
6. Add CI smoke and docs.
7. Start PPO rollout storage and actor-critic implementation.

## Architecture Validation Results

### Coherence Validation

**Decision compatibility**

The decisions are compatible. The small `Env` trait remains independent of
Burn, while Burn is still a required crate dependency through the trainer layer.
The Bevy runner emits typed transitions; trainer storage consumes them; Burn
models optimize from storage; checkpoints and eval close the acceptance loop.

**Pattern consistency**

Naming, file placement, run directories, metrics, and checkpoint conventions all
support the same product boundary: environment contracts are small, trainer
internals are first-class but isolated, and artifacts are stable.

**Structure alignment**

The proposed structure extends the current source tree without relocating the
existing runner modules. The new `src/training/*` boundary is clear enough for
multiple implementation agents to avoid mixing responsibilities.

### Requirements Coverage Validation

**Epic coverage**

- Epic 1, Trainer Foundation And Artifact Contract, is covered by Burn
  dependency, trainer skeleton, backend, config, metrics, and checkpoint
  decisions.
- Epic 2, CartPole DQN Acceptance Slice, is covered by DQN model, replay,
  train/eval command shape, `best.mpk`, CI smoke, and useful-policy validation.
- Epic 3, PPO Parallel-Environment Slice, is covered by rollout storage and PPO
  actor-critic module boundaries, with explicit sequencing after DQN acceptance.

**Functional requirements coverage**

All FR-1 through FR-11 have a target module, data flow, and validation path.

**Non-functional requirements coverage**

Headless operation, deterministic seeds, artifact stability, avoidance of
runner-only framing, and no `rl-traits` or `ember-rl` product-core dependency
are all covered.

### Gap Analysis Results

**Critical gaps**

None.

**Important implementation calibration**

- CI threshold may need one implementation calibration pass if `+10.0` mean eval
  improvement is flaky. The architecture preserves the invariant that CI must
  prove positive improvement, `best.mpk`, and eval-from-disk.

**Future enhancements**

- GPU/WGPU/CUDA backend features.
- Full training resume from optimizer and replay state.
- PPO continuous action support.
- Python or SB3 baseline oracle.

### Architecture Completeness Checklist

**Requirements Analysis**

- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed
- [x] Technical constraints identified
- [x] Cross-cutting concerns mapped

**Architectural Decisions**

- [x] Critical decisions documented with versions
- [x] Technology stack fully specified
- [x] Integration patterns defined
- [x] Performance considerations addressed

**Implementation Patterns**

- [x] Naming conventions established
- [x] Structure patterns defined
- [x] Communication patterns specified
- [x] Process patterns documented

**Project Structure**

- [x] Complete directory structure defined
- [x] Component boundaries established
- [x] Integration points mapped
- [x] Requirements to structure mapping complete

### Architecture Readiness Assessment

**Overall status:** READY FOR IMPLEMENTATION

**Confidence level:** High for architecture readiness. Medium for the exact CI
reward threshold until the first DQN implementation calibrates variance on this
checkout and CI environment.

**Key strengths**

- Product direction is no longer runner-only.
- Burn is required and isolated in a first-class trainer layer.
- CartPole DQN acceptance is behavioral, artifact-backed, and deterministic.
- PPO is sequenced after DQN and tied to Bevy parallel environments.
- Source module boundaries are concrete enough for implementation agents.

### Implementation Handoff

AI agents implementing from this architecture must:

- Preserve `Env`, `Reset`, `Step`, and `EpisodeStatus` as the small environment
  contract.
- Add Burn as required trainer infrastructure.
- Put all trainer internals under `src/training/*`.
- Make CartPole DQN train/eval the first accepted path.
- Save `best.mpk` and prove eval loads it from disk.
- Keep PPO second and use multiple Bevy environment entities.
- Avoid `rl-traits`, `ember-rl` as product core, rendering-as-proof, and
  trainer internals in `Env`.

First implementation priority: BG-TRAIN-001, required Burn dependency and
trainer skeleton, immediately followed by checkpoint/metrics scaffolding so DQN
acceptance cannot drift away from `best.mpk` and deterministic eval.
