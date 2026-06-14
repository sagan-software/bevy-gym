---
stepsCompleted: [1, 2, 3, 4]
status: complete
completedAt: 2026-06-13
inputDocuments:
  - prd-burn-backed-bevy-gym-training-core-2026-06-13.md
  - architecture.md
  - implementation-readiness-report-2026-06-13.md
  - ../specs/spec-bevy-gym-api-redesign/SPEC.md
  - sprint-change-proposal-2026-06-13.md
---

# bevy-gym - Epic Breakdown

## Overview

This document decomposes the approved Burn-backed training-core replan into implementable stories. DQN and CartPole checkpoint/eval acceptance are the first slice. PPO is explicitly second.

## Requirements Inventory

### Functional Requirements

- FR-1: Define Env with repo-owned types.
- FR-2: Preserve terminated versus truncated.
- FR-3: Spawn and step parallel environments.
- FR-4: Provide typed action request/response flow.
- FR-5: Add required Burn dependency and trainer modules.
- FR-6: Provide DQN model, optimizer, and replay storage.
- FR-7: Provide metrics and checkpoints.
- FR-8: Provide deterministic eval from checkpoint.
- FR-9: Make CartPole DQN the first acceptance example.
- FR-10: Add PPO rollout storage.
- FR-11: Add PPO actor-critic training and eval.

### NonFunctional Requirements

- Deterministic train/eval seeds must be explicit.
- Training must run headless without rendering.
- Large tensors, replay buffers, rollout buffers, and checkpoints must live in trainer/run storage, not ordinary ECS components.
- CI smoke thresholds must be stable enough to avoid flaky reward gates.
- Docs must not present `rl-traits`, `ember-rl`, or runner-only behavior as the product core.
- `Env`, `Reset`, `Step`, and `EpisodeStatus` must stay small and must not absorb model, optimizer, replay, rollout, metrics, or checkpoint internals.
- Burn must be a required dependency with the architecture-selected default backend/recorder path unless a later approved architecture change supersedes it.
- CartPole DQN acceptance must prove reward improvement, `best.mpk`, and deterministic eval from disk; structural trainer code alone is not sufficient.

### Additional Architecture Requirements

- Use Burn `0.21` with required `std`, `autodiff`, and `flex` features as the default CPU/headless backend path.
- Centralize backend aliases in `src/training/backend.rs`: inference on `burn::backend::Flex`, training on `burn::backend::Autodiff<burn::backend::Flex>`.
- Use `NamedMpkFileRecorder<FullPrecisionSettings>` behind `training::checkpoint` for policy artifacts.
- Preserve the artifact contract: `runs/<env>-<algorithm>/<run-id>/` with `config.json`, `seeds.json`, `metrics.jsonl`, `eval.jsonl`, `summary.json`, `checkpoints/latest.mpk`, optional periodic `checkpoints/step-*.mpk`, and root `best.mpk`.
- Expose CartPole DQN train/eval through library API and an example command surface shaped as `cargo run --example cartpole_dqn --release -- train ...` and `... eval --checkpoint ...`.
- Do not add a new CLI dependency for the first slice unless explicitly approved.
- Put all Burn, optimizer, replay/rollout storage, metrics, checkpoint, eval, and algorithm loops under `src/training/*`.
- Keep PPO second and start it only after DQN, metrics, checkpoint, and deterministic eval are accepted.

### NFR Coverage Map

| NonFunctional Requirement | Stories | Acceptance Hook |
| --- | --- | --- |
| Explicit deterministic train/eval seeds | BG-TRAIN-001, BG-TRAIN-003, BG-TRAIN-004, BG-TRAIN-005, BG-TRAIN-006, BG-TRAIN-007 | Config/seeds metadata, deterministic replay sampling, repeated eval tolerance, CI smoke fixed seeds |
| Headless training without rendering | BG-TRAIN-001, BG-TRAIN-004, BG-TRAIN-006 | MinimalPlugins/headless example path, docs reject rendering as correctness proof |
| Trainer storage outside ordinary ECS components | BG-TRAIN-001, BG-TRAIN-003, BG-TRAIN-004, BG-TRAIN-007 | Replay/rollout modules under `src/training/storage/*`; minibatches do not read large tensors from ECS components |
| Stable artifact contract and `best.mpk` | BG-TRAIN-003, BG-TRAIN-005, BG-TRAIN-006, BG-TRAIN-008 | Run directory files exist; `best.mpk` saved on eval improvement; eval loads from disk |
| CI smoke stability | BG-TRAIN-004, BG-TRAIN-005, BG-TRAIN-006 | Same-run baseline, positive reward improvement threshold, deterministic checkpoint eval |
| Docs position Burn training as product core | BG-TRAIN-001, BG-TRAIN-006 | README/docs/examples remove runner-only framing and show DQN train/eval as primary |
| No `rl-traits`/`ember-rl` product-core dependency | BG-TRAIN-001, BG-TRAIN-006 | `cargo check` and grep gates show no first-party product-surface dependency |
| Small `Env` contract remains trainer input only | BG-TRAIN-001, BG-TRAIN-002, BG-TRAIN-004, BG-TRAIN-007 | Minimal env compiles; trainer requirements surface as tensorization/spec errors, not `Env` trait expansion |

### FR Coverage Map

| Requirement | Stories |
| --- | --- |
| FR-1 | BG-TRAIN-001 |
| FR-2 | BG-TRAIN-001, BG-TRAIN-004, BG-TRAIN-007 |
| FR-3 | BG-TRAIN-001, BG-TRAIN-007 |
| FR-4 | BG-TRAIN-001 |
| FR-5 | BG-TRAIN-001 |
| FR-6 | BG-TRAIN-002, BG-TRAIN-004 |
| FR-7 | BG-TRAIN-003, BG-TRAIN-005 |
| FR-8 | BG-TRAIN-005 |
| FR-9 | BG-TRAIN-004, BG-TRAIN-006 |
| FR-10 | BG-TRAIN-007 |
| FR-11 | BG-TRAIN-008 |

### Story Dependency Map

| Story | Explicit Dependencies | Notes |
| --- | --- | --- |
| BG-TRAIN-001 | None | Foundation story; must produce concrete trainer API types, not empty modules. |
| BG-TRAIN-002 | BG-TRAIN-001 | Uses backend/config/tensor boundaries from foundation. |
| BG-TRAIN-003 | BG-TRAIN-001, BG-TRAIN-002 | Run metadata can start after BG-TRAIN-001, but concrete model checkpoint save/load requires BG-TRAIN-002. |
| BG-TRAIN-004 | BG-TRAIN-001, BG-TRAIN-002 | DQN replay/training loop uses concrete model and optimizer. |
| BG-TRAIN-005 | BG-TRAIN-002, BG-TRAIN-003, BG-TRAIN-004 | Eval loads the DQN model checkpoint emitted by the train loop and checkpoint layer. |
| BG-TRAIN-006 | BG-TRAIN-003, BG-TRAIN-004, BG-TRAIN-005 | Example and smoke gate prove the vertical CartPole DQN acceptance slice. |
| BG-TRAIN-007 | BG-TRAIN-001, BG-TRAIN-003, BG-TRAIN-006 | PPO storage follows accepted trainer/storage conventions and waits for DQN acceptance. |
| BG-TRAIN-008 | BG-TRAIN-007 | PPO actor-critic depends on rollout storage. |

## Epic List

1. Trainable Crate Foundation And Artifact Contract
2. CartPole DQN Acceptance Slice
3. PPO Parallel-Environment Slice

## Epic 1: Trainable Crate Foundation And Artifact Contract

Goal: Make `bevy-gym` visibly trainable at the crate boundary: a user can compile with required Burn support, see concrete trainer API/export surfaces, keep the small `Env` contract intact, and inspect stable run/checkpoint scaffolding that later DQN stories use. This epic is an enabling user-value slice, not a shippable trainer endpoint; it must not be accepted as complete if it only adds empty modules.

### Story 1.1: BG-TRAIN-001 Required Burn Dependency And Trainer Skeleton

As a Rust RL crate user,
I want a concrete Burn-backed trainer API boundary that compiles,
So that `bevy-gym` is visibly a training crate instead of only an environment runner.

**Depends on:** None.
**Covers:** FR-1, FR-2, FR-3, FR-4, FR-5.

**Acceptance Criteria:**

**Given** a clean checkout
**When** `cargo check` runs
**Then** the crate builds with required Burn support
**And** the public contract no longer depends on `rl-traits` or `ember-rl`.

**Given** the trainer module root exists
**When** a downstream user imports `bevy_gym::training`
**Then** concrete public trainer-boundary types are exported
**And** those exports include backend aliases or accessors, run/config types, seed config, tensorization error types, metrics writer types, checkpoint error/path types, and algorithm module roots
**And** empty placeholder modules or private-only stubs do not satisfy this story.

**Given** the architecture-selected Burn defaults
**When** the trainer backend module is compiled
**Then** the default inference backend is `burn::backend::Flex`
**And** the default training backend is `burn::backend::Autodiff<burn::backend::Flex>`
**And** the required Burn features include `std`, `autodiff`, and `flex`.

**Given** a minimal environment
**When** it implements `Env`, `Reset`, `Step`, and `EpisodeStatus`
**Then** the environment trait stays free of model, optimizer, replay, rollout, metrics, and checkpoint internals.

**Given** trainer modules are added
**When** docs or exports mention training
**Then** Burn-backed trainer responsibilities are first-class crate code, not example-only adapters.

**Given** first-party source and docs
**When** the dependency and product-surface scan runs
**Then** `rl-traits`, `rl_traits`, `ember-rl`, and `ember_rl` are absent from first-party product contracts
**And** any mention of runner-only behavior is framed as superseded or as a manual runner demonstration, not as product core.

### Story 1.2: BG-TRAIN-002 DQN Model And Optimizer

As a developer training discrete-control policies,
I want a default Burn DQN model and optimizer configuration,
So that CartPole can train without users writing neural-network boilerplate first.

**Depends on:** BG-TRAIN-001.
**Covers:** FR-6.

**Acceptance Criteria:**

**Given** a discrete observation/action environment such as CartPole
**When** the DQN model config is constructed
**Then** a Burn MLP Q-network can be initialized deterministically.

**Given** the same DQN config and model seed
**When** the model is initialized twice on the default training backend
**Then** equivalent CartPole-shaped observations produce identical Q-values within the documented deterministic tolerance.

**Given** an optimizer config
**When** training starts
**Then** optimizer state is initialized and can update model parameters.

**Given** a deterministic batch of CartPole-shaped observations, actions, rewards, next observations, and statuses
**When** one DQN loss and optimizer update is applied
**Then** the update runs without Bevy rendering
**And** at least one trainable parameter changes
**And** the resulting scalar loss and selected Q-values are finite.

**Given** DQN requires action/observation specs or tensorization
**When** an environment omits them
**Then** the error points to trainer-boundary requirements without changing the minimal `Env` trait.

**Given** DQN tensorization code is added
**When** a CartPole observation `[f32; 4]` and discrete action are converted
**Then** tensors match the documented observation and action dimensions
**And** tensorization code lives under `src/training/*`, not in `Env`.

### Story 1.3: BG-TRAIN-003 Metrics, Run Directory, And Checkpoint Contract

As a user running experiments,
I want stable metrics and checkpoint artifacts,
So that training runs are inspectable and eval can reload saved policies.

**Depends on:** BG-TRAIN-001 for run metadata and metrics. Depends on BG-TRAIN-002 for concrete DQN model checkpoint save/load. This story is not complete until the concrete DQN model checkpoint path is proven against the model from BG-TRAIN-002.
**Covers:** FR-7.

**Acceptance Criteria:**

**Given** a training run starts
**When** the run directory is created
**Then** it records config, seeds, metrics, eval records, checkpoints, and `best.mpk` using a stable layout.

**Given** a run is created for CartPole DQN
**When** the run directory is inspected
**Then** it follows `runs/cartpole-dqn/<run-id>/`
**And** it contains `config.json`, `seeds.json`, `metrics.jsonl`, `eval.jsonl`, `summary.json`, `checkpoints/latest.mpk` when a latest checkpoint exists, and root `best.mpk` when a best checkpoint exists.

**Given** metrics are written during training or eval
**When** `metrics.jsonl` and `eval.jsonl` are parsed line by line
**Then** every line is valid JSON
**And** records include enough fields to reconstruct global step, seed/run id, train loss or eval mean reward, and checkpoint-best status where applicable.

**Given** eval reward improves
**When** the best policy changes
**Then** `best.mpk` is written or replaced.

**Given** a saved checkpoint exists
**When** checkpoint loading is requested
**Then** model state is restored from disk rather than reused from an in-memory model.

**Given** the architecture-selected recorder contract
**When** checkpoint save/load is implemented
**Then** `NamedMpkFileRecorder<FullPrecisionSettings>` or an equivalent Burn named MessagePack recorder wrapper is used behind `training::checkpoint`
**And** callers and tests interact with the public artifact path `best.mpk`.

**Given** checkpoint save/load errors occur
**When** the error is reported
**Then** it includes the path and whether the failed operation was save or load.

## Epic 2: CartPole DQN Acceptance Slice

Goal: Prove the full trainer loop with CartPole DQN: train, improve, save `best.mpk`, and eval from checkpoint.

### Story 2.1: BG-TRAIN-004 DQN Replay And Training Loop

As a user training CartPole,
I want DQN replay and optimization connected to Bevy transitions,
So that the crate trains from its own runner output.

**Depends on:** BG-TRAIN-001 and BG-TRAIN-002.
**Covers:** FR-2, FR-6, FR-9.

**Acceptance Criteria:**

**Given** Bevy emits CartPole transitions
**When** DQN training runs
**Then** transitions enter replay storage and sampled batches optimize the Q-network.

**Given** transitions enter replay storage
**When** replay batches are sampled for optimization
**Then** replay data lives in `src/training/storage/replay.rs` or equivalent trainer storage
**And** large replay tensors are not stored as ordinary ECS components.

**Given** a transition has `EpisodeStatus::Terminated` or `EpisodeStatus::Truncated`
**When** DQN targets are computed
**Then** terminated transitions use zero bootstrap value
**And** truncated transitions preserve bootstrap semantics.

**Given** training is configured with fixed seeds
**When** exploration samples actions
**Then** exploration is reproducible for the same seed.

**Given** target network updates are due
**When** the configured update interval is reached
**Then** target parameters are updated according to the DQN config.

### Story 2.2: BG-TRAIN-005 Deterministic Eval From best.mpk

As a user validating training,
I want deterministic eval to load `best.mpk`,
So that checkpoint artifacts are proven usable.

**Depends on:** BG-TRAIN-002, BG-TRAIN-003, and BG-TRAIN-004.
**Covers:** FR-7, FR-8.

**Acceptance Criteria:**

**Given** `best.mpk` exists
**When** deterministic eval runs with fixed eval seeds
**Then** eval loads the checkpoint from disk and reports CartPole reward.

**Given** deterministic eval runs twice with the same checkpoint and seeds
**When** no training occurs between runs
**Then** reported reward is identical or within the documented deterministic tolerance.

**Given** no checkpoint exists
**When** eval is requested
**Then** the error clearly names the missing checkpoint path.

### Story 2.3: BG-TRAIN-006 CartPole Train/Eval Example And Smoke Gate

As a new user,
I want an out-of-box CartPole DQN command path,
So that I can see reward improve and checkpoint eval succeed.

**Depends on:** BG-TRAIN-003, BG-TRAIN-004, and BG-TRAIN-005.
**Covers:** FR-9.

**Acceptance Criteria:**

**Given** the CartPole DQN training example or CLI is run with fixed seeds
**When** training completes
**Then** deterministic eval reward improves over the initial or random policy baseline.

**Given** the example command surface is implemented
**When** a user runs `cargo run --example cartpole_dqn --release -- train --preset smoke --seed 7 --run-id ci-smoke`
**Then** the command starts the smoke training path without requiring rendering
**And** the corresponding eval command accepts `eval --checkpoint <path> --seed <seed> --episodes <n>`.

**Given** training completes
**When** the run directory is inspected
**Then** metrics/config metadata and `best.mpk` exist.

**Given** the smoke gate runs in CI
**When** training budget is constrained
**Then** it validates improvement, checkpoint creation, and checkpoint eval with stable thresholds.

**Given** docs are updated for CartPole
**When** README, crate docs, plugin docs, and CartPole docs describe the example
**Then** DQN train/eval is the primary CartPole path
**And** heuristic or random behavior is described only as a baseline or runner demonstration.

## Epic 3: PPO Parallel-Environment Slice

Goal: Add PPO after DQN acceptance, using Bevy's parallel environment strengths.

### Story 3.1: BG-TRAIN-007 PPO Rollout Storage Across Bevy Environments

As a trainer author,
I want PPO rollout storage collected from multiple Bevy environment entities,
So that on-policy training uses Bevy parallelism effectively.

**Depends on:** BG-TRAIN-001, BG-TRAIN-003, and BG-TRAIN-006.
**Covers:** FR-2, FR-3, FR-10.

**Acceptance Criteria:**

**Given** multiple environment entities are running
**When** PPO rollout collection starts
**Then** rollout storage records observation, action, log probability, value, reward, status, and env id for each step.

**Given** episodes terminate or truncate during rollout
**When** returns and advantages are computed
**Then** terminated and truncated statuses are handled according to bootstrap semantics.

**Given** rollout storage fills
**When** a PPO update is requested
**Then** minibatches can be produced without reading large tensors from ECS components.

### Story 3.2: BG-TRAIN-008 PPO Actor-Critic Training And Eval

As a user training with parallel environments,
I want a Burn actor-critic PPO implementation,
So that `bevy-gym` proves an on-policy algorithm after DQN.

**Depends on:** BG-TRAIN-007.
**Covers:** FR-11.

**Acceptance Criteria:**

**Given** PPO config and rollout storage exist
**When** PPO training runs
**Then** the actor-critic model updates from collected rollouts.

**Given** PPO eval improves
**When** best policy selection runs
**Then** a PPO `best.mpk` is saved using the same artifact contract as DQN.

**Given** PPO `best.mpk` exists
**When** deterministic eval loads it
**Then** eval reports fixed-seed reward from the saved checkpoint.
