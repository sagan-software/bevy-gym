---
title: Burn-Backed Bevy Gym Training Core
created: 2026-06-13
updated: 2026-06-13
source: sprint-change-proposal-2026-06-13.md
---

# PRD: Burn-Backed Bevy Gym Training Core

## 0. Document Purpose

This PRD gives PM, architecture, implementation, and review workflows the product contract for the approved course correction. It replaces the environment-runner-only interpretation with a Burn-backed training-core product direction while preserving the small `Env` contract as the environment boundary.

## 1. Vision

`bevy-gym` is a Rust and Bevy-native reinforcement-learning training crate. A user brings a Rust environment, gets Bevy ECS rollout orchestration, trains with Burn, saves policy artifacts, and evaluates checkpoints without leaving the crate.

The redesigned environment API remains intentionally small. `Env`, `Reset`, `Step`, and `EpisodeStatus` describe how an environment resets and steps; model definitions, optimizers, replay/rollout storage, metrics, checkpointing, and deterministic eval live in the required trainer layer above it.

The first proof is CartPole DQN. The crate is not accepted until CartPole training improves reward, writes `best.mpk`, and deterministic eval can load that checkpoint. PPO follows because it uses Bevy parallel environments well.

## 2. Target User

### 2.1 Jobs To Be Done

- Define a Rust RL environment without adopting `rl-traits`, `ember-rl`, or Python as the public contract.
- Run many environment instances through Bevy ECS for deterministic rollout collection.
- Train a useful Burn-backed policy from crate-owned examples or APIs.
- Inspect run metrics, configs, seeds, and checkpoint artifacts.
- Reproduce evaluation from a saved checkpoint.

### 2.2 Non-Users (v1)

- Users who need a mature Python-first RL framework instead of a Rust-native crate.
- Users who need robotics-grade sim-to-real dynamics validation from Bevy alone.
- Users who need broad algorithm coverage before DQN and PPO are accepted.

### 2.3 Key User Journeys

- **UJ-1. Rust developer trains CartPole DQN.** A Rust developer checks out the crate, runs the CartPole DQN training path with fixed seeds, watches metrics improve against a baseline, and finds `best.mpk` in the run directory.
- **UJ-2. Rust developer evaluates a saved policy.** The developer runs deterministic eval against `best.mpk`; the crate loads the checkpoint from disk and reports CartPole reward under fixed eval seeds.
- **UJ-3. Environment author keeps the contract small.** The author implements `Env` for a custom environment and can run it manually through typed action requests before adding trainer-specific specs or tensorization.
- **UJ-4. Trainer author uses Bevy parallel rollout.** The author uses multiple Bevy environment entities to collect PPO rollout batches without changing the single-environment `Env` trait.

## 3. Glossary

- **Env** - The crate-owned single-environment trait with `reset` and `step`.
- **Step** - Atomic result containing observation, reward, episode status, and info.
- **EpisodeStatus** - Continuing, terminated, or truncated state marker used by trainers for target/bootstrap behavior.
- **Bevy Runner** - ECS orchestration that owns environment entities, reset policy, action request/response messages, and transition emission.
- **Burn Trainer** - Required crate layer that owns Burn models, optimizers, storage, metrics, checkpoints, and eval.
- **DQN** - First accepted algorithm slice for discrete control and CartPole.
- **PPO** - Second accepted algorithm slice using parallel Bevy rollout collection.
- **best.mpk** - Required best-policy checkpoint artifact saved by training and loaded by deterministic eval.

## 4. Features

### 4.1 Small Environment Contract

**Description:** Users implement a minimal Rust environment contract that feeds both manual policy systems and the trainer. Realizes UJ-3.

#### FR-1: Define Env With Repo-Owned Types

Users can define environments with `Env`, `Reset`, `Step`, and `EpisodeStatus`.

**Consequences:**
- A minimal environment compiles without `rl-traits` or `ember-rl`.
- The same environment can feed the crate's trainer once trainer-required specs/tensorization are supplied.

#### FR-2: Preserve Terminated Versus Truncated

The system preserves terminated and truncated episode status from environment step through trainer storage.

**Consequences:**
- DQN and PPO target/advantage logic can distinguish natural termination from time-limit truncation.
- Auto-reset does not emit stale post-step action requests after terminal/truncated states.

### 4.2 Bevy Runner Rollout Collection

**Description:** The Bevy runner manages many environment entities and emits typed transitions for manual policy systems and trainer storage. Realizes UJ-3 and UJ-4.

#### FR-3: Spawn And Step Parallel Environments

Users can spawn many environment instances from a factory and step them through `FixedUpdate`.

**Consequences:**
- A runner example creates at least 16 environments.
- Transitions include env id, entity, observation, action, reward, next observation, status, and info.

#### FR-4: Provide Typed Action Request/Response Flow

Manual policies can respond to typed action requests without mutating `PendingAction`.

**Consequences:**
- Docs show `ActionRequest<E>` and `ActionResponse<E>` as the beginner manual path.
- Low-level ECS access remains an advanced escape hatch.

### 4.3 Required Burn Training Engine

**Description:** Burn-backed training is required product functionality, not an optional example. Realizes UJ-1 and UJ-2.

#### FR-5: Add Required Burn Dependency And Trainer Modules

The crate builds with Burn as a first-class dependency and exposes trainer-layer modules.

**Consequences:**
- `cargo check` succeeds with Burn support.
- Trainer code does not expand the minimal `Env` trait.

#### FR-6: Provide DQN Model, Optimizer, And Replay Storage

The crate provides a default Burn MLP Q-network, optimizer setup, epsilon schedule, target network policy, and replay buffer for discrete control.

**Consequences:**
- CartPole DQN can optimize from Bevy-collected transitions.
- Replay storage lives outside ECS components.

#### FR-7: Provide Metrics And Checkpoints

Training writes metrics, config/seed metadata, checkpoints, and `best.mpk`.

**Consequences:**
- Each run has a stable directory layout.
- `best.mpk` is updated based on deterministic eval improvement.

#### FR-8: Provide Deterministic Eval From Checkpoint

Eval loads a checkpoint from disk and runs fixed-seed episodes without exploration.

**Consequences:**
- Eval from `best.mpk` succeeds after CartPole DQN training.
- Eval does not rely on the in-memory training model.

#### FR-9: Make CartPole DQN The First Acceptance Example

CartPole is the first train/eval acceptance target.

**Consequences:**
- Training reward or deterministic eval reward improves over initial/random baseline.
- Docs and examples present DQN training as primary, with heuristic/random only as baselines.

### 4.4 PPO Second Slice

**Description:** PPO follows after DQN to prove on-policy training with parallel Bevy environments. Realizes UJ-4.

#### FR-10: Add PPO Rollout Storage

The crate records observations, actions, log probabilities, values, rewards, statuses, and env ids from multiple Bevy entities.

**Consequences:**
- Rollout storage supports advantage/return computation.
- Terminated and truncated states are handled correctly.

#### FR-11: Add PPO Actor-Critic Training And Eval

The crate trains a Burn actor-critic policy and evaluates a saved PPO checkpoint.

**Consequences:**
- PPO uses multiple Bevy environment entities.
- PPO checkpoint/eval follows the same artifact conventions as DQN.

## 5. Non-Goals

- Broad algorithm coverage before DQN and PPO.
- Making `rl-traits` or `ember-rl` the public contract.
- Hiding training in examples only.
- Adding a generic rendering abstraction.
- Robotics-grade dynamics validation or sim-to-real guarantees from Bevy alone.
- Multi-agent APIs in the first training-core slice.

## 6. MVP Scope

### 6.1 In Scope

- Required Burn dependency.
- Small environment contract retained.
- Bevy runner transition stream retained.
- DQN model, optimizer, replay storage, training loop, metrics, checkpointing, and deterministic eval.
- CartPole DQN training and eval docs/examples.
- PPO planning and second-slice story set.

### 6.2 Out of Scope for MVP

- PPO implementation before DQN acceptance.
- SAC/TD3/offline RL/imitation learning.
- Rendered training dashboards.
- Python bridge as the primary training path.
- Backwards compatibility guarantee for the rejected runner-only API.

## 7. Success Metrics

**Primary**
- **SM-1:** CartPole DQN deterministic eval reward improves over initial/random baseline. Validates FR-6, FR-8, FR-9.
- **SM-2:** Training writes `best.mpk` plus metrics/config metadata. Validates FR-7.
- **SM-3:** Deterministic eval loads `best.mpk` from disk and reports successful CartPole evaluation. Validates FR-8.

**Secondary**
- **SM-4:** Minimal `Env` examples still compile without trainer internals in the trait. Validates FR-1.
- **SM-5:** PPO rollout collection uses multiple Bevy environment entities. Validates FR-10.

**Counter-metrics**
- **SM-C1:** Do not optimize for a large first algorithm menu; DQN and checkpoint/eval correctness come before breadth.
- **SM-C2:** Do not optimize for rendering demos as proof of training correctness.

## 8. Open Questions

1. What exact CartPole reward threshold and training duration should define the fast CI smoke versus longer manual validation?
2. Which Burn backend, autodiff backend, and recorder settings should be default?
3. What exact command surface should training and eval expose: examples only, CLI, library API, or a combination?
4. What release/deprecation policy should apply to the already completed runner-only redesign changes?

## 9. Assumptions Index

- Assumes the simplified environment contract and typed runner are retained from the current redesign.
- Assumes `best.mpk` remains the required best-policy artifact name.
- Assumes PPO waits until DQN proves the trainer, metrics, checkpoint, and eval path.
