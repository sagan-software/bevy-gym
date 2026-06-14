---
title: Training Engine
type: companion
spec: SPEC-bevy-gym-api-redesign
---

# Training Engine

## Product Boundary

The training engine is required crate functionality. It sits above the small `Env` contract and Bevy runner, consumes typed transitions, and owns Burn model definitions, optimizer state, storage, metrics, checkpoints, and eval.

## Module Responsibilities

Expected implementation surface:

- `training::config` - algorithm config, run config, seed config, eval config.
- `training::models` - Burn model definitions for DQN first, actor-critic PPO second.
- `training::optim` - optimizer construction and learning-rate configuration.
- `training::storage::replay` - DQN transition replay.
- `training::storage::rollout` - PPO rollout batches with advantages/returns.
- `training::metrics` - append-only run metrics and summary records.
- `training::checkpoint` - save/load of latest, periodic, and best checkpoints.
- `training::eval` - deterministic policy evaluation from an in-memory model or checkpoint.
- `training::dqn` - first-slice trainer loop and target-network updates.
- `training::ppo` - second-slice on-policy trainer loop.

Exact module names may change, but these responsibilities must remain first-class crate code.

## DQN First Slice

CartPole DQN is the first acceptance target:

- required Burn MLP Q-network for discrete control;
- optimizer setup;
- replay buffer;
- epsilon-greedy exploration schedule;
- target network update policy;
- fixed seeds for environment resets, action sampling, and model initialization;
- deterministic eval episodes with exploration disabled;
- `best.mpk` saved when eval improves;
- metrics that show training reward/eval reward over time.

Acceptance is not "DQN code exists." Acceptance is that CartPole reward improves over the initial or random policy baseline, `best.mpk` is saved, and eval from `best.mpk` succeeds.

## PPO Second Slice

PPO follows DQN because it benefits from Bevy parallel environments:

- multiple Bevy environment entities collect on-policy rollout batches;
- actor-critic Burn model supports discrete first and can be extended to continuous actions later;
- rollout storage records observations, actions, log probabilities, values, rewards, status, and env ids;
- advantage and return computation respects terminated versus truncated status;
- checkpoint/eval uses the same artifact conventions as DQN.

PPO should not start until the DQN trainer, metrics, checkpoint, and deterministic eval path are accepted.

## Run Directory Contract

Each training run writes a stable directory such as:

```text
runs/<env>-<algorithm>/<run-id>/
  config.json
  metrics.jsonl
  eval.jsonl
  checkpoints/
    latest.mpk
    step-000000.mpk
  best.mpk
```

`best.mpk` is the primary acceptance artifact. Metrics and config metadata must be enough to explain the seed, algorithm settings, environment count, evaluation protocol, and best checkpoint selection.

## Determinism Rules

- Training and evaluation seed streams must be explicit.
- Evaluation disables exploration and uses fixed eval seeds.
- Checkpoint eval must load from disk rather than reusing the in-memory training model.
- Tests should compare improvement against a baseline and a modest threshold chosen for stability, not a single fragile max reward target.
- Rendering must not affect training correctness or eval results.

## Dependency Policy

Burn is required. `rl-traits` must not return as the public environment contract. `ember-rl` must not become the product core. Additional trainer dependencies require a narrow justification and should not hide the crate's Burn-backed trainer behind an external algorithm framework.
