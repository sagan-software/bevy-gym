---
title: 'Burn-Backed Bevy Gym Training Core'
type: 'feature'
created: '2026-06-12'
updated: '2026-06-13'
status: 'ready-for-dev'
baseline_commit: '84b1c781f3c0704c094c66d8ac15426df7695efd'
supersedes: 'environment-runner-only Bevy Gym API Redesign handoff'
context:
  - '{project-root}/ai/bmad-output/specs/spec-bevy-gym-api-redesign/SPEC.md'
  - '{project-root}/ai/bmad-output/specs/spec-bevy-gym-api-redesign/api-contract.md'
  - '{project-root}/ai/bmad-output/specs/spec-bevy-gym-api-redesign/architecture-and-migration.md'
  - '{project-root}/ai/bmad-output/specs/spec-bevy-gym-api-redesign/training-engine.md'
  - '{project-root}/ai/bmad-output/planning-artifacts/architecture.md'
  - '{project-root}/ai/bmad-output/planning-artifacts/prd-burn-backed-bevy-gym-training-core-2026-06-13.md'
  - '{project-root}/ai/bmad-output/planning-artifacts/epics-burn-backed-bevy-gym-training-core-2026-06-13.md'
---

<frozen-after-approval reason="approved 2026-06-13 course correction - do not revert to runner-only product framing">

## Intent

**Problem:** The prior redesign correctly simplified the environment contract but incorrectly removed Burn-backed training from the product core. That left `bevy-gym` positioned as an environment runner, which conflicts with the approved direction.

**Approach:** Keep the small `Env` / `Reset` / `Step` / `EpisodeStatus` contract and typed Bevy runner as the trainer input boundary. Add a required Burn-backed trainer layer with model definitions, optimizers, replay/rollout storage, metrics, checkpoint save/load, and deterministic eval. Prove the first slice with CartPole DQN; plan PPO as the second slice using Bevy parallel environments.

## Boundaries & Constraints

**Always:** Keep `Env`, `Reset`, `Step`, `EpisodeStatus`, transition, request, and response types repo-owned. Keep the single-environment trait small. Preserve terminated versus truncated semantics. Keep batching in the runner/trainer orchestration layer. Make Burn a required first-class dependency. Save metrics/config/checkpoints and `best.mpk`. Evaluate checkpoints from disk with fixed seeds.

**Ask First:** Ask before adding non-Burn trainer dependencies, making `ember-rl` a runtime dependency, adding Python as the primary training path, deleting `ref/` reference directories, changing the required `best.mpk` artifact name, or expanding beyond DQN/PPO.

**Never:** Do not reintroduce `rl-traits` as the public contract, make `ember-rl` the product core, hide training in examples only, expand `Env` with trainer internals, rely on rendering for training correctness, or claim the runner-only redesign is accepted.

</frozen-after-approval>

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Minimal env | A user implements `Env` for a small local type | The type compiles using `bevy_gym::{Env, Reset, Step, EpisodeStatus}` and remains free of trainer internals | Compile failure points to missing associated types or methods |
| Trainer-bound env | An env supplies trainer-required specs/tensorization | Burn trainer can collect transitions and optimize a policy | Missing specs/tensorization produce trainer-boundary errors, not `Env` trait changes |
| CartPole DQN train | Fixed seeds and DQN config | Reward improves over initial/random baseline and metrics are written | Unstable improvement must tune threshold/budget before acceptance |
| Best checkpoint | Eval reward improves during training | `best.mpk` is saved in the run directory with config and metrics | Checkpoint write/load errors include the artifact path |
| Checkpoint eval | `best.mpk` exists | Deterministic eval loads checkpoint from disk and reports CartPole reward | Missing checkpoint reports the missing path |
| Terminal auto-reset | `Env::step` returns `Terminated` or `Truncated` and auto-reset is enabled | Runner emits one transition, resets, and emits exactly one post-reset action request | No stale post-step request is emitted for the terminal state |
| PPO rollout | Multiple Bevy env entities collect on-policy steps | Rollout storage records per-env observations, actions, log probabilities, values, rewards, and statuses | Rollout overflow/incomplete batches are explicit trainer states |

## Code Map

- `Cargo.toml` - add required Burn dependency and any narrowly justified trainer support dependencies; do not restore `rl-traits` or make `ember-rl` core.
- `src/lib.rs` - export the trainer surface while keeping the existing environment/runner exports clear.
- `src/core.rs` - keep `Env`, `Reset`, `Step`, `EpisodeStatus`, optional spaces/checkers, and transition vocabulary small.
- `src/components.rs`, `src/events.rs`, `src/plugin.rs`, `src/systems/step.rs`, `src/systems/reset.rs` - preserve typed runner flow and transition emission for trainer consumption.
- `src/training/mod.rs` - new trainer module root.
- `src/training/config.rs` - algorithm/run/eval seed configuration.
- `src/training/models.rs` - Burn DQN Q-network first, PPO actor-critic second.
- `src/training/optim.rs` - optimizer construction and learning-rate configuration.
- `src/training/storage/replay.rs` - DQN replay storage.
- `src/training/storage/rollout.rs` - PPO rollout storage.
- `src/training/metrics.rs` - metrics writer and run summaries.
- `src/training/checkpoint.rs` - checkpoint save/load and `best.mpk` handling.
- `src/training/eval.rs` - deterministic eval from in-memory model or checkpoint path.
- `src/training/dqn.rs` - DQN trainer loop, target network updates, epsilon schedule, batch optimization.
- `src/training/ppo.rs` - PPO trainer loop after DQN acceptance.
- `examples/classic-control/cart_pole.rs` or a new CartPole training example - make CartPole DQN train/eval the primary path; heuristic/random remains only as baseline.
- `README.md`, `docs/plugins/bevy_gym_plugin.md`, `docs/examples/cartpole.md` - update product positioning and commands after implementation.

## Tasks & Acceptance

**Execution:**
- [ ] BG-TRAIN-001 - add required Burn dependency and trainer module skeleton while preserving the small `Env` trait.
- [ ] BG-TRAIN-002 - add default Burn DQN MLP Q-network, optimizer config, action/observation tensorization, and deterministic initialization.
- [ ] BG-TRAIN-003 - add run directory layout, metrics writer, checkpoint save/load, and `best.mpk` best-policy handling.
- [ ] BG-TRAIN-004 - implement DQN replay buffer, epsilon schedule, target-network updates, and batch optimization from Bevy-collected CartPole transitions.
- [ ] BG-TRAIN-005 - implement deterministic checkpoint eval that loads `best.mpk` from disk.
- [ ] BG-TRAIN-006 - convert CartPole into the first train/eval acceptance example and add a stable smoke gate.
- [ ] BG-TRAIN-007 - add PPO rollout storage using multiple Bevy environment entities after DQN acceptance.
- [ ] BG-TRAIN-008 - add PPO actor-critic training/checkpoint/eval after rollout storage is accepted.
- [ ] Docs - rewrite README, crate docs, plugin docs, and CartPole docs around "bring a Rust env, get a Bevy runner and Burn trainer."

**Acceptance Criteria:**
- Given a clean checkout, when `cargo check` runs, then the crate builds with required Burn support and without `rl-traits` or `ember-rl`.
- Given a minimal env, when it implements `Env`, then it does not need model, optimizer, replay, rollout, metrics, or checkpoint code in the trait.
- Given CartPole DQN training runs with fixed seeds, then deterministic eval reward improves over the initial or random policy baseline.
- Given CartPole DQN training completes, then `best.mpk` is written in the run directory with accompanying metrics and config metadata.
- Given `best.mpk` exists, then deterministic eval loads the checkpoint from disk and reports successful CartPole evaluation.
- Given an env step returns `Terminated` or `Truncated`, when auto-reset runs in the same fixed update, then exactly one post-reset `ActionRequest<E>` is readable for that env.
- Given PPO work starts, then rollout collection uses multiple Bevy environment entities and does not store large tensor batches as ordinary ECS components.

## Design Notes

The trainer layer is first-class crate code. Keep this separation:

```text
Env reset/step -> Bevy runner transitions -> trainer storage -> Burn optimization -> checkpoint/eval
```

The environment contract stays beginner-simple. Algorithm-specific specs, tensorization, replay, rollout, models, optimizers, and checkpoint recorders sit at the trainer boundary.

The first acceptance target is behavioral, not structural: CartPole DQN must actually improve against a baseline, save `best.mpk`, and evaluate from that checkpoint.

Resolved architecture decisions from `ai/bmad-output/planning-artifacts/architecture.md`:

- Default backend: Burn `0.21` with required features `std`, `autodiff`, and `flex`; centralize aliases in `src/training/backend.rs`.
- Default trainer backend: `burn::backend::Autodiff<burn::backend::Flex>` for training and `burn::backend::Flex` for inference/eval.
- Default recorder: `NamedMpkFileRecorder<FullPrecisionSettings>` behind `training::checkpoint`, with the public artifact contract fixed to `best.mpk`.
- Run layout: `runs/<env>-<algorithm>/<run-id>/` with `config.json`, `seeds.json`, `metrics.jsonl`, `eval.jsonl`, `summary.json`, `checkpoints/latest.mpk`, optional periodic `checkpoints/step-*.mpk`, and root `best.mpk`.
- Command shape: `cargo run --example cartpole_dqn --release -- train ...` and `cargo run --example cartpole_dqn --release -- eval --checkpoint ...`; do not add a CLI dependency for the first slice without approval.
- CI gate: smoke preset proves deterministic positive reward improvement over a same-run baseline, `best.mpk` creation, and eval-from-disk; useful-policy preset is longer manual/release validation.
- Module boundary: all Burn, optimizer, storage, metrics, checkpoint, and eval logic lives under `src/training/*`, not in `Env` or ordinary ECS components.

## Verification

**Commands and gates:**
- `cargo fmt --check` - expected: changed Rust files are formatted.
- `cargo check` - expected: crate builds with Burn and without `rl-traits` or `ember-rl`.
- `cargo clippy --all-targets -- -D warnings` - expected: strict lint gate passes.
- `cargo test` - expected: unit tests for core, runner, storage, checkpoint, and deterministic eval pass.
- `cargo check --examples` - expected: examples build against the new API and trainer.
- CartPole DQN train command chosen during implementation - expected: reward improves over initial/random baseline, metrics are written, and `best.mpk` exists.
- CartPole DQN eval command chosen during implementation - expected: loads `best.mpk` from disk and reports deterministic reward.
- `rg -n "rl_traits|rl-traits|ember_rl|ember-rl|GymRender|GymRenderPlugin|ActionRequestEvent|ExperienceEvent" Cargo.toml Cargo.lock README.md src examples docs` - expected: no first-party product-surface matches.
