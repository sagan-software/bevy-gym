---
baseline_commit: 84b1c781f3c0704c094c66d8ac15426df7695efd
---

# Story 1.1: BG-TRAIN-001 Required Burn Dependency And Trainer Skeleton

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Rust RL crate user,
I want a concrete Burn-backed trainer API boundary that compiles,
so that `bevy-gym` is visibly a training crate instead of only an environment runner.

## Acceptance Criteria

1. Given a clean checkout, when `cargo check` runs, then the crate builds with required Burn support and the public contract no longer depends on `rl-traits` or `ember-rl`.
2. Given the trainer module root exists, when a downstream user imports `bevy_gym::training`, then concrete public trainer-boundary types are exported, including backend aliases or accessors, run/config types, seed config, tensorization error types, metrics writer types, checkpoint error/path types, and algorithm module roots.
3. Empty placeholder modules or private-only stubs do not satisfy this story.
4. Given the architecture-selected Burn defaults, when the trainer backend module is compiled, then the default inference backend is `burn::backend::Flex`, the default training backend is `burn::backend::Autodiff<burn::backend::Flex>`, and required Burn features include `std`, `autodiff`, and `flex`.
5. Given a minimal environment, when it implements `Env`, `Reset`, `Step`, and `EpisodeStatus`, then the environment trait stays free of model, optimizer, replay, rollout, metrics, and checkpoint internals.
6. Given trainer modules are added, when docs or exports mention training, then Burn-backed trainer responsibilities are first-class crate code, not example-only adapters.
7. Given first-party source and docs, when the dependency and product-surface scan runs, then `rl-traits`, `rl_traits`, `ember-rl`, and `ember_rl` are absent from first-party product contracts.
8. Any mention of runner-only behavior is framed as superseded or as a manual runner demonstration, not as product core.

## Tasks / Subtasks

- [x] Add Burn as a required dependency. (AC: 1, 4)
  - [x] Update `Cargo.toml` with `burn = "0.21"` using `default-features = false` and features `["std", "autodiff", "flex"]`.
  - [x] Do not add `rl-traits`, `ember-rl`, `burn-rl`, Python bridge dependencies, or GPU/backend feature defaults.
  - [x] Avoid additional dependencies in this story unless the implementation proves they are necessary; if extra crates are added, document why in the Dev Agent Record.
- [x] Add the first-class trainer module root and exports. (AC: 2, 3, 6)
  - [x] Add `src/training/mod.rs` and expose it from `src/lib.rs` as `pub mod training;`.
  - [x] Export concrete public boundary modules at minimum: `backend`, `config`, `rng`, `tensor`, `metrics`, `checkpoint`, `dqn`, and `ppo`.
  - [x] Keep Burn references inside `src/training/*` except for the top-level `pub mod training` export.
- [x] Implement the backend boundary. (AC: 2, 4)
  - [x] Add `src/training/backend.rs`.
  - [x] Define `pub type InferenceBackend = burn::backend::Flex;`.
  - [x] Define `pub type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;`.
  - [x] Add concrete backend/device accessors or helper functions so the module is not just two unused aliases.
- [x] Implement concrete run/config/seed boundary types. (AC: 2, 3)
  - [x] Add `src/training/config.rs` with concrete public types such as `AlgorithmKind`, `RunConfig`, `RunPaths`, `RunId`, and an environment/algorithm run-name helper.
  - [x] Add `src/training/rng.rs` with a concrete `SeedConfig` and derived seed fields for env reset, action exploration, replay sampling, model initialization, and eval.
  - [x] Keep serialization or JSON writing minimal in this story unless needed for compile-visible types; full run artifact writing belongs to BG-TRAIN-003.
- [x] Implement tensorization boundary types. (AC: 2, 5)
  - [x] Add `src/training/tensor.rs`.
  - [x] Define concrete public error/spec types such as `TensorizationError`, `ObservationSpec`, `ActionSpec` or `DiscreteActionSpec`.
  - [x] Error messages must point to trainer-boundary requirements and must not suggest editing `Env`.
- [x] Implement metrics writer boundary types. (AC: 2, 3)
  - [x] Add `src/training/metrics.rs`.
  - [x] Define concrete public types such as `MetricRecord`, `MetricValue`, `MetricsWriter`, and `MetricsError`.
  - [x] It is acceptable for this story to create a minimal append-oriented JSONL writer boundary, but it must not claim full BG-TRAIN-003 artifact completion unless implemented and tested.
- [x] Implement checkpoint boundary types. (AC: 2, 3)
  - [x] Add `src/training/checkpoint.rs`.
  - [x] Define concrete public types such as `CheckpointPaths`, `CheckpointError`, `CheckpointOperation`, and a recorder alias/wrapper for `NamedMpkFileRecorder<FullPrecisionSettings>`.
  - [x] Preserve the user-facing artifact names `best.mpk`, `checkpoints/latest.mpk`, and optional `checkpoints/step-*.mpk`.
  - [x] Checkpoint errors must include the path and operation (`save` or `load`).
- [x] Add non-empty algorithm module roots. (AC: 2, 3)
  - [x] Add `src/training/dqn.rs` with concrete public boundary types, for example `DqnConfig`, `DqnTrainer`, and `DqnReport` shells that compile and encode intended responsibilities.
  - [x] Add `src/training/ppo.rs` with concrete public boundary types, for example `PpoConfig` and `PpoTrainer` shells, but do not implement PPO behavior yet.
  - [x] Do not start DQN model/optimizer/replay behavior in this story unless it is needed to make the public boundary coherent; BG-TRAIN-002 and BG-TRAIN-004 own that work.
- [x] Preserve the core environment and runner boundaries. (AC: 5)
  - [x] Do not add model, optimizer, replay, rollout, metrics, checkpoint, or eval fields/methods to `Env`.
  - [x] Preserve `EpisodeStatus::Terminated` versus `EpisodeStatus::Truncated` semantics and `bootstrap_mask`.
  - [x] Preserve typed runner messages: `ActionRequest<E>`, `ActionResponse<E>`, `TransitionEvent<E>`, and `EpisodeEndEvent`.
  - [x] Preserve same-tick reset behavior and duplicate-response behavior unless a failing test proves a regression already exists.
- [x] Update product-surface docs and crate docs for the new boundary. (AC: 6, 8)
  - [x] Update `src/lib.rs` crate docs to mention the first-class Burn trainer layer while keeping the small environment contract.
  - [x] Update `README.md` stale wording that currently says training algorithms and replay buffers stay outside the core crate.
  - [x] Update `docs/examples/cartpole.md` stale wording that says the heuristic runner proves the API without pulling trainer code into this crate; frame it as a manual/baseline runner demonstration.
  - [x] Keep full CartPole DQN command docs for BG-TRAIN-006 unless this story introduces an actual command.
- [x] Verify the implementation. (AC: 1, 4, 5, 7, 8)
  - [x] Run `cargo fmt`.
  - [x] Run `cargo check`.
  - [x] Run `rg -n "rl-traits|rl_traits|ember-rl|ember_rl" Cargo.toml src README.md docs examples` and ensure no first-party product contract contains these names.
  - [x] Run `rg -n "Training algorithms, replay buffers|without pulling trainer code into this crate|runner-only" README.md docs src examples` and ensure stale runner-only product framing is gone or clearly marked as manual/baseline/superseded.
  - [x] If examples are touched, run the nearest real example check, such as `cargo check --example cartpole`; do not assume `cargo check --examples` covers targets without checking `Cargo.toml` or `cargo metadata`.

### Review Findings

- [x] [Review][Patch] Validate trainer run path segments before constructing artifact paths [src/training/config.rs:48]
- [x] [Review][Patch] Escape all JSON control characters in metrics JSONL strings [src/training/metrics.rs:160]
- [x] [Review][Patch] Reject zero-sized observation dimensions in tensorization specs [src/training/tensor.rs:32]
- [x] [Review][Defer] Restore or formally migrate removed render feature/module compatibility [Cargo.toml:14] — deferred, pre-existing
- [x] [Review][Defer] Restore or formally migrate removed legacy public re-export compatibility [src/lib.rs:81] — deferred, pre-existing

## Dev Notes

### Current Source State

- `Cargo.toml` currently depends on Bevy only and has no Burn dependency. It declares Bevy `0.18` with `default-features = false` and the `bevy_log` and `multi_threaded` features. [Source: Cargo.toml]
- `src/lib.rs` currently exports `components`, `core`, `events`, `plugin`, and `systems`, but no `training` module. Its crate docs describe runner/typed policy integration but not a Burn trainer. [Source: src/lib.rs]
- `src/core.rs` already owns the small environment vocabulary: `Env`, `Reset`, `Step`, `EpisodeStatus`, `Transition`, optional `Space`/`HasSpaces`, and `check_env_with_action`. Preserve this shape. [Source: src/core.rs]
- `EpisodeStatus` already distinguishes `Continuing`, `Terminated`, and `Truncated`; `bootstrap_mask()` returns `0.0` only for `Terminated` and `1.0` for `Continuing` or `Truncated`. Do not collapse these states. [Source: src/core.rs]
- `src/events.rs` already defines typed runner messages and transition events. `TransitionEvent<E>` carries `env_id`, `entity`, and full `Transition`. `EpisodeEndEvent` carries final status, reward, length, and extras. [Source: src/events.rs]
- `src/plugin.rs` wires `BevyGymPlugin<E>`, `GymConfig`, `GymSet`, message registration, `FixedUpdate` ordering, and startup environment spawning. Do not make the plugin depend on Burn for this story. [Source: src/plugin.rs]
- `src/systems/step.rs` consumes typed `ActionResponse<E>`, calls `Env::step`, emits `TransitionEvent<E>`, emits `EpisodeEndEvent` on done states, and only emits another `ActionRequest<E>` for continuing states. Preserve this behavior. [Source: src/systems/step.rs]
- `src/systems/reset.rs` performs auto and manual resets and emits fresh `ActionRequest<E>` after reset. Preserve this behavior. [Source: src/systems/reset.rs]
- `src/training/` does not exist yet. This story creates the trainer layer instead of modifying the core trait. [Source: repo scan]
- README and CartPole docs currently contain runner-only/stale positioning. They must be updated as part of this story so product docs no longer imply trainer code belongs outside the crate. [Source: README.md; docs/examples/cartpole.md]

### Architecture Compliance

- Burn-backed training is required product functionality, not an optional example. The useful historical constraint that remains is that the environment contract stays small. [Source: ai/bmad-output/planning-artifacts/prd-burn-backed-bevy-gym-training-core-2026-06-13.md#4.3-required-burn-training-engine]
- The approved layer order is: `Env` reset/step -> Bevy runner entities and typed transitions -> trainer storage -> Burn model optimization -> metrics/checkpoints -> deterministic eval from checkpoint. [Source: ai/bmad-output/planning-artifacts/architecture.md#product-layers]
- Layer 1 (`src/core.rs`) may contain optional spaces/checkers and transition vocabulary, but no Burn model, optimizer, replay, rollout, metrics, checkpoint, or eval implementation. [Source: ai/bmad-output/planning-artifacts/architecture.md#product-layers]
- Layer 2 runner code lives in `src/components.rs`, `src/events.rs`, `src/plugin.rs`, and `src/systems/*`; it remains required infrastructure but is not product-complete by itself. [Source: ai/bmad-output/planning-artifacts/architecture.md#product-layers]
- Layer 3 trainer code lives in `src/training/*` and owns Burn backend aliases, model definitions, optimizer setup, replay/rollout storage, training loops, metrics, checkpoint save/load, and deterministic eval. [Source: ai/bmad-output/planning-artifacts/architecture.md#product-layers]
- BG-TRAIN-001 is an enabling user-value slice and must not be accepted if it only adds empty modules. [Source: ai/bmad-output/planning-artifacts/epics-burn-backed-bevy-gym-training-core-2026-06-13.md#epic-1-trainable-crate-foundation-and-artifact-contract]

### Library And Framework Requirements

- Use Burn `0.21` as the required first-class trainer dependency. [Source: ai/bmad-output/planning-artifacts/architecture.md#burn-backend-autodiff-and-recorder]
- Dependency target:

```toml
[dependencies.burn]
version = "0.21"
default-features = false
features = ["std", "autodiff", "flex"]
```

- Centralize backend defaults in `src/training/backend.rs`:

```rust
pub type InferenceBackend = burn::backend::Flex;
pub type TrainingBackend = burn::backend::Autodiff<InferenceBackend>;
```

- Use `NamedMpkFileRecorder<FullPrecisionSettings>` behind `training::checkpoint` for policy artifacts. If Burn APIs expect an extensionless base path, hide that in `training::checkpoint`; callers should still pass and inspect `best.mpk`. [Source: ai/bmad-output/planning-artifacts/architecture.md#burn-backend-autodiff-and-recorder]
- Do not implement with `burn-rl`, `ember-rl`, `rl-traits`, Python SB3, GPU-specific backends, or a custom microservice boundary in this story. [Source: ai/bmad-output/planning-artifacts/architecture.md#anti-patterns]
- Latest external verification on 2026-06-13:
  - Burn 0.21.0 docs list `std`, `autodiff`, and `flex` feature flags: https://docs.rs/crate/burn/latest/features
  - Burn docs describe `Flex` as a pure-Rust CPU backend and `Autodiff` as the backend decorator for backpropagation: https://docs.rs/burn/latest/burn/
  - Burn Flex docs describe `Flex` as a fast, portable CPU backend with pure Rust and thread-safe design: https://docs.rs/burn-flex/latest/burn_flex/
  - `NamedMpkFileRecorder` is the named MessagePack file recorder surface: https://burn.dev/docs/burn/record/struct.NamedMpkFileRecorder.html
  - Burn 0.21 release notes say `burn-flex` replaced `burn-ndarray`; do not choose `NdArray` as the new default unless an approved architecture change supersedes this story: https://github.com/Tracel-AI/burn/releases

### File Structure Requirements

- Required updates:
  - `Cargo.toml`
  - `src/lib.rs`
  - `README.md`
  - `docs/examples/cartpole.md`
- Required new files for this story:
  - `src/training/mod.rs`
  - `src/training/backend.rs`
  - `src/training/config.rs`
  - `src/training/rng.rs`
  - `src/training/tensor.rs`
  - `src/training/metrics.rs`
  - `src/training/checkpoint.rs`
  - `src/training/dqn.rs`
  - `src/training/ppo.rs`
- Optional if useful for clean module organization, but do not overbuild:
  - `src/training/models/mod.rs`
  - `src/training/storage/mod.rs`
- Do not move the current CartPole example in this story unless required by docs or compile errors. BG-TRAIN-006 owns the `cartpole_dqn` command surface.

### Testing Requirements

- `cargo check` is the required acceptance check for this story.
- `cargo fmt` should run before final status.
- Add focused unit tests where cheap and useful, especially for:
  - backend alias/device helper compilation;
  - seed derivation determinism if implemented with actual derived fields;
  - run-path construction such as `runs/cartpole-dqn/<run-id>/best.mpk`;
  - checkpoint path helpers returning `best.mpk` and `checkpoints/latest.mpk`;
  - tensor/spec error messages staying trainer-boundary focused.
- Do not add flaky reward thresholds in this story. CI reward calibration belongs to later DQN/smoke stories.

### Product-Surface Scan Requirements

Run these scans before marking the story complete:

```sh
rg -n "rl-traits|rl_traits|ember-rl|ember_rl" Cargo.toml src README.md docs examples
rg -n "Training algorithms, replay buffers|without pulling trainer code into this crate|runner-only" README.md docs src examples
```

Expected result:

- No `rl-traits`, `rl_traits`, `ember-rl`, or `ember_rl` in first-party product contracts.
- Any runner-only or heuristic-only text is explicitly framed as a manual runner demonstration, baseline, historical/superseded behavior, or non-core path.

### Previous Story Intelligence

No previous story exists. `BG-TRAIN-001` has no dependencies and anchors the trainer boundary for BG-TRAIN-002 and BG-TRAIN-003.

### Project Context Reference

No `project-context.md` file exists in this checkout. Use the planning artifacts and live crate files cited above as the source of truth.

### Completion Note

Ultimate context engine analysis completed - comprehensive developer guide created.

## Change Log

- 2026-06-13: Implemented BG-TRAIN-001 Burn dependency and trainer API boundary; moved story to review.
- 2026-06-13: Addressed code-review patch findings and moved story to done.

## Dev Agent Record

### Agent Model Used

GPT-5 Codex

### Debug Log References

- Red phase: `cargo check` failed on missing `src/training/mod.rs` after adding `pub mod training;`.
- Green/refactor: `cargo test training::` passed 12 focused training-boundary tests.
- Final validation: `cargo fmt`, `cargo check`, `cargo test`, `cargo check --example cartpole`, `cargo clippy`, `yamllint ai/bmad-output/implementation-artifacts/sprint-status.yaml`.
- Product scans: `rg -n "rl-traits|rl_traits|ember-rl|ember_rl" Cargo.toml src README.md docs examples` returned no matches; `rg -n "Training algorithms, replay buffers|without pulling trainer code into this crate|runner-only" README.md docs src examples` returned no matches.

### Completion Notes List

- Added required Burn `0.21` dependency with `std`, `autodiff`, and `flex` features and no additional external dependencies.
- Added `bevy_gym::training` with concrete backend, config, seed, tensorization, metrics, checkpoint, DQN, and PPO boundary types.
- Preserved the small `Env` contract and runner modules; no model, optimizer, replay, rollout, metrics, checkpoint, or eval responsibilities were added to `Env`.
- Updated crate docs, README, CartPole docs, and CartPole example framing to present the heuristic path as a manual baseline beside the crate-owned Burn trainer boundary.
- Added focused unit tests for backend alias/device construction, run paths, deterministic seeds, tensorization errors/specs, metrics JSONL output, and checkpoint paths/errors.

### File List

- Cargo.toml
- Cargo.lock
- README.md
- docs/examples/cartpole.md
- examples/classic-control/cart_pole.rs
- src/lib.rs
- src/training/mod.rs
- src/training/backend.rs
- src/training/config.rs
- src/training/rng.rs
- src/training/tensor.rs
- src/training/metrics.rs
- src/training/checkpoint.rs
- src/training/dqn.rs
- src/training/ppo.rs
- ai/bmad-output/implementation-artifacts/1-1-bg-train-001-required-burn-dependency-and-trainer-skeleton.md
- ai/bmad-output/implementation-artifacts/sprint-status.yaml
- ai/bmad-output/implementation-artifacts/deferred-work.md
